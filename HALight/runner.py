import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.get_neighbour import build_neighbors_from_file
import random
import numpy as np
import torch
from harl.common.valuenorm import ValueNorm

from actor_buffer import ActorBuffer
from harl.utils.trans_tools import _t2n
from cityflow_env_wrapper import CityflowEnvWrapper
from HALight_Agent import Agent
from tqdm import tqdm
from harl.utils.envs_tools import check
from critic_buffer import OnPolicyCriticBufferEP
from critic_network import Critic
from typing import Dict
from utils.comm_buffers import CommBuffers

class OnPolicyBaseRunner:

    def __init__(self, args):
        self.args = args
        self.hidden_sizes = args.hidden_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_length = args.episode_length
        self.fixed_order = args.fixed_order
        self.id2neighbours = build_neighbors_from_file(args.net)
        self.infor_dim = args.infor_dim

        if "high" in args.flow:
            flow = "high"
        else:
            flow = "low"
        if "4x4" in args.net:
            net = "4x4"
        else:
            net = "3x3"
        self.log_file = "./HALight_RNN_" + net + "_" + str(args.obs_drop_prob) + "_" + flow + "_" + "training_log.txt"
        self.metric_file = "./HALight_RNN_" + net + "_" + str(args.obs_drop_prob) + "_" + flow + "_" + "metric.txt"

        self.env = CityflowEnvWrapper(self.args)
        self.ts_ids = []
        for inter_id in self.env.intersection_ids:
            self.ts_ids.append(inter_id)
        self.num_agents = args.N
        self.comm_buffer = CommBuffers(episode_length=self.episode_length,N=self.num_agents,h_dim=self.args.ht_dim,infor_dim=self.args.infor_dim,device=self.device)
        # actor
        self.actors = {}
        for ts in self.ts_ids:
            self.actors[int(ts)] = Agent(
                args=args,
                action_size=self.env.interid2actiondim[ts],
                state_size=self.env.interid2statedim[ts],
                neighbours=self.id2neighbours[ts],
                ts_id=ts,
            )


        self.actor_buffer = {
            int(ts): ActorBuffer(
                args=args,
                action_size=self.env.interid2actiondim[ts],
                state_size=self.env.interid2statedim[ts],
                neighbours=self.id2neighbours[ts],
                ts_id=ts,
            )
            for ts in self.ts_ids
        }
        self.rewards = np.zeros((self.episode_length, self.num_agents))

        share_obs_size = 0
        for ts in self.env.intersection_ids:
            share_obs_size += self.env.interid2statedim[ts]
        share_obs_size = share_obs_size + self.num_agents * self.args.infor_dim
        self.critic = Critic(args=self.args,share_states_size=share_obs_size)
        self.critic_buffer = OnPolicyCriticBufferEP(args=args,share_obs_size=share_obs_size)
        if args.use_valuenorm is True:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
    def run(self):
        self.setup_seed()
        print("start running")
        episodes = self.args.episode
        for episode in tqdm(range(1, episodes + 1)):
            self.comm_buffer.clear_()
            deterministic = False
            total_reward = 0
            self.prep_rollout()  # change to eval mode
            self.warmup()
            for step in range(self.episode_length + 1):
                h_all = torch.empty(self.num_agents, self.args.ht_dim, device=self.device)
                for ts_id in self.actors.keys():
                    neigh_infor_prev_dict_i = self.actors[ts_id].policy_network.build_neigh_infor_prev_dict(
                        self.comm_buffer, 1)
                    h_i_t = self.actors[ts_id].get_self_ht(self.actor_buffer[ts_id].states[step],
                                                           neigh_infor_prev_dict_i)  # [h_dim], no grad
                    h_all[ts_id].copy_(h_i_t.squeeze(0))
                self.comm_buffer.set_step_ht(step, h_all)
                # Sample actions from actors and values from critics
                (
                    actions,
                    action_log_probs,
                    ts2action,
                    rnn_states
                ) = self.collect(step,deterministic=deterministic)
                if step != self.episode_length:
                    next_states, rewards, dones, _ = self.env.step(ts2action)
                    states_array, actions_array, rewards_array, rewards_sum = self.handle_states_actions_rewards(ts2action,rewards,next_states)
                    total_reward += rewards_sum
                    dones = np.zeros_like(dones)

                    data = (
                        states_array,
                        rnn_states,
                        rewards_array,
                        dones,
                        actions,
                        action_log_probs,
                        rewards_sum,
                    )
                    self.insert(data, step)  # insert data into buffer
            self.insert_critic_buffer()
            for ts_id in self.actor_buffer.keys():  # i = agent_id
                self.actor_buffer[ts_id].attach_comm_snapshot(self.comm_buffer, agent_id=ts_id, detach=True)

            throughput = self.env.get_throughput()
            average_queue_length = total_reward / len(self.actors.keys()) / self.episode_length
            average_travel_time = self.env.get_average_travel_time()
            average_waiting_time = self.env.get_average_queue_time()
            metric_text = "Episode {}:throughput {},average_queue_length {},average_travel_time {},average_waiting_time {}".format(
                episode, throughput, average_queue_length, average_travel_time, average_waiting_time)
            with open(self.metric_file, "a") as f:
                f.write(metric_text + "\n")
            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            self.train()
            log_text = "Episode {}:{}".format(episode, total_reward)
            print(log_text)

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_text + "\n")

    def warmup(self):
        # reset env
        initial_states = self.env.reset()
        # replay buffer
        for ts in self.actors.keys():
            self.actor_buffer[ts].states[0] = self.get_agent_obs(initial_states,ts).copy()
            self.actor_buffer[ts].reset_rnn_states()
        self.critic_buffer.reset_rnn_states()


    def get_agent_obs(self, states, ts_id):
        state = states[str(ts_id)]
        state = np.array(state, dtype=np.float32)
        return state

    def handle_states_actions_rewards(self, actions, rewards, states):
        all_states = []
        all_actions = []
        all_rewards = []

        share_obs_list = []

        # Process each agent's state, action, and reward
        for ts in self.actors.keys():
            # Encode the state for the current agent
            state = states[str(ts)]
            state = np.array(state, dtype=np.float32)

            share_obs_list.append(state.flatten())

            # Get the action and process it
            action = np.array(actions[str(ts)], dtype=np.int64)
            reward = np.array(rewards[str(ts)], dtype=np.float32)

            # Append the processed values
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)

        states_array = np.array(all_states, dtype=object)

        actions_array = np.array(all_actions, dtype=np.int64)  # [n]
        rewards_array = np.array(all_rewards, dtype=np.float32)  # [n]

        rewards_sum = np.sum(rewards_array)
        return states_array, actions_array, rewards_array, rewards_sum

    @torch.no_grad()
    def collect(self, step,deterministic=False):
        # collect actions, action_log_probs, rnn_states from n actors
        infor_all = torch.empty(self.num_agents, self.infor_dim, device=self.device)
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        ts2action = {}
        for ts_id in self.actors.keys():
            neigh_h_dict = {int(nid): self.comm_buffer.get_ht(step, int(nid), detach=True).unsqueeze(0) for nid in self.id2neighbours[str(ts_id)]}
            neigh_infor_prev_dict_i = self.actors[ts_id].policy_network.build_neigh_infor_prev_dict(self.comm_buffer,1)
            obs_i_t = check(self.actor_buffer[ts_id].states[step]).to(self.device)
            obs_i_t = obs_i_t.unsqueeze(0)
            action, action_log_prob, infor_i_t,rnn_state = self.actors[ts_id].policy_network.forward(
                obs_i_t=obs_i_t,
                rnn_states=self.actor_buffer[ts_id].rnn_states[step],
                neigh_infor_prev_dict=neigh_infor_prev_dict_i,
                neigh_h_t=neigh_h_dict,
                deterministic=deterministic,
                stop_grad_neighbors=True
            )
            infor_all[ts_id].copy_(infor_i_t.squeeze(0) if infor_i_t.ndim == 2 else infor_i_t)
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            ts2action[str(ts_id)] = _t2n(action)[0][0]
        self.comm_buffer.set_step_infor(step, infor_all)
        actions = np.array(action_collector)
        action_log_probs = np.array(action_log_prob_collector)
        rnn_states = np.array(rnn_state_collector)
        return actions, action_log_probs,ts2action,rnn_states

    def insert(self, data,step):
        """Insert data into buffer."""
        (
            obs,
            rnn_states,
            rewards,
            dones,
            actions,
            action_log_probs,
            rewards_sum,
        ) = data

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[agent_id],
                rnn_states[agent_id],
                actions[agent_id],
                action_log_probs[agent_id],
            )
        self.rewards[step] = rewards.copy()

        self.critic_buffer.insert(
            rewards_sum,
        )

    def train(self):
        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.episode_length,
                1,
            ),
            dtype=np.float32,
        )

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                         :-1
                         ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                    self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].advantage = advantages
        # normalize advantages for FP

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor
            # compute action log probs for the actor before update.
            neigh_prev_info_ep = self.actors[agent_id].policy_network.build_episode_neigh_infor_prev_dict(self.comm_buffer, T=self.episode_length, B=1)
            neigh_h_ep = self.actors[agent_id].policy_network.build_episode_neigh_h_dict(self.comm_buffer, T=self.episode_length, B=1, detach=True)


            old_actions_logprob, _, _ = self.actors[agent_id].evaluate_actions(
                local_state=self.actor_buffer[agent_id].states[:-1].reshape(-1, *self.actor_buffer[agent_id].states.shape[1:]),
                rnn_states=self.actor_buffer[agent_id].rnn_states[0],
                neigh_infor_prev_dict=neigh_prev_info_ep,
                neigh_ht_dict=neigh_h_ep,
                action=self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[1:]),
            )

            # update actor
            self.actors[agent_id].train(
                self.actor_buffer[agent_id],neigh_prev_info_ep,neigh_h_ep
            )
            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actors[agent_id].evaluate_actions(
                local_state=self.actor_buffer[agent_id].states[:-1].reshape(-1, *self.actor_buffer[agent_id].states.shape[1:]),
                rnn_states=self.actor_buffer[agent_id].rnn_states[0],
                neigh_infor_prev_dict=neigh_prev_info_ep,
                neigh_ht_dict=neigh_h_ep,
                action=self.actor_buffer[agent_id].actions.reshape(-1, *self.actor_buffer[agent_id].actions.shape[1:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                    torch.exp(new_actions_logprob - old_actions_logprob)
                .reshape(
                    self.episode_length,
                    1,
                )
            )

        # update critic
        self.critic.train(self.critic_buffer, self.value_normalizer)

    def prep_rollout(self):
        for ts in self.actors.keys():
            self.actors[ts].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        for ts in self.actors.keys():
            self.actors[ts].prep_training()
        self.critic.prep_training()

    def setup_seed(self):
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)
            torch.backends.cudnn.deterministic = True

    def insert_critic_buffer(self):
        _, infor = self.comm_buffer.to_cpu_and_numpy()

        # 遍历每个 step
        for step in range(self.episode_length + 1):
            step_share_obs = []

            for ts_id in self.actors.keys():
                actor_state = self.actor_buffer[ts_id].states[step]
                actor_infor = infor[step, ts_id]

                share_state = np.concatenate([actor_state.flatten(), actor_infor.flatten()],
                                             axis=-1)  # [share_obs_shape]

                step_share_obs.append(share_state)

            share_obs = np.concatenate([obs.flatten() for obs in step_share_obs])  # shape: [N, share_obs_shape]
            value,rnn_state_critic = self.critic.get_values(share_obs,rnn_states_critic=self.critic_buffer.rnn_states_critic[step])
            value = _t2n(value)
            rnn_state_critic = _t2n(rnn_state_critic)


            self.critic_buffer.share_obs[step] = share_obs
            self.critic_buffer.value_preds[step] = value
            if step != self.episode_length:
                self.critic_buffer.rnn_states_critic[step + 1] = rnn_state_critic


    @torch.no_grad()
    def compute(self):
        next_value= self.critic_buffer.value_preds[-1]
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)




