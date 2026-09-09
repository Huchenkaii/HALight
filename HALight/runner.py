"""Base runner for on-policy algorithms."""
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
from utils.comm_buffers import CommBuffers
import glob


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args):
        """Initialize the OnPolicyBaseRunner class."""
        self.args = args
        self.hidden_sizes = args.hidden_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.episode_length = args.episode_length
        self.fixed_order = args.fixed_order
        self.id2neighbours = build_neighbors_from_file(args.net)
        self.infor_dim = args.infor_dim
        self.base_seed = 0 if self.args.seed is None else self.args.seed
        self.flow_rng = random.Random(self.base_seed)

        self.flow_files = self._collect_flow_files(self.args.flow_dir)
        self.flow = self.flow_files[0]
        self.current_flow_file = self.flow

        flow_dir_lower = self.args.flow_dir.lower()
        if "high" in flow_dir_lower:
            flow_type = "high"
        elif "moderate" in flow_dir_lower:
            flow_type = "moderate"
        elif "low" in flow_dir_lower:
            flow_type = "low"
        else:
            flow_type = os.path.basename(os.path.normpath(self.args.flow_dir)) or "flow"
        net_name = os.path.splitext(os.path.basename(args.net))[0]
        prefix = "./HALight_RNN_" + net_name + "_" + str(args.obs_drop_prob) + "_" + flow_type + "_"
        self.log_file = prefix + "training_log.txt"
        self.step_metric_file = prefix + "metric_step.txt"
        self.validate_metric_file = prefix + "validate_metric.txt"

        """
        初始化env
        """
        self.env = CityflowEnvWrapper(self.args, self.flow, env_seed=self.base_seed)
        self.ts_ids = []
        for inter_id in self.env.intersection_ids:
            self.ts_ids.append(inter_id)
        self.num_agents = len(self.ts_ids)
        self.comm_buffer = CommBuffers(
            episode_length=self.episode_length,
            N=self.num_agents,
            h_dim=self.args.ht_dim,
            infor_dim=self.args.infor_dim,
            device=self.device,
        )
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
        # EP stands for Environment Provided, as phrased by MAPPO paper.
        # In EP, the global states for all agents are the same.
        self.rewards = np.zeros((self.episode_length, self.num_agents))

        share_obs_size = 0
        for ts in self.env.intersection_ids:
            share_obs_size += self.env.interid2statedim[ts]
        share_obs_size = share_obs_size + self.num_agents * self.args.infor_dim
        self.critic = Critic(args=self.args, share_states_size=share_obs_size)
        # EP stands for Environment Provided, as phrased by MAPPO paper.
        # In EP, the global states for all agents are the same.
        self.critic_buffer = OnPolicyCriticBufferEP(args=args, share_obs_size=share_obs_size)
        if args.use_valuenorm is True:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    # ====================== flow helpers ======================
    @staticmethod
    def _collect_flow_files(flow_dir):
        """只从 flow_dir 文件夹中收集 json flow 文件；不兼容旧版单 flow 文件写法。"""
        if flow_dir is None or str(flow_dir).strip() == "":
            raise ValueError("必须通过 -flow_dir 指定 flow 文件夹。")

        flow_dir = os.path.expanduser(str(flow_dir))
        if not os.path.isdir(flow_dir):
            raise NotADirectoryError(f"-flow_dir 必须是一个文件夹，而不是单个 flow 文件: {flow_dir}")

        flow_files = sorted(glob.glob(os.path.join(flow_dir, "*.json")))
        if len(flow_files) == 0:
            raise FileNotFoundError(f"flow 文件夹中没有找到 json 文件: {flow_dir}")
        return flow_files

    def _sample_train_flow(self):
        """训练时每个 episode 随机抽取一个 flow 文件。

        使用局部 RNG，保证不同算法在相同 args.seed 下得到一致的 flow 序列，
        不受其他 random 调用影响。
        """
        return self.flow_rng.choice(self.flow_files)

    def _set_env_flow(self, flow_file, env_seed):
        """
        切换当前环境使用的 flow 文件，并为本次 rollout 设置独立 env_seed。

        注意：即使 flow_file 与当前 flow 相同，也必须重新创建环境，
        因为同一个 flow 在不同 episode 下也应该生成不同的 missing mask。
        """
        self.flow = flow_file
        self.env = CityflowEnvWrapper(self.args, self.flow, env_seed=env_seed)
        self.current_flow_file = flow_file

    # ====================== rollout / train / validate ======================
    def _rollout_episode(self, episode, deterministic=False, update_buffer=True, write_step_metric=True):
        """
        执行一个完整 episode。

        Args:
            episode: 当前 episode 编号。训练时对应训练 episode，验证时对应 flow 序号。
            deterministic: False 表示训练采样动作；True 表示验证时使用确定性动作。
            update_buffer: True 时写入 actor/critic buffer，用于后续 PPO 更新；False 时只缓存下一步状态供 rollout 继续。
            write_step_metric: 是否记录 step-level metric。
        """
        self.comm_buffer.clear_()
        total_reward = 0
        self.prep_rollout()
        self.warmup()

        for step in range(self.episode_length + 1):
            h_all = torch.empty(self.num_agents, self.args.ht_dim, device=self.device)
            for ts_id in self.actors.keys():
                neigh_infor_prev_dict_i = self.actors[ts_id].policy_network.build_neigh_infor_prev_dict(
                    self.comm_buffer, step
                )
                h_i_t = self.actors[ts_id].get_self_ht(
                    self.actor_buffer[ts_id].states[step],
                    neigh_infor_prev_dict_i,
                )  # [h_dim], no grad
                h_all[ts_id].copy_(h_i_t.squeeze(0))
            self.comm_buffer.set_step_ht(step, h_all)

            (
                actions,
                action_log_probs,
                ts2action,
                rnn_states,
            ) = self.collect(step, deterministic=deterministic)

            if step != self.episode_length:
                next_states, rewards, dones, _ = self.env.step(ts2action)
                states_array, actions_array, rewards_array, rewards_sum = self.handle_states_actions_rewards(
                    ts2action, rewards, next_states
                )
                total_reward += rewards_sum

                if write_step_metric:
                    average_queue_length_step = -rewards_sum / float(self.num_agents)
                    throughput_step = self.env.get_throughput()
                    average_travel_time_step = self.env.get_average_travel_time()
                    average_waiting_time_step = self.env.get_average_queue_time()

                    step_metric_text = "Episode {} Step {}:throughput {},average_queue_length {},average_travel_time {},average_waiting_time {}".format(
                        episode,
                        step,
                        throughput_step,
                        average_queue_length_step,
                        average_travel_time_step,
                        average_waiting_time_step,
                    )
                    with open(self.step_metric_file, "a", encoding="utf-8") as f:
                        f.write(step_metric_text + "\n")

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

                if update_buffer:
                    self.insert(data, step)
                else:
                    self.cache_next_step_for_rollout(states_array, rnn_states, step)

        if deterministic:
            throughput = self.env.get_throughput()
            average_queue_length = -total_reward / len(self.actors.keys()) / self.episode_length
            average_travel_time = self.env.get_average_travel_time()
            average_waiting_time = self.env.get_average_queue_time()

            return {
                "total_reward": total_reward,
                "throughput": throughput,
                "average_queue_length": average_queue_length,
                "average_travel_time": average_travel_time,
                "average_waiting_time": average_waiting_time,
            }
        else:
            return {
                "total_reward": total_reward,
            }

    def run(self):
        self.setup_seed()
        print("start running")
        episodes = self.args.episode
        for episode in tqdm(range(1, episodes + 1)):
            selected_flow = self._sample_train_flow()
            env_seed = self.base_seed + episode
            self._set_env_flow(selected_flow, env_seed=env_seed)

            rollout_info = self._rollout_episode(
                episode=episode,
                deterministic=False,
                update_buffer=True,
                write_step_metric=False,
            )

            self.insert_critic_buffer()
            # rollout 结束后，将全局 CommBuffers 拷到每个 ActorBuffer（all_agents）
            for ts_id in self.actor_buffer.keys():
                self.actor_buffer[ts_id].attach_comm_snapshot(self.comm_buffer, agent_id=ts_id, detach=True)

            # compute return and update network
            self.compute()
            self.prep_training()
            self.train()

            log_text = "Episode {}:{}".format(episode, rollout_info["total_reward"])
            print(log_text)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_text + "\n")

        # 训练结束后，直接用内存中的当前模型遍历同一个 flow_dir 做验证。
        self.validate()

    @torch.no_grad()
    def validate(self):
        """
        验证阶段：训练结束后，使用内存中的模型，遍历 flow_dir 下的每一个 flow 文件。
        这里不重新加载 checkpoint，也不通过 mode/test_flow_dir 控制。

        metric_step 文件只记录 validate 阶段的 step-level metric。
        """
        print(f"start validation on {len(self.flow_files)} flow files")
        results = []

        # validate 开始前清空 step-level metric 文件，避免混入训练阶段或历史运行的记录。
        with open(self.step_metric_file, "w", encoding="utf-8") as f:
            f.write("")

        with open(self.validate_metric_file, "w", encoding="utf-8") as f:
            f.write("flow\tthroughput\taverage_queue_length\taverage_travel_time\taverage_waiting_time\ttotal_reward\n")

        for flow_idx, flow_file in enumerate(tqdm(self.flow_files), start=1):
            env_seed = self.base_seed + 999999 + flow_idx
            self._set_env_flow(flow_file, env_seed=env_seed)

            rollout_info = self._rollout_episode(
                episode=flow_idx,
                deterministic=True,
                update_buffer=False,
                write_step_metric=True,
            )
            rollout_info["flow"] = flow_file
            results.append(rollout_info)

            validate_text = "{}\t{}\t{}\t{}\t{}\t{}".format(
                flow_file,
                rollout_info["throughput"],
                rollout_info["average_queue_length"],
                rollout_info["average_travel_time"],
                rollout_info["average_waiting_time"],
                rollout_info["total_reward"],
            )
            with open(self.validate_metric_file, "a", encoding="utf-8") as f:
                f.write(validate_text + "\n")

        if len(results) > 0:
            metric_names = [
                "throughput",
                "average_queue_length",
                "average_travel_time",
                "average_waiting_time",
                "total_reward",
            ]
            with open(self.validate_metric_file, "a", encoding="utf-8") as f:
                f.write("\nsummary(mean±std)\n")
                for metric in metric_names:
                    values = np.array([r[metric] for r in results], dtype=np.float64)
                    f.write(f"{metric}:{values.mean():.2f}±{values.std():.2f}\n")

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        initial_states = self.env.reset()
        # replay buffer
        for ts in self.actors.keys():
            self.actor_buffer[ts].step = 0
            self.actor_buffer[ts].states[0] = self.get_agent_obs(initial_states, ts).copy()
            self.actor_buffer[ts].reset_rnn_states()
        self.critic_buffer.step = 0
        self.critic_buffer.reset_rnn_states()

    def cache_next_step_for_rollout(self, states_array, rnn_states, step):
        """验证时不写训练 buffer，只缓存下一步状态和 RNN 状态，保证 rollout 能继续。"""
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].states[step + 1] = states_array[agent_id].copy()
            self.actor_buffer[agent_id].rnn_states[step + 1] = rnn_states[agent_id].copy()

    def get_agent_obs(self, states, ts_id):
        state = states[str(ts_id)]
        state = np.array(state, dtype=np.float32)
        # 转换为 NumPy 数组，支持变长
        return state

    def handle_states_actions_rewards(self, actions, rewards, states):
        all_states = []
        all_actions = []
        all_rewards = []

        # 用于存储共享观测值（拼接所有状态）
        share_obs_list = []

        # Process each agent's state, action, and reward
        for ts in self.actors.keys():
            # Encode the state for the current agent
            state = states[str(ts)]
            state = np.array(state, dtype=np.float32)  # 转换为 NumPy 数组

            # 记录到共享观测列表
            share_obs_list.append(state.flatten())  # 展平，保证拼接时一致

            # Get the action and process it
            action = np.array(actions[str(ts)], dtype=np.int64)  # 确保是 NumPy 数组
            reward = np.array(rewards[str(ts)], dtype=np.float32)  # 确保是 NumPy 数组

            # Append the processed values
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)

        # 将变长 state 存储为 object 类型 NumPy 数组
        states_array = np.array(all_states, dtype=object)

        # actions 和 rewards 作为普通 NumPy 数组
        actions_array = np.array(all_actions, dtype=np.int64)  # [n]
        rewards_array = np.array(all_rewards, dtype=np.float32)  # [n]

        # 计算 reward 总和
        rewards_sum = np.sum(rewards_array)
        return states_array, actions_array, rewards_array, rewards_sum

    @torch.no_grad()
    def collect(self, step, deterministic=False):
        """Collect actions and values from actors and critics."""
        # collect actions, action_log_probs, rnn_states from n actors
        infor_all = torch.empty(self.num_agents, self.infor_dim, device=self.device)
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        ts2action = {}
        for ts_id in self.actors.keys():
            neigh_h_dict = {
                int(nid): self.comm_buffer.get_ht(step, int(nid), detach=True).unsqueeze(0)
                for nid in self.id2neighbours[str(ts_id)]
            }
            neigh_infor_prev_dict_i = self.actors[ts_id].policy_network.build_neigh_infor_prev_dict(
                self.comm_buffer, step
            )
            obs_i_t = check(self.actor_buffer[ts_id].states[step]).to(self.device)
            obs_i_t = obs_i_t.unsqueeze(0)
            action, action_log_prob, infor_i_t, rnn_state = self.actors[ts_id].policy_network.forward(
                obs_i_t=obs_i_t,
                rnn_states=self.actor_buffer[ts_id].rnn_states[step],
                neigh_infor_prev_dict=neigh_infor_prev_dict_i,
                neigh_h_t=neigh_h_dict,
                deterministic=deterministic,
                stop_grad_neighbors=True,
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
        return actions, action_log_probs, ts2action, rnn_states

    def insert(self, data, step):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            rnn_states,
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
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
        """Train the model."""
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
            advantages = self.critic_buffer.returns[:-1] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            advantages = self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
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
            neigh_prev_info_ep = self.actors[agent_id].policy_network.build_episode_neigh_infor_prev_dict(
                self.comm_buffer, T=self.episode_length, B=1
            )
            neigh_h_ep = self.actors[agent_id].policy_network.build_episode_neigh_h_dict(
                self.comm_buffer, T=self.episode_length, B=1, detach=True
            )

            old_actions_logprob, _, _ = self.actors[agent_id].evaluate_actions(
                local_state=self.actor_buffer[agent_id].states[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].states.shape[1:]
                ),
                rnn_states=self.actor_buffer[agent_id].rnn_states[0],
                neigh_infor_prev_dict=neigh_prev_info_ep,
                neigh_ht_dict=neigh_h_ep,
                action=self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[1:]
                ),
            )

            # update actor
            self.actors[agent_id].train(
                self.actor_buffer[agent_id], neigh_prev_info_ep, neigh_h_ep
            )
            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actors[agent_id].evaluate_actions(
                local_state=self.actor_buffer[agent_id].states[:-1].reshape(
                    -1, *self.actor_buffer[agent_id].states.shape[1:]
                ),
                rnn_states=self.actor_buffer[agent_id].rnn_states[0],
                neigh_infor_prev_dict=neigh_prev_info_ep,
                neigh_ht_dict=neigh_h_ep,
                action=self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[1:]
                ),
            )

            # update factor for next agent
            factor = factor * _t2n(
                torch.exp(new_actions_logprob - old_actions_logprob).reshape(
                    self.episode_length,
                    1,
                )
            )

        # update critic
        self.critic.train(self.critic_buffer, self.value_normalizer)

    def prep_rollout(self):
        """Prepare for rollout."""
        for ts in self.actors.keys():
            self.actors[ts].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
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
        """
        将每一步的 share_state 插入 Critic Buffer 中
        先组装好单个 actor 的 state：[obs_t, infor_t]，再将所有的 state 拼接得到 share_obs
        最后写入 self.critic_buffer.share_obs[step] = share_obs
        """
        # 获取所有的 `infor` 和 `ht` 数据，并转移到 CPU，转换为 NumPy 数组
        _, infor = self.comm_buffer.to_cpu_and_numpy()  # 获取整个 episode 的 infor 数据（已转为 NumPy 数组）

        # 遍历每个 step
        for step in range(self.episode_length + 1):
            step_share_obs = []

            # 遍历所有 actor，获取本地观测和当前 step 自己计算的 infor
            for ts_id in self.actors.keys():
                # 获取当前 step 步的 actor state（本地观测）
                actor_state = self.actor_buffer[ts_id].states[step]  # 获取当前 step 的状态

                # 获取当前时间步的自己计算的 infor（即该智能体自己计算的 `infor_t`）
                actor_infor = infor[step, ts_id]  # 获取当前智能体的 infor（自己计算的）

                # 拼接本地状态 `obs_t` 和 当前计算的 `infor_t`
                share_state = np.concatenate([actor_state.flatten(), actor_infor.flatten()], axis=-1)

                # 将拼接后的 `share_state` 存入该 step 的 list
                step_share_obs.append(share_state)

            # 将所有智能体的 `share_state` 拼接成该 step 的 `share_obs`
            share_obs = np.concatenate([obs.flatten() for obs in step_share_obs])
            value, rnn_state_critic = self.critic.get_values(
                share_obs,
                rnn_states_critic=self.critic_buffer.rnn_states_critic[step],
            )
            value = _t2n(value)
            rnn_state_critic = _t2n(rnn_state_critic)

            # 将当前 step 的 `share_obs` 写入 Critic Buffer
            self.critic_buffer.share_obs[step] = share_obs
            self.critic_buffer.value_preds[step] = value
            if step != self.episode_length:
                self.critic_buffer.rnn_states_critic[step + 1] = rnn_state_critic

    @torch.no_grad()
    def compute(self):
        """
        Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        next_value = self.critic_buffer.value_preds[-1]
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)
