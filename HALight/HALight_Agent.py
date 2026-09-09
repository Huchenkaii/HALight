import numpy as np
from harl.utils.envs_tools import check
import random
import torch.optim as optim
from policy_network import HALight_policy_network
import torch
import copy
from typing import Dict, List, Union, Iterable, Optional

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Agent:
    def __init__(self, args,action_size,state_size,neighbours,ts_id):
        self.state_size = state_size
        self.action_size = action_size
        self.ts_id = ts_id
        self.lr = args.lr
        self.gamma = args.gamma
        if args.seed != None:
            setup_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.actor_num_mini_batch = args.actor_num_mini_batch
        self.entropy_coef = args.entropy_coef
        self.neighbours = neighbours
        self.neighbours = [int(x) for x in neighbours]
        self.policy_network = HALight_policy_network(state_size,action_size,neighbours=self.neighbours,infor_dim=args.infor_dim,h_hidden=args.ht_dim,trunk_hidden=args.hidden_sizes,attn_dim=args.attn_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr,eps=0.00001)

    def update(self, sample):
        (states_batch,rnn_states_batch,
         neigh_prev_info_batch_dict, neigh_h_batch_dict,
         actions_batch, old_action_log_probs_batch, adv_targ, factor_batch)\
        = sample


        states_batch = check(states_batch).to(self.device)
        rnn_states_batch = check(rnn_states_batch).to(self.device)
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(self.device)
        adv_targ = check(adv_targ).to(self.device)
        factor_batch = check(factor_batch).to(self.device)


        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = self.policy_network.evaluate_actions(states_batch,rnn_states_batch,neigh_prev_info_batch_dict,neigh_h_batch_dict,action=actions_batch)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = (torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ)

        policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        # actor_grad_norm = nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10)

        self.actor_optimizer.step()

        # return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer,neigh_prev_info_ep,neigh_h_ep):
        advantages_copy = actor_buffer.advantage.copy()

        for _ in range(self.ppo_epoch):
            data_generator = actor_buffer.naive_recurrent_generator_actor(advantages_copy, neigh_prev_info_ep, neigh_h_ep,actor_num_mini_batch=self.actor_num_mini_batch)

            for sample in data_generator:
                self.update(sample)
        return
    def prep_training(self):
        """Prepare for training."""
        self.policy_network.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.policy_network.eval()


    def evaluate_actions(self,local_state,rnn_states,neigh_infor_prev_dict,neigh_ht_dict,action):
        local_state = check(local_state).to(self.device)
        (
            action_log_probs,
            dist_entropy,
            action_distribution,
        ) = self.policy_network.evaluate_actions(obs_i_t=local_state,rnn_states=rnn_states,neigh_infor_prev_dict=neigh_infor_prev_dict,neigh_h_t=neigh_ht_dict,action=action)
        return action_log_probs, dist_entropy, action_distribution

    def get_self_ht(self,obs,neigh_infor_prev_dict):
        obs = check(obs).to(self.device)
        obs = obs.unsqueeze(0)
        return self.policy_network.compute_ht(obs,neigh_infor_prev_dict)
