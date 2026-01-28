import torch
import numpy as np
from typing import Dict, List, Optional

class ActorBuffer:
    """On-policy buffer for actor data storage."""
    def __init__(self, args, state_size, action_size, neighbours, ts_id,
                 h_dim: Optional[int] = None, infor_dim: Optional[int] = None):
        self.state_size = state_size
        self.episode_length = args.episode_length
        self.hidden_sizes = args.hidden_sizes
        self.rnn_hidden_size = args.hidden_sizes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.neighbours = [int(x) for x in neighbours]
        self.ts_id = ts_id
        self.action_size = action_size

        self.states = np.zeros((self.episode_length + 1, self.state_size), dtype=np.float32)
        self.rnn_states = np.zeros((self.episode_length + 1, self.rnn_hidden_size), dtype=np.float32, )
        self.actions = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.advantage        = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.states_array_dict = {}
        self.states_array = np.empty(self.episode_length + 1, dtype=object)

        self.factor = None
        self.step = 0

        self.comm_all_ht: Optional[torch.Tensor]   = None  # [steps, N, h_dim]
        self.comm_all_info: Optional[torch.Tensor] = None  # [steps, N, infor_dim]
        self.h_dim = h_dim
        self.infor_dim = infor_dim
        self.num_agents: Optional[int] = None
        self.steps: Optional[int] = None

        self.agent_id: Optional[int] = None

    def update_factor(self, factor):
        self.factor = factor.copy()

    def set_states_array(self):
        for step in self.states_array_dict.keys():
            self.states_array[step] = self.states_array_dict[step]

    def insert(self, state,rnn_states, actions, action_log_probs):
        self.states[self.step + 1] = state.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.step = (self.step + 1) % self.episode_length

    def reset_rnn_states(self):
        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.rnn_hidden_size),
            dtype=np.float32
        )
        return self.rnn_states[0]


    @torch.no_grad()
    def attach_comm_snapshot(
        self,
        comm_buf,
        agent_id: int,
        detach: bool = True,
        move_to_device: Optional[torch.device] = None,
    ):
        dev = move_to_device or self.device

        ht = comm_buf.ht    # [steps, N, h_dim]
        info = comm_buf.infor  # [steps, N, infor_dim]
        if detach:
            ht = ht.detach()
            info = info.detach()

        self.steps = ht.shape[0]
        self.num_agents = ht.shape[1]
        if self.h_dim is None:     self.h_dim = ht.shape[-1]
        if self.infor_dim is None: self.infor_dim = info.shape[-1]
        self.agent_id = int(agent_id)

        self.comm_all_ht   = ht.to(dev).clone()     # [steps, N, h_dim]
        self.comm_all_info = info.to(dev).clone()   # [steps, N, infor_dim]


    def get_step_ht(self, t: int, *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t)
        out = self.comm_all_ht[t]
        return out.detach() if detach else out

    def get_step_infor(self, t: int, *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t)
        out = self.comm_all_info[t]
        return out.detach() if detach else out

    def get_agent_ht(self, t: int, agent_id: int, *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t); self._check_i(agent_id)
        out = self.comm_all_ht[t, agent_id]
        return out.detach() if detach else out

    def get_agent_infor(self, t: int, agent_id: int, *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t); self._check_i(agent_id)
        out = self.comm_all_info[t, agent_id]
        return out.detach() if detach else out

    def get_self_ht(self, t: int, *, detach: bool = True) -> torch.Tensor:
        return self.get_agent_ht(t, self.agent_id, detach=detach)

    def get_self_infor(self, t: int, *, detach: bool = True) -> torch.Tensor:
        return self.get_agent_infor(t, self.agent_id, detach=detach)

    def get_neighbors_ht(self, t: int, neighbor_ids: List[int], *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t)
        idx = torch.as_tensor(neighbor_ids, device=self.comm_all_ht.device, dtype=torch.long)
        out = self.comm_all_ht[t].index_select(0, idx)
        return out.detach() if detach else out

    def get_neighbors_infor(self, t: int, neighbor_ids: List[int], *, detach: bool = True) -> torch.Tensor:
        self._check_ready()
        self._check_t(t)
        idx = torch.as_tensor(neighbor_ids, device=self.comm_all_info.device, dtype=torch.long)
        out = self.comm_all_info[t].index_select(0, idx)
        return out.detach() if detach else out

    def _check_ready(self):
        if self.comm_all_ht is None or self.comm_all_info is None:
            raise RuntimeError("Communication snapshot is not attached: please call attach_comm_snapshot(..., mode='all_agents') first.")

    def _check_t(self, t: int):
        if self.steps is None:
            raise RuntimeError("steps Not initialized.")
        if not (0 <= t < self.steps):
            raise IndexError(f"t={t} out of range [0, {self.steps - 1}]")

    def _check_i(self, i: int):
        if self.num_agents is None:
            raise RuntimeError("num_agents Not initialized.")
        if not (0 <= i < self.num_agents):
            raise IndexError(f"agent_id={i} out of range [0, {self.num_agents - 1}]")


    def naive_recurrent_generator_actor(
            self, advantages, neigh_prev_info_ep, neigh_h_ep,
            actor_num_mini_batch=None, mini_batch_size=None
    ):
        T = self.episode_length
        obs_batch = self.states[:-1].reshape(T, -1)  #  (T+1, obs_dim) → (T, obs_dim)
        actions_batch = self.actions.reshape(T, -1)
        old_action_log_probs_batch = self.action_log_probs.reshape(T, -1)
        adv_targ = advantages.reshape(T, 1)

        factor_batch = None
        if self.factor is not None:
            factor_batch = self.factor.reshape(T, -1)  # (T, factor_dim)
        rnn_states_batch = self.rnn_states[0]
        full_indices = np.arange(T, dtype=np.int64)
        neigh_prev_info_batch_dict = {}
        for nid, info_seq in neigh_prev_info_ep.items():
            idx_t = torch.as_tensor(full_indices, device=info_seq.device, dtype=torch.long)
            neigh_prev_info_batch_dict[nid] = info_seq.index_select(dim=0, index=idx_t)
        neigh_h_batch_dict = {}
        for nid, h_seq in neigh_h_ep.items():
            idx_t = torch.as_tensor(full_indices, device=h_seq.device, dtype=torch.long)
            neigh_h_batch_dict[nid] = h_seq.index_select(dim=0, index=idx_t)

        if self.factor is not None:
            yield obs_batch, rnn_states_batch,neigh_prev_info_batch_dict, neigh_h_batch_dict,actions_batch, old_action_log_probs_batch, adv_targ, factor_batch
        else:
            yield obs_batch, rnn_states_batch, neigh_prev_info_batch_dict, neigh_h_batch_dict,actions_batch, old_action_log_probs_batch, adv_targ
