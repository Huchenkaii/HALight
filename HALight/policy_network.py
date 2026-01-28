import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from typing import Dict, List, Optional, Tuple
from models.rnn_layer import RNNLayer

class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class ACTLayer(nn.Module):
    def __init__(self, action_dim, inputs_dim):
        super(ACTLayer, self).__init__()
        self.action_dim = action_dim
        self.fc = nn.Linear(inputs_dim, action_dim)
        nn.init.orthogonal_(self.fc.weight, gain=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x,deterministic=False):
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        actions = (
            action_distribution.mode()
            if deterministic
            else action_distribution.sample()
        )
        action_log_probs = action_distribution.log_probs(actions)
        return actions, action_log_probs

    def get_logits(self, x):
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        action_logits = action_distribution.logits
        return action_logits

    def evaluate_actions(self, x, action):
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        action_log_probs = action_distribution.log_probs(action)
        dist_entropy = action_distribution.entropy().mean()
        return action_log_probs, dist_entropy, action_distribution


class HALight_policy_network(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        infor_dim: int = 8,
        h_hidden: int = 16,
        attn_dim: int = 16,
        trunk_hidden: int = 64,
        neighbours: Optional[List[int]] = None,
    ):
        super().__init__()
        self.neighbours = [] if neighbours is None else [int(x) for x in neighbours]
        self.M = len(self.neighbours)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc_in = state_size + self.M * infor_dim
        self.enc = nn.Sequential(
            nn.Linear(enc_in, h_hidden), nn.GELU(), nn.LayerNorm(h_hidden)
        )
        for m in self.enc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

        self.Wq = nn.Linear(h_hidden, attn_dim, bias=False)
        self.Wk = nn.Linear(h_hidden, attn_dim, bias=False)
        self.Wv = nn.Linear(h_hidden, attn_dim, bias=False)
        nn.init.orthogonal_(self.Wq.weight, gain=1.0)
        nn.init.orthogonal_(self.Wk.weight, gain=1.0)
        nn.init.orthogonal_(self.Wv.weight, gain=1.0)
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.scale = attn_dim ** 0.5

        self.infor_head = nn.Sequential(
            nn.Linear(attn_dim, infor_dim)
        )
        for m in self.infor_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

        fused_dim = state_size + infor_dim
        self.trunk_norm = nn.LayerNorm(fused_dim)
        self.trunk_fc1 = nn.Linear(fused_dim, trunk_hidden)
        self.trunk_fc2 = nn.Linear(trunk_hidden, trunk_hidden)
        self.trunk_ln1 = nn.LayerNorm(trunk_hidden)
        self.trunk_ln2 = nn.LayerNorm(trunk_hidden)
        nn.init.orthogonal_(self.trunk_fc1.weight, gain=1.0); nn.init.zeros_(self.trunk_fc1.bias)
        nn.init.orthogonal_(self.trunk_fc2.weight, gain=1.0); nn.init.zeros_(self.trunk_fc2.bias)

        self.rnn = RNNLayer(trunk_hidden, trunk_hidden)
        self.act = ACTLayer(action_size, trunk_hidden)

        self.state_size = state_size
        self.infor_dim = infor_dim
        self.h_hidden = h_hidden
        self.attn_dim = attn_dim

    def _trunk(self, fused: torch.Tensor) -> torch.Tensor:
        x = self.trunk_norm(fused)
        x = F.relu(self.trunk_fc1(x)); x = self.trunk_ln1(x)
        x = F.relu(self.trunk_fc2(x)); x = self.trunk_ln2(x)
        return x

    @torch.no_grad()
    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(mask == 0, float('-inf'))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        obs_i_t: torch.Tensor,                               # [B, state_size]
        rnn_states:Dict[numpy.ndarray, torch.Tensor],
        neigh_infor_prev_dict: Dict[int, torch.Tensor],
        neigh_h_t: Dict[int, torch.Tensor],
        neigh_mask: Optional[torch.Tensor] = None,
        neigh_ids_in_order: Optional[List[int]] = None,
        deterministic: bool = False,
        stop_grad_neighbors: bool = True,
    ):
        """
        Returns:
            actions:          [B]
            action_log_probs: [B]
            infor_i_t:        [B, infor_dim]
        """
        B = obs_i_t.size(0)
        rnn_states = check(rnn_states).to(self.device)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "The number of neighbors must be consistent with initialization (each agent has a fixed number of first-order neighbors)."


        prev_list = []
        for nid in order:
            x = neigh_infor_prev_dict[nid]
            if x is None:
                x = torch.zeros(B, self.infor_dim, device=obs_i_t.device, dtype=obs_i_t.dtype)
            prev_list.append(x)
        prev_stack = torch.stack(prev_list, dim=1)
        prev_flat = prev_stack.reshape(B, self.M * self.infor_dim)
        enc_in = torch.cat([obs_i_t, prev_flat], dim=-1)
        h_i_t = self.enc(enc_in)

        if self.M == 0:
            h_prime = torch.zeros(B, self.attn_dim, device=obs_i_t.device)
        else:
            H_nei = torch.stack(
                [(neigh_h_t[nid].detach() if stop_grad_neighbors else neigh_h_t[nid]) for nid in order],
                dim=1
            )
            # [B, M, h_hidden]
            Q = self.Wq(h_i_t).unsqueeze(1)                        # [B,1,D]
            K = self.Wk(H_nei)                                     # [B,M,D]
            V = self.Wv(H_nei)                                     # [B,M,D]
            attn_logits = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B,1,M]
            if neigh_mask is not None:
                attn = self._masked_softmax(attn_logits.squeeze(1), neigh_mask).unsqueeze(1)
            else:
                attn = torch.softmax(attn_logits, dim=-1)          # [B,1,M]
            h_prime = torch.matmul(attn, V).squeeze(1)             # [B,D]
            h_prime = self.attn_norm(h_prime)

        infor_i_t = torch.sigmoid(self.infor_head(h_prime))         # [B, infor_dim]

        fused = torch.cat([obs_i_t, infor_i_t], dim=-1)             # [B, state_size+infor_dim]
        actor_feat = self._trunk(fused)
        actor_features, rnn_states = self.rnn(actor_feat, rnn_states)
        actions, action_log_probs = self.act(actor_features, deterministic=deterministic)

        return actions, action_log_probs, infor_i_t,rnn_states

    @torch.no_grad()
    def compute_ht(
        self,
        obs_i_t: torch.Tensor,                              # [B, state_size]
        neigh_infor_prev_dict: Dict[int, torch.Tensor],
        neigh_ids_in_order: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B = obs_i_t.size(0)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "The number of neighbors must be consistent with the initialization."

        prev_list = []
        for nid in order:
            x = neigh_infor_prev_dict[nid]
            if x is None:
                x = torch.zeros(B, self.infor_dim, device=obs_i_t.device, dtype=obs_i_t.dtype)
            prev_list.append(x)
        prev_stack = torch.stack(prev_list, dim=1)                 # [B, M, infor_dim]
        prev_flat  = prev_stack.reshape(B, self.M * self.infor_dim)# [B, M*infor_dim]
        enc_in = torch.cat([obs_i_t, prev_flat], dim=-1)           # [B, state + M*infor]

        h_i_t = self.enc(enc_in)                                       # [B, h_hidden]
        return h_i_t

    @torch.no_grad()
    def build_neigh_infor_prev_dict(
        self,
        comm_buf,
        t: int,
        B: int = 1,
        neigh_ids_in_order: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        device = comm_buf.device
        dtype = comm_buf.infor.dtype

        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        neigh_infor_prev_dict: Dict[int, torch.Tensor] = {}

        if t <= 0:
            zero_row = torch.zeros(1, self.infor_dim, device=device, dtype=dtype).expand(B, -1)  # [B, D]
            for m, nid in enumerate(order):
                neigh_infor_prev_dict[nid] = zero_row
            return neigh_infor_prev_dict  # 全 0

        prev_all = comm_buf.get_step_infor(t-1, detach=True)  # [N, infor_dim]

        for m, nid in enumerate(order):
            row = prev_all[nid]  # [infor_dim]
            neigh_infor_prev_dict[nid] = row.unsqueeze(0).expand(B, -1)

        return neigh_infor_prev_dict

    @torch.no_grad()
    def build_neigh_h_t_dict(
            self,
            comm_buf,  # CommBuffers
            t: int,
            neigh_ids_in_order: Optional[List[int]] = None,
            detach: bool = True,
    ) -> Dict[int, torch.Tensor]:
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        H_t = comm_buf.get_step_ht(t, detach=detach)  # [N, h_dim]
        out = {}
        for nid in order:
            out[nid] = H_t[nid].unsqueeze(0)  # [1, h_dim]，和 B=1 的前向对齐；需要更大 B 时你自己堆叠
        return out

    @torch.no_grad()
    def build_episode_neigh_infor_prev_dict(
            self,
            comm_buf,
            T: int,
            B: int = 1,
            neigh_ids_in_order: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        device = comm_buf.device
        dtype = comm_buf.infor.dtype
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        T_eff = min(T, comm_buf.steps)  # comm_buf.steps=episode_length+1；这里 T 通常≤episode_length

        out: Dict[int, torch.Tensor] = {}
        if T_eff <= 0 or len(order) == 0:
            return {nid: torch.zeros(0, B, self.infor_dim, device=device, dtype=dtype) for nid in order}

        zero_blk = torch.zeros(1, B, self.infor_dim, device=device, dtype=dtype)

        for nid in order:
            if T_eff == 1:
                seq = zero_blk
            else:
                prev_seq = comm_buf.infor[:T_eff - 1, nid, :]  # [T-1, D]
                prev_seq = prev_seq.unsqueeze(1).expand(-1, B, -1)  # [T-1, B, D]
                seq = torch.cat([zero_blk, prev_seq], dim=0)  # [T, B, D]

            out[nid] = seq

        return out

    @torch.no_grad()
    def build_episode_neigh_h_dict(
        self,
        comm_buf,
        T: int,
        B: int = 1,
        neigh_ids_in_order: Optional[List[int]] = None,
        detach: bool = True,
    ) -> Dict[int, torch.Tensor]:
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        T_eff = min(T, comm_buf.steps)
        ht_all = comm_buf.ht            # [steps, N, h_dim]
        if detach:
            ht_all = ht_all.detach()

        out: Dict[int, torch.Tensor] = {}
        if T_eff <= 0 or len(order) == 0:
            device = comm_buf.device
            out = {nid: torch.zeros(0, B, self.h_hidden, device=device, dtype=comm_buf.ht.dtype) for nid in order}
            return out

        for nid in order:
            h_seq = ht_all[:T_eff, nid, :]                      # [T, D]
            h_seq = h_seq.unsqueeze(1).expand(-1, B, -1)        # [T, B, D]
            out[nid] = h_seq
        return out

    def evaluate_actions(
        self,
        obs_i_t: torch.Tensor,                               # [B, state_size]
        rnn_states: Dict[numpy.ndarray, torch.Tensor],
        neigh_infor_prev_dict: Dict[int, torch.Tensor],
        neigh_h_t: Dict[int, torch.Tensor],
        neigh_mask: Optional[torch.Tensor] = None,
        neigh_ids_in_order: Optional[List[int]] = None,
        stop_grad_neighbors: bool = True,
        action = None
    ):
        rnn_states = check(rnn_states).to(self.device)
        action = check(action).to(self.device)
        B = obs_i_t.size(0)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "The number of neighbors must be consistent with the initialization."

        prev_list = []
        for nid in order:
            x = neigh_infor_prev_dict[nid]
            if x is None:
                x = torch.zeros(B, self.infor_dim, device=obs_i_t.device, dtype=obs_i_t.dtype)
            prev_list.append(x)
        prev_stack = torch.stack(prev_list, dim=1)                 # [B, M, infor_dim]
        prev_flat  = prev_stack.reshape(B, self.M * self.infor_dim)# [B, M*infor_dim]
        enc_in = torch.cat([obs_i_t, prev_flat], dim=-1)
        h_i_t = self.enc(enc_in)                                   # [B, h_hidden]

        if self.M == 0:
            h_prime = torch.zeros(B, self.attn_dim, device=obs_i_t.device)
        else:
            H_nei = torch.stack(
                [(neigh_h_t[nid].detach() if stop_grad_neighbors else neigh_h_t[nid]) for nid in order],
                dim=1
            )                                                      # [B, M, h_hidden]
            Q = self.Wq(h_i_t).unsqueeze(1)                        # [B,1,D]
            K = self.Wk(H_nei)
            V = self.Wv(H_nei)
            K = K.squeeze(-2)                                      # [B, M, D]
            V = V.squeeze(-2)                                      # [B, M, D]
            attn_logits = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [B,1,M]
            if neigh_mask is not None:
                attn = self._masked_softmax(attn_logits.squeeze(1), neigh_mask).unsqueeze(1)
            else:
                attn = torch.softmax(attn_logits, dim=-1)          # [B,1,M]
            h_prime = torch.matmul(attn, V).squeeze(1)             # [B,D]
            h_prime = self.attn_norm(h_prime)

        infor_i_t = torch.sigmoid(self.infor_head(h_prime))         # [B, infor_dim]

        fused = torch.cat([obs_i_t, infor_i_t], dim=-1)             # [B, state_size+infor_dim]
        actor_feat = self._trunk(fused)
        actor_features, rnn_states = self.rnn(actor_feat, rnn_states)

        # Evaluate action distribution
        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(actor_features, action)
        return action_log_probs, dist_entropy, action_distribution
