"""HAPPO Agent class."""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from typing import Dict, List, Optional, Tuple
from models.rnn_layer import RNNLayer

class FixedCategorical(torch.distributions.Categorical):
    """Modify standard PyTorch Categorical."""

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
    """MLP Module to compute actions."""
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
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            x: (torch.Tensor) input to network.
            action: (torch.Tensor) actions whose entropy and log probability to evaluate.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        x = self.fc(x)
        action_distribution = FixedCategorical(logits=x)
        action_log_probs = action_distribution.log_probs(action)
        dist_entropy = action_distribution.entropy().mean()
        return action_log_probs, dist_entropy, action_distribution


class HALight_policy_network(nn.Module):
    """
    (9)  h_i^t = Enc_i([obs_i^t, infor_{j1}^{t-1}, ..., infor_{jM}^{t-1}])      # 逐邻居拼接
    (10/11) h'_i^t = Attn_i(h_i^t, {h_j^t}_{j∈N(i)})                             # 邻居 h^t 以 stop-grad 使用
    (13) infor_i^t = sigmoid(MLP_i(h'_i^t))
    policy 输入: [obs_i^t, infor_i^t]
    """

    def __init__(
        self,
        state_size: int,            # obs_i^t 维度（已包含上一轮动作）
        action_size: int,
        infor_dim: int = 8,        # 每个邻居 infor 的维度（统一维）
        h_hidden: int = 16,         # h_i^t 维度
        attn_dim: int = 16,         # 注意力通道维
        trunk_hidden: int = 64,
        neighbours: Optional[List[int]] = None,  # 固定一阶邻居 id 列表（顺序固定）
    ):
        super().__init__()
        self.neighbours = [] if neighbours is None else [int(x) for x in neighbours]
        self.M = len(self.neighbours)                     # 该智能体的邻居数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ---- Enc_i ：式(9)（本地 + 逐邻居 infor_{t-1} 拼接）----
        enc_in = state_size + self.M * infor_dim          # ★ 修正点：逐邻居拼接后的总维
        self.enc = nn.Sequential(
            nn.Linear(enc_in, h_hidden), nn.GELU(), nn.LayerNorm(h_hidden)
        )
        for m in self.enc:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

        # ---- 注意力：式(10)(11) ----
        self.Wq = nn.Linear(h_hidden, attn_dim, bias=False)
        self.Wk = nn.Linear(h_hidden, attn_dim, bias=False)
        self.Wv = nn.Linear(h_hidden, attn_dim, bias=False)
        nn.init.orthogonal_(self.Wq.weight, gain=1.0)
        nn.init.orthogonal_(self.Wk.weight, gain=1.0)
        nn.init.orthogonal_(self.Wv.weight, gain=1.0)
        self.attn_norm = nn.LayerNorm(attn_dim)
        self.scale = attn_dim ** 0.5

        # ---- infor 解码头：式(13) ----
        self.infor_head = nn.Sequential(
            nn.Linear(attn_dim, infor_dim)
        )
        for m in self.infor_head:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0); nn.init.zeros_(m.bias)

        # ---- policy trunk：输入 [obs, infor] ----
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

        # meta
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
        neigh_infor_prev_dict: Dict[int, torch.Tensor],      # {nid: [B, infor_dim]} 逐邻居的 t-1 infor
        neigh_h_t: Dict[int, torch.Tensor],                  # {nid: [B, h_hidden]}  邻居的 h_j^t（rollout 缓存）
        neigh_mask: Optional[torch.Tensor] = None,           # [B, M] 1/0，可选
        neigh_ids_in_order: Optional[List[int]] = None,      # 若不传则默认用 self.neighbours 的顺序
        deterministic: bool = False,
        stop_grad_neighbors: bool = True,
    ):
        """
        Returns:
            actions:          [B]
            action_log_probs: [B]
            infor_i_t:        [B, infor_dim]  —— 本步发送给邻居
        """
        B = obs_i_t.size(0)
        rnn_states = check(rnn_states).to(self.device)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "邻居数量需与初始化一致（每体固定一阶邻居数）。"

        # ---- (9) 逐邻居拼接 infor_{t-1} + 本地 obs → h_i^t ----
        # 若某邻居暂缺消息，自动用 0 向量占位（保证维度一致）
        prev_list = []
        for nid in order:
            x = neigh_infor_prev_dict[nid]
            if x is None:
                x = torch.zeros(B, self.infor_dim, device=obs_i_t.device, dtype=obs_i_t.dtype)
            prev_list.append(x)
        prev_stack = torch.stack(prev_list, dim=1)                 # [B, M, infor_dim]
        prev_flat  = prev_stack.reshape(B, self.M * self.infor_dim)# [B, M*infor_dim]
        enc_in = torch.cat([obs_i_t, prev_flat], dim=-1)           # ★ 修正点
        h_i_t = self.enc(enc_in)                                   # [B, h_hidden]

        # ---- (10)(11) 注意力：h_i^t 与 {h_j^t} → h'_i^t ----
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

        # ---- (13) infor_i^t ----
        infor_i_t = torch.sigmoid(self.infor_head(h_prime))         # [B, infor_dim]

        # ---- policy 输入：[obs, infor] ----
        fused = torch.cat([obs_i_t, infor_i_t], dim=-1)             # [B, state_size+infor_dim]
        actor_feat = self._trunk(fused)
        actor_features, rnn_states = self.rnn(actor_feat, rnn_states)
        actions, action_log_probs = self.act(actor_features, deterministic=deterministic)

        return actions, action_log_probs, infor_i_t,rnn_states

    @torch.no_grad()
    def compute_ht(
        self,
        obs_i_t: torch.Tensor,                              # [B, state_size]
        neigh_infor_prev_dict: Dict[int, torch.Tensor],     # {nid: [B, infor_dim]} 逐邻居 t-1 infor
        neigh_ids_in_order: Optional[List[int]] = None,     # 不传则按 self.neighbours 的固定顺序
    ) -> torch.Tensor:
        """
        计算本轮要发送给邻居的 h_i^t（式(9)），不保留梯度。

        输入：
          - obs_i_t: 本地观测（已包含上一轮动作）
          - neigh_infor_prev_dict: 每个一阶邻居在 t-1 的 infor（逐邻居提供）
          - neigh_ids_in_order: 可覆盖默认邻居顺序（可选）

        返回：
          - h_i_t: [B, h_hidden]，可直接缓存到 rollout buffer 供邻居读取
        """
        B = obs_i_t.size(0)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "邻居数量需与初始化一致。"

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
        comm_buf,                 # CommBuffers 实例
        t: int,                   # 当前 step（我们会取 t-1）
        B: int = 1,               # 当前 batch 大小（obs 的 batch 维）；单步 rollout 就是 1
        neigh_ids_in_order: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        从 comm_buf 取出 t-1 时刻的一阶邻居 infor，组装为:
            { nid: [B, infor_dim] }，以及邻居 mask [B, M] (1/0)

        规则：
          - t==0 时没有上一帧，返回全 0。
          - 若某邻居在 t-1 未写入且 use_zero_if_missing=True，则用 0 占位。
          - 返回的张量 device/dtype 与 buffer 保持一致；必要时会在 batch 维上做 expand。
        """
        device = comm_buf.device
        dtype  = comm_buf.infor.dtype

        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        neigh_infor_prev_dict: Dict[int, torch.Tensor] = {}

        if t <= 0:
            # 没有上一帧，全部 0
            zero_row = torch.zeros(1, self.infor_dim, device=device, dtype=dtype).expand(B, -1)  # [B, D]
            for m, nid in enumerate(order):
                neigh_infor_prev_dict[nid] = zero_row
            return neigh_infor_prev_dict  # 全 0

        # 正常从 t-1 取
        prev_all = comm_buf.get_step_infor(t-1, detach=True)  # [N, infor_dim]
        # 可选：如果你在 CommBuffers 里维护了写入标记，也可以在这里做严格检查

        for m, nid in enumerate(order):
            # 取出该邻居在 t-1 的 infor: [D] -> [B, D]
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
        """
        从 comm_buf 取出当前步 t 的邻居 h^t，返回 { nid: [1, h_hidden] }（带 batch 维 1）。
        训练时会作为 stop-grad 输入；rollout 时也是即时读取。
        """
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        H_t = comm_buf.get_step_ht(t, detach=detach)  # [N, h_dim]
        out = {}
        for nid in order:
            out[nid] = H_t[nid].unsqueeze(0)  # [1, h_dim]，和 B=1 的前向对齐；需要更大 B 时你自己堆叠
        return out

    @torch.no_grad()
    def build_episode_neigh_infor_prev_dict(
            self,
            comm_buf,  # CommBuffers
            T: int,  # 动作步数（通常是 episode_length）
            B: int = 1,  # 批大小（obs 的 batch 维）
            neigh_ids_in_order: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        返回整个 episode 的 t-1 帧邻居 infor 快照：
            { nid: [T, B, infor_dim] }
        规则：
          - 第 0 步自动补 0（没有上一帧）
          - 第 t>0 步使用 comm_buf.infor[t-1, nid]
        """
        device = comm_buf.device
        dtype = comm_buf.infor.dtype
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        T_eff = min(T, comm_buf.steps)  # comm_buf.steps=episode_length+1；这里 T 通常≤episode_length

        out: Dict[int, torch.Tensor] = {}
        if T_eff <= 0 or len(order) == 0:
            return {nid: torch.zeros(0, B, self.infor_dim, device=device, dtype=dtype) for nid in order}

        # 预先构造一个 [1, B, D] 的全 0，用于 t=0
        zero_blk = torch.zeros(1, B, self.infor_dim, device=device, dtype=dtype)

        for nid in order:
            # comm_buf.infor 索引：
            #   t=1..T_eff-1 用 infor[t-1, nid] => 切片 [0..T_eff-2, nid, :]
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
        comm_buf,                                  # CommBuffers
        T: int,                                    # 动作步数（通常是 episode_length）
        B: int = 1,
        neigh_ids_in_order: Optional[List[int]] = None,
        detach: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        返回整个 episode 的邻居 h^t：
            { nid: [T, B, h_hidden] }
        规则：
          - 第 t 步使用 comm_buf.ht[t, nid]
        """
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        T_eff = min(T, comm_buf.steps)  # 通常 steps = episode_length + 1；我们只取 0..T-1
        ht_all = comm_buf.ht            # [steps, N, h_dim]
        if detach:
            ht_all = ht_all.detach()

        out: Dict[int, torch.Tensor] = {}
        if T_eff <= 0 or len(order) == 0:
            device = comm_buf.device
            out = {nid: torch.zeros(0, B, self.h_hidden, device=device, dtype=comm_buf.ht.dtype) for nid in order}
            return out

        # 仅取 0..T_eff-1 步
        for nid in order:
            h_seq = ht_all[:T_eff, nid, :]                      # [T, D]
            h_seq = h_seq.unsqueeze(1).expand(-1, B, -1)        # [T, B, D]
            out[nid] = h_seq
        return out

    def evaluate_actions(
        self,
        obs_i_t: torch.Tensor,                               # [B, state_size]
        rnn_states: Dict[numpy.ndarray, torch.Tensor],
        neigh_infor_prev_dict: Dict[int, torch.Tensor],      # {nid: [B, infor_dim]} 逐邻居的 t-1 infor
        neigh_h_t: Dict[int, torch.Tensor],                  # {nid: [B, h_hidden]}  邻居的 h_j^t（rollout 缓存）
        neigh_mask: Optional[torch.Tensor] = None,           # [B, M] 1/0，可选
        neigh_ids_in_order: Optional[List[int]] = None,      # 若不传则默认用 self.neighbours 的顺序
        stop_grad_neighbors: bool = True,
        action = None
    ):
        rnn_states = check(rnn_states).to(self.device)
        action = check(action).to(self.device)
        B = obs_i_t.size(0)
        order = self.neighbours if neigh_ids_in_order is None else [int(x) for x in neigh_ids_in_order]
        assert len(order) == self.M, "邻居数量需与初始化一致（每体固定一阶邻居数）。"

        # ---- (9) 逐邻居拼接 infor_{t-1} + 本地 obs → h_i^t ----
        # 若某邻居暂缺消息，自动用 0 向量占位（保证维度一致）
        prev_list = []
        for nid in order:
            x = neigh_infor_prev_dict[nid]
            if x is None:
                x = torch.zeros(B, self.infor_dim, device=obs_i_t.device, dtype=obs_i_t.dtype)
            prev_list.append(x)
        prev_stack = torch.stack(prev_list, dim=1)                 # [B, M, infor_dim]
        prev_flat  = prev_stack.reshape(B, self.M * self.infor_dim)# [B, M*infor_dim]
        enc_in = torch.cat([obs_i_t, prev_flat], dim=-1)           # ★ 修正点
        h_i_t = self.enc(enc_in)                                   # [B, h_hidden]

        # ---- (10)(11) 注意力：h_i^t 与 {h_j^t} → h'_i^t ----
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

        # ---- (13) infor_i^t ----
        infor_i_t = torch.sigmoid(self.infor_head(h_prime))         # [B, infor_dim]

        # ---- policy 输入：[obs, infor] ----
        fused = torch.cat([obs_i_t, infor_i_t], dim=-1)             # [B, state_size+infor_dim]
        actor_feat = self._trunk(fused)
        actor_features, rnn_states = self.rnn(actor_feat, rnn_states)

        # Evaluate action distribution
        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(actor_features, action)
        return action_log_probs, dist_entropy, action_distribution
