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

        # -------- 原有 on-policy 缓冲（按你现有写法） --------
        self.states = np.zeros((self.episode_length + 1, self.state_size), dtype=np.float32)
        self.rnn_states = np.zeros((self.episode_length + 1, self.rnn_hidden_size), dtype=np.float32, )
        self.actions = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.advantage        = np.zeros((self.episode_length, 1), dtype=np.float32)
        self.states_array_dict = {}
        self.states_array = np.empty(self.episode_length + 1, dtype=object)

        self.factor = None
        self.step = 0

        # -------- 通信快照（all_agents 存法） --------
        self.comm_all_ht: Optional[torch.Tensor]   = None  # [steps, N, h_dim]
        self.comm_all_info: Optional[torch.Tensor] = None  # [steps, N, infor_dim]
        self.h_dim = h_dim
        self.infor_dim = infor_dim
        self.num_agents: Optional[int] = None
        self.steps: Optional[int] = None

        # 本体 id（供便捷函数使用），由外部赋值或 attach 时传入
        self.agent_id: Optional[int] = None

    # ---------------- 原有接口 ----------------
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
        """清空actor的RNN隐藏状态，重置为全零初始状态（每个新episode调用）"""
        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.rnn_hidden_size),
            dtype=np.float32
        )
        # 可选：返回重置后的初始状态（用于验证）
        return self.rnn_states[0]

    # =============== 新增：拷贝 CommBuffers 到本类（all_agents） ===============

    @torch.no_grad()
    def attach_comm_snapshot(
        self,
        comm_buf,                  # CommBuffers 实例（含 ht: [steps, N, h_dim], infor: [steps, N, infor_dim]）
        agent_id: int,             # 当前 ActorBuffer 对应的 agent id（用于便捷读取）
        detach: bool = True,       # 训练常用 stop-grad 数据：detach=True
        move_to_device: Optional[torch.device] = None,  # 若传则移动到该 device（默认用本类 device）
    ):
        """
        将本轮 rollout 的 '全部智能体' 的通信数据拷贝到 ActorBuffer：
          - self.comm_all_ht   <- comm_buf.ht    # [steps, N, h_dim]
          - self.comm_all_info <- comm_buf.infor # [steps, N, infor_dim]
        说明：
          - steps = episode_length + 1（含最后一帧）
          - 若 detach=True，则从计算图分离；训练时推荐 True
          - 数据会 clone()，不依赖外部 comm_buf 生命周期
        """
        dev = move_to_device or self.device

        ht = comm_buf.ht    # [steps, N, h_dim]
        info = comm_buf.infor  # [steps, N, infor_dim]
        if detach:
            ht = ht.detach()
            info = info.detach()

        # 记录维度
        self.steps = ht.shape[0]
        self.num_agents = ht.shape[1]
        if self.h_dim is None:     self.h_dim = ht.shape[-1]
        if self.infor_dim is None: self.infor_dim = info.shape[-1]
        self.agent_id = int(agent_id)

        # 深拷贝到本类（放到目标设备）
        self.comm_all_ht   = ht.to(dev).clone()     # [steps, N, h_dim]
        self.comm_all_info = info.to(dev).clone()   # [steps, N, infor_dim]

    # ---------------- 便捷读取（all_agents） ----------------

    def get_step_ht(self, t: int, *, detach: bool = True) -> torch.Tensor:
        """取第 t 步所有智能体的 h^t: [N, h_dim]"""
        self._check_ready()
        self._check_t(t)
        out = self.comm_all_ht[t]
        return out.detach() if detach else out

    def get_step_infor(self, t: int, *, detach: bool = True) -> torch.Tensor:
        """取第 t 步所有智能体的 infor^t: [N, infor_dim]"""
        self._check_ready()
        self._check_t(t)
        out = self.comm_all_info[t]
        return out.detach() if detach else out

    def get_agent_ht(self, t: int, agent_id: int, *, detach: bool = True) -> torch.Tensor:
        """取指定 agent 在步 t 的 h^t: [h_dim]"""
        self._check_ready()
        self._check_t(t); self._check_i(agent_id)
        out = self.comm_all_ht[t, agent_id]
        return out.detach() if detach else out

    def get_agent_infor(self, t: int, agent_id: int, *, detach: bool = True) -> torch.Tensor:
        """取指定 agent 在步 t 的 infor^t: [infor_dim]"""
        self._check_ready()
        self._check_t(t); self._check_i(agent_id)
        out = self.comm_all_info[t, agent_id]
        return out.detach() if detach else out

    def get_self_ht(self, t: int, *, detach: bool = True) -> torch.Tensor:
        """本体在步 t 的 h^t: [h_dim]"""
        if self.agent_id is None:
            raise RuntimeError("actor_id/agent_id 未设置。调用 attach_comm_snapshot(...) 时请传入 agent_id。")
        return self.get_agent_ht(t, self.agent_id, detach=detach)

    def get_self_infor(self, t: int, *, detach: bool = True) -> torch.Tensor:
        """本体在步 t 的 infor^t: [infor_dim]"""
        if self.agent_id is None:
            raise RuntimeError("actor_id/agent_id 未设置。调用 attach_comm_snapshot(...) 时请传入 agent_id。")
        return self.get_agent_infor(t, self.agent_id, detach=detach)

    def get_neighbors_ht(self, t: int, neighbor_ids: List[int], *, detach: bool = True) -> torch.Tensor:
        """取邻居在步 t 的 h^t，形状 [len(neighbor_ids), h_dim]，便于注意力聚合。"""
        self._check_ready()
        self._check_t(t)
        idx = torch.as_tensor(neighbor_ids, device=self.comm_all_ht.device, dtype=torch.long)
        out = self.comm_all_ht[t].index_select(0, idx)
        return out.detach() if detach else out

    def get_neighbors_infor(self, t: int, neighbor_ids: List[int], *, detach: bool = True) -> torch.Tensor:
        """取邻居在步 t 的 infor^t，形状 [len(neighbor_ids), infor_dim]。"""
        self._check_ready()
        self._check_t(t)
        idx = torch.as_tensor(neighbor_ids, device=self.comm_all_info.device, dtype=torch.long)
        out = self.comm_all_info[t].index_select(0, idx)
        return out.detach() if detach else out

    # ---------------- 内部检查 ----------------

    def _check_ready(self):
        if self.comm_all_ht is None or self.comm_all_info is None:
            raise RuntimeError("通信快照未附加：请先调用 attach_comm_snapshot(..., mode='all_agents')")

    def _check_t(self, t: int):
        if self.steps is None:
            raise RuntimeError("steps 未初始化。")
        if not (0 <= t < self.steps):
            raise IndexError(f"t={t} out of range [0, {self.steps - 1}]")

    def _check_i(self, i: int):
        if self.num_agents is None:
            raise RuntimeError("num_agents 未初始化。")
        if not (0 <= i < self.num_agents):
            raise IndexError(f"agent_id={i} out of range [0, {self.num_agents - 1}]")


    def naive_recurrent_generator_actor(
            self, advantages, neigh_prev_info_ep, neigh_h_ep,
            actor_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for actor that uses RNN network.
        This generator does not split the trajectories into chunks.
        (n_rollout_threads=1, recurrent_n=1, no N dimension)
        """
        # 无需N维度（固定为1，直接删除）
        T = self.episode_length  # 轨迹长度（如360步）

        # 准备唯一的mini-batch数据（直接处理单环境数据，无需考虑N）
        # 原_flatten(T, 1, data)等价于data.reshape(T, ...)，直接简化为reshape
        obs_batch = self.states[:-1].reshape(T, -1)  # self.states形状: (T+1, obs_dim) → (T, obs_dim)
        actions_batch = self.actions.reshape(T, -1)  # self.actions形状: (T, action_dim) → 保持形状
        old_action_log_probs_batch = self.action_log_probs.reshape(T, -1)  # 形状: (T,) → (T, 1)
        adv_targ = advantages.reshape(T, 1)  # 形状: (T,) → (T, 1)


        factor_batch = None
        if self.factor is not None:
            factor_batch = self.factor.reshape(T, -1)  # (T, factor_dim)

        # RNN初始状态（无recurrent_n和N维度）
        # self.rnn_states形状: (T+1, outputs_dim)
        rnn_states_batch = self.rnn_states[0]

        # -------------------------- 核心修改：sampler仅包含一个批次，覆盖所有时间步 --------------------------
        # 生成索引：0 ~ episode_length-1（包含整个episode的所有时间步）
        full_indices = np.arange(T, dtype=np.int64)  # 如 [0,1,2,...,T-1]
        neigh_prev_info_batch_dict = {}
        for nid, info_seq in neigh_prev_info_ep.items():
            # info_seq: [T, B, neigh_info_dim]（完整episode的邻居时序信息）
            # 索引为完整时间步，确保包含所有邻居的时序数据
            idx_t = torch.as_tensor(full_indices, device=info_seq.device, dtype=torch.long)
            # 采样后形状：[T, B, neigh_info_dim]（保留完整时序）
            neigh_prev_info_batch_dict[nid] = info_seq.index_select(dim=0, index=idx_t)
        # -------------------------- neigh_h_ep：获取完整episode的邻居RNN状态 --------------------------
        neigh_h_batch_dict = {}
        for nid, h_seq in neigh_h_ep.items():
            # h_seq: [T, B, rnn_hidden_size]（完整episode的邻居RNN状态时序）
            idx_t = torch.as_tensor(full_indices, device=h_seq.device, dtype=torch.long)
            # 采样后形状：[T, B, rnn_hidden_size]（与邻居信息时序严格对齐）
            neigh_h_batch_dict[nid] = h_seq.index_select(dim=0, index=idx_t)

        # 返回唯一的mini-batch
        if self.factor is not None:
            yield obs_batch, rnn_states_batch,neigh_prev_info_batch_dict, neigh_h_batch_dict,actions_batch, old_action_log_probs_batch, adv_targ, factor_batch
        else:
            yield obs_batch, rnn_states_batch, neigh_prev_info_batch_dict, neigh_h_batch_dict,actions_batch, old_action_log_probs_batch, adv_targ
