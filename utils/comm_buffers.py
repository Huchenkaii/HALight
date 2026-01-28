from typing import List, Optional
import torch

class CommBuffers:
    def __init__(self, episode_length: int, N: int, h_dim: int, infor_dim: int,
                 device=None, dtype=torch.float32):
        self.steps, self.N = episode_length + 1, N
        self.h_dim, self.infor_dim = h_dim, infor_dim
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.ht = torch.zeros(self.steps, N, h_dim, device=self.device, dtype=self.dtype)
        self.infor = torch.zeros(self.steps, N, infor_dim, device=self.device, dtype=self.dtype)

        self._w_ht = torch.zeros(self.steps, N, dtype=torch.bool, device=self.device)
        self._w_infor = torch.zeros(self.steps, N, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def set_ht(self, t: int, agent_id: int, h_t: torch.Tensor):
        self._check_t_i(t, agent_id)
        self._check_shape(h_t, (self.h_dim,))
        self.ht[t, agent_id].copy_(h_t)
        self._w_ht[t, agent_id] = True

    @torch.no_grad()
    def set_step_ht(self, t: int, h_t_all: torch.Tensor):
        self._check_t(t)
        self._check_shape(h_t_all, (self.N, self.h_dim))
        self.ht[t].copy_(h_t_all)
        self._w_ht[t].fill_(True)

    @torch.no_grad()
    def set_infor(self, t: int, agent_id: int, infor_t: torch.Tensor):
        self._check_t_i(t, agent_id)
        self._check_shape(infor_t, (self.infor_dim,))
        self.infor[t, agent_id].copy_(infor_t)
        self._w_infor[t, agent_id] = True

    @torch.no_grad()
    def set_step_infor(self, t: int, infor_t_all: torch.Tensor):
        self._check_t(t)
        self._check_shape(infor_t_all, (self.N, self.infor_dim))
        self.infor[t].copy_(infor_t_all)
        self._w_infor[t].fill_(True)

    def get_ht(self, t: int, agent_id: int, *, detach: bool = False, check_written: bool = False) -> torch.Tensor:
        self._check_t_i(t, agent_id)
        if check_written and not self._w_ht[t, agent_id]:
            raise RuntimeError(f"h_t not written yet at (t={t}, agent={agent_id}).")
        out = self.ht[t, agent_id]
        return out.detach() if detach else out

    def get_infor(self, t: int, agent_id: int, *, detach: bool = False, check_written: bool = False) -> torch.Tensor:
        self._check_t_i(t, agent_id)
        if check_written and not self._w_infor[t, agent_id]:
            raise RuntimeError(f"infor_t not written yet at (t={t}, agent={agent_id}).")
        out = self.infor[t, agent_id]
        return out.detach() if detach else out

    def get_step_ht(self, t: int, *, detach: bool = False, check_written: bool = False) -> torch.Tensor:
        self._check_t(t)
        if check_written and not bool(self._w_ht[t].all()):
            raise RuntimeError(f"h_t for step {t} not fully written.")
        out = self.ht[t]
        return out.detach() if detach else out

    def get_step_infor(self, t: int, *, detach: bool = False, check_written: bool = False) -> torch.Tensor:
        self._check_t(t)
        if check_written and not bool(self._w_infor[t].all()):
            raise RuntimeError(f"infor_t for step {t} not fully written.")
        out = self.infor[t]
        return out.detach() if detach else out

    def gather_neighbors_ht(self, t: int, neighbor_ids: List[int], *, detach: bool = True) -> torch.Tensor:
        """
        Return a tensor of shape [len(neighbor_ids), h_dim] for attention aggregation.
        """
        self._check_t(t)
        idx = torch.as_tensor(neighbor_ids, device=self.device, dtype=torch.long)
        out = self.ht[t].index_select(dim=0, index=idx)
        return out.detach() if detach else out

    def to(self, device):
        self.device = device
        self.ht = self.ht.to(device); self.infor = self.infor.to(device)
        self._w_ht = self._w_ht.to(device); self._w_infor = self._w_infor.to(device)
        return self

    def clear_(self):
        self.ht.zero_(); self.infor.zero_()
        self._w_ht.zero_(); self._w_infor.zero_()

    def _check_t(self, t: int):
        if not (0 <= t < self.steps):
            raise IndexError(f"t={t} out of range [0, {self.steps-1}]")

    def _check_t_i(self, t: int, i: int):
        self._check_t(t)
        if not (0 <= i < self.N):
            raise IndexError(f"agent_id={i} out of range [0, {self.N-1}]")

    def _check_shape(self, x: torch.Tensor, expect_tail: tuple):
        if tuple(x.shape[-len(expect_tail):]) != expect_tail:
            raise ValueError(f"shape {tuple(x.shape)} does not end with {expect_tail}")

    def to_cpu_and_numpy(self):
        """
        After the entire episode ends, transfer `infor` and `ht` from GPU to CPU
        and convert them to NumPy arrays, ensuring that data from all steps
        has been processed.
        """
        if not bool(self._w_ht.all()):
            raise RuntimeError(f"h_t not fully written for some steps.")
        if not bool(self._w_infor.all()):
            raise RuntimeError(f"infor_t not fully written for some steps.")

        ht_cpu = self.ht.cpu().numpy()  # [steps, N, h_dim]
        infor_cpu = self.infor.cpu().numpy()  # [steps, N, infor_dim]

        return ht_cpu, infor_cpu