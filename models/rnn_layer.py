import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, episode_length=360):
        super(RNNLayer, self).__init__()
        self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=1)
        self.outputs_dim = outputs_dim
        self.episode_length = episode_length

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks=None):
        # === hxs reshape ===
        if hxs.dim() == 1:
            hxs = hxs.unsqueeze(0)
        N = hxs.size(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_rows = x.size(0)
        T = x_rows // N
        assert T >= 1, f"T must be ≥1, got {T}"

        if masks is None:
            if T == 1:
                masks = torch.ones(N, dtype=torch.float32, device=hxs.device)
            else:
                masks = torch.ones(T, N, dtype=torch.float32, device=hxs.device)

                if self.episode_length is not None and T == self.episode_length:
                    masks[-1] = 0.0  # 自动截断

        if T == 1:
            masked_hxs = hxs * masks.unsqueeze(-1)
            masked_hxs_3d = masked_hxs.unsqueeze(0)
            x_3d = x.unsqueeze(0)
            x_rnn_3d, hxs_rnn_3d = self.rnn(x_3d, masked_hxs_3d)
            x = x_rnn_3d.squeeze(0)
            hxs = hxs_rnn_3d.squeeze(0)
        else:
            x_rnn = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = [0, T]
            hxs_rnn_3d = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                temp = hxs_rnn_3d * masks[start_idx].view(1, -1, 1)
                rnn_scores, hxs_rnn_3d = self.rnn(x_rnn[start_idx:end_idx], temp)
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0).reshape(T * N, -1)
            hxs = hxs_rnn_3d.squeeze(0)

        x = self.norm(x)
        return x, hxs
