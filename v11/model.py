
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class UNetRegConfig:
    in_ch: int = 56
    mid_ch: int = 16
    out_ch: int = 64
    sam_kernel: int = 7
    cam_reduction: int = 16

class REBNCONV(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=False)  # Set to False to protect skip connection autograd graphs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

def _upsample_like(x: torch.Tensor, target_len: int) -> torch.Tensor:
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        assert height >= 2
        self.height = height
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.enc = nn.ModuleList()
        self.enc.append(REBNCONV(out_ch, mid_ch))
        for _ in range(1, height - 1):
            self.enc.append(REBNCONV(mid_ch, mid_ch))

        self.bottleneck = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.dec = nn.ModuleList()
        for _ in range(height - 2):
            self.dec.append(REBNCONV(mid_ch * 2, mid_ch))
        self.dec_last = REBNCONV(mid_ch * 2, out_ch)
        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        enc_feats = [self.enc[0](hxin)]
        for i in range(1, self.height - 1):
            enc_feats.append(self.enc[i](self.pool(enc_feats[-1])))

        hx = self.bottleneck(enc_feats[-1])
        hx = self.dec[0](torch.cat([hx, enc_feats[-1]], dim=1))
        for i in range(1, self.height - 1):
            skip = enc_feats[self.height - 2 - i]
            hx = _upsample_like(hx, skip.shape[-1])
            hx = self.dec[i](torch.cat([hx, skip], dim=1)) if i < self.height - 2 else self.dec_last(torch.cat([hx, skip], dim=1))

        return hx + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilation=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat([hx4, hx3], dim=1))
        hx2d = self.rebnconv2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.rebnconv1d(torch.cat([hx2d, hx1], dim=1))
        return hx1d + hxin

class SAM(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y1: torch.Tensor) -> torch.Tensor:
        max_c = torch.max(y1, dim=1, keepdim=True)[0]
        avg_c = torch.mean(y1, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_c, avg_c], dim=1))) * y1

class CAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=False),
            nn.Linear(hidden, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, y1: torch.Tensor) -> torch.Tensor:
        max_s = torch.max(y1, dim=2)[0]
        avg_s = torch.mean(y1, dim=2)
        attn = self.mlp(max_s) + self.mlp(avg_s)
        return self.sigmoid(attn).unsqueeze(-1) * y1

class USAE(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int, sam_kernel: int = 7):
        super().__init__()
        self.rsu = RSU(height, in_ch, mid_ch, out_ch)
        self.sam = SAM(out_ch, kernel_size=sam_kernel)

    def forward(self, x: torch.Tensor):
        y1 = self.rsu(x)
        return y1, self.sam(y1)

class UCAE(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, reduction: int = 16, bottom: bool = False):
        super().__init__()
        self.rsu = RSU4F(in_ch, mid_ch, out_ch)
        self.cam = CAM(out_ch, reduction=reduction)
        self.bottom = bottom

    def forward(self, x: torch.Tensor):
        y1 = self.rsu(x)
        y2 = self.cam(y1)
        return y2 if self.bottom else (y1, y2)

class SCA_U2Net_Reg(nn.Module):
    def __init__(self, cfg: UNetRegConfig = None):
        super().__init__()
        cfg = cfg or UNetRegConfig()
        c, m, o = cfg.in_ch, cfg.mid_ch, cfg.out_ch
        sk, r = cfg.sam_kernel, cfg.cam_reduction

        self.e1 = USAE(7, c, m, o, sam_kernel=sk)
        self.e2 = USAE(6, o, m, o, sam_kernel=sk)
        self.e3 = USAE(5, o, m, o, sam_kernel=sk)
        self.e4 = USAE(4, o, m, o, sam_kernel=sk)
        self.e5 = UCAE(o, m, o, reduction=r, bottom=False)
        self.e6 = UCAE(o, m, o, reduction=r, bottom=True)

        self.d5 = RSU4F(o * 2, m, o)
        self.d4 = RSU(4, o * 2, m, o)
        self.d3 = RSU(5, o * 2, m, o)
        self.d2 = RSU(6, o * 2, m, o)
        self.d1 = RSU(7, o * 2, m, o)

        self.pool = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.side1 = nn.Conv1d(o, 1, kernel_size=3, padding=1)
        self.side2 = nn.Conv1d(o, 1, kernel_size=3, padding=1)
        self.side3 = nn.Conv1d(o, 1, kernel_size=3, padding=1)
        self.side4 = nn.Conv1d(o, 1, kernel_size=3, padding=1)
        self.side5 = nn.Conv1d(o, 1, kernel_size=3, padding=1)
        self.side6 = nn.Conv1d(o, 1, kernel_size=3, padding=1)

        self.outconv = nn.Conv1d(6, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        e1c1, e1c2 = self.e1(x)
        e2c1, e2c2 = self.e2(self.pool(e1c1))
        e3c1, e3c2 = self.e3(self.pool(e2c1))
        e4c1, e4c2 = self.e4(self.pool(e3c1))
        e5c1, e5c2 = self.e5(self.pool(e4c1))
        e6c = self.e6(self.pool(e5c1))

        hx6up = _upsample_like(e6c, e5c2.shape[-1])
        hx5d = self.d5(torch.cat([hx6up, e5c2], dim=1))
        hx5dup = _upsample_like(hx5d, e4c2.shape[-1])
        hx4d = self.d4(torch.cat([hx5dup, e4c2], dim=1))
        hx4dup = _upsample_like(hx4d, e3c2.shape[-1])
        hx3d = self.d3(torch.cat([hx4dup, e3c2], dim=1))
        hx3dup = _upsample_like(hx3d, e2c2.shape[-1])
        hx2d = self.d2(torch.cat([hx3dup, e2c2], dim=1))
        hx2dup = _upsample_like(hx2d, e1c2.shape[-1])
        hx1d = self.d1(torch.cat([hx2dup, e1c2], dim=1))

        L = x.shape[-1]
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), L)
        d3 = _upsample_like(self.side3(hx3d), L)
        d4 = _upsample_like(self.side4(hx4d), L)
        d5 = _upsample_like(self.side5(hx5d), L)
        d6 = _upsample_like(self.side6(e6c), L)

        fuse_out = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], dim=1)).squeeze(1)
        sides = [d1.squeeze(1), d2.squeeze(1), d3.squeeze(1), d4.squeeze(1), d5.squeeze(1), d6.squeeze(1)]

        return fuse_out, sides

def deep_supervision_rmse(fuse: torch.Tensor, sides: list[torch.Tensor], target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Computes masked multi-output deep supervision loss across all branches (No in-place addition)."""
    def _masked_rmse(pred, tgt, m):
        valid = m.float(); n_valid = valid.sum()
        if n_valid < 1: return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.sqrt((((pred - tgt) ** 2) * valid).sum() / n_valid + eps)

    total = _masked_rmse(fuse, target, mask)
    for s in sides: 
        total = total + _masked_rmse(s, target, mask)  # Changed from += to protect the backprop graph
    return total
