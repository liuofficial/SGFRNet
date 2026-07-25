import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """简单 1x1 Conv + BN + ReLU，用来对齐通道."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class HaarWaveletTransform2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            channels,
            4 * channels,
            kernel_size=2,
            stride=2,
            groups=channels,
            bias=False,
        )
        self._init_weights()
        for p in self.conv.parameters():
            p.requires_grad = False

    def _init_weights(self):
        C = self.channels
        LL = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        LH = torch.tensor([[0.5, 0.5],
                           [-0.5, -0.5]], dtype=torch.float32)
        HL = torch.tensor([[0.5, -0.5],
                           [0.5, -0.5]], dtype=torch.float32)
        HH = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        k = torch.stack([LL, LH, HL, HH], dim=0)  # (4, 2, 2)

        weight = torch.zeros(4 * C, 1, 2, 2, dtype=torch.float32)
        for c in range(C):
            weight[4 * c:4 * c + 4, 0, :, :] = k

        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HaarInverseWaveletTransform2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H2, W2 = x.shape
        C = self.channels
        assert C4 == 4 * C, f"IDWT expect 4*C channels, got {C4}"

        LL = x[:, 0:C, :, :]
        LH = x[:, C:2*C, :, :]
        HL = x[:, 2*C:3*C, :, :]
        HH = x[:, 3*C:4*C, :, :]

        # 重建公式 (2x2 block):
        # x00 = LL + LH + HL + HH
        # x01 = LL + LH - HL - HH
        # x10 = LL - LH + HL - HH
        # x11 = LL - LH - HL + HH
        H = H2 * 2
        W = W2 * 2
        out = x.new_zeros(B, C, H, W)

        out[:, :, 0::2, 0::2] = LL + LH + HL + HH
        out[:, :, 0::2, 1::2] = LL + LH - HL - HH
        out[:, :, 1::2, 0::2] = LL - LH + HL - HH
        out[:, :, 1::2, 1::2] = LL - LH - HL + HH

        # 与正变换的 0.5 系数匹配，这里总体再缩放 0.5
        out = 0.5 * out
        return out


class SobelConv(nn.Module):
    """
    每个通道做 Sobel 梯度 (depthwise).
    输出: 梯度幅值 (B, C, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        )
        kernel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        )
        weight = torch.stack([kernel_x, kernel_y], dim=0)  # (2, 3, 3)
        weight = weight.unsqueeze(1)                       # (2, 1, 3, 3)
        weight = weight.repeat(channels, 1, 1, 1)          # (2C, 1, 3, 3)

        self.conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=channels,
        )
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        for p in self.conv.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx_gy = self.conv(x)
        gx, gy = torch.chunk(gx_gy, 2, dim=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return grad


class EGFIR(nn.Module):

    def __init__(
        self,
        in_channels: List[int],
        mid_channels: int = 64,
        attn_channels: int = None,
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.attn_channels = attn_channels or (3 * mid_channels)

        # 1) 通道对齐
        self.enc_projs = nn.ModuleList(
            [ConvBNReLU(c, mid_channels) for c in in_channels]
        )

        # 2) DWT / IDWT
        self.dwt_blocks = nn.ModuleList(
            [HaarWaveletTransform2D(mid_channels) for _ in range(self.num_scales)]
        )
        self.idwt_blocks = nn.ModuleList(
            [HaarInverseWaveletTransform2D(mid_channels) for _ in range(self.num_scales)]
        )

        # 3) Sobel 在 HF 分支上做：先 1x1 降到 Cmid，再 Sobel
        self.hf_edge_proj = nn.Conv2d(
            3 * mid_channels, mid_channels, kernel_size=1, bias=False
        )
        self.sobel = SobelConv(mid_channels)

        self.edge_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.edge_to_mask = nn.ModuleList(
            [
                nn.Conv2d(
                    mid_channels,
                    3 * mid_channels,  # 对 HF 三个方向分别学通道权重
                    kernel_size=1,
                    bias=True,
                )
                for _ in range(self.num_scales)
            ]
        )

        # 4) 全局高频池 (K/V 共同使用)
        self.global_hf_conv = nn.Conv2d(
            3 * mid_channels * self.num_scales,
            3 * mid_channels,
            kernel_size=1,
            bias=True,
        )

        # 5) Q/K/V 投影
        self.q_projs = nn.ModuleList(
            [
                nn.Conv2d(3 * mid_channels, self.attn_channels, kernel_size=1, bias=True)
                for _ in range(self.num_scales)
            ]
        )
        self.k_proj = nn.Conv2d(
            3 * mid_channels, self.attn_channels, kernel_size=1, bias=True
        )
        self.v_proj = nn.Conv2d(
            3 * mid_channels, self.attn_channels, kernel_size=1, bias=True
        )
        self.out_projs = nn.ModuleList(
            [
                nn.Conv2d(self.attn_channels, 3 * mid_channels, kernel_size=1, bias=True)
                for _ in range(self.num_scales)
            ]
        )

        # 6) IDWT 后的空间残差 -> 原通道
        self.idwt_to_spatial = nn.ModuleList(
            [
                nn.Conv2d(mid_channels, c, kernel_size=1, bias=True)
                for c in in_channels
            ]
        )
        self.conv1 =  nn.ModuleList(
            [
                nn.Conv2d(c,c,kernel_size=1)
                for c in in_channels
            ])
        # # 残差缩放参数, 初始为 0, 方便稳定训练
        # self.gamma = nn.Parameter(torch.zeros(self.num_scales))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats) == self.num_scales, \
            f"expect {self.num_scales} scales, got {len(feats)}"

        B = feats[0].shape[0]

        aligned_feats = []
        hf_list = []
        ll_list = []
        masks = []
        hf_sizes = []

        # ------ 逐尺度: 通道对齐 + Haar 高频 + Sobel 边缘 mask ------
        for i, x in enumerate(feats):
            # 1) 通道对齐
            x_align = self.enc_projs[i](x)  # (B, Cmid, H, W)
            aligned_feats.append(x_align)

            # 2) DWT -> LL + HF(3*Cmid)
            dwt_out = self.dwt_blocks[i](x_align)  # (B, 4*Cmid, H/2, W/2)
            C = self.mid_channels
            LL, HF = torch.split(dwt_out, [C, 3 * C], dim=1)

            # 3) 在 HF 分支上算 Sobel 边缘
            HF_for_edge = self.hf_edge_proj(HF)        # (B, Cmid, H/2, W/2)
            edge = self.sobel(HF_for_edge)             # (B, Cmid, H/2, W/2)

            edge_max = self.edge_maxpool(edge)
            edge_avg = self.edge_avgpool(edge)
            edge_delta = edge_max - edge_avg

            # 生成 高频 mask: (B, 3*Cmid, H/2, W/2)
            mask = torch.sigmoid(self.edge_to_mask[i](edge_delta))

            ll_list.append(LL)
            hf_list.append(HF)
            masks.append(mask)
            hf_sizes.append(HF.shape[-2:])  # (h_i, w_i)

        # ------ 构建全局高频池 (不带 Sobel 处理) ------
        # 选择一个参考尺度做对齐 (这里用中间层，你也可以换成最后一层)
        ref_h, ref_w = hf_sizes[2]
        hf_up_list = []
        for HF in hf_list:
            if HF.shape[-2:] != (ref_h, ref_w):
                HF_up = F.interpolate(
                    HF, size=(ref_h, ref_w), mode="bilinear", align_corners=False
                )
            else:
                HF_up = HF
            hf_up_list.append(HF_up)

        # 全局高频池
        H_cat = torch.cat(hf_up_list, dim=1)  # (B, 3*Cmid*num_scales, ref_h, ref_w)
        H_glob = self.global_hf_conv(H_cat)   # (B, 3*Cmid, ref_h, ref_w)

        # K / V
        K = self.k_proj(H_glob)  # (B, Cattn, ref_h, ref_w)
        V = self.v_proj(H_glob)  # (B, Cattn, ref_h, ref_w)
        Bk, Cattn, Hk, Wk = K.shape
        K_flat = K.view(Bk, Cattn, -1)        # (B, Cattn, Nk)
        V_flat = V.view(Bk, Cattn, -1)        # (B, Cattn, Nk)

        # ------ 对每个尺度: 边缘引导 cross-attention + 高频去噪 + IDWT 回空间 ------
        outputs = []
        for i, (x, x_align, LL, HF, mask) in enumerate(
            zip(feats, aligned_feats, ll_list, hf_list, masks)
        ):
            hi, wi = HF.shape[-2:]

            # 1) 边缘高频作为 Q
            HF_edge = HF * mask
            Q = self.q_projs[i](HF_edge)
            Bq, Cq, Hq, Wq = Q.shape
            Q_flat = Q.view(Bq, Cq, -1).transpose(1, 2)

            # 2) 注意力: Q(边缘) x K(全局高频池)
            scores = torch.bmm(Q_flat, K_flat) / math.sqrt(Cq)
            attn = torch.softmax(scores, dim=-1)

            # 3) 用 V 重写边缘高频
            V_flat_t = V_flat.transpose(1, 2)
            out_flat = torch.bmm(attn, V_flat_t)
            out = out_flat.transpose(1, 2).view(Bq, Cq, Hq, Wq)


            out = self.out_projs[i](out)

            dwt_den = torch.cat([LL, out], dim=1)
            x_refine = self.idwt_blocks[i](dwt_den)
            x_refine = self.idwt_to_spatial[i](x_refine)
            x_out = self.conv1[i](x) + x_refine
            outputs.append(x_out)

        return outputs


class ChannelAttention_CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        w = self.mlp(avg) + self.mlp(mx)
        return self.sigmoid(w)

class SpatialAttention_CBAM(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 5, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.sigmoid(self.conv(s))
        return x * s
class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels, ca_reduction=8, sa_ks=3, bottleneck_ratio=2):
        super().__init__()

        # ---------- bottleneck 降维 ----------
        mid_channels = channels // bottleneck_ratio # 防止太小或为 0

        # 注意：CBAM 的通道数也改成 mid_channels
        self.ca1 = ChannelAttention_CBAM(mid_channels, reduction=ca_reduction)
        self.sa1 = SpatialAttention_CBAM(kernel_size=sa_ks)
        self.ca2 = ChannelAttention_CBAM(mid_channels, reduction=ca_reduction)
        self.sa2 = SpatialAttention_CBAM(kernel_size=sa_ks)

        # 1x1 conv 降维：C -> C_mid
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 原来的 conv_1：在 C_mid 上做 DW + PW
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=1, padding=1,
                groups=mid_channels, bias=False  # Depthwise
            ),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 原来的 conv_2：膨胀卷积，同样在 C_mid 上
        self.conv_2 = nn.Conv2d(
            mid_channels, mid_channels,
            kernel_size=3, stride=1, padding=2,
            dilation=2, bias=False
        )
        self.bn_2_1 = nn.BatchNorm2d(mid_channels)

        # 融合两个分支后的 1x1 conv，同步改成 C_mid 维度
        self.conv_ca1 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_ca2 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_ca3 = nn.Conv2d(2 * mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)

        # ---------- bottleneck 升维 ----------
        self.conv_expand = nn.Sequential(
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x                      # 残差从原始输入走，更标准的 ResNet 写法

        # C -> C_mid
        out = self.conv_reduce(x)

        # 两个感受野分支
        out_1 = self.conv_1(out)         # 普通 3x3 (DW+PW)
        out_2 = self.conv_2(out)
        out_2 = self.bn_2_1(out_2)

        # 空间注意力
        sa1 = self.sa1(out_1)
        sa2 = self.sa2(out_2)

        # 拼接两个分支的空间信息
        cat_out = torch.cat([sa1, sa2], dim=1)  # [B, 2*C_mid, H, W]

        # 通道注意力 + 特征重标定
        ca1 = self.ca1(self.conv_ca1(cat_out)) * out_1
        ca2 = self.ca2(self.conv_ca2(cat_out)) * out_2

        # 再拼接，然后 1x1 conv 融合
        ca = torch.cat([ca1, ca2], dim=1)       # [B, 2*C_mid, H, W]
        ca_out = self.conv_ca3(ca)              # [B, C_mid, H, W]

        # C_mid -> C，并做残差
        out = self.conv_expand(ca_out)
        out = self.relu(out + identity)

        return out


class CFIF(nn.Module):
    def __init__(self, in_channels_list, out_ch, branch_ch=32, sa_ks=3):
        super().__init__()
        self.branch_maps = nn.ModuleList([Conv1x1BN(c, branch_ch) for c in in_channels_list])
        self._ups = UpsampleTo()
        self._downs = DownTo()

        fused_ch = branch_ch * len(in_channels_list)
        self.ra = ResidualAttentionBlock(fused_ch, sa_ks=sa_ks)
        self.fuse = Conv3x3BN(fused_ch, out_ch)


    def _resize_like(self, x, ref):
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return self._ups(x, ref) if x.shape[-2] < ref.shape[-2] else self._downs(x, ref)

    def forward(self, feats, target_feat):
        mapped = []
        for i, f in enumerate(feats):
            m = self.branch_maps[i](f)
            m = self._resize_like(m, target_feat)
            mapped.append(m)
        x = torch.cat(mapped, dim=1)  # (B, branch_ch*5, Ht, Wt)
        x = self.ra(x)
        x = self.fuse(x)              # (B, out_ch, Ht, Wt)
        return x


def eca_kernel(channels, gamma=2, b=1):
    k = int(abs((math.log2(channels) / gamma) + b))
    return k if k % 2 == 1 else k + 1

class ECA(nn.Module):
    def __init__(self, channels, k_size=None):
        super().__init__()
        if k_size is None: k_size = max(3, eca_kernel(channels))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = self.conv(y.squeeze(-1).transpose(1, 2))
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)
        return x * y

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2):
        super().__init__()
        mid_channels = int(in_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)

class CDC_conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        kernel_size=3,
        padding=1,
        dilation=1,
        theta=0.7,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        c_out, c_in, kernel_size, _ = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(
            input=x,
            weight=kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
        )
        out = norm_out - self.theta * diff_out
        return out

class ProgressiveFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 3, 3], act=nn.ReLU(inplace=True)):
        super().__init__()
        C = out_channels
        self.C = C
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, 3 * C, 1, bias=False),
            nn.BatchNorm2d(3 * C),
            act
        )
        self.conv_A = self._make_branch(C, C, kernel_size=scales[0])
        self.conv_B = self._make_branch(C, C, kernel_size=scales[1])
        self.conv_C = self._make_branch(C, C, kernel_size=scales[2])

    def _make_branch(self, in_ch, out_ch, kernel_size):
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.expand(x)  # (B, 3C, H, W)
        a, b, c = torch.split(x, self.C, dim=1)
        a_out = self.conv_A(a)
        b_out = self.conv_B(b + a_out)
        c_out = self.conv_C(c + b_out)
        fused = torch.cat([a_out, b_out, c_out], dim=1)  # (B, 3C, H, W)
        return fused

class PCDE(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[3, 3, 3]):
        super().__init__()
        self.out_channels = out_channels
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.enhance_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.progressive_block = ProgressiveFusionBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            scales=scales
        )
        self.concat_proj = nn.Conv2d(4 * out_channels, out_channels, 1, bias=False)
        self.attn = ECA(out_channels, k_size=None)
        self.final_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        self.cdc_layer = nn.Sequential(
            CDC_conv(out_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        skip = self.skip_conv(x)
        enhanced = self.progressive_block(self.enhance_conv(x))
        cat = torch.cat([skip, enhanced], dim=1)
        cat = self.concat_proj(cat)
        cdc = self.cdc_layer(cat)
        cat = cdc + cat
        cat = self.attn(cat)
        out = self.final_conv(cat)
        identity = self.shortcut(x)
        return F.relu(out + identity)



class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.conv = nn.Conv2d(2*in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.act(out)
        return out

class UpsampleTo(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
    def forward(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode=self.mode, align_corners=False)

class DownTo(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, ref):
        H, W = ref.shape[-2:]
        return F.adaptive_avg_pool2d(x, (H, W))

class Conv1x1BN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class Conv3x3BN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class DSHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pred = nn.Conv2d(in_ch, 1, 1)
    def forward(self, x, size_hw):
        x = self.pred(x)
        return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)


class SGFRNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=16, branch_ch=32, use_se=True, deep_supervision=True):
        super().__init__()
        C = base_ch
        self.e1 = PCDE(in_ch, C)
        self.down1 = DownSample(C, C)
        self.e2 =  PCDE(C, 2 * C)
        self.down2 = DownSample(2 * C, 2 * C)
        self.e3 =  PCDE(2 * C, 4 * C)
        self.down3 = DownSample(4 * C, 4 * C)
        self.e4 = PCDE(4 * C, 8 * C)
        self.down4 = DownSample(8 * C, 8 * C)
        self.e5 = Residual(8 * C, 8 * C, expansion=2)

        self.d4_refine  = Residual(4*C, 4*C, expansion=1)

        self.d3_refine = Residual(2*C, 2*C, expansion=1)

        self.d2_refine = Residual(C, C, expansion=1)

        self.d1_refine = Residual(C, C, expansion=1)

        self.d4_fuse = CFIF([C, 2 * C, 4 * C, 8 * C, 8 * C], out_ch=4 * C, branch_ch=16)
        self.d3_fuse = CFIF([C, 2 * C, 4 * C, 4 * C, 8 * C], out_ch=2 * C, branch_ch=16)
        self.d2_fuse = CFIF([C, 2 * C, 2 * C, 4 * C, 8 * C], out_ch=C, branch_ch=16)
        self.d1_fuse = CFIF([C, C, 2 * C, 4 * C, 8 * C], out_ch=C, branch_ch=16)
        self.head = nn.Conv2d(C, 1, 1)

        self.block = EGFIR(
            in_channels=[C, 2 * C, 4 * C, 8 * C, 8 * C],  # 五个编码器输出通道
            mid_channels=32,  # 统一到的中间通道，可自己改
            attn_channels=None,  # 默认=3*mid_channels
        )


        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.ds5 = DSHead(8 * C)
            self.ds4 = DSHead(4 * C)
            self.ds3 = DSHead(2 * C)
            self.ds2 = DSHead(C)
            self.ds1 = DSHead(C)

    def forward(self, x):
        H, W = x.shape[-2:]

        # 编码
        e1 = self.e1(x)                  # 1/1,   C
        e2 = self.e2(self.down1(e1))     # 1/2,  2C
        e3 = self.e3(self.down2(e2))     # 1/4,  4C
        e4 = self.e4(self.down3(e3))     # 1/8,  8C
        e5 = self.e5(self.down4(e4))     # 1/16, 8C

        outs = self.block([e1, e2, e3, e4, e5])

        e1, e2, e3, e4, e5 = outs

        d4 = self.d4_fuse([e1, e2, e3, e4, e5], target_feat=e4)
        d4 = self.d4_refine(d4)          # 1/8, 4C

        d3 = self.d3_fuse([e1, e2, e3, d4, e5], target_feat=e3)
        d3 = self.d3_refine(d3)          # 1/4, 2C

        d2 = self.d2_fuse([e1, e2, d3, d4, e5], target_feat=e2)


        d2 = self.d2_refine(d2)          # 1/2, C

        d1 = self.d1_fuse([e1, d2, d3, d4, e5], target_feat=e1)



        d1 = self.d1_refine(d1)

        logit = self.head(d1)
        seg = torch.sigmoid(logit)

        if not self.deep_supervision:
            return seg

        ds5 = torch.sigmoid(self.ds5(e5, (H, W)))
        ds4 = torch.sigmoid(self.ds4(d4, (H, W)))
        ds3 = torch.sigmoid(self.ds3(d3, (H, W)))
        ds2 = torch.sigmoid(self.ds2(d2, (H, W)))

        return seg,ds2, ds3, ds4, ds5

if __name__ == '__main__':
    from torchinfo import summary
    x = torch.randn(1, 1, 256, 256)
    model = SGFRNet(in_ch=1, base_ch=16, branch_ch=32, use_se=True, deep_supervision=True)
    summary_result = summary(
        model,
        input_size=(1,) +  (1, 256, 256),
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        verbose=0
    )

    print(summary_result)
