import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class SelfAttention_w_attnscore(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)


    # —— 区域引导（质心版，极省显存；direction 控制增强方向）
    def region_guidance_centroid(self, q, k, q_idx, left_k_idx, right_k_idx, tau=1):
        """
        q, k: (B, S, H, D)
        q_idx: 需要施加强引导的 queries(1D 索引)
        """
        if q_idx.numel() == 0:
            return q.new_zeros(())
        B, S, H, D = q.shape

        # 采样 head/query 进一步控成本
        kL = k[:, left_k_idx].mean(1)         # (B, Hs, D)
        kR = k[:, right_k_idx].mean(1)        # (B, Hs, D)

        qh = q.permute(0,2,1,3)                        # # (B, Qs, Hs, D) --> (B,Hs,Qs,D)
        logitL = (qh * kL[:, :, None, :]).sum(-1) / (math.sqrt(D) * tau)  # (B,Hs,Qs)
        logitR = (qh * kR[:, :, None, :]).sum(-1) / (math.sqrt(D) * tau)

        # 两类 softmax 概率
        logits = torch.stack([logitL, logitR], dim=-1)     # (...,2)
        probs = logits.softmax(dim=-1)
        attn_L, attn_R = probs[..., 0], probs[..., 1]

        # 你的公式：mean(A[~M_fg]) - mean(A[M_fg])
        # - 若 enhance_right：希望 A_R 均值大、A_L 小 -> loss = mean(A_L) - mean(A_R)
        # - 若 enhance_right=False：相反
        # if enhance_right:
        attn_score = (attn_L - attn_R).mean()
        # else:
        #     loss = A_R.mean() - A_L.mean()
        # return loss
        # attn_score = {'attn_L': attn_L, 'attn_R': attn_R}

        return attn_score


    def split_lr_indices_with_mask_5d(self, F:int, H:int, W:int, L:int = 1,
                                    boost_mask_5d: torch.Tensor | None = None,
                                    device: str | torch.device = "cuda"):
        """
        展平顺序固定为 (F, H, W, L) -> S   ←←← 与你的一致
        生成：
        - left_k_idx, right_k_idx ：按 W 维左右半区的所有 key 的线性索引
        - right_q_idx_masked      ：右半 ∩ boost_mask 的 query 索引
        - right_q_idx_other       ：右半 \ boost_mask 的 query 索引
        参数：
        boost_mask_5d: (F, H, W, 1) 的 0/1 或 bool mask（单个样本）
        """
        # 构造线性索引栅格（F, H, W, L）
        grid = torch.arange(F * H * W*2 * L, device=device).reshape(F, H, W*2, L)

        # 左右半边 keys（整半边）
        left_k_idx  = grid[:, :, :W, :].reshape(-1).contiguous()
        right_k_idx = grid[:, :,  W:, :].reshape(-1).contiguous()

        # 右半 queries 的两组
        if boost_mask_5d is not None:
            bm = boost_mask_5d
            assert bm.dim() == 4 and bm.shape[-1] == 1, "boost_mask 应为 (F,H,W,1)"

            bm_right = bm[:, :, :, 0].to(dtype=torch.bool, device=device)  # (F,H,W_right)
            right_all = grid[:, :, W:, :].reshape(-1)                        # 与 bm_right 展平对齐
            mask_flat = bm_right.reshape(-1)

            right_q_idx_masked = right_all[mask_flat]
        else:
            right_q_idx_masked = torch.empty(0, device=device, dtype=left_k_idx.dtype)


        return {
            "left_k_idx":           left_k_idx.long(),
            "right_k_idx":          right_k_idx.long(),
            "right_q_idx_masked":   right_q_idx_masked.long(),
        }


    def split_lr_indices_with_mask_5d_left(
        self, F:int, H:int, W:int, L:int = 1,
        boost_mask_5d: torch.Tensor | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        展平顺序固定为 (F, H, W, L) -> S
        生成：
        - left_k_idx_in   ：左半 ∩ boost_mask 的 key 线性索引
        - left_k_idx_out  ：左半 \ boost_mask 的 key 线性索引
        - right_q_idx_masked：右半 ∩ boost_mask 的 query 线性索引
        参数：
        boost_mask_5d: (F, H, W, 1) 的 0/1 或 bool mask（单个样本）
                        这个 W 与半幅对齐：用于贴到左半/右半各自的 W 宽度上
        """
        # 构造线性索引栅格（F, H, 2W, L）
        grid = torch.arange(F * H * (W * 2) * L, device=device).reshape(F, H, W * 2, L)

        # 为空时直接返回空索引
        if boost_mask_5d is None:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return {
                "left_k_idx_in":   empty,
                "left_k_idx_out":  empty,
                "right_q_idx_masked": empty,
            }

        bm = boost_mask_5d
        assert bm.dim() == 4 and bm.shape[-1] == 1 and bm.shape[2] == W, "boost_mask 应为 (F,H,W,1) 且 W 为半幅宽度"
        bm_bool = bm[..., 0].to(dtype=torch.bool, device=device)   # (F,H,W)

        # —— 左半 keys 的两组（与 bm 对齐）
        left_all  = grid[:, :, :W, :].reshape(-1)                  # (F,H,W,L)->(-1,)
        lmask     = bm_bool.reshape(-1)                            # 与 left_all 对齐
        left_k_idx_in  = left_all[lmask]
        left_k_idx_out = left_all[~lmask]

        # —— 右半 queries（mask 内）
        right_all = grid[:, :, W:, :].reshape(-1)                  # (F,H,W,L)->(-1,)
        rmask     = bm_bool.reshape(-1)                            # 同一张 mask 贴到右半
        right_q_idx_masked = right_all[rmask]

        return {
            "left_k_idx_in":        left_k_idx_in.long().contiguous(),
            "left_k_idx_out":       left_k_idx_out.long().contiguous(),
            "right_q_idx_masked":   right_q_idx_masked.long().contiguous(),
        }

    # —— 区域引导（质心版）：仅比较 “左半 mask内” vs “左半 mask外”，queries 取右半(mask内)
    def region_guidance_centroid_left_in_out(
        self,
        q: torch.Tensor,   # (B, S, Hs, D)
        k: torch.Tensor,   # (B, S, Hs, D)
        q_idx: torch.Tensor,           # 右半(mask内)的 queries 线性索引 (1D, on S)
        left_k_idx_in: torch.Tensor,   # 左半(mask内) keys 线性索引
        left_k_idx_out: torch.Tensor,  # 左半(mask外) keys 线性索引
        tau: float = 1.0,
    ):
        """
        返回： mean(attn_L_in[q_idx]) - mean(attn_L_out[q_idx])
        做法：对 keys 两组（左半 mask内 / 左半 mask外）各自做质心，然后把 q 与两质心做二分类 softmax，取概率差。
        """
        # if (q_idx.numel() == 0) or (left_k_idx_in.numel() == 0) or (left_k_idx_out.numel() == 0):
        #     return q.new_zeros(())

        if left_k_idx_out.numel() == 0:
            B, S, Hs, D = q.shape
            device = q.device
            q_idx = q_idx.to(device=device, dtype=torch.long)
            left_k_idx_in  = left_k_idx_in.to(device=device, dtype=torch.long)

            # —— 两组 key 的质心 (沿 S 聚合)
            kL_in  = k[:, left_k_idx_in].mean(dim=1)    # (B, Hs, D)

            # —— 计算每个 head 上 q 与两质心的点积 logits
            # q 先换到 (B, Hs, S, D)，便于逐 head 计算
            qh = q.permute(0, 2, 1, 3)                  # (B,Hs,S,D)

            scale = (math.sqrt(D) * tau)
            logit_in  = (qh * kL_in[:, :, None, :]).sum(-1)  / scale   # (B,Hs,S)

            # —— 二分类 softmax 概率：[...,0]=in, [...,1]=out
            logits = torch.stack([logit_in], dim=-1)        # (B,Hs,S,2)
            probs  = logits.softmax(dim=-1)
            attn_in = probs[..., 0]

            # 只在右半(mask内)的 queries 上取均值
            attn_in_sel  = attn_in[:, :, q_idx]                        # (B,Hs,|Q|)
            scores = 0 - attn_in_sel.mean()
            
            return scores

        B, S, Hs, D = q.shape
        device = q.device
        q_idx = q_idx.to(device=device, dtype=torch.long)
        left_k_idx_in  = left_k_idx_in.to(device=device, dtype=torch.long)
        left_k_idx_out = left_k_idx_out.to(device=device, dtype=torch.long)

        # —— 两组 key 的质心 (沿 S 聚合)
        kL_in  = k[:, left_k_idx_in].mean(dim=1)    # (B, Hs, D)
        kL_out = k[:, left_k_idx_out].mean(dim=1)   # (B, Hs, D)

        # —— 计算每个 head 上 q 与两质心的点积 logits
        # q 先换到 (B, Hs, S, D)，便于逐 head 计算
        qh = q.permute(0, 2, 1, 3)                  # (B,Hs,S,D)

        scale = (math.sqrt(D) * tau)
        logit_in  = (qh * kL_in[:, :, None, :]).sum(-1)  / scale   # (B,Hs,S)
        logit_out = (qh * kL_out[:, :, None, :]).sum(-1) / scale   # (B,Hs,S)

        # —— 二分类 softmax 概率：[...,0]=in, [...,1]=out
        logits = torch.stack([logit_in, logit_out], dim=-1)        # (B,Hs,S,2)
        probs  = logits.softmax(dim=-1)
        attn_in, attn_out = probs[..., 0], probs[..., 1]           # (B,Hs,S)

        # 只在右半(mask内)的 queries 上取均值
        attn_in_sel  = attn_in[:, :, q_idx]                        # (B,Hs,|Q|)
        attn_out_sel = attn_out[:, :, q_idx]                       # (B,Hs,|Q|)

        return attn_in_sel.mean() - attn_out_sel.mean()

    # —— 生成索引：右半(mask内)的 queries；左半(mask外) 与 右半(mask外) 的 keys；另外给出右半(mask外)的 queries 以备需要
    def split_lr_indices_with_mask_5d_left_outs(
        self, F:int, H:int, W:int, L:int = 1,
        boost_mask_5d: torch.Tensor | None = None,
        device: str | torch.device = "cuda",
    ):
        """
        展平顺序固定为 (F, H, W, L) -> S，实际总宽度是 2W。
        返回：
        - left_k_idx_out   ：左半 \ mask 的 key 线性索引
        - right_k_idx_out  ：右半 \ mask 的 key 线性索引
        - right_q_idx_masked：右半 ∩ mask 的 query 线性索引（你要在这部分上取均值）
        - right_q_idx_other ：右半 \ mask 的 query 线性索引（若你后续也需要）
        参数：
        boost_mask_5d: (F, H, W, 1) 的 0/1 或 bool mask，W 为半幅宽度
        """
        grid = torch.arange(F * H * (W * 2) * L, device=device).reshape(F, H, W * 2, L)

        if boost_mask_5d is None:
            empty = torch.empty(0, device=device, dtype=torch.long)
            return {
                "left_k_idx_out":     empty,
                "right_k_idx_out":    empty,
                "right_q_idx_masked": empty,
                "right_q_idx_other":  empty,
            }

        bm = boost_mask_5d
        assert bm.dim() == 4 and bm.shape[-1] == 1 and bm.shape[2] == W, "boost_mask 应为 (F,H,W,1) 且 W 为半幅宽度"
        bm_bool = bm[..., 0].to(dtype=torch.bool, device=device)   # (F,H,W)
        mflat   = bm_bool.reshape(-1)

        # 左半 keys
        left_all = grid[:, :, :W, :].reshape(-1)          # (F*H*W*L,)
        left_k_idx_out = left_all[~mflat]

        # 右半 keys
        right_all_k = grid[:, :, W:, :].reshape(-1)       # (F*H*W*L,)
        right_k_idx_out = right_all_k[~mflat]

        # 右半 queries：mask内 / mask外
        right_all_q = right_all_k                          # 同一展开
        right_q_idx_masked = right_all_q[mflat]
        right_q_idx_other  = right_all_q[~mflat]

        return {
            "left_k_idx_out":       left_k_idx_out.long().contiguous(),
            "right_k_idx_out":      right_k_idx_out.long().contiguous(),
            "right_q_idx_masked":   right_q_idx_masked.long().contiguous(),
            "right_q_idx_other":    right_q_idx_other.long().contiguous(),
        }


    # —— 区域引导（质心版）：比较 “左半 mask外(L_out)” vs “右半 mask外(R_out)”，在 “右半 mask内的 queries” 上统计
    def region_guidance_centroid_lr_out(
        self, 
        q: torch.Tensor,   # (B, S, Hs, D)
        k: torch.Tensor,   # (B, S, Hs, D)
        q_idx: torch.Tensor,            # 右半(mask内) queries 的索引 (1D on S)
        left_k_idx_out: torch.Tensor,   # 左半(mask外) keys
        right_k_idx_out: torch.Tensor,  # 右半(mask外) keys
        tau: float = 1.0,
    ):
        """
        返回： mean(attn_L_out[q_idx]) - mean(attn_R_out[q_idx])
        做法：对两组 keys（L_out / R_out）做质心，与 q 做二分类 softmax，取概率差并在 q_idx 上取均值。
        """
        if (q_idx.numel() == 0) or (left_k_idx_out.numel() == 0) or (right_k_idx_out.numel() == 0):
            return q.new_zeros(())

        B, S, Hs, D = q.shape
        dev = q.device
        q_idx          = q_idx.to(device=dev, dtype=torch.long)
        left_k_idx_out = left_k_idx_out.to(device=dev, dtype=torch.long)
        right_k_idx_out= right_k_idx_out.to(device=dev, dtype=torch.long)

        # keys 质心
        kL_out = k[:, left_k_idx_out].mean(dim=1)   # (B,Hs,D)
        kR_out = k[:, right_k_idx_out].mean(dim=1)  # (B,Hs,D)

        # q -> (B,Hs,S,D)
        qh = q.permute(0, 2, 1, 3)

        scale = (math.sqrt(D) * tau)
        logit_L_out = (qh * kL_out[:, :, None, :]).sum(-1) / scale   # (B,Hs,S)
        logit_R_out = (qh * kR_out[:, :, None, :]).sum(-1) / scale   # (B,Hs,S)

        logits = torch.stack([logit_L_out, logit_R_out], dim=-1)     # (B,Hs,S,2)
        probs  = logits.softmax(dim=-1)
        attn_L_out, attn_R_out = probs[..., 0], probs[..., 1]        # (B,Hs,S)

        # 只在右半(mask内) queries 上取均值
        score = (attn_L_out[:, :, q_idx] - attn_R_out[:, :, q_idx]).mean()
        return score


    def forward(self, x, freqs, obj_mask=None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)

        # ================== Return attn_L and attn_R, acc to input mask
        # q,k 来自你 flash-attn 前的投影；(B,S,H*D) --> (B,S,H,D)
        if obj_mask is not None:
            self.patch_size = [1,2,2]
            bs,f, h_img_attn, w_img_attn, c = obj_mask.shape       # half video

            # skip ip latent...
            if q.shape[1]!=f*h_img_attn*w_img_attn*2:
                q_l = rearrange(q, "b (f h w) d -> b f h w d", h=h_img_attn, w=w_img_attn*2)
                k_l = rearrange(k, "b (f h w) d -> b f h w d", h=h_img_attn, w=w_img_attn*2)
                q_l = q_l[:,1:]
                k_l = k_l[:,1:]
                q = rearrange(q_l, "b f h w d -> b (f h w) d")
                k = rearrange(k_l, "b f h w d -> b (f h w) d")

            # compute attention score
            q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
            k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)

            # get indexs
            attn_scores = []
            for b in range(bs):
                bm = obj_mask[b]                 # (F,H,W,1)

                # judge obj_mask type-- True no unedited area
                all_true = bool(obj_mask.bool().all())

                # attn edited area---weaken...
                key_dict = self.split_lr_indices_with_mask_5d_left(
                    F=f, H=h_img_attn, W=w_img_attn, L=1,
                    boost_mask_5d=bm, device=q.device
                )
                score_masked_in = self.region_guidance_centroid_left_in_out(
                    q[b:b+1], k[b:b+1],
                    key_dict["right_q_idx_masked"],
                    key_dict["left_k_idx_in"], key_dict["left_k_idx_out"], tau=1.0
                )
                # score 即 attn_L_in - attn_L_out
                attn_scores.append(score_masked_in)

                if not all_true:            # have unedited area
                    key_dict = self.split_lr_indices_with_mask_5d_left_outs(
                        F=f, H=h_img_attn, W=w_img_attn, L=1,
                        boost_mask_5d=bm, device=q.device
                    )
                    score_masked_out = self.region_guidance_centroid_lr_out(
                        q[b:b+1], k[b:b+1],
                        q_idx=key_dict["right_q_idx_masked"],
                        left_k_idx_out=key_dict["left_k_idx_out"],
                        right_k_idx_out=key_dict["right_k_idx_out"],
                        tau=1.0
                    )
                    # score 即 attn_L_out - attn_R_out
                    attn_scores.append(score_masked_out)

            return self.o(x), attn_scores
        else:
            return self.o(x)



class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, ):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class DiTBlock_w_attnscore(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention_w_attnscore(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs, obj_mask=None):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if obj_mask is not None:
            attn_o, attn_score = self.self_attn(input_x, freqs, obj_mask)
            x = self.gate(x, gate_msa, attn_o)

            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            return x, attn_score
        else:
            x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))

            x = x + self.cross_attn(self.norm3(x), context)
            input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = self.gate(x, gate_mlp, self.ffn(input_x))
            return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel_w_attnscore(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock_w_attnscore(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                obj_mask: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        
        print('Disabled function...')
        # t = self.time_embedding(
        #     sinusoidal_embedding_1d(self.freq_dim, timestep))
        # t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        # context = self.text_embedding(context)
        
        # if self.has_image_input:
        #     x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        #     clip_embdding = self.img_emb(clip_feature)
        #     context = torch.cat([clip_embdding, context], dim=1)
        
        # x, (f, h, w) = self.patchify(x)
        
        # freqs = torch.cat([
        #     self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        #     self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        #     self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        # ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        # def create_custom_forward(module):
        #     def custom_forward(*inputs):
        #         return module(*inputs)
        #     return custom_forward

        # all_attnscore = []
        # for block in self.blocks:
        #     if self.training and use_gradient_checkpointing:
        #         if use_gradient_checkpointing_offload:
        #             with torch.autograd.graph.save_on_cpu():
        #                 x, attn_score = torch.utils.checkpoint.checkpoint(
        #                     create_custom_forward(block),
        #                     x, context, t_mod, freqs,
        #                     use_reentrant=False, obj_mask=obj_mask,
        #                 )
        #         else:
        #             x, attn_score = torch.utils.checkpoint.checkpoint(
        #                 create_custom_forward(block),
        #                 x, context, t_mod, freqs,
        #                 use_reentrant=False, obj_mask=obj_mask,
        #             )
        #     else:
        #         x, attn_score = block(x, context, t_mod, freqs, obj_mask=obj_mask)

        #     all_attnscore.extend(attn_score)

        # x = self.head(x, t)
        # x = self.unpatchify(x, (f, h, w))
        # if obj_mask is not None:
        #     return x, all_attnscore
        # else:
        #     return x


    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    

class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb

    def patchify(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6d6ccde6845b95ad9114ab993d917893":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "349723183fc063b2bfc10bb2835cf677":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "efa44cddf936c70abd0ea28b6cbe946c":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        else:
            config = {}
        return state_dict, config
