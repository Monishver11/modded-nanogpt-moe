import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl
from kernels import get_kernel
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    Code adapted from https://github.com/NoahAmsel/PolarExpress/tree/main by @varunneal.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# NorMuon optimizer

class NorMuon(torch.optim.Optimizer):
    """
    Single-GPU version of NorMuon optimizer.
    Removes all distributed communication logic.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = 1  # Always 1 for single GPU
        
        # Always use standard param groups for single GPU
        param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            if "momentum_buffer" in group:
                group["momentum_buffer"].zero_()
            if "second_momentum_buffer" in group:
                group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """Creates one param group per module type."""
        from collections import defaultdict
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            param_groups.append(dict(params=group_params))
        
        return param_groups

    @torch.no_grad()
    def step(self):
        """Single-GPU step - no distributed communication."""
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            # Stack all gradients for this group
            grad_list = [p.grad for p in params]
            stacked_grads = torch.stack(grad_list)
            
            # Initialize momentum buffers if needed
            if "momentum_buffer" not in group:
                group["momentum_buffer"] = torch.zeros_like(stacked_grads)
            momentum_buffer = group["momentum_buffer"]
            
            # Apply momentum update
            momentum_buffer.lerp_(stacked_grads, 1 - group["momentum"])
            updated_grads = stacked_grads.lerp_(momentum_buffer, group["momentum"])
            
            grad_shape = updated_grads.shape
            ref_param = params[0]
            param_shape = ref_param.shape
            
            # Handle attn param reshaping
            if ref_param.label == 'attn':
                for p in params:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
            
            # Initialize second momentum buffer if needed
            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (
                    torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1] 
                    else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]
            
            # Initialize learning rate multipliers if needed
            if "param_lr" not in group:
                group["param_lr"] = (
                    max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                    * ref_param.new_tensor(
                        [getattr(param, "lr_mul", 1.0) for param in params]
                    ).view(-1, 1, 1)
                )
                group["param_wd"] = ref_param.new_tensor(
                    [getattr(param, "wd_mul", 1.0) for param in params]
                ).view(-1, 1, 1)
            
            # Compute effective LR and weight decay
            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["lr"] * group["weight_decay"] * group["param_wd"]
            
            # Apply Polar Express orthogonalization
            v_chunk = polar_express(updated_grads)
            
            # NorMuon: track squared magnitude along one dimension
            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(
                dim=-1 if param_shape[-2] >= param_shape[-1] else -2, 
                keepdim=True
            )
            second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))
            
            # Reshape back if needed
            v_chunk = v_chunk.view(grad_shape)
            
            # Stack parameters
            param_chunk = torch.stack(params)
            
            # "Cautious" weight decay
            mask = (v_chunk * param_chunk) >= 0
            v_chunk.addcmul_(param_chunk, (eff_wd * mask).to(ref_param.dtype))
            
            # Update parameters
            param_chunk.addcmul_(v_chunk, -eff_lr)
            
            # Write back to original parameters
            for i, p in enumerate(params):
                p.copy_(param_chunk[i])

class DistAdam(torch.optim.AdamW):
    """Simplified AdamW for single GPU - replaces the distributed version"""
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.should_sync = False  # Keep this attribute for compatibility
    
    def step(self):
        # Just call the parent class step
        super().step()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        # Disable FP8 by default for 4070 compatibility
        self.use_fp8 = False  # Force disable FP8 for Ada Lovelace
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        # Always use standard linear (BF16)
        return F.linear(x, self.weight.type_as(x))

# yarn implementation @classiclarryd
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()
        
    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # make matrices the same shape as MLP to enable batched call in optimizer
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim*4))
        # label module to enable custom optimizer sizing
        self.qkvo_w.label='attn'
        with torch.no_grad():
            self.qkvo_w.view(4,self.hdim, self.dim)[:3].uniform_(-bound, bound) # init QKV weights
            self.qkvo_w.view(4,self.hdim, self.dim)[3].zero_() # init output weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        # label module to enable custom optimizer sizing
        self.attn_gate.weight.label = 'attn_gate'
        self.attn_gate.weight.detach().zero_()

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v) # @ KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # Reshape for standard attention: [1, T, H, D] -> [1, H, T, D]
        q_attn = q.transpose(1, 2)  # [B, H, T, D]
        k_attn = k.transpose(1, 2)
        v_attn = v.transpose(1, 2)

        # Use PyTorch's native Flash Attention 2 (works on 4070)
        y = F.scaled_dot_product_attention(
            q_attn, k_attn, v_attn,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=attn_scale
        )

        # Reshape back: [B, H, T, D] -> [B, T, H, D]
        y = y.transpose(1, 2).contiguous()

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y

class MoEMLP(nn.Module):
    """
    Mixture of Experts MLP with top-1 routing.
    Compatible with NorMuon optimizer and torch.compile (no data-dependent branching).
    """
    def __init__(self, dim: int, num_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        hdim = 4 * dim
        
        # Router: maps from dim to num_experts logits
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.router.weight.label = 'moe_router'
        
        # Initialize router to zeros for uniform routing at start
        with torch.no_grad():
            self.router.weight.zero_()
        
        # Create experts - each has same structure as original MLP
        # Store as separate parameters to maintain compatibility with optimizer
        self.expert_fc = nn.ParameterList([
            nn.Parameter(torch.empty(dim, hdim)) for _ in range(num_experts)
        ])
        self.expert_proj = nn.ParameterList([
            nn.Parameter(torch.empty(dim, hdim)) for _ in range(num_experts)
        ])
        
        # Label all expert params as 'mlp' for NorMuon optimizer
        for i in range(num_experts):
            self.expert_fc[i].label = 'mlp'
            self.expert_proj[i].label = 'mlp'
        
        # Initialize expert weights same as original MLP
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            for i in range(num_experts):
                self.expert_fc[i].uniform_(-bound, bound)
                self.expert_proj[i].zero_()
    
    def forward(self, x: Tensor, return_aux_loss: bool = False):
        """
        Args:
            x: Input tensor of shape [B, T, dim]
            return_aux_loss: If True, return (output, aux_loss) tuple
        
        Returns:
            output: Tensor of shape [B, T, dim]
            aux_loss (optional): Load balancing loss scalar
        """
        B, T, D = x.shape
        assert D == self.dim
        
        # Compute router logits and probabilities
        router_logits = self.router(x)  # [B, T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, T, num_experts]
        
        # Top-1 routing: select expert with highest probability for each token
        expert_weights, expert_ids = torch.max(router_probs, dim=-1)  # [B, T], [B, T]
        
        # Flatten for easier processing
        x_flat = x.view(B * T, D)  # [B*T, D]
        expert_ids_flat = expert_ids.view(B * T)  # [B*T]
        
        # Initialize output (will accumulate expert outputs)
        output_flat = torch.zeros_like(x_flat)  # [B*T, D]
        
        # Process each expert WITHOUT data-dependent branching
        for expert_idx in range(self.num_experts):
            # Create mask for tokens assigned to this expert
            expert_mask = (expert_ids_flat == expert_idx).float()  # [B*T] - use float for masking
            
            # Always process all tokens, but mask out non-assigned ones
            # This avoids data-dependent control flow
            
            # Forward through this expert's MLP
            h = F.linear(x_flat, self.expert_fc[expert_idx].T.type_as(x_flat))
            h = F.relu(h).square()
            expert_output = F.linear(h, self.expert_proj[expert_idx].type_as(h))  # [B*T, D]
            
            # Mask and accumulate: only add output for tokens assigned to this expert
            output_flat = output_flat + expert_output * expert_mask.unsqueeze(-1)
        
        # Reshape back
        output = output_flat.view(B, T, D)
        
        # Compute auxiliary load balancing loss if requested
        if return_aux_loss:
            # Encourage uniform expert usage
            avg_expert_probs = router_probs.mean(dim=(0, 1))  # [num_experts]
            
            # Entropy-based loss
            eps = 1e-10
            aux_loss = -torch.sum(avg_expert_probs * torch.log(avg_expert_probs + eps))
            aux_loss = aux_loss / math.log(self.num_experts)
            aux_loss = 1.0 - aux_loss
            
            return output, aux_loss
        
        return output
class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make matrices the same shape to enable batched call in optimizer
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        # label modules to enable custom optimizer sizing
        self.c_fc.label='mlp'
        self.c_proj.label='mlp'
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int, use_moe: bool = False, num_experts: int = 4):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        # Optionally replace with MoE
        if layer_idx != 0:
            if use_moe:
                self.mlp = MoEMLP(dim, num_experts=num_experts)
            else:
                self.mlp = MLP(dim)
        else:
            self.mlp = None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, 
                 model_dim: int, max_seq_len: int, use_moe: bool = False, num_experts: int = 4):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.detach().zero_()
        self.smear_gate.weight.label = 'smear_gate'
        
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        
        # Create blocks with optional MoE
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads, i, use_moe=use_moe, num_experts=num_experts) 
            for i in range(num_layers)
        ])
        
        self.yarn = Yarn(head_dim, max_seq_len)
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, 
                                    x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        self.lm_head.weight.detach().zero_()
        
        # Rest of __init__ remains the same...
        assert num_layers % 2 == 0
        pad = 0  # No padding for world_size=1
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5 * torch.ones(num_layers),
                    *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
                    *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
                    torch.zeros(1),
                    0.5*torch.ones(1),
                    torch.ones(pad),
                ]
            )
        )
        
        # Set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0
    
        # forward() method remains unchanged

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq)

        # smear token embed forward 1 position @classiclarryd
        smear_lambda = self.scalars[5 * len(self.blocks)]
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # U-net design by @brendanh0gan
        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        backout_lambda = self.scalars[5 * len(self.blocks)+1]

        n = len(self.blocks) // 2

        x_backout = None
        backout_layer = 8
        # skip layer zero
        for i in range(1,len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            # since layer 0 is skipped, layer 11 does not have skip_connection
            if i >= n and i<11:
                gate = torch.sigmoid(skip_weights[i - n])  # in (0, 1)
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 8 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()
    
    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()
    
    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter==5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead of position {cur}; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter+=1
        return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()
    
    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()
    
    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()
    
    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens  # No sharding for single GPU

        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[0]), torch.tensor(seq_ends[0])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos # No rank offset for single GPU
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 524288  # Reduced from 10485760 (20x smaller)
    
    # CRITICAL: Reduce batch sizes for 12GB VRAM
    train_batch_size: int = 4096   # was 2048
    train_max_seq_len: int = 1024  # Reduced from 2048
    val_batch_size: int = 4096     # match train batch
    
    # optimization
    num_scheduled_iterations: int = 11500
    num_extension_iterations: int = 500
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.50
    
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13
    ws_validate_post_yarn_ext: int = 20

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# Single GPU setup
rank = 0
world_size = 1
grad_accum_steps = 1  # Adjust based on your target batch size
assert torch.cuda.is_available()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
master_process = True  # Always true for single GPU

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=max(args.train_batch_size, args.val_batch_size) // (grad_accum_steps * world_size),
    use_moe=True,      # Enable MoE
    num_experts=4      # Number of experts per MoE layer
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() 
                        if p.ndim >= 2 and "embed" not in n and "gate" not in n and "router" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n or "router" in n]

# init the optimizer(s)
optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=0.008,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(
    hidden_matrix_params + gate_params,  # gate_params now includes router
    lr=0.03, 
    momentum=0.95, 
    beta2=0.95, 
    weight_decay=1.2,
    custom_sizing=False
)

optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then linear decay
def get_lr(step: int):
    x = min(0.9999, step / args.num_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    # set short window size to half of long window size
    # on final step return specific ws for validation
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

def step_optimizers(step: int, optimizers, model):
    # update lr
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # set muon momentum based on step
    momentum = get_muon_momentum(step)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    # on even steps, only step Muon params
    # on odd steps, step all params
    if step%2==0:
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)

# Option 1: Keep compilation but be aware of shape constraints
# model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)

# Option 2: Use dynamic shapes for more flexibility (slightly slower)
model: nn.Module = torch.compile(model, dynamic=True, fullgraph=True)

# Option 3: Disable compilation initially for debugging
# model: nn.Module = model  # No compilation

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    # each window size is a new graph, need to warm up each with Yarn.attn_scale
    ws_idx = step % len(args.ws_schedule)
    if ws_idx==0:
        model.yarn.reset()
        ws_long = args.ws_schedule[0]
    else:
        new_ws_long = args.ws_schedule[ws_idx]  
        if new_ws_long != ws_long:
            model.yarn.apply(ws_long, new_ws_long)
            ws_long = new_ws_long
    model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.yarn.reset() # rotary buffer is not stored in state_dict
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(args.train_files, args.train_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
ws_short, ws_long = get_ws(0)
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long=new_ws_long

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
        val_loss /= val_steps
        del val_loader
        
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    for _ in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        model(inputs, targets, cum_seqlens, ws_short, ws_long).backward()
    step_optimizers(step, optimizers, model)
     
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
