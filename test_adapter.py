# test_adapter.py
import torch
from deepseek_moe_fused import FusedDeepSeekMoEMLP

moe = FusedDeepSeekMoEMLP(dim=768, num_routed_experts=4, top_k=2).cuda()
x = torch.randn(1, 1024, 768, dtype=torch.bfloat16, device='cuda')
out, aux = moe(x)
print(f"Input: {x.shape}, Output: {out.shape}")
print(f"Aux losses: {list(aux.keys())}")
print(f"Aux losses: {list(aux.values())}")