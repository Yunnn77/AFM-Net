# moe_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal, List, Dict, Any
from dataclasses import dataclass
import math

_DSV2_WORLD_SIZE = 1
_DSV2_RANK = 0
def get_dsv2_world_size():
    return _DSV2_WORLD_SIZE
def get_dsv2_rank():
    return _DSV2_RANK

@dataclass
class DSV2ModelArgs:
    dim: int = 2048
    moe_inter_dim: int = 1408
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    aux_loss_alpha: float = 0.01
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8", "fp32", "float32", "float16"] = "bf16"
    vocab_size: int = 102400
    inter_dim: int = 10944
    n_layers: int = 27

class GateDSV2(nn.Module):
    def __init__(self, args: DSV2ModelArgs, factory_kwargs=None):
        super().__init__()
        fk = factory_kwargs if factory_kwargs else {}
        _device, _dtype = fk.get('device'), fk.get('dtype')
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.n_routed_experts = args.n_routed_experts
        if self.n_routed_experts > 0:
            self.gate_proj = nn.Linear(self.dim, self.n_routed_experts, bias=False, device=_device, dtype=_dtype)
        else:
            self.gate_proj = nn.Identity()
        self.bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B_N = x.size(0)
        if self.n_routed_experts == 0 or self.topk == 0:
            return (torch.empty((B_N, 0), device=x.device, dtype=x.dtype),
                    torch.empty((B_N, 0), device=x.device, dtype=torch.long),
                    torch.empty((B_N, 0), device=x.device, dtype=torch.float32))
        scores_logits = self.gate_proj(x)
        if self.bias is not None:
            scores_logits = scores_logits + self.bias
        if self.score_func == "softmax":
            gate_probs_for_aux = scores_logits.softmax(dim=-1, dtype=torch.float32)
        else:
            gate_probs_for_aux = scores_logits.sigmoid().to(dtype=torch.float32)
        final_scores_for_routing = scores_logits
        original_activations = gate_probs_for_aux.to(x.dtype)
        if self.n_groups > 1:
            logits_g = scores_logits.view(B_N, self.n_groups, -1)
            group_s_logits = logits_g.amax(dim=-1) if self.bias is None else logits_g.topk(min(2, logits_g.size(-1)), dim=-1)[0].sum(dim=-1)
            top_g_indices = group_s_logits.topk(self.topk_groups, dim=-1)[1]
            g_mask_for_fill = logits_g.new_ones(B_N, self.n_groups, dtype=torch.bool).scatter_(1, top_g_indices, False)
            final_scores_for_routing = logits_g.masked_fill(g_mask_for_fill.unsqueeze(-1), float("-inf")).flatten(1)
        if self.score_func == "softmax":
            proc_s_routing = final_scores_for_routing.softmax(dim=-1, dtype=torch.float32).to(x.dtype)
        else:
            proc_s_routing = final_scores_for_routing.sigmoid().to(x.dtype)
        actual_topk = min(self.topk, self.n_routed_experts)
        if actual_topk == 0:
            return (torch.empty((B_N, 0), device=x.device, dtype=x.dtype),
                    torch.empty((B_N, 0), device=x.device, dtype=torch.long),
                    gate_probs_for_aux)
        _, top_k_indices = torch.topk(proc_s_routing, actual_topk, dim=-1)
        weights_f = original_activations.gather(dim=1, index=top_k_indices)
        if self.score_func == "sigmoid":
            weights_f = weights_f / (weights_f.sum(dim=-1, keepdim=True) + 1e-6)
        return weights_f * self.route_scale, top_k_indices, gate_probs_for_aux

class ExpertDSV2(nn.Module):
    def __init__(self, dim: int, inter_dim: int, factory_kwargs=None):
        super().__init__()
        fk = factory_kwargs if factory_kwargs else {}
        _device, _dtype = fk.get('device'), fk.get('dtype')
        self.w1 = nn.Linear(dim, inter_dim, bias=False, device=_device, dtype=_dtype)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, device=_device, dtype=_dtype)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, device=_device, dtype=_dtype)
        for w_param in [self.w1.weight, self.w2.weight, self.w3.weight]:
            nn.init.kaiming_uniform_(w_param, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MLPDSV2(nn.Module):
    def __init__(self, dim: int, inter_dim: int, factory_kwargs=None):
        super().__init__()
        fk = factory_kwargs if factory_kwargs else {}
        _device, _dtype = fk.get('device'), fk.get('dtype')
        self.w1 = nn.Linear(dim, inter_dim, bias=False, device=_device, dtype=_dtype)
        self.w3 = nn.Linear(dim, inter_dim, bias=False, device=_device, dtype=_dtype)
        self.w2 = nn.Linear(inter_dim, dim, bias=False, device=_device, dtype=_dtype)
        for w_param in [self.w1.weight, self.w2.weight, self.w3.weight]:
            nn.init.kaiming_uniform_(w_param, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEDSV2(nn.Module):
    def __init__(self, args: DSV2ModelArgs, factory_kwargs=None):
        super().__init__()
        fk = factory_kwargs if factory_kwargs else {}
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.aux_loss_alpha = args.aux_loss_alpha

        ws = get_dsv2_world_size()
        rk = get_dsv2_rank()
        if self.n_routed_experts > 0:
            if self.n_routed_experts % ws != 0:
                raise ValueError(f"n_routed_experts must be divisible by world_size")
            self.n_local_experts = self.n_routed_experts // ws
        else:
            self.n_local_experts = 0

        self.exp_start = rk * self.n_local_experts
        self.exp_end = (rk + 1) * self.n_local_experts
        self.gate = GateDSV2(args, factory_kwargs=fk)
        self.experts = nn.ModuleList()
        if self.n_routed_experts > 0:
            for i in range(self.n_routed_experts):
                if self.exp_start <= i < self.exp_end:
                    self.experts.append(ExpertDSV2(args.dim, args.moe_inter_dim, factory_kwargs=fk))
                else:
                    self.experts.append(None)

        self.shared_mlp = None
        if args.n_shared_experts > 0 and args.moe_inter_dim > 0:
            s_int_dim = args.n_shared_experts * args.moe_inter_dim
            if s_int_dim > 0:
                self.shared_mlp = MLPDSV2(args.dim, s_int_dim, factory_kwargs=fk)
            else:
                print(f"Warning (MoEDSV2): Shared expert inter_dim = {s_int_dim}, disabled.")

        self._last_batch_stats: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        x_f = x.view(-1, D)
        aux_l = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        if self.n_routed_experts == 0 and self.shared_mlp is None:
            return torch.zeros_like(x), aux_l.to(x.dtype)

        y = torch.zeros_like(x_f)

        if self.n_routed_experts > 0 and self.n_activated_experts > 0:
            weights, indices, gate_p_aux = self.gate(x_f)
            if weights.numel() > 0 and indices.numel() > 0:
                if self.training and self.aux_loss_alpha > 0 and gate_p_aux.numel() > 0:
                    toks_per_expert_mask = F.one_hot(indices, num_classes=self.n_routed_experts).sum(dim=1)
                    toks_p_exp_f = toks_per_expert_mask.float().mean(dim=0)
                    avg_g_p_exp = gate_p_aux.mean(dim=0)
                    if toks_p_exp_f.shape[0] == avg_g_p_exp.shape[0] == self.n_routed_experts:
                        aux_l = (toks_p_exp_f * avg_g_p_exp).sum() * self.n_routed_experts * self.aux_loss_alpha

                for i in range(self.n_routed_experts):
                    if not (self.exp_start <= i < self.exp_end and self.experts[i] is not None):
                        continue
                    tok_idx_i, k_pos_i = torch.where(indices == i)
                    if tok_idx_i.numel() > 0:
                        expert_out = self.experts[i](x_f[tok_idx_i])
                        weight_i = weights[tok_idx_i, k_pos_i].unsqueeze(-1)
                        y.index_add_(0, tok_idx_i, expert_out * weight_i)

        if self.shared_mlp is not None:
            y = y + self.shared_mlp(x_f)

        return y.view(B, N, D), aux_l.to(y.dtype)