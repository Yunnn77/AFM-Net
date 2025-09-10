# base_mamba_vision_block_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional, Tuple, Literal, Sequence, List, Dict, Any, Union
import math
import random
import inspect
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
import logging

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# --- Mamba Imports ---
MAMBA_IMPORTED = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layer_norm import (
        RMSNorm as MambaRMSNorm,
        layer_norm_fn as mamba_layer_norm_fn,
        rms_norm_fn as mamba_rms_norm_fn,
    )
    MAMBA_IMPORTED = True
    logger.info("Successfully imported mamba_ssm components from base_mamba_vision_block_arch.")
except ImportError:
    Mamba, MambaRMSNorm, mamba_layer_norm_fn, mamba_rms_norm_fn = None, None, None, None
    logger.warning(
        "Warning (base_mamba_vision_block_arch): mamba_ssm or its triton ops not found..."
    )

    if Mamba is None:
        class Mamba(nn.Module):  # Placeholder to avoid TypeError
            def __init__(
                self,
                d_model,
                d_state=16,
                d_conv=4,
                expand=2,
                dt_rank="auto",
                dt_min=0.001,
                dt_max=0.1,
                dt_init="random",
                dt_scale=1.0,
                dt_init_floor=1e-4,
                conv_bias=True,
                bias=False,
                use_fast_path=True,
                layer_idx=None,
                device=None,
                dtype=None,
                factory_kwargs=None,  # capture factory_kwargs
                **other_kwargs,       # capture any other unexpected kwargs
            ):
                super().__init__()
                fk = factory_kwargs if factory_kwargs else {}
                _dev = fk.get("device", device)
                _dt = fk.get("dtype", dtype)
                eff_d = d_model if isinstance(d_model, int) else 192
                self.fc = nn.Linear(eff_d, eff_d, device=_dev, dtype=_dt)
                logger.info(f"Placeholder Mamba initialized: d_model={eff_d}")
                if other_kwargs:
                    logger.warning(
                        f"Placeholder Mamba received unexpected kwargs: {list(other_kwargs.keys())}"
                    )

            def forward(self, x, inference_params=None):
                return self.fc(x)

            def allocate_inference_cache(self, *args, **kwargs):
                return None

try:
    from rope import VisionRotaryEmbeddingFast

    ROPE_IMPORTED = True
except ImportError:
    VisionRotaryEmbeddingFast = None
    ROPE_IMPORTED = False
    logger.info(
        "VisionRotaryEmbeddingFast not found. RoPE features will be disabled if configured."
    )


class DSV2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__()

        factory_kwargs_to_use = kwargs.pop('factory_kwargs', {})
        device_arg = kwargs.pop('device', factory_kwargs_to_use.get('device'))
        dtype_arg = kwargs.pop('dtype', factory_kwargs_to_use.get('dtype'))

        final_fk = {}
        if device_arg is not None:
            final_fk['device'] = device_arg
        if dtype_arg is not None:
            final_fk['dtype'] = dtype_arg

        for k_dict in [factory_kwargs_to_use, kwargs]:
            for k, v in k_dict.items():
                if k not in final_fk:
                    final_fk[k] = v

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, **final_fk))

    def forward(self, x: torch.Tensor):
        if self.weight.device != x.device:
            self.weight = nn.Parameter(self.weight.to(x.device))

        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)

        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)

        return (self.weight * x_fp32).to(input_dtype)

class ViMPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        factory_kwargs=None,
    ):
        super().__init__()

        fk = factory_kwargs if factory_kwargs else {}
        _dev, _dt = fk.get('device'), fk.get('dtype')

        img_s = to_2tuple(img_size)
        patch_s = to_2tuple(patch_size)
        str_v = to_2tuple(stride)

        self.img_size = img_s
        self.patch_size = patch_s
        self.grid_size = (
            (img_s[0] - patch_s[0]) // str_v[0] + 1,
            (img_s[1] - patch_s[1]) // str_v[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Patch projection
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_s,
            stride=str_v,
            device=_dev,
            dtype=_dt,
        )

        # Optional normalization
        self.norm = norm_layer(embed_dim, **fk) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Optional assertion for fixed-size input
        # assert H == self.img_size[0] and W == self.img_size[1], "Input image size doesn't match model config."

        x = self.proj(x)  # Shape: [B, embed_dim, H', W']

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # Shape: [B, N_patches, embed_dim]

        return self.norm(x)

class VisionMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mamba_mixer_instance: nn.Module,
        norm_cls_constructor: Any,
        drop_path: float = 0.0
    ):
        super().__init__()
        self.norm = norm_cls_constructor(dim)
        self.mixer = mamba_mixer_instance
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: Tensor, inference_params=None) -> Tensor:
        residual = hidden_states
        norm_weight_dtype = getattr(self.norm, 'weight', hidden_states).dtype
        x_normed = self.norm(hidden_states.to(norm_weight_dtype))
        x_mixed = self.mixer(x_normed, inference_params=inference_params)
        x_dropped = self.drop_path(x_mixed)
        hidden_states_out = residual + x_dropped
        return hidden_states_out

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self.mixer, 'allocate_inference_cache'):
            return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None

def create_vision_mamba_block(
    d_model: int,
    d_state: int = 16,
    ssm_cfg: Optional[Dict[str, Any]] = None,
    norm_epsilon: float = 1e-5,
    drop_path: float = 0.0,
    rms_norm: bool = False,
    layer_idx: Optional[int] = None,
    factory_kwargs: Optional[Dict[str, Any]] = None,
    d_conv: int = 4,
    expand: int = 2,
    dt_rank: Union[str, int] = "auto",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dt_init: str = "random",
    dt_scale: float = 1.0,
    dt_init_floor: float = 1e-4,
    conv_bias: bool = True,
    bias: bool = False,
    use_fast_path: bool = True,
):
    fk = factory_kwargs if factory_kwargs else {}
    _ssm_cfg = ssm_cfg if ssm_cfg is not None else {}

    mamba_init_args = {
        'd_model': d_model,
        'd_state': _ssm_cfg.get('d_state', d_state),
        'd_conv': _ssm_cfg.get('d_conv', d_conv),
        'expand': int(_ssm_cfg.get('expand', expand)),
        'dt_rank': _ssm_cfg.get('dt_rank', dt_rank),
        'dt_min': _ssm_cfg.get('dt_min', dt_min),
        'dt_max': _ssm_cfg.get('dt_max', dt_max),
        'dt_init': _ssm_cfg.get('dt_init', dt_init),
        'dt_scale': _ssm_cfg.get('dt_scale', dt_scale),
        'dt_init_floor': _ssm_cfg.get('dt_init_floor', dt_init_floor),
        'conv_bias': _ssm_cfg.get('conv_bias', conv_bias),
        'bias': _ssm_cfg.get('bias', bias),
        'use_fast_path': _ssm_cfg.get('use_fast_path', use_fast_path),
        'layer_idx': layer_idx,
    }
    if 'device' in fk:
        mamba_init_args['device'] = fk['device']
    if 'dtype' in fk:
        mamba_init_args['dtype'] = fk['dtype']

    if Mamba is None or (Mamba.__name__ == 'Mamba' and not MAMBA_IMPORTED):
        mamba_instance = Mamba(**mamba_init_args, factory_kwargs=fk)
    else:
        mamba_sig_params = inspect.signature(Mamba.__init__).parameters
        valid_mamba_params = {k: v for k, v in mamba_init_args.items() if k in mamba_sig_params}
        mamba_instance = Mamba(**valid_mamba_params)

    NormTypeToUse = MambaRMSNorm if rms_norm and MAMBA_IMPORTED else (DSV2RMSNorm if rms_norm else nn.LayerNorm)
    norm_cls_constructor_partial = partial(NormTypeToUse, eps=norm_epsilon, **fk)

    block = VisionMambaBlock(
        dim=d_model,
        mamba_mixer_instance=mamba_instance,
        norm_cls_constructor=norm_cls_constructor_partial,
        drop_path=drop_path,
    )
    return block

@register_model
class VisionMambaPathArch(nn.Module):
    OUT_TYPES = {'featmap', 'avg_featmap', 'cls_token', 'raw'}

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_channels=3,
        embed_dim=192,
        depth=12,
        d_state=16,
        ssm_cfg: Optional[Dict[str, Any]] = None,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        if_abs_pos_embed=True,
        if_rope=False,
        final_norm=True,
        path_type='forward',
        cls_position='middle',
        pe_type='learnable',
        final_pool_type='avg_featmap',
        output_stage_block_counts: Optional[List[int]] = None,
        device=None,
        dtype=None,
        pt_hw_seq_len=14,
        **kwargs
    ):
        super().__init__()

        final_fk = {}
        _p_fk = kwargs.pop('factory_kwargs', {})
        dev_a = device if device is not None else _p_fk.get('device', None)
        dt_a = dtype if dtype is not None else _p_fk.get('dtype', None)
        if dev_a is not None:
            final_fk['device'] = torch.device(dev_a) if isinstance(dev_a, str) else dev_a
        if dt_a is not None:
            final_fk['dtype'] = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'float32': torch.float32}.get(
                str(dt_a).lower(), dt_a
            ) if isinstance(dt_a, str) else dt_a
        for k_iter, v_iter in _p_fk.items():
            if k_iter not in final_fk:
                final_fk[k_iter] = v_iter

        self.embed_dim = embed_dim
        self.path_type = path_type
        self.cls_position = cls_position
        self.pe_type = pe_type
        self.final_pool_type = final_pool_type
        self.num_classes = num_classes
        self.final_norm_flag = final_norm
        self.depth = depth
        self.output_stage_block_counts = output_stage_block_counts
        self.output_stage_indices: Optional[List[int]] = None

        if self.output_stage_block_counts:
            if not (isinstance(self.output_stage_block_counts, list) and all(isinstance(n, int) for n in self.output_stage_block_counts)):
                raise TypeError("output_stage_block_counts must be list of int.")
            if sum(self.output_stage_block_counts) != depth:
                raise ValueError(f"Sum of output_stage_block_counts must equal depth.")
            self.output_stage_indices = [int(x) - 1 for x in np.cumsum(self.output_stage_block_counts)]

        NormTypeForEmbedAndFinal = MambaRMSNorm if rms_norm and MAMBA_IMPORTED else (DSV2RMSNorm if rms_norm else nn.LayerNorm)
        norm_layer_for_extras = partial(NormTypeForEmbedAndFinal, eps=norm_epsilon, **final_fk)

        self.patch_embed = ViMPatchEmbed(
            img_size,
            patch_size,
            stride if stride else patch_size,
            in_channels,
            self.embed_dim,
            norm_layer_for_extras,
            factory_kwargs=final_fk
        )

        num_patches = self.patch_embed.num_patches
        self.patch_resolution = (self.patch_embed.grid_size[0], self.patch_embed.grid_size[1])
        self.num_extra_tokens = 0
        if self.cls_position not in ['none', None]:
            self.num_extra_tokens = 2 if self.cls_position == 'head_tail' else 1
            self.cls_token = nn.Parameter(torch.zeros(1, self.num_extra_tokens, self.embed_dim, **final_fk))
        else:
            self.cls_token = None

        self.pos_embed = None
        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dim, **final_fk))

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.if_rope = if_rope
        self.rope_for_vim = None
        if self.if_rope and VisionRotaryEmbeddingFast is not None and ROPE_IMPORTED:
            try:
                ft_seq_len = img_size // patch_size
                self.rope_for_vim = VisionRotaryEmbeddingFast(dim=embed_dim // 2, pt_seq_len=pt_hw_seq_len, ft_seq_len=ft_seq_len)
                logger.info("RoPE for VisionMambaCore initialized.")
            except Exception as e_rope:
                logger.warning(f"Failed to init RoPE: {e_rope}. RoPE disabled.")
        elif self.if_rope:
            logger.warning("VisionRotaryEmbeddingFast not available, RoPE disabled.")

        dpr_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList()
        self.path_gate_layers: Optional[nn.ModuleList] = None

        num_mamba_paths_active = 1 + ('reverse' in self.path_type) + ('shuffle' in self.path_type)
        if 'gate' in self.path_type and num_mamba_paths_active > 1:
            self.path_gate_layers = nn.ModuleList()
            for _ in range(depth):
                self.path_gate_layers.append(
                    nn.Sequential(
                        nn.Linear(num_mamba_paths_active * self.embed_dim, num_mamba_paths_active, bias=False, **final_fk),
                        nn.Softmax(dim=-1)
                    )
                )

        for i in range(depth):
            gate_mod = self.path_gate_layers[i] if self.path_gate_layers and i < len(self.path_gate_layers) else None
            self.layers.append(
                create_vision_mamba_block(
                    d_model=embed_dim,
                    d_state=d_state,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    drop_path=dpr_list[i],
                    rms_norm=rms_norm,
                    layer_idx=i,
                    factory_kwargs=final_fk,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_init=dt_init,
                    dt_scale=dt_scale,
                    dt_init_floor=dt_init_floor,
                    conv_bias=conv_bias,
                    bias=bias,
                    use_fast_path=use_fast_path
                )
            )

        self.norm_f = norm_layer_for_extras(embed_dim, **final_fk) if self.final_norm_flag else nn.Identity()
        self.pool_norm = norm_layer_for_extras(embed_dim, **final_fk) if self.final_pool_type == 'avg_featmap' and self.final_norm_flag else None
        self.head = nn.Linear(self.embed_dim, num_classes, **final_fk) if self.num_classes > 0 else nn.Identity()
        self.init_weights()

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)  # type: ignore
        self.apply(self._init_weights_general_vim)

    def _init_weights_general_vim(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, DSV2RMSNorm, (MambaRMSNorm if MAMBA_IMPORTED else type(None)))):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _pos_embed_fn(self, x):
        active_pe = self.pos_embed if isinstance(self.pos_embed, nn.Parameter) else getattr(self, '_pos_embed_tensor',
                                                                                            None)
        pos_drop_layer = self.pos_drop if hasattr(self, 'pos_drop') else nn.Identity()
        if active_pe is None:
            return pos_drop_layer(x)
        active_pe = active_pe.to(x.device)
        if x.size(1) == active_pe.size(1):
            x = x + active_pe
        elif x.size(1) < active_pe.size(1):
            x = x + active_pe[:, :x.size(1), :]
        elif x.size(1) > active_pe.size(1) and active_pe.size(1) > 0:
            repeat_factor = math.ceil(x.size(1) / active_pe.size(1))
            expanded_pe = active_pe.repeat(1, repeat_factor, 1)[:, :x.size(1), :]
            x = x + expanded_pe
        return pos_drop_layer(x)

    def _insert_cls_token(self, x: Tensor) -> Tensor:
        if self.cls_token is None:
            return x
        B = x.shape[0]
        cls_tok = self.cls_token.expand(B, -1, -1).to(x.device)
        if self.cls_position == 'head':
            x = torch.cat((cls_tok, x), dim=1)
        elif self.cls_position == 'tail':
            x = torch.cat((x, cls_tok), dim=1)
        elif self.cls_position == 'head_tail':
            if self.num_extra_tokens != 2 or cls_tok.size(1) != 2:
                raise ValueError("cls_position='head_tail' requires num_extra_tokens=2 and cls_token size[1]=2")
            x = torch.cat((cls_tok[:, :1], x, cls_tok[:, 1:]), dim=1)
        elif self.cls_position == 'middle':
            if self.num_extra_tokens != 1:
                raise ValueError("cls_position='middle' typically requires num_extra_tokens=1")
            mid_idx = x.size(1) // 2
            x = torch.cat((x[:, :mid_idx], cls_tok, x[:, mid_idx:]), dim=1)
        else:
            pass
        return x

    def _extract_output_features(self, x, patch_res):
        if self.final_pool_type == 'raw':
            return x
        if self.final_pool_type == 'cls_token':
            if self.cls_token is None:
                return x.mean(dim=1)
            if self.cls_position == 'head':
                return x[:, 0]
            else:
                return x[:, 0]
        patch_tok = x
        expected_num_patches = patch_res[0] * patch_res[1]
        if patch_tok.size(1) != expected_num_patches:
            return self.pool_norm(patch_tok.mean(dim=1)) if self.pool_norm else patch_tok.mean(dim=1)
        if self.final_pool_type == 'featmap':
            B, H_p, W_p = patch_tok.size(0), patch_res[0], patch_res[1]
            return patch_tok.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
        if self.final_pool_type == 'avg_featmap':
            return self.pool_norm(patch_tok.mean(dim=1)) if self.pool_norm else patch_tok.mean(dim=1)
        return x.mean(dim=1)

    def _apply_single_mamba_block(self, hidden_states: Tensor, layer_idx: int, inference_params=None) -> Tensor:
        hs_for_block = hidden_states
        if self.if_rope and self.rope_for_vim is not None:
            hs_for_block = self.rope_for_vim(hs_for_block)
        return self.layers[layer_idx](hs_for_block, inference_params=inference_params)

    def forward_features(self, x: Tensor, inference_params=None) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        x_p = self.patch_embed(x)
        x_cls = self._insert_cls_token(x_p)
        hidden_states = self._pos_embed_fn(x_cls)
        intermediate_features: List[Tensor] = []
        for i in range(self.depth):
            current_block_input = hidden_states
            if self.path_type == 'forward' or not (('reverse' in self.path_type) or ('shuffle' in self.path_type)):
                hidden_states = self._apply_single_mamba_block(current_block_input, i, inference_params)
            else:
                paths_out = []
                paths_out.append(self._apply_single_mamba_block(current_block_input, i, inference_params))
                if 'reverse' in self.path_type:
                    paths_out.append(torch.flip(
                        self._apply_single_mamba_block(torch.flip(current_block_input, [1]), i, inference_params), [1]))
                if 'shuffle' in self.path_type:
                    r_idx = torch.randperm(current_block_input.size(1), device=current_block_input.device)
                    shuffled_output = self._apply_single_mamba_block(current_block_input[:, r_idx], i, inference_params)
                    paths_out.append(shuffled_output[:, torch.argsort(r_idx)])
                if 'mean' in self.path_type:
                    hidden_states = torch.stack(paths_out, dim=0).mean(dim=0)
                elif 'gate' in self.path_type and self.path_gate_layers and i < len(self.path_gate_layers) and \
                        self.path_gate_layers[i] is not None:
                    g_in = torch.cat([p.mean(dim=1) for p in paths_out], dim=-1)
                    p_w = self.path_gate_layers[i](g_in).view(x.size(0), 1, len(paths_out), 1)
                    hidden_states = torch.sum(torch.stack(paths_out, dim=2) * p_w, dim=2)
                else:
                    hidden_states = torch.stack(paths_out, dim=0).mean(dim=0)
            if self.output_stage_indices and i in self.output_stage_indices:
                intermediate_features.append(hidden_states.clone())
        processed_output_for_head = self.norm_f(hidden_states) if self.final_norm_flag else hidden_states
        output_payload = self._extract_output_features(processed_output_for_head, self.patch_resolution)
        return (output_payload, intermediate_features) if self.output_stage_indices else output_payload

    def forward(self, x: Tensor, inference_params: Optional[Any] = None, return_features: bool = False):
        forward_output = self.forward_features(x, inference_params)
        features_for_head: Tensor
        intermediate_stages: List[Tensor] = []
        if self.output_stage_indices and isinstance(forward_output, tuple):
            features_for_head, intermediate_stages = forward_output
        else:
            features_for_head = forward_output
        if return_features:
            return features_for_head, torch.tensor(0.0, device=x.device, dtype=x.dtype), intermediate_stages
        return self.head(features_for_head)
