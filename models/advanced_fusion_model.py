# models/advanced_fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Tuple, Optional, Dict, Any, Union, List
import inspect
import logging

# --- Setup Logger and Dependencies (with fallbacks) ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .attention_blocks import FEDAB_Enhanced_DWSC

_ATTN_BLOCKS_IMPORTED = True

from base_mamba_vision_block_arch import VisionMambaPathArch

_VMPA_IMPORTED = True

from .moe_blocks import DSV2ModelArgs, MoEDSV2

_MOE_BLOCKS_IMPORTED = True


# --- Helper Classes ---
class MLPHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[List[int]] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        layers = []
        current_dim = in_features

        if hidden_features and all(h > 0 for h in hidden_features):
            for h_dim in hidden_features:
                layers.extend([
                    nn.Linear(current_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                ])
                current_dim = h_dim

        layers.append(nn.Linear(current_dim, out_features))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)


class AdvancedFusionUnit_FEDAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_planes_factor: float,
        num_blocks: int,
        output_channels: int,
        use_dwsc: bool,
    ):
        super().__init__()
        self.actual_out_channels = output_channels

        bottleneck_p = max(1, int(in_channels * bottleneck_planes_factor))
        self.block = FEDAB_Enhanced_DWSC(
            in_channels=in_channels,
            bottleneck_planes=bottleneck_p,
            out_channels_after_att=output_channels,
            num_bottleneck_blocks=num_blocks,
            use_dwsc_for_branches=use_dwsc,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# --- Main Model ---
class AdvancedFusionModel(nn.Module):
    def __init__(
        self,
        mamba_config: Dict[str, Any],
        num_classes: int,
        resnet_variant: str = "resnet50",
        branch_fusion_type: str = "concat",
        fusion_aggregation_mode: str = "concat",
        resnet_pretrained: bool = False,
        cnn_output_layers: Optional[List[str]] = ["layer2", "layer3", "layer4"],
        mamba_stage_block_counts: Optional[List[int]] = [3, 3, 4],
        adapter_channel_dim: int = 128,
        fusion_moe_config: Dict[str, Any] = {},
        mamba_branch_trainable: bool = True,
        cnn_branch_trainable: bool = True,
        use_cnn_branch: bool = True,
        use_mamba_branch: bool = True,
        use_fusion_moe: bool = True,
        mlp_head_hidden_dims: Optional[List[int]] = [512],
        mlp_head_dropout: float = 0.3,
        use_fedab_on_cnn: bool = False,
        fedab_cnn_bottleneck_planes_factor: float = 0.25,
        fedab_cnn_num_blocks: int = 1,
        fedab_cnn_use_dwsc: bool = True,
        use_fedab_on_mamba: bool = False,
        fedab_mamba_bottleneck_planes_factor: float = 0.5,
        fedab_mamba_num_blocks: int = 1,
        fedab_mamba_use_dwsc: bool = True,
        fusion_unit_type: str = "simple_conv",
        advanced_fedab_fusion_bottleneck_planes_factor: float = 0.125,
        advanced_fedab_fusion_num_blocks: int = 1,
        advanced_fedab_fusion_output_channels: int = 256,
        advanced_fedab_fusion_use_dwsc: bool = True,
        simple_conv_fusion_output_channels: int = 256,
        **kwargs,
    ):
        super().__init__()
        # --- 1. Parameter Validation & Storage ---
        self.branch_fusion_type = branch_fusion_type
        self.fusion_aggregation_mode = fusion_aggregation_mode

        if self.branch_fusion_type not in ["concat", "add", "multiply"]:
            raise ValueError(f"Invalid branch_fusion_type: '{branch_fusion_type}'")

        if self.fusion_aggregation_mode not in ["concat", "residual", "dense"]:
            raise ValueError(f"Invalid fusion_aggregation_mode: '{fusion_aggregation_mode}'")

        if fusion_unit_type not in ["simple_conv", "advanced_fedab"]:
            raise ValueError(f"Invalid fusion_unit_type: '{fusion_unit_type}'.")

        if (
                self.fusion_aggregation_mode == "dense"
                and self.branch_fusion_type != "concat"
        ):
            raise ValueError("'dense' aggregation requires 'concat' branch fusion.")

        self.num_classes = num_classes
        self.adapter_channel_dim = adapter_channel_dim

        self.use_cnn_branch = use_cnn_branch
        self.use_mamba_branch = use_mamba_branch
        self.use_fusion_moe = use_fusion_moe

        self.use_fedab_on_cnn = use_fedab_on_cnn and _ATTN_BLOCKS_IMPORTED
        self.use_fedab_on_mamba = use_fedab_on_mamba and _ATTN_BLOCKS_IMPORTED

        self.fusion_unit_type = fusion_unit_type
        self.cnn_output_layers = cnn_output_layers or []
        self.mamba_stage_block_counts = mamba_stage_block_counts or []

        # --- 2. CNN Branch Initialization ---
        self.cnn_backbone: Optional[nn.Module] = None
        self.cnn_adapter_layers: Optional[nn.ModuleList] = None
        self.cnn_fedab_enhancers: Optional[nn.ModuleList] = None

        if self.use_cnn_branch:
            if not self.cnn_output_layers:
                raise ValueError("`cnn_output_layers` cannot be empty if `use_cnn_branch` is True.")

            # --- Select ResNet Variant ---
            if resnet_variant == "resnet18":
                weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if resnet_pretrained else None
                self.cnn_backbone = tv_models.resnet18(weights=weights)
                _cnn_layer_dims = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
            elif resnet_variant == "resnet50":
                weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if resnet_pretrained else None
                self.cnn_backbone = tv_models.resnet50(weights=weights)
                _cnn_layer_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
            else:
                raise ValueError(
                    f"Unsupported resnet_variant: '{resnet_variant}'. Supported: 'resnet18', 'resnet50'."
                )

            logger.info(f"CNN Backbone: {resnet_variant} initialized.")

            # --- Freeze CNN Backbone if not trainable ---
            if not cnn_branch_trainable:
                for p in self.cnn_backbone.parameters():
                    p.requires_grad_(False)

            # --- Feature Dimension and Hook Setup ---
            self.cnn_feature_dims_orig = [_cnn_layer_dims[name] for name in self.cnn_output_layers]
            self._cnn_features_cache: Dict[str, torch.Tensor] = {}
            for name in self.cnn_output_layers:
                if hasattr(self.cnn_backbone, name):
                    dict(self.cnn_backbone.named_modules())[name].register_forward_hook(
                        self._get_cnn_hook(name)
                    )
                else:
                    logger.error(f"Layer '{name}' not found in '{resnet_variant}'.")

            # --- Adapter Layers ---
            self.cnn_adapter_layers = nn.ModuleList(
                [nn.Conv2d(dim, self.adapter_channel_dim, kernel_size=1) for dim in self.cnn_feature_dims_orig]
            )

            # --- Optional FEDAB Enhancers ---
            if self.use_fedab_on_cnn:
                self.cnn_fedab_enhancers = nn.ModuleList(
                    [
                        AdvancedFusionUnit_FEDAB(
                            in_channels=self.adapter_channel_dim,
                            output_channels=self.adapter_channel_dim,
                            bottleneck_planes_factor=fedab_cnn_bottleneck_planes_factor,
                            num_blocks=fedab_cnn_num_blocks,
                            use_dwsc=fedab_cnn_use_dwsc,
                        )
                        for _ in self.cnn_output_layers
                    ]
                )
                logger.info("Post-Adapter CNN FEDAB enhancement is enabled (Lightweight Mode).")

        # --- 3. Mamba Branch Initialization ---
        self.mamba_branch: Optional[VisionMambaPathArch] = None
        self.mamba_adapter_layers: Optional[nn.ModuleList] = None
        self.mamba_fedab_enhancers: Optional[nn.ModuleList] = None

        if self.use_mamba_branch:
            if not _VMPA_IMPORTED or VisionMambaPathArch is None:
                raise ImportError("Mamba branch enabled, but VisionMambaPathArch is not available.")

            # --- Initialize Mamba Branch ---
            vm_params = mamba_config.copy()
            vm_params.update({
                "depth": sum(self.mamba_stage_block_counts),
                "output_stage_block_counts": self.mamba_stage_block_counts,
                "num_classes": 0
            })
            self.mamba_branch = VisionMambaPathArch(**vm_params)

            # Freeze Mamba branch if not trainable
            if not mamba_branch_trainable:
                for p in self.mamba_branch.parameters():
                    p.requires_grad_(False)

            self.mamba_embed_dim = self.mamba_branch.embed_dim
            self.mamba_h_patch, self.mamba_w_patch = self.mamba_branch.patch_embed.grid_size

            # --- Adapter Layers for Mamba ---
            self.mamba_adapter_layers = nn.ModuleList(
                [nn.Linear(self.mamba_embed_dim, self.adapter_channel_dim) for _ in self.mamba_stage_block_counts]
            )

            # --- Optional FEDAB Enhancers for Mamba ---
            if self.use_fedab_on_mamba:
                self.mamba_fedab_enhancers = nn.ModuleList(
                    [
                        AdvancedFusionUnit_FEDAB(
                            in_channels=self.adapter_channel_dim,
                            output_channels=self.adapter_channel_dim,
                            bottleneck_planes_factor=fedab_mamba_bottleneck_planes_factor,
                            num_blocks=fedab_mamba_num_blocks,
                            use_dwsc=fedab_mamba_use_dwsc,
                        )
                        for _ in self.mamba_stage_block_counts
                    ]
                )

        # --- 4. Multi-scale Fusion Blocks ---
        self.scale_fusion_blocks = nn.ModuleList()
        num_fusion_stages = len(self.cnn_output_layers or self.mamba_stage_block_counts or [])

        if use_cnn_branch and use_mamba_branch and len(self.cnn_output_layers) != len(mamba_stage_block_counts):
            raise ValueError("CNN and Mamba stage counts must match for fusion.")

        # Determine output channels for fusion blocks
        self.fusion_block_output_channels = (
            advanced_fedab_fusion_output_channels
            if self.fusion_unit_type == "advanced_fedab"
            else simple_conv_fusion_output_channels
        )

        # --- Build Fusion Blocks ---
        accumulated_channels = 0
        for _ in range(num_fusion_stages):
            if self.branch_fusion_type == "concat":
                base_fusion_ch = self.adapter_channel_dim * (
                            (1 if use_cnn_branch else 0) + (1 if use_mamba_branch else 0))
            else:
                base_fusion_ch = self.adapter_channel_dim

            current_fusion_input_ch = base_fusion_ch + (
                accumulated_channels if self.fusion_aggregation_mode == "dense" else 0)

            if self.fusion_aggregation_mode == "dense":
                accumulated_channels += self.fusion_block_output_channels

            if self.fusion_unit_type == "advanced_fedab":
                self.scale_fusion_blocks.append(
                    AdvancedFusionUnit_FEDAB(
                        in_channels=current_fusion_input_ch,
                        output_channels=self.fusion_block_output_channels,
                        bottleneck_planes_factor=advanced_fedab_fusion_bottleneck_planes_factor,
                        num_blocks=advanced_fedab_fusion_num_blocks,
                        use_dwsc=advanced_fedab_fusion_use_dwsc,
                    )
                )
            else:
                self.scale_fusion_blocks.append(
                    nn.Conv2d(current_fusion_input_ch, self.fusion_block_output_channels, kernel_size=1)
                )


        # --- 5. Final Classifier Head ---
        self.final_pool = nn.AdaptiveAvgPool2d(1)

        # Determine classifier input dimension
        if self.fusion_aggregation_mode in ['concat', 'dense']:
            classifier_input_dim = num_fusion_stages * self.fusion_block_output_channels
        else:  # 'residual'
            classifier_input_dim = self.fusion_block_output_channels

        # Handle special case: no fusion stages
        if num_fusion_stages == 0:
            classifier_input_dim = self.adapter_channel_dim if use_cnn_branch or use_mamba_branch else 0

        if classifier_input_dim == 0:
            raise ValueError("Classifier input dimension is zero.")

        # Initialize MoE layer and final classifier
        self.fusion_moe_layer, self.final_classifier = None, None

        if use_fusion_moe:
            if not _MOE_BLOCKS_IMPORTED:
                raise ImportError("MoE is enabled but dependencies are not installed.")

            moe_params = fusion_moe_config.copy()
            moe_params['dim'] = classifier_input_dim
            if 'moe_inter_dim_ratio' in moe_params:
                moe_params['moe_inter_dim'] = int(classifier_input_dim * moe_params.pop('moe_inter_dim_ratio'))

            if MoEDSV2 is not None and DSV2ModelArgs is not None:
                self.fusion_moe_layer = MoEDSV2(args=DSV2ModelArgs(**moe_params))
                self.final_classifier = nn.Linear(classifier_input_dim, num_classes)

        # Fallback to MLP head if MoE is not used or fails
        if self.final_classifier is None:
            self.final_classifier = MLPHead(
                in_features=classifier_input_dim,
                out_features=num_classes,
                hidden_features=mlp_head_hidden_dims,
                dropout_rate=mlp_head_dropout
            )

        logger.info(
            f"Model initialized with branch_fusion='{self.branch_fusion_type}', "
            f"aggregation='{self.fusion_aggregation_mode}'. Classifier input dim: {classifier_input_dim}"
        )

    def _get_cnn_hook(self, name: str):
        """
        Returns a forward hook function to cache CNN features for a specific layer.
        """

        def hook(model, input_val, output_val):
            self._cnn_features_cache[name] = output_val

        return hook

    def print_trainable_parameters_summary(self):
        """
        Logs a summary of trainable parameters for each model component.
        """
        logger.info("-" * 60 + "\nModel Trainable Parameters Summary:")

        total_params = 0
        trainable_params = 0

        parts = {
            "CNN Backbone": self.cnn_backbone,
            "Mamba Branch": self.mamba_branch,
            "CNN FEDABs": self.cnn_fedab_enhancers,
            "Mamba FEDABs": self.mamba_fedab_enhancers,
            "CNN Adapters": self.cnn_adapter_layers,
            "Mamba Adapters": self.mamba_adapter_layers,
            "Scale Fusion Blocks": self.scale_fusion_blocks,
            "Classifier": self.final_classifier
        }

        if self.fusion_moe_layer:
            parts["Fusion MoE"] = self.fusion_moe_layer

        for name, part in parts.items():
            if part:
                part_total = sum(p.numel() for p in part.parameters())
                part_trainable = sum(p.numel() for p in part.parameters() if p.requires_grad)
                total_params += part_total
                trainable_params += part_trainable
                logger.info(f"  {name:<25}: Total={part_total / 1e6:.2f}M, "
                            f"Trainable={part_trainable / 1e6:.2f}M")

        logger.info(f"  {'Total Model':<25}: Total={total_params / 1e6:.2f}M, "
                    f"Trainable={trainable_params / 1e6:.2f}M\n" + "-" * 60)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = x.shape[0]
        num_fusion_layers = len(self.scale_fusion_blocks)

        # --- Step 1: Feature Extraction ---
        cnn_feats_stages: List[Optional[torch.Tensor]] = [None] * num_fusion_layers
        mamba_feats_stages: List[Optional[torch.Tensor]] = [None] * num_fusion_layers
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # CNN Branch Feature Extraction
        if self.use_cnn_branch and self.cnn_backbone and self.cnn_adapter_layers:
            self._cnn_features_cache.clear()
            _ = self.cnn_backbone(x)
            raw_cnn_feats = [self._cnn_features_cache.get(name) for name in self.cnn_output_layers]

            for i in range(num_fusion_layers):
                if i < len(raw_cnn_feats) and raw_cnn_feats[i] is not None:
                    feat = raw_cnn_feats[i].to(dtype=x.dtype)
                    adapted_feat = self.cnn_adapter_layers[i](feat)

                    if self.use_fedab_on_cnn and self.cnn_fedab_enhancers:
                        enhanced_feat = self.cnn_fedab_enhancers[i](adapted_feat)
                    else:
                        enhanced_feat = adapted_feat

                    # Determine target spatial size
                    if self.use_mamba_branch:
                        h_target, w_target = self.mamba_h_patch, self.mamba_w_patch
                    else:
                        h_target, w_target = enhanced_feat.shape[2], enhanced_feat.shape[3]

                    # Resize features to match target size
                    cnn_feats_stages[i] = F.interpolate(
                        enhanced_feat,
                        size=(h_target, w_target),
                        mode='bilinear',
                        align_corners=False
                    )
        # Mamba Branch Feature Extraction
        if self.use_mamba_branch and self.mamba_branch and self.mamba_adapter_layers:
            _, mamba_aux, raw_mamba_feats = self.mamba_branch(x, return_features=True)
            if mamba_aux is not None:
                aux_loss += mamba_aux

            for i in range(num_fusion_layers):
                if i >= len(raw_mamba_feats):
                    continue

                feat_seq = raw_mamba_feats[i].to(dtype=x.dtype)
                num_patches = self.mamba_h_patch * self.mamba_w_patch

                # Select patch tokens based on cls_position
                if self.mamba_branch.cls_position == 'head':
                    patch_tokens = feat_seq[:, self.mamba_branch.num_extra_tokens:]
                else:
                    patch_tokens = feat_seq[:, :num_patches]

                # Adapter and reshape to feature map
                adapted_feat = self.mamba_adapter_layers[i](patch_tokens)
                mamba_feat_map = adapted_feat.transpose(1, 2).reshape(
                    B,
                    self.adapter_channel_dim,
                    self.mamba_h_patch,
                    self.mamba_w_patch
                )

                # Optional FEDAB enhancement
                if self.use_fedab_on_mamba and self.mamba_fedab_enhancers:
                    mamba_feat_map = self.mamba_fedab_enhancers[i](mamba_feat_map)

                mamba_feats_stages[i] = mamba_feat_map
        # --- Step 2 & 3: Flexible Fusion & Aggregation ---
        fused_outputs_maps = []

        for i in range(num_fusion_layers):
            cnn_f = cnn_feats_stages[i]
            mamba_f = mamba_feats_stages[i]
            fusion_block = self.scale_fusion_blocks[i]

            # --- Branch Fusion ---
            branch_fused_map: Optional[torch.Tensor] = None
            if self.branch_fusion_type == 'add':
                branch_fused_map = cnn_f + mamba_f if cnn_f is not None and mamba_f is not None else (cnn_f or mamba_f)
            elif self.branch_fusion_type == 'multiply':
                branch_fused_map = cnn_f * mamba_f if cnn_f is not None and mamba_f is not None else (cnn_f or mamba_f)
            else:  # 'concat'
                feats_to_cat = [f for f in [cnn_f, mamba_f] if f is not None]
                if feats_to_cat:
                    branch_fused_map = torch.cat(feats_to_cat, dim=1) if len(feats_to_cat) > 1 else feats_to_cat[0]

            if branch_fused_map is None:
                continue

            # --- Fusion Aggregation ---
            fusion_input = branch_fused_map
            if self.fusion_aggregation_mode == 'dense' and fused_outputs_maps:
                target_size = fusion_input.shape[2:]
                prev_maps_upsampled = [
                    F.interpolate(prev_map, size=target_size, mode='bilinear', align_corners=False)
                    for prev_map in fused_outputs_maps
                ]
                fusion_input = torch.cat([fusion_input] + prev_maps_upsampled, dim=1)

            # --- Apply Fusion Block ---
            fused_map = fusion_block(fusion_input)
            if fused_map is not None:
                fused_outputs_maps.append(fused_map)

        if not fused_outputs_maps:
            if not self.use_cnn_branch and not self.use_mamba_branch:
                raise RuntimeError("Both branches are disabled, no features to process.")
            logger.warning(
                "No features were fused. This might happen in single-branch mode without fusion stages. "
                "Returning zero logits."
            )
            return torch.zeros(B, self.num_classes, device=x.device, dtype=x.dtype), aux_loss

        # --- Aggregate Fused Features ---
        if self.fusion_aggregation_mode in ['concat', 'dense']:
            final_vec = torch.cat(
                [self.final_pool(fm).flatten(1) for fm in fused_outputs_maps], dim=1
            )
        else:  # 'residual'
            aggregated_map = fused_outputs_maps[-1]
            for i in range(len(fused_outputs_maps) - 2, -1, -1):
                upsampled_deep_map = F.interpolate(
                    aggregated_map, size=fused_outputs_maps[i].shape[2:], mode='bilinear', align_corners=False
                )
                aggregated_map = fused_outputs_maps[i] + upsampled_deep_map
            final_vec = self.final_pool(aggregated_map).flatten(1)

        # --- Step 4: Classifier Head ---
        if not self.training:
            if self.use_fusion_moe and self.fusion_moe_layer:
                moe_out, _ = self.fusion_moe_layer(final_vec.unsqueeze(1))
                logits = self.final_classifier(moe_out.squeeze(1))  # type: ignore
            else:
                logits = self.final_classifier(final_vec)  # type: ignore
            return logits

        if self.use_fusion_moe and self.fusion_moe_layer:
            moe_out, moe_aux = self.fusion_moe_layer(final_vec.unsqueeze(1))
            logits = self.final_classifier(moe_out.squeeze(1))  # type: ignore
            aux_loss += moe_aux
        else:
            logits = self.final_classifier(final_vec)  # type: ignore

        return logits, aux_loss