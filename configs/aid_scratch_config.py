import torch

def get_aid_from_scratch_config():
    config = {}

    # Dataset & Dataloader
    config['data'] = {
        'dataset_type': 'aid',
        'aid_data_root': 'path/to/AID',
        'num_classes': 30,
        'img_size': 224,
        'batch_size': 128,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_workers': 4,
        'val_split_ratio': 0.5,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
    }

    # Model configuration
    config['model'] = {
        'name': 'AdvancedFusionModel',
        'params': {
            'num_classes': config['data']['num_classes'],
            'branch_fusion_type': 'concat',
            'fusion_unit_type': 'advanced_fedab',
            'use_cnn_branch': True,
            'use_mamba_branch': True,
            'resnet_variant': "resnet50",
            'resnet_pretrained': False,
            'cnn_output_layers': ['layer2', 'layer3', 'layer4'],
            'mamba_stage_block_counts': [3, 3, 4],
            'use_fedab_on_cnn': True,
            'use_fedab_on_mamba': True,
            'mlp_head_hidden_dims': [512],
            'mlp_head_dropout': 0.3,
            'mamba_config': {
                'img_size': config['data']['img_size'],
                'patch_size': 16,
                'stride': 16,
                'in_channels': 3,
                'embed_dim': 192,
                'd_state': 16,
                'd_conv': 4,
                'expand': 2,
                'path_type': 'forward_shuffle_reverse_gate',
                'cls_position': 'head',
                'if_abs_pos_embed': True,
                'if_rope': True,
                'rms_norm': True,
                'drop_rate': 0.1,
                'drop_path_rate': 0.1,
            }
        }
    }

    # Training
    config['training'] = {
        'epochs': 100,
        'optimizer': 'AdamW',
        'optimizer_params': { 'lr': 5e-4, 'weight_decay': 0.05 },
        'scheduler': 'CosineAnnealingLR',
        'scheduler_params': { 'T_max': 100, 'eta_min': 1e-6 },
        'criterion': 'CrossEntropyLoss',
        'clip_grad_norm': 1.0,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'float32',
        'seed': 42,
        'eval_freq': 5,
        'print_freq': 50,
        'checkpoint_dir': './checkpoints',
        'experiment_name': 'aid_experiment',
        'max_recent_checkpoints_to_keep': 5,
    }

    mamba_cfg_ref = config['model']['params']['mamba_config']
    mamba_cfg_ref['device'] = config['training']['device']
    mamba_cfg_ref['dtype'] = config['training']['dtype']

    return config