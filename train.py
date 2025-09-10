# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import os
import sys
import time
import shutil
import glob
import argparse
import importlib
from datetime import datetime
import numpy as np
from tqdm import tqdm
import logging
import datasets
from utils.training_utils import logger, setup_file_logger, set_seed, AverageMeter, save_checkpoint, load_checkpoint
from utils.visualization import plot_per_class_accuracy, generate_cam_visualizations
from utils.metrics import calculate_metrics, calculate_per_class_accuracy

# Check for Mamba availability for logging purposes
try:
    from base_mamba_vision_block_arch import MAMBA_IMPORTED

    if not MAMBA_IMPORTED:
        logger.warning("Mamba implementation (mamba_ssm) not found. Mamba branch will use a placeholder Linear layer.")
except ImportError:
    MAMBA_IMPORTED = False
    logger.warning("base_mamba_vision_block_arch.py not found. Mamba branch will use a placeholder Linear layer.")


# --- Main Training and Evaluation Functions ---

def train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg, device, dtype, epoch_idx):
    """
    Performs one full epoch of training.
    """
    model.train()

    losses, main_losses, aux_losses = AverageMeter(), AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch_idx}/{cfg['training']['epochs']}",
        leave=True,
        file=sys.stdout,
        dynamic_ncols=True
    )

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        data_time.update(time.time() - end)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(scaler is not None)):
            # The model is expected to return (logits, aux_loss) in training mode
            logits, aux_loss = model(inputs)
            main_loss = criterion(logits, targets)
            total_loss = main_loss + cfg['training']['aux_loss_weight'] * aux_loss

        if scaler:
            scaler.scale(total_loss).backward()
            if cfg['training']['clip_grad_norm']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if cfg['training']['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['clip_grad_norm'])
            optimizer.step()

        # Record metrics
        losses.update(total_loss.item(), inputs.size(0))
        main_losses.update(main_loss.item(), inputs.size(0))
        aux_losses.update(aux_loss.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'main': f"{main_losses.avg:.4f}",
            'aux': f"{aux_losses.avg:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })

    return losses.avg, main_losses.avg, aux_losses.avg


def evaluate_model(model, data_loader, criterion, cfg, device, dtype, eval_name="Eval", is_final_test_run=False,
                   current_epoch=0, save_plots=False, run_dir=None):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    losses = AverageMeter()
    all_preds, all_targets = [], []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Running {eval_name}", leave=False, file=sys.stdout, dynamic_ncols=True)
        for inputs, targets in progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                # The model returns only logits in eval mode
                logits = model(inputs)
                if criterion:
                    loss = criterion(logits, targets)
                    losses.update(loss.item(), inputs.size(0))

            _, preds = torch.max(logits.data, 1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            if criterion:
                progress_bar.set_postfix({f'{eval_name}_loss': f"{losses.avg:.4f}"})

    if not all_targets:
        logger.warning(f"{eval_name} dataloader was empty. Skipping metrics calculation.")
        return 0.0, 0.0, {}

    all_targets_np = torch.cat(all_targets).numpy()
    all_preds_np = torch.cat(all_preds).numpy()

    avg_mode = cfg['training'].get('metrics_average_mode', 'weighted')
    metrics = calculate_metrics(all_targets_np, all_preds_np, average=avg_mode)

    loss_str = f"Avg Loss: {losses.avg:.4f}, " if criterion else ""
    logger.info(f"{eval_name} Results: {loss_str}Overall Acc: {metrics['accuracy'] * 100:.2f}%")
    logger.info(
        f"  P ({avg_mode}): {metrics['precision']:.4f}, R ({avg_mode}): {metrics['recall']:.4f}, F1 ({avg_mode}): {metrics['f1_score']:.4f}")

    if save_plots and run_dir:
        class_names = cfg['data'].get('class_names', [f"Class_{i}" for i in range(cfg['data']['num_classes'])])
        per_class_acc = calculate_per_class_accuracy(all_targets_np, all_preds_np, cfg['data']['num_classes'])
        viz_dir = os.path.join(run_dir, "visualizations")
        filename_prefix = f"{eval_name.lower()}_{'final' if is_final_test_run else f'epoch_{current_epoch}'}"
        plot_per_class_accuracy(per_class_acc, class_names, viz_dir, filename_prefix, metrics['accuracy'])

    return losses.avg, metrics['accuracy'], metrics


def main(args):
    """
    Main function to run the training and evaluation pipeline.
    """
    # --- 1. Load Configuration ---
    try:
        config_module = importlib.import_module(f"configs.{args.dataset.lower()}_scratch_config")
        config_getter = getattr(config_module, f"get_{args.dataset.lower()}_from_scratch_config")
        cfg = config_getter()
    except (ImportError, AttributeError) as e:
        logger.critical(f"Failed to load configuration for dataset '{args.dataset}'. Please ensure "
                        f"'configs/{args.dataset.lower()}_scratch_config.py' and the corresponding "
                        f"'get_{args.dataset.lower()}_from_scratch_config' function exist. Error: {e}")
        return

    # --- 2. Override Config with Command-Line Arguments ---
    if args.lr: cfg['training']['optimizer_params']['lr'] = args.lr
    if args.batch_size: cfg['data']['batch_size'] = args.batch_size
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
        if 'scheduler_params' in cfg['training'] and 'T_max' in cfg['training']['scheduler_params']:
            cfg['training']['scheduler_params']['T_max'] = args.epochs

    # --- 3. Setup Environment and Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg['training']['checkpoint_dir'], cfg['training']['experiment_name'], timestamp)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "visualizations", "cam"), exist_ok=True)
    setup_file_logger(run_dir, timestamp)

    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: {cfg['training']['experiment_name']}")
    logger.info(f"RUN DIRECTORY: {run_dir}")
    logger.info(f"COMMAND-LINE ARGS: {args}")
    logger.info("=" * 60)

    try:
        shutil.copy2(f"configs/{args.dataset.lower()}_scratch_config.py", os.path.join(run_dir, "config_snapshot.py"))
    except Exception as e:
        logger.warning(f"Could not save config snapshot: {e}")

    # --- 4. Setup Device, Seed, and DType ---
    train_cfg = cfg['training']
    set_seed(train_cfg['seed'])
    device = torch.device(train_cfg['device'])
    dtype_str = train_cfg.get('dtype', 'float32')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}.get(dtype_str,
                                                                                                 torch.float32)
    use_amp = (dtype != torch.float32) and (device.type == 'cuda')
    if use_amp and not torch.cuda.is_available():
        logger.warning("AMP requested but CUDA is not available. Disabling AMP.")
        use_amp = False

    logger.info(f"Using device: {device} | Data type: {dtype_str} | AMP enabled: {use_amp}")

    # --- 5. Load Data ---
    data_cfg = cfg['data']
    try:
        dataloaders_getter = getattr(datasets, f"get_{data_cfg['dataset_type'].lower()}_dataloaders")
        train_loader, val_loader, test_loader = dataloaders_getter(cfg)
        logger.info(
            f"Dataloaders for '{data_cfg['dataset_type']}' loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    except (AttributeError, Exception) as e:
        logger.critical(f"Failed to load dataloaders for dataset type '{data_cfg['dataset_type']}': {e}")
        return

    # --- 6. Create Model ---
    model_cfg = cfg['model']
    try:
        model_module = importlib.import_module('models.advanced_fusion_model')
        ModelClass = getattr(model_module, model_cfg['name'])
        # The configuration is now perfectly aligned, just unpack the params.
        model = ModelClass(**model_cfg['params'])
        model.to(device=device, dtype=dtype)
        logger.info(f"Model '{model_cfg['name']}' created successfully.")
        model.print_trainable_parameters_summary()  # Print summary after creation
    except Exception as e:
        logger.critical(f"Failed to create model '{model_cfg['name']}': {e}", exc_info=True)
        return

    # --- 7. Setup Optimizer, Scheduler, and Criterion ---
    criterion = nn.CrossEntropyLoss(**train_cfg.get('criterion_params', {})).to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    opt_params = train_cfg['optimizer_params']
    optimizer = optim.AdamW(model.parameters(), **opt_params)
    logger.info(f"Optimizer: AdamW with params: {opt_params}")

    scheduler = None
    if 'scheduler' in train_cfg and train_cfg['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **train_cfg['scheduler_params'])
        logger.info(f"Scheduler: CosineAnnealingLR with params: {train_cfg['scheduler_params']}")

    # --- 8. Resume from Checkpoint (if specified) ---
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        try:
            start_epoch, best_metrics = load_checkpoint(args.resume, model, optimizer, scheduler, device)
            best_val_f1 = best_metrics.get('best_val_f1_score', 0.0)
            logger.info(f"Resumed from checkpoint: {args.resume} at epoch {start_epoch}. Best F1: {best_val_f1:.4f}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {args.resume}: {e}. Starting from scratch.")

    # --- 9. Main Training Loop ---
    logger.info(f"Starting training from epoch {start_epoch + 1} to {train_cfg['epochs']}...")
    for epoch in range(start_epoch, train_cfg['epochs']):
        epoch_display = epoch + 1
        epoch_start_time = time.time()

        train_loss, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg, device, dtype,
                                           epoch_display)

        if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()

        logger.info(
            f"Epoch {epoch_display}/{train_cfg['epochs']} | Train Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Validation and Checkpointing ---
        if (epoch_display % train_cfg['eval_freq'] == 0) or (epoch_display == train_cfg['epochs']):
            val_loss, val_acc, val_metrics = evaluate_model(model, val_loader, criterion, cfg, device, dtype,
                                                            "Validation", current_epoch=epoch_display)

            if scheduler and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            is_best = val_metrics.get('f1_score', 0.0) > best_val_f1
            if is_best:
                best_val_f1 = val_metrics['f1_score']
                logger.info(
                    f"*** NEW BEST MODEL (Epoch {epoch_display}) | Val F1: {best_val_f1:.4f}, Val Acc: {val_acc * 100:.2f}% ***")
                # Save plot for the new best model
                evaluate_model(model, val_loader, None, cfg, device, dtype, "Best_Validation",
                               current_epoch=epoch_display, save_plots=True, run_dir=run_dir)

            checkpoint_data = {
                'epoch': epoch_display,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'best_val_f1_score': best_val_f1,
                'config_snapshot': cfg
            }
            save_checkpoint(checkpoint_data, is_best, run_dir,
                            max_recent_checkpoints=train_cfg['max_recent_checkpoints_to_keep'])

        logger.info(
            f"Epoch {epoch_display} finished in {time.time() - epoch_start_time:.2f}s. Best F1 so far: {best_val_f1:.4f}")
        logger.info("-" * 60)

    # --- 10. Final Evaluation and Visualization ---
    logger.info("=" * 60)
    logger.info("TRAINING FINISHED. Performing final evaluation on the best model.")
    logger.info("=" * 60)

    best_model_path = os.path.join(run_dir, "checkpoints", "model_best.pth.tar")
    if os.path.exists(best_model_path):
        # Create a fresh model instance for final eval to ensure no state leakage
        FinalModelClass = getattr(importlib.import_module('models.advanced_fusion_model'), model_cfg['name'])
        final_model = FinalModelClass(**model_cfg['params'])
        final_model.to(device=device, dtype=dtype)

        _, best_metrics_loaded = load_checkpoint(best_model_path, final_model, None, None, device)
        logger.info(
            f"Best model loaded. Recorded best val F1 was: {best_metrics_loaded.get('best_val_f1_score', 0.0):.4f}")

        eval_loader = test_loader if test_loader and len(test_loader) > 0 else val_loader
        eval_set_name = "Test Set" if test_loader and len(test_loader) > 0 else "Validation Set"

        logger.info(f"Evaluating best model on the {eval_set_name}...")
        _, _, final_metrics = evaluate_model(final_model, eval_loader, None, cfg, device, dtype, "Final Eval",
                                             is_final_test_run=True, current_epoch=train_cfg['epochs'], save_plots=True,
                                             run_dir=run_dir)
        logger.info(
            f"FINAL PERFORMANCE on {eval_set_name}: Acc={final_metrics['accuracy'] * 100:.2f}%, F1={final_metrics['f1_score']:.4f}, P={final_metrics['precision']:.4f}, R={final_metrics['recall']:.4f}")

        if args.cam_samples > 0:
            logger.info(f"Generating {args.cam_samples} CAM visualizations...")
            generate_cam_visualizations(
                model=final_model, data_loader=eval_loader, device=device, cfg=cfg,
                run_dir=run_dir, num_samples=args.cam_samples, cam_algorithm_name=args.cam_alg,
                target_layer_type=args.cam_target_layer, eval_name="Final"
            )
    else:
        logger.warning("Could not find best model checkpoint 'model_best.pth.tar' for final evaluation.")

    logger.info(f"Experiment finished. All artifacts are in: {run_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Advanced Fusion Model From Scratch")
    parser.add_argument('--dataset', type=str, required=True, choices=['aid', 'nwpu', 'ucmerced'],
                        help="Name of the dataset to use.")
    parser.add_argument('--lr', type=float, default=None, help="Override learning rate from config.")
    parser.add_argument('--batch_size', type=int, default=None, help="Override batch size from config.")
    parser.add_argument('--epochs', type=int, default=None, help="Override number of epochs from config.")
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument('--cam_samples', type=int, default=10,
                        help="Number of samples for CAM visualization after training.")
    parser.add_argument('--cam_alg', type=str, default="GradCAM", choices=["GradCAM", "GradCAMPlusPlus", "ScoreCAM"],
                        help="CAM algorithm to use.")
    parser.add_argument('--cam_target_layer', type=str, default="cnn_layer4",
                        help="Target layer for CAM. E.g., 'cnn_layer4', 'mamba_block_11'. Needs to be adapted to your model implementation.")

    cmd_args = parser.parse_args()
    main(cmd_args)