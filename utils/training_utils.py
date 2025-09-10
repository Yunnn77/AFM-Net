# utils/training_utils.py
import torch
import torch.optim as optim
import random
import numpy as np
import os
import glob
import shutil
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import torch.nn as nn

logger = logging.getLogger("train_logger")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)


def setup_file_logger(log_dir_for_run: str, timestamp: str):
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    log_filename = f"{timestamp}.log"
    log_filepath = os.path.join(log_dir_for_run, log_filename)

    file_handler = logging.FileHandler(log_filepath, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"File logger setup complete. Logging to: {log_filepath}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def manage_checkpoints(checkpoint_run_dir_checkpoints_subdir: str, prefix="checkpoint_epoch_", max_keep=5):
    glob_pattern = os.path.join(checkpoint_run_dir_checkpoints_subdir, f"{prefix}*.pth.tar")
    checkpoint_files = sorted(glob.glob(glob_pattern), key=os.path.getmtime, reverse=True)
    if len(checkpoint_files) > max_keep:
        files_to_delete = checkpoint_files[max_keep:]
        for f_del in files_to_delete:
            try:
                os.remove(f_del)
                logger.info(f"Deleted old checkpoint: {f_del}")
            except OSError as e:
                logger.warning(f"Error deleting old checkpoint {f_del}: {e}")


def save_checkpoint(state: Dict[str, Any], is_best: bool, run_dir: str,
                    epoch_filename_prefix="checkpoint_epoch_",
                    best_filename="model_best.pth.tar",
                    max_recent_checkpoints=5):
    ckpt_subdir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_subdir, exist_ok=True)

    current_epoch = state.get('epoch', 0)
    epoch_filename = f"{epoch_filename_prefix}{current_epoch:04d}.pth.tar"
    epoch_filepath = os.path.join(ckpt_subdir, epoch_filename)
    torch.save(state, epoch_filepath)
    logger.info(f"Saved current epoch checkpoint to {epoch_filepath}")

    if max_recent_checkpoints is not None and max_recent_checkpoints > 0:
        manage_checkpoints(ckpt_subdir, prefix=epoch_filename_prefix, max_keep=max_recent_checkpoints)

    if is_best:
        best_filepath = os.path.join(ckpt_subdir, best_filename)
        torch.save(state, best_filepath)
        logger.info(f" => Saved new best model to {best_filepath}")

    last_ckpt_path_in_subdir = os.path.join(ckpt_subdir, "last_checkpoint.pth.tar")
    target_for_symlink = os.path.basename(epoch_filepath)
    try:
        if os.path.exists(last_ckpt_path_in_subdir):
            if os.path.islink(last_ckpt_path_in_subdir):
                os.unlink(last_ckpt_path_in_subdir)
            else:
                os.remove(last_ckpt_path_in_subdir)

        if sys.platform == "win32":
            shutil.copyfile(epoch_filepath, last_ckpt_path_in_subdir)
        else:
            os.symlink(target_for_symlink, last_ckpt_path_in_subdir)
        logger.info(f"Updated last_checkpoint to point to {epoch_filename} in {ckpt_subdir}")
    except Exception as e:
        logger.warning(f"Could not create/update last_checkpoint in {ckpt_subdir}: {e}")
        logger.info(f"Attempting to copy file for last_checkpoint as fallback.")
        try:
            shutil.copyfile(epoch_filepath, last_ckpt_path_in_subdir)
            logger.info(f"Fallback: Copied {epoch_filename} to last_checkpoint.pth.tar")
        except Exception as e_copy:
            logger.error(f"Fallback copy for last_checkpoint also failed: {e_copy}")


def load_checkpoint(checkpoint_path: str, model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    device: str = 'cpu') -> Tuple[int, Dict[str, float]]:
    start_epoch = 0
    best_metrics = {
        'best_val_f1_score': 0.0, 'best_val_precision_at_best_f1': 0.0,
        'best_val_recall_at_best_f1': 0.0, 'best_val_accuracy_at_best_f1': 0.0
    }
    if os.path.isfile(checkpoint_path):
        logger.info(f"=> loading checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.error(f"torch.load failed (weights_only=False), trying with weights_only=True: {e}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
                logger.warning("Successfully loaded with weights_only=True, optimizer/scheduler states may be missing.")
            except Exception as e2:
                logger.error(f"Loading failed even with weights_only=True: {e2}")
                return start_epoch, best_metrics

        start_epoch = checkpoint.get('epoch', 0)
        best_metrics['best_val_f1_score'] = checkpoint.get('best_val_f1_score', checkpoint.get('best_metric', 0.0))
        best_metrics['best_val_precision_at_best_f1'] = checkpoint.get('best_val_precision_at_best_f1', 0.0)
        best_metrics['best_val_recall_at_best_f1'] = checkpoint.get('best_val_recall_at_best_f1', 0.0)
        best_metrics['best_val_accuracy_at_best_f1'] = checkpoint.get('best_val_accuracy_at_best_f1', 0.0)

        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state', checkpoint))
        if state_dict:
            new_state_dict = {}
            [(name := k[7:] if k.startswith('module.') else k, new_state_dict.update({name: v})) for k, v in
             state_dict.items()]
            try:
                model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e_strict:
                logger.warning(f"Strict=True load failed: {e_strict}. Trying strict=False...")
                try:
                    model.load_state_dict(new_state_dict, strict=False)
                except Exception as e_non_strict:
                    logger.error(f"Strict=False load also failed: {e_non_strict}")
        else:
            logger.warning("No state_dict or model_state found in checkpoint.")

        if optimizer and 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                logger.warning("Failed to load optimizer state. Possibly due to weights_only=True or mismatch.")
        if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                logger.warning("Failed to load scheduler state. Possibly due to weights_only=True or mismatch.")
        logger.info(
            f"=> loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_f1 {best_metrics['best_val_f1_score']:.4f})")
    else:
        logger.info(f"=> no checkpoint found at '{checkpoint_path}'")
    return start_epoch, best_metrics