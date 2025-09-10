# utils/visualization.py (Final Publication Quality Version)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from typing import Optional, List, Dict, Any
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from contextlib import contextmanager

# --- Logger Setup ---
try:
    from .training_utils import logger
except (ImportError, ModuleNotFoundError):
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Matplotlib Backend Setup ---
plt.switch_backend('agg')

FONT_CONFIG = {
    'title':    {'fontsize': 25, 'fontweight': 'bold'},
    'suptitle': {'fontsize': 25, 'fontweight': 'bold'},
    'axis':     {'fontsize': 20},
    'tick':     {'fontsize': 20},
    'legend':   {'fontsize': 20},
    'legend_title': {'fontsize': 18},
    'bar_label':    {'fontsize': 18, 'fontweight': 'bold'}
}

@contextmanager
def temp_font_settings(font_config_dict):
    original_rc = plt.rcParams.copy()
    flat_config = {
        'axes.titlesize': font_config_dict['title']['fontsize'],
        'axes.labelsize': font_config_dict['axis']['fontsize'],
        'xtick.labelsize': font_config_dict['tick']['fontsize'],
        'ytick.labelsize': font_config_dict['tick']['fontsize'],
        'legend.fontsize': font_config_dict['legend']['fontsize'],
        'legend.title_fontsize': font_config_dict['legend_title']['fontsize'],
        'figure.titlesize': font_config_dict['suptitle']['fontsize'],
        'font.family': 'sans-serif',
    }
    try:
        plt.rcParams.update(flat_config)
        yield
    finally:
        plt.rcParams.update(original_rc)

def plot_per_class_accuracy(class_accuracies, class_names, save_dir, prefix, overall_accuracy):
    with temp_font_settings(FONT_CONFIG):
        try:
            plt.figure(figsize=(max(12, len(class_names) * 0.4), 8))
            plt.bar(np.arange(len(class_names)), class_accuracies, align='center', alpha=0.75)
            plt.xticks(np.arange(len(class_names)), class_names, rotation=65, ha="right")
            plt.ylabel('Per-Class Accuracy (Recall)')
            plt.title(f'Per-Class Accuracy\nOverall: {overall_accuracy*100:.2f}% ({prefix.replace("_", " ").title()})')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"{prefix}_per_class_acc.png")
            plt.savefig(filename, dpi=200)
            plt.close()
            logger.info(f"Per-class accuracy plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to plot per-class accuracy: {e}", exc_info=True)

def plot_confusion_matrix(y_true, y_pred, class_names, run_dir, prefix="cm"):
    with temp_font_settings(FONT_CONFIG):
        try:
            cm = confusion_matrix(y_true, y_pred)
            figsize = max(10, len(class_names) * 0.4)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='d')
            ax.set_title(f'Confusion Matrix ({prefix.replace("_", " ").title()})')
            plt.tight_layout()
            save_path = os.path.join(run_dir, "visualizations")
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"{prefix}_confusion_matrix.png")
            plt.savefig(filename, dpi=200)
            plt.close(fig)
            logger.info(f"Confusion matrix saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}", exc_info=True)

# --- Feature Map & CAM Visualization ---
feature_maps_storage = {}


def plot_tsne_comparison(features_before, features_after, labels, class_names, save_dir, prefix="tsne_comparison"):
    FONT_PROPS = {
        'suptitle': {'fontsize': 40, 'fontweight': 'bold', 'family': 'sans-serif'},
        'title': {'fontsize': 35, 'pad': 20, 'family': 'sans-serif'},
        'axis': {'fontsize': 28, 'family': 'sans-serif'},
        'tick': {'labelsize': 18},
        'legend': {'fontsize': 16},
        'legend_title': {'size': 18}
    }

    logger.info("Calculating t-SNE... This may take a while.")
    tsne_before = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate='auto',
                       init='pca').fit_transform(features_before)
    tsne_after = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, learning_rate='auto',
                      init='pca').fit_transform(features_after)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
    fig.suptitle('t-SNE Visualization: Before vs. After Training', **FONT_PROPS['suptitle'])

    axes[0].scatter(tsne_before[:, 0], tsne_before[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", len(class_names)), s=15)
    axes[0].set_title('Before Training (Randomly Initialized)', **FONT_PROPS['title'])
    axes[0].set_xlabel('t-SNE Dimension 1', **FONT_PROPS['axis'])
    axes[0].set_ylabel('t-SNE Dimension 2', **FONT_PROPS['axis'])
    axes[0].tick_params(axis='both', which='major', **FONT_PROPS['tick'])
    axes[0].grid(True, linestyle='--', alpha=0.5)

    scatter2 = axes[1].scatter(
        tsne_after[:, 0],
        tsne_after[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap("jet", len(class_names)),
        s=15
    )
    axes[1].set_title('After Training (Converged)', **FONT_PROPS['title'])
    axes[1].set_xlabel('t-SNE Dimension 1', **FONT_PROPS['axis'])
    axes[1].tick_params(axis='both', which='major', **FONT_PROPS['tick'])
    axes[1].grid(True, linestyle='--', alpha=0.5)

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                                 markerfacecolor=scatter2.cmap(scatter2.norm(i)), markersize=12) for i in
                      range(len(class_names))]
    fig.legend(
        handles=legend_handles,
        title="Classes",
        bbox_to_anchor=(1.0, 0.9),
        loc='upper left',
        prop={'size': FONT_PROPS['legend']['fontsize'], 'family': 'sans-serif'},
        title_fontproperties={'size': FONT_PROPS['legend_title']['size'], 'weight': 'bold', 'family': 'sans-serif'}
    )

    plt.tight_layout(rect=[0, 0, 0.9, 0.9])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_tsne.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info(f"t-SNE comparison plot saved to {save_path}")


def get_feature_map_hook(name: str):
    def hook(model, input, output):
        feature_maps_storage[name] = output.detach()

    return hook


def get_target_layers_by_path(model: nn.Module, layer_path_str: str) -> List[nn.Module]:
    layers = []
    try:
        current_module = model
        for part in layer_path_str.split('.'):
            if part.endswith(']'):
                name, index = part.strip(']').split('[')
                current_module = getattr(current_module, name)[int(index)]
            else:
                current_module = getattr(current_module, part)
        layers.append(current_module)
        logger.info(f"Successfully located target layer: '{layer_path_str}'")
    except (AttributeError, IndexError) as e:
        logger.warning(f"Could not find target layer at path '{layer_path_str}': {e}")
    return layers

def plot_feature_maps(
    feature_maps_dict: Dict[str, torch.Tensor],
    ordered_layer_names: List[str],
    run_dir: str,
    prefix: str = "feature_evolution",
    num_channels_to_show: int = 4
):
    try:
        ROW_TITLE_FONT = {'fontsize': 28, 'fontweight': 'bold'}
        COL_TITLE_FONT = {'fontsize': 26}
        SUPTITLE_FONT = {'fontsize': 36, 'fontweight': 'bold'}
        CBAR_TICK_FONT = {'labelsize': 18}

        num_layers = len(ordered_layer_names)
        if num_layers == 0:
            logger.warning("Layer name list is empty. Skipping plot.")
            return

        cols = num_channels_to_show
        rows = num_layers
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)

        for i, layer_name in enumerate(ordered_layer_names):
            if layer_name not in feature_maps_dict:
                for j in range(cols):
                    axes[i, j].axis('off')
                continue

            fm_tensor = feature_maps_dict[layer_name]
            fm = fm_tensor[0].cpu().float()
            channel_indices = list(range(min(fm.shape[0], cols)))
            axes[i, 0].set_ylabel(layer_name, rotation=90, labelpad=40, **ROW_TITLE_FONT)

            for j, channel_idx in enumerate(channel_indices):
                ax = axes[i, j]
                feature_map = fm[channel_idx]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_xticks([])
                ax.set_yticks([])

                if i == 0:
                    ax.set_title(f'Channel {channel_idx}', **COL_TITLE_FONT)

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(**CBAR_TICK_FONT)

        fig.suptitle('Feature Map Evolution', **SUPTITLE_FONT)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.05, wspace=0.3, hspace=0.3)

        save_path = os.path.join(run_dir, "visualizations", "feature_evolution")
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"{prefix}_evolution.png")
        plt.savefig(filename, dpi=200)
        plt.close(fig)
        logger.info(f"Feature map evolution plot saved to {filename}")

    except Exception as e:
        logger.error(f"Failed to plot feature map evolution for '{prefix}': {e}", exc_info=True)


def generate_cam_visualizations(
        model: nn.Module, data_loader: DataLoader, device: torch.device,
        cfg: dict, run_dir: str,
        num_samples: int = 10, cam_algorithm_name: str = "GradCAM",
        target_layer_str: str = "scale_fusion_blocks[-1]", **kwargs
):
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, LayerCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.error("`pytorch-grad-cam` is not installed. Skipping CAM.")
        return

    model.eval()
    target_layers = get_target_layers_by_path(model, target_layer_str)
    if not target_layers:
        logger.warning(f"No valid CAM layers for '{target_layer_str}'. Skipping.")
        return

    CAM_ALG_MAP = {"GradCAM": GradCAM, "GradCAMPlusPlus": GradCAMPlusPlus, "ScoreCAM": ScoreCAM, "LayerCAM": LayerCAM}
    cam_constructor = CAM_ALG_MAP.get(cam_algorithm_name, GradCAM)
    cam_save_dir = os.path.join(run_dir, "visualizations", "cam",
                                target_layer_str.replace('.', '_').replace('[', '').replace(']', ''))
    os.makedirs(cam_save_dir, exist_ok=True)
    logger.info(f"Generating up to {num_samples} CAMs using {cam_algorithm_name} on '{target_layer_str}'...")

    samples_done = 0
    class_names = cfg['data'].get('class_names', [f"C_{i}" for i in range(cfg['data']['num_classes'])])

    for inputs, targets in data_loader:
        for i in range(inputs.size(0)):
            if samples_done >= num_samples:
                break

            input_tensor = inputs[i:i + 1].to(device)
            target_label = targets[i].item()
            true_class = class_names[target_label]

            img_for_overlay = input_tensor.squeeze(0).cpu().detach()
            mean = torch.tensor(cfg['data']['normalize_mean']).view(3, 1, 1)
            std = torch.tensor(cfg['data']['normalize_std']).view(3, 1, 1)
            img_for_overlay = np.float32(torch.clamp(img_for_overlay * std + mean, 0, 1).permute(1, 2, 0))

            with torch.no_grad():
                pred_label = model(input_tensor).argmax(dim=1).item()
            pred_class = class_names[pred_label]

            try:
                with cam_constructor(model=model, target_layers=target_layers) as cam:
                    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_label)],
                                        aug_smooth=True, eigen_smooth=True)[0, :]
                    cam_image = show_cam_on_image(img_for_overlay, grayscale_cam, use_rgb=True, image_weight=0.5)

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
                    fig.suptitle(f"CAM Analysis for Class: '{true_class}'", fontsize=24, fontweight='bold')

                    axes[0].imshow(img_for_overlay)
                    axes[0].set_title(f'Original Image', fontsize=20)
                    axes[0].axis('off')

                    axes[1].imshow(cam_image)
                    axes[1].set_title(f'{cam_algorithm_name} Overlay (Pred: {pred_class})', fontsize=20)
                    axes[1].axis('off')

                    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, wspace=0.1, hspace=0.1)

                    save_name = f"sample{samples_done}_pred_{pred_class}_true_{true_class}.png"
                    plt.savefig(os.path.join(cam_save_dir, save_name))
                    plt.close(fig)

            except Exception as e_cam:
                logger.error(f"Failed to generate CAM for sample {samples_done}: {e_cam}", exc_info=True)

            samples_done += 1
        if samples_done >= num_samples:
            break

    logger.info(f"Finished generating {samples_done} CAM comparison images.")


def plot_tsne_visualization(features, labels, class_names, run_dir, prefix="tsne"):
    with temp_font_settings(FONT_CONFIG):
        try:
            logger.info(f"Running t-SNE for {features.shape[0]} samples...")
            n_samples = features.shape[0]
            perplexity_value = min(30.0, float(n_samples - 1))
            if perplexity_value <= 1.0:
                logger.warning(f"Samples ({n_samples}) too few for t-SNE. Skipping.")
                return
            tsne = TSNE(n_components=2, perplexity=perplexity_value, n_iter=1000, random_state=42, init='pca',
                        learning_rate='auto')
            tsne_results = tsne.fit_transform(features)
            df = pd.DataFrame({"tsne-2d-one": tsne_results[:, 0], "tsne-2d-two": tsne_results[:, 1],
                               "label": [class_names[i] for i in labels]})

            plt.figure(figsize=(16, 16))
            show_legend = "full" if len(class_names) <= 30 else False
            sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="label",
                            palette=sns.color_palette("hsv", len(class_names)), data=df, legend=show_legend, alpha=0.7)
            plt.title(f't-SNE Visualization of Feature Space ({prefix.replace("_", " ").title()})')
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            if show_legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            save_path = os.path.join(run_dir, "visualizations")
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"{prefix}_tsne_distribution.png")
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"t-SNE plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to generate t-SNE plot for '{prefix}': {e}", exc_info=True)


def plot_performance_gain_bar_chart(model_scores, baseline_scores, class_names, save_dir, metric_name="F1-score Gain",
                                    title="Performance Gain"):
    with temp_font_settings(FONT_CONFIG):
        try:
            gains = model_scores - baseline_scores
            colors = ['g' if x >= 0 else 'r' for x in gains]
            plt.figure(figsize=(max(15, len(class_names) * 0.5), 8))
            bars = plt.bar(np.arange(len(class_names)), gains, color=colors)
            plt.ylabel(f"Performance Gain ({metric_name})")
            plt.title(title)
            plt.xticks(np.arange(len(class_names)), class_names, rotation=65, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.axhline(0, color='black', linewidth=0.8)
            for bar in bars:
                yval = bar.get_height()
                va = 'bottom' if yval >= 0 else 'top'
                label = f"+{yval * 100:.1f}%" if yval >= 0 else f"{yval * 100:.1f}%"
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval, label, va=va, ha='center', color='black',
                         **FONT_CONFIG['bar_label'])
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, "performance_gain_comparison.png")
            plt.savefig(filename, dpi=200)
            plt.close()
            logger.info(f"Performance gain plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to plot performance gain chart: {e}", exc_info=True)


def plot_ablation_study_results(results_dict, save_dir, metric_name="F1-score", title="Ablation Study Results"):
    with temp_font_settings(FONT_CONFIG):
        try:
            labels, values = list(results_dict.keys()), list(results_dict.values())
            plt.figure(figsize=(14, 8))
            colors = plt.cm.viridis_r(np.linspace(0.1, 1.0, len(labels)))
            bars = plt.bar(labels, values, color=colors)
            plt.ylabel(f"Performance ({metric_name})")
            plt.title(title)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(min(values) * 0.98, max(values) * 1.02)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f'{yval:.4f}', va='bottom', ha='center', **FONT_CONFIG['bar_label'])
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, "ablation_study_comparison.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            logger.info(f"Ablation study plot saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to plot ablation study results: {e}", exc_info=True)