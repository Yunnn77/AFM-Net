import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import json

class UCMercedDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, class_to_idx=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            placeholder_image = Image.new('RGB', (256, 256), color='red')
            return placeholder_image, torch.tensor(-1)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def _load_paths_labels_from_ucmerced_dir(data_dir, class_to_idx_map=None):
    all_image_paths = []
    all_labels = []
    if not os.path.isdir(data_dir):
        print(f"Directory '{data_dir}' is invalid or does not exist.")
        return all_image_paths, all_labels, {}

    if class_to_idx_map is None:
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not class_names:
            print(f"No class subdirectories found in '{data_dir}'.")
            return all_image_paths, all_labels, {}
        class_to_idx_map = {cls_name: i for i, cls_name in enumerate(class_names)}
    else:
        class_names = list(class_to_idx_map.keys())

    print(f"Loading UCMerced data from '{data_dir}'. Expected classes: {class_names}")

    for class_name in class_names:
        if class_name not in class_to_idx_map:
            print(f"Class '{class_name}' in '{data_dir}' not in provided class_to_idx map. Skipping.")
            continue
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            label_idx = class_to_idx_map[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.tif', '.jpg', '.png')):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_labels.append(label_idx)
    return all_image_paths, all_labels, class_to_idx_map

def get_ucmerced_dataloaders(config: dict):
    data_cfg = config['data']
    ucm_root_train_val = data_cfg.get('ucmerced_data_root')
    ucm_root_test = data_cfg.get('ucmerced_test_data_root', None)
    img_size = data_cfg['img_size']
    batch_size = data_cfg['batch_size']
    val_batch_size = data_cfg['val_batch_size']
    test_batch_size = data_cfg.get('test_batch_size', val_batch_size)
    num_workers = data_cfg['num_workers']
    val_split_ratio = data_cfg.get('val_split_ratio', 0.2)
    random_state_split = config['training'].get('seed', 42)

    if not ucm_root_train_val or not os.path.isdir(ucm_root_train_val):
        raise ValueError(f"UCMerced training/validation directory '{ucm_root_train_val}' is invalid.")

    all_train_val_image_paths, all_train_val_labels = [], []
    class_names = sorted([d for d in os.listdir(ucm_root_train_val) if os.path.isdir(os.path.join(ucm_root_train_val, d))])
    if not class_names:
        raise ValueError(f"No class subdirectories found in '{ucm_root_train_val}'.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    actual_num_classes = len(class_names)
    if data_cfg.get('num_classes') != actual_num_classes:
        print(f"Detected {actual_num_classes} classes in UCMerced directory. Updating config.")
        data_cfg['num_classes'] = actual_num_classes
        if 'model' in config and 'params' in config['model']:
            config['model']['params']['num_classes'] = actual_num_classes

    print(f"Loading UCMerced training/validation data. Found {actual_num_classes} classes.")

    for class_name in class_names:
        class_dir = os.path.join(ucm_root_train_val, class_name)
        label_idx = class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                all_train_val_image_paths.append(os.path.join(class_dir, img_name))
                all_train_val_labels.append(label_idx)

    if not all_train_val_image_paths:
        raise ValueError(f"No images found in '{ucm_root_train_val}'.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_val_image_paths, all_train_val_labels,
        test_size=val_split_ratio, random_state=random_state_split, stratify=all_train_val_labels
    )

    mean_ucm_0_1 = [0.485, 0.456, 0.406]
    std_ucm_0_1 = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_ucm_0_1, std=std_ucm_0_1)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_ucm_0_1, std=std_ucm_0_1)
    ])

    train_dataset = UCMercedDataset(train_paths, train_labels, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = UCMercedDataset(val_paths, val_labels, transform=eval_transform, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"UCMerced training loader: {len(train_loader)} batches, total {len(train_dataset)} samples.")
    print(f"UCMerced validation loader: {len(val_loader)} batches, total {len(val_dataset)} samples.")

    test_loader = None
    test_image_paths, test_labels = [], []

    if ucm_root_test and os.path.isdir(ucm_root_test):
        print(f"Loading test data from '{ucm_root_test}'...")
        test_image_paths, test_labels = _load_paths_and_labels_from_dir(ucm_root_test, class_to_idx)
        if test_image_paths:
            test_dataset = UCMercedDataset(test_image_paths, test_labels, transform=eval_transform, class_to_idx=class_to_idx)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            print(f"UCMerced test loader: {len(test_loader)} batches, total {len(test_dataset)} samples.")
        else:
            print(f"No images found in test directory '{ucm_root_test}'.")
    elif ucm_root_test:
        print(f"Invalid test directory '{ucm_root_test}' in config.")

    save_dir = config.get('dataset_split_save_dir', f'./dataset_splits/ucmerced_split_{random_state_split}')
    os.makedirs(save_dir, exist_ok=True)

    for split_name, paths, labels in [('train', train_paths, train_labels), ('val', val_paths, val_labels), ('test', test_image_paths, test_labels)]:
        if paths:
            split_data = {
                'image_paths': paths, 'labels': labels, 'class_names': class_names,
                'class_to_idx': class_to_idx, 'sample_count': len(paths)
            }
            with open(os.path.join(save_dir, f'{split_name}_split.json'), 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)

    print(f"UCMerced dataset splits saved to: {save_dir}")

    data_cfg['class_names'] = class_names
    data_cfg['class_to_idx'] = class_to_idx

    return train_loader, val_loader, test_loader

def get_ucmerced_retrieval_loader(config, batch_size):
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from torch.utils.data import DataLoader

    data_cfg = config['data']
    uc_root = data_cfg.get('ucmerced_data_root')
    img_size = data_cfg['img_size']
    num_workers = data_cfg.get('num_workers', 4)
    val_split_ratio = data_cfg.get('val_split_ratio', 0.2)
    random_state_split = config['training'].get('seed', 42)

    all_image_paths = []
    all_labels = []
    class_names = sorted([d for d in os.listdir(uc_root) if os.path.isdir(os.path.join(uc_root, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    for class_name in class_names:
        class_dir = os.path.join(uc_root, class_name)
        label_idx = class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                all_image_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(label_idx)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_image_paths, all_labels, test_size=val_split_ratio,
        random_state=random_state_split, stratify=all_labels
    )

    mean = data_cfg.get('normalize_mean', [0.485, 0.456, 0.406])
    std = data_cfg.get('normalize_std', [0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print("Info: Using test split from UC Merced for retrieval.")
    retrieval_dataset = UCMercedDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=eval_transform,
        class_to_idx=class_to_idx
    )

    if retrieval_dataset is None or len(retrieval_dataset) == 0:
        return None, None

    retrieval_loader = DataLoader(
        retrieval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return retrieval_loader, retrieval_dataset