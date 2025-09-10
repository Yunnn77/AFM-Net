import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import json

class NWPUDataset(Dataset):
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
            placeholder_image = Image.new('RGB', (256, 256), color='red')
            return placeholder_image, torch.tensor(-1)

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def _load_paths_labels_from_nwpu_dir(data_dir, class_to_idx_map=None):
    all_image_paths = []
    all_labels = []

    if not os.path.isdir(data_dir):
        return all_image_paths, all_labels, {}

    if class_to_idx_map is None:
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if not class_names:
            return all_image_paths, all_labels, {}
        class_to_idx_map = {cls_name: i for i, cls_name in enumerate(class_names)}
    else:
        class_names = list(class_to_idx_map.keys())

    for class_name in class_names:
        if class_name not in class_to_idx_map:
            continue
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            label_idx = class_to_idx_map[class_name]
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    all_image_paths.append(os.path.join(class_path, img_name))
                    all_labels.append(label_idx)
    return all_image_paths, all_labels, class_to_idx_map

def get_nwpu_dataloaders(config: dict):
    data_cfg = config['data']
    nwpu_root_train_val = data_cfg.get('nwpu_data_root')
    nwpu_root_test = data_cfg.get('nwpu_test_data_root', None)
    img_size = data_cfg['img_size']
    batch_size = data_cfg['batch_size']
    val_batch_size = data_cfg['val_batch_size']
    test_batch_size = data_cfg.get('test_batch_size', val_batch_size)
    num_workers = data_cfg['num_workers']
    val_split_ratio = data_cfg.get('val_split_ratio', 0.2)
    random_state_split = config['training'].get('seed', 42)

    if not nwpu_root_train_val or not os.path.isdir(nwpu_root_train_val):
        raise ValueError(f"NWPU training/validation root '{nwpu_root_train_val}' is not set or invalid.")

    all_train_val_image_paths = []
    all_train_val_labels = []
    class_names = sorted([d for d in os.listdir(nwpu_root_train_val) if os.path.isdir(os.path.join(nwpu_root_train_val, d))])

    if not class_names:
        raise ValueError(f"No class subdirectories found in '{nwpu_root_train_val}'.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    actual_num_classes = len(class_names)
    if data_cfg.get('num_classes') != actual_num_classes:
        print(f"Info: Detected {actual_num_classes} classes from NWPU directory. Updating config.")
        data_cfg['num_classes'] = actual_num_classes
        if 'model' in config and 'params' in config['model']:
            config['model']['params']['num_classes'] = actual_num_classes

    print(f"Loading NWPU training/validation data. Found {actual_num_classes} classes.")

    for class_name in class_names:
        class_dir = os.path.join(nwpu_root_train_val, class_name)
        label_idx = class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_train_val_image_paths.append(os.path.join(class_dir, img_name))
                all_train_val_labels.append(label_idx)

    if not all_train_val_image_paths:
        raise ValueError(f"No image files found in '{nwpu_root_train_val}'.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_val_image_paths, all_train_val_labels,
        test_size=val_split_ratio,
        random_state=random_state_split,
        stratify=all_train_val_labels
    )

    mean_nwpu_0_1 = [0.367, 0.382, 0.358]
    std_nwpu_0_1 = [0.135, 0.131, 0.132]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nwpu_0_1, std=std_nwpu_0_1)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_nwpu_0_1, std=std_nwpu_0_1)
    ])

    train_dataset = NWPUDataset(train_paths, train_labels, transform=train_transform, class_to_idx=class_to_idx)
    val_dataset = NWPUDataset(val_paths, val_labels, transform=eval_transform, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    print(f"NWPU training loader: {len(train_loader)} batches, total {len(train_dataset)} samples.")
    print(f"NWPU validation loader: {len(val_loader)} batches, total {len(val_dataset)} samples.")

    test_loader = None
    test_image_paths, test_labels = [], []

    if nwpu_root_test and os.path.isdir(nwpu_root_test):
        print(f"Loading independent test data from '{nwpu_root_test}'...")
        test_image_paths, test_labels = _load_paths_and_labels_from_dir(nwpu_root_test, class_to_idx)

        if test_image_paths:
            test_dataset = NWPUDataset(test_image_paths, test_labels, transform=eval_transform,
                                       class_to_idx=class_to_idx)
            test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,
                                     pin_memory=True)
            print(f"NWPU test loader: {len(test_loader)} batches, total {len(test_dataset)} samples.")
        else:
            print(f"Warning: No image files found in test directory '{nwpu_root_test}'.")
    elif nwpu_root_test:
        print(f"Warning: Configured test directory '{nwpu_root_test}' is invalid.")

    save_dir = config.get('dataset_split_save_dir', f'./dataset_splits/nwpu_split_{random_state_split}')
    os.makedirs(save_dir, exist_ok=True)

    for split_name, paths, labels in [('train', train_paths, train_labels),
                                      ('val', val_paths, val_labels),
                                      ('test', test_image_paths, test_labels)]:
        if paths:
            split_data = {
                'image_paths': paths,
                'labels': labels,
                'class_names': class_names,
                'class_to_idx': class_to_idx,
                'sample_count': len(paths)
            }
            with open(os.path.join(save_dir, f'{split_name}_split.json'), 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)

    print(f"NWPU dataset split saved to: {save_dir}")

    data_cfg['class_names'] = class_names
    data_cfg['class_to_idx'] = class_to_idx

    return train_loader, val_loader, test_loader

def get_nwpu_retrieval_loader(config, batch_size):
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from torch.utils.data import DataLoader

    data_cfg = config['data']
    nwpu_root_train_val = data_cfg.get('nwpu_data_root')
    nwpu_root_test = data_cfg.get('nwpu_test_data_root', None)
    img_size = data_cfg['img_size']
    num_workers = data_cfg.get('num_workers', 4)
    val_split_ratio = data_cfg.get('val_split_ratio', 0.2)
    random_state_split = config['training'].get('seed', 42)

    all_train_val_image_paths = []
    all_train_val_labels = []
    class_names = sorted([d for d in os.listdir(nwpu_root_train_val) if os.path.isdir(os.path.join(nwpu_root_train_val, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(nwpu_root_train_val, class_name)
        label_idx = class_to_idx[class_name]
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_train_val_image_paths.append(os.path.join(class_dir, img_name))
                all_train_val_labels.append(label_idx)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_train_val_image_paths, all_train_val_labels,
        train_size=val_split_ratio,
        random_state=random_state_split,
        stratify=all_train_val_labels
    )

    mean = data_cfg.get('normalize_mean', [0.485, 0.456, 0.406])
    std = data_cfg.get('normalize_std', [0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    print("Info: Using test split from NWPU for retrieval.")
    retrieval_dataset = NWPUDataset(
        image_paths=test_paths,
        labels=test_labels,
        transform=eval_transform,
        class_to_idx=class_to_idx
    )

    if retrieval_dataset is None or len(retrieval_dataset) == 0:
        return None, None

    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return retrieval_loader, retrieval_dataset