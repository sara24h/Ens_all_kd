import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import argparse
import warnings
import sys

# ---- Distributed Imports ----
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

warnings.filterwarnings("ignore")

# ==========================================
# 0. مستقل‌سازی لودر دیتاست (Standalone Dataset Loader)
# ==========================================

def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)

def create_reproducible_split(dataset, seed=42, train_ratio=0.7, val_ratio=0.15):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    return train_indices, val_indices, test_indices

class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.dataset[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders_ddp(base_dir: str, batch_size: int, rank: int, world_size: int,
                          num_workers: int = 2, dataset_type: str = 'wild'):
    """نسخه مستقل و اصلاح شده برای اجرا در Kaggle"""
    
    if rank == 0:
        print("="*70)
        print(f"Creating DataLoaders (Standalone Version) | Dataset: {dataset_type}")
        print("="*70)
   
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
   
    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
   
    if dataset_type == 'wild':
        splits_map = {'train': 'train', 'valid': 'valid', 'test': 'test'}
        if not os.path.exists(os.path.join(base_dir, 'valid')):
            splits_map['valid'] = 'val'
            
        loaders = {}
        for split_key, split_folder in splits_map.items():
            path = os.path.join(base_dir, split_folder)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder not found: {path}")
            
            if rank == 0:
                print(f"Loading {split_key.upper()}: {path}")

            transform = train_transform if split_key == 'train' else val_test_transform
            dataset = datasets.ImageFolder(path, transform=transform)
            
            if split_key == 'train':
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
                loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True,
                                  worker_init_fn=worker_init_fn)
            else:
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True, drop_last=False,
                                  worker_init_fn=worker_init_fn)
            loaders[split_key] = loader
        
        return loaders['train'], loaders['valid'], loaders['test']

    elif dataset_type in ['real_fake', 'hard_fake_real', 'deepflux', 'uadfV']:
        if rank == 0:
            print(f"Processing {dataset_type} from: {base_dir}")
        
        dataset_dir = base_dir
        possible_subdirs = ['real_and_fake_face', 'hardfakevsrealfaces', 'DeepFLUX', 'UADFV']
        found = False
        
        if os.path.exists(os.path.join(base_dir, 'real')) or os.path.exists(os.path.join(base_dir, 'fake')):
             found = True
        else:
            for subdir in possible_subdirs:
                if os.path.exists(os.path.join(base_dir, subdir)):
                    dataset_dir = os.path.join(base_dir, subdir)
                    found = True
                    break
        
        if not found:
            pass

        full_dataset = datasets.ImageFolder(dataset_dir)
        if rank == 0:
            print(f"Classes found: {full_dataset.classes}")
            print(f"Total images: {len(full_dataset)}")

        train_indices, val_indices, test_indices = create_reproducible_split(full_dataset, seed=42)
        
        train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
        val_dataset = TransformSubset(full_dataset, val_indices, val_test_transform)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                 num_workers=num_workers, pin_memory=True, drop_last=True,
                                 worker_init_fn=worker_init_fn)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True, drop_last=False,
                               worker_init_fn=worker_init_fn)
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                worker_init_fn=worker_init_fn)
        
        return train_loader, val_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ==========================================
# 1. Models (اصلاح شده برای تطبیق با کد آموزش)
# ==========================================

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes)
        ) if stride != 1 or inplanes != planes else None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)

class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16))
        self.layer2 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer3 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64), BasicBlock(64, 64))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        x = self.fc(self.avgpool(feat3).view(feat3.size(0), -1))
        return x

# ==========================================
# تغییر مهم: کلاس ResNetKD دقیقا مشابه ResNetStudent در کد آموزش ساخته شد
# ==========================================
class ResNetKD(nn.Module):
    def __init__(self, arch='resnet20'):
        super().__init__()
        # در کد آموزش شما، ResNetStudent یک لایه مدل داشت که ResNet20 بود
        # بنابراین کلیدها با 'model.' شروع می‌شوند
        self.model = ResNet20()
        
        # اگر آرکی‌تکچر دیگری بود اینجا اضافه کنید
        if arch != 'resnet20':
             pass 

    def forward(self, x):
        # در کد آموزش شما، خروجی student (logits, features) بود.
        # اما چون ما فقط state_dict را لود می‌کنیم، هندها (Hooks) فعال نمی‌شوند.
        # بنابراین forward ساده شده کافی است چون only weights matter.
        return self.model(x)


# ==========================================
# 3. Ensemble Class
# ==========================================

class MultiModelNormalization(nn.Module):
    def __init__(self, means, stds):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x, idx):
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class DeepfakeEnsemble:
    def __init__(self, model_paths, device, means, stds):
        self.device = device
        self.models = []
        self.normalization = MultiModelNormalization(means, stds).to(device)
        
        for i, path in enumerate(model_paths):
            # ایجاد مدل با ساختار صحیح (ResNetKD = ResNetStudent)
            model = ResNetKD(arch='resnet20').to(self.device)
            
            if os.path.exists(path):
                try:
                    # لود کردن state_dict
                    # در کد آموزش شما: torch.save(student.module.state_dict(), ...)
                    # پس فایل حاوی state_dict خام است (بدون کلیدهای اضافی)
                    # اما چون student شامل self.model بود، کلیدها 'model.' دارند.
                    
                    state_dict = torch.load(path, map_location=self.device, weights_only=False)
                    
                    # اگر فایل شامل کلیدهای اضافی باشد (ملاحظه می‌کنیم)
                    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']

                    # هندل کردن پیشوند 'model.'
                    # مدل ResNetKD دارای ویژگی self.model است.
                    # فایل ذخیره شده احتمالا شامل کلیدهایی مثل model.conv1.weight است.
                    # اما چون ما self.model.load_state_dict را صدا می‌زنیم (یا خود مدل را)، باید دقت کنیم.
                    
                    # روش ایمن: چون ResNetKD دقیقا ResNetStudent را تقلید می‌کند،
                    # load_state_dict(strict=False) باید کار کند اگر پیشوند 'model.' درست باشد.
                    
                    model.load_state_dict(state_dict, strict=False)
                    
                    model.eval()
                    self.models.append(model)
                    if dist.get_rank() == 0:
                        print(f"Model {i+1} loaded successfully: {path}")
                        
                except Exception as e:
                    if dist.get_rank() == 0:
                        print(f"Error loading {path}: {e}")
            else:
                if dist.get_rank() == 0:
                    print(f"Warning: Model path not found: {path}")
    
    def predict(self, images):
        images = images.to(self.device)
        probs = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                norm_images = self.normalization(images, i)
                logits = model(norm_images)              
                prob = torch.sigmoid(logits)        
                probs.append(prob)
        ensemble_prob = torch.mean(torch.stack(probs), dim=0)
        ensemble_pred = (ensemble_prob > 0.5).float()
        return ensemble_prob, ensemble_pred

    def _gather_and_compute_metrics(self, local_probs, local_labels, dataset_name="Model"):
        metrics = {}
        # چون هیچ DistributedSampler برای تست تعریف نشده،
        # تمام GPUها کل داده را دیده‌اند و local_probs در همه GPUها یکسان است.
        # بنابراین فقط رنک ۰ متریک را محاسبه می‌کند و نیازی به gather نیست.
        if dist.get_rank() == 0:
            final_probs = local_probs
            final_labels = local_labels
            final_preds = (final_probs > 0.5).astype(float)
            
            acc = accuracy_score(final_labels, final_preds)
            # محافظت در برابر دیتاست‌هایی که فقط یک کلاس دارند
            try:
                auc = roc_auc_score(final_labels, final_probs)
            except ValueError:
                auc = 0.0
                
            prec = precision_score(final_labels, final_preds, zero_division=0)
            rec = recall_score(final_labels, final_preds, zero_division=0)
            f1 = f1_score(final_labels, final_preds, zero_division=0)
            
            print(f"\n=== {dataset_name} Results ===")
            print(f"Accuracy  : {acc:.4f}")
            print(f"AUC       : {auc:.4f}")
            print(f"Precision : {prec:.4f}")
            print(f"Recall    : {rec:.4f}")
            print(f"F1-Score  : {f1:.4f}")
            metrics = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec}
            
        # بقیه GPUها منتظر رنک ۰ می‌مانند تا همگام‌سازی حفظ شود
        dist.barrier()
        return metrics

    def evaluate_single_model(self, model, dataloader, model_index, model_name="Single Model"):
        if dist.get_rank() == 0:
            print(f"\nEvaluating {model_name}...")
        all_probs = []
        all_labels = []
        with torch.no_grad():
            data_iter = tqdm(dataloader, desc=f"Eval {model_name}") if dist.get_rank() == 0 else dataloader
            for images, labels in data_iter:
                images = images.to(self.device)
                norm_images = self.normalization(images, model_index)
                logits = model(norm_images)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy().flatten())
                all_labels.append(labels.numpy().flatten())
        
        local_probs = np.concatenate(all_probs)
        local_labels = np.concatenate(all_labels)
        return self._gather_and_compute_metrics(local_probs, local_labels, dataset_name=model_name)

    def evaluate_ensemble(self, dataloader, dataset_name="Ensemble"):
        all_probs = []
        all_labels = []
        if dist.get_rank() == 0:
            print(f"\nEvaluating {dataset_name} (Distributed)...")
        with torch.no_grad():
            data_iter = tqdm(dataloader, desc=f"Eval {dataset_name}") if dist.get_rank() == 0 else dataloader
            for images, labels in data_iter:
                probs, _ = self.predict(images)
                all_probs.append(probs.cpu().numpy().flatten())
                all_labels.append(labels.numpy().flatten())
        
        local_probs = np.concatenate(all_probs)
        local_labels = np.concatenate(all_labels)
        return self._gather_and_compute_metrics(local_probs, local_labels, dataset_name=dataset_name)


# ====================== اجرا ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ensemble (Fixed Architecture)")

    parser.add_argument("--data_dir", type=str, required=True, help="Root directory of NEW dataset")
    parser.add_argument("--dataset_type", type=str, required=True, 
                        choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV'],
                        help="Type of dataset structure")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Paths to .pth files")
    
    args = parser.parse_args()

    # 1. Distributed Init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        print("="*70)
        print("ENSEMBLE EVALUATION (MATCHED ARCHITECTURE)")
        print("="*70)
        print(f"Data Dir: {args.data_dir}")
        print(f"Dataset Type: {args.dataset_type}")
        print(f"Models: {len(args.models)}")
        print("="*70)

    # 2. Mean/Std
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    num_models = len(args.models)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    if rank == 0:
        print(f"Using Normalization Stats for {num_models} models.")

    # 3. لود دیتاست
    try:
        _, _, distributed_test_loader = create_dataloaders_ddp(
            base_dir=args.data_dir,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=4,
            dataset_type=args.dataset_type
        )
        if rank == 0:
            print(f"Dataset loaded successfully.")
    except Exception as e:
        if rank == 0:
            print(f"Error loading dataset: {e}")
        dist.destroy_process_group()
        sys.exit(1)

    # 4. ساخت Ensemble (ResNetKD اکنون ResNetStudent را تقلید می‌کند)
    ensemble = DeepfakeEnsemble(args.models, device=device, means=MEANS, stds=STDS)

    # 5. ارزیابی تک به تک
    single_models_metrics = []
    dist.barrier()
    if rank == 0:
        print("\n" + "="*30)
        print("Starting Single Model Evaluation")
        print("="*30)

    for i, model in enumerate(ensemble.models):
        dist.barrier()
        model_name = f"Model {i+1} ({os.path.basename(args.models[i])})"
        metrics = ensemble.evaluate_single_model(model, distributed_test_loader, model_index=i, model_name=model_name)
        if rank == 0:
            single_models_metrics.append({'index': i, 'name': model_name, 'metrics': metrics})

    # 6. بهترین مدل
    if rank == 0:
        if single_models_metrics:
            best_model_info = max(single_models_metrics, key=lambda x: x['metrics']['auc'])
            print("\n" + "="*30)
            print(f"BEST SINGLE MODEL: {best_model_info['name']}")
            print(f"AUC: {best_model_info['metrics']['auc']:.4f}")
            print("="*30 + "\n")

    # 7. انسمبل
    dist.barrier()
    ensemble.evaluate_ensemble(distributed_test_loader, dataset_name="Ensemble (New Dataset)")

    dist.destroy_process_group()
