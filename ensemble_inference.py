import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.distributed as dist

warnings.filterwarnings("ignore")

# ==========================================
# 0. Dataset Utilities
# ==========================================
class TransformSubset(Subset):
    def __init__(self, dataset, indices, transform):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        if hasattr(self.dataset, 'samples'):
            img_path, label = self.dataset.samples[original_idx]
            img = Image.open(img_path).convert('RGB')
        else:
            img, label = self.dataset[original_idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_sample_info(dataset, index):
    if hasattr(dataset, 'samples'):
        return dataset.samples[index]
    elif hasattr(dataset, 'dataset'):
        return get_sample_info(dataset.dataset, index)
    else:
        raise AttributeError("Cannot find samples in dataset")

def create_standard_reproducible_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    labels = [dataset.samples[i][1] for i in indices]
    train_val_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=seed, stratify=labels)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size_adjusted, random_state=seed, stratify=[labels[i] for i in train_val_indices])
    return train_indices, val_indices, test_indices

def create_local_dataloaders(base_dir, batch_size, dataset_type, seed=42, is_distributed=False):
    val_test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
    ])
    
    # تعریف مسیر پوشه‌ها برای دیتاست‌های ImageFolder
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
        'real_fake_dataset': ['face_fake', 'face_real'], 
        'deepfake_lab': ['training_fake', 'training_real'], 
    }

    print(f"\n[Dataset Loading] Processing: {dataset_type}")

    # 1. دیتاست Wild
    if dataset_type == 'wild':
        splits = ['train', 'valid', 'test']
        datasets_dict = {}
        for split in splits:
            path = os.path.join(base_dir, split)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Folder not found: {path}")
            transform = train_transform if split == 'train' else val_test_transform
            datasets_dict[split] = datasets.ImageFolder(path, transform=transform)
        
        test_dataset = datasets_dict['test']
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader, test_dataset, list(range(len(test_dataset)))

    # 2. دیتاست UADFV
    elif dataset_type == 'uadfV':
        # استفاده از کلاس دیتاست اختصاصی شما
        class UADFVDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.samples = []
                self.class_to_idx = {'fake': 0, 'real': 1}
                for class_name in ['fake', 'real']:
                    frames_dir = os.path.join(self.root_dir, class_name, 'frames')
                    if os.path.exists(frames_dir):
                        for subdir in os.listdir(frames_dir):
                            subdir_path = os.path.join(frames_dir, subdir)
                            if os.path.isdir(subdir_path):
                                for img_file in os.listdir(subdir_path):
                                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        self.samples.append((os.path.join(subdir_path, img_file), self.class_to_idx[class_name]))
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                if self.transform: img = self.transform(img)
                return img, label

        full_dataset = UADFVDataset(base_dir, transform=val_test_transform)
        _, _, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader, full_dataset, test_indices

    # 3. دیتاست‌های Custom GenAI
    elif dataset_type in ['custom_genai', 'custom_genai_v2']:
        # استفاده از کلاس NewGenAIDataset شما
        class NewGenAIDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.samples = []
                self.label_map = {'fake': 0, 'real': 1}
                for dirpath, dirnames, filenames in os.walk(self.root_dir):
                    current_folder_name = os.path.basename(dirpath)
                    if current_folder_name in ['real', 'fake']:
                        label = self.label_map[current_folder_name]
                        valid_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        for img_file in valid_files:
                            self.samples.append((os.path.join(dirpath, img_file), label))
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                if self.transform: img = self.transform(img)
                return img, label

        full_dataset = NewGenAIDataset(base_dir, transform=val_test_transform)
        _, _, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader, full_dataset, test_indices

    # 4. دیتاست‌های استاندارد ImageFolder (real_fake, hard_fake_real, etc.)
    elif dataset_type in dataset_paths:
        folders = dataset_paths[dataset_type]
        dataset_dir = base_dir
        if not all(os.path.exists(os.path.join(dataset_dir, f)) for f in folders):
            possible_sub_dir = os.path.join(base_dir, dataset_type)
            if all(os.path.exists(os.path.join(possible_sub_dir, f)) for f in folders):
                dataset_dir = possible_sub_dir
            else:
                for dirpath, dirnames, _ in os.walk(base_dir):
                    if all(f in dirnames for f in folders):
                        dataset_dir = dirpath
                        break
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        full_dataset = datasets.ImageFolder(dataset_dir, transform=temp_transform)
        train_indices, val_indices, test_indices = create_standard_reproducible_split(full_dataset, seed=seed)
        
        test_dataset = TransformSubset(full_dataset, test_indices, val_test_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader, full_dataset, test_indices
    
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported.")

# ==========================================
# 1. Models (ResNet20 Structure)
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(self.avgpool(x).view(x.size(0), -1))
        return x

class ResNetKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet20()

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. Normalization & Ensemble Classes
# ==========================================
class MultiModelNormalization(nn.Module):
    def __init__(self, means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        for i, (m, s) in enumerate(zip(means, stds)):
            self.register_buffer(f'mean_{i}', torch.tensor(m).view(1, 3, 1, 1))
            self.register_buffer(f'std_{i}', torch.tensor(s).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return (x - getattr(self, f'mean_{idx}')) / getattr(self, f'std_{idx}')

class PaperKDEnsemble(nn.Module):
    def __init__(self, models: List[nn.Module], means: List[Tuple[float]], stds: List[Tuple[float]]):
        super().__init__()
        self.num_models = len(models)
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                if isinstance(out, (tuple, list)): out = out[0]
                if out.dim() == 1: out = out.unsqueeze(1)
            outputs[:, i] = out
        

        probs = torch.sigmoid(outputs)   # ابتدا probability
        final_output = probs.mean(dim=1) # سپس میانگین — soft voting واقعی
        
        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_output, weights, None, outputs
        return final_output, None

# ================== UNIFIED FINAL EVALUATION ==================
# ================== UNIFIED FINAL EVALUATION (BATCH-BASED) ==================
@torch.no_grad()
def final_evaluation_unified(model, test_loader, device, save_dir, model_name, args, is_main, is_ensemble=True):
    if not is_main: return 0.0

    model.eval()
    all_y_true, all_y_score, all_y_pred = [], [], []
    
    TP, TN, FP, FN = 0, 0, 0, 0
    correct_count, total_samples = 0, 0

    print(f"\nRunning Fast Batch Evaluation on {len(test_loader.dataset)} samples for [{model_name}]...")
    
    for images, labels in tqdm(test_loader, desc=f"Eval {model_name}"):
        images = images.to(device)
        labels_int = labels.long().tolist()  # تبدیل برچسب‌های این بچ به لیست پایتونی
        
        if is_ensemble:
            # خروجی انسمبل شما (final_output) خودش میانگین احتمالات پس از سیگموئید است
            # نیازی به بازپخش سیگموئید روی stacked_logits نیست
            final_output, _ = model(images)
            probs = final_output.squeeze(1).cpu().tolist()
        else:
            output = model(images)
            if isinstance(output, (tuple, list)): output = output[0]
            probs = torch.sigmoid(output.squeeze(1)).cpu().tolist()
            
        # پردازش نتایج بچ جاری
        for prob, label_int in zip(probs, labels_int):
            pred_int = int(prob > 0.5)
            
            all_y_true.append(label_int)
            all_y_score.append(prob)
            all_y_pred.append(pred_int)
            
            if pred_int == label_int: 
                correct_count += 1
            
            if label_int == 1:
                if pred_int == 1: TP += 1
                else: FN += 1
            else:
                if pred_int == 1: FP += 1
                else: TN += 1
                
            total_samples += 1

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    if is_ensemble:
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {model_name}")
        print(f"{'='*70}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | Specificity: {spec:.4f}")
        print(f"Confusion Matrix:\n                 Predicted Real  Predicted Fake")
        print(f"    Actual Real      {TP:<15} {FN:<15}")
        print(f"    Actual Fake      {FP:<15} {TN:<15}")
        print(f"Correct: {correct_count} ({acc*100:.2f}%) | Incorrect: {total - correct_count} ({(1-acc)*100:.2f}%)")
        print("="*70)

        roc_json_path = os.path.join(save_dir, "roc_data_test.json")
        roc_data_json = {
            "metadata": {"dataset": args.dataset_type, "num_samples": int(total_samples), "model": "paper_kd_ensemble"},
            "y_true": all_y_true, "y_score": all_y_score, "y_pred": all_y_pred
        }
        with open(roc_json_path, 'w', encoding='utf-8') as f: 
            json.dump(roc_data_json, f, indent=2)
        print(f"✅ ROC data saved to: {roc_json_path}")

    return acc * 100

# ================== MODEL LOADING ==================
def load_kd_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    if is_main: print(f"Loading {len(model_paths)} KD Student models (ResNet20)...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path): continue
        try:
            model = ResNetKD().to(device)
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict: state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            models.append(model)
            if is_main: print(f" [{i+1}/{len(model_paths)}] Loaded: {os.path.basename(path)}")
        except Exception as e:
            if is_main: print(f" [ERROR] Failed {path}: {e}")
    if len(models) == 0: raise ValueError("No models loaded!")
    return models

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Paper KD Ensemble")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_type', type=str, required=True, choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV', 'real_fake_dataset', 'deepfake_lab'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_kd')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

        # ================== بخش اصلاح شده ==================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 🟢 راه‌حل ارور gloo: استفاده از device_id به جای torch.cuda.set_device
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=local_rank  # این خط ارور را برطرف می‌کند
        )
        
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank, world_size, local_rank, device = 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_main = rank == 0
    # ==================================================
    if len(args.model_names) != len(args.models):
        raise ValueError("Number of model_names must match model_paths")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.models)]
    STDS = STDS[:len(args.models)]

    base_models = load_kd_models(args.models, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]
    
    normalizations = MultiModelNormalization(MEANS, STDS).to(device)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        test_loader, base_dataset, test_indices = create_local_dataloaders(
            args.data_dir, args.batch_size, args.dataset_type, args.seed)

        # 1. ارزیابی تک‌تک مدل‌ها (فقط چاپ دقت درصدی)
        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)
        
        individual_accs = []
        for i, model in enumerate(base_models):
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    out = model(normalizations(images, i))
                    if isinstance(out, (tuple, list)): out = out[0]
                    pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
                    correct += pred.eq(labels.long()).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            individual_accs.append(acc)
            print(f" Model {i+1} ({MODEL_NAMES[i]}): {acc:.2f}%")

        best_single = max(individual_accs)
        best_idx = individual_accs.index(best_single)
        print(f"\nBest Single Model: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")
        print("="*70)

        
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)
        
        ensemble = PaperKDEnsemble(base_models, MEANS, STDS).to(device)
        
        # تغییر کلیدی: پاس دادن test_loader به جای base_dataset و حذف آرگومان test_indices
        ensemble_acc = final_evaluation_unified(
            ensemble, 
            test_loader, 
            device, 
            args.save_dir, 
            "Paper KD Ensemble", 
            args, 
            is_main, 
            is_ensemble=True
        )

        # 3. مقایسه نهایی
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
        print(f"Improvement: {ensemble_acc - best_single:+.2f}%")
        print("="*70)

        final_results = {
            'method': 'Paper_KD_Ensemble',
            'best_single_model': {'name': MODEL_NAMES[best_idx], 'accuracy': float(best_single)},
            'ensemble': {'test_accuracy': float(ensemble_acc)},
            'improvement': float(ensemble_acc - best_single)
        }
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
