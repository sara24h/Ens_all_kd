import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import time  # <======= ۱. اضافه شدن ماژول زمان
from sklearn.model_selection import train_test_split
from PIL import Image
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ==========================================
# 0. Dataset Utilities (بدون تغییر)
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
    # ✅ ۱. ترنسفورم ولیدیشن و تست: تغییر سایز مستقیم به (256, 256) بدون CenterCrop
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    # ✅ ۲. ترنسفورم آموزش: حذف RandomCrop، تغییر سایز مستقیم و تنظیم دقیق آگمنتیشن‌ها مطابق روش اول
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),          # تغییر سایز مستقیم (حفظ لبه‌های چهره)
        transforms.RandomHorizontalFlip(p=0.5), # آگمنتیشن افقی
        transforms.RandomRotation(degrees=10),   # آگمنتیشن چرخش
        transforms.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.1, 
            hue=0.05
        ),                                      # تنظیم دقیق پارامترهای رنگ جتر
        transforms.ToTensor(),                  # تبدیل به تانسور [0, 1] (بدون اعمال خطی نرمالایز)
    ])
    
    # =========================================================================
    # مابقی منطق مدیریت دیتاست‌ها (بدون تغییر نسبت به نسخه قبل شما برای حفظ سازگاری)
    # =========================================================================
    dataset_paths = {
        'real_fake': ['training_fake', 'training_real'],
        'hard_fake_real': ['fake', 'real'],
        'deepflux': ['Fake', 'Real'],
        'real_fake_dataset': ['face_fake', 'face_real'], 
        'deepfake_lab': ['training_fake', 'training_real'], 
    }

    print(f"\n[Dataset Loading] Processing: {dataset_type}")

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

    elif dataset_type == 'uadfV':
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

    elif dataset_type in ['custom_genai', 'custom_genai_v2']:
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
# 1. Models (بدون تغییر)
# ==========================================
class ResNetKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. Normalization & Ensemble Classes (بدون تغییر)
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
    def __init__(self, models, means, stds):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.normalizations = MultiModelNormalization(means, stds)

    def forward(self, x):
        probs_list = []
        for i in range(len(self.models)):
            # 1. اعمال نرمال‌سازی مخصوص هر مدل
            x_n = self.normalizations(x, i)
            
            # 2. دریافت خروجی خام مدل (Logits)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)): 
                out = out[0]
            
            # 3. تبدیل خروجی خام به احتمال (Probability) با Sigmoid
            # استفاده از float() برای جلوگیری از مشکلات احتمالی با Mixed Precision (fp16)
            prob = torch.sigmoid(out.float()) 
            probs_list.append(prob)
        
        # 4. میانگین‌گیری از احتمالات (Standard Soft Voting)
        # stack(..., dim=0) باعث میشه یک بُعد جدید برای مدل‌ها ساخته بشه
        # mean(dim=0) میانگین احتمالات همه مدل‌ها رو برای هر نمونه دیتا حساب می‌کنه
        final_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
        
        return final_probs, None

# ================== UNIFIED FINAL EVALUATION (افزوده شده: Time, Precision, Recall, F1) ==================
@torch.no_grad()
def final_evaluation_unified(model, test_loader, device, save_dir, model_name, args, is_main, is_ensemble=True):
    if not is_main: return 0.0, 0.0, 0.0, 0.0, {}

    model.eval()
    all_y_true, all_y_score = [], []
    
    TP, TN, FP, FN = 0, 0, 0, 0
    correct_count, total_samples = 0, 0

    print(f"\nRunning Fast Batch Evaluation on {len(test_loader.dataset)} samples for [{model_name}]...")
    
    # ======= ۲. شروع زمان‌سنجی =======
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()
    # =====================================

    for images, labels in tqdm(test_loader, desc=f"Eval {model_name}"):
        images = images.to(device)
        labels_int = labels.long().tolist()
        
        if is_ensemble:
            final_output, _ = model(images)
            probs = final_output.squeeze(1).cpu().tolist()
        else:
            output = model(images)
            if isinstance(output, (tuple, list)): output = output[0]
            probs = torch.sigmoid(output.squeeze(1)).cpu().tolist()
            
        for prob, label_int in zip(probs, labels_int):
            pred_int = int(prob > 0.5)
            
            all_y_true.append(label_int)
            all_y_score.append(prob)
            
            if pred_int == label_int: 
                correct_count += 1
            
            if label_int == 1:  # Real = 1
                if pred_int == 1: TP += 1
                else: FN += 1
            else:               # Fake = 0
                if pred_int == 1: FP += 1
                else: TN += 1
            total_samples += 1

    # ======= ۳. توقف و محاسبه زمان استنتاج =======
    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        total_inference_time_ms = start_event.elapsed_time(end_event)
    else:
        total_inference_time_ms = (time.time() - start_time) * 1000.0
        
    total_real_samples = len(test_loader.dataset)
    avg_time_per_sample_ms = total_inference_time_ms / total_real_samples if total_real_samples > 0 else 0
    fps = 1000.0 / avg_time_per_sample_ms if avg_time_per_sample_ms > 0 else 0
    # ================================================

    # محاسبه متریک‌ها
    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # محاسبه AUC
    auc_score = roc_auc_score(all_y_true, all_y_score)
    
    if is_ensemble:
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {model_name}")
        print(f"{'='*70}")
        
        # چاپ زمان استنتاج
        print(f"\nInference Time Statistics:")
        print(f"  Total Time:     {total_inference_time_ms/1000:.2f} seconds")
        print(f"  Avg per Image:  {avg_time_per_sample_ms:.2f} ms")
        print(f"  Throughput:     {fps:.2f} FPS (Frames Per Second)")
        
        print(f"\nAccuracy:  {acc*100:.2f}%")
        print(f"Precision: {precision:.4f} (دفقت در تشخیص Real)")
        print(f"Recall:    {recall:.4f} (حساسیت در تشخیص Real)")
        print(f"F1-Score:  {f1_score:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"{'='*70}")
        
        # آمار سرعت
        inference_stats = {
            'total_time_sec': float(total_inference_time_ms / 1000),
            'avg_time_per_sample_ms': float(avg_time_per_sample_ms),
            'fps': float(fps)
        }

        # ذخیره در JSON
        roc_json_path = os.path.join(save_dir, "roc_data_test.json")
        roc_data_json = {
            "metadata": {
                "dataset": args.dataset_type, 
                "auc": float(auc_score), 
                "accuracy": float(acc*100),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "model": "paper_kd_ensemble",
                "inference_stats": inference_stats  # <======= اضافه شدن به JSON
            },
            "y_true": all_y_true, 
            "y_score": all_y_score
        }
        with open(roc_json_path, 'w', encoding='utf-8') as f: 
            json.dump(roc_data_json, f, indent=2)
        print(f"✅ ROC data (including metrics & time) saved to: {roc_json_path}")

    return acc * 100, precision, recall, f1_score, inference_stats

# ================== MODEL LOADING (بدون تغییر) ==================
def load_kd_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    if is_main: print(f"Loading {len(model_paths)} KD Student models (ResNet18)...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path): 
            if is_main: print(f" [❌ ERROR] File not found: {path}")
            continue
            
        try:
            model = ResNetKD().to(device)
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
            if isinstance(state_dict, dict) and 'state_dict' in state_dict: 
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                elif k.startswith('module.'):  
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.model.load_state_dict(new_state_dict, strict=False)
            
            model.eval()
            models.append(model)
            
            if is_main: print(f" [✅ {len(models)}/{len(model_paths)}] Loaded: {os.path.basename(path)}")
            
        except Exception as e:
            if is_main: print(f" [❌ ERROR] Failed {os.path.basename(path)}: {e}")
            
    if len(models) == 0: raise ValueError("No models loaded!")
    return models

# ================== MAIN FUNCTION (افزوده شده متریک‌ها به مدل‌های تکی و مقایسه نهایی) ==================
def main():
    parser = argparse.ArgumentParser(description="Paper KD Ensemble")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_type', type=str, required=True, 
                        choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV', 'real_fake_dataset', 'deepfake_lab'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_kd')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        is_main = True
        rank, world_size, local_rank = 0, 1, 0
    else:
        device = torch.device('cpu')
        is_main = True
        rank, world_size, local_rank = 0, 1, 0

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

        print("\n" + "="*70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("="*70)
        
        individual_accs = []
        individual_f1s = []
        
        for i, model in enumerate(base_models):
            TP, TN, FP, FN = 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    out = model(normalizations(images, i))
                    if isinstance(out, (tuple, list)): out = out[0]
                    pred = (torch.sigmoid(out.squeeze()) > 0.5).long()
                    
                    # محاسبه ماتریس درهم‌ریختگی برای هر مدل تکی
                    labels_cpu = labels.long().cpu()
                    pred_cpu = pred.cpu()
                    
                    TP += ((pred_cpu == 1) & (labels_cpu == 1)).sum().item()
                    TN += ((pred_cpu == 0) & (labels_cpu == 0)).sum().item()
                    FP += ((pred_cpu == 1) & (labels_cpu == 0)).sum().item()
                    FN += ((pred_cpu == 0) & (labels_cpu == 1)).sum().item()

            total = TP + TN + FP + FN
            acc = (TP + TN) / total if total > 0 else 0
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            individual_accs.append(acc * 100)
            individual_f1s.append(f1)
            print(f" Model {i+1} ({MODEL_NAMES[i]}): Acc={acc*100:.2f}% | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

        best_single_idx = individual_accs.index(max(individual_accs))
        best_single_acc = individual_accs[best_single_idx]
        best_single_f1 = individual_f1s[best_single_idx]
        
        print(f"\nBest Single Model: Model {best_single_idx+1} ({MODEL_NAMES[best_single_idx]}) → Acc: {best_single_acc:.2f}% | F1: {best_single_f1:.4f}")
        print("="*70)

        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)
        
        ensemble = PaperKDEnsemble(base_models, MEANS, STDS).to(device)
        
        # دریافت تمامی متریک‌ها و زمان استنتاج از تابع ارزیابی
        ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1, inference_stats = final_evaluation_unified(
            ensemble, test_loader, device, args.save_dir, 
            "Paper KD Ensemble", args, is_main, is_ensemble=True
        )

        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Model: Acc={best_single_acc:.2f}% | F1={best_single_f1:.4f}")
        print(f"Ensemble Model:    Acc={ensemble_acc:.2f}% | F1={ensemble_f1:.4f}")
        print(f"Accuracy Improvement: {ensemble_acc - best_single_acc:+.2f}%")
        print(f"F1-Score Improvement: {ensemble_f1 - best_single_f1:+.4f}")
        print("="*70)

        final_results = {
            'method': 'Paper_KD_Ensemble',
            'best_single_model': {
                'name': MODEL_NAMES[best_single_idx], 
                'accuracy': float(best_single_acc),
                'f1_score': float(best_single_f1)
            },
            'ensemble': {
                'test_accuracy': float(ensemble_acc), 
                'precision': float(ensemble_prec),
                'recall': float(ensemble_rec),
                'f1_score': float(ensemble_f1)
            },
            'accuracy_improvement': float(ensemble_acc - best_single_acc),
            'f1_improvement': float(ensemble_f1 - best_single_f1),
            'inference_stats': inference_stats  # <======= اضافه شدن به فایل نتایج نهایی
        }
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
