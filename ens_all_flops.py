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
import time
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import roc_auc_score

# --- اصلاح ۱: مسیر درست import برای fvcore ---
from fvcore.nn import FlopCountAnalysis

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
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_ratio, random_state=seed, stratify=labels)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size_adjusted, random_state=seed,
        stratify=[labels[i] for i in train_val_indices])
    return train_indices, val_indices, test_indices


def create_local_dataloaders(base_dir, batch_size, dataset_type, seed=42, is_distributed=False):
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05
        ),
        transforms.ToTensor(),
    ])

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
                                        self.samples.append(
                                            (os.path.join(subdir_path, img_file), self.class_to_idx[class_name]))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
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

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
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
            x_n = self.normalizations(x, i)
            out = self.models[i](x_n)
            if isinstance(out, (tuple, list)):
                out = out[0]
            prob = torch.sigmoid(out.float())
            probs_list.append(prob)

        final_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)
        return final_probs, None


# ==========================================
# 3. Model Complexity Calculator (اصلاح‌شده)
# ==========================================
def compute_complexity(model, input_size=(1, 3, 256, 256), device='cuda', model_name="", count_trainable_only=False):
    """
    محاسبه FLOPs و Params با گزارش صریح عملیات‌های پشتیبانی‌نشده توسط fvcore.

    اصلاحات نسبت به نسخه اولیه:
      - fvcore عملاً MACs (Multiply-Accumulate) را می‌شمارد، نه FLOPs.
        اینجا با ضرب در ۲ به FLOPs تبدیل می‌شود (استاندارد رایج در ادبیات فشرده‌سازی).
      - unsupported_ops() فراخوانی می‌شود تا مشخص شود آیا بخشی از گراف
        (مثلاً normalization سفارشی، sigmoid، mean/stack در aggregation)
        از شمارش جا افتاده است یا نه. این را باید قبل از گزارش نهایی در مقاله
        چک کنید، نه فرض کنید که صفر است.
      - در صورت نیاز، فقط پارامترهای trainable (غیر منجمد) شمرده می‌شود؛
        برای HFDE که base modelها frozen هستند این مهم است.
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    flop_analysis = FlopCountAnalysis(model, dummy_input)

    total_macs = flop_analysis.total()
    unsupported = flop_analysis.unsupported_ops()  # dict: op_name -> count

    total_flops = total_macs * 2  # MACs -> FLOPs

    if count_trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())

    print(f"\n  -> [{model_name}] Computational Complexity:")
    print(f"     FLOPs:  {total_flops / 1e9:.3f} GFLOPs (= 2 x {total_macs / 1e9:.3f} GMACs)")
    print(f"     Params: {total_params / 1e6:.3f} M"
          f" ({'trainable only' if count_trainable_only else 'all parameters'})")

    if unsupported:
        print(f"     WARNING: unsupported ops NOT counted in FLOPs "
              f"(verify impact before reporting in the paper):")
        for op_name, count in unsupported.items():
            print(f"         - {op_name}: {count} occurrence(s)")
    else:
        print(f"     OK: no unsupported operators encountered - FLOPs count is complete.")

    return {
        "model_name": model_name,
        "total_flops": float(total_flops),
        "total_gflops": float(total_flops / 1e9),
        "total_macs": float(total_macs),
        "total_params": int(total_params),
        "total_params_m": float(total_params / 1e6),
        "unsupported_ops": dict(unsupported) if unsupported else {},
    }


# ================== UNIFIED FINAL EVALUATION (با warmup اضافه‌شده) ==================
@torch.no_grad()
def final_evaluation_unified(model, test_loader, device, save_dir, model_name, args, is_main,
                              is_ensemble=True, warmup_batches=10):
    if not is_main:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    model.eval()
    all_y_true, all_y_score = [], []

    TP, TN, FP, FN = 0, 0, 0, 0
    total_samples = 0
    total_inference_time_ms = 0.0

    # --- اصلاح: warmup قبل از اندازه‌گیری واقعی ---
    print(f"\nWarming up ({warmup_batches} batches) before timing [{model_name}]...")
    warm_iter = iter(test_loader)
    for i in range(warmup_batches):
        try:
            w_images, _ = next(warm_iter)
        except StopIteration:
            break
        w_images = w_images.to(device)
        if is_ensemble:
            _ = model(w_images)
        else:
            _ = model(w_images)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"\nRunning Fast Batch Evaluation on {len(test_loader.dataset)} samples for [{model_name}]...")

    for images, labels in tqdm(test_loader, desc=f"Eval {model_name}"):
        images = images.to(device)
        labels_int = labels.long().tolist()

        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        if is_ensemble:
            final_output, _ = model(images)
            probs = final_output.squeeze(1).cpu().tolist()
        else:
            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            probs = torch.sigmoid(output.squeeze(1)).cpu().tolist()

        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            total_inference_time_ms += start_event.elapsed_time(end_event)
        else:
            total_inference_time_ms += (time.time() - start_time) * 1000.0

        for prob, label_int in zip(probs, labels_int):
            pred_int = int(prob > 0.5)

            all_y_true.append(label_int)
            all_y_score.append(prob)

            if label_int == 1:
                if pred_int == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred_int == 1:
                    FP += 1
                else:
                    TN += 1
            total_samples += 1

    avg_time_per_sample_ms = total_inference_time_ms / total_samples if total_samples > 0 else 0
    fps = 1000.0 / avg_time_per_sample_ms if avg_time_per_sample_ms > 0 else 0

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    auc_score = roc_auc_score(all_y_true, all_y_score)

    if is_ensemble:
        print(f"\n{'=' * 70}")
        print(f"FINAL RESULTS - {model_name}")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1_score:.4f}")
        print(f"AUC Score: {auc_score:.4f}")

        print(f"\nInference Time Statistics (post-warmup, batch_size={test_loader.batch_size}):")
        print(f"  Total Time:     {total_inference_time_ms / 1000:.2f} seconds")
        print(f"  Avg per Image:  {avg_time_per_sample_ms:.2f} ms")
        print(f"  Throughput:     {fps:.2f} FPS")

        print(f"{'=' * 70}")

        roc_json_path = os.path.join(save_dir, "roc_data_test.json")
        roc_data_json = {
            "metadata": {
                "dataset": args.dataset_type,
                "auc": float(auc_score),
                "accuracy": float(acc * 100),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "model": model_name
            },
            "y_true": all_y_true,
            "y_score": all_y_score
        }
        with open(roc_json_path, 'w', encoding='utf-8') as f:
            json.dump(roc_data_json, f, indent=2)
        print(f"ROC data saved to: {roc_json_path}")

    return acc * 100, precision, recall, f1_score, total_inference_time_ms, avg_time_per_sample_ms, fps


# ================== MODEL LOADING (بدون تغییر) ==================
def load_kd_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    if is_main:
        print(f"Loading {len(model_paths)} KD Student models (ResNet18)...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main:
                print(f" [ERROR] File not found: {path}")
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

            if is_main:
                print(f" [{len(models)}/{len(model_paths)}] Loaded: {os.path.basename(path)}")

        except Exception as e:
            if is_main:
                print(f" [ERROR] Failed {os.path.basename(path)}: {e}")

    if len(models) == 0:
        raise ValueError("No models loaded!")
    return models


# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Paper KD Ensemble - Final Corrected Version")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_type', type=str, required=True,
                         choices=['wild', 'real_fake', 'hard_fake_real', 'deepflux', 'uadfV',
                                  'real_fake_dataset', 'deepfake_lab'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_kd')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_batches', type=int, default=10,
                         help='Number of batches to run before timing starts (discarded).')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    is_main = True

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

        # ==========================================
        # بخش پیچیدگی محاسباتی: هم مدل تکی، هم ensemble
        # ==========================================
        print("\n" + "=" * 70)
        print("MODEL COMPUTATIONAL COMPLEXITY (FLOPs & Params)")
        print("=" * 70)

        complexity_results = {}

        print("\n[1] Single Student Model (ResNet18):")
        complexity_results["single_model"] = compute_complexity(
            base_models[0], device=device, model_name=MODEL_NAMES[0])

        print("\n[2] Ensemble Model (Nx ResNet18, soft voting):")
        ensemble_temp = PaperKDEnsemble(base_models, MEANS, STDS).to(device)
        complexity_results["ensemble"] = compute_complexity(
            ensemble_temp, device=device, model_name="PaperKDEnsemble")

        print("=" * 70)

        # ذخیره جداگانه پیچیدگی محاسباتی (مستقل از نتایج دقت)
        with open(os.path.join(args.save_dir, 'complexity_results.json'), 'w') as f:
            json.dump(complexity_results, f, indent=4)
        print(f"Complexity results saved to: "
              f"{os.path.join(args.save_dir, 'complexity_results.json')}")

        # ==========================================
        # ادامه ارزیابی دیتاست (دقت/F1 روی مدل‌های تکی)
        # ==========================================
        test_loader, base_dataset, test_indices = create_local_dataloaders(
            args.data_dir, args.batch_size, args.dataset_type, args.seed)

        print("\n" + "=" * 70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("=" * 70)

        individual_accs = []
        individual_f1s = []

        for i, model in enumerate(base_models):
            TP, TN, FP, FN = 0, 0, 0, 0
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    out = model(normalizations(images, i))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    pred = (torch.sigmoid(out.squeeze()) > 0.5).long()

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
            print(f" Model {i + 1} ({MODEL_NAMES[i]}): "
                  f"Acc={acc * 100:.2f}% | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

        best_single_idx = individual_accs.index(max(individual_accs))
        best_single_acc = individual_accs[best_single_idx]
        best_single_f1 = individual_f1s[best_single_idx]

        print(f"\nBest Single Model: Model {best_single_idx + 1} "
              f"({MODEL_NAMES[best_single_idx]}) -> Acc: {best_single_acc:.2f}% | F1: {best_single_f1:.4f}")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("FINAL ENSEMBLE EVALUATION")
        print("=" * 70)

        ensemble = ensemble_temp  # همان مدلی که برای FLOPs استفاده شد

        (ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1,
         total_time, avg_time, fps) = final_evaluation_unified(
            ensemble, test_loader, device, args.save_dir,
            "Paper KD Ensemble", args, is_main, is_ensemble=True,
            warmup_batches=args.warmup_batches
        )

        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        print(f"Best Single Model: Acc={best_single_acc:.2f}% | F1={best_single_f1:.4f}")
        print(f"Ensemble Model:    Acc={ensemble_acc:.2f}% | F1={ensemble_f1:.4f}")
        print(f"Accuracy Improvement: {ensemble_acc - best_single_acc:+.2f}%")
        print(f"F1-Score Improvement: {ensemble_f1 - best_single_f1:+.4f}")
        print("=" * 70)

        final_results = {
            'method': 'Paper_KD_Ensemble',
            'best_single_model': {
                'name': MODEL_NAMES[best_single_idx],
                'accuracy': float(best_single_acc),
                'f1_score': float(best_single_f1),
                'GFLOPs': complexity_results["single_model"]["total_gflops"],
                'Params_M': complexity_results["single_model"]["total_params_m"],
            },
            'ensemble': {
                'test_accuracy': float(ensemble_acc),
                'precision': float(ensemble_prec),
                'recall': float(ensemble_rec),
                'f1_score': float(ensemble_f1),
                'GFLOPs': complexity_results["ensemble"]["total_gflops"],
                'Params_M': complexity_results["ensemble"]["total_params_m"],
            },
            'accuracy_improvement': float(ensemble_acc - best_single_acc),
            'f1_improvement': float(ensemble_f1 - best_single_f1),
            'inference_stats': {
                'total_time_sec': float(total_time / 1000),
                'avg_time_per_sample_ms': float(avg_time),
                'fps': float(fps),
                'batch_size_used': args.batch_size,
                'warmup_batches': args.warmup_batches,
            },
            'unsupported_flop_ops': {
                'single_model': complexity_results["single_model"]["unsupported_ops"],
                'ensemble': complexity_results["ensemble"]["unsupported_ops"],
            }
        }
        with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nFinal results saved to: {os.path.join(args.save_dir, 'final_results.json')}")


if __name__ == "__main__":
    main()
