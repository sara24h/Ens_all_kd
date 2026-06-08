import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")

# ---- Importing your custom utilities ----
from dataset_utils import create_dataloaders, get_sample_info

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
        
        # Soft Voting روی Logitها
        final_output = outputs.mean(dim=1)
        
        if return_details:
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            return final_output, weights, None, outputs
        return final_output, None

# ================== UNIFIED FINAL EVALUATION ==================
@torch.no_grad()
def final_evaluation_and_report(ensemble, loader, device, save_dir, model_name, args, is_main):
    if not is_main or loader is None: return 0.0, None, None

    ensemble.eval()
    
    # استخراج دیتاست پایه و اندیس‌ها
    base_dataset = loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
        
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices'):
        test_indices = loader.sampler.indices
    elif hasattr(loader.dataset, 'indices'):
        test_indices = loader.dataset.indices
    else:
        test_indices = list(range(len(base_dataset)))

    all_y_true, all_y_score, all_y_pred = [], [], []
    lines = []

    lines.append("="*100)
    lines.append("SAMPLE-BY-SAMPLE PREDICTIONS (For McNemar Test Comparison):")
    lines.append("="*100)
    header = f"{'Sample_ID':<10} {'Sample_Path':<60} {'True_Label':<12} {'Predicted_Label':<15} {'Correct':<10}"
    lines.append(header)
    lines.append("-"*100)

    TP, TN, FP, FN = 0, 0, 0, 0
    correct_count, total_samples = 0, 0

    print(f"\nRunning Final Evaluation on {len(test_indices)} samples...")
    
    for i, global_idx in enumerate(tqdm(test_indices, desc="Final Eval")):
        try:
            image, label = base_dataset[global_idx]
            path, _ = get_sample_info(base_dataset, global_idx)
        except Exception as e:
            continue

        image = image.unsqueeze(0).to(device)
        label_int = int(label)
        
        final_output, weights, _, stacked_logits = ensemble(image, return_details=True)
        probs = torch.sigmoid(stacked_logits).mean(dim=1).item()
        pred_int = int(probs > 0.5)
        
        all_y_true.append(label_int)
        all_y_score.append(probs)
        all_y_pred.append(pred_int)
        
        is_correct = (pred_int == label_int)
        if is_correct: correct_count += 1
        
        if label_int == 1:
            if pred_int == 1: TP += 1
            else: FN += 1
        else:
            if pred_int == 1: FP += 1
            else: TN += 1
            
        total_samples += 1
        
        filename = os.path.basename(path)
        if len(filename) > 55: filename = filename[:25] + "..." + filename[-27:]
        line = f"{i+1:<10} {filename:<60} {label_int:<12} {pred_int:<15} {'Yes' if is_correct else 'No':<10}"
        lines.append(line)

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    print(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nSpecificity: {spec:.4f}")
    print(f"\nConfusion Matrix:\n                 Predicted Real  Predicted Fake")
    print(f"    Actual Real      {TP:<15} {FN:<15}")
    print(f"    Actual Fake      {FP:<15} {TN:<15}")
    print(f"\nCorrect Predictions: {correct_count} ({acc*100:.2f}%)")
    print("="*70)

    output_str = []
    output_str.append("-" * 100)
    output_str.append("SUMMARY STATISTICS:")
    output_str.append(f"Accuracy: {acc*100:.2f}%")
    output_str.append(f"Precision: {prec:.4f}\nRecall: {rec:.4f}\nSpecificity: {spec:.4f}")
    output_str.append(f"\nCorrect Predictions: {correct_count} ({acc*100:.2f}%)")
    output_str.extend(lines)

    log_path = os.path.join(save_dir, 'prediction_log.txt')
    with open(log_path, 'w') as f: f.write("\n".join(output_str))

    y_true_np, y_score_np, y_pred_np = np.array(all_y_true), np.array(all_y_score), np.array(all_y_pred)
    roc_json_path = os.path.join(save_dir, "roc_data_test.json")
    roc_data_json = {
        "metadata": {"dataset": args.dataset_type, "num_samples": int(total_samples), "model": "paper_kd_ensemble"},
        "y_true": y_true_np.tolist(), "y_score": y_score_np.tolist(), "y_pred": y_pred_np.tolist()
    }
    with open(roc_json_path, 'w', encoding='utf-8') as f: json.dump(roc_data_json, f, indent=2)
    print(f"✅ Prediction log & ROC data saved to: {save_dir}")

    return acc * 100, y_true_np, y_score_np

# ================== MODEL LOADING ==================
def load_kd_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    if is_main: print(f"Loading {len(model_paths)} KD Student models (ResNet20)...")
    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main: print(f" [WARNING] File not found: {path}")
            continue
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
    parser.add_argument('--save_dir', type=str, default='./output_kd')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, world_size, local_rank = int(os.environ["RANK"]), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank, world_size, local_rank, device = 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    is_main = rank == 0

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.models)]
    STDS = STDS[:len(args.models)]

    base_models = load_kd_models(args.models, device, is_main)
    ensemble = PaperKDEnsemble(base_models, MEANS, STDS).to(device)

    # برای ارزیابی نهایی با جزئیات، دیتالودر غیرتوزیع‌شده می‌سازیم (فقط روی رنک 0)
    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        _, _, test_loader_full = create_dataloaders(
            args.data_dir, args.batch_size, num_workers=2,
            dataset_type=args.dataset_type, is_distributed=False, 
            seed=args.seed, is_main=True
        )
        ensemble_test_acc, y_true, y_score = final_evaluation_and_report(
            ensemble, test_loader_full, device, args.save_dir, "Paper KD Ensemble", args, is_main
        )
        print(f"\nEnsemble Accuracy: {ensemble_test_acc:.2f}%")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
