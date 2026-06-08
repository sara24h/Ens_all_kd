import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import warnings
import argparse
import json
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")

# ---- Importing your custom utilities ----
from dataset_utils import create_dataloaders, get_sample_info
from metrics_utils import plot_roc_and_f1
from visualization_utils import generate_visualizations

# ================== UTILITY FUNCTIONS ==================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        x = self.fc(self.avgpool(feat3).view(feat3.size(0), -1))
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
        # ایجاد تنسور برای ذخیره خروجی (Logit) هر مدل
        outputs = torch.zeros(x.size(0), self.num_models, 1, device=x.device)
        
        for i in range(self.num_models):
            x_n = self.normalizations(x, i)
            with torch.no_grad():
                out = self.models[i](x_n)
                # مدل‌های ResNetKD خروجی تک‌بعدی میدن، ابعاد رو هم‌ساز می‌کنیم
                if isinstance(out, tuple) or isinstance(out, list):
                    out = out[0]
                if out.dim() == 1:
                    out = out.unsqueeze(1)
            outputs[:, i] = out

        # 🟢 انسمبل روی Logitها (Soft Voting مطابق معادله 9 مقاله برای تسک باینری)
        final_output = outputs.mean(dim=1)
        
        if return_details:
            # وزن‌ها یکنواخت هستن (1/تعداد مدل)
            weights = torch.ones(x.size(0), self.num_models, device=x.device) / self.num_models
            # خروجی دقیقاً هم‌فرمت کلاس SimpleAveragingEnsemble شماست
            return final_output, weights, None, outputs
        
        return final_output, None

# ================== MODEL LOADING ==================
def load_kd_models(model_paths: List[str], device: torch.device, is_main: bool) -> List[nn.Module]:
    models = []
    if is_main:
        print(f"Loading {len(model_paths)} KD Student models (ResNet20)...")

    for i, path in enumerate(model_paths):
        if not os.path.exists(path):
            if is_main: print(f" [WARNING] File not found: {path}")
            continue
        if is_main: print(f" [{i+1}/{len(model_paths)}] Loading: {os.path.basename(path)}")
        try:
            model = ResNetKD().to(device)
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
                
            # لود با strict=True برای اطمینان از صحت وزن‌ها
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            
            param_count = sum(p.numel() for p in model.parameters())
            if is_main: print(f" → Parameters: {param_count:,}")
            models.append(model)
        except Exception as e:
            if is_main: print(f" [ERROR] Failed to load {path}: {e}")
            continue

    if len(models) == 0:
        raise ValueError("No models loaded!")
    if is_main: print(f"All {len(models)} models loaded!\n")
    return models

# ================== EVALUATION FUNCTIONS ==================
@torch.no_grad()
def evaluate_single_model_ddp(model: nn.Module, loader: DataLoader, device: torch.device,
                              name: str, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float], is_main: bool) -> float:
    model.eval()
    normalizer = MultiModelNormalization([mean], [std]).to(device)
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=f"Evaluating {name}", disable=not is_main):
        images, labels = images.to(device), labels.to(device).float()
        images = normalizer(images, 0)
        out = model(images)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # استفاده از sigmoid و آستانه 0.5 برای ResNet20
        pred = (torch.sigmoid(out.squeeze(1)) > 0.5).long()
        total += labels.size(0)
        correct += pred.eq(labels.long()).sum().item()

    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    acc = 100. * correct_tensor.item() / total_tensor.item()
    if is_main: print(f" {name}: {acc:.2f}%")
    return acc

# ================== DISTRIBUTED SETUP ==================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return device, local_rank, rank, world_size
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device, 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ================== MAIN FUNCTION ==================
def main():
    parser = argparse.ArgumentParser(description="Paper KD Ensemble (Logits, AT, RKD)")
    parser.add_argument('--epochs', type=int, default=1, help="Unused")
    parser.add_argument('--lr', type=float, default=0.0001, help="Unused")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_grad_cam_samples', type=int, default=5)
    parser.add_argument('--num_lime_samples', type=int, default=5)
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['wild', 'deepfake_lab', 'hard_fake_real', 'uadfV','real_fake_dataset'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_paths', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--save_dir', type=str, default='./output_kd')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_paths):
        raise ValueError("Number of model_names must match model_paths")

    set_seed(args.seed)
    device, local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if is_main:
        print("="*70)
        print(f"PAPER KD ENSEMBLE (Soft Voting on Logits)")
        print(f"Distributed on {world_size} GPU(s) | Seed: {args.seed}")
        print("="*70)
        print(f"Dataset: {args.dataset} | Batch size: {args.batch_size}")
        print(f"Models: {len(args.model_paths)}")
        print("="*70 + "\n")

    MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    STDS = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    MEANS = MEANS[:len(args.model_paths)]
    STDS = STDS[:len(args.model_paths)]

    base_models = load_kd_models(args.model_paths, device, is_main)
    MODEL_NAMES = args.model_names[:len(base_models)]

    # ساخت انسمبل
    ensemble = PaperKDEnsemble(base_models, MEANS, STDS).to(device)

    if world_size > 1:
        ensemble = DDP(ensemble, device_ids=[local_rank], output_device=local_rank)

    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, dataset_type=args.dataset,
        is_distributed=(world_size > 1), seed=args.seed, is_main=is_main)

    # ارزیابی تک‌تک مدل‌ها
    if is_main:
        print("\n" + "="*70)
        print("INDIVIDUAL KD MODEL PERFORMANCE")
        print("="*70)

    individual_accs = []
    for i, model in enumerate(base_models):
        acc = evaluate_single_model_ddp(
            model, test_loader, device,
            f"Student {i+1} ({MODEL_NAMES[i]})",
            MEANS[i], STDS[i], is_main)
        individual_accs.append(acc)

    best_single = max(individual_accs)
    best_idx = individual_accs.index(best_single)

    if is_main:
        print(f"\nBest Single Student: Model {best_idx+1} ({MODEL_NAMES[best_idx]}) → {best_single:.2f}%")

    # ارزیابی نهایی انسمبل
    if is_main:
        print("\n" + "="*70)
        print("FINAL ENSEMBLE EVALUATION")
        print("="*70)

        ensemble_module = ensemble.module if hasattr(ensemble, 'module') else ensemble
        os.makedirs(args.save_dir, exist_ok=True)
        
        # ساخت دیتالودر تست غیرتوزیع‌شده (همانطور که در کد خودتان وجود دارد)
        _, _, test_loader_full = create_dataloaders(
            args.data_dir, args.batch_size, dataset_type=args.dataset,
            is_distributed=False, seed=args.seed, is_main=True
        )

        # فراخوانی تابع ارزیابی یکپارچه شما (نیازی به تغییر در این تابع نیست)
        ensemble_test_acc, y_true, y_score = final_evaluation_and_report(
            ensemble_module, test_loader_full, device, args.save_dir, 
            "Paper KD Ensemble", args, is_main
        )

        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Best Single Student: {best_single:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_test_acc:.2f}%")
        print(f"Improvement: {ensemble_test_acc - best_single:+.2f}%")
        print("="*70)

        final_results = {
            'method': 'Paper KD Ensemble',
            'best_single_model': {
                'name': MODEL_NAMES[best_idx],
                'accuracy': float(best_single)
            },
            'ensemble': {
                'test_accuracy': float(ensemble_test_acc),
                'strategy': 'Soft Voting on Logits (Eq. 9)'
            },
            'improvement': float(ensemble_test_acc - best_single)
        }

        results_path = os.path.join(args.save_dir, 'final_results_paper_kd.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)
        print(f"\nResults saved: {results_path}")

        final_model_path = os.path.join(args.save_dir, 'final_ensemble_model_paper_kd.pt')
        torch.save({
            'ensemble_state_dict': ensemble_module.state_dict(),
            'test_accuracy': ensemble_test_acc,
            'model_names': MODEL_NAMES,
            'means': MEANS,
            'stds': STDS
        }, final_model_path)
        print(f"Model saved: {final_model_path}")

        vis_dir = os.path.join(args.save_dir, 'visualizations')
        generate_visualizations(
            ensemble_module, test_loader_full, device, vis_dir, MODEL_NAMES,
            args.num_grad_cam_samples, args.num_lime_samples,
            args.dataset, is_main)

        plot_roc_and_f1(
            ensemble_module,
            test_loader_full, 
            device, 
            args.save_dir, 
            MODEL_NAMES,
            is_main
        )

    cleanup_distributed()

if __name__ == "__main__":
    main()
