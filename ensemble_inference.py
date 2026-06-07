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

# ---- Distributed Imports ----
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")

# ==========================================
# 1. Models (ResNet20 & Wrapper)
# ==========================================

class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1)

    def _make_layer(self, inplanes, planes, stride):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNetKD(nn.Module):
    def __init__(self, arch='resnet20'):
        super().__init__()
        if arch == 'resnet20':
            self.model = ResNet20()
        else:
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 2. Dataset (Simple & Generic for New Data)
# ==========================================

class FaceDataset(Dataset):
    def __init__(self, data_frame, root_dir, transform=None, img_column='images_id'):
        self.data = data_frame
        self.root_dir = root_dir
        self.transform = transform
        self.img_column = img_column
        self.label_map = {1: 1, 0: 0, 'real': 1, 'fake': 0, 'Real': 1, 'Fake': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # هندل کردن مسیر فایل
        path_in_csv = self.data[self.img_column].iloc[idx]
        
        # اگر مسیر کامل نیست به root_dir اضافه کن
        if not os.path.isabs(path_in_csv):
            img_name = os.path.join(self.root_dir, path_in_csv)
        else:
            img_name = path_in_csv
            
        if not os.path.exists(img_name):
            # اگر عکس پیدا نشد (برای جلوگیری از کرش)
            image = Image.new('RGB', (256, 256))
            label = 0
            if dist.get_rank() == 0:
                print(f"Warning: Image not found {img_name}, using dummy.")
        else:
            image = Image.open(img_name).convert('RGB')
            label = self.label_map[str(self.data['label'].iloc[idx]).lower()]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

# ==========================================
# 3. Ensemble Class (با Normalize اختصاصی)
# ==========================================

class MultiModelNormalization(nn.Module):
    """کلاس نرمال‌ساز مشابه کد Fuzzy Hesitant"""
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
        
        # شیء نرمال‌سازی مشترک برای همه مدل‌ها
        self.normalization = MultiModelNormalization(means, stds).to(device)
        
        for i, path in enumerate(model_paths):
            model = ResNetKD(arch='resnet20').to(self.device)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=self.device)
                state = ckpt.get('state_dict') or ckpt.get('model') or ckpt.get('model_state_dict')
                model.load_state_dict(state, strict=False)
                model.eval()
                self.models.append(model)
                if dist.get_rank() == 0:
                    print(f"Model {i+1} loaded: {path}")
            else:
                if dist.get_rank() == 0:
                    print(f"Warning: Model path not found: {path}")
    
    def predict(self, images):
        """Soft Voting با نرمال‌سازی اختصاصی"""
        images = images.to(self.device)
        probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                # نرمال‌سازی مخصوص مدل i ام
                norm_images = self.normalization(images, i)
                
                logits = model(norm_images)              
                prob = torch.sigmoid(logits)        
                probs.append(prob)
        
        ensemble_prob = torch.mean(torch.stack(probs), dim=0)
        ensemble_pred = (ensemble_prob > 0.5).float()
        
        return ensemble_prob, ensemble_pred

    def _gather_and_compute_metrics(self, local_probs, local_labels, dataset_name="Model"):
        """جمع‌آوری نتایج از تمام GPUها و محاسبه معیارها"""
        gathered_probs = [None for _ in range(dist.get_world_size())]
        gathered_labels = [None for _ in range(dist.get_world_size())]
        
        dist.all_gather_object(gathered_probs, local_probs)
        dist.all_gather_object(gathered_labels, local_labels)

        metrics = {}
        if dist.get_rank() == 0:
            final_probs = np.concatenate(gathered_probs)
            final_labels = np.concatenate(gathered_labels)
            final_preds = (final_probs > 0.5).astype(float)
            
            acc = accuracy_score(final_labels, final_preds)
            auc = roc_auc_score(final_labels, final_probs)
            prec = precision_score(final_labels, final_preds, zero_division=0)
            rec = recall_score(final_labels, final_preds)
            f1 = f1_score(final_labels, final_preds)
            
            print(f"\n=== {dataset_name} Results ===")
            print(f"Accuracy  : {acc:.4f}")
            print(f"AUC       : {auc:.4f}")
            print(f"Precision : {prec:.4f}")
            print(f"Recall    : {rec:.4f}")
            print(f"F1-Score  : {f1:.4f}")
            
            metrics = {'acc': acc, 'auc': auc, 'f1': f1, 'prec': prec, 'rec': rec}
        return metrics

    def evaluate_single_model(self, model, dataloader, model_index, model_name="Single Model"):
        """ارزیابی یک مدل خاص با Normalize اختصاصی"""
        if dist.get_rank() == 0:
            print(f"\nEvaluating {model_name}...")
            
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            data_iter = tqdm(dataloader, desc=f"Eval {model_name}") if dist.get_rank() == 0 else dataloader
            for images, labels in data_iter:
                images = images.to(self.device)
                
                # نرمال‌سازی با ایندکس مدل
                norm_images = self.normalization(images, model_index)
                
                logits = model(norm_images)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy().flatten())
                all_labels.append(labels.numpy().flatten())
        
        local_probs = np.concatenate(all_probs)
        local_labels = np.concatenate(all_labels)
        
        return self._gather_and_compute_metrics(local_probs, local_labels, dataset_name=model_name)

    def evaluate_ensemble(self, dataloader, dataset_name="Ensemble"):
        """ارزیابی انسمبل"""
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
    parser = argparse.ArgumentParser(description="Evaluate Ensemble & Single Models on NEW Dataset")

    # تنظیمات دیتاست جدید
    parser.add_argument("--test_csv", type=str, required=True, help="Path to NEW test CSV")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of NEW images")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # مسیر مدل‌های آموزش داده شده
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Paths to .pth files")
    
    args = parser.parse_args()

    # 1. مقداردهی اولیه Distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        print("="*70)
        print("ENSEMBLE & SINGLE MODEL EVALUATION (NEW DATASET)")
        print("="*70)
        print(f"Test CSV: {args.test_csv}")
        print(f"Root Dir: {args.root_dir}")
        print(f"Models: {len(args.models)}")
        print("="*70)

    # 2. آماده‌سازی Mean/Std (طبق استاندارد مدل‌های شما)
    # اگر مدل‌های شما Mean/Std متفاوتی دارند، این لیست را تغییر دهید
    DEFAULT_MEANS = [(0.5207, 0.4258, 0.3806), (0.4460, 0.3622, 0.3416), (0.4668, 0.3816, 0.3414)]
    DEFAULT_STDS  = [(0.2490, 0.2239, 0.2212), (0.2057, 0.1849, 0.1761), (0.2410, 0.2161, 0.2081)]
    
    num_models = len(args.models)
    if num_models > len(DEFAULT_MEANS):
        MEANS = DEFAULT_MEANS + [DEFAULT_MEANS[-1]] * (num_models - len(DEFAULT_MEANS))
        STDS = DEFAULT_STDS + [DEFAULT_STDS[-1]] * (num_models - len(DEFAULT_STDS))
    else:
        MEANS = DEFAULT_MEANS[:num_models]
        STDS = DEFAULT_STDS[:num_models]

    if dist.get_rank() == 0:
        print(f"Using Normalization Stats for {num_models} models.")

    # 3. لود دیتاست (بدون Normalize، تبدیل خام به Tensor)
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
        # Normalize حذف شد
    ])
    
    test_data = pd.read_csv(args.test_csv)
    
    # تشخیص نام ستون عکس (اختیاری)
    img_col = 'images_id'
    if img_col not in test_data.columns:
        if 'path' in test_data.columns: img_col = 'path'
        elif 'filename' in test_data.columns: img_col = 'filename'
        else: img_col = test_data.columns[0] # ستون اول

    test_dataset = FaceDataset(test_data, args.root_dir, transform=transform_test, img_column=img_col)
    
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False 
    )
    
    distributed_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 4. ساخت Ensemble با Mean/Std ها
    ensemble = DeepfakeEnsemble(
        args.models, 
        device=device, 
        means=MEANS, 
        stds=STDS
    )

    # 5. ارزیابی مدل‌های تکی
    single_models_metrics = []
    dist.barrier()

    if dist.get_rank() == 0:
        print("\n" + "="*30)
        print("Starting Single Model Evaluation")
        print("="*30)

    for i, model in enumerate(ensemble.models):
        dist.barrier()
        model_name = f"Model {i+1} ({os.path.basename(args.models[i])})"
        
        metrics = ensemble.evaluate_single_model(
            model, 
            distributed_test_loader, 
            model_index=i,
            model_name=model_name
        )
        
        if dist.get_rank() == 0:
            single_models_metrics.append({
                'index': i,
                'name': model_name,
                'metrics': metrics
            })

    # 6. نمایش بهترین مدل تکی
    if dist.get_rank() == 0:
        if single_models_metrics:
            best_model_info = max(single_models_metrics, key=lambda x: x['metrics']['auc'])
            print("\n" + "="*30)
            print(f"BEST SINGLE MODEL: {best_model_info['name']}")
            print(f"AUC: {best_model_info['metrics']['auc']:.4f}")
            print(f"Accuracy: {best_model_info['metrics']['acc']:.4f}")
            print("="*30 + "\n")

    # 7. ارزیابی نهایی انسمبل
    dist.barrier()
    ensemble.evaluate_ensemble(
        distributed_test_loader,
        dataset_name="Ensemble (New Dataset)"
    )

    dist.destroy_process_group()
