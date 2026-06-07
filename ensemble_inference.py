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
# 1. Dataset & Models (از کد اول)
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
        # ما فرض می‌کنیم تمام مدل‌های دانشجو ResNet20 هستند
        if arch == 'resnet20':
            self.model = ResNet20()
        else:
            # اگر لازم بود ResNet50 هم اضافه شود
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

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
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            # اگر عکس پیدا نشد، یک عکس سیاه ایجاد می‌کنیم تا کرش نکند (یا خطا بدهید)
            # اینجا برای اطمینان از اجرا، هندل ساده می‌کنیم
            image = Image.new('RGB', (256, 256))
            label = 0
        else:
            image = Image.open(img_name).convert('RGB')
            label = self.label_map[str(self.data['label'].iloc[idx]).lower()]
        
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector:
    def __init__(self, dataset_mode, realfake140k_train_csv=None, realfake140k_valid_csv=None, 
                 realfake140k_test_csv=None, realfake140k_root_dir=None, realfake200k_train_csv=None, 
                 realfake200k_val_csv=None, realfake200k_test_csv=None, realfake200k_root_dir=None, 
                 realfake190k_root_dir=None, train_batch_size=32, eval_batch_size=32, num_workers=4, 
                 pin_memory=True, ddp=False, rank=0, world_size=1):
        
        if dataset_mode not in ['140k', '190k', '200k']:
            raise ValueError("dataset_mode must be '140k', '190k', '200k'")
        self.dataset_mode = dataset_mode
        image_size = (256, 256)

        # تعریف Mean/Std بر اساس دیتاست
        if dataset_mode == '140k':
            mean, std = (0.5207, 0.4258, 0.3806), (0.2490, 0.2239, 0.2212)
        elif dataset_mode == '200k':
            mean, std = (0.4460, 0.3622, 0.3416), (0.2057, 0.1849, 0.1761)
        elif dataset_mode == '190k':
            mean, std = (0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081)

        # Transform تست (بدون آگمنتیشن)
        transform_test = transforms.Compose([
            transforms.Resize(image_size), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        img_column = 'path' if dataset_mode == '140k' else 'images_id'
        root_dir = train_data = val_data = test_data = None

        # لود دیتافریم‌ها بر اساس مد
        if dataset_mode == '140k':
            if not all([realfake140k_test_csv, realfake140k_root_dir]): 
                raise ValueError("140k test paths missing")
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

        elif dataset_mode == '200k':
            if not all([realfake200k_test_csv, realfake200k_root_dir]): 
                raise ValueError("200k test paths missing")
            test_data = pd.read_csv(realfake200k_test_csv)
            root_dir = realfake200k_root_dir
            def create_image_path(row):
                folder = 'real' if str(row['label']).lower() in ['1', 'real'] else 'fake'
                # تلاش برای پیدا کردن نام فایل
                img_name = row.get('filename_clean', row.get('filename', row.get('image', row.get('path', ''))))
                return os.path.join('test', folder, os.path.basename(img_name))
            test_data['images_id'] = test_data.apply(lambda r: create_image_path(r), axis=1)

        elif dataset_mode == '190k':
            if not realfake190k_root_dir: raise ValueError("190k path missing")
            root_dir = realfake190k_root_dir
            def collect_images(split):
                data = []
                for label in ['Real', 'Fake']:
                    folder_path = os.path.join(root_dir, split, label)
                    if os.path.exists(folder_path):
                        for img_name in os.listdir(folder_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                data.append({'images_id': os.path.join(split, label, img_name), 'label': label})
                return pd.DataFrame(data)
            test_data = collect_images('Test')

        # ساخت دیتاست تست
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)
        
        # ساخت لودر تست (بدون شافل)
        # نکته: در اینجا sampler را بیرون از کلاس می‌سازیم تا کنترل DDP دست ما باشد
        self.test_dataset = test_dataset
        self.loader_test = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, 
                                      num_workers=num_workers, pin_memory=pin_memory)
        if rank == 0: print(f"Test Dataset loaded: {len(test_dataset)} samples.")

# ==========================================
# 2. Ensemble Class (گسترش یافته)
# ==========================================

class DeepfakeEnsemble:
    def __init__(self, model_paths, device):
        self.device = device
        self.models = []
        
        for i, path in enumerate(model_paths):
            model = ResNetKD(arch='resnet20').to(self.device)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location=self.device)
                # هندل کردن فرمت‌های مختلف ذخیره‌سازی
                state = ckpt.get('state_dict') or ckpt.get('model') or ckpt.get('model_state_dict')
                # اگر کلیدها شامل 'model.' هستند و مدل ResNetKD است که خودش self.model دارد
                # معمولا state_dict دانشجو با ResNetKD سازگار است
                model.load_state_dict(state, strict=False)
                model.eval()
                self.models.append(model)
                if dist.get_rank() == 0:
                    print(f"Model {i+1} loaded from: {path}")
            else:
                if dist.get_rank() == 0:
                    print(f"Warning: Model path not found: {path}")
    
    def predict(self, images):
        """Soft Voting"""
        images = images.to(self.device)
        probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(images)              
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

    def evaluate_single_model(self, model, dataloader, model_name="Single Model"):
        """ارزیابی یک مدل خاص"""
        if dist.get_rank() == 0:
            print(f"\nEvaluating {model_name}...")
            
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            data_iter = tqdm(dataloader, desc=f"Eval {model_name}") if dist.get_rank() == 0 else dataloader
            for images, labels in data_iter:
                images = images.to(self.device)
                logits = model(images)
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
                probs, _ = self.predict(images) # پیش‌بینی انسمبل
                
                all_probs.append(probs.cpu().numpy().flatten())
                all_labels.append(labels.numpy().flatten())
        
        local_probs = np.concatenate(all_probs)
        local_labels = np.concatenate(all_labels)

        return self._gather_and_compute_metrics(local_probs, local_labels, dataset_name=dataset_name)


# ====================== اجرا ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ensemble & Single Models")

    # تنظیمات دیتاست
    parser.add_argument("--dataset_mode", type=str, default="200k", choices=["140k", "190k", "200k"])
    parser.add_argument("--batch_size", type=int, default=64)
    
    # مسیر فایل‌های تست
    # برای هر مد باید مسیر تست مربوط به همان را بدهید
    parser.add_argument("--test_csv", type=str, default="", help="Path to test CSV")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory of images")

    # مسیر مدل‌های آموزش داده شده (Student models)
    parser.add_argument("--models", type=str, nargs='+', required=True, help="Paths to .pth files")

    args = parser.parse_args()

    # 1. مقداردهی اولیه Distributed
    dist.init_process_group(backend="nccl") # برای چند GPU در یک سیستم nccl بهتر است
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    if dist.get_rank() == 0:
        print("="*70)
        print("ENSEMBLE & SINGLE MODEL EVALUATION")
        print("="*70)
        print(f"Dataset Mode: {args.dataset_mode}")
        print(f"Models to evaluate: {len(args.models)}")
        print(f"GPU World Size: {world_size}")
        print("="*70)

    # 2. لود دیتاست
    # برای سادگی، مسیرهای CSV و Root را بر اساس dataset_mode نگاشت می‌کنیم
    # اگر args.dataset_mode با فایلی که دارید متفاوت است، اینجا باید دستی مسیر دهید
    ds = Dataset_selector(
        dataset_mode=args.dataset_mode,
        realfake140k_test_csv=args.test_csv if args.dataset_mode == '140k' else None,
        realfake140k_root_dir=args.root_dir if args.dataset_mode == '140k' else None,
        realfake200k_test_csv=args.test_csv if args.dataset_mode == '200k' else None,
        realfake200k_root_dir=args.root_dir if args.dataset_mode == '200k' else None,
        realfake190k_root_dir=args.root_dir if args.dataset_mode == '190k' else None,
        eval_batch_size=args.batch_size,
        ddp=False 
    )

    test_dataset = ds.test_dataset
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

    # 3. ساخت Ensemble (لود مدل‌ها)
    ensemble = DeepfakeEnsemble(args.models, device=device)

    # 4. ارزیابی مدل‌های تکی
    single_models_metrics = []
    dist.barrier() # همگام‌سازی

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
            model_name=model_name
        )
        
        if dist.get_rank() == 0:
            single_models_metrics.append({
                'index': i,
                'name': model_name,
                'metrics': metrics
            })

    # 5. نمایش بهترین مدل تکی
    if dist.get_rank() == 0:
        if single_models_metrics:
            best_model_info = max(single_models_metrics, key=lambda x: x['metrics']['auc'])
            print("\n" + "="*30)
            print(f"BEST SINGLE MODEL: {best_model_info['name']}")
            print(f"AUC: {best_model_info['metrics']['auc']:.4f}")
            print(f"Accuracy: {best_model_info['metrics']['acc']:.4f}")
            print("="*30 + "\n")

    # 6. ارزیابی نهایی انسمبل
    dist.barrier()
    ensemble.evaluate_ensemble(
        distributed_test_loader,
        dataset_name=f"Ensemble ({args.dataset_mode})"
    )

    dist.destroy_process_group()
