import os
import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# ---- ایمپورت‌های ضروری برای Distributed Evaluation ----
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# فرض بر این است که کلاس‌های ResNet20 و Dataset_selector در فایل یا سلول‌های قبلی تعریف شده‌اند.

class ResNetKD(torch.nn.Module):
    def __init__(self, arch='resnet50'):
        super().__init__()
        if arch == 'resnet20':
            self.model = ResNet20()
        else:
            self.model = resnet50(pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

class DeepfakeEnsemble:
    def __init__(self, model_paths, device):
        self.device = device
        self.models = []
        
        for i, path in enumerate(model_paths):
            model = ResNetKD(arch='resnet20').to(self.device)
            ckpt = torch.load(path, map_location=self.device)
            
            if isinstance(ckpt, dict):
                state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
                model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            
            model.eval()
            self.models.append(model)
            
            # فقط GPU اصلی لاگ را چاپ کند
            if dist.get_rank() == 0:
                print(f"Model {i+1} loaded: {path}")
    
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

    def evaluate(self, dataloader, dataset_name="Test"):
        all_probs = []
        all_preds = []
        all_labels = []
        
        if dist.get_rank() == 0:
            print(f"\nEvaluating Ensemble on {dataset_name} (Distributed)...")
            
        with torch.no_grad():
            # tqdm فقط در رنک 0 نمایش داده شود
            data_iter = tqdm(dataloader, desc=f"Evaluating") if dist.get_rank() == 0 else dataloader
            
            for images, labels in data_iter:
                probs, preds = self.predict(images)
                
                all_probs.append(probs.cpu().numpy().flatten())
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.numpy().flatten())
        
        # تبدیل لیست‌های محلی این GPU به آرایه نامپای
        local_probs = np.concatenate(all_probs)
        local_preds = np.concatenate(all_preds)
        local_labels = np.concatenate(all_labels)

        # ---- جمع‌آوری نتایج از تمام GPUها ----
        # استفاده از all_gather_object برای دسته‌های احتمالا نابرابر (مخصوصاً آخرین بچ)
        gathered_probs = [None for _ in range(dist.get_world_size())]
        gathered_labels = [None for _ in range(dist.get_world_size())]
        
        dist.all_gather_object(gathered_probs, local_probs)
        dist.all_gather_object(gathered_labels, local_labels)

        # محاسبه معیارها فقط در GPU اصلی (برای جلوگیری از تکرار)
        if dist.get_rank() == 0:
            # ترکیب کردن آرایه‌های برگشتی از هر دو GPU
            final_probs = np.concatenate(gathered_probs)
            final_preds = np.concatenate(gathered_preds)
            final_labels = np.concatenate(gathered_labels)
            
            acc = accuracy_score(final_labels, final_preds)
            auc = roc_auc_score(final_labels, final_probs)
            prec = precision_score(final_labels, final_preds)
            rec = recall_score(final_labels, final_preds)
            f1 = f1_score(final_labels, final_preds)
            
            print(f"\n=== Ensemble Results ({dataset_name}) ===")
            print(f"Accuracy  : {acc:.4f}")
            print(f"AUC       : {auc:.4f}")
            print(f"Precision : {prec:.4f}")
            print(f"Recall    : {rec:.4f}")
            print(f"F1-Score  : {f1:.4f}")
            
            return acc, auc, f1


# ====================== اجرا ======================
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_mode", type=str, default="200k", choices=["140k", "190k", "200k"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)

    args = parser.parse_args()

    # 1. راه‌اندازی DDP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()

    model_paths = [
        "student_140k_at.pth",
        "student_190k_logits.pth",
        "student_200k_rkd.pth"
    ]

    ensemble = DeepfakeEnsemble(model_paths, device=device)

    # 2. لود دیتاست (با ddp=False چون می‌خواهیم خودمان sampler بسازیم)
    ds = Dataset_selector(
        dataset_mode=args.dataset_mode,
        realfake200k_test_csv=args.test_csv,
        realfake200k_root_dir=args.root_dir,
        eval_batch_size=args.batch_size,
        ddp=False # دیتاست سلکتور را دست نمی‌زنیم
    )

    # 3. استخراج دیتاست خام از داخل DataLoader و ساخت DistributedSampler برای تست
    test_dataset = ds.loader_test.dataset
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False # در تست هرگز داده‌ها را شافل نمی‌کنیم
    )
    
    # ساخت DataLoader جدید با Sampler توزیع شده
    distributed_test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 4. ارزیابی
    ensemble.evaluate(
        distributed_test_loader,
        dataset_name=f"{args.dataset_mode} Test Set"
    )

    # 5. پاکسازی
    dist.destroy_process_group()
