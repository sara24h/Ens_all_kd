import torch
import torch.nn.functional as F
from torchvision.models import resnet50
import os
from tqdm import tqdm

# ====================== ResNetKD (همان کلاس قبلی) ======================
class ResNetKD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        
    def forward(self, x):
        return self.model(x)

# ====================== Ensemble Model ======================
class EnsembleModel:
    def __init__(self, model_paths, device='cuda'):
        self.device = device
        self.models = []
        
        for path in model_paths:
            model = ResNetKD().to(device)
            ckpt = torch.load(path, map_location=device)
            
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'], strict=False)
                elif 'model' in ckpt:
                    model.load_state_dict(ckpt['model'], strict=False)
                else:
                    model.load_state_dict(ckpt, strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            
            model.eval()
            self.models.append(model)
        
        print(f"{len(self.models)} مدل با موفقیت لود شد.")

    def predict(self, images):
        """images: tensor با shape (batch, 3, H, W)"""
        images = images.to(self.device)
        probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(images)                    # (batch, 1)
                prob = torch.sigmoid(logits)              # تبدیل به احتمال
                probs.append(prob)
        
        # Soft Voting: میانگین احتمال‌ها
        ensemble_prob = torch.mean(torch.stack(probs), dim=0)   # (batch, 1)
        
        # پیش‌بینی نهایی
        predictions = (ensemble_prob > 0.5).float()
        
        return ensemble_prob, predictions

    def evaluate(self, dataloader):
        """ارزیابی روی دیتالودر"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                probs, preds = self.predict(images)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # محاسبه معیارها
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        import numpy as np
        
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Ensemble Accuracy : {acc:.4f}")
        print(f"Ensemble AUC      : {auc:.4f}")
        print(f"Precision         : {prec:.4f}")
        print(f"Recall            : {rec:.4f}")
        print(f"F1-Score          : {f1:.4f}")
        
        return acc, auc, f1

# ====================== استفاده ======================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # مسیر مدل‌های دانش‌آموز
    model_paths = [
        "student_140k_at.pth",      # مدل AT روی 140k
        "student_190k_logits.pth",  # مدل Logits روی 190k
        "student_200k_rkd.pth"      # مدل RKD روی 200k
    ]
    
    ensemble = EnsembleModel(model_paths, device=device)
    
    # مثال: ارزیابی روی تست ست یکی از دیتاست‌ها (مثلاً 200k)
    from your_dataset_file import Dataset_selector   # یا همان فایل قبلی
    
    ds = Dataset_selector(dataset_mode='200k',          # یا 140k یا 190k
                          realfake200k_test_csv=..., 
                          realfake200k_root_dir=...,
                          eval_batch_size=64, ddp=False)
    
    test_loader = ds.loader_test
    
    print("ارزیابی Ensemble روی تست ست:")
    ensemble.evaluate(test_loader)
