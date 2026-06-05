import torch
import torch.nn.functional as F
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

class ResNetKD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)  # Binary output

    def forward(self, x):
        return self.model(x)

class DeepfakeEnsemble:
    def __init__(self, model_paths, device='cuda'):
        self.device = device
        self.models = []
        
        for i, path in enumerate(model_paths):
            model = ResNetKD().to(device)
            ckpt = torch.load(path, map_location=device)
            
            if isinstance(ckpt, dict):
                state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
                model.load_state_dict(state, strict=False)
            else:
                model.load_state_dict(ckpt, strict=False)
            
            model.eval()
            self.models.append(model)
            print(f"Model {i+1} loaded: {path}")
    
    def predict(self, images):
        """Soft Voting - دقیقاً مثل مقاله"""
        images = images.to(self.device)
        probs = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(images)              # (batch, 1)
                prob = torch.sigmoid(logits)        # تبدیل به احتمال
                probs.append(prob)
        
        # Soft Voting (میانگین احتمال‌ها) ← مثل مقاله
        ensemble_prob = torch.mean(torch.stack(probs), dim=0)
        ensemble_pred = (ensemble_prob > 0.5).float()
        
        return ensemble_prob, ensemble_pred

    def evaluate(self, dataloader, dataset_name="Test"):
        all_probs = []
        all_preds = []
        all_labels = []
        
        print(f"\nEvaluating Ensemble on {dataset_name}...")
        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                probs, preds = self.predict(images)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # محاسبه معیارها
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"\n=== Ensemble Results ({dataset_name}) ===")
        print(f"Accuracy  : {acc:.4f}")
        print(f"AUC       : {auc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        
        return acc, auc, f1


# ====================== اجرا ======================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # مسیر سه مدل دانش‌آموز (دقیقاً مثل مقاله: سه مدل با روش متفاوت)
    model_paths = [
        "student_140k_at.pth",       # Feature-based (AT)
        "student_190k_logits.pth",   # Response-based (Logits)
        "student_200k_rkd.pth"       # Relation-based (RKD)
    ]
    
    ensemble = DeepfakeEnsemble(model_paths, device=device)
    
    # مثال: ارزیابی روی تست ست 200k (یا هر دیتاست دیگری)
    from your_dataset_file import Dataset_selector   # فایل Dataset_selector خودت
    
    ds = Dataset_selector(
        dataset_mode='200k',
        realfake200k_test_csv='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
        realfake200k_root_dir='/kaggle/input/undersampled-200k/balanced_unique_200k_dataset',
        eval_batch_size=64,
        ddp=False
    )
    
    test_loader = ds.loader_test
    ensemble.evaluate(test_loader, dataset_name="200k Test Set")
