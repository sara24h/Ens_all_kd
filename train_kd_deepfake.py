import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ====================== Dataset Classes ======================
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
            raise FileNotFoundError(f"image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)

class Dataset_selector:
    # ... (تمام کد Dataset_selector که نوشتی را دقیقاً اینجا کپی کن)
    # (برای صرفه‌جویی در فضا، همان کد قبلی‌ات را نگه دار)
    pass  # ← اینجا کد کامل Dataset_selector را بچسبان

# ====================== KD Components ======================
def logits_loss(teacher_logits, student_logits, T=4.0):
    teacher_prob = torch.sigmoid(teacher_logits / T)
    student_prob = torch.sigmoid(student_logits / T)
    return F.kl_div(torch.log(student_prob + 1e-8), teacher_prob, reduction='batchmean') * (T ** 2)

def at_loss(teacher_features, student_features):
    loss = 0.0
    for t_feat, s_feat in zip(teacher_features, student_features):
        t_att = F.normalize(t_feat.pow(2).mean(1).view(t_feat.size(0), -1), dim=1)
        s_att = F.normalize(s_feat.pow(2).mean(1).view(s_feat.size(0), -1), dim=1)
        loss += F.mse_loss(s_att, t_att)
    return loss

class RKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, teacher_emb, student_emb):
        with torch.no_grad():
            t_dist = self.pairwise_distance(teacher_emb)
            t_angle = self.pairwise_angle(teacher_emb)
        s_dist = self.pairwise_distance(student_emb)
        s_angle = self.pairwise_angle(student_emb)
        dist_loss = F.smooth_l1_loss(s_dist, t_dist)
        angle_loss = F.smooth_l1_loss(s_angle, t_angle)
        return self.alpha * dist_loss + self.beta * angle_loss

    def pairwise_distance(self, x):
        x = F.normalize(x, p=2, dim=1)
        return torch.cdist(x, x, p=2)

    def pairwise_angle(self, x):
        x = F.normalize(x, p=2, dim=1)
        cosine = torch.mm(x, x.t())
        return torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

class ResNetKD(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output)
        
        self.model.layer2[-1].register_forward_hook(hook_fn)
        self.model.layer3[-1].register_forward_hook(hook_fn)
        self.model.layer4[-1].register_forward_hook(hook_fn)

    def forward(self, x):
        self.features.clear()
        logit = self.model(x)
        return logit, self.features

# ====================== Training Function ======================
def train_student(teacher_path, dataset_mode, kd_method='logits',
                  epochs=25, lr=0.005, device='cuda', batch_size=48):
    
    teacher = ResNetKD().to(device)
    ckpt = torch.load(teacher_path, map_location=device)
    
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            teacher.model.load_state_dict(ckpt['state_dict'], strict=False)
        elif 'model' in ckpt:
            teacher.model.load_state_dict(ckpt['model'], strict=False)
        else:
            teacher.model.load_state_dict(ckpt, strict=False)
    else:
        teacher.model.load_state_dict(ckpt, strict=False)
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = ResNetKD().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = nn.BCEWithLogitsLoss()

    # ایجاد دیتاست
    if dataset_mode == '140k':
        ds = Dataset_selector(dataset_mode='140k', 
                              realfake140k_train_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/train.csv', # مسیرها را پر کن
                              realfake140k_valid_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/valid.csv',
                              realfake140k_test_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/test.csv',
                              realfake140k_root_dir='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces',
                              train_batch_size=batch_size, eval_batch_size=64, ddp=False)
    elif dataset_mode == '190k':
        ds = Dataset_selector(dataset_mode='190k',
                              realfake190k_root_dir='/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset', 
                              train_batch_size=batch_size, eval_batch_size=64, ddp=False)
    elif dataset_mode == '200k':
        ds = Dataset_selector(dataset_mode='200k',
                              realfake200k_val_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv',
                              realfake200k_test_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv',
                              realfake200k_root_dir='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset',
                              realfake190k_root_dir='/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset',
                              train_batch_size=batch_size, eval_batch_size=64, ddp=False)

    train_loader = ds.loader_train

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            with torch.no_grad():
                teacher_logits, teacher_feats = teacher(images)
            
            student_logits, student_feats = student(images)
            
            base_loss = criterion(student_logits, labels)
            
            if kd_method == 'logits':
                kd_loss = logits_loss(teacher_logits, student_logits, T=4.0)
                loss = 0.6 * base_loss + 0.4 * kd_loss
            elif kd_method == 'at':
                kd_loss = at_loss(teacher_feats, student_feats)
                loss = base_loss + 0.4 * kd_loss
            elif kd_method == 'rkd':
                t_emb = teacher.model.avgpool(teacher_feats[-1]).flatten(1)
                s_emb = student.model.avgpool(student_feats[-1]).flatten(1)
                rkd_loss = RKDLoss().to(device)(t_emb, s_emb)
                loss = base_loss + 0.5 * rkd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(), f"student_{dataset_mode}_{kd_method}.pth")
    print(f"Student saved → student_{dataset_mode}_{kd_method}.pth")

# ====================== اجرا ======================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 140k → AT
    train_student(teacher_path="/kaggle/input/models/sara24h/teacher_model_best/pytorch/default/1/teacher_model_best.pth",
                  dataset_mode='140k',
                  kd_method='at',
                  epochs=30, lr=0.005, device=device)

    # 190k → Logits
    train_student(teacher_path="/kaggle/input/datasets/sara24h/kdfs-190k-transfer-learning-data/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth",
                  dataset_mode='190k',
                  kd_method='logits',
                  epochs=30, lr=0.005, device=device)

    # 200k → RKD
    train_student(teacher_path="/kaggle/input/datasets/sarah20079/teacher-model-best-200k/KDFS-Pearson-2/teacher_dir/teacher_model_best.pth",
                  dataset_mode='200k',
                  kd_method='rkd',
                  epochs=30, lr=0.005, device=device)
