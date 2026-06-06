import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet20   # ← اضافه شد
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# ====================== Dataset Classes ======================
# (FaceDataset و Dataset_selector را کامل کپی کنید)

# ====================== KD Losses ======================
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

# ====================== Models ======================
class ResNetTeacher(nn.Module):
    """معلم: ResNet-50"""
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


import torch
import torch.nn as nn
import torch.nn.functional as F

# کلاس پایه برای ResNet-20 (مخصوص کارهای KD و سبک‌سازی)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
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
        feat2 = self.layer2(x) # Hook point
        feat3 = self.layer3(feat2) # Hook point
        x = self.avgpool(feat3)
        return self.fc(x.view(x.size(0), -1))


class ResNetStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet20() # استفاده از مدل بالا
        
        # ثبت هوک‌ها
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output)

        # در کلاس ResNetStudent:
        self.model.layer1.register_forward_hook(hook_fn) 
        self.model.layer2.register_forward_hook(hook_fn)
        self.model.layer3.register_forward_hook(hook_fn)

    def forward(self, x):
        self.features.clear()
        logit = self.model(x)
        return logit, self.features

# ====================== Training Function ======================
def train_student(teacher_path, dataset_mode, kd_method='logits',
                  epochs=30, lr=0.005, device='cuda', batch_size=64):
    
    # Teacher (ResNet-50)
    teacher = ResNetTeacher().to(device)
    ckpt = torch.load(teacher_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
        teacher.model.load_state_dict(state, strict=False)
    else:
        teacher.model.load_state_dict(ckpt, strict=False)
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student (ResNet-20)
    student = ResNetStudent().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Dataset (توصیه: همه روی 200k)
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

            if epoch == 0 and i == 0:
                print(f"\n--- Debugging Feature Dimensions ---")
                print(f"Teacher feats shapes: {[f.shape for f in teacher_feats]}")
                print(f"Student feats shapes: {[f.shape for f in student_feats]}")
                print(f"-------------------------------------\n")
            
            base_loss = criterion(student_logits, labels)
            
            if kd_method == 'logits':
                kd_loss = logits_loss(teacher_logits, student_logits, T=4.0)
                loss = 0.6 * base_loss + 0.4 * kd_loss
            elif kd_method == 'at':
                kd_loss = at_loss(teacher_feats, student_feats)
                loss = base_loss + 0.4 * kd_loss
            elif kd_method == 'rkd':
                t_emb = teacher.model.avgpool(teacher_feats[-1]).flatten(1) if hasattr(teacher.model, 'avgpool') else teacher_feats[-1].mean([2,3]).flatten(1)
                s_emb = student.model.avgpool(student_feats[-1]).flatten(1) if hasattr(student.model, 'avgpool') else student_feats[-1].mean([2,3]).flatten(1)
                rkd_loss = RKDLoss().to(device)(t_emb, s_emb)
                loss = base_loss + 0.5 * rkd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(), f"baseline_student_{dataset_mode}_{kd_method}.pth")
    print(f"Saved: baseline_student_{dataset_mode}_{kd_method}.pth")

# ====================== اجرا ======================
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KD Training")
    parser.add_argument('--mode', type=str, required=True, help="140k, 190k, or 200k")
    parser.add_argument('--method', type=str, required=True, help="logits, at, or rkd")
    parser.add_argument('--path', type=str, required=True, help="Path to teacher pth")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_student(teacher_path=args.path, 
                  dataset_mode=args.mode, 
                  kd_method=args.method, 
                  device=device)
