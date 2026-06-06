import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# در ابتدای فایل
from torch.amp import GradScaler, autocast
from torchvision.models import resnet50, ResNet50_Weights

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
    def __init__(
        self,
        dataset_mode,
        realfake140k_train_csv=None,
        realfake140k_valid_csv=None,
        realfake140k_test_csv=None,
        realfake140k_root_dir=None,
        realfake200k_train_csv=None,
        realfake200k_val_csv=None,
        realfake200k_test_csv=None,
        realfake200k_root_dir=None,
        realfake190k_root_dir=None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
        pin_memory=True,
        ddp=False,
    ):
        if dataset_mode not in ['140k', '190k', '200k']:
            raise ValueError("dataset_mode must be  '140k', '190k', '200k'")
        self.dataset_mode = dataset_mode

        image_size = (256, 256) if dataset_mode in ['140k', '190k', '200k'] else (300, 300)

        if dataset_mode == '140k':
            mean = (0.5207, 0.4258, 0.3806)
            std = (0.2490, 0.2239, 0.2212)
        elif dataset_mode == '200k':
            mean = (0.4460, 0.3622, 0.3416)
            std = (0.2057, 0.1849, 0.1761)
        elif dataset_mode == '190k':
            mean = (0.4668, 0.3816, 0.3414)
            std = (0.2410, 0.2161, 0.2081)

        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        img_column = 'path' if dataset_mode in ['140k'] else 'images_id'

        root_dir = None
        train_data = val_data = test_data = None

        if dataset_mode == '140k':
            if not realfake140k_train_csv or not realfake140k_valid_csv or not realfake140k_test_csv or not realfake140k_root_dir:
                raise ValueError("realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, and realfake140k_root_dir must be provided")
            train_data = pd.read_csv(realfake140k_train_csv)
            val_data = pd.read_csv(realfake140k_valid_csv)
            test_data = pd.read_csv(realfake140k_test_csv)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')
            if 'path' not in train_data.columns:
                raise ValueError("CSV files for 140k dataset must contain a 'path' column")
            train_data = train_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=3407).reset_index(drop=True)

        elif dataset_mode == '200k':
            if not realfake200k_train_csv or not realfake200k_val_csv or not realfake200k_test_csv or not realfake200k_root_dir:
                raise ValueError("realfake200k_train_csv, realfake200k_val_csv, realfake200k_test_csv, and realfake200k_root_dir must be provided")

            train_data = pd.read_csv(realfake200k_train_csv)
            val_data = pd.read_csv(realfake200k_val_csv)
            test_data = pd.read_csv(realfake200k_test_csv)

            root_dir = realfake200k_root_dir  # /kaggle/input/undersampled-200k/balanced_unique_200k_dataset

            def create_image_path(row, split):
                folder = 'real' if row['label'] in [1, 'real', 'Real'] else 'fake'
                img_name = os.path.basename(row.get('filename_clean', row.get('filename', row.get('image', row.get('path', '')))))
                return os.path.join(split, folder, img_name)

            train_data['images_id'] = train_data.apply(lambda row: create_image_path(row, 'train'), axis=1)
            val_data['images_id'] = val_data.apply(lambda row: create_image_path(row, 'val'), axis=1)
            test_data['images_id'] = test_data.apply(lambda row: create_image_path(row, 'test'), axis=1)

        elif dataset_mode == '190k':
            if not realfake190k_root_dir:
                raise ValueError("realfake190k_root_dir must be provided")
            root_dir = realfake190k_root_dir
            def collect_images_from_folder(split):
                data = []
                for label in ['Real', 'Fake']:
                    folder_path = os.path.join(root_dir, split, label)
                    if not os.path.exists(folder_path):
                        raise FileNotFoundError(f"Folder not found: {folder_path}")
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(split, label, img_name)
                            data.append({'images_id': img_path, 'label': label})
                return pd.DataFrame(data)
            train_data = collect_images_from_folder('Train')
            val_data = collect_images_from_folder('Validation')
            test_data = collect_images_from_folder('Test')
            train_data = train_data.sample(frac=1, random_state=None).reset_index(drop=True)
            val_data = val_data.sample(frac=1, random_state=None).reset_index(drop=True)
            test_data = test_data.sample(frac=1, random_state=None).reset_index(drop=True)
            
        # Debug statistics
        print(f"{dataset_mode} dataset statistics:")
        print(f"Total train: {len(train_data)} | val: {len(val_data)} | test: {len(test_data)}")
        print(f"Train label distribution:\n{train_data['label'].value_counts()}")
        print(f"Val label distribution:\n{val_data['label'].value_counts()}")
        print(f"Test label distribution:\n{test_data['label'].value_counts()}")

        # Check missing images
        #for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    # استفاده از img_column به جای images_id
         #   missing = [os.path.join(root_dir, p) for p in data[img_column] if not os.path.exists(os.path.join(root_dir, p))]
          #  print(f"{split_name} missing images: {len(missing)}")
           # if missing:
            #    print(f"Sample missing: {missing[:3]}")

        # Create datasets
        train_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column)
        val_dataset = FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column)
        test_dataset = FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

       
        self.loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_val = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_test = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        print(f"DataLoaders ready - Train batches: {len(self.loader_train)}, Val: {len(self.loader_val)}, Test: {len(self.loader_test)}")

        # Test sample batch
        try:
            sample_train = next(iter(self.loader_train))
            print(f"Sample train batch shape: {sample_train[0].shape}, labels: {sample_train[1][:5]}")
        except Exception as e:
            print(f"Error in train loader: {e}")

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
        # در کلاس ResNetTeacher

        self.model = resnet50(weights=None) # یا ResNet50_Weights.DEFAULT اگر می‌خواهید وزن‌های پیش‌فرض داشته باشد
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook_fn)
        self.model.layer3[-1].register_forward_hook(hook_fn)
        self.model.layer4[-1].register_forward_hook(hook_fn)
        # در __init__ کلاس ResNetTeacher:
        self.model.eval() # اضافه کردن این خط

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
    
    # 1. تنظیمات اولیه
    teacher = ResNetTeacher().to(device)
    ckpt = torch.load(teacher_path, map_location=device)
    state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
    teacher.model.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = ResNetStudent().to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    # تعریف GradScaler برای Mixed Precision
    scaler = GradScaler()

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
                              realfake200k_train_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv',
                              realfake200k_root_dir='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset',
                              realfake190k_root_dir='/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset',
                              train_batch_size=batch_size, eval_batch_size=64, ddp=False)
                      
    train_loader = ds.loader_train

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        
        for i, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            scaler = GradScaler('cuda') # مشخص کردن دستگاه

# در حلقه آموزش (داخل باک autocast)
            with autocast('cuda'): 
                with torch.no_grad():
                    teacher_logits, teacher_feats = teacher(images)
                
                student_logits, student_feats = student(images)
                
                base_loss = criterion(student_logits, labels)
                
                # محاسبه Loss بر اساس متد انتخابی
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

            # 3. بهینه‌سازی با Scaler
            optimizer.zero_grad()
            # مقیاس‌بندی loss و انجام Backward
            scaler.scale(loss).backward()
            # استپ کردن optimizer ( scaler خودش unscale را انجام می‌دهد)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(), f"student_{dataset_mode}_{kd_method}_amp.pth")
    print(f"Model saved successfully.")

# ====================== اجرا ======================
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KD Training")
    parser.add_argument('--mode', type=str, required=True, help="140k, 190k, or 200k")
    parser.add_argument('--method', type=str, required=True, help="logits, at, or rkd")
    parser.add_argument('--path', type=str, required=True, help="Path to teacher pth")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_student(
        teacher_path=args.path, 
        dataset_mode=args.mode, 
        kd_method=args.method, 
        device=device
    )
