import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torchvision.models import resnet50

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ==========================================
# 1. Dataset & DataLoader (بدون تغییر)
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
        img_name = os.path.join(self.root_dir, self.data[self.img_column].iloc[idx])
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"image not found: {img_name}")
        image = Image.open(img_name).convert('RGB')
        label = self.label_map[self.data['label'].iloc[idx]]
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
            raise ValueError("dataset_mode must be  '140k', '190k', '200k'")
        self.dataset_mode = dataset_mode
        image_size = (256, 256)

        if dataset_mode == '140k':
            mean, std = (0.5207, 0.4258, 0.3806), (0.2490, 0.2239, 0.2212)
        elif dataset_mode == '200k':
            mean, std = (0.4460, 0.3622, 0.3416), (0.2057, 0.1849, 0.1761)
        elif dataset_mode == '190k':
            mean, std = (0.4668, 0.3816, 0.3414), (0.2410, 0.2161, 0.2081)

        transform_train = transforms.Compose([
            transforms.Resize(image_size), transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), transforms.Normalize(mean=mean, std=std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size), transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        img_column = 'path' if dataset_mode in ['140k'] else 'images_id'
        root_dir = train_data = val_data = test_data = None

        if dataset_mode == '140k':
            if not all([realfake140k_train_csv, realfake140k_valid_csv, realfake140k_test_csv, realfake140k_root_dir]): raise ValueError("140k paths missing")
            train_data = pd.read_csv(realfake140k_train_csv).sample(frac=1, random_state=3407).reset_index(drop=True)
            val_data = pd.read_csv(realfake140k_valid_csv).sample(frac=1, random_state=3407).reset_index(drop=True)
            test_data = pd.read_csv(realfake140k_test_csv).sample(frac=1, random_state=3407).reset_index(drop=True)
            root_dir = os.path.join(realfake140k_root_dir, 'real_vs_fake', 'real-vs-fake')

        elif dataset_mode == '200k':
            if not all([realfake200k_train_csv, realfake200k_val_csv, realfake200k_test_csv, realfake200k_root_dir]): raise ValueError("200k paths missing")
            train_data, val_data, test_data = pd.read_csv(realfake200k_train_csv), pd.read_csv(realfake200k_val_csv), pd.read_csv(realfake200k_test_csv)
            root_dir = realfake200k_root_dir
            def create_image_path(row, split):
                folder = 'real' if row['label'] in [1, 'real', 'Real'] else 'fake'
                img_name = os.path.basename(row.get('filename_clean', row.get('filename', row.get('image', row.get('path', '')))))
                return os.path.join(split, folder, img_name)
            train_data['images_id'], val_data['images_id'], test_data['images_id'] = train_data.apply(lambda r: create_image_path(r, 'train'), axis=1), val_data.apply(lambda r: create_image_path(r, 'val'), axis=1), test_data.apply(lambda r: create_image_path(r, 'test'), axis=1)

        elif dataset_mode == '190k':
            if not realfake190k_root_dir: raise ValueError("190k path missing")
            root_dir = realfake190k_root_dir
            def collect_images_from_folder(split):
                data = []
                for label in ['Real', 'Fake']:
                    folder_path = os.path.join(root_dir, split, label)
                    if not os.path.exists(folder_path): raise FileNotFoundError(f"Folder not found: {folder_path}")
                    for img_name in os.listdir(folder_path):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')): data.append({'images_id': os.path.join(split, label, img_name), 'label': label})
                return pd.DataFrame(data)
            train_data, val_data, test_data = collect_images_from_folder('Train').sample(frac=1).reset_index(drop=True), collect_images_from_folder('Validation').sample(frac=1).reset_index(drop=True), collect_images_from_folder('Test').sample(frac=1).reset_index(drop=True)

        train_dataset, val_dataset, test_dataset = FaceDataset(train_data, root_dir, transform=transform_train, img_column=img_column), FaceDataset(val_data, root_dir, transform=transform_test, img_column=img_column), FaceDataset(test_data, root_dir, transform=transform_test, img_column=img_column)

        self.train_sampler = None
        shuffle_train = True
        if ddp:
            self.train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            shuffle_train = False

        self.loader_train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=pin_memory, sampler=self.train_sampler)
        self.loader_val = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        self.loader_test = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        if rank == 0: print(f"DataLoaders ready - Train: {len(self.loader_train)}, Val: {len(self.loader_val)}, Test: {len(self.loader_test)}")

# ==========================================
# 2. KD Losses (بدون تغییر)
# ==========================================
def logits_loss(teacher_logits, student_logits):
    return F.mse_loss(teacher_logits, student_logits)

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
        self.alpha, self.beta = alpha, beta
    def forward(self, teacher_emb, student_emb):
        with torch.no_grad():
            t_dist, t_angle = self.pairwise_distance(teacher_emb), self.pairwise_angle(teacher_emb)
        s_dist, s_angle = self.pairwise_distance(student_emb), self.pairwise_angle(student_emb)
        return self.alpha * F.smooth_l1_loss(s_dist, t_dist) + self.beta * F.smooth_l1_loss(s_angle, t_angle)
    def pairwise_distance(self, x):
        x = F.normalize(x, p=2, dim=1); return torch.cdist(x, x, p=2)
    def pairwise_angle(self, x):
        x = F.normalize(x, p=2, dim=1); return torch.acos(torch.clamp(torch.mm(x, x.t()), -1.0 + 1e-7, 1.0 - 1e-7))

# ==========================================
# 3. Models (بدون تغییر)
# ==========================================
class ResNetTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        self.features = []
        def hook_fn(module, input, output): self.features.append(output)
        self.model.layer2[-1].register_forward_hook(hook_fn)
        self.model.layer3[-1].register_forward_hook(hook_fn)
        self.model.layer4[-1].register_forward_hook(hook_fn)
        self.model.eval()
    def forward(self, x):
        self.features.clear(); return self.model(x), self.features

def conv3x3(in_planes, out_planes, stride=1): return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1, self.bn1, self.relu = conv3x3(inplanes, planes, stride), nn.BatchNorm2d(planes), nn.ReLU(inplace=True)
        self.conv2, self.bn2 = conv3x3(planes, planes), nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes)) if stride != 1 or inplanes != planes else None
    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + identity)

class ResNet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 16
        self.conv1, self.bn1, self.relu = conv3x3(3, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16))
        self.layer2 = nn.Sequential(BasicBlock(16, 32, 2), BasicBlock(32, 32), BasicBlock(32, 32))
        self.layer3 = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 64), BasicBlock(64, 64))
        self.avgpool, self.fc = nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(64, 1)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x))); x = self.layer1(x); feat2 = self.layer2(x); feat3 = self.layer3(feat2)
        return self.fc(self.avgpool(feat3).view(feat3.size(0), -1))

class ResNetStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ResNet20()
        self.features = []
        def hook_fn(module, input, output): self.features.append(output)
        self.model.layer1.register_forward_hook(hook_fn)
        self.model.layer2.register_forward_hook(hook_fn)
        self.model.layer3.register_forward_hook(hook_fn)
    def forward(self, x):
        self.features.clear(); return self.model(x), self.features

# ==========================================
# 4. Training Function (اصلاح شده)
# ==========================================
def train_student(local_rank, teacher_path, dataset_mode, kd_method='logits',
                  epochs=200, lr=0.1, momentum=0.9, weight_decay=0.0005, batch_size=64):
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    teacher = ResNetTeacher().to(device)
    ckpt = torch.load(teacher_path, map_location=device)
    state = ckpt.get('state_dict') or ckpt.get('model') or ckpt
    teacher.model.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    student = ResNetStudent().to(device)
    student = DDP(student, device_ids=[local_rank])

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    
    # ---- اصلاح باگ اسکجولر ----
    # اگر ایپاک ها کمتر از 10 باشد، دیگر از MultiStepLR استفاده نمی‌کنیم تا LR صفر نشود
    if epochs > 10:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
        if local_rank == 0: print(f"\n[Info] LR milestones at epochs: {int(epochs*0.5)} and {int(epochs*0.75)}")
    else:
        scheduler = None
        if local_rank == 0: print(f"\n[Info] Epochs <= 10, Scheduler disabled to prevent LR crash.")

    criterion = nn.BCEWithLogitsLoss()
    
    # ---- تنظیمات فوق‌العاده پایدار GradScaler ----
    scaler = GradScaler(
        device='cuda',
        init_scale=1024,         # شروع ملایم
        growth_factor=2.0,       # تهاجمی بالا رفتن اسکیل
        backoff_factor=0.25,     # بسیار محافظه کارانه پایین آمدن اسکیل اگر NaN دید (پیش فرض 0.5 است)
        growth_interval=2000     # آپدیت هر 2000 بچ
    )
    
    rkd_criterion = RKDLoss().to(device) if kd_method == 'rkd' else None

    if dataset_mode == '140k':
        ds = Dataset_selector(dataset_mode='140k', realfake140k_train_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/train.csv', realfake140k_valid_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/valid.csv', realfake140k_test_csv='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces/test.csv', realfake140k_root_dir='/kaggle/input/datasets/xhlulu/140k-real-and-fake-faces', train_batch_size=batch_size, eval_batch_size=64, ddp=True, rank=local_rank, world_size=dist.get_world_size())
    elif dataset_mode == '190k':
        ds = Dataset_selector(dataset_mode='190k', realfake190k_root_dir='/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset', train_batch_size=batch_size, eval_batch_size=64, ddp=True, rank=local_rank, world_size=dist.get_world_size())
    elif dataset_mode == '200k':
        ds = Dataset_selector(dataset_mode='200k', realfake200k_val_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/val_labels.csv', realfake200k_test_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/test_labels.csv', realfake200k_train_csv='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset/train_labels.csv', realfake200k_root_dir='/kaggle/input/datasets/saraaskari/undersampled-200k/balanced_unique_200k_dataset', train_batch_size=batch_size, eval_batch_size=64, ddp=True, rank=local_rank, world_size=dist.get_world_size())
                              
    train_loader, train_sampler = ds.loader_train, ds.train_sampler

    for epoch in range(epochs):
        student.train()
        if train_sampler is not None: train_sampler.set_epoch(epoch)
        
        running_loss, running_corrects, total_samples = 0.0, 0, 0
        data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if local_rank == 0 else train_loader
        
        for i, (images, labels) in enumerate(data_iter):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            with autocast(device_type='cuda', dtype=torch.float16): 
                with torch.no_grad():
                    teacher_logits, teacher_feats = teacher(images)
                
                student_logits, student_feats = student(images)
                base_loss = criterion(student_logits, labels)
                
                if kd_method == 'logits': loss = base_loss + logits_loss(teacher_logits, student_logits)
                elif kd_method == 'at': loss = base_loss + at_loss(teacher_feats, student_feats)
                elif kd_method == 'rkd':
                    t_emb = teacher.model.avgpool(teacher_feats[-1]).flatten(1)
                    s_emb = student_feats[-1] if not isinstance(student_feats, list) else student.module.model.avgpool(student_feats[-1]).flatten(1)
                    loss = base_loss + rkd_criterion(t_emb, s_emb)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(student_logits) > 0.5).float()
                running_corrects += (preds == labels).sum().item()
                total_samples += labels.size(0)

            if local_rank == 0:
                data_iter.set_postfix({'Loss': f'{running_loss/(i+1):.4f}', 'Acc': f'{(running_corrects/total_samples)*100:.2f}%', 'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'})

        if scheduler is not None:
            scheduler.step()
        if local_rank == 0: print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {(running_corrects/total_samples)*100:.2f}%")

    # ==========================================
    # 5. Test Evaluation
    # ==========================================
    student.eval()
    test_corrects = 0
    test_total = 0
    
    with torch.no_grad():
        test_iter = tqdm(ds.loader_test, desc="Evaluating Test Set") if local_rank == 0 else ds.loader_test
        
        for images, labels in test_iter:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = student(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
            test_corrects += (preds == labels).sum().item()
            test_total += labels.size(0)

    if dist.is_initialized():
        tensor_corrects = torch.tensor(test_corrects, dtype=torch.float64, device=device)
        tensor_total = torch.tensor(test_total, dtype=torch.float64, device=device)
        
        dist.all_reduce(tensor_corrects, op=dist.ReduceOp.SUM)
        dist.all_reduce(tensor_total, op=dist.ReduceOp.SUM)
        
        test_corrects = tensor_corrects.item()
        test_total = tensor_total.item()

    if local_rank == 0:
        test_acc = (test_corrects / test_total) * 100
        print("\n" + "="*50)
        print(f"==> Final Test Accuracy: {test_acc:.2f}%")
        print("="*50 + "\n")

        torch.save(student.module.state_dict(), f"student_{dataset_mode}_{kd_method}_amp_ddp.pth")
        print(f"Model saved successfully.")

# ==========================================
# 6. Exec
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KD Training with DDP")
    parser.add_argument('--mode', type=str, required=True, help="140k, 190k, or 200k")
    parser.add_argument('--method', type=str, required=True, help="logits, at, or rkd")
    parser.add_argument('--path', type=str, required=True, help="Path to teacher pth")
    parser.add_argument('--epochs', type=int, default=200, help="Total epochs")
    parser.add_argument('--lr', type=float, default=0.1, help="Initial learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay")
    args = parser.parse_args()
    
    dist.init_process_group(backend="gloo") 
    local_rank = int(os.environ["LOCAL_RANK"])
    
    train_student(
        local_rank=local_rank, teacher_path=args.path, dataset_mode=args.mode, 
        kd_method=args.method, epochs=args.epochs, lr=args.lr, 
        momentum=args.momentum, weight_decay=args.weight_decay
    )
    
    dist.destroy_process_group()
