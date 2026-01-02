import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, BatchSampler
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    result = {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    if 'dataset_id' in batch[0]:
        result['dataset_id'] = torch.tensor([item['dataset_id'] for item in batch])
    return result


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, prompt, target_size=320, augment=False, dataset_id=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.processor = processor
        self.prompt = prompt
        self.target_size = target_size
        self.augment = augment
        self.dataset_id = dataset_id
        
        image_files = sorted(self.images_dir.glob('*.jpg')) + sorted(self.images_dir.glob('*.png'))
        self.samples = []
        for img_file in image_files:
            mask_file = self.masks_dir / (img_file.stem + '.png')
            if mask_file.exists():
                self.samples.append((img_file, mask_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]
        img = Image.open(img_file).convert('RGB')
        mask = Image.open(mask_file).convert('L')
        
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = img.rotate(angle, fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, fillcolor=0)
            
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                img = color_jitter(img)
        
        pixel_values = self.processor(images=[img], return_tensors="pt")['pixel_values'].squeeze(0)
        text_inputs = self.processor.tokenizer(
            self.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        
        mask_array = np.array(mask)
        mask_tensor = torch.from_numpy((mask_array > 127).astype(np.float32))
        result = {
            'pixel_values': pixel_values,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'labels': mask_tensor
        }
        if self.dataset_id is not None:
            result['dataset_id'] = self.dataset_id
        return result


def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice


def bce_loss(pred_logits, target, pos_weight=1.0):
    return nn.functional.binary_cross_entropy_with_logits(
        pred_logits.view(-1), target.view(-1), pos_weight=torch.tensor(pos_weight, device=pred_logits.device)
    )


def train_epoch(model, dataloader, optimizer, device, cracks_weight=1.0, drywall_weight=2.0, bce_weight=1.0, dice_weight=1.0, cracks_pos_weight=2.0, drywall_pos_weight=2.0):
    model.train()
    total_loss = 0
    total_dice_loss = 0
    total_bce_loss = 0
    total_iou = 0
    total_dice = 0
    
    cracks_loss = 0
    cracks_dice_loss = 0
    cracks_bce_loss = 0
    cracks_iou = 0
    cracks_dice = 0
    cracks_count = 0
    
    drywall_loss = 0
    drywall_dice_loss = 0
    drywall_bce_loss = 0
    drywall_iou = 0
    drywall_dice = 0
    drywall_count = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        dataset_ids = batch.get('dataset_id', None)
        
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits.unsqueeze(1), size=labels.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(1)
        
        pred_probs = torch.sigmoid(logits)
        dice_loss_val = dice_loss(pred_probs, labels)
        
        if dataset_ids is not None:
            dataset_ids = dataset_ids.to(device)
            cracks_mask = (dataset_ids == 0)
            drywall_mask = (dataset_ids == 1)
            
            cracks_dice_batch = dice_loss(pred_probs[cracks_mask], labels[cracks_mask]) if cracks_mask.any() else torch.tensor(0.0, device=device)
            drywall_dice_batch = dice_loss(pred_probs[drywall_mask], labels[drywall_mask]) if drywall_mask.any() else torch.tensor(0.0, device=device)
            
            cracks_bce_batch = bce_loss(logits[cracks_mask], labels[cracks_mask], pos_weight=cracks_pos_weight) if cracks_mask.any() else torch.tensor(0.0, device=device)
            drywall_bce_batch = bce_loss(logits[drywall_mask], labels[drywall_mask], pos_weight=drywall_pos_weight) if drywall_mask.any() else torch.tensor(0.0, device=device)
            
            cracks_loss_batch = bce_weight * cracks_bce_batch + dice_weight * cracks_dice_batch
            drywall_loss_batch = bce_weight * drywall_bce_batch + dice_weight * drywall_dice_batch
            
            loss = cracks_weight * cracks_loss_batch + drywall_weight * drywall_loss_batch
            
            if cracks_mask.any():
                cracks_loss += cracks_loss_batch.item()
                cracks_bce_loss += cracks_bce_batch.item()
                cracks_dice_loss += cracks_dice_batch.item()
                total_bce_loss += cracks_bce_batch.item()
                pred_mask_cracks = (pred_probs[cracks_mask] > 0.5).float()
                labels_cracks = labels[cracks_mask]
                for i in range(pred_mask_cracks.shape[0]):
                    intersection = (pred_mask_cracks[i] * labels_cracks[i]).sum()
                    union = pred_mask_cracks[i].sum() + labels_cracks[i].sum() - intersection
                    iou = intersection / (union + 1e-8)
                    dice_score = 2 * intersection / (pred_mask_cracks[i].sum() + labels_cracks[i].sum() + 1e-8)
                    cracks_iou += iou.item()
                    cracks_dice += dice_score.item()
                    cracks_count += 1
            
            if drywall_mask.any():
                drywall_loss += drywall_loss_batch.item()
                drywall_bce_loss += drywall_bce_batch.item()
                drywall_dice_loss += drywall_dice_batch.item()
                total_bce_loss += drywall_bce_batch.item()
                pred_mask_drywall = (pred_probs[drywall_mask] > 0.5).float()
                labels_drywall = labels[drywall_mask]
                for i in range(pred_mask_drywall.shape[0]):
                    intersection = (pred_mask_drywall[i] * labels_drywall[i]).sum()
                    union = pred_mask_drywall[i].sum() + labels_drywall[i].sum() - intersection
                    iou = intersection / (union + 1e-8)
                    dice_score = 2 * intersection / (pred_mask_drywall[i].sum() + labels_drywall[i].sum() + 1e-8)
                    drywall_iou += iou.item()
                    drywall_dice += dice_score.item()
                    drywall_count += 1
        else:
            bce_loss_val = bce_loss(logits, labels, pos_weight=cracks_pos_weight)
            loss = bce_weight * bce_loss_val + dice_weight * dice_loss_val
            total_bce_loss += bce_loss_val.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_dice_loss += dice_loss_val.item()
        pred_mask = (pred_probs > 0.5).float()
        intersection = (pred_mask * labels).sum()
        union = pred_mask.sum() + labels.sum() - intersection
        iou = intersection / (union + 1e-8)
        dice_score = 2 * intersection / (pred_mask.sum() + labels.sum() + 1e-8)
        total_iou += iou.item()
        total_dice += dice_score.item()
    
    n = len(dataloader)
    result = {
        'total': (total_loss/n, total_dice_loss/n, total_bce_loss/n, total_iou/n, total_dice/n),
        'cracks': (cracks_loss/cracks_count if cracks_count > 0 else 0, cracks_dice_loss/cracks_count if cracks_count > 0 else 0, cracks_bce_loss/cracks_count if cracks_count > 0 else 0, cracks_iou/cracks_count if cracks_count > 0 else 0, cracks_dice/cracks_count if cracks_count > 0 else 0),
        'drywall': (drywall_loss/drywall_count if drywall_count > 0 else 0, drywall_dice_loss/drywall_count if drywall_count > 0 else 0, drywall_bce_loss/drywall_count if drywall_count > 0 else 0, drywall_iou/drywall_count if drywall_count > 0 else 0, drywall_dice/drywall_count if drywall_count > 0 else 0)
    }
    return result


def evaluate(model, dataloader, device, bce_weight=1.0, dice_weight=1.0, cracks_pos_weight=2.0, drywall_pos_weight=2.0):
    model.eval()
    total_loss = 0
    total_dice_loss = 0
    total_bce_loss = 0
    total_iou = 0
    total_dice = 0
    
    cracks_loss = 0
    cracks_dice_loss = 0
    cracks_bce_loss = 0
    cracks_iou = 0
    cracks_dice = 0
    cracks_count = 0
    
    drywall_loss = 0
    drywall_dice_loss = 0
    drywall_bce_loss = 0
    drywall_iou = 0
    drywall_dice = 0
    drywall_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            dataset_ids = batch.get('dataset_id', None)
            
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(1), size=labels.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(1)
            
            pred_probs = torch.sigmoid(logits)
            dice_loss_val = dice_loss(pred_probs, labels)
            
            if dataset_ids is not None:
                dataset_ids = dataset_ids.to(device)
                cracks_mask = (dataset_ids == 0)
                drywall_mask = (dataset_ids == 1)
                
                cracks_dice_batch = dice_loss(pred_probs[cracks_mask], labels[cracks_mask]) if cracks_mask.any() else torch.tensor(0.0, device=device)
                drywall_dice_batch = dice_loss(pred_probs[drywall_mask], labels[drywall_mask]) if drywall_mask.any() else torch.tensor(0.0, device=device)
                
                cracks_bce_batch = bce_loss(logits[cracks_mask], labels[cracks_mask], pos_weight=cracks_pos_weight) if cracks_mask.any() else torch.tensor(0.0, device=device)
                drywall_bce_batch = bce_loss(logits[drywall_mask], labels[drywall_mask], pos_weight=drywall_pos_weight) if drywall_mask.any() else torch.tensor(0.0, device=device)
                
                cracks_loss_batch = bce_weight * cracks_bce_batch + dice_weight * cracks_dice_batch
                drywall_loss_batch = bce_weight * drywall_bce_batch + dice_weight * drywall_dice_batch
                
                if cracks_mask.any():
                    cracks_loss += cracks_loss_batch.item()
                    cracks_bce_loss += cracks_bce_batch.item()
                    cracks_dice_loss += cracks_dice_batch.item()
                    pred_mask_cracks = (pred_probs[cracks_mask] > 0.5).float()
                    labels_cracks = labels[cracks_mask]
                    for i in range(pred_mask_cracks.shape[0]):
                        intersection = (pred_mask_cracks[i] * labels_cracks[i]).sum()
                        union = pred_mask_cracks[i].sum() + labels_cracks[i].sum() - intersection
                        iou = intersection / (union + 1e-8)
                        dice_score = 2 * intersection / (pred_mask_cracks[i].sum() + labels_cracks[i].sum() + 1e-8)
                        cracks_iou += iou.item()
                        cracks_dice += dice_score.item()
                        cracks_count += 1
                
                if drywall_mask.any():
                    drywall_loss += drywall_loss_batch.item()
                    drywall_bce_loss += drywall_bce_batch.item()
                    drywall_dice_loss += drywall_dice_batch.item()
                    total_bce_loss += drywall_bce_batch.item()
                    pred_mask_drywall = (pred_probs[drywall_mask] > 0.5).float()
                    labels_drywall = labels[drywall_mask]
                    for i in range(pred_mask_drywall.shape[0]):
                        intersection = (pred_mask_drywall[i] * labels_drywall[i]).sum()
                        union = pred_mask_drywall[i].sum() + labels_drywall[i].sum() - intersection
                        iou = intersection / (union + 1e-8)
                        dice_score = 2 * intersection / (pred_mask_drywall[i].sum() + labels_drywall[i].sum() + 1e-8)
                        drywall_iou += iou.item()
                        drywall_dice += dice_score.item()
                        drywall_count += 1
                
                loss = cracks_loss_batch + drywall_loss_batch
            else:
                bce_loss_val = bce_loss(logits, labels, pos_weight=cracks_pos_weight)
                loss = bce_weight * bce_loss_val + dice_weight * dice_loss_val
                total_bce_loss += bce_loss_val.item()
            
            total_loss += loss.item()
            total_dice_loss += dice_loss_val.item()
            pred_mask = (pred_probs > 0.5).float()
            intersection = (pred_mask * labels).sum()
            union = pred_mask.sum() + labels.sum() - intersection
            iou = intersection / (union + 1e-8)
            dice_score = 2 * intersection / (pred_mask.sum() + labels.sum() + 1e-8)
            total_iou += iou.item()
            total_dice += dice_score.item()
    
    n = len(dataloader)
    result = {
        'total': (total_loss/n, total_dice_loss/n, total_bce_loss/n, total_iou/n, total_dice/n),
        'cracks': (cracks_loss/cracks_count if cracks_count > 0 else 0, cracks_dice_loss/cracks_count if cracks_count > 0 else 0, cracks_bce_loss/cracks_count if cracks_count > 0 else 0, cracks_iou/cracks_count if cracks_count > 0 else 0, cracks_dice/cracks_count if cracks_count > 0 else 0),
        'drywall': (drywall_loss/drywall_count if drywall_count > 0 else 0, drywall_dice_loss/drywall_count if drywall_count > 0 else 0, drywall_bce_loss/drywall_count if drywall_count > 0 else 0, drywall_iou/drywall_count if drywall_count > 0 else 0, drywall_dice/drywall_count if drywall_count > 0 else 0)
    }
    return result


def plot_metrics(metrics, save_path):
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    axes[0, 0].plot(epochs, metrics['train_loss'], 'b-', label='Train', marker='o')
    axes[0, 0].plot(epochs, metrics['val_loss'], 'r-', label='Val', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, metrics['train_iou'], 'b-', label='Train', marker='o')
    axes[0, 1].plot(epochs, metrics['val_iou'], 'r-', label='Val', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Overall mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].plot(epochs, metrics['train_dice'], 'b-', label='Train', marker='o')
    axes[0, 2].plot(epochs, metrics['val_dice'], 'r-', label='Val', marker='s')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice Score')
    axes[0, 2].set_title('Overall Dice Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].plot(epochs, metrics['train_cracks_loss'], 'b-', label='Train', marker='o')
    axes[1, 0].plot(epochs, metrics['val_cracks_loss'], 'r-', label='Val', marker='s')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Cracks Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, metrics['train_cracks_iou'], 'b-', label='Train', marker='o')
    axes[1, 1].plot(epochs, metrics['val_cracks_iou'], 'r-', label='Val', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('mIoU')
    axes[1, 1].set_title('Cracks mIoU')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].plot(epochs, metrics['train_cracks_dice'], 'b-', label='Train', marker='o')
    axes[1, 2].plot(epochs, metrics['val_cracks_dice'], 'r-', label='Val', marker='s')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Dice Score')
    axes[1, 2].set_title('Cracks Dice Score')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    axes[2, 0].plot(epochs, metrics['train_drywall_loss'], 'b-', label='Train', marker='o')
    axes[2, 0].plot(epochs, metrics['val_drywall_loss'], 'r-', label='Val', marker='s')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Loss')
    axes[2, 0].set_title('Drywall Loss')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(epochs, metrics['train_drywall_iou'], 'b-', label='Train', marker='o')
    axes[2, 1].plot(epochs, metrics['val_drywall_iou'], 'r-', label='Val', marker='s')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('mIoU')
    axes[2, 1].set_title('Drywall mIoU')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    axes[2, 2].plot(epochs, metrics['train_drywall_dice'], 'b-', label='Train', marker='o')
    axes[2, 2].plot(epochs, metrics['val_drywall_dice'], 'r-', label='Val', marker='s')
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Dice Score')
    axes[2, 2].set_title('Drywall Dice Score')
    axes[2, 2].legend()
    axes[2, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 100
    num_workers = 2
    target_size = 352
    
    bce_weight = 1.0
    dice_weight = 1.0
    cracks_pos_weight = 2.0
    drywall_pos_weight = 20.0
    cracks_loss_weight = 1.0
    drywall_loss_weight = 2.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    
    for param in model.clip.parameters():
        param.requires_grad = False
    
    base_data_dir = Path('data')
    
    train_dataset_cracks = SegmentationDataset(
        base_data_dir / 'cracks.v1i.yolov5pytorch' / 'train' / 'images',
        base_data_dir / 'cracks.v1i.yolov5pytorch' / 'train' / 'label_mask',
        processor, "segment crack", target_size, augment=True, dataset_id=0
    )
    
    train_dataset_drywall = SegmentationDataset(
        base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch' / 'train' / 'images',
        base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch' / 'train' / 'label_mask',
        processor, "segment taping area", target_size, augment=True, dataset_id=1
    )
    
    print(f"Cracks dataset: {len(train_dataset_cracks)} samples")
    print(f"Drywall dataset: {len(train_dataset_drywall)} samples")
    
    val_dataset_cracks = SegmentationDataset(
        base_data_dir / 'cracks.v1i.yolov5pytorch' / 'valid' / 'images',
        base_data_dir / 'cracks.v1i.yolov5pytorch' / 'valid' / 'label_mask',
        processor, "segment crack", target_size, dataset_id=0
    )
    
    val_dataset_drywall = SegmentationDataset(
        base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch' / 'valid' / 'images',
        base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch' / 'valid' / 'label_mask',
        processor, "segment taping area", target_size, dataset_id=1
    )
    
    val_dataset = ConcatDataset([val_dataset_cracks, val_dataset_drywall])
    
    class BalancedBatchSampler:
        def __init__(self, dataset1, dataset2, batch_size, shuffle=True):
            self.dataset1 = dataset1
            self.dataset2 = dataset2
            self.batch_size = batch_size
            self.half_batch = batch_size // 2
            self.shuffle = shuffle
            self.len1 = len(dataset1)
            self.len2 = len(dataset2)
            self.min_len = min(self.len1, self.len2)
            self.num_batches = (self.min_len * 2) // batch_size
        
        def __iter__(self):
            indices1 = list(range(self.len1))
            indices2 = list(range(self.len2))
            if self.shuffle:
                random.shuffle(indices1)
                random.shuffle(indices2)
            
            for i in range(self.num_batches):
                start_idx1 = (i * self.half_batch) % self.len1
                start_idx2 = (i * self.half_batch) % self.len2
                
                batch_indices1 = indices1[start_idx1:start_idx1 + self.half_batch]
                batch_indices2 = indices2[start_idx2:start_idx2 + self.half_batch]
                
                if len(batch_indices1) < self.half_batch:
                    batch_indices1 += indices1[:self.half_batch - len(batch_indices1)]
                if len(batch_indices2) < self.half_batch:
                    batch_indices2 += indices2[:self.half_batch - len(batch_indices2)]
                
                combined_indices = []
                for idx in batch_indices1:
                    combined_indices.append(idx)
                for idx in batch_indices2:
                    combined_indices.append(self.len1 + idx)
                
                yield combined_indices
        
        def __len__(self):
            return self.num_batches
    
    train_sampler = BalancedBatchSampler(train_dataset_cracks, train_dataset_drywall, batch_size, shuffle=True)
    train_loader = DataLoader(ConcatDataset([train_dataset_cracks, train_dataset_drywall]), batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    metrics = {
        'train_loss': [],
        'train_dice_loss': [],
        'train_bce_loss': [],
        'train_iou': [],
        'train_dice': [],
        'train_cracks_loss': [],
        'train_cracks_bce_loss': [],
        'train_cracks_iou': [],
        'train_cracks_dice': [],
        'train_drywall_loss': [],
        'train_drywall_bce_loss': [],
        'train_drywall_iou': [],
        'train_drywall_dice': [],
        'val_loss': [],
        'val_dice_loss': [],
        'val_bce_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_cracks_loss': [],
        'val_cracks_bce_loss': [],
        'val_cracks_iou': [],
        'val_cracks_dice': [],
        'val_drywall_loss': [],
        'val_drywall_bce_loss': [],
        'val_drywall_iou': [],
        'val_drywall_dice': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_results = train_epoch(model, train_loader, optimizer, device, cracks_loss_weight, drywall_loss_weight, bce_weight, dice_weight, cracks_pos_weight, drywall_pos_weight)
        val_results = evaluate(model, val_loader, device, bce_weight, dice_weight, cracks_pos_weight, drywall_pos_weight)
        
        train_total = train_results['total']
        train_cracks = train_results['cracks']
        train_drywall = train_results['drywall']
        val_total = val_results['total']
        val_cracks = val_results['cracks']
        val_drywall = val_results['drywall']
        
        metrics['train_loss'].append(train_total[0])
        metrics['train_dice_loss'].append(train_total[1])
        metrics['train_bce_loss'].append(train_total[2])
        metrics['train_iou'].append(train_total[3])
        metrics['train_dice'].append(train_total[4])
        metrics['train_cracks_loss'].append(train_cracks[0])
        metrics['train_cracks_bce_loss'].append(train_cracks[2])
        metrics['train_cracks_iou'].append(train_cracks[3])
        metrics['train_cracks_dice'].append(train_cracks[4])
        metrics['train_drywall_loss'].append(train_drywall[0])
        metrics['train_drywall_bce_loss'].append(train_drywall[2])
        metrics['train_drywall_iou'].append(train_drywall[3])
        metrics['train_drywall_dice'].append(train_drywall[4])
        
        metrics['val_loss'].append(val_total[0])
        metrics['val_dice_loss'].append(val_total[1])
        metrics['val_bce_loss'].append(val_total[2])
        metrics['val_iou'].append(val_total[3])
        metrics['val_dice'].append(val_total[4])
        metrics['val_cracks_loss'].append(val_cracks[0])
        metrics['val_cracks_bce_loss'].append(val_cracks[2])
        metrics['val_cracks_iou'].append(val_cracks[3])
        metrics['val_cracks_dice'].append(val_cracks[4])
        metrics['val_drywall_loss'].append(val_drywall[0])
        metrics['val_drywall_bce_loss'].append(val_drywall[2])
        metrics['val_drywall_iou'].append(val_drywall[3])
        metrics['val_drywall_dice'].append(val_drywall[4])
        
        print(f"Train - Loss: {train_total[0]:.4f} (Dice: {train_total[1]:.4f}, BCE: {train_total[2]:.4f}), mIoU: {train_total[3]:.4f}, Dice: {train_total[4]:.4f}")
        print(f"  Cracks - mIoU: {train_cracks[3]:.4f}, Dice: {train_cracks[4]:.4f}")
        print(f"  Drywall - mIoU: {train_drywall[3]:.4f}, Dice: {train_drywall[4]:.4f}")
        print(f"Val - Loss: {val_total[0]:.4f} (Dice: {val_total[1]:.4f}, BCE: {val_total[2]:.4f}), mIoU: {val_total[3]:.4f}, Dice: {val_total[4]:.4f}")
        print(f"  Cracks - mIoU: {val_cracks[3]:.4f}, Dice: {val_cracks[4]:.4f}")
        print(f"  Drywall - mIoU: {val_drywall[3]:.4f}, Dice: {val_drywall[4]:.4f}")
        
        val_loss = val_total[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path("checkpoints").mkdir(exist_ok=True)
            model.save_pretrained("checkpoints/best")
            print(f"Saved best checkpoint (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        scheduler.step(val_loss)
        
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in val loss for {patience} epochs.")
            break
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    plot_metrics(metrics, 'training_curves.png')


if __name__ == '__main__':
    main()

