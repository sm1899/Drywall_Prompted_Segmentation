import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union


def calculate_dice(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    if mask1.sum() + mask2.sum() == 0:
        return 1.0
    return 2 * intersection / (mask1.sum() + mask2.sum())


def process_dataset(images_dir, masks_dir, prompt, output_dir, model, processor, device, model_name=""):
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(images_path.glob('*.jpg')) + sorted(images_path.glob('*.png'))
    
    ious = []
    dices = []
    
    for img_file in tqdm(image_files, desc=model_name):
        mask_file = masks_path / (img_file.stem + '.png')
        if not mask_file.exists():
            continue
        
        img_orig = Image.open(img_file).convert('RGB')
        img = img_orig.resize((352, 352), Image.BILINEAR)
        gt_mask = Image.open(mask_file).convert('L')
        gt_mask = gt_mask.resize((352, 352), Image.NEAREST)
        gt_mask = np.array(gt_mask)
        gt_mask = (gt_mask > 127).astype(np.uint8)
        
        pixel_values = processor(images=[img], return_tensors="pt")['pixel_values'].to(device)
        text_inputs = processor.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'])
            logits = outputs.logits[0].cpu().numpy()
        
        pred_mask = torch.sigmoid(torch.from_numpy(logits)).numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
        
        iou = calculate_iou(gt_mask, pred_mask)
        dice = calculate_dice(gt_mask, pred_mask)
        
        pred_save_orig = Image.fromarray(pred_mask * 255, mode='L')
        if img_orig.size != (352, 352):
            pred_save_orig = pred_save_orig.resize((img_orig.size[0], img_orig.size[1]), Image.NEAREST)
        
        ious.append(iou)
        dices.append(dice)
        
        pred_save_orig.save(output_path / (img_file.stem + '.png'))
    
    return ious, dices


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    base_data_dir = Path('data')
    results_dir = Path('results')
    
    print("=" * 60)
    print("Zero-shot CLIPSeg (Pretrained)")
    print("=" * 60)
    
    model_pretrained = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model_pretrained.eval()
    
    cracks_dataset = base_data_dir / 'cracks.v1i.yolov5pytorch'
    if cracks_dataset.exists():
        print("\nProcessing Cracks dataset...")
        for split in ['train', 'valid', 'test']:
            images_dir = cracks_dataset / split / 'images'
            masks_dir = cracks_dataset / split / 'label_mask'
            
            if images_dir.exists() and masks_dir.exists():
                output_dir = results_dir / 'cracks_pretrained' / split
                ious, dices = process_dataset(
                    images_dir, masks_dir, "segment crack",
                    output_dir, model_pretrained, processor, device, f"Pretrained Cracks {split}"
                )
                
                if ious:
                    print(f"Cracks {split} - mIoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")
    
    drywall_dataset = base_data_dir / 'Drywall-Join-Detect.v2i.yolov5pytorch'
    if drywall_dataset.exists():
        print("\nProcessing Drywall dataset...")
        for split in ['train', 'valid']:
            images_dir = drywall_dataset / split / 'images'
            masks_dir = drywall_dataset / split / 'label_mask'
            
            if images_dir.exists() and masks_dir.exists():
                output_dir = results_dir / 'drywall_pretrained' / split
                ious, dices = process_dataset(
                    images_dir, masks_dir, "segment taping area",
                    output_dir, model_pretrained, processor, device, f"Pretrained Drywall {split}"
                )
                
                if ious:
                    print(f"Drywall {split} - mIoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")
    
    checkpoint_path = Path('checkpoints/best')
    if checkpoint_path.exists():
        print("\n" + "=" * 60)
        print("Fine-tuned CLIPSeg (Trained)")
        print("=" * 60)
        
        model_trained = CLIPSegForImageSegmentation.from_pretrained(str(checkpoint_path)).to(device)
        model_trained.eval()
        
        if cracks_dataset.exists():
            print("\nProcessing Cracks dataset...")
            for split in ['train', 'valid', 'test']:
                images_dir = cracks_dataset / split / 'images'
                masks_dir = cracks_dataset / split / 'label_mask'
                
                if images_dir.exists() and masks_dir.exists():
                    output_dir = results_dir / 'cracks_trained' / split
                    ious, dices = process_dataset(
                        images_dir, masks_dir, "segment crack",
                        output_dir, model_trained, processor, device, f"Trained Cracks {split}"
                    )
                    
                    if ious:
                        print(f"Cracks {split} - mIoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")
        
        if drywall_dataset.exists():
            print("\nProcessing Drywall dataset...")
            for split in ['train', 'valid']:
                images_dir = drywall_dataset / split / 'images'
                masks_dir = drywall_dataset / split / 'label_mask'
                
                if images_dir.exists() and masks_dir.exists():
                    output_dir = results_dir / 'drywall_trained' / split
                    ious, dices = process_dataset(
                        images_dir, masks_dir, "segment taping area",
                        output_dir, model_trained, processor, device, f"Trained Drywall {split}"
                    )
                    
                    if ious:
                        print(f"Drywall {split} - mIoU: {np.mean(ious):.4f}, Dice: {np.mean(dices):.4f}")
    else:
        print(f"\nCheckpoint not found at {checkpoint_path}")


if __name__ == '__main__':
    main()

