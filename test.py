import os
import json
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.optim as optim
from torch.optim import AdamW
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tqdm

# === AYARLAR ===
DATASET_DIR = 'C:/Users/Bilge/Downloads/dataset'
DATA_YAML = os.path.join(DATASET_DIR, 'data.yaml')
DATA_COCO_TRAIN = os.path.join(DATASET_DIR, 'train', 'annotations.coco.json')
DATA_COCO_VAL = os.path.join(DATASET_DIR, 'valid', 'annotations.coco.json')
RESULTS_DIR = './results_comparison'
os.makedirs(RESULTS_DIR, exist_ok=True)

# === COCOeval ile Metrikleri CSV'ye Yaz ===
def run_coco_eval(gt_json, pred_json, out_csv):
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(pred_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    metrics_df = pd.DataFrame([{
        'mAP@[IoU=0.5:0.95]': cocoEval.stats[0],
        'mAP@0.5': cocoEval.stats[1],
        'mAP@0.75': cocoEval.stats[2],
        'Recall@1': cocoEval.stats[6],
        'Recall@10': cocoEval.stats[7],
        'Recall@100': cocoEval.stats[8],
    }])
    metrics_df.to_csv(out_csv, index=False)
    print(f'Metrics saved to {out_csv}')

# === YOLOv8 (Eğitim + Değerlendirme) ===
def train_yolov8():
    print("Training YOLOv8...")
    model = YOLO('yolov8m.pt')
    model.train(data=DATA_YAML, epochs=25, imgsz=640, project=RESULTS_DIR, name='YOLOv8_run', save=True)
    print("Evaluating YOLOv8...")
    model = YOLO(os.path.join(RESULTS_DIR, "YOLOv8_run", "weights", "best.pt"))
    metrics = model.val()
    metrics_df = pd.DataFrame([{
        'mAP@0.5': metrics.box.map50,
        'mAP@0.75': metrics.box.map75,
        'mAP@[.5:.95]': metrics.box.map,
        'Precision': metrics.box.mp,
        'Recall': metrics.box.mr
    }])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'yolov8_metrics.csv'), index=False)
    print("YOLOv8 metrics saved.")

# === RetinaNet ===
def train_retinanet():
    print("Training RetinaNet...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = retinanet_resnet50_fpn(weights='DEFAULT').to(device).train()

    transform = T.Compose([T.ToTensor()])
    train_dataset = CocoDetection(os.path.join(DATASET_DIR, 'train/images'), DATA_COCO_TRAIN, transform=transform)
    val_dataset = CocoDetection(os.path.join(DATASET_DIR, 'valid/images'), DATA_COCO_VAL, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    num_epochs = 25
    best_map = 0
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        print(f"[RetinaNet] Epoch {epoch+1}/{num_epochs}")
        model.train()
        for images, targets in tqdm.tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Değerlendirme
        model.eval()
        results = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                for i, output in enumerate(outputs):
                    boxes = output['boxes'].cpu()
                    scores = output['scores'].cpu()
                    labels = output['labels'].cpu()
                    image_id = targets[i]['image_id'].item()
                    for box, score, label in zip(boxes, scores, labels):
                        x1, y1, x2, y2 = box.tolist()
                        results.append({
                            'image_id': image_id,
                            'category_id': label.item(),
                            'bbox': [x1, y1, x2 - x1, y2 - y1],
                            'score': score.item()
                        })
        pred_json = os.path.join(RESULTS_DIR, 'retinanet_predictions.json')
        with open(pred_json, 'w') as f:
            json.dump(results, f)
        run_coco_eval(DATA_COCO_VAL, pred_json, os.path.join(RESULTS_DIR, 'retinanet_metrics.csv'))
        current_map = pd.read_csv(os.path.join(RESULTS_DIR, 'retinanet_metrics.csv'))['mAP@[IoU=0.5:0.95]'][0]
        if current_map > best_map:
            best_map = current_map
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'retinanet_best.pth'))
            print(f"Yeni en iyi RetinaNet modeli kaydedildi (mAP={best_map:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Erken durdurma tetiklendi.")
                break

# === DETR ===
def train_detr():
    print("Training DETR...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).train()

    train_dataset = CocoDetection(os.path.join(DATASET_DIR, 'train/images'), DATA_COCO_TRAIN, transform=T.ToTensor())
    val_dataset = CocoDetection(os.path.join(DATASET_DIR, 'valid/images'), DATA_COCO_VAL, transform=T.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    num_epochs = 25
    best_map = 0
    patience = 5
    no_improve = 0

    for epoch in range(num_epochs):
        print(f"[DETR] Epoch {epoch+1}/{num_epochs}")
        model.train()
        for images, targets in tqdm.tqdm(train_loader):
            pixel_values = torch.stack([img for img in images]).to(device)
            outputs = model(pixel_values=pixel_values)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Değerlendirme
        model.eval()
        results = []
        with torch.no_grad():
            for images, targets in val_loader:
                pixel_values = torch.stack([img for img in images]).to(device)
                outputs = model(pixel_values=pixel_values)
                for i, logits in enumerate(outputs.logits):
                    probas = logits.softmax(-1)[..., :-1]
                    scores, labels = probas.max(-1)
                    boxes = outputs.pred_boxes[i].cpu()
                    image_id = targets[i]['image_id'].item()
                    for box, score, label in zip(boxes, scores.cpu(), labels.cpu()):
                        cx, cy, w, h = box.tolist()
                        x = cx - w / 2
                        y = cy - h / 2
                        results.append({
                            'image_id': image_id,
                            'category_id': label.item(),
                            'bbox': [x, y, w, h],
                            'score': score.item()
                        })
        pred_json = os.path.join(RESULTS_DIR, 'detr_predictions.json')
        with open(pred_json, 'w') as f:
            json.dump(results, f)
        run_coco_eval(DATA_COCO_VAL, pred_json, os.path.join(RESULTS_DIR, 'detr_metrics.csv'))
        current_map = pd.read_csv(os.path.join(RESULTS_DIR, 'detr_metrics.csv'))['mAP@[IoU=0.5:0.95]'][0]
        if current_map > best_map:
            best_map = current_map
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'detr_best.pth'))
            print(f"Yeni en iyi DETR modeli kaydedildi (mAP={best_map:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Erken durdurma tetiklendi.")
                break

# === GRAFİKLERİ OLUŞTUR VE KAYDET ===
def plot_metric_comparison():
    csv_paths = {
        "YOLOv8": os.path.join(RESULTS_DIR, "yolov8_metrics.csv"),
        "RetinaNet": os.path.join(RESULTS_DIR, "retinanet_metrics.csv"),
        "DETR": os.path.join(RESULTS_DIR, "detr_metrics.csv")
    }

    metrics = ['mAP@0.5', 'mAP@0.75', 'mAP@[.5:.95]', 'Precision', 'Recall']
    model_scores = {metric: [] for metric in metrics}
    models = []

    for model_name, csv_path in csv_paths.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            models.append(model_name)
            for metric in metrics:
                value = df.iloc[0].get(metric, None)
                model_scores[metric].append(value if pd.notna(value) else 0.0)

    for metric in metrics:
        plt.figure(figsize=(6, 4))
        plt.bar(models, model_scores[metric], color='skyblue')
        plt.title(f'{metric} Karşılaştırması')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.6)
        filename = f"{metric.replace('@', '').replace('[', '').replace(']', '').replace('.', '')}_comparison.png"
        plot_path = os.path.join(RESULTS_DIR, filename)
        plt.savefig(plot_path)
        plt.close()

# === ANA ===
if __name__ == "__main__":
    train_yolov8()
    train_retinanet()
    train_detr()
    plot_metric_comparison()
    print("\nTüm işlemler tamamlandı. Sonuçlar 'results_comparison/' klasörüné kaydedildi.")
