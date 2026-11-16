"""
Fruit Detection Dataset Loader
Supports loading bounding boxes in standard YOLO format (.txt) or Pascal VOC format (.xml).
Converts annotations to PyTorch torchvision detection format:
  boxes: Tensor[N, 4] in (xmin, ymin, xmax, ymax) absolute pixel coordinates
  labels: Tensor[N] with class IDs (1-indexed; 0 is reserved for background)
"""

import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class FruitBoundingBoxDataset(Dataset):
    def __init__(self, root_dir, split="train", transforms=None, label_map=None):
        """
        root_dir should contain:
          fruit_dataset/
            images/
              train/
              val/
            labels/ (or annotations/)
              train/
              val/
        """
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # Default label map for 2 fruits (can be expanded)
        self.label_map = label_map or {
            "fruit_1": 1,
            "fruit_2": 2,
            "apple": 1,
            "banana": 2
        }

        self.img_dir = os.path.join(root_dir, "images", split)
        self.lbl_dir = os.path.join(root_dir, "labels", split)

        # Collect image files
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = []
        if os.path.exists(self.img_dir):
            for ext in valid_exts:
                self.image_paths.extend(glob.glob(os.path.join(self.img_dir, ext)))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def _parse_yolo_label(self, txt_path, img_w, img_h):
        boxes = []
        labels = []
        if not os.path.exists(txt_path):
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0]) + 1  # 1-indexed for PyTorch detection
                    xc = float(parts[1]) * img_w
                    yc = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h

                    xmin = max(0.0, xc - w / 2.0)
                    ymin = max(0.0, yc - h / 2.0)
                    xmax = min(float(img_w), xc + w / 2.0)
                    ymax = min(float(img_h), yc + h / 2.0)

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def _parse_voc_label(self, xml_path):
        boxes = []
        labels = []
        if not os.path.exists(xml_path):
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            cls_id = self.label_map.get(name, 1)

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Check for YOLO .txt or VOC .xml label
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        yolo_path = os.path.join(self.lbl_dir, f"{base_name}.txt")
        voc_path = os.path.join(self.lbl_dir, f"{base_name}.xml")

        if os.path.exists(yolo_path):
            boxes, labels = self._parse_yolo_label(yolo_path, w, h)
        elif os.path.exists(voc_path):
            boxes, labels = self._parse_voc_label(voc_path)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        # Basic tensor transform
        img_tensor = TF.to_tensor(img)

        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))
