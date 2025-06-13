# imports
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import cv2

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, split) :

        self.data_dir = data_dir
        self.split = split

        # open the configuration
        annotation_path = os.path.join(self.data_dir, self.split, "_annotations.coco.json")
        with open(annotation_path, "r") as f :
            self.annotations = json.load(f)

        self.annotations_ = self.useful_json()
        self.dir = os.path.join(data_dir, split)

        print(f"Loaded split {split} with length : {len(self.annotations_)}")


    def useful_json(self) :
        useful_ = []

        # hash map
        img_map = {img["id"]: img for img in self.annotations["images"]}

        for ann in self.annotations["annotations"] :
            img_id = ann["image_id"]
            if img_id in img_map:
                img_info = img_map[img_id]
                use = {
                    "id": img_id,
                    "file_name": img_info["file_name"],
                    "category_id": ann["category_id"] - 1,
                    "bbox": ann["bbox"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                }
                useful_.append(use)

        return useful_


    def __getitem__(self, index):
        ann = self.annotations_[index]
        img, mask = self.draw_mask(index)

        mask = torch.from_numpy(mask).long()

        if hasattr(self, "transforms"):
            img = self.transforms(img)

        return {
            "image": img, # (1, H, W)
            "mask": mask,
            "category_id": ann["category_id"],
            "file_name": ann["file_name"],
        }


    def __len__(self):
        return len(self.annotations_)

    def draw_mask(self, index):
        img_info = self.annotations_[index]
        img = Image.open(os.path.join(self.dir, img_info["file_name"])).convert("L")
        category_id = img_info["category_id"]
        width, height = img.size

        mask = np.zeros((height, width), dtype=np.uint8)

        if not img_info["segmentation"]:
            return img, mask

        segmentations = img_info["segmentation"]
        for segmentation in segmentations:
            if len(segmentation) % 2 != 0:
                continue

            polygon = np.array(segmentation).reshape(-1, 2)
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon], color=category_id)

        return img, mask

    def visualize(self, index) :
        img, mask = self.draw_mask(index=index)
        img_info = self.annotations_[index]
        img_np = np.array(img)

        fig, ax = plt.subplots(1,3, figsize = (15,5))

        # plot the original image
        ax[0].imshow(img)
        ax[0].set_title(f"Image: {img_info['file_name']}")
        ax[0].axis("off")

        # plot the mask
        ax[1].imshow(mask, cmap = "gray")
        ax[1].set_title(f"Segmentation Mask : Category : {img_info['category_id']}")
        ax[1].axis("off")

        colored_mask = np.zeros_like(img_np)
        colored_mask[mask == 1] = [255, 0, 0]
        overlay = cv2.addWeighted(img_np, 1, colored_mask, 1, 0)

        ax[2].imshow(overlay)
        ax[2].set_title("Overlay")
        ax[2].axis("off")


        plt.tight_layout()
        plt.show()

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transforms = transform

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory = torch.cuda.is_available()
            )
