import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as maskUtils

MAX_NORMALIZED_COORDINATE = 999

def xywh2xyxy(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

def resize_if_exceeds_max_size(image, mask, box, max_size):
    """Resize image, mask and box if either dimension exceeds max_size"""
    if max(image.shape[:2]) <= max_size:
        return image, mask, box
    scale = max_size / max(image.shape[:2])
    new_size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    resized_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_CUBIC)
    resized_box = [round(coord * scale) for coord in box]
    return resized_image, resized_mask, resized_box

def make_square(image, mask, box, is_random_pad=True):
    """Make image and mask square by padding"""
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    max_img_size = max(img_h, img_w)
    if is_random_pad:
        pad_left = random.randint(0, max_img_size - img_w)
        pad_top = random.randint(0, max_img_size - img_h)
    else:
        pad_left = 0
        pad_top = 0
    square_image = np.zeros((max_img_size, max_img_size, 3), dtype=image.dtype)
    square_image[pad_top:pad_top+img_h, pad_left:pad_left+img_w] = image
    mask_scale_w = mask_w / img_w
    mask_scale_h = mask_h / img_h
    max_mask_size = int(max_img_size * max(mask_scale_w, mask_scale_h))
    mask_pad_left = int(pad_left * mask_scale_w)
    mask_pad_top = int(pad_top * mask_scale_h)
    square_mask = np.zeros((max_mask_size, max_mask_size), dtype=mask.dtype)
    square_mask[mask_pad_top:mask_pad_top+mask_h, mask_pad_left:mask_pad_left+mask_w] = mask
    adjusted_box = [box[0] + pad_left, box[1] + pad_top, box[2] + pad_left, box[3] + pad_top]
    return square_image, square_mask, adjusted_box

def get_crop_box(orig_box, height, width):
    """Calculate a square crop box that contains the original box"""
    box_w = orig_box[2] - orig_box[0]
    box_h = orig_box[3] - orig_box[1]
    box_area = box_w * box_h
    center_x = (orig_box[0] + orig_box[2]) // 2
    center_y = (orig_box[1] + orig_box[3]) // 2
    target_area = box_area * random.uniform(4, 50)
    crop_size = int(np.sqrt(target_area))
    min_size = max(box_w, box_h) * 2
    crop_size = max(crop_size, min_size)
    crop_x1 = center_x - crop_size // 2
    crop_y1 = center_y - crop_size // 2
    if crop_x1 < 0:
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y1 = 0
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size
    if crop_x2 > width:
        crop_x1 = max(0, width - crop_size)
        crop_x2 = width
    if crop_y2 > height:
        crop_y1 = max(0, height - crop_size)
        crop_y2 = height
    return [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)]

def random_crop(image, mask, box):
    height, width = image.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    assert abs(mask_height/mask_width - height/width) < 1e-3
    crop_box = get_crop_box(box, height, width)
    cropped_img = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    box2 = [box[0]-crop_box[0], box[1]-crop_box[1], box[2]-crop_box[0], box[3]-crop_box[1]]
    mask_crop_box = [round(crop_box[0]*mask_width/width), round(crop_box[1]*mask_height/height), 
                     round(crop_box[2]*mask_width/width), round(crop_box[3]*mask_height/height)]
    cropped_mask = mask[mask_crop_box[1]:mask_crop_box[3], mask_crop_box[0]:mask_crop_box[2]]
    return cropped_img, cropped_mask, box2

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        config,
        is_train=True,
        fixed_size=1024,
        mask_size=256,
        max_normalized_coordinate=999,
        min_area_ratio=0.005,
    ):
        super(SupervisedDataset, self).__init__()
        data_path = config.dataset.params.train_sam1b_path if is_train else config.dataset.params.eval_sam1b_path   
        self.is_train = is_train
        self.max_normalized_coordinate = max_normalized_coordinate
        self.min_area_ratio = min_area_ratio
        self.mask_size = mask_size
        self.max_image_size = fixed_size
        self.square_pad = True

        # Load data list (assume jsonl or json list of file paths)
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line.strip()) for line in f]
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        if self.is_train:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Load sample meta
        sample = self.data[index]
        json_path = sample["json_path"] if "json_path" in sample else sample
        json_data = json.load(open(json_path, 'r'))
        image_path = os.path.join(os.path.dirname(json_path), json_data["image"]["file_name"])
        width, height = json_data["image"]["width"], json_data["image"]["height"]
        raw_anns = json_data["annotations"]
        min_area = max(height, width) ** 2 * self.min_area_ratio
        raw_anns = [ann for ann in raw_anns if ann['area'] > min_area]
        if len(raw_anns) == 0:
            return self.__getitem__(random.randint(0, len(self.data)-1))
        max_sample_n = min(20, max(3, len(raw_anns)//2))
        sample_n = random.randint(1, min(max_sample_n, len(raw_anns))) if self.is_train else min(index % max_sample_n + 1, len(raw_anns))
        anns = random.sample(raw_anns, sample_n) if self.is_train else raw_anns[:sample_n]
        image = np.array(Image.open(image_path))
        masks = np.zeros((height, width), dtype=np.uint8)
        boxes = None
        mask_count = 0
        for ann in anns:
            mask = maskUtils.decode(ann['segmentation']) * 255
            current_box_xywh = ann['bbox']
            current_box_xyxy = xywh2xyxy(current_box_xywh)
            if boxes is None:
                boxes = current_box_xyxy
            else:
                boxes[0] = min(boxes[0], current_box_xyxy[0])
                boxes[1] = min(boxes[1], current_box_xyxy[1])
                boxes[2] = max(boxes[2], current_box_xyxy[2])
                boxes[3] = max(boxes[3], current_box_xyxy[3])
            masks = masks | mask
            mask_count += 1
        box = boxes
        mask = masks
        if self.is_train:
            image, mask, box = random_crop(image, mask, box)
        image, mask, box = resize_if_exceeds_max_size(image, mask, box, self.max_image_size)
        if self.square_pad:
            image, mask, box = make_square(image, mask, box, is_random_pad=self.is_train)
        # Resize image to self.max_image_size
        if image.shape[1] != self.max_image_size:
            image = cv2.resize(image, (self.max_image_size, self.max_image_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        mask = cv2.resize(mask, (self.mask_size, self.mask_size))
        mask = torch.from_numpy(mask).float() / 255
        mask = mask.unsqueeze(0)
        mask = mask.repeat(3, 1, 1)
        return {
            "image_src": image,
            "mask": mask,
            "filename": f"{os.path.basename(json_path)}_{index}",
            "mask_count": mask_count
        }