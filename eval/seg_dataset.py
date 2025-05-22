import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from pycocotools import mask
from PIL import Image, ImageOps
from glob import glob
import cv2

class ReferSegDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        image_dir,
        refer_seg_data="refcoco||refcoco+||refcocog||grefcoco",
        split="val",
        text_mode='all'
    ):
        self.image_dir = image_dir
        self.text_mode = text_mode
        if refer_seg_data == "refcocom":
            self.data = json.load(open(os.path.join(dataset_dir, f"annotations/{split}.json"), "r"))
            self.mask_path_template=os.path.join(dataset_dir,'masks/{}.png')
        else:
            self.data = json.load(open(os.path.join(dataset_dir, f"{refer_seg_data}/{refer_seg_data}_{split}.json"), "r"))
        if refer_seg_data == "grefcoco":
            self.data = [item for item in self.data if len(item['instruction'])>0]
        self.refer_seg_data = refer_seg_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]    
        if self.refer_seg_data == "refcocom":
            image_path=os.path.join(self.image_dir,item['img_name'])
            mask_path=self.mask_path_template.format(item['segment_id'])
            sentences=item['sentences']
            m_final=(np.array(Image.open(mask_path).convert("L"))/255.0).astype(np.uint8)
        else:
            image_info = item['image_info']
            m_final = np.zeros(
                            (image_info["height"], image_info["width"])
                        ).astype(np.uint8)
            instruction = item['instruction']
            sentences = [instruction[j]['sent'] for j in range(len(instruction))]
            if len(sentences) != 0:
                for ann in item['anns']:
                    if len(ann["segmentation"]) == 0:
                        m = np.zeros(
                            (image_info["height"], image_info["width"])
                        ).astype(np.uint8)
                    else:
                        if type(ann["segmentation"]) == list:  # polygon
                            rle = mask.frPyObjects(
                                ann["segmentation"], image_info["height"], image_info["width"], )
                        else:
                            rle = ann["segmentation"]
                            # 处理counts为列表的情况
                            if isinstance(rle["counts"], list):
                                # 将counts列表转换为bytes格式
                                rle = mask.frPyObjects(
                                    [rle], image_info["height"], image_info["width"]
                                )
                            elif not isinstance(rle["counts"], bytes):
                                rle["counts"] = rle["counts"].encode()
                        m = mask.decode(rle)
                        m = np.sum(
                            m, axis=2
                        )  # sometimes there are multiple binary map (corresponding to multiple segs)
                        m = m.astype(np.uint8)  # convert to np.uint8
                    m_final = m_final | m

            image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        sentence = "Meet all the descriptions: "
        for i, sent in enumerate(sentences):
            sentence += f"{i+1}. {sent}. "
        if self.text_mode == 'adaptive':
            prompt = "Segment <ref>{}</ref> by adaptive length.".format(sentence)
        else:
            prompt = "Segment <ref>{}</ref>.".format(sentence)

        data_dict = {
            "image":image,
            "mask":m_final,
            "prompt":prompt
        }
        if self.refer_seg_data == "grefcoco":
            pre_prompt = "Does the image contain <ref>{}</ref>?\nAnswer \"yes\" or \"no\" directly.".format(sentence)
            data_dict["pre_prompt"] = pre_prompt
        return data_dict

class MOVSegDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_dir,
        text_mode='all',
    ):
        with open(data_path, "r") as f:
            lines = f.readlines()
        self.data = [json.loads(line) for line in lines]

        self.image_dir = image_dir
        self.text_mode = text_mode
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image_path = os.path.join(self.image_dir, item["image"])
        text = ", ".join(item["description"])
        if self.text_mode == 'adaptive':
            prompt = "Segment <ref>{}</ref> by adaptive length.".format(text)
        else:
            prompt = "Segment <ref>{}</ref>.".format(text)
        data_dict = {
            "image":Image.open(image_path).convert("RGB"),
            "prompt":prompt,
            "mask":(np.array(Image.open(os.path.join(self.image_dir, item["mask"])).convert("L"))/255.0).astype(np.uint8),
        }
        return data_dict