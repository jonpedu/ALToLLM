import os
from typing import List, Dict
from tqdm import tqdm
import re
import argparse
from eval.seg_dataset import MOVSegDataset
from eval.utils import AverageMeter, Summary
from eval.predict import Predictor

def init_trackers() -> Dict:
    return {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM),
    }

def print_dataset_results(dataset_name, trackers):
    intersection = trackers['intersection'].sum
    union = trackers['union'].sum
    miou = intersection / (union + 1e-10)
    print(f"{dataset_name} results:")
    print(f"cIoU: {miou:.4f}")
    print(f"gIoU: {trackers['gIoU'].avg:.4f}")

def evaluate_worker(predictor, dataset, batch_size):
    trackers = init_trackers()
    
    total_samples = len(dataset)
    total_samples = 100
    
    for batch_idx, idx in enumerate(tqdm(range(0, total_samples, batch_size), 
                    desc=f"Evaluating ...")):

        batch_end = min(idx + batch_size, total_samples)
        batch_samples = [dataset[i] for i in range(idx, batch_end)]

        mask_images = predictor.predict(batch_samples)

        mask_images = mask_images.float().cpu().numpy()
        predictor.update_metrics(mask_images, batch_samples, trackers)
    return trackers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/Multi-Class-OV')
    parser.add_argument('--image-dir', type=str, default='./data')
    parser.add_argument('--datasets', type=str, default='ade')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-num', type=int, default=1)
    parser.add_argument('--text-mode', type=str, default='all')
    args = parser.parse_args()

    predictor = Predictor(args.checkpoint, max_num=args.max_num)
    dataset_names = args.datasets.split(',')

    for dataset in dataset_names:

        ds = MOVSegDataset(data_path=os.path.join(args.data_dir, f"{dataset}.jsonl"), image_dir=args.image_dir, text_mode=args.text_mode)
        trackers = evaluate_worker(predictor, ds, args.batch_size)
        print_dataset_results(f"{dataset}", trackers)


if __name__ == "__main__":
    main()
