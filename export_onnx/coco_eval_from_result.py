
import argparse
import json
import os
import sys
import numpy as np
import datetime
import os
import time
import torch
from common import load_image, preprocess_prompt, to_numpy, postprocess_boxes, plot_boxes_to_image
from demo.test_ap_on_coco import CocoDetection, PostProcessCocoGrounding, load_model
import torch
from torch.utils.data import DataLoader
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
import groundingdino.datasets.transforms as T


def evaluate(args):
    transform = T.Compose(
        [
            T.RandomResize([(1280, 720)], max_size=1440),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    read_count = 0
    for i, (images, targets) in enumerate(data_loader):
        cocogrounding_res = {}
        # target["image_id"]: output for target, output in zip(targets, results)}

        for target in targets:
            image_id = target["image_id"]
            fname = f'idx{i}-im{image_id}.json'
            fp = os.path.join(args.out_dir, fname)
            if not os.path.isfile(fp):
                print(f'not found: {fp}')
                break
            with open(fp) as fi:
                jd = json.load(fi)[str(image_id)]
                jd = {key: torch.tensor(val) for key, val in jd.items()}
                cocogrounding_res[image_id] = jd
        if len(cocogrounding_res) > 0:
            evaluator.update(cocogrounding_res)
            read_count += 1
        else:
            break
    print(f'read det result for {read_count} images')
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    print(f'note that the above result is based on {read_count} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--anno_path", type=str,
                        required=False, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=False, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers for dataloader")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="if set, the det result will be saved to the target dir")

    args = parser.parse_args()
    evaluate(args)
