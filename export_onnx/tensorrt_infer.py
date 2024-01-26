import numpy as np
import datetime
import os

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_bytes,
)
from polygraphy.logger import G_LOGGER
import pathlib
from common import load_image, preprocess_prompt, to_numpy, postprocess_boxes, plot_boxes_to_image
import pdb


def convert_int64_to_int32(tensor: np.ndarray) -> np.ndarray:
    if tensor.dtype == np.int64:
        return tensor.astype(np.int32)
    return tensor


def set_pdb(args):
    if args.debug:
        pdb.set_trace()


def infer_single_image():
    '''
    "samples", "input_ids", "token_type_ids", "text_token_mask", "text_self_attention_masks", "position_ids"
    torch.Size([1, 3, 800, 1440])
    torch.Size([1, 5])
    torch.Size([1, 5])
    torch.Size([1, 5])
    torch.Size([1, 5, 5])
    torch.Size([1, 5])
    '''
    import argparse
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--engine_path", "-m", type=pathlib.Path,
                        required=True, help="path to config file")
    parser.add_argument("--image_path", "-i", type=pathlib.Path,
                        required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str,
                        required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=pathlib.Path, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float,
                        default=0.3, help="box threshold")
    parser.add_argument("--debug", type=int,
                        default=0, help="use debug mode")
    parser.add_argument("--text_threshold", type=float,
                        default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help="The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    args = parser.parse_args()

    with open(args.engine_path, "rb") as engine_file:
        engine = engine_from_bytes(engine_file.read())
    print("engine loaded")
    runner = TrtRunner(engine.create_execution_context())
    print("execution context created")
    with runner:
        image_raw, image_tensor = load_image(args.image_path)
        image_tensor = image_tensor.unsqueeze(0)

        prompt = args.text_prompt
        if not prompt.endswith('.'):
            prompt = prompt + '.'
        text_encoder_type = "bert-base-uncased"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
        inputs = preprocess_prompt(prompt=prompt, tokenizer=tokenizer)
        inputs["samples"] = image_tensor
        inputs = {name: to_numpy(value) for name, value in inputs.items()}
        # tensorrt doesn't support int64, so convert it to int32
        inputs = {name: convert_int64_to_int32(
            value) for name, value in inputs.items()}
        outputs = runner.infer(inputs)

        logits = outputs['logits']
        boxes = outputs['boxes']
        boxes_filt, pred_phrases = postprocess_boxes(logits, boxes, tokenizer, prompt,
                                                     box_threshold=args.box_threshold,
                                                     text_threshold=args.text_threshold,
                                                     token_spans=None)
        # visualize pred
        size = image_raw.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        set_pdb(args)
        image_with_box = plot_boxes_to_image(image_raw, pred_dict)[0]
        fname = os.path.basename(args.image_path)
        timestr = datetime.datetime.now().strftime("%m%d.%H%M%S")
        fname = os.path.join(args.output_dir, os.path.splitext(fname)[
                             0] + '.' + timestr + '.jpg')
        print(f'save result to {fname}. {len(boxes_filt)} objs detected')
        image_with_box.save(fname)


def test_ap_on_coco():
    from demo.test_ap_on_coco import CocoDetection, PostProcessCocoGrounding, load_model

    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    # parser.add_argument(
    #     "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    # )
    parser.add_argument("--engine_path", "-m", type=pathlib.Path,
                        required=True, help="path to config file")
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    with open(args.engine_path, "rb") as engine_file:
        engine = engine_from_bytes(engine_file.read())
    print("engine loaded")
    runner = TrtRunner(engine.create_execution_context())
    print("execution context created")
    with runner:
        # run inference
        start = time.time()
        gpu_computing_time = 0
        gpu_computing_post_time = 0
        for i, (images, targets) in enumerate(data_loader):
            # get images and captions
            images = images.tensors.to(args.device)
            bs = images.shape[0]
            input_captions = [caption] * bs

            # feed to the model
            t0 = time.time()
            outputs = model(images, captions=input_captions)
            t1 = time.time()

            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0).to(images.device)
            results = postprocessor(outputs, orig_target_sizes)
            t2 = time.time()

            cocogrounding_res = {
                target["image_id"]: output for target, output in zip(targets, results)}
            evaluator.update(cocogrounding_res)

            gpu_computing_time += (t1 - t0)
            gpu_computing_post_time += (t2 - t0)
            if (i+1) % 30 == 0:
                used_time = time.time() - start
                eta = len(data_loader) / (i+1e-5) * used_time - used_time
                print(
                    f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s. gpu_computing_postprocess_time={gpu_computing_post_time:.2f} gpu_computing_time={gpu_computing_time:.2f}s")

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

    with runner:
        image_raw, image_tensor = load_image(args.image_path)
        image_tensor = image_tensor.unsqueeze(0)

        prompt = args.text_prompt
        if not prompt.endswith('.'):
            prompt = prompt + '.'
        text_encoder_type = "bert-base-uncased"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
        inputs = preprocess_prompt(prompt=prompt, tokenizer=tokenizer)
        inputs["samples"] = image_tensor
        inputs = {name: to_numpy(value) for name, value in inputs.items()}
        # tensorrt doesn't support int64, so convert it to int32
        inputs = {name: convert_int64_to_int32(
            value) for name, value in inputs.items()}
        outputs = runner.infer(inputs)

        logits = outputs['logits']
        boxes = outputs['boxes']
        boxes_filt, pred_phrases = postprocess_boxes(logits, boxes, tokenizer, prompt,
                                                     box_threshold=args.box_threshold,
                                                     text_threshold=args.text_threshold,
                                                     token_spans=None)
        # visualize pred
        size = image_raw.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        set_pdb(args)
        image_with_box = plot_boxes_to_image(image_raw, pred_dict)[0]
        fname = os.path.basename(args.image_path)
        timestr = datetime.datetime.now().strftime("%m%d.%H%M%S")
        fname = os.path.join(args.output_dir, os.path.splitext(fname)[
                             0] + '.' + timestr + '.jpg')
        print(f'save result to {fname}. {len(boxes_filt)} objs detected')

    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    gpu_computing_time = 0
    gpu_computing_post_time = 0
    for i, (images, targets) in enumerate(data_loader):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        t0 = time.time()
        outputs = model(images, captions=input_captions)
        t1 = time.time()

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
        t2 = time.time()

        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        gpu_computing_time += (t1 - t0)
        gpu_computing_post_time += (t2 - t0)
        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s. gpu_computing_postprocess_time={gpu_computing_post_time:.2f} gpu_computing_time={gpu_computing_time:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == "__main__":
    # infer_single_image()
    test_ap_on_coco()
