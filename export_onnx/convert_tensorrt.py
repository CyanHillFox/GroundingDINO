import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
import pathlib
import tensorrt.tensorrt as trt

import re
linear1_re = re.compile(R"/transformer/decoder/layers.\d+/linear1/.*")
linear2_re = re.compile(R"/transformer/decoder/layers.\d+/linear2/.*")
activation_re = re.compile(R"/transformer/decoder/layers.\d+/Relu")
def need_fp32(layer):
    if layer.name and any([regex.match(layer.name) for regex in [linear1_re, activation_re, linear2_re]]):
        # transformer ffn should be fp32
        return True
    # ipt = layer.get_input(0)
    # if ipt is not None \
    #     and ipt.dtype == trt.DataType.FLOAT \
    #     and layer.type not in [trt.LayerType.CAST]:
    #     return True
    return False

def main():
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
    parser.add_argument("--onnx_path", "-m", type=pathlib.Path, required=True, help="path to onnx model")
    parser.add_argument("--engine_path", "-o", type=pathlib.Path, required=True, help="path to output tensorrt engine file")
    parser.add_argument("--use_fp16", action="store_true", help="enable fp16 inference")
    parser.add_argument("--use_dla", action="store_true", help="enable DLA inference")
    args = parser.parse_args()


    profile = Profile()
    H, W = 720, 1280
    # the max_text_len in grounding-dino model is 256, not sure how we can relax the limitation.
    min_seq_len, max_seq_len = 1, 256 
    profile.add("samples", min=[1, 3, H, W], opt=[1, 3, H, W], max=[1, 3, H, W])
    profile.add("input_ids", min=[1, min_seq_len], opt=[1, max_seq_len], max=[1, max_seq_len])
    profile.add("token_type_ids", min=[1, min_seq_len], opt=[1, max_seq_len], max=[1, max_seq_len])
    profile.add("text_token_mask", min=[1, min_seq_len], opt=[1, max_seq_len], max=[1, max_seq_len])
    profile.add("text_self_attention_masks", min=[1, min_seq_len, min_seq_len], 
                opt=[1, max_seq_len, max_seq_len], max=[1, max_seq_len, max_seq_len])
    profile.add("position_ids", min=[1, min_seq_len], opt=[1, max_seq_len], max=[1, max_seq_len])
    profiles = [profile]
    # See examples/api/06_immediate_eval_api for details on immediately evaluated functional loaders like `engine_from_network`.
    # Note that we can freely mix lazy and immediately-evaluated loaders.
    builder, network, parser = network_from_onnx_path(str(args.onnx_path))
    # set certain layers to fp32
    for layer in network:
        if need_fp32(layer):
            print("setting {0} to fp32".format(layer.name))
            layer.precision = trt.DataType.FLOAT
    build_config = CreateConfig(profiles=profiles,
                                fp16=args.use_fp16,
                                precision_constraints="prefer",
                                use_dla=args.use_dla,
                                allow_gpu_fallback=True,
                                )
    print("building engine")
    engine = engine_from_network((builder, network, parser), config=build_config)
    print("engine built")
    # We'll save the engine so that we can inspect it with `inspect model`.
    # This should make it easy to see how the engine bindings are laid out.
    save_path = str(args.engine_path)
    save_engine(engine, save_path)
    print("tensorrt engine saved to {0}".format(save_path))

if __name__ == "__main__":
    main()