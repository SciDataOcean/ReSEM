import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.LISA_qwen import LISAQwenForCausalLM
# from model.LISA_qwsa import QWSAForCausalLM
from model.llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower, SigLipImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, EXPLANATORY_QUESTION_LIST)
import pdb
import random
from train_ds import validate
from tqdm import tqdm
import gc
import json

import traceback

random.seed(42)


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    print()
    pdb.pm()

sys.excepthook = info

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--weight", default="", type=str, required=False)
    parser.add_argument("--chat_json", default="chat_sample.json", type=str, required=False)
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    # 清理cuda cache
    torch.cuda.empty_cache()
    args = parse_args(args)
    args.vis_save_path = os.path.join(args.vis_save_path, args.version.split("/")[-1])
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = "[PAD]"  # 设置默认填充标记

    # 确保 pad_token 在词汇表中
    if tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})

    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.default_im_start_token_idx = tokenizer(DEFAULT_IM_START_TOKEN, add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    if "qwenvl" in args.version:
        model = QWSAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )
    elif "qwen" in args.version:
        model = LISAQwenForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
        )
    else:
        model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, default_im_start_token_idx=args.default_im_start_token_idx,  **kwargs
        )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if "qwenvl" in args.version:
        pass
    else:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
    


    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    if 'siglip' in model.config.vision_tower:
        clip_image_processor = SigLipVisionTower(model.config.vision_tower).image_processor
    else:
        clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
        
    transform = ResizeLongestSide(args.image_size)

    if args.weight != "":
        index_file = os.path.join(args.weight, "pytorch_model.bin.index.json")
        with open(index_file, "r", encoding="utf-8") as f:
            index_dict = json.load(f)
        weight_map = index_dict["weight_map"]  # dict
        shard_files = set(weight_map.values()) 

        for shard_file in tqdm(shard_files, desc="Loading shards into model"):
            shard_path = os.path.join(args.weight, shard_file)

            shard_state = torch.load(shard_path, map_location="cpu", weights_only=True)

            # 加载到 model（只写入本 shard 内的参数）
            model.load_state_dict(shard_state, strict=False)
            # 释放内存
            del shard_state
            gc.collect()
            torch.cuda.empty_cache()
    
    model.eval()

    def chat(prompt,image_path,class_name, answer='',return_similarity=True):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt + random.choice(EXPLANATORY_QUESTION_LIST)
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        image_name=image_path.split('.')[0].split('/')[-1]

        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            return
        
        if "tiff" in image_path:
            image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
            image_np=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        else:
            image_np = cv2.imread(image_path)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        # if input_ids.shape[1]==1:
        # import pdb; pdb.set_trace()
        output_ids, pred_masks, similarity_map = model.evaluate(
            image_clip, # torch.Size([1, 3, 224, 224])
            image, # torch.Size([1, 3, 1024, 1024])
            input_ids, # torch.Size([1, 76])
            resize_list, #[(689, 1024)]
            original_size_list, # [(3230, 4800)]
            max_new_tokens=512,
            tokenizer=tokenizer,
            return_similarity=return_similarity
        )
        if return_similarity:
            for i, sim_map in enumerate(similarity_map):
                sim_map = sim_map.detach().cpu().numpy()
                # 提取通道维度 [H, W, C] -> [H, W]
                sim_map = sim_map[0, ..., 0]
                # 归一化到0-1
                sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
                # 转换为0-255的单通道图
                sim_map_uint8 = (sim_map * 255).astype(np.uint8)
                # 应用热力图色彩映射 (Jet)
                sim_map_color = cv2.applyColorMap(sim_map_uint8, cv2.COLORMAP_JET)
                # 与原始图像混合：30% 原图 + 70% 热力图
                blended = image_np * 0.3 + sim_map_color * 0.7
                blended = blended.astype(np.uint8)
                blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

                save_path = "{}/{}_similarity_map_{}.jpg".format(
                    args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                )
                cv2.imwrite(save_path, blended)
                print("{} has been saved.".format(save_path))
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, class_name, i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_{}_masked_img_{}.jpg".format(
                args.vis_save_path,image_name, class_name, i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))
        return {"image": image_path, "prompt": prompt, "class": class_name, "answer": text_output.split('ASSISTANT: ')[-1]}
    sample_dict = json.load(open(args.chat_json))
    result_json=[]
    for i in range(len(sample_dict)):
        result=chat(sample_dict[i]["prompt"],sample_dict[i]["image"],sample_dict[i]["class"])
        result_json.append(result)
    
    result_save_path = os.path.join(args.vis_save_path,args.chat_json.split("/")[-1])
    with open(result_save_path,"w") as f:
        json.dump(result_json,f,indent=4)

if __name__ == "__main__":
    main(sys.argv[1:])
