# 应该不需要initialize lisa model了
import torch.distributed as dist
import builtins
import os
# 保存原始 print
def print_rank0(*args, **kwargs):
    if int(os.getenv('RANK', '0')) == 0:  # 通过环境变量判断
        print(*args, **kwargs)



print_rank0("test 1")

import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from transformers import BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.LISA import LISAForCausalLM
# from model.LISA_qwen import LISAQwenForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn_grpo, ValDataset_EM
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX, 
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, evaluate_text_metrics)
import pdb
import traceback
import torch.nn as nn
import random

from model.grpo.data_parallel_ds import train_with_grpo, train_with_grpo_epoch
from model.grpo.utils import optimize_model_memory
from model.grpo.reward import combined_reward
import copy


def info(type, value, tb):
    traceback.print_exception(type, value, tb)
    print()
    pdb.pm()

sys.excepthook = info


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_lora_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                x not in name
                for x in [
                    "visual_model",
                    "vision_tower",
                    "mm_projector",
                    "text_hidden_fcs",
                ]
            )
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def normalize_checkpoint_key(key):
    prefixes = ("module.",)
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def compatible_load_state_dict(src, dst):
    dst_state = dst.state_dict()
    normalized = {}
    loaded = 0
    skipped = 0

    for key, value in src.items():
        key = normalize_checkpoint_key(key)
        candidates = [
            key,
            key[len("base_model."):] if key.startswith("base_model.") else None,
            "base_model." + key if not key.startswith("base_model.") else None,
            "base_model.model." + key if not key.startswith("base_model.model.") else None,
        ]
        matched_key = None
        for candidate in candidates:
            if candidate is None:
                continue
            if candidate in dst_state and dst_state[candidate].shape == value.shape:
                matched_key = candidate
                break
        if matched_key is None:
            skipped += 1
            continue
        normalized[matched_key] = value
        loaded += 1

    missing, unexpected = dst.load_state_dict(normalized, strict=False)
    print_rank0(
        "compatible checkpoint load: loaded {}, skipped {}, missing {}, unexpected {}".format(
            loaded, skipped, len(missing), len(unexpected)
        )
    )


def parse_tuple(input_str):
    try:
        # 去除括号和空格，分割成列表，再转为整数元组
        tuple_result=tuple(map(int, input_str.strip("()").split(',')))
        print_rank0(tuple_result)
        return tuple_result
    except:
        raise argparse.ArgumentTypeError("Tuple must be in format '(a,b,c,d)'")


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
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
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--use_gpt_qa", action="store_true", default=False)
    parser.add_argument("--train_from_scratch", action="store_true", default=False)
    parser.add_argument('--train_mask_decoder_only', action='store_true', default=False)
    parser.add_argument('--full_finetune', action='store_true', default=False)
    parser.add_argument('--full_from_scratch', action='store_true', default=False)
    parser.add_argument('--score_text', action='store_true', default=False)
    parser.add_argument('--lora_module_full_finetune', action='store_true', default=False)
    parser.add_argument("--num_generations", default=4, type=int)
    parser.add_argument("--weight", default="", type=str, required=False)
    # reward_weight default is (1,1,10,0)
    parser.add_argument(
        "--reward_weights",
        nargs=4,  # 要求4个值
        type=float,
        default=[1, 1, 10, 1],
        help="Reward weights as 4 integers (default: 1 1 10 1)"
    )
    return parser.parse_args(args)


def main(args):
    torch.cuda.empty_cache()
    args = parse_args(args)
    set_random_seed(args.seed)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    os.makedirs(args.vis_save_path, exist_ok=True)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

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

    kwargs = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "torch_dtype": torch_dtype
    }
    # if args.load_in_4bit:
    #     kwargs.update(
    #         {
    #             "torch_dtype": torch.half,
    #             "load_in_4bit": True,
    #             "quantization_config": BitsAndBytesConfig(
    #                 load_in_4bit=True,
    #                 bnb_4bit_compute_dtype=torch.float16,
    #                 bnb_4bit_use_double_quant=True,
    #                 bnb_4bit_quant_type="nf4",
    #                 llm_int8_skip_modules=["visual_model"],
    #             ),
    #         }
    #     )
    # elif args.load_in_8bit:
    #     kwargs.update(
    #         {
    #             "torch_dtype": torch.half,
    #             "quantization_config": BitsAndBytesConfig(
    #                 llm_int8_skip_modules=["visual_model"],
    #                 load_in_8bit=True,
    #             ),
    #         }
    #     )
    if "qwen" in args.version:
        model = LISAQwenForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    else:
        kwargs["default_im_start_token_idx"] = args.default_im_start_token_idx
        model = LISAForCausalLM.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    # if args.precision == "bf16":
    #     model = model.bfloat16().cuda()
    # elif (
    #     args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    # ):
    #     vision_tower = model.get_model().get_vision_tower()
    #     model.model.vision_tower = None
    #     model_engine = deepspeed.init_inference(
    #         model=model,
    #         dtype=torch.half,
    #         replace_with_kernel_inject=True,
    #         replace_method="auto",
    #     )
    #     model = model_engine.module
    #     model.model.vision_tower = vision_tower.half().cuda()
    # elif args.precision == "fp32":
    #     model = model.float().cuda()

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r if not args.train_mask_decoder_only else 0
    if args.full_finetune or args.full_from_scratch or args.eval_only or args.lora_module_full_finetune:
        lora_r = 0
    if lora_r > 0:
        lora_target_modules = find_lora_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        if args.local_rank == 0:
            model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    
    if args.train_mask_decoder_only:
        trainable_params=["mask_decoder"]
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_params
                ]
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False
    elif args.lora_module_full_finetune:
        # 找出所有需要full finetune的 之前用lora训练的模块
        full_finetune_module_names = set()
        cls = torch.nn.Linear
        for name, module in model.named_modules():
            if (
                isinstance(module, cls)
                and all(
                    [
                        x not in name
                        for x in [
                            "visual_model",
                            "vision_tower",
                            "mm_projector",
                            "text_hidden_fcs",
                        ]
                    ]
                )
                and any([x in name for x in args.lora_target_modules.split(",")])
            ):
                full_finetune_module_names.add(name)
    
        for n, p in model.named_parameters():
            is_target = any([module_name in n for module_name in full_finetune_module_names])
            if is_target:
                p.requires_grad = True
        # 之前的训练模块
        trainable_params=["lm_head", "embed_tokens", "text_hidden_fcs"]
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_params
                ]
            ):
                p.requires_grad = True
    elif args.full_finetune:
        for n, p in model.named_parameters():
            p.requires_grad = True
    elif args.full_from_scratch:
        # 只在rank0打印
        print_rank0('initialize all params')
        for n, p in model.named_parameters():
            p.requires_grad = True
            # initialize
            if len(p.shape) > 1:  # 如果是权重矩阵
                nn.init.kaiming_uniform_(p, nonlinearity='relu')  # 或者使用 nn.init.kaiming_normal_
            else:  # 如果是偏置
                nn.init.zeros_(p)  # 偏置通常初始化为零，但你可以根据需要选择其他方法

    else:
        trainable_params=["lm_head", "embed_tokens", "text_hidden_fcs"]
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in trainable_params
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    print_rank0("##########")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print_rank0(name)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
        use_gpt_qa=args.use_gpt_qa,
    )

    if args.no_eval == False:
        if args.val_dataset == "reason_seg_em":
            val_dataset = ValDataset_EM(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.reason_seg_data+"_val",
                args.image_size,
                use_gpt_qa=args.use_gpt_qa,
            )
        else:
            val_dataset = ValDataset(
                args.dataset_dir,
                tokenizer,
                args.vision_tower,
                args.val_dataset,
                args.image_size,
            )
        print_rank0(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print_rank0(f"Training with {len(train_dataset)} examples.")
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "allgather_bucket_size": 2e8,
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True
        }
    }
    model_engine, _ , train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn_grpo,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    train_iter = iter(train_loader)
    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model_grpo")
        if os.path.exists(resume):
            args.resume = resume
        else:
            resume = os.path.join(args.log_dir, "ckpt_model")
            if os.path.exists(resume):
                args.resume = resume

    if args.resume:
        is_grpo_resume = "grpo" in args.resume
        if is_grpo_resume:
            load_path, client_state = model_engine.load_checkpoint(args.resume)
        else:
            load_path, client_state = model_engine.load_checkpoint(
                args.resume,
                load_module_strict=False,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_only=True,
                custom_load_fn=compatible_load_state_dict,
            )
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        if is_grpo_resume:
            args.start_epoch = (
                int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            )
        else:
            args.start_epoch = 0
        print_rank0(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )
    

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_grpo,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )


    if False:
        if args.score_text:
            # import pdb; pdb.set_trace()
            giou, ciou, text_metrics = validate_text(val_loader, model_engine, 0, writer, tokenizer, args)
            # print(giou,ciou,text_metrics)
        else:
            giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        best_score=giou
        cur_ciou=ciou
    else:
        best_score=0.5929
        cur_ciou=0.6460
    torch.cuda.empty_cache()
    print_rank0("\nStarting RL fine-tuning using GRPO...")
    # This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
    training_config = {
        'num_iterations': args.epochs,
        'num_steps': args.steps_per_epoch,
        'batch_size': args.batch_size, # reduce if you have fewer GPUs
        'num_generations': args.num_generations, # reduce if you have GPUs with less VRAM
        'max_completion_length': args.model_max_length, # reduce if you have GPUs with less VRAM
        'beta': 0.04,
        'learning_rate': args.lr,
        'mu': 1,
        'epsilon': 0.1
    }
    num_gpus = torch.cuda.device_count()

    dtype = next(model_engine.module.parameters()).dtype
    # ------------------------------------------------
    # 构造 reference model —— 深拷贝训练模型并冻结
    # ------------------------------------------------
    ref_base_model = copy.deepcopy(model_engine.module).cpu().eval()
    for p in ref_base_model.parameters():
        p.requires_grad = False

    ref_infer_engine = deepspeed.init_inference(
        model=ref_base_model,
        mp_size=1,
        dtype=dtype,
        replace_with_kernel_inject=False     # ← 禁止 kernel 注入
    )
    
    # ------------------------------------------------
    # 构造 policy 推理模型 —— 参数与训练 engine 同步，但不参与梯度
    # ------------------------------------------------
    policy_base_model = copy.deepcopy(model_engine.module).cpu().eval()
    for p in policy_base_model.parameters():
        p.requires_grad = False

    policy_infer_engine = deepspeed.init_inference(
        model=policy_base_model,
        mp_size=1,
        dtype=dtype,
        replace_with_kernel_inject=False     # ← 禁止 kernel 注入
    )

    for epoch in range(args.start_epoch, args.epochs):
        print_rank0("Epoch: ", epoch)
        model_engine = train_with_grpo_epoch(
            ref_infer_engine=ref_infer_engine,
            policy_infer_engine=policy_infer_engine,
            policy_base_model=policy_base_model,
            model_engine=model_engine,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            train_iter=train_iter,
            reward_function=partial(combined_reward, weight=args.reward_weights),
            device_ids=list(range(num_gpus)) if num_gpus > 1 else None,
            **training_config
        )
        if args.no_eval == False:
            giou, ciou = validate(val_loader, model, 0, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou
                
        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model_grpo")
            if args.local_rank == 0:
                torch.save({},
                    os.path.join(
                        args.log_dir,
                            "meta_log_epoch{}_giou{:.3f}_ciou{:.3f}.pth".format(
                            epoch, best_score, cur_ciou
                        ),
                    ))
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
                




def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    # pdb.set_trace()
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print_rank0("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


def validate_text(val_loader, model_engine, epoch, writer, tokenizer, args):
    '''
    加入text相关指标
    '''
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    bleu_meter=AverageMeter("Bleu", ":6.3f", Summary.SUM)
    cider_meter=AverageMeter("CIDEr", ":6.3f", Summary.SUM)
    bertscorep_meter=AverageMeter("BERTScore_P", ":6.3f", Summary.SUM)
    bertscorer_meter=AverageMeter("BERTScore_R", ":6.3f", Summary.SUM)
    bertscoref1_meter=AverageMeter("BERTScore_F1", ":6.3f", Summary.SUM)


    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        
        for i in range(len(input_dict["input_ids"])):
            
            with torch.no_grad():
                output_ids, pred_masks = model_engine.module.evaluate(
                    input_dict["images_clip"],
                    input_dict["images"],
                    input_dict["input_ids"][i].unsqueeze(0),
                    input_dict["resize_list"],
                    # input_dict["resize_list"],
                    [(input_dict["masks_list"][0].shape[1], input_dict["masks_list"][0].shape[2])],
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
            
            mask_i = input_dict["masks_list"][0][i].int()
            output_i = (pred_masks[0][0] > 0).int()
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ").replace('<unk>', '')
            text_output = text_output.split('ASSISTANT: ')[-1]
            text_output_gt = input_dict["conversation_list"][i].split('ASSISTANT: ')[-1] # todo
            
            
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            intersection, union, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            acc_iou = intersection / (union + 1e-5) 
            acc_iou[union == 0] += 1.0 
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            acc_iou_meter.update(acc_iou, n=1)

            # import pdb; pdb.set_trace()
            text_metrics=evaluate_text_metrics(candidate=text_output, reference=text_output_gt)
            bleu_meter.update(text_metrics['BLEU'])
            cider_meter.update(text_metrics['CIDEr'])
            bertscorep_meter.update(text_metrics['BERTScore_P'])
            bertscorer_meter.update(text_metrics['BERTScore_R'])
            bertscoref1_meter.update(text_metrics['BERTScore_F1'])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    bleu_meter.all_reduce()
    cider_meter.all_reduce()
    bertscorep_meter.all_reduce()
    bertscorer_meter.all_reduce()
    bertscoref1_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    bleu= bleu_meter.avg
    cider = cider_meter.avg
    bertscorep = bertscorep_meter.avg
    bertscorer = bertscorer_meter.avg
    bertscoref1 = bertscoref1_meter.avg

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/bleu", bleu, epoch)
        writer.add_scalar("val/cider", cider, epoch)
        writer.add_scalar("val/bertscorep", bertscorep, epoch)
        writer.add_scalar("val/bertscorer", bertscorer, epoch)
        writer.add_scalar("val/bertscoref1", bertscoref1, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))
        print("bleu: {:.4f}, cider: {:.4f}".format(bleu, cider))
        print("bert score p: {:.4f}, r: {:.4f}, f1: {:.4f}".format(bertscorep, bertscorer, bertscoref1))

    return giou, ciou, {'bleu': bleu, 'cider': cider, 'bert_score_p': bertscorep, 'bert_score_r': bertscorer, 'bert_score_f1': bertscoref1}

if __name__ == "__main__":
    main(sys.argv[1:])
