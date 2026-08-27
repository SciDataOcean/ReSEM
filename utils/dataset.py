import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

from model.llava import conversation as conversation_lib
from model.qwen import conversation as conversation_lib_qwen
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from model.qwen.mm_utils import tokenizer_image_token_qwen
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .reason_seg_dataset_em import ReasonSegDataset_EM
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST, COT_ANSWER_LIST)
from .vqa_dataset import VQADataset
import json
from PIL import Image
import pdb

def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    classes_list = []
    prompts_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    if len(batch[0])==12:
        for (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            classes, # classes
            prompts, # prompts
            inference,
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            images_clip_list.append(images_clip)
            conversation_list.extend(conversations)
            label_list.append(label)
            masks_list.append(masks.float())
            resize_list.append(resize)
            questions_list.append(questions)
            sampled_classes_list.append(sampled_classes)
            classes_list.append(classes)
            prompts_list.extend(prompts)
            cnt += len(conversations)
            offset_list.append(cnt)
            inferences.append(inference)
    else:
        for (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
            inference,
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            images_clip_list.append(images_clip)
            conversation_list.extend(conversations)
            label_list.append(label)
            masks_list.append(masks.float())
            resize_list.append(resize)
            questions_list.append(questions)
            sampled_classes_list.append(sampled_classes)
            cnt += len(conversations)
            offset_list.append(cnt)
            inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    prompt_ids=[
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in prompts_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    prompt_ids = torch.nn.utils.rnn.pad_sequence(
        prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    attention_masks_prompts=prompt_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    elif conv_type == "qwen_2":
        sep = conv.roles[1]
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1 if conv_type == "llava_v1" else 0 # warning
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou, sep)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            prompt_ids = prompt_ids[:, :truncate_len]
            attention_masks_prompts=attention_masks_prompts[:, :truncate_len]


    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "classes_list": classes_list,
        "prompt_ids": prompt_ids,
        "attention_masks_prompts": attention_masks_prompts,
    
    }

def collate_fn_grpo(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, precision='bf16'
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    classes_list = []
    prompts_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        classes,
        prompts,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        classes_list.append(classes)
        prompts_list.extend(prompts)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    prompt_ids=[
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in prompts_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    prompt_ids = torch.nn.utils.rnn.pad_sequence(
        prompt_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    attention_masks_prompts=prompt_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    elif conv_type == "qwen_2":
        sep = conv.roles[1]
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1 if conv_type == "llava_v1" else 0 # warning
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou, sep, conv.sep2, conv_type)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
            prompt_ids = prompt_ids[:, :truncate_len]
            attention_masks_prompts=attention_masks_prompts[:, :truncate_len]

    processed_batch= {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        ### add classes_list for the new dataset
        "classes_list": classes_list,
        "prompt_ids": prompt_ids,
        "attention_masks_prompts": attention_masks_prompts,
    }

    if precision == "fp16":
        for key, value in processed_batch.items():
            if isinstance(value, torch.Tensor):
                # 对于浮点数张量转换为 fp16
                if value.dtype == torch.float32:
                    processed_batch[key] = value.half()
                # 对于长整型等其他类型保持不变
                elif value.dtype == torch.int64 or value.dtype == torch.int32:
                    processed_batch[key] = value
    elif precision == "bf16":
        for key, value in processed_batch.items():
            if isinstance(value, torch.Tensor):
                # 对于浮点数张量转换为 bf16
                if value.dtype == torch.float32:
                    processed_batch[key] = value.bfloat16()
                elif value.dtype == torch.int64 or value.dtype == torch.int32:
                    processed_batch[key] = value
    
    return processed_batch

'''
def collate_fn_qwen(
    batch, tokenizer=None, conv_type="qwen", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = "<|image_pad|>"
            replace_token = "<|vision_start|>" + replace_token + "<|vision_end|>"
            conversation_list[i] = conversation_list[i].replace("<image>", replace_token)

    input_ids = [
        tokenizer_image_token_qwen(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib_qwen.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    elif conv_type == "qwen":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) # 计算了目标张量中非 pad_token_id 的有效长度
        rounds = conversation.split(conv.sep2) # 将对话拆分为轮次
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX # 将目标张量的起始部分标记为 IGNORE_INDEX，忽略起始标记的损失。
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (parts, rou)
            parts[0] += sep

            if "<|image_pad|>" in conversation:
                round_len = len(tokenizer_image_token_qwen(rou, tokenizer))
                instruction_len = len(tokenizer_image_token_qwen(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )
        # pdb.set_trace()
        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len, (cur_len, total_len) # AssertionError: (116, 117)

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }
'''
class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        explanatory=0.1,
        use_gpt_qa=False,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )
            elif dataset == "reason_seg_em":
                self.all_datasets.append(
                    ReasonSegDataset_EM(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data+"_train",
                        explanatory=-1,
                        use_gpt_qa=use_gpt_qa,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(self.base_image_dir, ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        base_image_dir, "images/saiapr_tc-12", item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        base_image_dir,
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} Please output segmentation mask.".format(text),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n What is {} in this image? Please output segmentation mask.".format(
                        text
                    ),
                )
                conv.append_message(conv.roles[1], "[SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )


class ValDataset_EM(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([140, 140, 140]).view(-1, 1, 1)
    pixel_std = torch.Tensor([40, 40, 40]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset="organelle||plantorgan||cremi_val",
        image_size=1024,
        explanatory=-1,
        use_gpt_qa=False
    ):
        self.base_image_dir = base_image_dir
        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.cot_answer_list= COT_ANSWER_LIST
        self.explanatory = explanatory

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        if "clip" in vision_tower:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        elif "siglip" in vision_tower:
            self.clip_image_processor = SigLipVisionTower(vision_tower).image_processor
        reason_seg_data,splits=val_dataset.split("_")[0], val_dataset.split("_")[1]
        reason_seg_data=reason_seg_data.replace("material","ceramic||defect||micronet||nanoparticle")
        reason_seg_data_ls = reason_seg_data.split("||")
        # prepare self.img_to_explanation for all datasets
        self.json_data_list = []
        for data in reason_seg_data_ls:
            if use_gpt_qa:
                gpt_qa_suffix = os.getenv("RESEM_VAL_GPT_QA_SUFFIX", "d_qa")
                json_path=os.path.join(base_image_dir,  data, f"{splits}_{gpt_qa_suffix}.json")
            else:
                json_path=os.path.join(base_image_dir,  data, f"{splits}.json")
            json_data=json.load(open(json_path)) if os.path.exists(json_path) else []
            for item in json_data:
                item['image_path']=os.path.join(base_image_dir, data, item['image_name'])
                item['data_root']=os.path.join(base_image_dir, data)
            self.json_data_list.extend(json_data)
        self.use_gpt_qa = use_gpt_qa
        self.explanatory_question_list = EXPLANATORY_QUESTION_LIST


    def __len__(self):
        return len(self.json_data_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # images, jsons = self.reason_seg_data
        idx = random.randint(0, len(self.json_data_list) - 1)
        image_path = self.json_data_list[idx]["image_path"]

        # if "tif" in image_path:
        #     image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        # else:
        #     image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 使用Pillow打开图像
        # image = Image.open(image_path)

        # # 如果需要将图像转换为RGB模式（例如，图像是灰度或其他模式）
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        # image=np.array(image)
        if "tiff" in image_path:
            image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
            image=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.imread(image_path)
        # pdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        # preprocess image for clip
        if np.max(image)>1 or np.min(image)<0:
            image = (image-np.min(image))/(np.max(image)-np.min(image))
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # mask, sents, is_sentence = get_mask_from_json(json_path, image)
        is_sentence = self.json_data_list[idx]['is_sentence']
        sents=self.json_data_list[idx]['text']
        if "ceramic" in image_path.lower() or "nanoparticle" in image_path.lower():
            masks=[np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], s["mask_name"])).convert('L'))==s["color_id"] if "color_id" in s.keys() 
               else np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], s["mask_name"])).convert('L'))!=0
               for s in self.json_data_list[idx]['shapes']]
        else:
            masks=[np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], s["mask_name"])))==s["color_id"] if "color_id" in s.keys() 
               else np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], s["mask_name"])))!=0
               for s in self.json_data_list[idx]['shapes']]

        sampled_sents = sents
        sampled_masks = masks
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        if self.explanatory != -1 and image_name in self.img_to_explanation:
            if random.random() < self.explanatory:
                choice = 2
            else:
                choice = random.randint(0, 1)

        questions = []
        answers = []
        classes=[]
        for i in range(len(self.json_data_list[idx]["shapes"])):
            text=sampled_sents[i]
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                if self.use_gpt_qa:
                    """
                    use my qa pair generated by gpt
                    """
                    # pdb.set_trace()
                    class_name=self.json_data_list[idx]["shapes"][i]["class_name"]
                    classes.append(class_name)
                    # qa_list=self.json_data_list[idx]["gpt_qa"]
                    # qa_list_class=[qa for qa in qa_list if qa["class_name"].lower() == class_name.lower()]
                    qa_list_class=[]
                    for shape in self.json_data_list[idx]["shapes"]:
                        if shape["class_name"].lower() == class_name.lower():
                            qa_list_class=shape["qa_list"]
                    if len(qa_list_class)>0:
                        qa_dict=random.choice(qa_list_class)
                        text=qa_dict["question"]
                        answer=qa_dict["answer"]
                        answer =  " {}".format(answer) + random.choice(self.cot_answer_list) 
                        questions.append(question_template.format(sent=text))
                        questions[-1] = (
                            DEFAULT_IMAGE_TOKEN
                            + "\n"
                            + text
                            + " {}".format(random.choice(self.explanatory_question_list))
                        )
                        answers.append(answer)
                        # print(text,answer)
                        conversations = []
                        prompts=[]
                        conv = conversation_lib.default_conversation.copy()
                        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

                        i = 0
                        while i < len(questions):
                            conv.messages = []
                            conv.append_message(conv.roles[0], questions[i])
                            conv.append_message(conv.roles[1], "")
                            prompts.append(conv.get_prompt())
                            conv.messages.pop()
                            conv.append_message(conv.roles[1], answers[i])
                            conversations.append(conv.get_prompt())
                            i += 1
                        continue
                questions.append(question_template.format(sent=text))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=text.lower()))

            # add explanation if applicable
            img_name = image_path.split("/")[-1]
            if self.explanatory != -1 and img_name in self.img_to_explanation:
                if choice == 0:  # [SEG] token
                    answers.append(random.choice(self.answer_list))
                elif choice == 1:  # [SEG] token + texhat answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    answer = random.choice(self.answer_list) + " {}".format(answer)
                    questions[-1] = (
                        DEFAULT_IMAGE_TOKEN
                        + "\n"
                        + text
                        + " {}".format(random.choice(self.explanatory_question_list))
                    )
                    answers.append(answer)
                elif choice == 2:  # vanilla text answer
                    image_name = image_path.split("/")[-1]
                    answer = self.img_to_explanation[image_name]["outputs"]
                    questions[-1] = DEFAULT_IMAGE_TOKEN + "\n" + text
                    answers.append(answer)
                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            prompts=[]
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], "")
                prompts.append(conv.get_prompt())
                conv.messages.pop()
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        image_name = image_path.split("/")[-1]
        if (
            self.explanatory != -1
            and image_name in self.img_to_explanation
            and choice == 2
        ):
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label
        else:
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        # print("Masks shape: ", masks.shape, "selected_sents: ", len(sampled_sents))
        # assert masks.shape[0] == len(sampled_sents), (
        #     masks.shape,
        #     len(sampled_sents),
        # )
        assert len(conversations)==len(prompts), (len(conversations), len(prompts))
        inference = True
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            None,
            None,
            classes,
            prompts, # prompts 要补
            inference,
        )
