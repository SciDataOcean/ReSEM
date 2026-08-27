import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST, COT_ANSWER_LIST)
from PIL import Image
import pdb

class ReasonSegDatasetQWSA_EM(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([140, 140, 140]).view(-1, 1, 1)
    pixel_std = torch.Tensor([40, 40, 40]).view(-1, 1, 1)
    # pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    # pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_data_dir="data",
        tokenizer=None,
        processer=None,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 4,
        exclude_val=False,
        reason_seg_data="organelle||plantorgan||cremi||ceramic||defect||micronet||nanoparticle_train",
        explanatory=-1,
        use_gpt_qa=False,
    ):
        self.exclude_val = exclude_val
        self.reason_seg_data = reason_seg_data
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample

        self.base_data_dir = base_data_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.cot_answer_list = COT_ANSWER_LIST

        reason_seg_data,splits=reason_seg_data.split("_")[0], reason_seg_data.split("_")[1]
        self.split=splits
        reason_seg_data=reason_seg_data.replace("material","ceramic||defect||micronet||nanoparticle")
        reason_seg_data_ls = reason_seg_data.split("||")
        # splits = splits.split("_")
        self.json_data_list=[]
        # prepare self.img_to_explanation for all datasets
        for data in reason_seg_data_ls:
            if use_gpt_qa:
                # json_path=os.path.join(base_data_dir,  data, f"{splits}_d_qa.json") 
                json_path=os.path.join(base_data_dir,  data, f"{splits}_revised.json") 
            else:
                json_path=os.path.join(base_data_dir,  data, f"{splits}.json")
            json_data=json.load(open(json_path))
            for item in json_data:
                item['image_path']=os.path.join(base_data_dir, data, item['image_name'])
                item['data_root']=os.path.join(base_data_dir, data)
            self.json_data_list.extend(json_data)
        self.use_gpt_qa = use_gpt_qa
        self.explanatory_question_list=EXPLANATORY_QUESTION_LIST
        self.processer=processer

    def __len__(self):
        return self.samples_per_epoch if self.split=='train' else len(self.json_data_list)

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
        if "tiff" in image_path:
            image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
            image = (image-np.min(image))/(np.max(image)-np.min(image)) *255
            image=cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.imread(image_path)
        # pdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print('1',image.shape)
        
        ori_size = image.shape[:2]
        image = self.transform.apply_image(image)  # preprocess image for sam
        # print('2',image.shape)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        # print('3',image.shape)# torch.Size([3, 1024, 1024])
        pil_image=Image.fromarray((image.permute(1, 2, 0).numpy()*255).astype(np.uint8))
        # preprocess image for clip
        # if np.max(image)>1 or np.min(image)<0:
        #     image = (image-np.min(image))/(np.max(image)-np.min(image))
        # print('in dataset:',image.shape, type(image)) # in dataset: (1024, 1024, 3) <class 'numpy.ndarray'>
        
        shape_idx=random.randint(0, len(self.json_data_list[idx]['shapes'])-1)
        shape=self.json_data_list[idx]['shapes'][shape_idx]

        if "ceramic" in image_path.lower() or "nanoparticle" in image_path.lower():
            masks=np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], shape["mask_name"])).convert('L'))==shape["color_id"] if "color_id" in shape.keys() \
            else np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], shape["mask_name"])).convert('L'))!=0
        else:
            masks=np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], shape["mask_name"])))==shape["color_id"] if "color_id" in shape.keys() \
            else np.array(Image.open(os.path.join(self.json_data_list[idx]["data_root"], shape["mask_name"])))!=0
        masks=torch.from_numpy(masks).float().unsqueeze(0)
        classes=[shape["class_name"]]
        qa_idx=random.randint(0, len(shape['qa_list']) - 1)
        query=shape['qa_list'][qa_idx]['question']+ " {}".format(random.choice(self.explanatory_question_list))
        outputs=shape['qa_list'][qa_idx]['answer'] + random.choice(self.cot_answer_list) 
        if self.split=="train":
            conversation = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": outputs}
                ]
            conversation_prompt=[
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": ""}
                ]
        else:
            conversation = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": outputs} # qaq
                ]
            conversation_prompt=[
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]},
                    {"role": "assistant", "content": ""}
                ]
        inputs=self.processer(
            text=self.processer.tokenizer.apply_chat_template(conversation, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        prompts=self.processer(
            text=self.processer.tokenizer.apply_chat_template(conversation_prompt, tokenize=False),
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        # Create labels
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        if self.split=='train':
            assistant_start = len(self.processer.tokenizer.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}, {"role": "assistant", "content": ""}],
                tokenize=False,
                add_generation_prompt=False
            ))
            labels[0, :assistant_start] = -100
        
        # return_dict= {
        #     "pixel_values": inputs["pixel_values"].squeeze(0),
        #     "input_ids": inputs["input_ids"].squeeze(0),
        #     "labels": labels.squeeze(0),
        #     "attention_mask": inputs["attention_mask"].squeeze(0),
        #     "images": image,
        #     "masks_list": masks,
        #     "label_list": torch.ones(masks.shape[1], masks.shape[2]) * 255,
        #     "resize_list": [resize],
        #     'classes': classes,
        # }

        # # print('pixel_values', return_dict['pixel_values'].shape)
        # # print('input_ids', return_dict['input_ids'].shape)
        # # print('labels', return_dict['labels'].shape)
        # # print('attention_mask', return_dict['attention_mask'].shape)
        # # print('images', return_dict['images'].shape)
        # # print('masks_list', return_dict['masks_list'].shape)
        # return return_dict
        inference=False if self.split == 'train' else True
        return (
            image_path,
            image,
            pil_image,
            conversation,
            masks,
            torch.ones(masks.shape[1], masks.shape[2]) * 255, # label_tensor
            resize,
            [query],
            classes,
            inference,
            ## new
            conversation_prompt,
        )


def unified_collate_fn(batch, tokenizer, processor=None):
    """Unified collate function for finetune mode"""
    return collate_fn_qwsa(batch, tokenizer=tokenizer, processor=processor)

def collate_fn_qwsa(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1, processor=None
):
    # 1. Filter out invalid samples (those returning None in __getitem__)
    batch = [item for item in batch if item is not None]
    
    # 2. If entire batch is empty after filtering, return a complete but empty structure
    if not batch:
        return {
            "images": torch.tensor([], dtype=torch.float32),
            "pixel_values": torch.tensor([], dtype=torch.float32),
            "input_ids": torch.tensor([], dtype=torch.long),
            "labels": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "masks_list": [],
            "label_list": [],
            "resize_list": [],
            "offset": torch.LongTensor([]),
            "inference": [],
            "image_paths": [],
            "questions_list": [],
            "sampled_classes_list": [],
            "image_grid_thw": None,
        }
    
    assert processor is not None, "Qwen requires processor to be passed to collate_fn"
    
    # Initialize collectors
    image_path_list, images_list, pil_images, all_messages_structured = [], [], [], []
    masks_list, label_list, resize_list, questions_list, sampled_classes_list = [], [], [], [], []
    offset_list = [0]
    cnt = 0
    inferences = []
    prompt_messages_structured = []
    
    # 3. Collect raw data from valid batch
    if len(batch[0])==12:
        for (
            image_path, images, pil_image, messages, masks, label, resize,
            questions, sampled_classes,_,_, inference, prompt_messages
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            pil_images.append(pil_image)
            all_messages_structured.append(messages)
            masks_list.append(masks.float())
            label_list.append(label)
            resize_list.append(resize)
            questions_list.append(questions)
            sampled_classes_list.append(sampled_classes)
            cnt += 1
            offset_list.append(cnt)
            inferences.append(inference)
            prompt_messages_structured.append(prompt_messages)
    else:
        for (
            image_path, images, pil_image, messages, masks, label, resize,
            questions, sampled_classes, inference, prompt_messages
        ) in batch:
            image_path_list.append(image_path)
            images_list.append(images)
            pil_images.append(pil_image)
            all_messages_structured.append(messages)
            masks_list.append(masks.float())
            label_list.append(label)
            resize_list.append(resize)
            questions_list.append(questions)
            sampled_classes_list.append(sampled_classes)
            cnt += 1
            offset_list.append(cnt)
            inferences.append(inference)
            prompt_messages_structured.append(prompt_messages)
    
    # 4. Use processor to handle text and images
    # For training, conversation includes answer. For validation, assistant's part is empty.
    texts_for_processing = [
        processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=(not conversation[-1]['content'])
        )
        for conversation in all_messages_structured
    ]
    prompt_texts_for_processing = [
        processor.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        for conversation in prompt_messages_structured
    ]
    
    # 5. Package the batch using processor
    inputs = processor(
        text=texts_for_processing,
        images=pil_images,
        return_tensors="pt",
        padding=True
    )
    prompts=processor(
        text=prompt_texts_for_processing,
        images=pil_images,
        return_tensors="pt",
        padding=True
    )
    
    # 6. Extract all required outputs
    input_ids = inputs['input_ids']
    attention_masks = inputs['attention_mask']
    images_clip = inputs['pixel_values']
    image_grid_thw = inputs.get('image_grid_thw')

    prompt_ids=prompts['input_ids']
    attention_masks_prompts = prompts['attention_mask']
    
    # 7. Create target tensor for 'labels' and mask padding
    targets = input_ids.clone()
    targets[targets == tokenizer.pad_token_id] = -100
    
    # 8. Build and return final batch dictionary
    return_dict = {
        "images": torch.stack(images_list, dim=0),
        "pixel_values": images_clip,
        "input_ids": input_ids,
        "labels": targets,
        "attention_mask": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "inference": inferences[0] if len(inferences) > 0 else False,  # Handle empty list case
        # Auxiliary info, filtered before model call
        "image_paths": image_path_list,
        "questions_list": questions_list,
        "classes_list": sampled_classes_list,
        "prompt_ids": prompt_ids,
        "attention_masks_prompts": attention_masks_prompts,
        # needed by qwen3.5
        "mm_token_type_ids": inputs.get("mm_token_type_ids"),
    }
    
    if image_grid_thw is not None:
        return_dict['image_grid_thw'] = image_grid_thw
    
    return return_dict