import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor, SigLipVisionTower

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN,
                    EXPLANATORY_QUESTION_LIST, LONG_QUESTION_LIST,
                    SHORT_QUESTION_LIST, COT_ANSWER_LIST)
from PIL import Image
import pdb

class ReasonSegDataset_EM(torch.utils.data.Dataset):
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
        vision_tower=None,
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
        if "clip" in vision_tower:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        elif "siglip" in vision_tower:
            self.clip_image_processor = SigLipVisionTower(vision_tower).image_processor

        self.short_question_list = SHORT_QUESTION_LIST
        self.long_question_list = LONG_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.cot_answer_list = COT_ANSWER_LIST

        reason_seg_data,splits=reason_seg_data.split("_")[0], reason_seg_data.split("_")[1]
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

    def __len__(self):
        return self.samples_per_epoch

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
        last_error = None
        for _ in range(20):
            candidate_idx = random.randint(0, len(self.json_data_list) - 1)
            try:
                return self._getitem_once(candidate_idx)
            except Exception as exc:
                last_error = exc
                print(
                    "WARNING: skip bad ReasonSegDataset_EM sample idx={} error={}".format(
                        candidate_idx, repr(exc)
                    )
                )
        raise last_error

    def _getitem_once(self, idx):
        # images, jsons = self.reason_seg_data
        image_path = self.json_data_list[idx]["image_path"]
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
        # if np.max(image)>1 or np.min(image)<0:
        #     image = (image-np.min(image))/(np.max(image)-np.min(image))
        # print('in dataset:',image.shape, type(image)) # in dataset: (1024, 1024, 3) <class 'numpy.ndarray'>
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
            
        valid_count = min(len(sents), len(masks), len(self.json_data_list[idx]["shapes"]))
        if valid_count <= 0:
            raise ValueError("no valid text/mask/shape entries in {}".format(image_path))
        if len(sents) != len(masks) or len(sents) != len(self.json_data_list[idx]["shapes"]):
            print(
                "WARNING: text/mask/shape length mismatch for {}: text={}, masks={}, shapes={}; using first {}".format(
                    image_path, len(sents), len(masks), len(self.json_data_list[idx]["shapes"]), valid_count
                )
            )

        if valid_count >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(valid_count)), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(valid_count))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        sampled_masks = [
            masks[int(i)] for i in sampled_inds
        ]

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
        
        for i in sampled_inds:
            i = int(i)
            text = sents[i]
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
                elif choice == 1:  # [SEG] token + text answer
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
        assert masks.shape[0] == len(sampled_sents), (
            masks.shape,
            len(sampled_sents),
        )
        assert len(conversations)==len(prompts), (len(conversations), len(prompts))
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_sents,
            classes,
            prompts
        )

if __name__ == "__main__":
    pass
