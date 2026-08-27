# ReSEM: Knowledge-Grounded Reasoning Segmentation for Electron Microscopy

<font size=7><div align='center'><b>ReSEM</b>: Knowledge-Grounded <b>Re</b>asoning <b>S</b>egmentation for <b>E</b>lectron <b>M</b>icroscopy</div></font>

<font size=7><div align='center'>
    <a href="https://arxiv.org/pdf/2308.00692.pdf"><strong>📄 Paper</strong></a> | 
    <a href="https://huggingface.co/SciDataOcean"><strong>🤗 Models</strong></a> | 
    <a href="#training"><strong>🛠️ Training</strong></a> | 
    <a href="#inference"><strong>🔍 Inference</strong></a> | 
    <a href="#dataset"><strong>🗂️ Dataset</strong></a>
</div></font>

ReSEM is a knowledge-grounded reasoning segmentation framework for electron microscopy images. It is designed to understand natural-language scientific queries, identify the relevant microscopic structures or material regions, produce precise segmentation masks, and provide explanatory responses grounded in domain knowledge. The project includes model checkpoints, training and inference scripts, and the ReasonEM dataset for building EM segmentation models with stronger reasoning and instruction-following abilities.

<!-- <p align="center"> <img src="imgs/teaser.jpg" width="100%"> </p> -->

<table class="center">
<tr>
  <td style="text-align:center;"><b>Input</b></td>
  <td style="text-align:center;"><b>Output</b></td>
  <td style="text-align:center;"><b>Input</b></td>
  <td style="text-align:center;"><b>Output</b></td>
</tr>
<tr>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/mitochondria.jpg"></td>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/mitochondria_mitochondria_masked_img_0.jpg"></td>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/cleft.png"></td>              
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/cleft_cleft_masked_img_0.jpg"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"Identify the organelle characterized by elongated oval structures with infoldings that increase surface area for biochemical reactions. Please output segmentation mask and explain why."</td>
  <td width=25% style="text-align:center;">"The organelle in question is identified by its elongated oval shape and the presence of infoldings, known as cristae. These infoldings increase the surface area for biochemical reactions, crucial for ATP production. The segmentation mask highlights these features, indicating the organelle is the mitochondrion. Therefore, the final segmentation is [SEG] .”</td>
  <td width=25% style="text-align:center;color:gray">"Identify the narrow, elongated region that facilitates signal transmission between two neuronal structures.Please output segmentation mask and explain why. "</td>
  <td width=25% style="text-align:center;">"The target region is the synaptic cleft, characterized by its narrow, elongated shape and electron-lucent appearance. It is centrally located between the presynaptic and postsynaptic neurons, maintaining a consistent width to support neurotransmitter diffusion. In conclusion, the segmentation map corresponds to [SEG]."</td>
</tr>

<tr>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/positive.png"></td>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/positive_positive_masked_img_0.jpg"></td>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/nanoparticle.png"></td>              
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/nanoparticle_nanoparticle_masked_img_0.jpg"></td>
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"Segment the areas that provide pathways for electron flow, influencing the material's conductivity.Please output segmentation mask and give some explanation. "</td>
  <td width=25% style="text-align:center;">"The areas providing pathways for electron flow are the interconnected regions with irregular, angular shapes. These are the positive phase, which is crucial for the material's superconducting properties by facilitating electron flow. In conclusion, the segmentation map corresponds to [SEG]."</td>
  <td width=25% style="text-align:center;color:gray">"Identify the uniformly distributed spherical entities across the field of view that exhibit a range of sizes.Please output segmentation mask and explain the reason. "</td>
  <td width=25% style="text-align:center;">"The entities are uniformly distributed and vary in size, indicating a consistent synthesis process. Their spherical shape and distribution suggest they are designed for applications requiring uniformity and stability. In conclusion, the segmentation map corresponds to [SEG]."</td>
</tr>

<tr>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/vacuole.png"></td>
  <td style="vertical-align: middle; text-align: center;"><img src="imgs/vacuole_vacuole_masked_img_0.jpg"></td>
  <!-- <td style="vertical-align: middle; text-align: center;"><img src="imgs/wash_hands.jpg"></td>              
  <td style="vertical-align: middle; text-align: center;"><img src="vis_output/wash_hands.jpg"></td> -->
</tr>
<tr>
  <td width=25% style="text-align:center;color:gray;">"Locate the organelle with an intact membrane, crucial for cellular compartmentalization and homeostasis.Please output segmentation mask and give some explanation. "</td>
  <td width=25% style="text-align:center;">"The organelle with an intact membrane is the vacuole. Its intact membrane is crucial for maintaining cellular compartmentalization and homeostasis, as it plays a significant role in storage and maintaining turgor pressure. Based on the analysis, the segmentation output is [SEG]."</td>
  <!-- <td width=25% style="text-align:center;color:gray">"Where to <ins>wash hands</ins> in this image? Please output segmentation mask."</td>
  <td width=25% style="text-align:center;">"Sure, [SEG]."</td> -->
</tr>

</table>

<!-- <p align="center"> <img src="imgs/fig_overview.jpg" width="100%"> </p> -->



<!-- ## Abstract
While automated segmentation for electron microscopy has advanced significantly, existing methods remain limited by rigid visual prompting mechanisms and a critical absence of integrated scientific priors. To address these challenges, we present **ReSEM**, a knowledge-aware multi-modal framework designed for scientific **Re**asoning **S**egmentation for **E**lectron **M**icroscopy, explicitly grounding pixel-level features in domain knowledge. We construct a large-scale, multi-domain Electron Microscopy benchmark that comprises $\sim$ 20k reasoning Q\&A pairs, generated via a hallucination-resistant pipeline to facilitate robust instruction following and inject domain-specific semantics. To further refine reasoning capabilities, we introduce Micro-Group Relative Policy Optimization, a novel post-training strategy that aligns model outputs with scientifically accurate and visually grounded reasoning paths. Extensive experiments demonstrate that ReSEM achieves state-of-the-art performance across diverse EM modalities, achieving a mean IoU of 62.4\% and significantly outperforming current baselines.
For more details, please refer to the [paper](https://arxiv.org/abs/2308.00692). -->

<!-- ## Highlights
**LISA** unlocks the new segmentation capabilities of multi-modal LLMs, and can handle cases involving: 
1. complex reasoning; 
2. world knowledge; 
3. explanatory answers; 
4. multi-turn conversation. 

**LISA** also demonstrates robust zero-shot capability when trained exclusively on reasoning-free datasets. In addition, fine-tuning the model with merely 239 reasoning segmentation image-instruction pairs results in further performance enhancement. -->


## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Dataset
The ReasonEM dataset can be downloaded from [this link](https://huggingface.co/datasets/SciDataOcean/ReasonEM).

## Training
### Data Preparation
The training and evaluation data should be organized as follows.

```
├── dataset_path
│   ├── ceramic
│   │   ├── images
│   │   ├── labels
│   │   ├── train_d_qa.json
│   │   └── val_d_qa.json
...
```
The json files should contain the following fields:

```
[
    {
        "image_name": "images/...png",
        "is_sentence": true,
        "shapes": [
            {
                "label": "target",
                "labels": [
                    "target"
                ],
                "shape_type": "mask",
                "image_name": "images/...png",
                "mask_name": "labels/...png",
                "class_name": "...",
                "color_id": ...,
                "description": "...",
                "qa_list": [
                    {
                        "question": "...",
                        "answer": "..."
                    },
                    ...
                ]
            },
            ...
        ]
    },
    ...
]

```

### Pre-trained weights

#### LLaVA
To train LISA-7B or 13B, you need to follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to merge the LLaVA delta weights. Typically, we use the final weights `LLaVA-Lightning-7B-v1-1` and `LLaVA-13B-v1-1` merged from `liuhaotian/LLaVA-Lightning-7B-delta-v1-1` and `liuhaotian/LLaVA-13b-delta-v1-1`, respectively. For Llama2, we can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Training
For sft training, run the following command:
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LLaVA" \
  --dataset_dir='dataset_path' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="reason_seg_em" \
  --val_dataset="reason_seg_em" \
  --reason_seg_data="organelle||ceramic..." \
  --sample_rates="1" \
  --exp_name="..." \
  --steps_per_epoch="..." \
  --epochs="..." \
  --lr="..." \
  --use_gpt_qa \
  --train_from_scratch
```
For grpo training, run the following command:
```
deepspeed --master_port=24998 train_grpo_ds.py \
  --version="sft_huggingface_path" \
  --dataset_dir='dataset_path' \
  --log_base_dir='runs' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="reason_seg_em" \
  --val_dataset="reason_seg_em" \
  --reason_seg_data="organelle||ceramic..." \
  --sample_rates="1" \
  --batch_size="..." \
  --num_generations="..." \
  --exp_name="..." \
  --steps_per_epoch="..." \
  --epochs="..."  \
  --lr="..." \
  --use_gpt_qa \
  --lora_module_full_finetune \
  --reward_weights ... \
```
Refer to scripts/train.sh for detailed examples and more training settings.

When training is finished, to get the full model weight:
```
cd ./runs/<exp_name>/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
```

### Merge LoRA Weight
Merge the LoRA weights of `pytorch_model.bin`, save the resulting model into your desired path in the Hugging Face format:
```
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LLaVA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```
Refer to scripts/merge.sh for detailed examples and more settings.



### Validation
```
deepspeed --master_port=24999 train_ds.py \
  --version="PATH_TO_LISA_HF_Model_Directory" \
  --dataset_dir='dataset_path' \
  --vision_pretrained="PATH_TO_SAM" \
  --dataset="reason_seg_em" \
  --val_dataset="reason_seg_em" \
  --reason_seg_data="organelle||ceramic..." \
  --sample_rates="1" \
  --exp_name="..." \
  --steps_per_epoch="..." \
  --epochs="..." \
  --lr="..." \
  --use_gpt_qa \
```
 
## Inference 

To chat with [ReSEM-13B-llama2-sft](https://huggingface.co/SciDataOcean/ReSEM-13B-sft) or [ReSEM-13B-llama2-grpo](https://huggingface.co/SciDataOcean/ReSEM-13B-rft):
```
CUDA_VISIBLE_DEVICES=0 python batch_chat.py --version='ReSEM-13B-llama2-xxx'
python batch_chat.py --precision='bf16' \
  --version="ReSEM-13B-llama2-xxx" \
  --chat_json="chat_sample.json" 
```
chat_sample.json is arranged as:
```
[
    {
        "image":"image_path",
        "prompt": "...",
        "class": "..."
    },
    ...
]

```


## Citation 
If you find this project useful in your research, please consider citing:

```

```

## Acknowledgement
- Thanks to the contribution of [LLaVA](https://github.com/haotian-liu/LLaVA), [SAM](https://github.com/facebookresearch/segment-anything), [LISA](https://github.com/JIA-Lab-research/LISA), and [PathChat](https://anonymous.4open.science/r/PathChat-Seg-3116/README.md), which provide valuable foundations for this work.
