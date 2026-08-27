from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score
from skimage.morphology import dilation, disk
import cv2


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

COT_ANSWER_LIST = [
    'Accordingly, the segmentation result is [SEG].',
    'Thus, the segmentation result is [SEG].',
    'Therefore, the final segmentation is [SEG].',
    'Based on the analysis, the segmentation output is [SEG].',
    'As a result, the segmented regions are marked as [SEG].',
    'In conclusion, the segmentation map corresponds to [SEG].',
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        # 用于在分布式环境中同步指标
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict


def calculate_bleu(candidate, references):
    """
    计算BLEU分数
    
    参数:
        candidate: str - 生成的文本
        references: list[str] - 参考文本列表
        
    返回:
        float: BLEU分数 (0-1)
    """
    # 分词
    candidate_tokens = candidate.split()
    reference_tokens = [ref.split() for ref in references]
    
    # 使用平滑函数处理短句子
    smoothing = SmoothingFunction().method1
    
    # 计算BLEU-4
    score = sentence_bleu(reference_tokens, candidate_tokens, 
                         smoothing_function=smoothing)
    return score

def calculate_cider(candidates, references):
    """
    计算CIDEr分数
    
    参数:
        candidates: dict - {id: 生成的文本}
        references: dict - {id: [参考文本列表]}
        
    返回:
        float: CIDEr分数
    """
    scorer = Cider()
    score, _ = scorer.compute_score(references, candidates)
    return score

def calculate_spice(candidates, references):
    """
    计算SPICE分数
    
    参数:
        candidates: dict - {id: 生成的文本}
        references: dict - {id: [参考文本列表]}
        
    返回:
        float: SPICE分数 (0-1)
    """
    scorer = Spice()
    score, _ = scorer.compute_score(references, candidates)
    return score

def calculate_bertscore(candidates, references, lang="en"):
    """
    计算BERTScore
    
    参数:
        candidates: list[str] - 生成的文本列表
        references: list[str] - 参考文本列表
        lang: str - 语言代码
        
    返回:
        tuple: (精确率, 召回率, F1)
    """
    P, R, F1 = bert_score(candidates, references, lang=lang , 
                            model_type="roberta-large",
                            num_layers=17)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluate_text_metrics(candidate, reference, candidate_id="1"):
    """
    综合计算所有指标
    
    参数:
        candidate: str - 生成的文本
        reference: str - 参考文本
        candidate_id: str - 文本ID
        
    返回:
        dict: 包含所有指标的字典
    """
    # import pdb; pdb.set_trace()
    references=[reference]
    # BLEU
    bleu = calculate_bleu(candidate, references)
    
    # CIDEr
    cider_score = calculate_cider({candidate_id: [candidate]}, {candidate_id: references})
    
    # SPICE
    # spice_result = calculate_spice({candidate_id: candidate}, {candidate_id: references})
    # spice = spice_result['All']['f']
    
    # BERTScore
    bert_P, bert_R, bert_F1 = calculate_bertscore([candidate], references)
    
    return {
        "BLEU": bleu,
        "CIDEr": cider_score,
        # "SPICE": spice,
        "BERTScore_P": bert_P,
        "BERTScore_R": bert_R,
        "BERTScore_F1": bert_F1
    }

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode

def compute_boundary_iou(gt_mask, pred_mask):
    """
    计算 gt_mask 和 pred_mask 的 Boundary IoU。
    """
    gt_boundary = mask_to_boundary(gt_mask)
    pred_boundary = mask_to_boundary(pred_mask)

    intersection = np.logical_and(gt_boundary, pred_boundary).sum()
    union = np.logical_or(gt_boundary, pred_boundary).sum()

    if union == 0:
        return 0.0  # 避免除零错误

    return intersection / union

from medpy.metric.binary import __surface_distances

def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float = 0.5, spacing: tuple = None, connectivity=1):
    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!

    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image

    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy

    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 0.0  # 如果任意一个掩码为空，直接返回 0.0

    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)

    numel_a = len(a_to_b)
    numel_b = len(b_to_a)

    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b

    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b

    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc