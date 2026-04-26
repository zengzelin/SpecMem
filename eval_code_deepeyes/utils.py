import math
from io import BytesIO
import base64
from PIL import Image
import io
import torch
import gc

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
tc_start_token = "<tool_call>"
tc_end_token = "</tool_call>"
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
stop_strings = ["</tool_call>", "</answer>"]

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Preserve the original smart resize heuristic."""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def map_box(left, top, right, bottom, width_scale, height_scale, ori_width, ori_height):
    left = int(left * width_scale)
    top = int(top * height_scale)
    right = int(right * width_scale)
    bottom = int(bottom * height_scale)
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, ori_width)
    bottom = min(bottom, ori_height)
    return left, top, right, bottom

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def safe_cuda_empty_cache():
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

def token_separability(logits_t, top_k=64, eps=1e-6):
    """
    logits_t: Tensor of shape [vocab_size]
    return: scalar separability score
    """
    # Select the top-k logits.
    topk_vals, _ = torch.topk(logits_t, k=top_k)

    # Score of the greedy token.
    z1 = topk_vals[0]

    # Competing tokens excluding the top-1 choice.
    competitors = topk_vals[1:]

    mu = competitors.mean()
    std = competitors.std(unbiased=False)

    score = (z1 - mu) / (std + eps)
    return score

def aggregate_token_scores(scores, mode="min", alpha=0.6):
    if mode == "min":
        score = scores.min()
        return torch.sigmoid(alpha * score)
    if mode == "mean":
        score = scores.mean()
        return torch.sigmoid(alpha * score)
    if mode == "bottom20":
        k = max(1, int(0.2 * len(scores)))
        score = torch.topk(scores, k=k, largest=False).values.mean()
        return torch.sigmoid(alpha * score)
    raise ValueError(f"Unknown aggregation mode: {mode}")



def build_score_profile(logits, top_k=64, mode="min", alpha=0.6, group_size=4, tail_length=8):
    token_scores = []
    for t in range(logits.size(0)):
        token_scores.append(token_separability(logits[t], top_k=top_k))

    token_scores = torch.stack(token_scores)
    token_scores_sigmoid = torch.sigmoid(alpha * token_scores)

    effective_group_size = max(1, min(group_size, len(token_scores_sigmoid)))
    group_scores = []
    for start in range(0, len(token_scores_sigmoid), effective_group_size):
        group_scores.append(token_scores_sigmoid[start:start + effective_group_size].mean())
    group_scores = torch.stack(group_scores)

    effective_tail_length = max(1, min(tail_length, len(token_scores_sigmoid)))
    effective_bottom_k = max(1, int(torch.ceil(torch.tensor(0.1 * len(group_scores))).item()))

    profile = {
        "confidence_score": aggregate_token_scores(token_scores, mode=mode, alpha=alpha),
        "tail_score": token_scores_sigmoid[-effective_tail_length:].mean(),
        "lowest_group_score": group_scores.min(),
        "bottom10_group_score": torch.topk(group_scores, k=effective_bottom_k, largest=False).values.mean(),
        "token_scores": token_scores_sigmoid,
        "group_scores": group_scores,
        "group_size": effective_group_size,
        "tail_length": effective_tail_length,
    }
    return profile



def answer_separability(logits, top_k=64, mode="min", alpha=0.6):
    """
    logits: Tensor of shape [n, vocab_size]
    mode: "min" | "mean" | "bottom20"
    """
    return build_score_profile(logits, top_k=top_k, mode=mode, alpha=alpha)["confidence_score"]
