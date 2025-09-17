import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# from decord import VideoReader
# from torchcodec.decoders import VideoDecoder
VideoReader = None
VideoDecoder = None
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
# Removed image_processor_wrapper - using direct processor with spatial merge compatible padding

# Setup logger for padding debug info
padding_logger = logging.getLogger('padding_debug')
padding_logger.setLevel(logging.DEBUG)
if not padding_logger.handlers:
    handler = logging.FileHandler('./logs/padding_debug.log', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    padding_logger.addHandler(handler)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    # Don't convert to tensor here - let DataCollator handle padding
    # This prevents issues with variable-length sequences
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels

        # Only set size attributes for Qwen2-VL, not for Qwen2.5-VL
        if data_args.model_type != "qwen2.5vl" and hasattr(self.data_args.image_processor, 'size'):
            self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
            self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        # Using spatial merge compatible padding approach for all models
        if data_args.model_type == "qwen2.5vl":
            rank0_print("Using spatial merge compatible padding for Qwen2.5-VL")

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def _pad_image_to_28_multiple(self, image):
        """
        Spatial merge compatible padding方案：
        确保 (h*w // 4) % 4 == 0，即 h*w % 16 == 0
        同时保持28的倍数（patch size要求）
        """
        width, height = image.size

        # 首先确保28的倍数
        new_width = ((width + 27) // 28) * 28
        new_height = ((height + 27) // 28) * 28

        # 计算patch数量
        h_patches = new_height // 28
        w_patches = new_width // 28
        total_patches = h_patches * w_patches

        # 检查spatial merge compatibility: total_patches % 16 == 0
        if total_patches % 16 != 0:
            # 使用搜索算法找到满足条件的最小padding方案
            best_solution = None
            min_extra_patches = float('inf')

            # 限制搜索范围：每个维度最多增加6个patches（168像素）
            # 这个范围已经足够覆盖大部分情况
            for extra_h in range(7):  # 0到6
                for extra_w in range(7):  # 0到6
                    candidate_h = h_patches + extra_h
                    candidate_w = w_patches + extra_w
                    candidate_total = candidate_h * candidate_w

                    # 检查是否满足spatial merge条件
                    if candidate_total % 16 == 0:
                        extra_patches = candidate_total - total_patches

                        # 寻找增加patches最少的方案
                        if extra_patches < min_extra_patches:
                            min_extra_patches = extra_patches
                            best_solution = (candidate_h, candidate_w)

            # 应用找到的最优解
            if best_solution is not None:
                new_height = best_solution[0] * 28
                new_width = best_solution[1] * 28
                padding_logger.info(f"Found solution: {h_patches}x{w_patches}={total_patches} → {best_solution[0]}x{best_solution[1]}={best_solution[0]*best_solution[1]} (added {min_extra_patches} patches)")
            else:
                # 如果搜索范围内没找到解，使用更激进的方案
                # 直接增加到下一个16的倍数所需的最小patches
                target_patches = ((total_patches + 15) // 16) * 16

                # 简单策略：优先增加较小的维度以保持纵横比
                if h_patches <= w_patches:
                    # 增加高度到能满足条件的值
                    for extra_h in range(1, 16):  # 最多增加15个高度patches
                        candidate_h = h_patches + extra_h
                        needed_w_patches = (target_patches + candidate_h - 1) // candidate_h  # 向上取整
                        if needed_w_patches >= w_patches and (candidate_h * needed_w_patches) % 16 == 0:
                            new_height = candidate_h * 28
                            new_width = needed_w_patches * 28
                            break
                else:
                    # 增加宽度到能满足条件的值
                    for extra_w in range(1, 16):
                        candidate_w = w_patches + extra_w
                        needed_h_patches = (target_patches + candidate_w - 1) // candidate_w
                        if needed_h_patches >= h_patches and (needed_h_patches * candidate_w) % 16 == 0:
                            new_height = needed_h_patches * 28
                            new_width = candidate_w * 28
                            break

                padding_logger.info(f"Used fallback solution: target_patches={target_patches}")

        if new_width == width and new_height == height:
            padding_logger.info(f"Image {width}x{height} already compatible, no padding needed")
            return image

        # 创建黑色背景的新图像
        padded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        padded_image.paste(image, (0, 0))  # 粘贴到左上角

        # 验证结果
        final_h_patches = new_height // 28
        final_w_patches = new_width // 28
        final_total_patches = final_h_patches * final_w_patches
        is_compatible = final_total_patches % 16 == 0

        padding_logger.info(f"PADDED: {width}x{height} → {new_width}x{new_height} (added {new_width-width}x{new_height-height} padding)")
        padding_logger.info(f"  Patches: {h_patches}x{w_patches}={total_patches} → {final_h_patches}x{final_w_patches}={final_total_patches}")
        padding_logger.info(f"  Spatial merge compatible: {is_compatible} (patches % 16 = {final_total_patches % 16})")

        return padded_image

    def _post_processor_spatial_fix(self, pixel_values, grid_thw, image_file):
        """
        POST-PROCESSOR修复：确保spatial merge兼容性
        基于测试结果的精确修复策略
        """
        import torch
        import math

        num_patches, hidden_dim = pixel_values.shape
        spatial_merge_unit = 4  # Qwen2.5-VL spatial merge unit

        padding_logger.info(f"POST-PROCESSOR CHECK: {image_file}")
        padding_logger.info(f"  Input tensor: [{num_patches}, {hidden_dim}]")
        padding_logger.info(f"  Grid THW: {grid_thw}")

        # 计算seq_len用于spatial merge
        seq_len = num_patches // spatial_merge_unit
        remainder = num_patches % spatial_merge_unit

        padding_logger.info(f"  Calculated seq_len: {seq_len}")
        padding_logger.info(f"  Patches % spatial_merge_unit: {remainder}")

        if remainder != 0:
            # 需要修复：补齐到4的倍数
            target_patches = ((num_patches + spatial_merge_unit - 1) // spatial_merge_unit) * spatial_merge_unit
            pad_patches = target_patches - num_patches

            padding_logger.info(f"  🔧 FIXING: need to add {pad_patches} patches")
            padding_logger.info(f"  Target patches: {target_patches}")

            # 在tensor末尾添加零填充
            padding_tensor = torch.zeros(pad_patches, hidden_dim, dtype=pixel_values.dtype, device=pixel_values.device)
            pixel_values = torch.cat([pixel_values, padding_tensor], dim=0)

            # 更新grid_thw以匹配新的patch数量
            # 策略：增加高度维度以保持宽度不变
            t, h, w = grid_thw
            new_h = target_patches // w
            if target_patches % w != 0:
                # 如果不能整除，调整宽度
                new_w = w
                while target_patches % new_w != 0:
                    new_w += 1
                new_h = target_patches // new_w
            else:
                new_w = w

            grid_thw = torch.tensor([t, new_h, new_w], dtype=grid_thw.dtype, device=grid_thw.device)

            # 验证修复结果
            final_patches = pixel_values.shape[0]
            final_seq_len = final_patches // spatial_merge_unit
            final_remainder = final_patches % spatial_merge_unit

            padding_logger.info(f"  ✅ FIXED tensor: [{final_patches}, {hidden_dim}]")
            padding_logger.info(f"  ✅ Updated Grid THW: {grid_thw}")
            padding_logger.info(f"  ✅ Final seq_len: {final_seq_len}")
            padding_logger.info(f"  ✅ Spatial merge compatible: {final_remainder == 0}")

        else:
            padding_logger.info(f"  ✅ Already compatible, no fix needed")

        return pixel_values, grid_thw

    def process_image_unified(self, image_file):
        processor = self.data_args.image_processor
        image = Image.open(image_file).convert("RGB")

        # Apply padding for Qwen2.5-VL to ensure 28-multiple dimensions
        padding_logger.info(f"DEBUG: model_type={self.model_type}, processing image {image_file}")
        if self.model_type == "qwen2.5vl":
            padding_logger.info("DEBUG: Applying padding check for qwen2.5vl")
            image = self._pad_image_to_28_multiple(image)
        else:
            padding_logger.info(f"DEBUG: Skipping padding - model_type is {self.model_type}, not qwen2.5vl")

        # Different processing for Qwen2.5-VL vs Qwen2-VL
        if self.model_type == "qwen2.5vl":
            # For Qwen2.5-VL, use the image processor (no text parameter for image-only processing)
            visual_processed = processor(images=image, return_tensors="pt")
        else:
            # For Qwen2-VL, use preprocess method
            visual_processed = processor.preprocess(image, return_tensors="pt")

        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]

        # 🔧 POST-PROCESSOR SPATIAL MERGE FIX
        if self.model_type == "qwen2.5vl":
            image_tensor, grid_thw = self._post_processor_spatial_fix(image_tensor, grid_thw, image_file)

        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if VideoReader is None:
            raise ImportError("VideoReader not available - video processing disabled for medical image training")
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        if VideoDecoder is None:
            raise ImportError("VideoDecoder not available - video processing disabled for medical image training")
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels

        # Only set size attributes for Qwen2-VL, not for Qwen2.5-VL
        if self.data_args.model_type != "qwen2.5vl" and hasattr(processor, 'size'):
            processor.size["longest_edge"] = processor.max_pixels
            processor.size["shortest_edge"] = processor.min_pixels

        # Different processing for Qwen2.5-VL vs Qwen2-VL
        if self.data_args.model_type == "qwen2.5vl":
            # For Qwen2.5-VL, use the image processor directly
            video_processed = processor(videos=video, return_tensors="pt")
        else:
            # For Qwen2-VL, use preprocess method
            video_processed = processor.preprocess(
                images=None, videos=video, return_tensors="pt"
            )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]

            # Smart path resolution: handle "images/filename.jpg" format
            def resolve_image_path(img_path, base_folder):
                if img_path.startswith("images/"):
                    # Remove "images/" prefix and find actual file location
                    filename = img_path[7:]  # Remove "images/" prefix

                    # Try different possible locations
                    possible_paths = [
                        os.path.join(base_folder, "train", "high_quality", filename),
                        os.path.join(base_folder, "train", "low_quality", filename),
                        os.path.join(base_folder, "val", "high_quality", filename),
                        os.path.join(base_folder, "val", "low_quality", filename),
                        os.path.join(base_folder, "test", "high_quality", filename),
                        os.path.join(base_folder, "test", "low_quality", filename),
                    ]

                    # Find the first existing file
                    for path in possible_paths:
                        if os.path.exists(path):
                            return path

                    # Fallback to original path if none found
                    return os.path.join(base_folder, img_path)
                else:
                    return os.path.join(base_folder, img_path)
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        resolve_image_path(file, image_folder) for file in image_file
                    ]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = resolve_image_path(image_file, image_folder)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = resolve_image_path(image_file, image_folder)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        # Convert input_ids to tensor for rope index calculation
        input_ids_tensor = (
            data_dict["input_ids"] if isinstance(data_dict["input_ids"], torch.Tensor)
            else torch.tensor(data_dict["input_ids"], dtype=torch.long)
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,  # Use merge_size=2 for RoPE calculation
            input_ids_tensor,
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            # Handle both tensor and list formats for text-only samples
            input_ids_for_pos = (
                data_dict["input_ids"] if isinstance(data_dict["input_ids"], torch.Tensor)
                else torch.tensor(data_dict["input_ids"], dtype=torch.long)
            )
            position_ids = (
                torch.arange(0, input_ids_for_pos.size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        # Handle attention mask calculation for both tensor and list formats
        if isinstance(data_dict["input_ids"], torch.Tensor):
            data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
        else:
            data_dict["attention_mask"] = [len(data_dict["input_ids"][0])]

        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        # Fix data structure for DataCollator compatibility
        # Convert nested list structure to flat structure expected by DataCollator
        if isinstance(data_dict["input_ids"], list) and len(data_dict["input_ids"]) == 1:
            data_dict["input_ids"] = data_dict["input_ids"][0]  # Flatten from [[tokens]] to [tokens]

        if isinstance(data_dict["labels"], list) and len(data_dict["labels"]) == 1:
            data_dict["labels"] = data_dict["labels"][0]  # Flatten from [[labels]] to [labels]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        # Handle both tensor and list formats for input_ids and labels
        input_ids = [
            ids.squeeze(0) if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
            for ids in input_ids
        ]
        labels = [
            lbls.squeeze(0) if isinstance(lbls, torch.Tensor) else torch.tensor(lbls, dtype=torch.long)
            for lbls in labels
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        # Convert list format to tensor format for flattened collator
        # Ensure consistent tensor shapes for concatenation
        processed_input_ids = []
        processed_labels = []

        for ids in input_ids:
            if isinstance(ids, torch.Tensor):
                # Already tensor, ensure 2D shape (batch_size, seq_len)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                processed_input_ids.append(ids)
            else:
                # List format, convert to tensor with shape (1, seq_len)
                tensor_ids = torch.tensor(ids, dtype=torch.long)
                if tensor_ids.dim() == 2:
                    # Already has batch dimension from list of lists
                    processed_input_ids.append(tensor_ids)
                else:
                    # Single sequence, add batch dimension
                    processed_input_ids.append(tensor_ids.unsqueeze(0))

        for lbls in labels:
            if isinstance(lbls, torch.Tensor):
                # Already tensor, ensure 2D shape (batch_size, seq_len)
                if lbls.dim() == 1:
                    lbls = lbls.unsqueeze(0)
                processed_labels.append(lbls)
            else:
                # List format, convert to tensor with shape (1, seq_len)
                tensor_lbls = torch.tensor(lbls, dtype=torch.long)
                if tensor_lbls.dim() == 2:
                    # Already has batch dimension from list of lists
                    processed_labels.append(tensor_lbls)
                else:
                    # Single sequence, add batch dimension
                    processed_labels.append(tensor_lbls.unsqueeze(0))

        input_ids = processed_input_ids
        labels = processed_labels
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
