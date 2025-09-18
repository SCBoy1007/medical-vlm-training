#!/usr/bin/env python3
"""
Simplified data processing for medical VLM training
Removes unnecessary padding complexity and follows official Qwen2.5-VL approach
"""

import os
import copy
import json
import logging
from typing import Dict, Optional, Sequence, List
from PIL import Image

import torch
from torch.utils.data import Dataset
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
DEFAULT_IMAGE_TOKEN = "<image>"

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
    """Simplified preprocessing without complex padding logic"""
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
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
                    for part_i, part in enumerate(parts):
                        if part_i > 0:
                            input_id += [IMAGE_TOKEN_INDEX]
                            target += [IGNORE_INDEX]
                            input_id += tokenizer.apply_chat_template(
                                [{"role": role, "content": part}]
                            )[1:]
                            target += [IGNORE_INDEX] * (len(input_id) - len(target))
                        else:
                            input_id += tokenizer.apply_chat_template(
                                [{"role": role, "content": part}]
                            )
                            target += [IGNORE_INDEX] * (len(input_id) - len(target))
                else:
                    input_id += tokenizer.apply_chat_template(
                        [{"role": role, "content": content}]
                    )
                    target += [IGNORE_INDEX] * (len(input_id) - len(target))
            elif role == "assistant":
                input_id += tokenizer.apply_chat_template(
                    [{"role": role, "content": content}]
                )
                target += [IGNORE_INDEX] * (
                    len(tokenizer.apply_chat_template([{"role": role, "content": ""}]))
                    - 1
                ) + tokenizer.apply_chat_template([{"role": role, "content": content}])[
                    len(tokenizer.apply_chat_template([{"role": role, "content": ""}])) - 1 :
                ]

        input_ids.append(input_id)
        targets.append(target)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Simplified dataset class following official approach"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.model_type = getattr(data_args, 'model_type', 'qwen2.5vl')

    def __len__(self):
        return len(self.list_data_dict)

    def process_image_simplified(self, image_file):
        """
        Simplified image processing following official Qwen2.5-VL approach
        No custom padding - rely on processor defaults
        """
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        # Official approach: direct processing without manual padding
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]

        return image_tensor, grid_thw

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        grid_thw = None

        # Handle image data
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.list_data_dict[i]["data_path"]

            if isinstance(image_file, list):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_simplified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_simplified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_simplified(image_file)
                image = [image]

            # Simplified grid_thw processing
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

        # Process conversations
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        )

        # Generate position ids
        if grid_thw is not None:
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            )
            data_dict["position_ids"] = position_ids

        # Handle image tensors
        if "image" in sources[0]:
            data_dict["image"] = image

        return data_dict

    def get_rope_index(self, merge_size, input_ids, image_grid_thw=None, video_grid_thw=None, second_per_grid_ts=None):
        """Simplified RoPE index generation"""
        if self.model_type == "qwen2.5vl":
            return get_rope_index_25(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts
            )
        else:
            return get_rope_index_2(
                input_ids, image_grid_thw, video_grid_thw, merge_size
            )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Simplified data collator"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # Handle images
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        # Handle position ids
        if "position_ids" in instances[0]:
            position_ids = [instance["position_ids"] for instance in instances]
            batch["position_ids"] = torch.nn.utils.rnn.pad_sequence(
                position_ids, batch_first=True, padding_value=0
            )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Create simplified data module"""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )