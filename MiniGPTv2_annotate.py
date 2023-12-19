import argparse
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr
import json

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        self.stopping_criteria = stopping_criteria


    def answer_prepare(self, prompt, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                        repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        # prompt = f"<s>[INST] <Img><ImageHere></Img> [grounding] please describe this image? [/INST]"
        print('prompt:', prompt)
        # print('img_list:', img_list)
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=None,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs
    
    def model_generate(self, *args, **kwargs):
        # print("Positional arguments (args):", args)
        # print("Keyword arguments (kwargs):", kwargs)
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output
    
    def stream_answer(self, prompt, img_list, **kargs):
        # print('stream_answer img shape: ', img_list[0].shape)
        # random_tensor = torch.randn(1, 1, 4096).to('cuda:0')
        # img_list[0] = torch.cat((img_list[0], random_tensor), dim=1)
        # print('merged_tensor shape: ', img_list[0].shape)

        generation_kwargs = self.answer_prepare(prompt, img_list, **kargs)
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        from threading import Thread
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer
    
    def escape_markdown(self, text):
        # List of Markdown special characters that need to be escaped
        md_chars = ['<', '>']

        # Escape each special character
        for char in md_chars:
            text = text.replace(char, '\\' + char)

        return text

    def encode_img(self, img_list):
        img_emb_list = []
        for image in img_list:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB')
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)

            image_emb, _ = self.model.encode_img(image)
            img_emb_list.append(image_emb)
        return img_emb_list


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def Generate4img(chat, image_path, prompt):
    image = Image.open(image_path)
    image = image.convert("RGB")
    img_list=[image]
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            img_list = chat.encode_img(img_list)

    prefix = f"<s>[INST] <Img><ImageHere></Img> "
    suffix = f" [/INST]"
    prompt = prefix + prompt + suffix

    streamer = chat.stream_answer(prompt=prompt,
                            img_list=img_list,
                            temperature=0.6,
                            max_new_tokens=500,
                            max_length=2000)

    output = ''
    # print('streamer:', streamer)
    for new_output in streamer:
        escapped = chat.escape_markdown(new_output)
        output += escapped
    output = " ".join(output.split()).replace("\n", "")
    print('output:', output)
    return output


def Generate4imgs(chat, image_file_path, label_file_path, save_annatation_path, questions):
    tmp = [x.strip().split(' ') for x in open(label_file_path)]
    print(('images number: %d' % (len(tmp))))
    
    annotations = []
    for t in tmp:
        image_path = os.path.join(image_file_path, t[0] + '.png')
        image = Image.open(image_path)
        image = image.convert("RGB")
        img_list=[image]
        if len(img_list) > 0:
            if not isinstance(img_list[0], torch.Tensor):
                img_list = chat.encode_img(img_list)

        # prompt: f"<s>[INST] <Img><ImageHere></Img> question [/INST]"
        rn = random.randint(0, 4)
        prefix = f"<s>[INST] <Img><ImageHere></Img> "
        suffix = f"[/INST]"
        prompt = prefix + questions[rn] + suffix
        streamer = chat.stream_answer(
            prompt=prompt,
            img_list=img_list,
            temperature=0.6,
            max_new_tokens=500,
            max_length=2000
        )

        output = ''
        for new_output in streamer:
            escapped = chat.escape_markdown(new_output)
            output += escapped
        output = " ".join(output.split()).replace("\n", "")
        print("img id:", t[0], ' output:', output)

        annotation = {"image_id": t[0], "caption": output}
        annotations.append(annotation)

    # save
    data = {"annotations": annotations}
    with open(save_annatation_path, "w") as json_file:
        json.dump(data, json_file)

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model = model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device=device)

    # # Generate caption based on one image
    # image_path = "./data/examples_v2/cockdial.png"
    # prompt = f"<s>[INST] <Img><ImageHere></Img> What animal is in the picture? [/INST]"
    # annotation = Generate4img(chat, image_path, prompt)

    # # Generate caption based on images file
    # image_file_path = "./data/heco/images"
    # label_file_path = "./data/heco/heco.txt"
    # save_annatation_path = "./data/heco/heco.json"
    # questions = [
    #     "What are the people doing in the picture?",
    #     "What activities can be observed in the picture with the people?",
    #     "How would you describe the actions of the individuals in the image?",
    #     "Could you give me some details about the person's activity captured in the picture?",
    #     "Please identify the specific actions or tasks being performed by the individuals in the picture.",
    # ]
    # annotation = Generate4imgs(chat, image_file_path, label_file_path, save_annatation_path, questions)



if __name__ == "__main__":
    main()