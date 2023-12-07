import argparse, os, requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torch
device = 'cuda' if torch.cuda.is_available() else "cpu"
print("using device is", device)
from transformers import AutoTokenizer, BitsAndBytesConfig, TextStreamer

from llava.model import LlavaLlamaForCausalLM
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

def load_image(image_path):
  # load image
  image = Image.open(image_path).convert('RGB')
  # Disable the redundant torch
  disable_torch_init()
  # image to image tensor
  image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

  return image_tensor

def create_prompt(prompt):
  # set role
  conv_mode = "llava_v1"
  conv = conv_templates[conv_mode].copy()
  roles = conv.roles

  # create inp
  inp = f"{roles[0]}: {prompt}"
  inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp

  # append message
  conv.append_message(conv.roles[0], inp)
  conv.append_message(conv.roles[1], None)

  return conv

def caption_image(image_tensor, raw_prompt):
  # encode
  input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
  # set stop string
  stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
  keywords = [stop_str]
  stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
  # print("stopping keywords:", stopping_criteria.keywords)

  # inference
  with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images = image_tensor,
        do_sample = True,
        temperature = 0.2,
        max_new_tokens = 1024,
        use_cache = True,
        stopping_criteria = [stopping_criteria])

  # decode
  outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
  conv.messages[-1][-1] = outputs
  output = outputs.rsplit('</s>', 1)[0]
  return output




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--input_img", type=str, default="../000/1_seed23_0.jpg")
    parser.add_argument("--prompt", type=str, default="Please understand this web page and generate the correct html and css code.")

    args = parser.parse_args()

    # set BitsAndByteConfig for 4bit
    kwargs = {"device_map": "auto"}
    kwargs['load_in_4bit'] = True
    kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )

    # load model
    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # load vision processor
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda')
    image_processor = vision_tower.image_processor

    # load image
    image_tensor = load_image(args.input_img)
    # create prompt
    conv = create_prompt(args.prompt)
    raw_prompt = conv.get_prompt()

    output = caption_image(image_tensor, raw_prompt)
    print(output)