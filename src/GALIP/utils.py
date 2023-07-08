import copy

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os, re
from glob import glob
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_multiprocessing

import json
import requests
import traceback

from transformers import CLIPTokenizer, CLIPModel
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import random
import clip

class ImageTextDataset(Dataset):
    def __init__(self, dataset_path, img_size, imagenet_normalization=False, max_length=77):
        self.image_samples, self.text_samples = self.listdir(dataset_path)
        self.max_length = max_length

        transform_list = image_preprocess(img_size, imagenet_normalization)
        self.transform = transforms.Compose(transform_list)

        # self.tokenizer, self.clip = FrozenNetwork(max_length=max_length).load()

    def listdir(self, dir_path):
        img_extensions = ['png', 'jpg', 'jpeg', 'JPG']
        image_list = []
        for ext in img_extensions:
            image_list += glob(os.path.join(dir_path, 'image', '*.' + ext))
        image_list.sort()

        txt_extensions = ['txt']
        text_list = []
        for ext in txt_extensions:
            text_list += glob(os.path.join(dir_path, 'text', '*.' + ext))
        text_list.sort()

        return image_list, text_list

    def __getitem__(self, index):
        image_path, text_path = self.image_samples[index], self.text_samples[index]
        img = Image.open(image_path).convert('RGB')
        txt = text_read(text_path)

        img = self.transform(img)

        # batch_encoding = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        # tokens = batch_encoding["input_ids"] # [1, 77]
        tokens = clip.tokenize(txt, truncate=True)
        tokens = torch.squeeze(tokens)
        # tokens = tokens.to(self.clip_text_encoder.device)
        # outputs = self.clip_text_encoder(input_ids=tokens)
        # txt_embed = outputs.last_hidden_state  # [77, 768]

        return img, tokens, txt

    def __len__(self):
        return len(self.image_samples)

class ImageDataset(Dataset):
    def __init__(self, dataset_path, img_size, imagenet_normalization=False):
        self.image_samples = self.listdir(dataset_path)

        transform_list = image_preprocess(img_size, imagenet_normalization)
        self.transform = transforms.Compose(transform_list)

    def listdir(self, dir_path):
        img_extensions = ['png', 'jpg', 'jpeg', 'JPG']
        image_list = []
        for ext in img_extensions:
            image_list += glob(os.path.join(dir_path, 'image', '*.' + ext))
        image_list.sort()

        return image_list

    def __getitem__(self, index):
        image_path = self.image_samples[index]
        img = Image.open(image_path).convert('RGB')

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_samples)

class FrozenNetwork(nn.Module):
    """Load Clip encoder (for text), SD-Autoencoder (for image)"""
    # https://github.com/baofff/U-ViT/blob/f0f35a9e710688ec669ae7154c490a8053f3139f/libs/clip.py
    def __init__(self, autoencoder_version="runwayml/stable-diffusion-v1-5", clip_version="openai/clip-vit-large-patch14", max_length=77):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_version)
        self.clip = CLIPModel.from_pretrained(clip_version)

        self.freeze()

    def freeze(self):
        self.clip.eval()

    def load(self):
        return self.tokenizer, self.clip


def image_preprocess(img_size, imagenet_normalization=False):
    # interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    if imagenet_normalization:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform_list = [
            transforms.Resize(size=[img_size, img_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True), # [h, w]
            transforms.ToTensor(),  # [0, 255] -> [0, 1] # [c, h, w]
            transforms.Normalize(mean=mean, std=std, inplace=True),  # [0, 1] -> [-1, 1]
        ]
    else:
        transform_list = [
            transforms.Resize(size=[img_size, img_size], interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # [0, 1] -> [-1, 1]
        ]

    return transform_list

def text_read(text_path):
    with open(text_path, 'r') as f:
        x = f.readlines()

    t = [text.strip() for text in x] # remove \n

    t_sample = random.choice(t)

    return t_sample


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def multi_gpu_run(ddp_fn, args): # in main
    # ddp_fn = train_fn
    world_size = torch.cuda.device_count() # ngpus
    torch_multiprocessing.spawn(fn=ddp_fn, args=(args, world_size), nprocs=world_size, join=True)


def build_init_procss(rank, world_size, device): # in build
    os.environ["MASTER_ADDR"] = "127.0.0.1" # localhost
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    synchronize()
    torch.cuda.set_device(device)


def distributed_sampler(dataset, rank, num_replicas, shuffle):
    return torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, num_replicas=num_replicas, shuffle=shuffle)
    # return torch.utils.data.RandomSampler(dataset)


def infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch

def find_latest_ckpt(folder):
    files = []
    for fname in os.listdir(folder):
        s = re.findall(r'\d+', fname)
        if len(s) == 1:
            files.append((int(s[0]), fname))
    if files:
        file_name = max(files)[1]
        index = os.path.splitext(file_name)[0]
        return file_name, index
    else:
        return None, 0


def broadcast_params(model):
    params = model.parameters()
    for param in params:
        dist.broadcast(param.data, src=0)
    dist.barrier()
    torch.cuda.synchronize()


def dataparallel_and_sync(model, local_rank, find_unused_parameters=False):
    # DistributedDataParallel
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)

    # broadcast
    broadcast_params(model)

    model = model.module

    return model

def cleanup():
    dist.destroy_process_group()

def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()

def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v.mean().item() for k, v in zip(keys, losses)}

    return reduced_losses


def dict_to_numpy(x_dict, python_value=False):
    losses_numpy = {}
    for k,v in x_dict.items():
        losses_numpy[k] = tensor_to_numpy(v, python_value=python_value)

    return losses_numpy

def tensor_to_numpy(x, python_value=False):
    if isinstance(x, torch.Tensor):
        if python_value:
            return x.detach().cpu().numpy().tolist()
        else:
            return x.detach().cpu().numpy()
    else:
        return x

def get_val(x):
    x_val = x.mean().item()

    return x_val

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
