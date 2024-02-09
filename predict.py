import io
import os
import sys
from typing import List, Optional

from cog import Path, Input, File

import imghdr

from urllib.parse import urlparse
import cv2
from PIL import Image, ImageOps, PngImagePlugin
import numpy as np
import torch
from const import MPS_SUPPORT_MODELS
from loguru import logger
from torch.hub import download_url_to_file, get_dir
import hashlib

from schema import Config

import os

from PIL import Image

import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch

from diffusers.utils import load_image

import requests

from helper import (
    norm_img,
    get_cache_path_by_url,
    load_jit_model,
    load_img,
    numpy_to_bytes,
    resize_max_size,
    pil_to_bytes,
)

from model_manager import ModelManager

def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "./big-lama.pt",
)
LAMA_MODEL_MD5 = os.environ.get("LAMA_MODEL_MD5", "e3aa4aaa15225a33ec84f9f4bc47e500")

class Predictor():
    def setup(self) -> None:
        self.model = ModelManager(
            name="lama",
            device="cpu"
        )

    """Run a single prediction on the model"""
    def predict(self, image_url: Path = Input(description="Image URL"), mask_url: Path = Input(description="Input image")) -> Path:

        # Send GET requests to the URLs to retrieve the images

        with open(str(image_url), "rb") as binary_file:
            # Read the entire file as a binary stream
            image_ = binary_file.read()

        with open(str(mask_url), "rb") as binary_file:
            # Read the entire file as a binary stream
            mask_ = binary_file.read()

        origin_image_bytes = image_
        image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)

        mask, _ = load_img(mask_, gray=True)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        original_shape = image.shape
        interpolation = cv2.INTER_CUBIC

        size_limit = max(image.shape)

        config = Config(
            ldm_steps = 25,
            ldm_sampler='plms',
            hd_strategy='Crop',
            zits_wireframe=True,
            hd_strategy_crop_margin=196,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=2048,
            prompt="",
            negative_prompt="",

            use_croper=False,
            croper_x=281,
            croper_y=244,
            croper_height=512,
            croper_width=512,

            sd_scale=1,
            sd_mask_blur=5,
            sd_strength=0.75,
            sd_steps=50,
            sd_guidance_scale=7.5,
            sd_sampler='uni_pc',
            sd_seed=-1,
            sd_match_histograms=False,
            cv2_flag='INPAINT_NS',
            cv2_radius=5,
            paint_by_example_steps=50,
            paint_by_example_guidance_scale=7.5,
            paint_by_example_mask_blur=5,
            paint_by_example_seed=-1,
            paint_by_example_match_histograms=False,
            paint_by_example_example_image=None,
            p2p_steps=50,
            p2p_image_guidance_scale=1.5,
            p2p_guidance_scale=7.5,
            controlnet_conditioning_scale=0.4,
            controlnet_method='control_v11p_sd15_canny',
        )

        logger.info(f"Origin image shape: {original_shape}")
        image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

        mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

        res_np_img = self.model(image, mask, config)

        res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
                )
            res_np_img = np.concatenate(
                (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
            )

        ext = get_image_ext(origin_image_bytes)

        opencv_image = cv2.cvtColor(res_np_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('result.jpg', opencv_image)

        return Path('result.jpg')

        