from typing import Tuple

import torch
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline, \
    StableDiffusionControlNetImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor


IllusionPipes = Tuple[StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline]


def illusion_pipes(
        safety_checker_ident: str | None = "CompVis/stable-diffusion-safety-checker") -> IllusionPipes:

    base_model = "SG161222/Realistic_Vision_V5.1_noVAE"

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16
    )

    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    safety_checker = (
        None if not safety_checker_ident else
        StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker").to("cuda")
    )

    main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        torch_dtype=torch.float16,
    ).to("cuda")

    image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components)

    return main_pipe, image_pipe
