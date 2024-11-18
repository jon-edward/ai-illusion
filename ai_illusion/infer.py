from dataclasses import dataclass
from datetime import datetime
import logging
import random
from typing import Tuple

from PIL import Image
from diffusers import DPMSolverMultistepScheduler
import torch

from img_utils import center_crop_resize, upscale
from illusion_pipes import IllusionPipes, illusion_pipes

@dataclass
class InferState:
    pipes: IllusionPipes | None = None


_state = InferState()


def infer(
        control_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float = 8.0,
        controlnet_conditioning_scale: float = 1,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        upscaler_strength: float = 0.5,
        seed: int | None = None) -> Tuple[Image.Image, int]:

    start_time = datetime.now()
    logging.info("Inference started at %s", start_time)

    if _state.pipes is None:
        _state.pipes = illusion_pipes()

    main_pipe, image_pipe = _state.pipes

    control_image_small = center_crop_resize(control_image)
    control_image_large = center_crop_resize(control_image, (1024, 1024))

    main_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        main_pipe.scheduler.config, use_karras=True, algorithm_type="sde-dpmsolver++")

    seed = random.randint(0, 2 ** 32 - 1) if seed is None else seed
    generator = torch.Generator(device="cuda").manual_seed(seed)

    out = main_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control_image_small,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        generator=generator,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        num_inference_steps=15,
        output_type="latent"
    )

    upscaled_latents = upscale(out, "nearest-exact", 2)

    out_image = image_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image_large,
        image=upscaled_latents,
        guidance_scale=float(guidance_scale),
        generator=generator,
        num_inference_steps=20,
        strength=upscaler_strength,
        control_guidance_start=float(control_guidance_start),
        control_guidance_end=float(control_guidance_end),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale)
    )

    end_time = datetime.now()

    logging.info("Finished at %s, taking %s", end_time, end_time - start_time)

    return out_image["images"][0], seed
