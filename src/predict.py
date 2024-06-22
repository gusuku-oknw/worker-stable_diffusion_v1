import os
from typing import List
import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    # StableDiffusionInpaintPipeline,
    StableDiffusionInpaintPipelineLegacy,

    DDIMScheduler,
    DDPMScheduler,
    # DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    # KarrasVeScheduler,
    PNDMScheduler,
    # RePaintScheduler,
    # ScoreSdeVeScheduler,
    # ScoreSdeVpScheduler,
    # UnCLIPScheduler,
    # VQDiffusionScheduler,
    LMSDiscreteScheduler
)
from safetensors.torch import load_file as safe_load

from PIL import Image
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

MODEL_CACHE = "diffusers-cache"

class Predictor:
    def __init__(self, model_tag="runwayml/stable-diffusion-v1-5"):
        self.model_tag = model_tag

    def setup(self):
        print("Loading pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            self.model_tag,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=False,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")
        self.inpaint_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        self.lora_loaded = False
        self.txt2img_pipe.unet.to(memory_format=torch.channels_last)
        self.img2img_pipe.unet.to(memory_format=torch.channels_last)
        self.inpaint_pipe.unet.to(memory_format=torch.channels_last)
        self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        self.img2img_pipe.enable_xformers_memory_efficient_attention()
        self.inpaint_pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, init_image, mask, prompt_strength, num_outputs, num_inference_steps, guidance_scale, scheduler, seed, lora, lora_scale):
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")

        if width * height > 786432:
            raise ValueError("Maximum size is 1024x768 or 768x1024 pixels, because of memory limits.")

        extra_kwargs = {}
        if mask:
            if not init_image:
                raise ValueError("mask was provided without init_image")

            pipe = self.inpaint_pipe
            init_image = Image.open(init_image).convert("RGB")
            extra_kwargs = {
                "mask_image": Image.open(mask).convert("RGB").resize(init_image.size),
                "image": init_image,
                "strength": prompt_strength,
            }
        elif init_image:
            pipe = self.img2img_pipe
            extra_kwargs = {
                "init_image": Image.open(init_image).convert("RGB"),
                "strength": prompt_strength,
            }
        else:
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        if not (lora is None):
            print(f"Loading LoRA file from: {lora}")
            # Load LoRA model from safetensors
            state_dict = safe_load(lora)
            pipe.unet.load_state_dict(state_dict, strict=False)
            self.lora_loaded = True

            # Apply LoRA scaling directly in the UNet layers
            for name, module in pipe.unet.named_modules():
                if hasattr(module, 'apply_lora_scale'):
                    module.apply_lora_scale(lora_scale)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt]*num_outputs if negative_prompt is not None else None,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i] and self.NSFW:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        if len(output_paths) == 0:
            raise Exception("NSFW content detected. Try running it again, or try a different prompt.")

        return output_paths

def make_scheduler(name, config):
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config)
    }[name]
