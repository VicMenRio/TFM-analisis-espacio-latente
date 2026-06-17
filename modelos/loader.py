"""
Unified loader for text-to-image diffusion models.

Provides a consistent interface for VAE encoding/decoding and latent
generation across SD 1.5, SD 2.1 and SDXL without leaking pipeline
internals into experiment code.
"""

from __future__ import annotations

import gc
import os
import threading

import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline


# ---------------------------------------------------------------------------
# Registro de modelos soportados
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: dict[str, dict] = {
    "sd15": {
        "model_id":        "runwayml/stable-diffusion-v1-5",
        "pipeline_cls":    StableDiffusionPipeline,
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_res":     512,
        "unet_type":       "UNet2DConditionModel",
        "text_encoder":    "CLIP ViT-L/14",
        "params_B":        0.86,
    },
    "sd21": {
        "model_id":        "sd2-community/stable-diffusion-2-1",
        "pipeline_cls":    StableDiffusionPipeline,
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_res":     768,
        "unet_type":       "UNet2DConditionModel",
        "text_encoder":    "OpenCLIP ViT-H/14",
        "params_B":        0.87,
    },

    "sdxl": {
        "model_id":        "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_cls":    StableDiffusionXLPipeline,
        "latent_channels": 4,
        "vae_scale_factor": 8,
        "default_res":     1024,
        "unet_type":       "UNet2DConditionModel",
        "text_encoder":    "CLIP ViT-L/14 + OpenCLIP ViT-bigG/14",
        "params_B":        2.60,
    },
}


# ---------------------------------------------------------------------------
# ModelWrapper
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Wrapper sobre un pipeline de Diffusers que expone una API estable para 
    experimentos en el espacio latente: codificar, decodificar y generar con captura de z0.
    """

    def __init__(
        self,
        model_key: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True,
    ):
        if model_key not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Supported keys: {list(SUPPORTED_MODELS)}"
            )

        self.model_key      = model_key
        self.cfg            = SUPPORTED_MODELS[model_key]
        self.device         = torch.device(device)
        self.dtype          = dtype
        self._generate_lock = threading.Lock()

        self._load(enable_xformers)

    # ------------------------------------------------------------------
    # Carga y descarga de pesos
    # ------------------------------------------------------------------

    def _load(self, enable_xformers: bool) -> None:
        cls         = self.cfg["pipeline_cls"]
        load_kwargs = {"torch_dtype": self.dtype}

        # Pass HF token if available (required for gated models like SDXL).
        # 
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            load_kwargs["token"] = hf_token

        # SD 1.x / 2.x ship with a safety checker; disable it for research.
        if cls is StableDiffusionPipeline:
            load_kwargs["safety_checker"] = None

        self.pipeline = cls.from_pretrained(
            self.cfg["model_id"], **load_kwargs
        ).to(self.device)

        self.pipeline.set_progress_bar_config(disable=True)

        if enable_xformers:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers unavailable; standard attention used instead

        self.vae  = self.pipeline.vae
        self.unet = self.pipeline.unet

        # SDXL's VAE overflows in fp16 and produces black images. upcast_vae()
        # casts only the VAE to float32 while the UNet stays in fp16.
        if cls is StableDiffusionXLPipeline:
            self.pipeline.upcast_vae()

    def offload(self) -> None:
        """Move pipeline to CPU and free GPU memory. Call before loading the next model."""
        self.pipeline.to("cpu")
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()

    def unload(self) -> None:
        """Delete pipeline weights from memory entirely (GPU + CPU RAM).

        The wrapper becomes unusable after this call. Use when done with a
        model and about to load the next one — frees the full ~3–7 GB of
        model weights so system RAM does not accumulate across models.
        """
        self.pipeline.to("cpu")
        del self.pipeline, self.vae, self.unet
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latent_channels(self) -> int:
        return self.cfg["latent_channels"]

    @property
    def vae_scale_factor(self) -> int:
        return self.cfg["vae_scale_factor"]

    @property
    def default_resolution(self) -> int:
        return self.cfg["default_res"]

    def latent_shape(self, image_size: Optional[int] = None) -> Tuple[int, int, int]:
        """Returns (C, H, W) of the latent space for a given image resolution."""
        res = image_size or self.default_resolution
        hw  = res // self.vae_scale_factor
        return (self.latent_channels, hw, hw)

    # ------------------------------------------------------------------
    # VAE encode / decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encodes a PIL image (or a [-1,1] NCHW tensor) to a latent vector.

        Uses the posterior *mean* (no sampling noise) so the encoding is
        deterministic and reproducible.

        Returns
        -------
        torch.Tensor  shape (1, C, H//8, W//8), on self.device
        """
        if isinstance(image, Image.Image):
            res   = image_size or self.default_resolution
            image = image.convert("RGB").resize((res, res), Image.LANCZOS)
            image = self._pil_to_tensor(image)

        # Use the VAE's own dtype — float32 for SDXL after upcast_vae(), fp16 otherwise.
        vae_dtype = next(self.vae.parameters()).dtype
        image     = image.to(self.device, dtype=vae_dtype)
        posterior = self.vae.encode(image)
        z         = posterior.latent_dist.mean
        return z * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decodes a latent tensor to a PIL image.

        Parameters
        ----------
        latent : torch.Tensor  shape (1, C, H, W) — in scaled latent space
        """
        vae_dtype = next(self.vae.parameters()).dtype
        latent = latent.to(self.device, dtype=vae_dtype)
        latent = latent / self.vae.config.scaling_factor
        image  = self.vae.decode(latent).sample
        image  = (image / 2 + 0.5).clamp(0, 1)
        image  = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Generation with z0 capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        image_size: Optional[int] = None,
        negative_prompt: str = "",
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Generates an image from a text prompt.

        Returns
        -------
        image  : PIL.Image.Image
        z0     : torch.Tensor  shape (1, C, H, W) — final denoised latent
                 before VAE decoding; in scaled latent space.
        """
        res       = image_size or self.default_resolution
        generator = torch.Generator(device=self.device).manual_seed(seed)
        captured  = {}

        # Hook the VAE decoder to capture z0 before it is decoded.
        # Diffusers passes `latent / scaling_factor` to vae.decode,
        # so we multiply back to restore the scaled representation.
        # Only the first call is captured; pipelines that invoke vae.decode
        # multiple times (tiled decoding, auxiliary passes) do not overwrite it.
        # The lock prevents two concurrent generate() calls on the same wrapper
        # from racing on self.vae.decode.
        with self._generate_lock:
            original_decode = self.vae.decode

            def _capture_and_decode(z, *args, **kwargs):
                if "z0" not in captured:
                    captured["z0"] = (z * self.vae.config.scaling_factor).detach().clone()
                vae_dtype = next(self.vae.parameters()).dtype
                return original_decode(z.to(vae_dtype), *args, **kwargs)

            self.vae.decode = _capture_and_decode
            try:
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=res,
                    width=res,
                    output_type="pil",
                )
            finally:
                self.vae.decode = original_decode  # always restore

        z0 = captured.get("z0")
        if z0 is None:
            raise RuntimeError(
                "VAE decode was never called during generation — z0 not captured. "
                "Check that output_type='pil' and the pipeline completed successfully."
            )
        return output.images[0], z0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Converts an RGB PIL image to a [-1, 1] NCHW float tensor."""
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    def __repr__(self) -> str:
        C, H, W = self.latent_shape()
        return (
            f"ModelWrapper("
            f"key='{self.model_key}', "
            f"latent=({C},{H},{W}), "
            f"encoder='{self.cfg['text_encoder']}', "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_model(model_key: str, **kwargs) -> ModelWrapper:
    """Shorthand for ModelWrapper(model_key, **kwargs)."""
    return ModelWrapper(model_key, **kwargs)
