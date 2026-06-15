"""
Cargador unificado para modelos de difusión texto-imagen.

Proporciona una interfaz consistente para codificación/decodificación VAE y
generación de latents en SD 1.5, SD 2.1 y SDXL, sin exponer los internos
del pipeline al código de experimentos.
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
# Registro de modelos
# ---------------------------------------------------------------------------

# Registro central de modelos soportados. Actúa como fuente única de verdad para
# configuración arquitectónica: IDs de HuggingFace, clase de pipeline, resolución
# nativa y metadatos del codificador de texto. Añadir un modelo nuevo implica
# únicamente extender este diccionario sin modificar ningún otro código.
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
    Envoltorio ligero sobre un pipeline de Diffusers que expone una API estable
    para experimentos con el espacio latente: encode, decode y generación con
    captura de z0.

    Uso
    ---
    >>> model = ModelWrapper("sd15")
    >>> image, z0 = model.generate("a red apple on a table", seed=0)
    >>> z0.shape          # torch.Size([1, 4, 64, 64])
    >>> reconstructed = model.decode_latent(z0)
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
    # Carga
    # ------------------------------------------------------------------

    def _load(self, enable_xformers: bool) -> None:
        # Carga el pipeline completo en el dtype indicado (fp16 por defecto para
        # reducir consumo de VRAM). El safety checker de SD 1.x/2.x se desactiva
        # porque consume ~600 MB de VRAM adicionales y no es necesario en este contexto
        # experimental donde los prompts están controlados.
        cls         = self.cfg["pipeline_cls"]
        load_kwargs = {"torch_dtype": self.dtype}

        # Se usa el token de HF si está disponible (necesario para modelos con acceso restringido como SDXL).
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            load_kwargs["token"] = hf_token

        # SD 1.x / 2.x incluyen safety checker; se desactiva para uso en investigación.
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
                pass  # xformers no disponible; se usa atención estándar

        self.vae  = self.pipeline.vae
        self.unet = self.pipeline.unet

        # El VAE de SDXL produce desbordamiento numérico en fp16 generando imágenes
        # completamente negras. upcast_vae() convierte únicamente el VAE a float32
        # dejando el UNet en fp16, lo que minimiza el impacto en VRAM (~200 MB extra).
        # En encode_image() y decode_latent() se consulta el dtype real del VAE con
        # next(self.vae.parameters()).dtype para aplicar el cast correcto.
        if cls is StableDiffusionXLPipeline:
            self.pipeline.upcast_vae()

    def offload(self) -> None:
        """Mueve el pipeline a CPU y libera memoria GPU. Llamar antes de cargar el siguiente modelo."""
        self.pipeline.to("cpu")
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()

    def unload(self) -> None:
        """Elimina los pesos del pipeline de memoria (GPU + RAM).

        El wrapper queda inutilizable tras esta llamada. Usar cuando se haya
        terminado con un modelo y se vaya a cargar el siguiente — libera los
        ~3–7 GB de pesos para que la RAM del sistema no se acumule entre modelos.
        """
        self.pipeline.to("cpu")
        del self.pipeline, self.vae, self.unet
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Propiedades
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
        """Devuelve (C, H, W) del espacio latente para una resolución de imagen dada."""
        res = image_size or self.default_resolution
        hw  = res // self.vae_scale_factor
        return (self.latent_channels, hw, hw)

    # ------------------------------------------------------------------
    # Codificación / decodificación VAE
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        image_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Codifica una imagen PIL (o tensor NCHW en [-1,1]) a un vector latente.

        Usa la *media* de la distribución posterior (sin ruido de muestreo) para
        que el encoding sea determinista y reproducible.

        Devuelve
        --------
        torch.Tensor  forma (1, C, H//8, W//8), en self.device
        """
        if isinstance(image, Image.Image):
            res   = image_size or self.default_resolution
            image = image.convert("RGB").resize((res, res), Image.LANCZOS)
            image = self._pil_to_tensor(image)

        # Se usa el dtype propio del VAE: float32 para SDXL tras upcast_vae(), fp16 en los demás.
        vae_dtype = next(self.vae.parameters()).dtype
        image     = image.to(self.device, dtype=vae_dtype)
        posterior = self.vae.encode(image)
        # Se usa la media de la distribución posterior en lugar de una muestra
        # (.sample()) para obtener un encoding determinista y reproducible.
        # Esto es esencial para comparar latentes de imágenes reales entre modelos.
        z         = posterior.latent_dist.mean
        return z * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decodifica un tensor latente a una imagen PIL.

        Parámetros
        ----------
        latent : torch.Tensor  forma (1, C, H, W) — en espacio latente escalado
        """
        vae_dtype = next(self.vae.parameters()).dtype
        latent = latent.to(self.device, dtype=vae_dtype)
        latent = latent / self.vae.config.scaling_factor
        image  = self.vae.decode(latent).sample
        image  = (image / 2 + 0.5).clamp(0, 1)
        image  = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return Image.fromarray((image * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Generación con captura de z0
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
        Genera una imagen a partir de un prompt de texto.

        Devuelve
        --------
        image  : PIL.Image.Image
        z0     : torch.Tensor  forma (1, C, H, W) — latente final desruidado
                 antes de la decodificación VAE; en espacio latente escalado.
        """
        res       = image_size or self.default_resolution
        generator = torch.Generator(device=self.device).manual_seed(seed)
        captured  = {}

        # Mecanismo de captura de z₀ mediante hook sobre vae.decode:
        # Diffusers no expone el latente final como salida del pipeline, así que
        # se reemplaza temporalmente vae.decode por una función closure que:
        #   1. Captura z₀ en el primer call (antes de que sea decodificado).
        #   2. Revierte la división por scaling_factor que aplica el pipeline
        #      antes de llamar a vae.decode, restaurando el espacio latente escalado.
        #   3. Delega en el decode original con el cast de dtype adecuado.
        # El bloque finally garantiza que vae.decode se restaura aunque falle la
        # generación, evitando dejar el modelo en estado inconsistente.
        # El lock impide que dos llamadas concurrentes a generate() compitan sobre
        # self.vae.decode en el mismo wrapper.
        with self._generate_lock:
            original_decode = self.vae.decode

            def _capture_and_decode(z, *args, **kwargs):
                if "z0" not in captured:
                    captured["z0"] = (z * self.vae.config.scaling_factor).detach().clone()
                # Se castea z al dtype del VAE. En SDXL, upcast_vae() pone el VAE en
                # float32 pero el pipeline sigue pasando latentes en fp16 (omite su propio
                # cast al ver que el VAE ya es float32). Sin este cast las capas conv lanzan
                # "Input type (Half) != bias type (float)".
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
                self.vae.decode = original_decode  # restaurar siempre

        z0 = captured.get("z0")
        if z0 is None:
            raise RuntimeError(
                "VAE decode was never called during generation — z0 not captured. "
                "Check that output_type='pil' and the pipeline completed successfully."
            )
        return output.images[0], z0

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convierte una imagen PIL RGB a un tensor float NCHW en [-1, 1]."""
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
# Función de carga rápida
# ---------------------------------------------------------------------------

def load_model(model_key: str, **kwargs) -> ModelWrapper:
    """Atajo para ModelWrapper(model_key, **kwargs)."""
    return ModelWrapper(model_key, **kwargs)
