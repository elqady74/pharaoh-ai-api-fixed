import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
from config import config


class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.face_app = None
        self.pipe = None
        self.ip_model = None

    def load_insightface(self):
        print("⏳ Loading InsightFace...")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.face_app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )

        self.face_app.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=(640, 640),
        )

        print("✅ InsightFace ready!")

    def load_stable_diffusion(self):
        print("⏳ Loading Stable Diffusion...")

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        vae = AutoencoderKL.from_pretrained(
            config.VAE_PATH,
            torch_dtype=dtype,
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            config.MODEL_PATH,
            scheduler=scheduler,
            vae=vae,
            safety_checker=None,
            torch_dtype=dtype,
        )

        self.pipe = self.pipe.to(self.device)

        # Optional (helps memory on CUDA)
        if self.device == "cuda":
            try:
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

        print("✅ Stable Diffusion ready!")

    def load_ip_adapter(self):
        print("⏳ Downloading IP-Adapter FaceID...")

        faceid_path = hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename="ip-adapter-faceid_sd15.bin",
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.ip_model = IPAdapterFaceID(
            self.pipe,
            faceid_path,
            self.device,
            torch_dtype=dtype,
        )

        # --- Force every sub-module to the SAME dtype/device -----------
        # Guards against "expected m1 and m2 to have the same dtype,
        # but got: float != c10::Half" which happens when any single
        # layer (unet, text_encoder, vae, or the ip-adapter's
        # projection/LoRA layers) ends up in a different precision
        # than the rest of the pipeline.
        self.pipe.text_encoder.to(self.device, dtype=dtype)
        self.pipe.unet.to(self.device, dtype=dtype)
        self.pipe.vae.to(self.device, dtype=dtype)

        self.ip_model.image_proj_model.to(self.device, dtype=dtype)

        for attn_processor in self.pipe.unet.attn_processors.values():
            attn_processor.to(self.device, dtype=dtype)
        # -----------------------------------------------------------------

        print("=" * 60)
        print("System Information")
        print("=" * 60)
        print(f"Device          : {self.device}")
        print(f"CUDA Available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU             : {torch.cuda.get_device_name(0)}")

        print(f"Pipeline dtype  : {next(self.pipe.unet.parameters()).dtype}")
        print(f"IPAdapter dtype : {self.ip_model.torch_dtype}")
        print("=" * 60)

        print("✅ IP-Adapter FaceID ready!")

    def load_all(self):
        if (
            self.face_app is not None
            and self.pipe is not None
            and self.ip_model is not None
        ):
            return self.face_app, self.pipe, self.ip_model

        self.load_insightface()
        self.load_stable_diffusion()
        self.load_ip_adapter()

        return self.face_app, self.pipe, self.ip_model


model_loader = ModelLoader()