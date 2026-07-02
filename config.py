import os
import torch
from dotenv import load_dotenv

load_dotenv()


class Config:

    # ===========================
    # Device
    # ===========================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    # ===========================
    # Models
    # ===========================

    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "runwayml/stable-diffusion-v1-5"
    )

    VAE_PATH = os.getenv(
        "VAE_PATH",
        "stabilityai/sd-vae-ft-mse"
    )

    # ===========================
    # Overlay Colors
    # ===========================

    GOLD = (218, 165, 32, 220)
    DARK_GOLD = (184, 134, 11, 220)
    LAPIS = (15, 82, 186, 220)
    RED = (183, 65, 14, 220)

    # ===========================
    # Prompts
    # Keep them under ~75 CLIP tokens
    # ===========================

    PHARAOH_PROMPT = (
        "portrait of an ancient Egyptian pharaoh, "
        "blue khepresh crown, "
        "golden nemes headdress, "
        "golden usekh collar, "
        "kohl eyes, "
        "royal throne, "
        "hieroglyphic background, "
        "ultra detailed, masterpiece"
    )

    QUEEN_PROMPT = (
        "portrait of ancient Egyptian queen, "
        "Nefertiti style, "
        "white atef crown, "
        "gold jewelry, "
        "lotus flowers, "
        "hieroglyphics, "
        "ultra detailed, masterpiece"
    )

    NEGATIVE_PROMPT = (
        "low quality, blurry, "
        "bad anatomy, "
        "extra limbs, "
        "cartoon, anime, "
        "modern clothes, "
        "text, watermark"
    )


config = Config()