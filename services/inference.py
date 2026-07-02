import traceback
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from config import config


def extract_face_embedding(face_app, img_pil):
    """
    Extract face embedding from InsightFace
    """

    img_cv = cv2.cvtColor(
        np.array(img_pil.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    faces = face_app.get(img_cv)

    if len(faces) == 0:
        return None, None

    face = faces[0]

    embedding = (
        torch.from_numpy(face.normed_embedding)
        .unsqueeze(0)
        .float()
    )

    bbox = face.bbox.astype(int)

    x1, y1, x2, y2 = bbox

    pad = 20

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(img_pil.width, x2 + pad)
    y2 = min(img_pil.height, y2 + pad)

    face_crop = img_pil.crop((x1, y1, x2, y2))

    return embedding, face_crop


def add_pharaoh_overlay(img, character):
    w, h = img.size

    overlay = Image.new(
        "RGBA",
        (w, h),
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    border = max(6, w // 60)

    draw.rectangle(
        [0, 0, w - 1, h - 1],
        outline=config.GOLD,
        width=border,
    )

    draw.rectangle(
        [
            border * 2,
            border * 2,
            w - border * 2 - 1,
            h - border * 2 - 1,
        ],
        outline=config.GOLD,
        width=max(2, border // 2),
    )

    symbols = "𓀀 𓂀 𓃭 𓅓 𓆣 𓇋 𓉐 𓋴"

    try:
        draw.text(
            (w // 2 - 80, h - 35),
            symbols,
            fill=config.GOLD,
        )
    except Exception:
        pass

    return Image.alpha_composite(
        img.convert("RGBA"),
        overlay,
    ).convert("RGB")


def transform_face(
    face_app,
    ip_model,
    img_pil,
    character="pharaoh",
    num_steps=40,
    scale=0.8,
):
    print("🔍 Extracting face features...")

    embedding, _ = extract_face_embedding(
        face_app,
        img_pil,
    )

    if embedding is None:
        print("⚠️ No face detected. Using zero embedding.")

        embedding = torch.zeros(
            (1, 512),
            dtype=torch.float32,
        )

    embedding = embedding.to(
        device=ip_model.device,
        dtype=ip_model.torch_dtype,
    )

    print("=" * 60)
    print(f"Embedding dtype : {embedding.dtype}")
    print(f"Embedding device: {embedding.device}")
    print(f"UNet dtype      : {next(ip_model.pipe.unet.parameters()).dtype}")
    print("=" * 60)

    prompt = (
        config.PHARAOH_PROMPT
        if character == "pharaoh"
        else config.QUEEN_PROMPT
    )

    print(f"🎨 Generating {character} image...")

    try:

        images = ip_model.generate(
            prompt=prompt,
            negative_prompt=config.NEGATIVE_PROMPT,
            faceid_embeds=embedding,
            num_samples=1,
            width=512,
            height=768,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
            scale=scale,
        )

    except Exception:

        print("=" * 80)
        print("❌ Generation Error")
        traceback.print_exc()
        print("=" * 80)

        raise

    result = images[0]

    result = result.resize(
        (512, 768),
        Image.LANCZOS,
    )

    result = add_pharaoh_overlay(
        result,
        character,
    )

    return result