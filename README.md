---
title: Pharaoh AI API
emoji: 🏺
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# Pharaoh AI API

FastAPI backend for the Pharaoh AI image generation service.

## Features

- Stable Diffusion 1.5
- IP-Adapter FaceID
- InsightFace
- FastAPI REST API


# Pharaoh Face Swap API 🏛️

This is a FastAPI implementation of the Pharaoh Face Swap project, based on IP-Adapter FaceID.

## Features
- Transform personal photos into Ancient Egyptian Pharaohs or Queens.
- Preserves facial features using IP-Adapter FaceID.
- FastAPI backend for easy integration.
- Dockerized for easy deployment.

## Project Structure
```
PHARAOH-API/
├── models/
│   └── ip_adapter/       # Store FaceID model here
├── outputs/              # Generated images
├── routers/
│   └── ai.py             # API endpoints
├── services/
│   ├── inference.py      # Core logic
│   └── model_loader.py   # Model loading logic
├── uploads/              # Temporary uploads
├── app.py                # Main entry point
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── Dockerfile            # Containerization
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models:**
   - Download `ip-adapter-faceid_sd15.bin` from [Hugging Face](https://huggingface.co/h94/IP-Adapter-FaceID) and place it in `models/ip_adapter/`.

3. **Run the API:**
   ```bash
   python app.py
   ```

## API Usage
Post an image to `/api/ai/transform` with the following parameters:
- `file`: The image file.
- `character`: "pharaoh" or "queen" (default: "pharaoh").
- `num_steps`: Number of inference steps (default: 40).
- `scale`: Face feature strength (default: 0.8).
