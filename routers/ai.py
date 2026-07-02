from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import io
import os
import uuid
from services.model_loader import model_loader
from services.inference import transform_face

router = APIRouter()

# Global models
face_app, pipe, ip_model = None, None, None

@router.on_event("startup")
async def startup_event():
    global face_app, pipe, ip_model
    face_app, pipe, ip_model = model_loader.load_all()

@router.post("/transform")
async def transform(
    file: UploadFile = File(...),
    character: str = Form("pharaoh"),
    num_steps: int = Form(40),
    scale: float = Form(0.8)
):
    try:
        # Read image
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform
        result = transform_face(
            face_app, 
            ip_model, 
            img_pil, 
            character=character, 
            num_steps=num_steps, 
            scale=scale
        )
        
        # Save result
        output_filename = f"{uuid.uuid4()}.png"
        output_path = os.path.join("outputs", output_filename)
        result.save(output_path)
        
        return FileResponse(output_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
