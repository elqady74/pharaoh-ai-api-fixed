from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ai
import uvicorn
from config import config

app = FastAPI(title="Pharaoh Face Swap API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ai.router, prefix="/api/ai", tags=["AI"])

@app.get("/")
async def root():
    return {"message": "Welcome to Pharaoh Face Swap API"}

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=True)
