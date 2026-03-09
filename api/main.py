import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks

from api.schemas import EstimationResponse, ErrorResponse, HealthResponse
from src.inference import run_pipeline
from src.utils.logger import get_logger
from config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
)

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = API_DESCRIPTION,
)

# CORS Middleware
# Allows frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Temporary folder for uploaded images
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# HEALTH CHECK ENDPOINT
# GET /health
# Used to verify API is running
@app.get("/health", response_model=HealthResponse)
def health_check():
    logger.info("Health check called")
    return HealthResponse(
        status  = "ok",
        version = API_VERSION,
        message = f"{API_TITLE} is running successfully"
    )


# ROOT ENDPOINT
# GET /
@app.get("/")
def root():
    return {
        "name":    API_TITLE,
        "version": API_VERSION,
        "status":  "running",
        "docs":    "/docs"
    }


# MAIN ESTIMATION ENDPOINT
# POST /estimate
# Upload image → get damage report + cost
@app.post("/estimate", response_model=EstimationResponse)
async def estimate_damage(file: UploadFile = File(...)):
    """
    Upload a car image to get:
    - Detected damage types
    - Affected car parts
    - Severity assessment
    - Repair vs replace recommendation
    - Cost estimate in PKR
    """

    # STEP 1: Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type: {file.content_type}. "
                          f"Please upload JPG, PNG or WEBP image."
        )

    # STEP 2: Save uploaded file temporarily
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path       = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Image saved temporarily: {temp_path}")

        # STEP 3: Run inference pipeline
        result = run_pipeline(temp_path)

        # STEP 4: Handle pipeline errors
        if not result["success"]:
            raise HTTPException(
                status_code = 422,
                detail      = result.get("error", "Pipeline failed")
            )

        logger.info("Estimation completed successfully")
        return EstimationResponse(**result)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code = 500,
            detail      = f"Internal server error: {str(e)}"
        )

    finally:
        # STEP 5: Always delete temp file
        # Even if something crashes above
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temp file deleted: {temp_path}")
            
# SERVE RESULT IMAGE
# GET /result-image/{filename}
@app.get("/result-image/{filename}")
async def get_result_image(filename: str, background_tasks: BackgroundTasks):
    """
    Returns the before/after comparison image.
    Automatically deletes file after serving.
    """
    file_path = os.path.join("temp_outputs", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete file after sending
    background_tasks.add_task(os.remove, file_path)

    return FileResponse(
        path         = file_path,
        media_type   = "image/jpeg",
        filename     = filename
    )


# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    logger.info(f"Starting {API_TITLE} on {HOST}:{PORT}")
    uvicorn.run(
        "api.main:app",
        host    = HOST,
        port    = PORT,
        reload  = True
    )