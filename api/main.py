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

# CORS Middleware (Allows frontend to talk to this API)
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


# Health check endpoint (to verify API is running)
@app.get("/health", response_model=HealthResponse)
def health_check():
    logger.info("Health check called")
    return HealthResponse(
        status  = "ok",
        version = API_VERSION,
        message = f"{API_TITLE} is running successfully"
    )


# Root endpoint
@app.get("/")
def root():
    return {
        "name":    API_TITLE,
        "version": API_VERSION,
        "status":  "running",
        "docs":    "/docs"
    }


# Main estimation endpoint (Upload image to get damage report and cost)
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

    # Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code = 400,
            detail      = f"Invalid file type: {file.content_type}. "
                          f"Please upload JPG, PNG or WEBP image."
        )

    # Save uploaded file temporarily
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path       = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Image saved temporarily: {temp_path}")

        # Run inference pipeline
        result = run_pipeline(temp_path)

        # Handle pipeline errors
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
        # Always delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Temp file deleted: {temp_path}")
            
# Serve result image
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
    
# Serve heatmap image
@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str, background_tasks: BackgroundTasks):
    """Returns the damage heatmap image."""
    file_path = os.path.join("temp_outputs", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Heatmap not found")

    background_tasks.add_task(os.remove, file_path)

    return FileResponse(
        path       = file_path,
        media_type = "image/jpeg",
        filename   = filename
    )
    
from src.models.cost_regresser import CostRegressor

# Initialize the regressor for evaluation
regressor = CostRegressor()

@app.post("/evaluate-accuracy")
async def evaluate_prediction(actual_manual_cost: float, file: UploadFile = File(...)):
    """
    Demonstration endpoint: Compares AI prediction with a Manual Manual Price.
    """
    # 1. Save and run pipeline (similar to your /estimate logic)
    unique_filename = f"eval_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    result = run_pipeline(temp_path)
    os.remove(temp_path) # Clean up

    if result["success"]:
        predicted_cost = result["total_cost_pkr"]
        
        # 2. CALL THE NEW METHOD WE ADDED TO COST_REGRESSER
        accuracy_report = CostRegressor.calculate_accuracy_metrics(predicted_cost, actual_manual_cost)
        
        return {
            "car_identified": result.get("car_model", "Unknown"),
            "ai_predicted_cost": predicted_cost,
            "manual_actual_cost": actual_manual_cost,
            "error_analysis": accuracy_report
        }
    
    raise HTTPException(status_code=422, detail="Pipeline failed during evaluation")


# Run server
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