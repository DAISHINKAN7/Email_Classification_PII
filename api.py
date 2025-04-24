"""
API endpoints for the email classification system.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import json

from utils import PIIMasker, preprocess_text
from models import EmailClassifier, train_model_from_dataset

# Create the router
router = APIRouter()

# Initialize the PII masker
pii_masker = PIIMasker()

# Try to load the classifier
classifier = None
model_path = "model/email_classifier.pkl"
try:
    if os.path.exists(model_path):
        classifier = EmailClassifier.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Request and response models
class EmailRequest(BaseModel):
    email_body: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: list
    masked_email: str
    category_of_the_email: str

class TrainingRequest(BaseModel):
    model_type: str = "random_forest"
    force_retrain: bool = False

@router.post("/classify")
async def classify_email(request: EmailRequest):
    """
    Classify an email and mask PII.
    
    Args:
        request (EmailRequest): Email to classify
    
    Returns:
        dict: Classification result with masked PII
    """
    global classifier
    
    # Check if classifier is loaded
    if not classifier:
        try:
            if os.path.exists(model_path):
                classifier = EmailClassifier.load(model_path)
            else:
                # Train a new model if dataset is available
                dataset_path = "combined_emails_with_natural_pii.csv"
                if os.path.exists(dataset_path):
                    classifier, _ = train_model_from_dataset(dataset_path)
                else:
                    raise HTTPException(
                        status_code=500, 
                        detail="No trained model available and dataset not found."
                    )
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error loading/training model: {str(e)}"
            )
    
    try:
        # Mask PII
        masking_result = pii_masker.process_email(request.email_body)
        masked_email = masking_result["masked_email"]
        entities = masking_result["list_of_masked_entities"]
        
        # Preprocess masked email for classification
        preprocessed_email = preprocess_text(masked_email)
        
        # Classify the email
        category = classifier.predict(preprocessed_email)
        
        # Prepare response
        response = {
            "input_email_body": request.email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing email: {str(e)}"
        )

@router.post("/train-model")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain the email classification model.
    
    Args:
        request (TrainingRequest): Training request
        background_tasks (BackgroundTasks): Background tasks
        
    Returns:
        dict: Status message
    """
    global classifier
    
    # Check if model exists and should not be retrained
    if os.path.exists(model_path) and not request.force_retrain:
        return {"message": "Model already exists. Use force_retrain=true to retrain."}
    
    # Check if dataset exists
    dataset_path = "combined_emails_with_natural_pii.csv"
    if not os.path.exists(dataset_path):
        raise HTTPException(
            status_code=404,
            detail=f"Dataset not found: {dataset_path}"
        )
    
    try:
        # Train in the background to avoid blocking the API
        def train_model_task():
            global classifier
            classifier, metrics = train_model_from_dataset(
                dataset_path, model_type=request.model_type
            )
            # Save metrics for reference
            with open("model/training_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        
        background_tasks.add_task(train_model_task)
        
        return {
            "message": "Model training started in the background. Check back later."
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@router.get("/model-status")
async def model_status():
    """
    Check the status of the model.
    
    Returns:
        dict: Model status
    """
    global classifier
    
    if os.path.exists(model_path):
        # Load model if not already loaded
        if not classifier:
            try:
                classifier = EmailClassifier.load(model_path)
            except Exception as e:
                return {
                    "model_exists": True,
                    "model_loaded": False,
                    "error": str(e)
                }
        
        # Check for training metrics
        metrics_path = "model/training_metrics.json"
        metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        
        return {
            "model_exists": True,
            "model_loaded": classifier is not None,
            "model_type": classifier.model_type if classifier else None,
            "categories": classifier.categories if classifier else None,
            "metrics": metrics
        }
    else:
        return {
            "model_exists": False,
            "model_loaded": False
        }