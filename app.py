"""
Main FastAPI application for email classification system.
"""
import os
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json

from utils import PIIMasker, preprocess_text
from models import EmailClassifier, train_model_from_dataset

# Create FastAPI app
app = FastAPI(
    title="Email Classification System",
    description="API for classifying emails with PII masking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure necessary directories exist
os.makedirs("model", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Define email request model
class EmailRequest(BaseModel):
    email_body: str

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with basic API information and testing form.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Email Classification System with PII Masking</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; background-color: #f5f8fa; }
                .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 25px; }
                p { color: #555; }
                form { margin: 20px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px; background-color: #f9f9f9; }
                textarea { width: 100%; min-height: 140px; margin-bottom: 15px; padding: 12px; border: 1px solid #ddd; border-radius: 4px; font-family: inherit; font-size: 14px; }
                button { background: #2980b9; color: white; border: none; padding: 12px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
                button:hover { background: #3498db; }
                .result { margin-top: 25px; padding: 20px; background: #f0f7fb; border-left: 5px solid #3498db; border-radius: 3px; }
                pre { background: #f5f5f5; padding: 15px; overflow: auto; border-radius: 4px; font-size: 14px; line-height: 1.5; }
                .examples { margin-top: 25px; }
                .example { background: #fffbf0; padding: 10px; margin: 10px 0; border-left: 3px solid #f39c12; cursor: pointer; }
                .footer { margin-top: 30px; font-size: 12px; color: #7f8c8d; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Email Classification System with PII Masking</h1>
                <p>This application automatically classifies support emails into four categories (Incident, Request, Problem, Change) while protecting personal information. It uses machine learning to determine the email category and masks sensitive data like names, email addresses, and other PII before processing.</p>
                
                <h2>Test the Classifier:</h2>
                <form id="testForm">
                    <div>
                        <label for="email_body"><strong>Enter email text:</strong></label>
                        <textarea id="email_body" name="email_body" placeholder="Enter an email to classify..."></textarea>
                    </div>
                    <button type="submit">Classify Email</button>
                </form>
                
                <div id="result" class="result" style="display: none;">
                    <h3>Classification Result:</h3>
                    <pre id="resultJson"></pre>
                </div>
                
                <div class="examples">
                    <h2>Example Emails:</h2>
                    <div class="example" onclick="useExample(0)">
                        Account locked - can't login (Incident)
                    </div>
                    <div class="example" onclick="useExample(1)">
                        Request for system access (Request)
                    </div>
                    <div class="example" onclick="useExample(2)">
                        System performance issue (Problem)
                    </div>
                    <div class="example" onclick="useExample(3)">
                        Update contact information (Change)
                    </div>
                </div>
                
                <div class="footer">
                    <p>For API documentation, visit <a href="/docs">/docs</a></p>
                </div>
                
                <script>
                    const examples = [
                        "Hello, my name is John Smith. My email is john.smith@example.com. I am experiencing an issue with my account. It appears to be locked and I cannot log in. Please help me resolve this issue urgently.",
                        "Hi team, I'm Sarah Johnson. I need access to the CRM system for my new role. My email is sarah.j@example.org. Please let me know what information you need from me.",
                        "The payment processing system is very slow. Reports that normally take 2 minutes are taking over 20 minutes to generate. This is Michael Chen from Operations (mike.chen@example.com). Can you investigate this performance problem?",
                        "Please update my department from Marketing to Sales in the HR system. This change is effective from next Monday. Thanks, Jennifer Lopez, ext. 3456, jennifer.lopez@example.net"
                    ];
                    
                    function useExample(index) {
                        document.getElementById('email_body').value = examples[index];
                    }
                    
                    document.getElementById('testForm').addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        const emailBody = document.getElementById('email_body').value;
                        document.getElementById('result').style.display = 'none';
                        
                        try {
                            const response = await fetch('/classify', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({ email_body: emailBody }),
                            });
                            
                            const data = await response.json();
                            
                            document.getElementById('resultJson').textContent = JSON.stringify(data, null, 2);
                            document.getElementById('result').style.display = 'block';
                        } catch (error) {
                            console.error('Error:', error);
                            document.getElementById('resultJson').textContent = 'Error: ' + error.message;
                            document.getElementById('result').style.display = 'block';
                        }
                    });
                </script>
            </div>
        </body>
    </html>
    """
    return html_content

@app.post("/classify", response_model=dict)
async def classify_email(request: EmailRequest):
    """
    Main endpoint for email classification and PII masking.
    
    Args:
        request (EmailRequest): Email to classify
        
    Returns:
        dict: Classification results with PII masking in the exact format required by the assignment
    """
    try:
        # Initialize components
        pii_masker = PIIMasker()
        
        # Check if model exists
        model_path = "model/email_classifier.pkl"
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
        
        # Process the email
        email_body = request.email_body
        
        # Detect and mask PII
        masking_result = pii_masker.process_email(email_body)
        masked_email = masking_result["masked_email"]
        entities = masking_result["list_of_masked_entities"]
        
        # Preprocess masked email for classification
        preprocessed_email = preprocess_text(masked_email)
        
        # Classify the email
        category = classifier.predict(preprocessed_email)
        
        # Prepare response in EXACTLY the format required by the assignment
        response = {
            "input_email_body": email_body,
            "list_of_masked_entities": entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        return response
        
    except Exception as e:
        print(f"Error processing email: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing email: {str(e)}"
        )

# Startup event to check dataset and model
@app.on_event("startup")
async def startup_event():
    """
    Run at application startup to check for dataset and model.
    """
    # Check for dataset
    dataset_path = "combined_emails_with_natural_pii.csv"
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
    else:
        print(f"Dataset found: {dataset_path}")
        # Check dataset columns
        try:
            df = pd.read_csv(dataset_path)
            columns = df.columns.tolist()
            print(f"Dataset columns: {columns}")
            # Map to our expected names
            email_column = 'email' if 'email' in columns else 'email_body'
            category_column = 'type' if 'type' in columns else 'category'
            print(f"Using {email_column} for email content and {category_column} for categories")
        except Exception as e:
            print(f"Error checking dataset: {e}")
    
    # Check for model
    model_path = "model/email_classifier.pkl"
    if os.path.exists(model_path):
        print(f"Model found: {model_path}")
    else:
        print(f"Model not found, will train on first request if dataset is available.")

# Run the application
if __name__ == "__main__":
    # Use the port that Hugging Face Spaces expects (7860)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)