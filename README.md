# Email Classification System with PII Masking

This project implements an email classification system for a company's support team. The system categorizes incoming support emails into predefined categories (Incident, Request, Problem, Change) while ensuring that personal information (PII) is masked before processing.

## Dataset Insights

The dataset used contains:
- 24,000 emails with 4 categories: Incident (40%), Request (28%), Problem (21%), and Change (11%)
- PII types commonly found include:
  - Full names (most common)
  - Email addresses
  - Dates of birth
  - Phone numbers (less common)
  - Financial information (rarely found)

## Features

1. **Email Classification**: Classifies emails into categories using a machine learning model optimized for imbalanced data.
2. **PII Masking**: Detects and masks personal information without using LLMs, including:
   - Full names
   - Email addresses
   - Phone numbers
   - Dates of birth
   - Aadhar card numbers
   - Credit/Debit card numbers
   - CVV numbers
   - Card expiry numbers
3. **API Deployment**: Exposes the solution as a REST API following the required format

## Project Structure

```
email-classification-system/
├── app.py                  # Main FastAPI application
├── api.py                  # API endpoints
├── utils.py                # PII masking utilities
├── models.py               # Classification model
├── analyze_dataset.py      # Dataset analysis script
├── requirements.txt        # Dependencies
├── README.md               # This file
├── combined_emails_with_natural_pii.csv  # Dataset
└── model/                  # Directory for saved models
    └── email_classifier.pkl
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository or create the files according to the project structure:

```bash
git clone <repository-url>
cd email-classification-system
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the dataset file in the project directory:

```bash
# Ensure the file is named exactly as follows
cp /path/to/your/dataset combined_emails_with_natural_pii.csv
```

### Analyzing the Dataset

Before training the model, you can analyze the dataset to understand its structure:

```bash
python analyze_dataset.py
```

This will generate:
- A JSON file with dataset statistics
- Plots showing the distribution of categories and PII types

### Training the Model

Train the classification model with:

```bash
python models.py
```

Alternatively, the model will be automatically trained on the first API request if it doesn't exist.

### Running the Application

Start the FastAPI server:

```bash
python app.py
```

The server will run at http://localhost:8000 by default.

## Using the API

### API Endpoints

1. **Classify Email**
   - Endpoint: `POST /classify`
   - Request body:
     ```json
     {
       "email_body": "string containing the email"
     }
     ```
   - Response:
     ```json
     {
       "input_email_body": "string containing the email",
       "list_of_masked_entities": [
         {
           "position": [start_index, end_index],
           "classification": "entity_type",
           "entity": "original_entity_value"
         }
       ],
       "masked_email": "string containing the masked email",
       "category_of_the_email": "string containing the class"
     }
     ```

2. **Train Model**
   - Endpoint: `POST /train-model`
   - Request body:
     ```json
     {
       "model_type": "random_forest",
       "force_retrain": false
     }
     ```
   - Response:
     ```json
     {
       "message": "Model training started in the background. Check back later."
     }
     ```

3. **Model Status**
   - Endpoint: `GET /model-status`
   - Response: Information about the current model

### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment to Hugging Face Spaces

To deploy the application to Hugging Face Spaces:

1. Create a new Space on Hugging Face (https://huggingface.co/spaces)
2. Choose the "Space SDK" as "Gradio"
3. Clone the Space repository
4. Copy your project files to the cloned repository
5. Push the changes to deploy

### Required Files for Hugging Face Spaces

Make sure to include:
- All Python files
- requirements.txt
- The dataset file

### Configuration for Hugging Face Spaces

Create an `app.py` file that uses the correct port for the environment:

```python
import os
import uvicorn
from fastapi import FastAPI
# ... rest of your code

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    uvicorn.run("app:app", host="0.0.0.0", port=port)
```

## Technical Details

### PII Masking

The system uses regular expressions to detect various types of PII without relying on LLMs. The masking process:
1. Detects entities using regex patterns
2. Records positions and values of detected entities
3. Replaces entities with their respective masks (e.g., `[full_name]`, `[email]`)
4. Maintains a mapping for potential demasking (if needed)

### Email Classification

The classification component uses machine learning to categorize emails:
- Default model: Random Forest with TF-IDF features
- Alternative models: Naive Bayes, SVM
- Model is trained on masked emails to ensure it works with PII-masked content

## Evaluation

The system is evaluated based on:
1. API Deployment: Correctly deployed and accessible
2. Code Quality: Adherence to PEP8 guidelines
3. API Format: Following the strict output format requirements
4. Test Case Coverage: Handling various email formats and PII patterns

## Limitations and Future Improvements

- The current PII detection relies on regex patterns, which may miss some complex or unusual PII formats
- More advanced NER models could be integrated for better PII detection (while still avoiding LLMs)
- The classification could be improved with more sophisticated features or model architectures