"""
Script to run the entire email classification pipeline from analysis to API deployment.
"""
import os
import subprocess
import time

def main():
    """Run the email classification pipeline."""
    print("=" * 80)
    print("EMAIL CLASSIFICATION SYSTEM PIPELINE")
    print("=" * 80)
    
    # Check for dataset
    dataset_path = "combined_emails_with_natural_pii.csv"
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset file not found at '{dataset_path}'")
        print("Please place the dataset file in the project directory with this exact name.")
        return
    
    # Ensure directories exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Step 1: Analyze dataset
    print("\n1. ANALYZING DATASET")
    print("-" * 50)
    subprocess.run(["python", "analyze_dataset.py"], check=True)
    
    # Wait for plots to be generated
    time.sleep(2)
    
    # Check if plots were generated
    if os.path.exists("plots/category_distribution.png") and os.path.exists("plots/pii_distribution.png"):
        print("✓ Dataset analysis completed successfully")
        print("  - Plots saved to 'plots' directory")
    else:
        print("⚠ Warning: Analysis plots were not generated")
    
    # Step 2: Train model
    print("\n2. TRAINING CLASSIFICATION MODEL")
    print("-" * 50)
    subprocess.run(["python", "models.py"], check=True)
    
    # Check if model was created
    if os.path.exists("model/email_classifier.pkl"):
        print("✓ Model trained successfully")
        print("  - Model saved to 'model/email_classifier.pkl'")
    else:
        print("⚠ Warning: Model training may have failed")
    
    # Step 3: Start API server
    print("\n3. STARTING API SERVER")
    print("-" * 50)
    print("Starting the API server...")
    print("✓ API will be available at http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the server
    subprocess.run(["python", "app.py"], check=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")