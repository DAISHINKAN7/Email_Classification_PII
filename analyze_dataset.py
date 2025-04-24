"""
Script to analyze the email dataset and detect PII patterns.
"""
import pandas as pd
import re
import json
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define PII regex patterns
PII_PATTERNS = {
    "full_name": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone_number": r'\b(\+\d{1,3}[-.\s]?)?(\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    "dob": r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2} [A-Za-z]+ \d{2,4})\b',
    "aadhar_num": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "credit_debit_no": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "cvv_no": r'\b[Cc][Vv][Vv][-:;]?\s*\d{3,4}\b|\b\d{3,4}\s*[Cc][Vv][Vv]\b',
    "expiry_no": r'\b(0[1-9]|1[0-2])[/\-](20)?[0-9]{2}\b'
}

def detect_pii(text):
    """
    Detect PII in a given text using regex patterns.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary with PII types and counts
    """
    pii_counts = {}
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        pii_counts[pii_type] = len(matches)
    
    return pii_counts

def analyze_dataset(file_path):
    """
    Analyze the email dataset for structure, categories, and PII.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Analysis results
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Basic dataset information
        num_rows = len(df)
        columns = df.columns.tolist()
        print(f"Columns in dataset: {columns}")
        
        # Map the dataset columns to our expected column names
        email_column = 'email' if 'email' in columns else 'email_body'
        category_column = 'type' if 'type' in columns else 'category'
        
        # Check for required columns
        if email_column not in columns:
            print("WARNING: No email content column found in dataset")
            return None
        
        if category_column not in columns:
            print("WARNING: No category column found in dataset")
            # If no category column, we'll need to infer categories or use a default set
        
        # Analyze email categories if available
        category_distribution = None
        if category_column in columns:
            category_distribution = df[category_column].value_counts().to_dict()
            print("\nEmail Category Distribution:")
            for category, count in category_distribution.items():
                print(f"  {category}: {count} ({count/num_rows:.1%})")
        
        # Analyze email lengths
        df['email_length'] = df[email_column].apply(len)
        avg_length = df['email_length'].mean()
        min_length = df['email_length'].min()
        max_length = df['email_length'].max()
        print(f"\nEmail Length Analysis:")
        print(f"  Average: {avg_length:.1f} characters")
        print(f"  Min: {min_length} characters")
        print(f"  Max: {max_length} characters")
        
        # Analyze PII presence
        print("\nAnalyzing PII in emails...")
        all_pii_counts = {pii_type: 0 for pii_type in PII_PATTERNS.keys()}
        emails_with_pii = 0
        
        # Analyze a sample of emails (or all if dataset is small)
        sample_size = min(num_rows, 1000)
        sample_df = df.sample(sample_size) if num_rows > 1000 else df
        
        for _, row in sample_df.iterrows():
            email_body = row[email_column]
            pii_counts = detect_pii(email_body)
            
            # Update overall counts
            for pii_type, count in pii_counts.items():
                all_pii_counts[pii_type] += count
            
            # Count emails with any PII
            if sum(pii_counts.values()) > 0:
                emails_with_pii += 1
        
        # Calculate percentage of emails with PII
        pii_percentage = (emails_with_pii / sample_size) * 100
        print(f"\nPII Detection Results (based on {sample_size} sample emails):")
        print(f"  Emails containing PII: {emails_with_pii} ({pii_percentage:.1f}%)")
        print("  PII types detected:")
        for pii_type, count in all_pii_counts.items():
            print(f"    {pii_type}: {count} instances")
        
        # Create plots directory if it doesn't exist
        os.makedirs("plots", exist_ok=True)
        
        # Plot category distribution if available
        if category_distribution:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(category_distribution.keys()), 
                        y=list(category_distribution.values()))
            plt.title('Email Category Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('plots/category_distribution.png')
            print("\nCategory distribution plot saved to plots/category_distribution.png")
        
        # Plot PII distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(all_pii_counts.keys()), y=list(all_pii_counts.values()))
        plt.title('PII Type Distribution')
        plt.xlabel('PII Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/pii_distribution.png')
        print("PII distribution plot saved to plots/pii_distribution.png")
        
        # Return analysis results
        return {
            "num_samples": num_rows,
            "columns": columns,
            "category_distribution": category_distribution,
            "email_length_stats": {
                "average": avg_length,
                "min": min_length,
                "max": max_length
            },
            "pii_analysis": {
                "emails_with_pii_percentage": pii_percentage,
                "pii_counts": all_pii_counts
            }
        }
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return None

if __name__ == "__main__":
    dataset_path = "combined_emails_with_natural_pii.csv"
    
    if os.path.exists(dataset_path):
        print(f"Analyzing dataset: {dataset_path}")
        results = analyze_dataset(dataset_path)
        
        if results:
            # Save analysis results to file
            with open("dataset_analysis.json", "w") as f:
                json.dump(results, f, indent=2)
            print("\nAnalysis complete. Results saved to dataset_analysis.json")
        else:
            print("\nAnalysis failed. Check errors above.")
    else:
        print(f"Error: Dataset file not found at {dataset_path}")