"""
Email classification models and training utilities.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

class EmailClassifier:
    """
    Email classifier using traditional machine learning.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type (str): Type of classification model to use
                Options: 'random_forest', 'naive_bayes', 'svm'
        """
        self.model_type = model_type
        self.model = None
        self.categories = None
        
        # Create model based on type
        if model_type == 'random_forest':
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, 
                    class_weight='balanced',  # Handle class imbalance
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                ))
            ])
        elif model_type == 'naive_bayes':
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', MultinomialNB())
            ])
        elif model_type == 'svm':
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                ('classifier', LinearSVC(
                    class_weight='balanced',  # Handle class imbalance
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, texts, labels, test_size=0.2):
        """
        Train the classifier on the given data.
        
        Args:
            texts (list): List of email texts (preprocessed)
            labels (list): List of email categories
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        # Store unique categories
        self.categories = list(set(labels))
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, 
            stratify=labels  # Use stratified sampling to maintain class distribution
        )
        
        # Calculate class weights for imbalanced dataset
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Get unique classes
        classes = np.unique(y_train)
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        
        # Create class weight dictionary
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        
        # Update classifier with class weights if it supports it
        if hasattr(self.model.named_steps['classifier'], 'class_weight'):
            self.model.named_steps['classifier'].class_weight = class_weight_dict
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        # Generate classification report
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate per-class metrics
        from sklearn.metrics import confusion_matrix
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=self.categories)
        
        # Calculate precision, recall, f1-score for each class
        class_metrics = {}
        for i, category in enumerate(self.categories):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(cm[i, :].sum())
            }
        
        # Prepare metrics
        metrics = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "classification_report": report,
            "class_metrics": class_metrics,
            "num_samples": len(texts),
            "num_categories": len(self.categories),
            "class_distribution": {category: labels.count(category) for category in self.categories}
        }
        
        return metrics
    
    def predict(self, text):
        """
        Predict the category of an email.
        
        Args:
            text (str): Email text (preprocessed)
            
        Returns:
            str: Predicted category
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make prediction
        return self.model.predict([text])[0]
    
    def save(self, file_path="model/email_classifier.pkl"):
        """
        Save the trained model to a file.
        
        Args:
            file_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model and categories
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'categories': self.categories,
                'model_type': self.model_type
            }, f)
    
    @classmethod
    def load(cls, file_path="model/email_classifier.pkl"):
        """
        Load a trained model from a file.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            EmailClassifier: Loaded classifier
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load the model and metadata
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.categories = data['categories']
        
        return classifier


def train_model_from_dataset(dataset_path, model_type='random_forest', masked=True):
    """
    Train a classification model from the dataset.
    
    Args:
        dataset_path (str): Path to the dataset CSV
        model_type (str): Type of model to train
        masked (bool): Whether to use masked emails for training
        
    Returns:
        tuple: (EmailClassifier, dict) - Trained classifier and metrics
    """
    from utils import PIIMasker, preprocess_text
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Map dataset columns to expected names
    email_column = 'email' if 'email' in df.columns else 'email_body'
    category_column = 'type' if 'type' in df.columns else 'category'
    
    # Check required columns
    if email_column not in df.columns or category_column not in df.columns:
        raise ValueError(f"Dataset must contain email content column ({email_column}) and category column ({category_column})")
    
    # Initialize PII masker if needed
    masker = PIIMasker() if masked else None
    
    # Prepare data for training
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        email_body = row[email_column]
        category = row[category_column]
        
        # Mask PII if required
        if masked:
            result = masker.process_email(email_body)
            email_body = result['masked_email']
        
        # Preprocess text
        preprocessed_text = preprocess_text(email_body)
        
        texts.append(preprocessed_text)
        labels.append(category)
    
    # Create and train the classifier
    classifier = EmailClassifier(model_type=model_type)
    metrics = classifier.train(texts, labels)
    
    # Save the trained model
    classifier.save()
    
    return classifier, metrics


if __name__ == "__main__":
    # Example usage for testing
    dataset_path = "combined_emails_with_natural_pii.csv"
    
    if os.path.exists(dataset_path):
        print(f"Training model on dataset: {dataset_path}")
        classifier, metrics = train_model_from_dataset(dataset_path)
        
        print(f"Model trained successfully:")
        print(f"  Training accuracy: {metrics['train_accuracy']:.2f}")
        print(f"  Testing accuracy: {metrics['test_accuracy']:.2f}")
        print(f"  Number of categories: {metrics['num_categories']}")
    else:
        print(f"Dataset not found: {dataset_path}")