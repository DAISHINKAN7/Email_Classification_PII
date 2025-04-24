"""
Utility functions for PII detection and masking.
"""
import re
import json

class PIIMasker:
    """
    Class for PII detection and masking without using LLMs.
    """
    
    def __init__(self):
        """
        Initialize the PII masker with regex patterns for different PII types.
        """
        # Define regex patterns for PII detection
        self.patterns = {
            # Enhanced pattern for names - handles more name formats
            "full_name": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b|\b([A-Z][a-z]+\s+[A-Z]\.)\b',
            
            # Enhanced email pattern
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            
            # Enhanced phone pattern - handles international formats better
            "phone_number": r'\b(?:\+\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
            
            # Enhanced DOB pattern - handles more date formats
            "dob": r'\b(?:0?[1-9]|[12]\d|3[01])[-/\s.](?:0?[1-9]|1[0-2])[-/\s.](?:19|20)\d{2}\b|\b(?:19|20)\d{2}[-/\s.](?:0?[1-9]|1[0-2])[-/\s.](?:0?[1-9]|[12]\d|3[01])\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b',
            
            # Other PII patterns (less common in your dataset but still needed)
            "aadhar_num": r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            "credit_debit_no": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "cvv_no": r'\b[Cc][Vv][Vv][-:;]?\s*\d{3,4}\b|\b\d{3,4}\s*[Cc][Vv][Vv]\b',
            "expiry_no": r'\b(0[1-9]|1[0-2])[/\-](20)?[0-9]{2}\b'
        }
        
        # Store mapping of masked to original values for demasking
        self.mask_to_original = {}
    
    def detect_pii(self, text):
        """
        Detect PII entities in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of detected PII entities with position, type and value
        """
        entities = []
        
        # Apply each pattern to detect PII
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.span()
                entity_value = match.group()
                
                # Add detected entity to the list
                entities.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": entity_value
                })
        
        # Sort entities by start position
        entities.sort(key=lambda x: x["position"][0])
        
        return entities
    
    def mask_text(self, text, entities):
        """
        Mask detected PII entities in text.
        
        Args:
            text (str): Original text
            entities (list): List of PII entities to mask
            
        Returns:
            str: Text with masked PII
        """
        # Create a working copy of the text
        masked_text = text
        
        # Keep track of position shifts due to masking
        offset = 0
        
        # Process entities in order of appearance
        for entity in sorted(entities, key=lambda x: x["position"][0]):
            start, end = entity["position"]
            entity_type = entity["classification"]
            original_value = entity["entity"]
            
            # Create the mask
            entity_mask = f"[{entity_type}]"
            
            # Adjust positions for previously applied masks
            start_adjusted = start + offset
            end_adjusted = end + offset
            
            # Replace the entity with its mask
            masked_text = masked_text[:start_adjusted] + entity_mask + masked_text[end_adjusted:]
            
            # Update the offset
            offset += len(entity_mask) - (end - start)
            
            # Store mapping for demasking
            if entity_mask not in self.mask_to_original:
                self.mask_to_original[entity_mask] = []
            self.mask_to_original[entity_mask].append(original_value)
        
        return masked_text
    
    def demask_text(self, masked_text):
        """
        Restore original PII in masked text (if needed for the application).
        
        Args:
            masked_text (str): Text with masked PII
            
        Returns:
            str: Text with original PII restored
        """
        demasked_text = masked_text
        
        # Restore each masked entity
        for mask, originals in self.mask_to_original.items():
            for original in originals:
                # Replace first occurrence only
                demasked_text = demasked_text.replace(mask, original, 1)
        
        return demasked_text
    
    def process_email(self, email_text):
        """
        Process an email: detect PII, mask it, and prepare the result.
        
        Args:
            email_text (str): Email text to process
            
        Returns:
            dict: Processing result with original email, masked email, and detected entities
        """
        # Reset mask-to-original mapping
        self.mask_to_original = {}
        
        # Detect PII entities
        entities = self.detect_pii(email_text)
        
        # Mask the text
        masked_text = self.mask_text(email_text, entities)
        
        # Prepare the result
        result = {
            "input_email_body": email_text,
            "list_of_masked_entities": entities,
            "masked_email": masked_text
        }
        
        return result


def preprocess_text(text):
    """
    Preprocess text for classification by normalizing, removing special characters,
    extra whitespace, etc.
    
    Args:
        text (str): Text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Replace email addresses (already masked) with generic token
    text = re.sub(r'\[email\]', 'EMAIL', text)
    
    # Replace names (already masked) with generic token
    text = re.sub(r'\[full_name\]', 'PERSON', text)
    
    # Replace other PII markers with generic tokens
    text = re.sub(r'\[phone_number\]', 'PHONE', text)
    text = re.sub(r'\[dob\]', 'BIRTHDATE', text)
    text = re.sub(r'\[aadhar_num\]', 'ID_NUMBER', text)
    text = re.sub(r'\[credit_debit_no\]', 'CARD_NUMBER', text)
    text = re.sub(r'\[cvv_no\]', 'CVV', text)
    text = re.sub(r'\[expiry_no\]', 'EXPIRY', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
    
    # Replace numbers with generic token
    text = re.sub(r'\b\d+\b', 'NUMBER', text)
    
    # Remove special characters except those in keywords that might be important
    text = re.sub(r'[^\w\s\-_]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text