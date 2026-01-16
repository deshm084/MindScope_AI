import torch
import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_fusion import MindScopeFusionModel

def test_model_output_shape():
    """
    Verifies that the fusion model produces the correct output shape (Batch_Size, Num_Classes)
    """
    batch_size = 4
    num_classes = 3
    num_tab_features = 4
    
    # Initialize model
    model = MindScopeFusionModel(num_tabular_features=num_tab_features, num_classes=num_classes)
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 64)) # Fake text tokens
    dummy_mask = torch.ones((batch_size, 64))
    dummy_tab = torch.rand((batch_size, num_tab_features)) # Fake numbers
    
    # Forward pass
    output = model(dummy_input_ids, dummy_mask, dummy_tab)
    
    # Check shape
    assert output.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {output.shape}"
    print("Shape Test Passed!")

def test_risk_logic():
    """
    Sanity check: Does high confidence in one class imply a valid probability distribution?
    """
    # Create a random logit output
    logits = torch.tensor([[0.1, 0.9, 0.1]]) # Should correspond to class 1
    probs = torch.softmax(logits, dim=1)
    
    assert torch.sum(probs).item() > 0.99, "Probabilities should sum to approx 1.0"
    assert torch.argmax(probs).item() == 1, "Argmax logic failed"
    print("Logic Test Passed!")

if __name__ == "__main__":
    test_model_output_shape()
    test_risk_logic()
