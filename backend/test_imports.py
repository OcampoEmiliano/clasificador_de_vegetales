"""Quick test to verify all modules import correctly."""
try:
    import torch
    import torchvision
    import fastapi
    import uvicorn
    from PIL import Image
    from classes import CLASSES
    from model import MLPClassifier, INPUT_SIZE, NUM_CLASSES, transform

    model_check = MLPClassifier(INPUT_SIZE, NUM_CLASSES)
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print(f"Transform: {transform}")
    print(f"Model parameters: {sum(p.numel() for p in model_check.parameters()):,}")
    print("All imports OK - backend ready!")
except Exception as e:
    print(f"ERROR: {e}")
    import sys
    sys.exit(1)
