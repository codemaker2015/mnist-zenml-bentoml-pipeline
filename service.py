import typing as t
import numpy as np
from PIL.Image import Image as PILImage
import bentoml
import torch
import warnings
from model import SimpleConvNet

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

@bentoml.service()
class MNISTPyTorchService:
    
    def __init__(self):
        # Add the model class to safe globals for PyTorch 2.6+
        torch.serialization.add_safe_globals([SimpleConvNet])
        
        # Temporarily patch torch.load to use weights_only=False
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **kwargs)
        
        try:
            # Load model using deprecated API (still works)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = bentoml.pytorch.load_model("pytorch_mnist")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        finally:
            # Restore original torch.load
            torch.load = original_load
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
    
    @bentoml.api
    def predict(self, inp: np.ndarray) -> np.ndarray:
        """Predict from numpy array (28, 28) shape"""
        # Validate input
        if inp.shape != (28, 28):
            raise ValueError(f"Expected shape (28, 28), got {inp.shape}")
        
        # Prepare input for model
        inp = inp.astype(np.float32)
        inp = np.expand_dims(np.expand_dims(inp, 0), 0)  # Shape: (1, 1, 28, 28)
        
        # Convert to tensor
        inp_tensor = torch.from_numpy(inp)
        
        # Run inference
        with torch.no_grad():
            output = self.model(inp_tensor)
            # Get the predicted class
            predicted = torch.argmax(output, dim=1)
        
        return predicted.numpy()
    
    @bentoml.api
    def predict_image(self, image: PILImage) -> np.ndarray:
        """Predict from PIL Image"""
        arr = np.array(image)
        
        # Validate shape
        if arr.shape != (28, 28):
            raise ValueError(f"Expected image size 28x28, got {arr.shape}")
        
        return self.predict(arr)
    
    @bentoml.api
    def health(self) -> dict:
        """Health check"""
        return {"status": "ok", "model_loaded": True}