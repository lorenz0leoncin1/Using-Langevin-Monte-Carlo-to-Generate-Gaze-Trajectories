import torch
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from scipy.ndimage import zoom, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt


def smooth_map(sal):
    """
    Applies Gaussian smoothing to a saliency map.

    Args:
        sal (numpy.ndarray): Input saliency map.

    Returns:
        numpy.ndarray: Smoothed and normalized saliency map.
    """
    sigma = 3 / 0.039
    Z = gaussian_filter(sal, sigma=sigma)
    Z /= np.max(Z) if np.max(Z) > 0 else 1  # Avoid division by zero
    return Z


def get_obj_map(img, text):
    """
    Computes the saliency map for a given image and text query using CLIPSeg.

    Args:
        img (numpy.ndarray or PIL.Image): Input image.
        text (list of str): List of text queries.

    Returns:
        numpy.ndarray: Processed saliency map.
    """
    # Load CLIPSeg processor and model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    # Prepare input for the model
    inputs = processor(text=text, images=[img] * len(text), return_tensors="pt")

    # Predict saliency maps
    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = outputs.logits.unsqueeze(1)  # Shape: (batch_size, 1, H, W)

    # Initialize the saliency map
    salmap = np.zeros((352, 352), dtype=np.float32)

    # Aggregate predictions for all text inputs
    for i in range(len(text)):
        salmap += torch.sigmoid(preds[i][0]).cpu().numpy()  # Convert tensor to NumPy
    
    # Normalize and refine the saliency map
    for _ in range(40):
        salmap = np.clip(salmap - salmap.mean(), 0, np.inf)

    # Resize to match input image dimensions
    salmap = zoom(
        salmap,
        (img.shape[0] / salmap.shape[0], img.shape[1] / salmap.shape[1]),
        order=1,  # Bilinear interpolation
    )

    # Apply smoothing and final normalization
    salmap = smooth_map(salmap)
    salmap /= np.max(salmap) if np.max(salmap) > 0 else 1  # Avoid division by zero
    
    return salmap
