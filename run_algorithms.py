


import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from classify_gaze_IVT import classify_raw_IVT
from mala import MALA
from mala_cauchy import MALA_Cauchy
from plotting import save_and_plot_everything
from saliency_to_potential import saliency_to_potential_EM
from ula import ULA
from zs_clip_seg import get_obj_map
from scipy.signal import savgol_filter

def load_and_preprocess_image(img_path):
    """
    Load an image from a given path, convert it to RGB format, and resize it if necessary.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image in RGB format.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Define standard shape
    typical_shape = (768, 1024, 3)

    # Resize while maintaining aspect ratio
    if img.shape != typical_shape:
        target_size = (768, 1024) if img.shape[0] > img.shape[1] else (1024, 768)
        img = cv2.resize(img, target_size)

    return img


def get_screen_params():
    """
    Retrieve screen parameters including resolution and pixel density.

    Returns:
        dict: Dictionary containing screen parameters.
    """
    screen_size_mm = np.array([28650, 50930])  # Fixed screen size (in mm)
    screen_res = np.array([1024, 768])  # Screen resolution

    return {
        'pix_per_mm': screen_res / screen_size_mm,  # Pixel density in pixels per mm
        'screen_dist_mm': 600,  # Fixed screen distance in mm
        'screen_res': screen_res  # Screen resolution
    }

def process_fixations(trajectory, gaze_sampling_rate, screen_params):
    """
    Processes a generated trajectory, calculates the average velocity, determines an adaptive threshold
    for fixation classification, and returns the results.

    Args:
        trajectory (numpy.ndarray): Simulated trajectory (N, 2) with (x, y) coordinates.
        gaze_sampling_rate (float): Gaze sampling frequency (Hz).
        screen_params (dict): Screen parameters.

    Returns:
        numpy.ndarray: Classified fixations with coordinates and duration.
    """
    
    # Smooth trajectory before velocity calculation
    smoothed_trajectory = savgol_filter(trajectory, window_length=5, polyorder=2, axis=0)


    # Compute velocities
    velocities = np.sqrt(np.sum(np.diff(smoothed_trajectory, axis=0) ** 2, axis=1)) / (1.0 / gaze_sampling_rate)

    # Compute a percentile-based threshold (adjusted to 50° percentile)
    adaptive_velocity_threshold = np.percentile(velocities, 50)

    # Classify fixations using the adaptive threshold
    fixations = classify_raw_IVT(trajectory, gaze_sampling_rate, screen_params, 
                                 vel_threshold=adaptive_velocity_threshold, min_fix_duration=0.15)  # Aumento a 150 ms

    # Add an initial fixation with a duration of 150 ms
    initial_fixation = np.array([[512, 384, 150]])
    
    if fixations.size == 0:
        print("For this image the adaptive threshold ahs to be set differently")
        print(f"Trajectory shape: {trajectory.shape}")  
        print(f"Fixations shape before vstack: {fixations.shape}")
        fixations = initial_fixation
    else:
        fixations = np.vstack((initial_fixation, fixations))


    return fixations

def process_image(input_dir, task, name, use_reshaped_map=True):
    """
    Processes an image by loading it, generating a saliency map, and creating a potential map.

    Args:
        input_dir (str): The directory where the images are stored.
        task (str): The task or object category for detection (e.g., "keyboard").
        name (str): The filename of the image to process.
        use_reshaped_map (bool, optional): Whether to use a reshaped saliency map (True) or the original saliency map (False). Defaults to True.

    Returns:
        tuple: A tuple containing the following:
            - img (Tensor): The processed image.
            - img_path (str): The path to the image.
            - saliency_map (Tensor): The original saliency map.
            - reshaped_saliency (Tensor): The reshaped saliency map (if applicable).
            - potential_map (Tensor): The potential map generated from the saliency map.
            - mean (list): The center coordinates of the image.
            - cov (list): The covariance matrix used for sampling.
            - ratio (np.ndarray): The ratio for resizing the saliency map based on the original image dimensions.
    """

    # Construct the main image path
    img_path = os.path.join(input_dir, task, name)

    # Check if the file exists
    if not os.path.exists(img_path):
        print(f"File not found at {img_path}, checking alternative location...")
        
        # Try looking for it directly in the 'data/' folder
        alt_path = os.path.join(input_dir, name)

        if os.path.exists(alt_path):
            print(f"Found image in {alt_path}, using this path instead.")
            img_path = alt_path
        else:
            raise FileNotFoundError(f"Image not found in either {img_path} or {alt_path}")

    print(f"Processing image: {img_path}")
    # Text description for object detection
    text = [task]
    
    # Load and preprocess the image
    img = load_and_preprocess_image(img_path)

    # Get the object map based on the image and text description
    obj_map = get_obj_map(img, text)
    obj_map = torch.tensor(obj_map[None, None, :, :])  # Convert to tensor
    saliency_map = obj_map

    # Resize the saliency map if needed
    reshaped_saliency = T.Resize(size=13)(saliency_map)

    # Initialize ratio and covariance for sampling
    ratio = [1.0, 1.0]
    cov = [[0.5, 0], [0, 0.5]]

    # Choose between reshaped or original saliency map
    if use_reshaped_map:
        img_size = reshaped_saliency.squeeze().shape
        mean = [img_size[0] // 2, img_size[1] // 2]  # Center of the image
        potential_map = saliency_to_potential_EM(reshaped_saliency, n_components=3)
        ratio = np.array(img.shape[:2]) / np.array(reshaped_saliency.shape[-2:])
    else:
        img_size = saliency_map.squeeze().shape
        mean = [img_size[0] // 2, img_size[1] // 2]
        potential_map = saliency_to_potential_EM(saliency_map, n_components=3)

    return img, img_path, saliency_map, reshaped_saliency, potential_map, mean, cov, ratio

def run_algorithm(input_dir, output_dir, task, name, algorithm, gamma):
    """
    Processes a single image to compute saliency, generate a gaze trajectory using the selected algorithm,
    detect fixations, and visualize the results.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save the output visualizations.
        task (str): The task or object category to be detected in the image.
        name (str): Name of the image to process.
        algorithm (str): Algorithm to use for gaze trajectory simulation ('ula', 'mala_norm', or 'mala_cauchy').

    Returns:
        tuple: A tuple containing the list of fixations and the image object.
    """

    #Set up all maps and values 
    img, img_path, saliency_map, reshaped_saliency, potential_map, mean, cov, ratio = process_image(input_dir, task, name)

    # Experiment Parameters ----------------------------------------------
    # Parameters for gaze trajectory simulation

    init_point = np.random.multivariate_normal(mean, cov, 1).astype(int)[0]  # Initial gaze point
    exp_dur = 3 # Expected duration of the gaze
    gaze_sample_rate = 500  # Sample rate for gaze data
    n_steps = int(exp_dur * gaze_sample_rate)  # Number of steps based on duration and sample rate
    step_size = 0.01  # Step size for trajectory simulation
    burn_in = 0  # Burn-in period for the algorithm

    # Selection of the algorithm ------------------------------------------
    # Select the algorithm for trajectory simulation
    
    if algorithm == 'ula':
        # ULA - Unadjusted Langevin Algorithm
        ula = ULA(
            potential_map=potential_map, 
            init_point=init_point, 
            n_steps=n_steps, 
            step_size=step_size, 
            burn_in=burn_in, 
            ratio=ratio
        )
        trajectory = ula.simulate_scanpath()
    
    elif algorithm == 'mala':
        # MALA - Metropolis-Adjusted Langevin Algorithm (Normal distribution)
        mala_norm = MALA(
            potential_map=potential_map, 
            init_point=init_point, 
            n_steps=n_steps, 
            step_size=step_size, 
            burn_in=burn_in, 
            ratio=ratio
        )
        trajectory = mala_norm.simulate_scanpath()
    
    elif algorithm == 'mala_cauchy':
        # MALA with Cauchy distribution
        if gamma is None:
            raise ValueError("Gamma parameter is required for MALA_Cauchy.")
        mala_cauchy = MALA_Cauchy(
            potential_map=potential_map, 
            init_point=init_point, 
            n_steps=n_steps, 
            step_size=step_size, 
            burn_in=burn_in, 
            ratio=ratio, 
            gamma = torch.tensor(gamma)
        )
        trajectory = mala_cauchy.simulate_scanpath()

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Adjust the trajectory (if needed) by swapping coordinates
    trajectory = trajectory[:, [1, 0]]

    # Get screen parameters (e.g., resolution)
    screen_params = get_screen_params()
    
    # Process fixations from the trajectory
    fixations = process_fixations(trajectory, gaze_sample_rate, screen_params)

    # Save and visualize the results (trajectory, fixations, saliency map, etc.)
    save_and_plot_everything(img, img_path, output_dir, potential_map, reshaped_saliency, trajectory, fixations)

    return fixations, img
