import os

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torch
from tqdm import tqdm
from classify_gaze_IVT import classify_raw_IVT
from metrics_eval import calculate_similarity, calculate_similarity_N, create_dataframe
from plotting import analyze_potential_continuity, analyze_saliency_continuity, save_and_plot_everything
from mala import MALA
from mala_cauchy import MALA_Cauchy
from ula_mala import saliency_to_potential_EM
from ula import ULA
from ula_mala import metropolis_adjusted_langevin_algorithm_cauchy, metropolis_adjusted_langevin_algorithm, saliency_to_potential, unadjusted_langevin_algorithm
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
    #plt.plot(trajectory[:, 0], trajectory[:, 1], label="Traiettoria rumorosa", linestyle="dashed", alpha=0.6)
    #plt.plot(smoothed_trajectory[:, 0], smoothed_trajectory[:, 1], label="Traiettoria filtrata", linewidth=2)
    #plt.legend()
    #plt.show()


    # Compute velocities
    velocities = np.sqrt(np.sum(np.diff(smoothed_trajectory, axis=0) ** 2, axis=1)) / (1.0 / gaze_sampling_rate)
    average_velocity = np.mean(velocities)

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


def run(input_dir, output_dir, task, name):

    img_path = os.path.join(input_dir, task, name)
    print(f"Processing image: {img_path}")

    """
    Process a single image: compute saliency, generate MALA trajectory, detect fixations, and visualize results.
    Args:
        img_path (str): Path to the input image.
        output_dir (str): Directory to save output visualizations.
        text (list): List of text descriptions for object detection.
    """
    use_reshaped_map = True  # Cambia a False per usare la mappa originale

    text = [task]
    
    img = load_and_preprocess_image(img_path)

    obj_map = get_obj_map(img, text)
    obj_map = torch.tensor(obj_map[None, None, :, :]) 
    saliency_map = obj_map

    reshaped_saliency = T.Resize(size=13)(saliency_map)

    ratio = [1.0,1.0]
    cov = [[0.5, 0], [0, 0.5]]

    if use_reshaped_map:

        img_size = reshaped_saliency.squeeze().shape
        mean = [img_size[0] // 2, img_size[1] // 2]
        potential_map = saliency_to_potential_EM(reshaped_saliency, n_components=3)
        ratio = np.array(img.shape[:2]) / np.array(reshaped_saliency.shape[-2:])
    else:
        img_size = saliency_map.squeeze().shape
        mean = [img_size[0] // 2, img_size[1] // 2]
        potential_map = saliency_to_potential_EM(saliency_map, n_components=3)

    # Parameters
    init_point = np.random.multivariate_normal(mean, cov, 1).astype(int)[0]
    exp_dur = 2# 
    gaze_sample_rate = 500 
    n_steps = int(exp_dur * gaze_sample_rate)  
    step_size = 0.01
    burn_in = 0

    #Param Cauchy Distr
    gamma = gamma= torch.tensor(0.1, dtype=torch.float32)

    screen_params = get_screen_params()
    
    ## Trajectories
    ula = ULA(
        potential_map=potential_map, 
        init_point=init_point, 
        n_steps=n_steps, 
        step_size=step_size, 
        burn_in=burn_in, 
        ratio=ratio
    )
    
    ula_trajectory = ula.simulate_scanpath()
    
    print(ula_trajectory)
    ula_trajectory = ula_trajectory[:, [1, 0]]
    '''
    mala_norm = MALA(
        potential_map=potential_map,
        init_point=init_point,
        n_steps=n_steps,
        step_size=step_size,
        burn_in=burn_in,
        ratio=ratio
    )

    mala_trajectory = mala_norm.simulate_scanpath()
    mala_trajectory = mala_trajectory[:, [1, 0]]
    
    mala_cauchy = MALA_Cauchy(
        potential_map, 
        init_point, 
        n_steps, 
        step_size, 
        burn_in, 
        ratio, 
        gamma
    )

    mala_cauchy_trajectory = mala_cauchy.simulate_scanpath()
    mala_cauchy_trajectory = mala_cauchy_trajectory[:, [1, 0]]
    '''
    fixations = process_fixations(ula_trajectory, gaze_sample_rate, screen_params)


    save_and_plot_everything(img, img_path, output_dir, potential_map, reshaped_saliency, ula_trajectory, fixations)


    return fixations, img
    
def evaluate_one_image():
    input_dir = "COCOSearch18-images-TP/images/"
    output_dir = "output/images/keyboard/"
    task = "keyboard"
    name = "000000006608.jpg"

    fixations, img = run(input_dir, output_dir, task, name)


def evaluation_pipeline_N(N):
    """
    Main function to process all images in the input directory and generate N simulated trajectories.

    Args:
        N (int): Number of simulated trajectories to generate per (name, task).
    """
    input_dir = "COCOSearch18-images-TP/images/"
    df = create_dataframe()
    df = df.sort_values(by=["name", "task", "subject"])
    #print(df)

    temp_df = pd.DataFrame(columns=df.columns)
    current_name, current_task = None, None 

    multimatch_df = pd.DataFrame(columns=['shape', 'direction', 'length', 'position', 'duration'])
    scanmatch_df = pd.DataFrame(columns=['ScanMatch'])
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        if current_name is None or (row["name"] == current_name and row["task"] == current_task):
            temp_df = pd.concat([temp_df, row.to_frame().T], ignore_index=True)
        else:
            #print(f"\nProcessing group (name={current_name}, task={current_task})")
            output_dir = f"output/images/{current_task}/"

            # Generiamo N traiettorie simulate
            for i in range(N):
                fixations, _ = run(input_dir, output_dir, current_task, current_name)
                fixations = np.array(fixations)

                X = fixations[:, 0]
                Y = fixations[:, 1]
                T = fixations[:, 2]
                length = len(X)

                new_row = {
                    "name": current_name,
                    "subject": 11 + i,  # Assegna un nuovo subject ID per ogni traiettoria simulata
                    "task": current_task,
                    "condition": temp_df["condition"].iloc[0],
                    "bbox": temp_df["bbox"].iloc[0],
                    "X": X,
                    "Y": Y,
                    "T": T,
                    "length": length,
                    "correct": 1,
                    "RT": 500,
                    "split": "valid"
                }
                
                temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

            #print(f"Generated {N} simulated trajectories for (name={current_name}, task={current_task})")
            #print(temp_df.tail(N+10))  # Log delle ultime N righe aggiunte (traiettorie simulate)

            # Calcoliamo la similarità
            MM_mean_similarity, SM_mean_similarity = calculate_similarity_N(temp_df)

            multimatch_df.loc[len(multimatch_df)] = MM_mean_similarity
            scanmatch_df.loc[len(scanmatch_df)] = SM_mean_similarity

            temp_df = row.to_frame().T
            current_name, current_task = row["name"], row["task"]

        if current_name is None:
            current_name, current_task = row["name"], row["task"]
    print("FINAL GROUP?")
    if not temp_df.empty:
        print(f"\nProcessing final group (name={current_name}, task={current_task})")
        output_dir = f"output/images/{current_task}/"

        for i in range(N):
            fixations, _ = run(input_dir, output_dir, current_task, current_name)
            fixations = np.array(fixations)

            X = fixations[:, 0]
            Y = fixations[:, 1]
            T = fixations[:, 2]
            length = len(X)

            new_row = {
                "name": current_name,
                "subject": 11 + i,
                "task": current_task,
                "condition": temp_df["condition"].iloc[0],
                "bbox": temp_df["bbox"].iloc[0],
                "X": X,
                "Y": Y,
                "T": T,
                "length": length,
                "correct": 1,
                "RT": 500,
                "split": "valid"
            }
            
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

        #print(f"Generated {N} simulated trajectories for (name={current_name}, task={current_task})")
        #print(temp_df.tail(N+10))

        MM_mean_similarity, SM_mean_similarity = calculate_similarity_N(temp_df)

        multimatch_df.loc[len(multimatch_df)] = MM_mean_similarity
        scanmatch_df.loc[len(scanmatch_df)] = SM_mean_similarity

    print("Final MultiMatch metrics:\n", multimatch_df.describe())
    print("Final ScanMatch metrics:\n", scanmatch_df.describe())