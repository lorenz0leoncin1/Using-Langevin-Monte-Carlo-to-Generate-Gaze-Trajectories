import os
from matplotlib import patches, pyplot as plt
import numpy as np
import torch
from vis import compute_density_image, draw_scanpath

def plot_trajectory_on_image(image, trajectory, output_path):

    trajectory = np.array(trajectory)  
    plt.imshow(image, cmap="gray")
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c="red", s=5, alpha=0.6, label="Gaze Trajectory")
    plt.legend()
    plt.title("Trajectory on background image")
    plt.savefig(output_path)  # Save the figure to the output path
    plt.close()

def plot_gaze_direction(image, trajectory, output_path):
    trajectory = np.array(trajectory)  
    plt.imshow(image, cmap="gray")
    

    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    
    
    dx = end_x - start_x
    dy = end_y - start_y
    
    
    plt.arrow(
        start_x, start_y, dx, dy,
        color="red", width=0.5, head_width=5, head_length=5,
        length_includes_head=True, label="Direzione media"
    )
    
    plt.legend()
    plt.title("Gaze direction")
    plt.savefig(output_path)  
    plt.close()

def plot_gaze_with_steps(image, trajectory, step_interval, output_path):
    trajectory = np.array(trajectory)  
    plt.imshow(image, cmap="gray")
    
    
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c="red", s=5, alpha=0.6, label="Gaze Trajectory")
    
    for i in range(0, len(trajectory), step_interval):
        plt.scatter(trajectory[i, 0], trajectory[i, 1], c="blue", s=50, edgecolor="black")
        plt.text(trajectory[i, 0] + 5, trajectory[i, 1] - 5, str(i), color="white", fontsize=12)
    
    plt.legend()
    plt.title("Trajectory on image with jump numbers")
    plt.savefig(output_path)  
    plt.close()


def extract_fixations(trajectory, distance_threshold=5, time_per_step=40):
    fixations = []
    current_fixation = [trajectory[0]]
    for i in range(1, len(trajectory)):
        if np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i - 1])) < distance_threshold:
            current_fixation.append(trajectory[i])
        else:
            
            if len(current_fixation) > 1:
                duration = len(current_fixation) * time_per_step
                fixations.append((np.mean(current_fixation, axis=0), duration))
            current_fixation = [trajectory[i]]

    
    if len(current_fixation) > 1:
        duration = len(current_fixation) * time_per_step
        fixations.append((np.mean(current_fixation, axis=0), duration))
    
    return fixations

def plot_fixations(img, fixations, trajectory):
    plt.imshow(img)
    for fixation, duration in fixations:
        x, y = fixation
        plt.plot(x, y, marker='o', markersize=duration / 10, color='red', alpha=0.6)
    plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue', alpha=0.5)
    plt.title("Scanpath with Fixations")
    plt.show()

def analyze_saliency_continuity(saliency_map):
    
    saliency_map = saliency_map.squeeze()

    
    plt.figure(figsize=(6, 5))
    plt.imshow(saliency_map)
    plt.colorbar(label="Saliency Value")
    plt.title("Saliency Map")
    plt.axis("off")
    plt.show()


def analyze_potential_continuity(potential_map):
    """
    Analizza la continuità e la differenziabilità di una mappa del potenziale.

    Args:
        potential_map (torch.Tensor): La mappa del potenziale da analizzare.
    """
    # Assicurati che il tensore sia su CPU
    potential_map = potential_map.cpu()

    # Rimuovi le dimensioni extra (1, 1) se presenti
    potential_map = potential_map.squeeze()  # Rimuove le dimensioni di valore 1

    # Se il tensore richiede il gradiente, usa .detach() per separarlo dal grafo computazionale
    potential_map_detached = potential_map.detach()

    # 1. Visualizzazione della mappa del potenziale
    plt.figure(figsize=(6, 5))
    plt.imshow(potential_map_detached.numpy(), cmap="jet")  # Convertiamo solo per visualizzare
    plt.colorbar(label="Potential Value")
    plt.title("Potential Map")
    plt.show()

def save_and_plot_everything(img, img_path, output_dir, potential_map, saliency_map, mala_trajectory, fixations):

    fix_x, fix_y, fix_d = fixations[:, 0], fixations[:, 1], fixations[:, 2]

    img_name = os.path.basename(img_path).split('.')[0]
    img_output_dir = os.path.join(output_dir, img_name)
    os.makedirs(img_output_dir, exist_ok=True)
    
    #plot_saliency_and_potential_maps(saliency_map_np, potential_map_np, os.path.join(img_output_dir, 'saliency_potential_np.png'))
    plot_trajectory_on_image(img, mala_trajectory, os.path.join(img_output_dir, 'trajectory.png'))

    plot_gaze_with_steps(img, mala_trajectory, 100, os.path.join(img_output_dir, 'gaze_steps.png'))
    plot_gaze_direction(img, mala_trajectory, os.path.join(img_output_dir, 'gaze_direction.png'))
    
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    draw_scanpath(fix_x, fix_y, fix_d)
    plt.axis("off")
    plt.title("Simulated Scan")

    # Saliency map generata dai dati di scanpath
    plt.subplot(1, 3, 3)
    sal = compute_density_image(mala_trajectory[:, :2], img.shape[:2])
    res = np.multiply(img, np.repeat(sal[:, :, None] / np.max(sal), 3, axis=2))
    res = res / np.max(res)
    plt.imshow(res)
    plt.axis("off")
    plt.title("Generated Saliency from Trajectory")
    
    
    plt.show()
    