import os
from matplotlib import patches, pyplot as plt
import numpy as np
import torch
from vis import compute_density_image, draw_scanpath

def plot_trajectory_on_image(image, trajectory, output_path):
    trajectory = np.array(trajectory)  # Converte la traiettoria in array NumPy
    plt.imshow(image, cmap="gray")
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c="red", s=5, alpha=0.6, label="Gaze Trajectory")
    plt.legend()
    plt.title("Trajectory on background image")
    plt.savefig(output_path)  # Save the figure to the output path
    plt.close()

def plot_gaze_direction(image, trajectory, output_path):
    trajectory = np.array(trajectory)  # Converti la traiettoria in array NumPy
    plt.imshow(image, cmap="gray")
    
    # Coordinate di partenza e di arrivo
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    
    # Calcola la differenza per determinare direzione e lunghezza
    dx = end_x - start_x
    dy = end_y - start_y
    
    # Traccia una freccia che indica la direzione
    plt.arrow(
        start_x, start_y, dx, dy,
        color="red", width=0.5, head_width=5, head_length=5,
        length_includes_head=True, label="Direzione media"
    )
    
    plt.legend()
    plt.title("Gaze direction")
    plt.savefig(output_path)  # Salva l'immagine nel percorso specificato
    plt.close()

def plot_gaze_with_steps(image, trajectory, step_interval, output_path):
    trajectory = np.array(trajectory)  # Converti la traiettoria in array NumPy
    plt.imshow(image, cmap="gray")
    
    # Plottare la traiettoria
    plt.scatter(trajectory[:, 0], trajectory[:, 1], c="red", s=5, alpha=0.6, label="Gaze Trajectory")
    # Aggiungere cerchi numerati ogni tot passi
    for i in range(0, len(trajectory), step_interval):
        plt.scatter(trajectory[i, 0], trajectory[i, 1], c="blue", s=50, edgecolor="black")
        plt.text(trajectory[i, 0] + 5, trajectory[i, 1] - 5, str(i), color="white", fontsize=12)
    
    plt.legend()
    plt.title("Trajectory on image with jump numbers")
    plt.savefig(output_path)  # Save the figure to the output path
    plt.close()

def plot_saliency_and_potential_maps(saliency_map, potential_map, output_path):
    
    # Verifica le forme per assicurarti che siano 2D
    print("Saliency map shape:", saliency_map.shape)  # Deve essere come (768, 1024)
    print("Potential map shape:", potential_map.shape)  # Deve essere come (768, 1024)
    print("Saliency map type:", saliency_map.dtype)
    print("Potential map type:", potential_map.dtype)
    # Traccia le mappe
    plt.figure(figsize=(10, 5))

    # Mappa di salienza
    plt.subplot(1, 2, 1)
    plt.imshow(saliency_map, cmap='hot', interpolation='bilinear')
    plt.colorbar(label="Saliency Value")
    plt.title("Saliency Map")

    # Mappa di potenziale
    plt.subplot(1, 2, 2)
    plt.imshow(potential_map.detach(), cmap='hot', interpolation='bilinear')  # Usiamo .detach() per evitare il gradiente
    plt.colorbar(label="Potential Value")
    plt.title("Potential Map")

    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure to the output path
    plt.close()


def plot_fixations_with_numbers(image, trajectory, num_steps, circle_radius, min_distance, output_path):
    """
    Plot the trajectory of fixations with numbered circles to simulate visual attention, 
    avoiding overlap between circles and connecting the circles with red lines.
    """
    step_interval = len(trajectory) // num_steps
    trajectory = np.array(trajectory)  # Convert trajectory to NumPy array
    plt.imshow(image, cmap="gray")
    
    ax = plt.gca()  # Get the current axis
    plotted_points = []  # List to track already plotted points
    prev_point = None  # To track the previous point for drawing lines

    # Draw circles along the trajectory
    for i in range(num_steps):
        idx = i * step_interval  # Index for each step
        x, y = trajectory[idx]  # Inverti x e y qui
        
        overlapping_circles = []  # To store overlapping circles
        for j, (py, px, pr) in enumerate(plotted_points):
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance < min_distance:  # If distance is less than min_distance, they overlap
                overlapping_circles.append((j, py, px, pr))
        
        if not overlapping_circles:  # No overlaps
            # Draw the circle
            circle = patches.Circle((y, x), radius=circle_radius, facecolor='red', alpha=0.4, edgecolor='black', lw=2)  # Inverti x e y qui
            ax.add_patch(circle)

            # Add the number inside the circle
            plt.text(y, x, str(i + 1), color="white", fontsize=12, ha='center', va='center', fontweight='bold')  # Inverti x e y qui

            # Update plotted points
            plotted_points.append((y, x, circle_radius))  # Inverti x e y qui
            
            # Draw the line connecting the previous circle
            if prev_point:
                plt.plot([prev_point[0], y], [prev_point[1], x], color='red', lw=1)  # Inverti x e y qui

            prev_point = (y, x)
        else:  # Handle overlaps
            # Sort overlapping circles by radius (larger ones come first)
            overlapping_circles.sort(key=lambda item: -item[3])

            # Use the largest radius to determine the new circle's size
            max_radius = overlapping_circles[0][3]
            new_radius = max_radius * 1.2  # Scale the new circle's size
            
            # Draw a larger circle for the current number
            larger_circle = patches.Circle((y, x), radius=new_radius, facecolor='orange', alpha=0.5, edgecolor='black', lw=2)  # Inverti x e y qui
            ax.add_patch(larger_circle)

            # Plot the smaller number inside the larger circle
            plt.text(y, x, str(i + 1), color="white", fontsize=12, ha='center', va='center', fontweight='bold')  # Inverti x e y qui
            
            # Update plotted points with the new circle
            plotted_points.append((y, x, new_radius))  # Inverti x e y qui

    # Finalize the plot
    plt.title("Simulation of fixations with numbering (without overlaps)")
    plt.axis("off")
    plt.savefig(output_path)  # Save the figure to the output path
    plt.close()


def extract_fixations(trajectory, distance_threshold=5, time_per_step=40):
    fixations = []
    current_fixation = [trajectory[0]]
    for i in range(1, len(trajectory)):
        if np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i - 1])) < distance_threshold:
            current_fixation.append(trajectory[i])
        else:
            # Calcola durata e aggiungi la fissazione
            if len(current_fixation) > 1:
                duration = len(current_fixation) * time_per_step
                fixations.append((np.mean(current_fixation, axis=0), duration))
            current_fixation = [trajectory[i]]

    # Aggiungi l'ultima fissazione
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
    # Assicurati che sia 2D
    saliency_map = saliency_map.squeeze()

    # Visualizzazione della mappa di salienza
    plt.figure(figsize=(6, 5))
    plt.imshow(saliency_map)
    plt.colorbar(label="Saliency Value")
    plt.title("Saliency Map")
    plt.show()

    # Calcolo del gradiente
    grad_x, grad_y = torch.gradient(saliency_map)

    # Visualizzazione delle derivate
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(grad_x)
    axes[0].set_title("Gradient along X")
    fig.colorbar(im1, ax=axes[0], label="dSaliency/dX")

    im2 = axes[1].imshow(grad_y)
    axes[1].set_title("Gradient along Y")
    fig.colorbar(im2, ax=axes[1], label="dSaliency/dY")

    plt.show()

    # Calcolo del gradiente totale
    grad_total = torch.sqrt(grad_x**2 + grad_y**2)

    # Visualizzazione del gradiente totale
    plt.figure(figsize=(6, 5))
    plt.imshow(grad_total)
    plt.colorbar(label="Total Gradient")
    plt.title("Total Gradient Map")
    plt.show()

    # Calcolo della seconda derivata (test di differenziabilità)
    grad_x2, grad_y2 = torch.gradient(grad_x)
    grad_xx2, grad_yy2 = torch.gradient(grad_y)

    # Visualizzazione delle derivate seconde
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(grad_x2)
    axes[0].set_title("Second Derivative along X")
    fig.colorbar(im1, ax=axes[0], label="d²Saliency/dX²")

    im2 = axes[1].imshow(grad_y2)
    axes[1].set_title("Second Derivative along Y")
    fig.colorbar(im2, ax=axes[1], label="d²Saliency/dY²")

    plt.show()

    # Istogramma dei valori di salienza
    plt.figure(figsize=(6, 5))
    plt.hist(saliency_map.flatten(), bins=50, color="blue", alpha=0.7)
    plt.xlabel("Saliency Value")
    plt.ylabel("Frequency")
    plt.title("Saliency Value Distribution")
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

    # Calcolo dei gradienti
    grad_x, grad_y = torch.gradient(potential_map_detached)

    # Visualizzazione dei gradienti
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im1 = axes[0].imshow(grad_x.numpy(), cmap="coolwarm")  # Convertiamo in NumPy per la visualizzazione
    axes[0].set_title("Gradient along X")
    fig.colorbar(im1, ax=axes[0], label="dPotential/dX")

    im2 = axes[1].imshow(grad_y.numpy(), cmap="coolwarm")  # Convertiamo in NumPy per la visualizzazione
    axes[1].set_title("Gradient along Y")  
    fig.colorbar(im2, ax=axes[1], label="dPotential/dY")

    plt.show()

    # Istogramma della distribuzione dei valori del potenziale
    plt.figure(figsize=(6, 5))
    plt.hist(potential_map_detached.flatten().numpy(), bins=50, color="green", alpha=0.7)  # Convertiamo in NumPy
    plt.xlabel("Potential Value")
    plt.ylabel("Frequency")
    plt.title("Potential Value Distribution")
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
    #plot_fixations_with_numbers(img, mala_trajectory, 4, 20, 40, os.path.join(img_output_dir, 'fixations.png'))
    
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    #plt.axis("off")
    plt.title("Original image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(img)
    draw_scanpath(fix_x, fix_y, fix_d)
    #plt.axis("off")
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
    