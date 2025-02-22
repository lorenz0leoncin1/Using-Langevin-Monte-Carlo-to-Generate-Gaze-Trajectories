import numpy as np
import dataset_helpers as helpers
from trajectory_split import ivt
from features import calculate_velocity

def classify_raw_IVT(xy, sample_rate, screen_params, vel_threshold, min_fix_duration):
    """ Generate feature vectors for all saccades and fixations in a trajectory

    :param xy: ndarray
        2D array of gaze points (x,y)
    :param label: any
         single class this trajectory belongs to
    :return: list, list
        list of feature vectors (each is a 1D ndarray) and list of classes(these will all be the same)
    """
    try:
        angle = helpers.convert_pixel_coordinates_to_angles(xy, *screen_params.values())
        smoothed_angle = np.asarray(helpers.savgol_filter_trajectory(angle)).T
        smoothed_vel_xy = calculate_velocity(smoothed_angle, sampleRate=sample_rate)
        smoothed_vel = np.linalg.norm(smoothed_vel_xy, axis=1)
        smoothed_pixels = helpers.convert_angles_to_pixel_coordinates(smoothed_angle, *screen_params.values())

        #print(smoothed_vel)
        
        try:
            sacs, fixs = ivt(smoothed_vel, vel_threshold=vel_threshold, min_fix_duration=min_fix_duration, sampleRate=sample_rate)
            fixations = []
            durs = []
            for fix in fixs:
                fixxy = xy[fix[0]:fix[1]]
                fixations.append(np.mean(fixxy, axis=0))
                durs.append((len(fixxy)/sample_rate)*1000)  # Qua si regola la grandezza delle T

            fixations = np.array(fixations)
            durs = np.array(durs).reshape(-1,1)
            
            gen_fix_plus_dur = np.hstack((fixations, durs)).astype(int) if fixations.shape[0] > 0 else np.array([])
            return gen_fix_plus_dur
        except ValueError as ve:
            print("Errore durante la classificazione IVT:", ve)
            return np.array([])
    except (ValueError, IndexError) as e:
        # Gestione centralizzata degli errori
        print(f"Errore nei dati di input o di indice: {e}")
        return np.array([])
