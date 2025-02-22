# Funzione per convertire stringhe di liste in array numpy
import ast
import pandas as pd
import numpy as np
import multimatch_gaze as mg
from scan_match import ScanMatch

def convert_to_numpy(value):
    """
    Convert a value to a NumPy array.
    
    Args:
        value (str, list, or other): The value to convert.

    Returns:
        np.ndarray: Converted NumPy array. Returns an empty array if conversion is not possible.
    """
    if isinstance(value, str):
        return np.array(ast.literal_eval(value))  # Safely convert string representation of a list to a NumPy array
    elif isinstance(value, list):
        return np.array(value)  # Directly convert list to NumPy array
    else:
        return np.array([])  # Return an empty array for unsupported types

def create_dataframe():
    """
    Load and process the fixation dataset, filtering based on specific criteria.

    Returns:
        pd.DataFrame: Processed DataFrame with filtered and transformed fixation data.
    """
    df = pd.read_json('COCOSearch18-fixations-TP/coco_search18_fixations_TP_validation_split2.json')

    # Compute the number of unique subjects for each (name, task) pair
    subjects_count = df.groupby(['name', 'task'])['subject'].nunique()

    # Select only (name, task) pairs that have exactly 10 unique subjects
    valid_pairs = subjects_count[subjects_count == 10].index

    # Filter the original DataFrame to retain only rows with valid (name, task) pairs
    df_filtered = df[df.set_index(['name', 'task']).index.isin(valid_pairs)]

    # Further filter to keep only pairs where all 'length' values are greater than 2
    df = df_filtered.groupby(['name', 'task']).filter(lambda x: (x['length'] > 2).all())

    # Convert 'X', 'Y', and 'T' columns to NumPy arrays
    df['X'] = df['X'].apply(convert_to_numpy)
    df['Y'] = df['Y'].apply(convert_to_numpy)
    df['T'] = df['T'].apply(convert_to_numpy)
        
    return df


def docomparison(
    fixation_vectors1,
    fixation_vectors2,
    screensize,
    grouping=False,
    TDir=0.0,
    TDur=0.0,
    TAmp=0.0,
):
    """Compare two scanpaths on five similarity dimensions.


    :param: fixation_vectors1: array-like n x 3 fixation vector of one scanpath
    :param: fixation_vectors2: array-like n x 3 fixation vector of one scanpath
    :param: screensize: list, screen dimensions in px.
    :param: grouping: boolean, if True, simplification is performed based on thresholds TAmp,
        TDir, and TDur. Default: False
    :param: TDir: float, Direction threshold, angle in degrees. Default: 0.0
    :param: TDur: float,  Duration threshold, duration in seconds. Default: 0.0
    :param: TAmp: float, Amplitude threshold, length in px. Default: 0.0

    :return: scanpathcomparisons: array
        array of 5 scanpath similarity measures. Vector (Shape), Direction
        (Angle), Length, Position, and Duration. 1 means absolute similarity, 0 means
        lowest similarity possible.

    >>> results = docomparison(fix_1, fix_2, screensize = [1280, 720], grouping = True, TDir = 45.0, TDur = 0.05, TAmp = 150)
    >>> print(results)
    >>> [[0.95075847681364678, 0.95637548674423822, 0.94082367355291008, 0.94491164030498609, 0.78260869565217384]]
    """
    # check if fixation vectors/scanpaths are long enough
    if (len(fixation_vectors1) >= 2) & (len(fixation_vectors2) >= 2):           # Change at 2 instead of 3
        # get the data into a geometric representation
        mg.multimatch_gaze
        path1 = mg.multimatch_gaze.gen_scanpath_structure(fixation_vectors1)
        path2 = mg.multimatch_gaze.gen_scanpath_structure(fixation_vectors2)
        if grouping:
            # simplify the data
            path1 = mg.multimatch_gaze.simplify_scanpath(path1, TAmp, TDir, TDur)
            path2 = mg.multimatch_gaze.simplify_scanpath(path2, TAmp, TDir, TDur)
        # create M, a matrix of all vector pairings length differences (weights)
        M = mg.multimatch_gaze.cal_vectordifferences(path1, path2)
        # initialize a matrix of size M for a matrix of nodes
        scanpath_dim = np.shape(M)
        M_assignment = np.arange(scanpath_dim[0] * scanpath_dim[1]).reshape(
            scanpath_dim[0], scanpath_dim[1]
        )
        # create a weighted graph of all possible connections per Node, and their weight
        numVert, rows, cols, weight = mg.multimatch_gaze.createdirectedgraph(scanpath_dim, M, M_assignment)
        # find the shortest path (= lowest sum of weights) through the graph using scipy dijkstra
        path, dist = mg.multimatch_gaze.dijkstra(
            numVert, rows, cols, weight, 0, scanpath_dim[0] * scanpath_dim[1] - 1
        )

        # compute similarities on aligned scanpaths and normalize them
        unnormalised = mg.multimatch_gaze.getunnormalised(path1, path2, path, M_assignment)
        normal = mg.multimatch_gaze.normaliseresults(unnormalised, screensize)
        return normal
    # return nan as result if at least one scanpath it too short
    else:
        return np.repeat(np.nan, 5)
    
def scanMatch_metric(s1, s2, stimulus_width, stimulus_height, Xbin=14, Ybin=8, TempBin=50):
    """
    Compute ScanMatch similarity score between two scanpaths.

    Args:
        s1 (np.ndarray): First scanpath, with shape [nfix,2] or [nfix,4].
        s2 (np.ndarray): Second scanpath, with shape [nfix,2] or [nfix,4].
        stimulus_width (int): Width of the stimulus.
        stimulus_height (int): Height of the stimulus.
        Xbin (int, optional): Number of spatial bins in the X dimension. Default is 14.
        Ybin (int, optional): Number of spatial bins in the Y dimension. Default is 8.
        TempBin (int, optional): Number of temporal bins. Default is 50.

    Returns:
        float: ScanMatch similarity score.
    """
    if s1.shape[1] > 2 and s2.shape[1] > 2:
        tb = TempBin
        duration1 = (s1[:, 3] - s1[:, 2]) * 1000.0
        s1 = np.hstack([s1[:, 0].reshape(-1, 1), s1[:, 1].reshape(-1, 1), duration1.reshape(-1, 1)])
        
        duration2 = (s2[:, 3] - s2[:, 2]) * 1000.0
        s2 = np.hstack([s2[:, 0].reshape(-1, 1), s2[:, 1].reshape(-1, 1), duration2.reshape(-1, 1)])

    elif s1.shape[1] == 2 and s2.shape[1] == 2:
        tb = 0.0
    else:
        raise ValueError("Scanpaths should be arrays of shape [nfix,2] or [nfix,4]! "
                         "Columns should represent x, y fixation coordinates and, optionally, start and end times.")

    match_object = ScanMatch(Xres=stimulus_width, Yres=stimulus_height, Xbin=Xbin, Ybin=Ybin, TempBin=TempBin)
    seq1 = match_object.fixationToSequence(s1).astype(int)
    seq2 = match_object.fixationToSequence(s2).astype(int)

    score, _, _ = match_object.match(seq1, seq2)

    return score


def multiMatch_metric(s1, s2, stimulus_width, stimulus_height):
    """
    Compute MultiMatch similarity score between two scanpaths.

    Args:
        s1 (np.ndarray): First scanpath, shape [nfix,4].
        s2 (np.ndarray): Second scanpath, shape [nfix,4].
        stimulus_width (int): Width of the stimulus.
        stimulus_height (int): Height of the stimulus.

    Returns:
        list: MultiMatch similarity scores across different dimensions.
    """
    assert s1.shape[1] > 2 and s2.shape[1] > 2, "Scanpaths should have shape [nfix,4] for MultiMatch computation!"

    duration1 = (s1[:, 3] - s1[:, 2]) * 1000.0
    ss1 = np.hstack([s1[:, 0].reshape(-1, 1), s1[:, 1].reshape(-1, 1), duration1.reshape(-1, 1)])

    duration2 = (s2[:, 3] - s2[:, 2]) * 1000.0
    ss2 = np.hstack([s2[:, 0].reshape(-1, 1), s2[:, 1].reshape(-1, 1), duration2.reshape(-1, 1)])

    scan1_pd = pd.DataFrame({'start_x': ss1[:, 0], 'start_y': ss1[:, 1], 'duration': ss1[:, 2]})
    scan2_pd = pd.DataFrame({'start_x': ss2[:, 0], 'start_y': ss2[:, 1], 'duration': ss2[:, 2]})

    mm_scores = docomparison(scan1_pd.to_records(), scan2_pd.to_records(),
                             screensize=[stimulus_width, stimulus_height])  # Using custom docomparison

    return mm_scores


def calculate_similarity(temp_df):
    """
    Compute the average similarity metrics (MultiMatch & ScanMatch) for a set of scanpaths.

    Args:
        temp_df (pd.DataFrame): DataFrame containing scanpath data with columns 'X', 'Y', 'T', and 'subject'.

    Returns:
        tuple: Mean MultiMatch and ScanMatch similarity scores.
    """
    MM_similarity_scores = []
    SM_similarity_scores = []
    scanpaths = []

    # Generate scanpath for each subject
    for idx, row in temp_df.iterrows():
        x = np.array(row['X'])
        y = np.array(row['Y'])
        durations = np.array(row['T'])

        print(f"\nProcessing subject {row['subject']}")
        print(f"  X: {x}")
        print(f"  Y: {y}")
        print(f"  Durations: {durations}")

        if len(x) == len(y) == len(durations) and len(durations) > 0:
            start_times = np.cumsum(np.insert(durations, 0, 0))[:-1] / 1000.0
            end_times = start_times + durations / 1000.0

            print(f"  Start times: {start_times}")
            print(f"  End times: {end_times}")

            if len(start_times) == len(end_times) == len(x):
                scanpaths.append(np.column_stack([x, y, start_times, end_times]))
                print(f"  Scanpath created for subject {row['subject']}")
            else:
                print(f" Dimension mismatch for subject {row['subject']}")

    # Compare scanpaths against the last subject
    if scanpaths:
        reference_scanpath = scanpaths[-1]  # The last subject is the reference

        for i in range(len(scanpaths) - 1):  # Compare all except the last one
            print(f"\nComparing subject {temp_df.iloc[i]['subject']} vs Reference (subject {temp_df.iloc[-1]['subject']})")

            MM_score = multiMatch_metric(scanpaths[i], reference_scanpath, 1280, 720)
            SM_score = scanMatch_metric(scanpaths[i], reference_scanpath, 1280, 720)

            print(f"ScanMatch Score: {SM_score}")
            print(f"MultiMatch Score: {MM_score}")

            MM_similarity_scores.append(MM_score)
            SM_similarity_scores.append(SM_score)

    # Compute average similarity scores
    if MM_similarity_scores and SM_similarity_scores:
        MM_mean_similarity = np.mean(np.array(MM_similarity_scores), axis=0)
        SM_mean_similarity = np.mean(np.array(SM_similarity_scores), axis=0)

        return MM_mean_similarity, SM_mean_similarity
    else:
        print("No similarity scores calculated.")
        return None, None

def calculate_similarity_N(temp_df):
    """
    Compute the average similarity metrics (MultiMatch & ScanMatch) for a set of scanpaths.

    Args:
        temp_df (pd.DataFrame): DataFrame containing scanpath data with columns 'X', 'Y', 'T', and 'subject'.

    Returns:
        tuple: Mean MultiMatch and ScanMatch similarity scores across all simulated scanpaths.
    """
    human_scanpaths = []
    simulated_scanpaths = []
    
    for idx, row in temp_df.iterrows():
        x = np.array(row['X'])
        y = np.array(row['Y'])
        durations = np.array(row['T'])

        start_times = np.cumsum(np.insert(durations, 0, 0))[:-1] / 1000.0
        end_times = start_times + durations / 1000.0

        scanpath = np.column_stack([x, y, start_times, end_times])

        if row['subject'] < 11:
            human_scanpaths.append(scanpath)
        else:
            simulated_scanpaths.append(scanpath)

    #print(f"Total Human Scanpaths: {len(human_scanpaths)}")
    #print(f"Total Simulated Scanpaths: {len(simulated_scanpaths)}")

    individual_means_MM = []
    individual_means_SM = []

    # Calculate mean scores for each simulated scanpath against all human scanpaths
    for sim_idx, sim_path in enumerate(simulated_scanpaths):
        #print(f"\nComparing Simulated Subject {11 + sim_idx} vs 10 Human Subjects")
        MM_scores = []
        SM_scores = []

        for human_idx, human_path in enumerate(human_scanpaths):
            MM_score = multiMatch_metric(human_path, sim_path, 1280, 720)
            SM_score = scanMatch_metric(human_path, sim_path, 1280, 720)
            
            MM_scores.append(MM_score)
            SM_scores.append(SM_score)

            #print(f"Human {human_idx + 1} → Simulated {11 + sim_idx}")
            #print(f"  MultiMatch Score: {MM_score}")
            #print(f"  ScanMatch Score: {SM_score}")
        
        # Compute mean for this simulated scanpath
        if MM_scores and SM_scores:
            individual_means_MM.append(np.mean(MM_scores, axis=0))
            individual_means_SM.append(np.mean(SM_scores, axis=0))

    # Debugging: Print individual means and check lengths
    print(f"Individual Means MM: {individual_means_MM}")
    print(f"Individual Means SM: {individual_means_SM}")
    print(f"Length check: {len(individual_means_MM)} == {len(simulated_scanpaths)}, {len(individual_means_SM)} == {len(simulated_scanpaths)}")

    # Final mean across all simulated scanpaths
    if individual_means_MM and individual_means_SM:

        MM_mean_similarity = np.mean(individual_means_MM, axis=0)
        print(f"MM Mean : {MM_mean_similarity}")

        SM_mean_similarity = np.mean(individual_means_SM, axis=0)
        print(f"SM Mean : {SM_mean_similarity}")

        return MM_mean_similarity, SM_mean_similarity
    else:
        print("No similarity scores calculated.")
        return None, None
    
