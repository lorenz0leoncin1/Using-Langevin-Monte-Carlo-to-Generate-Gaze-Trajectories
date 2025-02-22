import numpy as np
import pandas as pd
from tqdm import tqdm
from metrics_eval import calculate_similarity, calculate_similarity_N, create_dataframe
from run_algorithms import *


def evaluation_pipeline_N(N, algorithm, gamma):
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
                fixations, _ = run_algorithm(input_dir, output_dir, current_task, current_name, algorithm, gamma)
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
            fixations, _ = run_algorithm(input_dir, output_dir, current_task, current_name, algorithm, gamma)
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