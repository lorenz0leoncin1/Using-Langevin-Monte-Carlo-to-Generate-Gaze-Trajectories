from evaluation import evaluation_pipeline_N
from run_algorithms import run_algorithm
from saliency_to_potential import *
from zs_clip_seg import *
    
def main():

    # Data path -------------------------------------------------------------------------
    input_dir = "COCOSearch18-images-TP/images/"
    task = "keyboard"
    output_dir = f"output/images/{task}"
    name = "000000006608.jpg"

    # Choose the algorithm --------------------------------------------------------------
    algorithm = 'ula'   # Type of algorithm to use
    n = 10              # Number of scanpaths to simulate

    # Model Parameter -------------------------------------------------------------------
    gamma = 0.1  # This is used only in the MALA-Cauchy

    # Type of execution

    # Type 1: run a choosen algorithm on a given image ----------------------------------
    #fixations, img = run_algorithm(input_dir, output_dir, task, name, algorithm, gamma)
    
    # Type 2: Simulate N scanpaths, run a choosen algorithm and recieve all the stats of the performances with MM and SM 
    # Pipeline for evaluating all the images in split1 validation set
    evaluation_pipeline_N(n, algorithm, gamma)
main()
