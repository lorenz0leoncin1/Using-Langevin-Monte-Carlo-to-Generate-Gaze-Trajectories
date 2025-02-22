from evaluation import evaluation_pipeline_N
from run_algorithms import run_algorithm
from saliency_to_potential import *
from zs_clip_seg import *
    
def main():

    # Data path ----------------------------------------------------------
    input_dir = "COCOSearch18-images-TP/images/"
    task = "keyboard"
    output_dir = f"output/images/{task}"
    name = "000000006608.jpg"

    # Choose the algorithm -----------------------------------------------
    algorithm = 'ula' 

    # Model Parameter ----------------------------------------------------
    gamma = 0.1  # This is used only in the MALA-Cauchy

    # Use this to test one of the 3 algorithms on a choosen image
    #fixations, img = run_algorithm(input_dir, output_dir, task, name, algorithm, gamma)
    
    # Pipeline for evaluating all the images in split1 validation set
    evaluation_pipeline_N(5, algorithm, gamma)
main()
