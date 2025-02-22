
from evaluation import evaluate_one_image, evaluation_pipeline_N
from ula_mala import *
from zs_clip_seg import *
    
def main():

    #Pipeline for evaluating all the images in split1 validation set
    evaluation_pipeline_N(5)

    #evaluate_one_image()
main()
