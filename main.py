import os
import argparse
from PIL import Image
import PIL
import numpy as np
import argparse
import cv2
import re
from tensorflow.python.keras import models

#extract the arguments 
parser = argparse.ArgumentParser(description=
'Segment a document, fit the classifier, evaluate accuracy or predict class of symbol')

parser.add_argument('--task', type=str, default='pass',
                    help="""
                    task to perform: 
                    augment_images-->augment a set of images to extend dataset
                    fit-->fit the classifier (and optionaly save) the classifier; 
                    evaluate-->­­­calculate the accuracy on a given set of images
                    classify-->predict the probability that the image is from the possible classes
                    """)

parser.add_argument('--segment', type=str, default=None,
                    help="""
                    Path of the image when we want to perform document segmentation
                    """)

parser.add_argument('--evaluate_directory', type=str, default='test',
                    help="""
                    If we want to evaluate accuracy on images in "train", "val" or "test"
                    """)

parser.add_argument('--img', type=str, default=None,
                    help="""
                    Path of the image when we want to predict its class
                    """)

args = parser.parse_args()

#checking the format of given arguments
if args.task not in ['segment', 'fit', 'evaluate', 'classify']:
    print('Task not supported!')
    args.task = 'pass'

if args.task == 'segment':    
    if os.path.exists(args.seg):
        seg_path = args.seg
    else:
        print('Unknown path!')
        args.task = 'pass'

if args.task == 'evaluate_directory':    
    if args.evaluate_directory not in ['train', 'val', 'test']:
        print('evaluate_directory has to be train, val or test')
        args.task = 'pass'

if args.task == 'classify':    
    if os.path.exists(args.img):
        img_path = args.img
    else:
        print('Unknown path!')
        args.task = 'pass'


# function to preprocess the image
def read_image(folder):
    pass

'''
- function to fit a model with new data
- function calls read_image to read images
- import saved model and fit the new images
'''
def fit():
    pass

# function to classify a symbol
def classify(img_path):
    pass   

# testing function to calculate accuracy of model
def evaluate():
    pass

# pass the input document to the localization script
def segment():
    pass

# call functions based on --task values
if args.task == 'segment':
    segment(seg_path)

elif args.task == 'fit':
    fit()
  
elif args.task == 'classify':
    classify(img_path)
    
elif args.task == 'evaluate':
    evaluate()
