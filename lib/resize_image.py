import multiprocessing
import os
import sys
import cv2
from multiprocessing import Pool
import argparse

def parse_args():
    """
    parse the arguments

    :return: input and output filename
    :rtype: string, string
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="input file", required=True)
    parser.add_argument('-o', '--output', help="output file", required=True)
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    return input_filename, output_filename

input_directory = sys.argv[1]
output_directory = sys.argv[2]

def resize_image(args):
    """
    resize image from (width, height) to (x, 224) or (224, y) size and save it
    :param args: tuple (filename, input_directory, output_directory) 
    """
    filename = args[0]
    input_directory = args[1]
    output_directory = args[2]
    img = cv2.imread(input_directory + filename)
    width = img.shape[1]
    height = img.shape[0]
    width_divide = width / 224
    height_divide = height / 224
    if width > height:
        dim = (int(width / height_divide), int(height / height_divide))
    else:
        dim = (int(width / width_divide), int(height / width_divide))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_directory + filename, resized)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 resize_image.py {input} {output}")
        exit()
    nb_cpu = multiprocessing.cpu_count()
    input_directory, output_directory = parse_args()
    args = [(filename, input_directory, output_directory) for filename in os.listdir(input_directory)]
    with Pool(nb_cpu) as p:
        p.map(resize_image, args)