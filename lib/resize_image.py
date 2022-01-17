import multiprocessing
import os
import sys
import cv2
from multiprocessing import Pool

input_directory = sys.argv[1]
output_directory = sys.argv[2]

def resize_image(filename, input_directory=input_directory, output_directory=output_directory):
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
    with Pool(nb_cpu) as p:
        p.map(resize_image, os.listdir(input_directory))