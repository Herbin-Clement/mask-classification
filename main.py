from ast import parse
from logging import root
import pandas as pd
import os
import tensorflow as tf
import argparse

from lib import Data_processing, Data_visualisation, Model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm', '--loadmodel', help="load model", type=bool, required=True)

    args = parser.parse_args()

    loadmodel = args.loadmodel

    return loadmodel


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    loadmodel = parse_args()

    root_dir = os.path.dirname(__file__) + "/"
    absolute_root_dir = os.path.abspath(root_dir)

    image_dir = os.path.join(root_dir, "image_reduce_size")
    dataset_dir = os.path.join(root_dir, "set")

    images_filenames = os.listdir(image_dir)
    nb_images = len(images_filenames)
    
    csv_pathname = os.path.join(root_dir, "csv/df_part_7.csv")

    process_data = Data_processing.Data_processing(csv_pathname, image_dir, dataset_dir)
    Model = Model.Model(root_dir, dataset_dir, batch_size=32)

    if loadmodel:
        process_data.get_train_test_validation_csv()
        process_data.print_dataset_directory()
        Model.load_weights(root_dir + "Weights/TinyVGG/0/")
        model = Model.get_model()
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_1_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_2_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_3_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_4_028450_FEMALE_30.jpg"), model)
    
    else:
        process_data.test_train_validation_split_from_csv()
        process_data.create_train_test_validation_folder()
        process_data.print_dataset_directory()
        Model.fit_model()