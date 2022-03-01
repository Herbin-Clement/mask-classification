import pandas as pd
import os
# import tensorflow as tf
import argparse
import shutil

from lib import Data_processing, Data_visualisation, Model

def parse_args():
    """
    parse the arguments

    :rtype: bool, bool
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm', '--loadmodel', help="load model", action='store_true')    
    parser.add_argument('-nd', '--newdataset', help="newdataset", action='store_true')
    args = parser.parse_args()

    loadmodel = args.loadmodel
    newdataset = args.newdataset
    return loadmodel, newdataset

def new_dataset(dataset_dir):
    """
    remove the last dataset and create the new directory

    :param dataset_dir: the dataset folder 
    """
    shutil.rmtree(dataset_dir)
    for e in ["train", "test", "validation"]:
        for c in [0, 1, 2, 3]:
            os.makedirs(dataset_dir + '/' + e + '/' + str(c))
    os.makedirs(os.path.join(dataset_dir, "csv"))

if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    loadmodel, newdataset = parse_args()
    root_dir = os.path.dirname(__file__)
    absolute_root_dir = os.path.abspath(root_dir)
    image_dir = os.path.join(root_dir, "image_reduce_size")
    dataset_dir = os.path.join(root_dir, "dataset")
    save_dir = os.path.join(root_dir, "weights")

    images_filenames = os.listdir(image_dir)
    nb_images = len(images_filenames)
    csv_pathname = os.path.join(root_dir, "csv/df_part_7.csv")
    
    process_data = Data_processing.Data_processing(csv_pathname, image_dir, dataset_dir)
    Model = Model.TinyVGG(root_dir, dataset_dir, save_dir, batch_size=32)

    if newdataset:
        new_dataset(root_dir, data_dir)
        process_data.test_train_validation_split_from_csv()
        process_data.create_train_test_validation_folder()

    if loadmodel:
        print("Load model ...")
        Model.load_weights()
        model = Model.get_model()
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_1_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_2_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_3_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_4_028450_FEMALE_30.jpg"), model)
    
    else:
        print("Training new model ...")
        Model.fit_model()
        model = Model.get_model()
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_1_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_2_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_3_028450_FEMALE_30.jpg"), model)
        Data_visualisation.predictImage(os.path.join(r"image_reduce_size\060002_4_028450_FEMALE_30.jpg"), model)