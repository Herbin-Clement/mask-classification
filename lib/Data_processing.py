import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import shutil
from shutil import copy2
from multiprocessing import Pool, cpu_count


class Data_processing:

    def __init__(self, csv_filename, images_dir, dataset_dir, train_test_validation_ratio=(0.75, 0.15, 0.1)):
        self.data = pd.read_csv(csv_filename)
        self.images_dir = images_dir
        self.dataset_dir = dataset_dir
        self.train_test_validation_ratio = train_test_validation_ratio
        self.classes = self.data["TYPE"].unique()

    def test_train_validation_split_from_csv(self, data_ratio=1):
        """
        split a DataFrame into 3 DataFrames, respectively train, test and validation DataFrame

        :param df: the DataFrame to split
        :param train_test_validation_ratio: the ratio of data of each DataFrames
        :param data_ratio: the ratio of total data use
        :return: the 3 DataFrames
        """
        train_ratio, test_ratio, validation_ratio = self.train_test_validation_ratio
        self.train_df = pd.DataFrame(columns=self.data.columns)
        self.test_df = pd.DataFrame(columns=self.data.columns)
        self.validation_df = pd.DataFrame(columns=self.data.columns)
        nb_data = int(self.data.shape[0] * data_ratio)
        for i in range(nb_data):
            random_number = random.random()
            if random_number <= train_ratio:
                self.train_df = self.train_df.append(
                    self.data.iloc[i], ignore_index=True)
            elif random_number > train_ratio and random_number <= train_ratio + test_ratio:
                self.test_df = self.test_df.append(
                    self.data.iloc[i], ignore_index=True)
            else:
                self.validation_df = self.validation_df.append(
                    self.data.iloc[i], ignore_index=True)
        return self.train_df, self.test_df, self.validation_df

    def get_train_test_validation_csv(self):
        self.train_df = pd.read_csv(os.path.join(self.dataset_dir, "train.csv"))
        self.test_df = pd.read_csv(os.path.join(self.dataset_dir, "test.csv"))
        self.validation_df = pd.read_csv(os.path.join(self.dataset_dir, "validation.csv"))

    def remove_set(self):
        """
        remove the directory and all files and folder inside
        :param pathname: the pathname of the directory to remove
        """
        try:
            shutil.rmtree(self.dataset_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    def print_dataset_directory(self):
        """
        print the number of images of each folder in dataset_dir
        :param dataset_dir: the directory
        """
        for dirpath, dirnames, filesnames in os.walk(self.dataset_dir):
            if len(filesnames) > 0:
                print(
                    f"There are {len(dirnames)} directory and {len(filesnames)} images in {dirpath}.")

    def create_train_test_validation_folder(self):
        """
        create the train, test and validation folder from the csv data
        :param src_dir: the source directory
        :param dst_dir: the destination directory
        :param train_df: the train DataFrame
        :param test_df: the test DataFrame
        :param validation_df: the validation DataFrame
        :param classes: list of classes
        """
        
        # self.__create_dataset_dir(self.dataset_dir)
        nb_train_data = self.train_df.shape[0]
        nb_test_data = self.test_df.shape[0]
        nb_validation_data = self.validation_df.shape[0]
        train_args = [(self.images_dir + '/' + self.train_df.iloc[i]["name"],
                      self.dataset_dir + "/train/" + str(self.train_df.iloc[i]["TYPE"]) + '/' + self.train_df.iloc[i]["name"]) for i in range(nb_train_data)]
        test_args = [(self.images_dir + '/' + self.test_df.iloc[i]["name"],
                      self.dataset_dir + "/test/" + str(self.test_df.iloc[i]["TYPE"]) + '/' + self.test_df.iloc[i]["name"]) for i in range(nb_test_data)]
        validation_args = [(self.images_dir + '/' + self.validation_df.iloc[i]["name"],
                            self.dataset_dir + "/validation/" + str(self.validation_df.iloc[i]["TYPE"]) + '/' + self.validation_df.iloc[i]["name"]) for i in range(nb_validation_data)]
        nb_cpu = cpu_count()
        print(f"number of cpu: {nb_cpu}")
        # with Pool(nb_cpu) as p:
        #   p.map(self.copy_image, train_args)
        # with Pool(nb_cpu) as p:
        #   p.map(self.copy_image, test_args)
        # with Pool(nb_cpu) as p:
        #   p.map(self.copy_image, validation_args)
        for e in train_args:
            self.__copy_image(e)
        for e in test_args:
            self.__copy_image(e)
        for e in validation_args:
            self.__copy_image(e)

    def __create_dataset_dir(self, classes):
        """
        create folders train, test and validation in the directory, and this 3 folders, a folder for each classes
        :param dst_dir: the directory
        :param classes: list of classes
        """
        for e in ["train", "test", "validation"]:
            for c in classes:
                os.makedirs(self.dataset_dir + '/' + e + '/' + str(c))
                print(self.dataset_dir + '/' + e + '/' + str(c))

    def __copy_image(self, src_dst):
        """
        copy the image from source to destination
        :param src_dst: tuple with source and destination
        """
        copy2(src_dst[0], src_dst[1])
