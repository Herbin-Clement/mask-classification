import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import shutil
from shutil import copy2
from multiprocessing import Pool
import multiprocessing

def test_train_validation_split_from_csv(df, train_test_validation_ratio=(0.75, 0.15, 0.1), data_ratio=1):
	"""
	split a DataFrame into 3 DataFrames, respectively train, test and validation DataFrame

	:param df: the DataFrame to split
	:param train_test_validation_ratio: the ratio of data of each DataFrames
	:param data_ratio: the ratio of total data use
	:return: the 3 DataFrames
	"""
  train_ratio, test_ratio, validation_ratio = train_test_validation_ratio
  train_df = pd.DataFrame(columns=df.columns)
  test_df = pd.DataFrame(columns=df.columns)
  validation_df = pd.DataFrame(columns=df.columns)
  nb_data = int(df.shape[0] * data_ratio)
  for i in range(nb_data):
    random_number = random.random()
    if random_number <= train_ratio:
      train_df = train_df.append(df.iloc[i], ignore_index=True)
    elif random_number > train_ratio and random_number <= train_ratio + test_ratio:
      test_df = test_df.append(df.iloc[i], ignore_index=True)
    else:
      validation_df = validation_df.append(df.iloc[i], ignore_index=True)
  return (train_df, test_df, validation_df)

def remove_set(pathname):
	"""
	remove the directory and all files and folder inside
	:param pathname: the pathname of the directory to remove
	"""
  try:
    shutil.rmtree(pathname)
  except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

def print_dataset_directory(dataset_dir):
	"""
	print the number of images of each folder in dataset_dir
	:param dataset_dir: the directory
	"""
  for dirpath, dirnames, filesnames in os.walk(dataset_dir):
    if len(filesnames) > 0:
      print(f"There are {len(dirnames)} directory and {len(filesnames)} images in {dirpath}.")

def create_train_test_validation_folder(src_dir, dst_dir, train_df, test_df, validation_df, classes):
	"""
	create the train, test and validation folder from the csv data
	:param src_dir: the source directory
	:param dst_dir: the destination directory
	:param train_df: the train DataFrame
	:param test_df: the test DataFrame
	:param validation_df: the validation DataFrame
	:param classes: list of classes
	"""
  create_dataset_dir(dst_dir, classes)
  nb_train_data = train_df.shape[0]
  nb_test_data = test_df.shape[0]
  nb_validation_data = validation_df.shape[0]
  train_args = [(src_dir + '/' + train_df.iloc[i]["name"], 
                 dst_dir + "/train/" + str(train_df.iloc[i]["TYPE"]) + '/' + train_df.iloc[i]["name"]) for i in range(nb_train_data)]
  test_args = [(src_dir + '/' + test_df.iloc[i]["name"], 
                dst_dir + "/test/" + str(test_df.iloc[i]["TYPE"]) + '/' + test_df.iloc[i]["name"]) for i in range(nb_test_data)]
  validation_args = [(src_dir + '/' + validation_df.iloc[i]["name"], 
                      dst_dir + "/validation/" + str(validation_df.iloc[i]["TYPE"]) + '/' + validation_df.iloc[i]["name"]) for i in range(nb_validation_data)]
  nb_cpu = multiprocessing.cpu_count()
  print(f"number of cpu: {nb_cpu}")
  with Pool(nb_cpu) as p:
    p.map(copy_image, train_args)
  with Pool(nb_cpu) as p:
    p.map(copy_image, test_args)
  with Pool(nb_cpu) as p:
    p.map(copy_image, validation_args)
  
def create_dataset_dir(dst_dir, classes):
	"""
	create folders train, test and validation in the directory, and this 3 folders, a folder for each classes
	:param dst_dir: the directory
	:param classes: list of classes
	"""
  for e in ["train", "test", "validation"]:
    for c in classes:
      os.makedirs(dst_dir + '/' + e + '/' + str(c))
      print(dst_dir + '/' + e + '/' + str(c))


def copy_image(src_dst):
	"""
	copy the image from source to destination
	:param src_dst: tuple with source and destination
	"""
  copy2(src_dst[0], src_dst[1])