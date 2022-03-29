from ast import Try
import os, fnmatch
import argparse
from unittest import expectedFailure

from lib import Data_processing, Data_visualisation
from lib.video_capture.FaceRecognition import FaceRecognition
from lib.Model.TinyVGG import TinyVGG
from lib.Model.TL_VGG19 import TL_VGG19
from lib.Model.Conv4Pool import Conv4Pool

def parse_args():
    """
    parse the arguments

    :rtype: bool, bool
    """
     
    parser = argparse.ArgumentParser(prog="main.py", add_help=False, usage='%(prog)s ([-lm] or [-nd]) ([-lm] or [-nd] or [-T] or [-V] or [-I] or [-R] or [-X])')
    parser.add_argument('-lm', '--loadmodel', help="load model", action='store_true')
    parser.add_argument('-lmm', '--loadmodelManualy', help="choose the save of a  model", action='store_true')  
    parser.add_argument('-nd', '--newdataset', help="newdataset", action='store_true')
    parser.add_argument('-T', '--TinyVGG', help='TinyVGG model', action='store_true')
    parser.add_argument('-C', '--Conv4Pool', help='Conv4Pool model', action='store_true')
    parser.add_argument('-V', '--VGG19', help='TinyVGG19 model', action='store_true')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(1)

    if args.TinyVGG :
        model = TinyVGG
    elif args.Conv4Pool :
        model = Conv4Pool
    elif args.VGG19 : 
        model = TL_VGG19
    else : 
        model = TinyVGG

    loadmodel = args.loadmodel
    loadmodel_m = args.loadmodelManualy
    newdataset = args.newdataset
    return loadmodel, newdataset, model, loadmodel_m

def weight_directory_path(dir_model_name, manual_selection=0):
    list_dir = os.listdir(dir_model_name)
    
    if manual_selection == 1: # mode manuelle
        print(f"{dir_model_name}/?\n")
        for i in range(len(list_dir)):
            print(f'\t{list_dir[0]}\n')
        d = input("\nChoisissez un dossier : ")
        list_files = fnmatch.filter(os.listdir(os.path.join(dir_model_name,d)),'*.index')
        print(f"{dir_model_name}/{d}/?\n")
        for i in range(len(list_files)):
            print(f"\t {i} : {list_files[i]}\n")
        f = input("\nChoisissez un fichier : ")
        print(os.path.join(dir_model_name, d, list_files[int(f)][:-6]))
        return os.path.join(dir_model_name, d, list_files[int(f)][:-6])
    
    else: #mode automatique 
        d = str(max([int(f) for f in os.listdir(dir_model_name)]))
        f = [int(f[3:7]) for f in fnmatch.filter(os.listdir(os.path.join(dir_model_name, d) ),'*.index')]
        print(os.path.join(dir_model_name, d,f"cp-{max(f) :0>4d}.ckpt"))
        return os.path.join(dir_model_name, d,f"cp-{max(f) :0>4d}.ckpt")


if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    loadmodel, newdataset, Model, loadmodel_m = parse_args()
    root_dir = os.path.dirname(__file__)
    absolute_root_dir = os.path.abspath(root_dir)
    image_dir = os.path.join(root_dir, "resize_image")
    dataset_dir = os.path.join(root_dir, "dataset")

    # images_filenames = os.listdir(image_dir)
    # nb_images = len(images_filenames)
    csv_pathname = os.path.join(root_dir, "csv/df_1_7.csv")
    process_data = Data_processing.Data_processing(csv_pathname, image_dir, dataset_dir)
    print("dataset_dir", dataset_dir)
    print("image_dir", image_dir)
    if newdataset:
        print("Create new dataset ...")
        process_data.create_dataset_dir()
        process_data.test_train_validation_split_from_csv(data_ratio=0.01)
        process_data.create_train_test_validation_folder()
    m = Model(root_dir, dataset_dir, batch_size=32)

    if loadmodel or loadmodel_m:
        print(loadmodel_m)
        print("Load model ...")
        try:
            dir_model_name = os.path.join("weights",  type(m).__name__,)
            path = weight_directory_path(dir_model_name, loadmodel_m)
            print(path)
            m.load_weights(path)
        except (FileNotFoundError, ValueError) :
                print("No model train !")
                exit(1)
        model = m.get_model()
        #Data_visualisation.confusion_matrix(os.path.join(dataset_dir, "validation"), model, root_dir, "test")
    
    else:
        print("Training new model ...")
        Model.fit_model()
        # model = Model.get_model()

    camera = FaceRecognition(model)
    camera.detect_from_video()