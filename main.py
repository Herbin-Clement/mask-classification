from ast import Try
import os
import argparse

from lib import Data_processing, Data_visualisation
from lib.Model.TinyVGG import TinyVGG
from lib.video_capture.FaceRecognition import FaceRecognition
from lib.Model.TL_InceptionV3 import TL_InceptionV3
from lib.Model.TL_VGG19 import TL_VGG19
from lib.Model.TL_Resnet import TL_Resnet
from lib.Model.TL_Xception import TL_XCeption

def parse_args():
    """
    parse the arguments

    :rtype: bool, bool
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm', '--loadmodel', help="load model", action='store_true')    
    parser.add_argument('-nd', '--newdataset', help="newdataset", action='store_true')
    parser.add_argument('-T', '--TinyVGG', help='TinyVGG model', action='store_true')
    parser.add_argument('-V', '--VGG19', help='TinyVGG19 model', action='store_true')
    parser.add_argument('-I', '--InceptionV3', help='InceptionV3 model', action='store_true')
    parser.add_argument('-R', '--Resnet', help='Resnet model', action='store_true')
    parser.add_argument('-X', '--Xception', help='Xception model', action='store_true')
    args = parser.parse_args()

    if args.TinyVGG :
        model = TinyVGG
    elif args.VGG19 : 
        model = TL_VGG19
    elif args.IncetpionV3 : 
        model = TL_InceptionV3
    elif args.Resnet : 
        model = TL_Resnet
    elif args.Xception :
        model = TL_XCeption
    else : 
        model = TinyVGG

    loadmodel = args.loadmodel
    newdataset = args.newdataset
    return loadmodel, newdataset, model

if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    loadmodel, newdataset, Model = parse_args()
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
<<<<<<< HEAD
    m = Model(root_dir, dataset_dir, batch_size=32)
=======
    Model = TinyVGG(root_dir, dataset_dir, batch_size=32, nb_epochs=5)
>>>>>>> 03dc78dc7b1e3e0361bfce0967ef5c10fe769a98
    # Model = TL_InceptionV3(root_dir, dataset_dir, batch_size=32, nb_epochs=50)
    # Model = TinyVGG_grey(root_dir, dataset_dir, batch_size=32, nb_epochs=5)


    if loadmodel:
        print("Load model ...")
        try:
            if not (os.path.isdir(os.path.join("Trained_weights/",  type(Model).__name__))) :
            
        except:

        Model.load_weights(os.path.join("Trained_weights/TinyVGG", "cp-0030.ckpt"))
        # Model.load_weights(os.path.join("weights/TL_InceptionV3", "0", "cp-0012.ckpt"))
        
        model = Model.get_model()
        Data_visualisation.confusion_matrix(os.path.join(dataset_dir, "validation"), model)
    
    else:
        print("Training new model ...")
        Model.fit_model()
        # model = Model.get_model()

        camera = FaceRecognition(Model)
        camera.detect_from_video()