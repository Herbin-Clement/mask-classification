import os
import argparse

from lib import Data_processing, Data_visualisation
from lib.Model.TinyVGG import TinyVGG
# from lib.Model.TL_InceptionV3 import TL_InceptionV3
# from lib.Model.TinyVGG_grey import TinyVGG_grey
from lib.video_capture.face_recognition import detect_from_video

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

if __name__ == "__main__":
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    loadmodel, newdataset = parse_args()
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
        process_data.test_train_validation_split_from_csv(data_ratio=1)
        process_data.create_train_test_validation_folder()
    Model = TinyVGG(root_dir, dataset_dir, batch_size=32)
    # Model = TL_InceptionV3(root_dir, dataset_dir, batch_size=32, nb_epochs=50)
    # Model = TinyVGG_grey(root_dir, dataset_dir, batch_size=32, nb_epochs=5)


    if loadmodel:
        print("Load model ...")
        # Model.load_weights()
        Model.load_weights(os.path.join("Trained_weights/TinyVGG", "cp-0030.ckpt"))
        model = Model.get_model()
        Data_visualisation.confusion_matrix(os.path.join(dataset_dir, "validation"))
        # Model.save_model()
        # Model.load_model()
        # detect_from_video(model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image/000010_1_000010_MALE_24.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image/000010_2_000010_MALE_24.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image/000010_3_000010_MALE_24.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image/000010_4_000010_MALE_24.jpg"), model)
    
    else:
        # print("Training new model ...")
        Model.fit_model()
        # model = Model.get_model()
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image\060002_1_028450_FEMALE_30.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image\060002_2_028450_FEMALE_30.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image\060002_3_028450_FEMALE_30.jpg"), model)
        # Data_visualisation.predict_validation_image(os.path.join(r"resize_image\060002_4_028450_FEMALE_30.jpg"), model)