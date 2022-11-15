# Using the CloudSegNet model to detect clouds collected by Unibe

The CloudSegNet model is cited in the following paper:

> S. Dev, A. Nautiyal, Y. H. Lee, S. Winkler, CloudSegNet: A Deep Network for Nychthemeron Cloud Segmentation, IEEE Geoscience and Remote Sensing Letters, 2019.

![summary](./results/summary.png)

Please cite the above paper if you intend to use whole/part of the code. This code is only for academic and research purposes.


## Usage

1. Create the following folder structure inside `./temp`. 
    + `./temp/Dataset`: Contains the WSISEG dataset. The corresponding images, along with the corresponding ground-truth maps can be downloaded from [this](http://***.html) link. The images are saved inside `./temp/Dataset/WSISEG-Database/whole sky images` folder, and the corresponding ground-truth maps are saved inside `./temp/Dataset/WSISEG-Database/annotation`.
    + `./temp/Training`: Contains the trained model and corresponding test images as *.npy* type using the script `train_model.py`.
    + `./temp/Prediction`: Contains the prediction results based on WSISEG dataset and true Unibe images, which are computed using the script `predict_whole sky images.py` and `predict_Unibe images.py` respectively.
    + `./dataset/aug_SWINSEG`: Contains the augmented set of nightime images. It follows the similar structure, and is computed using the script `create_aug_night.py`. 
2. Run the script `python2 data_augmentation.py`. Please see the installation environment `requirements.txt`. You have to change this script, if you wish to make it compatible with the latest version of keras. It saves the augmented images inside `./temp/Dataset/WSISEG-Database/whole sky images_augmentation` and corresponding augmented masks inside `./temp/Dataset/WSISEG-Database/annotation_augmentation`.  
3. Run the script `python2 combine_images.py` for generating the `.h5` file for actual- and augmented- images. The results are stored in `./temp/Dataset/WSISEG-Database`.
4. Run the script `python2 train_model.py` for training the CloudSegNet model in the composite dataset containing actual- and augmented- images. The logfile and the model is saved inside the folder `./temp/Training`.
5. Run the script `python2 predict_whole sky images.py` for getting the prediction results of CloudSegNet model using the whole sky images. The corresponding predicted and actual images are stored in the `./temp/Prediction/WSISEG-Database/predicted images` and `./temp/Prediction/WSISEG-Database/original images`.
6. Run the script `python2 process_Unibe images.py` for processing the actual images collected by Unibe in order to satisfy the input sizes condition of the trained CloudSegNet model. The results are stored in `./temp/Dataset/Unibe images/processed_images`, and the corresponding inputs are stored in `./temp/Dataset/Unibe images/original images`
7. Run the script `python2 predict_Unibe images.py` for getting the prediction results of CloudSegNet model using the actual images collected by Unibe. The results are stored in `./temp/DataSet/Unibe images/predicted images` folder, and the corresponding actual images are stored in `./temp/DataSet/Unibe images/original images`.