import numpy as np
import imageio
import cv2
from keras.models import load_model
import h5py
import sys
import os

def score_card(input_map, groundtruth_image, threshold):
    binary_map = input_map
    binary_map[binary_map < threshold] = 0
    binary_map[binary_map == threshold] = 0
    binary_map[binary_map > threshold] = 1

    [rows, cols] = groundtruth_image.shape

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i, j] == 1 and binary_map[i, j] == 1):  # TP condition
                TP = TP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j] == 1)):  # FP condition
                FP = FP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j] == 0)):  # TN condition
                TN = TN + 1
            elif ((groundtruth_image[i, j] == 1) and (binary_map[i, j] == 0)):  # FN condition
                FN = FN + 1

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    fScore = float(2 * precision * recall) / float(precision + recall)

    error_count = 0
    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i, j] != binary_map[i, j]):
                error_count = error_count + 1
    error_rate = float(error_count) / float(rows * cols)

    return (precision, recall, fScore, error_rate)
def calculate_map(input_image, saved_model):
    test_image = np.expand_dims(input_image, axis=0)
    show_test_image = np.squeeze(test_image)

    decoded_img = saved_model.predict(test_image)
    show_decoded_image = np.squeeze(decoded_img)

    return show_decoded_image
def calculate_score_threshold(input_map, groundtruth_image, threshold):
    binary_map = input_map
    binary_map[binary_map < threshold] = 0
    binary_map[binary_map == threshold] = 0
    binary_map[binary_map > threshold] = 1

    [rows, cols] = groundtruth_image.shape

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(rows):
        for j in range(cols):
            if (groundtruth_image[i, j] == 1 and binary_map[i, j] == 1):  # TP condition
                TP = TP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j] == 1)):  # FP condition
                FP = FP + 1
            elif ((groundtruth_image[i, j] == 0) and (binary_map[i, j] == 0)):  # TN condition
                TN = TN + 1
            elif ((groundtruth_image[i, j] == 1) and (binary_map[i, j] == 0)):  # FN condition
                FN = FN + 1

    tpr = float(TP) / float(TP + FN)
    fpr = float(FP) / float(FP + TN)

    return (tpr, fpr)
def generate_predicted_image(input_map,threshold):
    binary_map = input_map
    binary_map[binary_map < threshold] = 0
    binary_map[binary_map == threshold] = 0
    binary_map[binary_map > threshold] = 1
    return binary_map


if __name__ == '__main__':


    # predict the whole sky images
    output_predict_path = '*/temp/Prediction/WSISEG-Database/predicted images'
    output_true_path = '*/temp/Prediction/WSISEG-Database/original images'
    model_cloud = load_model('*/temp/Training/cloudsegnet.hdf5')
    data_image = h5py.File('*/temp/Dataset/WSISEG-Database/day_scene_withAUG.h5', 'r')
    data_image = data_image['sceneimage'][:]
    nums_data = data_image.shape[0]
    for i in range(nums_data):
        temp_image = data_image[i][::]
        image_map = calculate_map(temp_image, model_cloud)
        temp_predicted = generate_predicted_image(image_map, threshold=0.5)
        imageio.imwrite(os.path.join(output_predict_path, str(int(i)) + '.jpg'), image_map)
        cv2.imwrite(os.path.join(output_true_path, str(int(i)) + '.jpg'), temp_image)
        print(i, ' has done')

