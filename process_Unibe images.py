# from scipy.misc import imread, imresize
import os
import numpy as np
import cv2
from keras.preprocessing import image
import h5py


if __name__ == '__main__':


    process_path = '*/temp/Dataset/Unibe images/original images'
    process_files = os.listdir(process_path)
    output_path = '*/temp/Dataset/Unibe images/processed_images'
    batches = []
    for i in range(len(process_files)):
        temp_process_path = os.path.join(process_path, process_files[i])
        print(['Reading scene image ', process_files[i]])
        temp_data = image.load_img(temp_process_path,
                                   target_size=(960, 1280))
        temp_data = image.img_to_array(temp_data)
        temp_data = temp_data[:,160:1120,:]
        temp_data = cv2.resize(temp_data, (300,300), interpolation=cv2.INTER_NEAREST)
        batches.append(temp_data)
    batches = np.array(batches, dtype=np.uint8)
    print(batches.shape)
    # saving the file
    h5f = h5py.File(os.path.join(output_path,'day_scene_withAUG.h5'),'w')
    h5f.create_dataset('sceneimage', data=batches)
    h5f.close()
    print('HDF file saved')


