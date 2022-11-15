# Import the libraries
import numpy as np
import os
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping


def randomize_data_alltimes(original_img_hdf, original_mask_hdf,  no_of_dayimages, percentage_training, percentage_testing):

    a = np.ones(no_of_dayimages)

    image_type_array =  a

    (number_of_original, _, _, _) = original_img_hdf.shape

    number_of_training = (percentage_training/100.0)*number_of_original
    number_of_testing = (percentage_testing / 100.0) * number_of_original

    number_of_training = int(number_of_training)
    number_of_testing = int(number_of_testing)

    print ('Number of training Parent images = ',number_of_training)
    print ('Number of testing Parent images = ', number_of_testing)

    a = np.arange(number_of_original)
    np.random.shuffle(a)


    index_of_training = a[:number_of_training]
    index_of_testing = a[number_of_training: ]

    X_train = original_img_hdf[index_of_training]
    Y_train = original_mask_hdf[index_of_training]

    X_testing = original_img_hdf[index_of_testing]
    Y_testing = original_mask_hdf[index_of_testing]

    imagetype_testing = image_type_array[index_of_testing]


    return (X_train, X_testing, Y_train, Y_testing, imagetype_testing)



if __name__ == '__main__':


    # Train the deep learning model
    input_img = Input(shape=(300, 300, 3))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    print("shape of encoded", K.int_shape(encoded))



    #==============================================================================


    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='valid')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Convolution2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)
    print("shape of decoded", K.int_shape(decoded))

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


    #===============================================================
    # Reading the HDF files
    #===============================================================

    # original scene image
    h5f = h5py.File('*/temp/Dataset/day_scene_withAUG.h5', 'r')
    original_sceneimage = h5f['sceneimage'][:]
    h5f.close()
    print("original scene hdf5 file's shape", original_sceneimage.shape)

    # original ground truth image
    h5f = h5py.File('*/temp/Dataset/day_withAUG_GT.h5','r')
    original_GTmasks = h5f['GTmasks'][:]
    h5f.close()
    print("original ground truth hdf5 file's shape", original_GTmasks.shape)

    # normalize the mask data
    original_GTmasks = (original_GTmasks / 255).astype('int')
    (no_of_dayimages, _, _, _) = original_sceneimage.shape
    print ('no_of_images',no_of_dayimages)
    print ('Unique',np.unique(original_GTmasks.flatten()))


    #===============================================================
    # Creating the dataset for training our model
    #===============================================================
    print ('Shuffling the dataset and creating the various sets')
    (X_train, X_testing, Y_train, Y_testing, imagetype_testing) = randomize_data_alltimes(original_sceneimage, original_GTmasks, no_of_dayimages, percentage_training=80, percentage_testing=20)


    print ('X_train.shape: ',X_train.shape)
    print ('X_testing.shape: ',X_testing.shape)


    print ('Y_train.shape: ',Y_train.shape)
    print ('Y_testing.shape: ',Y_testing.shape)

    print ('imagetype_testing.shape: ',imagetype_testing.shape)

    print(np.unique(Y_testing))



    # Saving the testing images and ground truths (as they are always randomized)
    np.save('*/temp/Training/xtesting.npy', X_testing)
    np.save('*/temp/Training/ytesting.npy', Y_testing)
    np.save('*/temp/Training/imagetypetesting.npy', imagetype_testing)

    data = np.load('*/temp/Training/xtesting.npy')
    print ('from the saved data')
    print (data.shape)



    #===============================================================
    # Model training
    #===============================================================

    csv_logger = CSVLogger('*/temp/Training/logfile.txt')
    '''
    saves the model weights after each epoch if the validation loss decreased
    '''
    checkpointer = ModelCheckpoint(filepath='*/temp/Training/cloudsegnet.hdf5', verbose=1, save_best_only=True)
    # Add early termination condition to shorten training time
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
    autoencoder.fit(X_train, Y_train, epochs=800, batch_size=32,
                    validation_data=(X_testing, Y_testing), verbose=1,callbacks=[csv_logger,early_stopping,checkpointer])

