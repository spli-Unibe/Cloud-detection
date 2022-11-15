# Import the libraries
import numpy as np
import keras
from keras.preprocessing import image
import imageio


def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = keras.preprocessing.image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = keras.preprocessing.image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
def random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5):
    if np.random.random() < u:
        theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = rotate(img, theta)
        mask = rotate(mask, theta)
    return img, mask
def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = keras.preprocessing.image.transform_matrix_offset_center(shear_matrix, h, w)
    x = keras.preprocessing.image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
def random_shear(img, mask, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = shear(img, sh)
        mask = shear(mask, sh)
    return img, mask
def random_flip(img, mask, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask
def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = keras.preprocessing.image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask
def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = keras.preprocessing.image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = keras.preprocessing.image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x
def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask
def get_image_and_mask(img_id):
    my_image = path_scene + 'ASC100-1006_' + str(img_id) + '.jpg'
    my_GT = path_mask + 'ASC100-1006_' + str(img_id) + '.jpg'
    img = image.load_img(my_image,
                         target_size=(input_size_height, input_size_width))
    img = image.img_to_array(img)
    mask = image.load_img(my_GT,
                          grayscale=True, target_size=(input_size_height, input_size_width))
    mask = image.img_to_array(mask)
    img, mask = img / 255., mask / 255.

    return img, mask
def random_augmentation(img, mask):

    img, mask = random_rotate(img, mask, rotate_limit=(-20, 20), u=0.5)
    img, mask = random_shear(img, mask, intensity_range=(-0.3, 0.3), u=0.2)
    img, mask = random_flip(img, mask, u=0.3)
    img, mask = random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.3)
    img, mask = random_zoom(img, mask, zoom_range=(0.8, 1), u=0.3)
    return img, mask
if __name__ == '__main__':

    # redefine your folder path
    path_scene = '*/temp/Dataset/WSISEG-Database/whole sky images/'
    path_mask = '*/temp/Dataset/WSISEG-Database/annotation/'
    path_output_scene = '*/temp/Dataset/WSISEG-Database/whole sky images_augmentation/'
    path_output_mask = '*/temp/Dataset/WSISEG-Database/annotation_augmentation/'

    # numbers of of your original images
    NO_OF_IMAGES = 400
    img_ids = []
    for i in range(NO_OF_IMAGES):
        if len(str(i+1)) == 3:
            item = str(i + 1)
        elif len(str(i+1)) == 2:
            item = '0' + str(i+1)
        elif len(str(i+1)) == 1:
            item = '00' + str(i+1)
        img_ids.append(item)
    # five data augmentation methods, which include rotation, shear, flip, shift and zoom.
    NO_OF_Xs = 5
    # define the dementions of the images you collect
    input_size_width = 480
    input_size_height = 450
    for img_id in img_ids:
        img, mask = get_image_and_mask(img_id)
        # Crop the arrays as 450*450 so that the length and width of the images are the same
        img = img[:,15:465]
        mask = mask[:,15:465]

        imageio.imwrite(path_output_scene + 'ASC100-1006_'+ img_id + '.jpg', img)
        imageio.imwrite(path_output_mask + 'ASC100-1006_' + img_id + '.jpg', mask[:, :, 0])

        print(['Processing input image for ', img_id])
        print(['Augmentation for ', img_id])
        for i in range(NO_OF_Xs):
            img_aug, mask_aug = random_augmentation(img, mask)

            imageio.imwrite(path_output_scene + 'ASC100-1006_'+ img_id + '_' + str(i) + '.jpg', img_aug)
            imageio.imwrite(path_output_mask + 'ASC100-1006_'+ img_id + '_' + str(i) + '.jpg', mask_aug[:, :, 0])