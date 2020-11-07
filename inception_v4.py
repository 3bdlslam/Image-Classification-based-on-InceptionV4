from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import plot_model
import keras 
import numpy as np
import cv2
from PIL import Image
import sys

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

TH_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_th_kernels.h5"
TH_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_th_kernels.h5"
TF_BACKEND_TF_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_tf_dim_ordering_tf_kernels.h5"
TF_BACKEND_TH_DIM_ORDERING = "https://github.com/titu1994/Inception-v4/releases/download/v1.2/inception_v4_weights_th_dim_ordering_tf_kernels.h5"


# define a block toaply convolusion 

def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    if K.image_dim_ordering() == "th":     #ckeck is the backend is theno or tensorflow          
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

# define a function to biult STEM LAYER  
def inception_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    x = keras.layers.concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid') # 

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = keras.layers.concatenate([x1, x2], axis=channel_axis)
   
    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

   
    x = keras.layers.concatenate([x1, x2], axis=channel_axis)
    return x

# define a function to biult inception A LAYER  
def inception_A(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = keras.layers.concatenate([a1, a2, a3, a4], axis=channel_axis)

    return m

# define a function to biult inception B LAYER  

def inception_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = keras.layers.concatenate([b1, b2, b3, b4], axis=channel_axis)
    return m

# define a function to biult inception C LAYER  

def inception_C(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c2 = keras.layers.concatenate([c2_1, c2_2], axis=channel_axis)
    
    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)

    c3 = keras.layers.concatenate([c3_1, c3_2], axis=channel_axis)
    
    c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = keras.layers.concatenate([c1, c2, c3, c4], axis=channel_axis)
    return m

# define a function to biult reduction A LAYER  

def reduction_A(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = keras.layers.concatenate([r1, r2, r3], axis=channel_axis)
    return m

# define a function to biult reduction B LAYER  

def reduction_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = keras.layers.concatenate([r1, r2, r3], axis=channel_axis)
    return m

# define a constructor for inception_V4 network

def create_inception_v4(nb_classes=1001, load_weights=True):
    '''
    Creates a inception v4 network

    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''

    if K.image_dim_ordering() == 'th':
        init = Input((3, 299, 299))
    else:
        init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')
# load weights 
    if load_weights:
        if K.backend() == "theano":
            if K.image_dim_ordering() == "th":
                weights = get_file('inception_v4_weights_th_dim_ordering_th_kernels.h5', TH_BACKEND_TH_DIM_ORDERING,
                                   cache_subdir='models')
            else:
                weights = get_file('inception_v4_weights_tf_dim_ordering_th_kernels.h5', TH_BACKEND_TF_DIM_ORDERING,
                                   cache_subdir='models')
        else:
            if K.image_dim_ordering() == "th":
                weights = get_file('inception_v4_weights_th_dim_ordering_tf_kernels.h5', TF_BACKEND_TH_DIM_ORDERING,
                                   cache_subdir='models')
            else:
                weights = get_file('inception_v4_weights_tf_dim_ordering_tf_kernels.h5', TH_BACKEND_TF_DIM_ORDERING,
                                   cache_subdir='models')

        model.load_weights(weights)
        print("Model weights loaded.")

    return model




def central_crop(image, central_fraction):
	"""Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
	if central_fraction <= 0.0 or central_fraction > 1.0:
		raise ValueError('central_fraction must be within (0, 1]')
	if central_fraction == 1.0:
		return image

	img_shape = image.shape
	depth = img_shape[2]
	fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
	bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
	bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

	bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
	bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

	image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
	return image


def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x

    
def get_processed_image(img_path):
    # Load image and convert from BGR to RGB
    im = np.asarray(cv2.imread(img_path))[:,:,::-1]
    im = central_crop(im, 0.875)
    im = cv2.resize(im, (299, 299))
    im = preprocess_input(im)
    if K.image_data_format() == "channels_first":
    	im = np.transpose(im, (2,0,1))
    	im = im.reshape(-1,3,299,299)
    else:
    	im = im.reshape(-1,299,299,3)
    return im


def get_pred(img_path):
     s = get_processed_image(img_path)   
     classes = eval(open('correct_classes.txt', 'r').read())
    
     #inception_v4.summary()
     inception_module = create_inception_v4()
     preds = inception_module.predict(s)
     
     cer = preds[0][np.argmax(preds)]
     
     
     clas = classes[np.argmax(preds)-1]
          
     return clas ,str(cer)    
          





if __name__ == "__main__":
     #from keras.utils.visualize_util import plot
     
    
     
    
 
     img_path = 'C:/Users/abdel/Downloads/48082373_217745382458425_1454068110138015744_n.jpg'
     img = get_processed_image(img_path)   
     classes = eval(open('correct_classes.txt', 'r').read())
    
     #inception_v4.summary()
     inception_module = create_inception_v4()
      #inception_module.summary()
     preds = inception_module.predict(img)
     print("Class is: " + classes[np.argmax(preds)-1])
     print("Certainty is: " + str(preds[0][np.argmax(preds)]))
 
     #predict(x =img)
     
     # inception_v4.input('basketball.png')
     # plot(inception_v4, to_file="father_circa_college.png", show_shapes=True)
     #plot_model( inception_module , to_file='basketball.png', show_shapes=True, show_layer_names=True)  