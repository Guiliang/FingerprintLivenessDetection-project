# aya43@sfu.ca; yataoz@sfu.ca; last modified 20161209

# Create Keras Models

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, merge, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

#vgg16 model
def create_vgg16(img_size, do=.5, include_top=True, weights=None):
    # Block 1
    inp = Input(shape=(3,)+img_size)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(inp)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x=Dropout(do)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x=Dropout(do)(x)
        x = Dense(1, activation='sigmoid', name='predictions')(x)

    # Create model
    model = Model(input=inp, output=x)

    return model


#bsif vector single-branch model
def create_mlp(vec_size, do=.5):
    img_inp=Input(shape=(3,)+img_size, name='img')
    inp = Input(shape=vec_size)
    x = Dense(4096, activation='relu')(inp)
    x=Dropout(do)(x)
    x=Dense(4096, activation='relu')(x)
    x=Dropout(do)(x)
    x=Dense(1, activation='sigmoid')(x)
    model=Model(input=inp, output=x)
    return model


#bsif+vgg16 multi-branch model
def create_multi_branch(img_size, vec_size, do=.5):
    img_inp=Input(shape=(3,)+img_size, name='img')
    cnn=create_vgg16(img_size, do=do, include_top=False)
    #cnn.layers.pop()
    #cnn.layers.pop()
    #cnn.layers.pop()
    #cnn.layers.pop()
    img_oup=cnn(img_inp)
    #img_oup=cnn.get_layer('fc1').output

    bsif_inp = Input(shape=vec_size, name='bsif')
    #mlp=create_mlp(vec_size)
    #bsif_oup=mlp(bsif_inp)

    merged=merge([img_oup, bsif_inp], mode='concat')
    merged=Dense(4096, activation='relu', name='fc2')(merged)
    merged=Dropout(do)(merged)
    merged=Dense(4096, activation='relu')(merged)
    merged=Dropout(do)(merged)
    output=Dense(1, activation='sigmoid')(merged)

    model=Model(input=[img_inp, bsif_inp], output=output)
    return model
    
