from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Cropping2D

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    return (ch1, ch2), (cw1, cw2)

def Unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # block 1
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='block1_conv2')(x)
    skip1 = x
    x = MaxPooling2D(pool_size=(2,2), name='block1_pool', padding='same')(x)

    # block 2
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='block2_conv2')(x)
    skip2 = x
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool', padding='same')(x)

    # block 3
    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='block3_conv2')(x)
    skip3 = x
    x = MaxPooling2D(pool_size=(2, 2), name='block3_pool', padding='same')(x)

    # block 4
    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='block4_conv2')(x)
    skip4 = x
    x = MaxPooling2D(pool_size=(2, 2), name='block4_pool', padding='same')(x)

    # block 5
    x = Conv2D(filters=516, kernel_size=3, activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(filters=516, kernel_size=3, activation='relu', padding='same', name='block5_conv2')(x)

    # block 6 (upsample from block 5 and concatenate from block 4)
    x = UpSampling2D(size=(2,2), name='block6_upsample')(x)
    ch, cw = get_crop_shape(x, skip4)
    x = Cropping2D(cropping=(ch, cw), name='block6_cropping')(x)
    x = Concatenate()([skip4, x])
    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='block6_conv1')(x)
    x = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', name='block6_conv2')(x)

    # block 7 (upsample from block 6 and concatenate from block 3)
    x = UpSampling2D(size=(2,2), name='block7_upsample')(x)
    ch, cw = get_crop_shape(x, skip3)
    x = Cropping2D(cropping=(ch, cw), name='block7_cropping')(x)
    x = Concatenate()([skip3, x])
    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='block7_conv1')(x)
    x = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', name='block7_conv2')(x)

    # block 8 (upsample from block 7 and concatenate from block 2)
    x = UpSampling2D(size=(2,2), name='block8_upsample')(x)
    ch, cw = get_crop_shape(x, skip2)
    x = Cropping2D(cropping=(ch, cw), name='block8_cropping')(x)
    x = Concatenate()([skip2, x])
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='block8_conv1')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='block8_conv2')(x)

    # block 9 (upsample from block 8 and concatenate from block 1)
    x = UpSampling2D(size=(2,2), name='block9_upsample')(x)
    ch, cw = get_crop_shape(x, skip1)
    x = Cropping2D(cropping=(ch, cw), name='block9_cropping')(x)
    x = Concatenate()([skip1, x])
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='block9_conv1')(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', name='block9_conv2')(x)

    # output layer
    ch, cw = get_crop_shape(x, inputs)
    x = Cropping2D(cropping=(ch, cw), name='output_cropping')(x)
    x = Conv2D(filters=1, kernel_size=1, activation='relu', padding='same', name='output')(x)

    model = Model(inputs, x)

    return model
