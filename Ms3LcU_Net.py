from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from efficientnet.keras import EfficientNetB5
K.set_image_data_format('channels_last')
INITIAL_WEIGHTS = 'imagenet'
kinit = 'he_normal'
lrelu_alpha = 0.3
dropout_rate=0.2

def BMA_Block(x, g, outdim):
    x2 = ConvAFBN(x, outdim//2, 3)
    x3 = ConvAFBN(x, outdim//2, 3)
    x3 = ConvAFBN(x3, outdim//2, 3)
    x3 = ConvAFBN(x3, outdim//2, 3)

    g2 = ConvAFBN(g, outdim//2, 3)
    g2 = ConvAFBN(g2, outdim//2, 3)
    g3 = ConvAFBN(g, outdim//2, 3)
    g3 = ConvAFBN(g3, outdim//2, 3)
    g3 = ConvAFBN(g3, outdim//2, 3)
    g3 = ConvAFBN(g3, outdim//2, 3)

    merge1 = add([x2, x3])
    merge11 = ConvAFBN(merge1, outdim, 1)

    merge2 = add([g2, g3])
    merge2 = ConvAFBN(merge2, outdim, 1)


    sigmoid_xg1 = Activation('sigmoid')(merge11)
    sigmoid_xg2 = Activation('sigmoid')(merge2)
    merge1 = multiply([sigmoid_xg2, merge11])
    merge2 = multiply([sigmoid_xg1, merge2])
    model_merge_backend = concatenate([merge1, merge2], axis=3)

    return model_merge_backend

def Dialeted_fusion_block(x, kernel_num):
    cur_enc = aspp(x[1], kernel_num//2)
    cur_enc = Conv2D(kernel_num//2, 1,kernel_initializer=kinit, padding='same',
                       use_bias=False)(cur_enc)

    cur_dec = aspp(x[2], kernel_num//2)
    cur_dec = Conv2D(kernel_num//2, 1,kernel_initializer=kinit, padding='same',
                       use_bias=False)(cur_dec)

    conca = concatenate([cur_enc, cur_dec], axis=3)

    pre_fusion = Conv2D(kernel_num, 3,use_bias=False, kernel_initializer=kinit, dilation_rate=2,
                padding='same')(x[0])
    pre_fusion = BatchNormalization()(pre_fusion)
    pre_fusion = Activation(LeakyReLU(alpha=lrelu_alpha))(pre_fusion)

    outfff = Add()([pre_fusion, conca])
    outedge = ConvAFBN(outfff, kernel_num//2, 1)
    outedge = ConvAFBN(outedge, kernel_num//2, 3)
    outconc = concatenate([outedge, outfff], axis=3)

    return outconc

def aspp(x, num_filters):
    x0 = Conv2D(num_filters//2, (1, 1), padding='same',use_bias=False,
                kernel_initializer=kinit)(x)

    d0 = MaxPooling2D(pool_size=(2, 2))(x)
    d0 = Conv2D(num_filters//2, (1,1), padding='same', strides=(1, 1), kernel_initializer=kinit,
                use_bias=False)(d0)
    d0 = BatchNormalization()(d0)
    d0 = Activation(LeakyReLU(alpha=lrelu_alpha))(d0)
    d0 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(d0)

    d1 = Conv2D(num_filters//2, (3, 3), dilation_rate=2, padding='same',use_bias=False,
                          kernel_initializer=kinit)(x)
    d1 = BatchNormalization()(d1)
    d1 = Activation(LeakyReLU(alpha=lrelu_alpha))(d1)

    d2 = Conv2D(num_filters//2, (3, 3), dilation_rate=4, padding='same',use_bias=False,
                          kernel_initializer=kinit)(x)
    d2 = BatchNormalization()(d2)
    d2 = Activation(LeakyReLU(alpha=lrelu_alpha))(d2)

    d3 = Conv2D(num_filters//2, (3, 3), dilation_rate=6, padding='same',use_bias=False,
                          kernel_initializer=kinit)(x)
    d3 = BatchNormalization()(d3)
    d3 = Activation(LeakyReLU(alpha=lrelu_alpha))(d3)

    c4 = concatenate([d0, d1, d2,d3,x0],axis=3)

    return c4

def tiny_unet(x, kernel_num):
    convt1 = ConvAFBN(x, kernel_num, 3)
    main_path = MaxPooling2D(pool_size=(2, 2))(convt1)
    main_path = ConvAFBN(main_path, kernel_num, 3)
    main_path = ConvAFBN(main_path, 2*kernel_num, 3)
    main_path = ConvAFBN(main_path, kernel_num, 3)
    main_path = Conv2DTranspose(kernel_num, (3, 3), strides=(2, 2), activation=LeakyReLU(alpha=lrelu_alpha),use_bias=False,
                                padding="same", kernel_initializer=kinit)(
        main_path)
    main_path = concatenate([main_path, convt1], axis=3)
    main_path = ConvAFBN(main_path, kernel_num, 3)

    return main_path

def residual_block(blockInput, num_filters):
    x0 = Conv2D(num_filters, 1,use_bias=False,  kernel_initializer=kinit,
                padding='same',  )(blockInput)
    x0 = BatchNormalization()(x0)
    x0 = Activation(LeakyReLU(alpha=lrelu_alpha))(x0)

    x1 = Conv2D(num_filters, 3, use_bias=False, kernel_initializer=kinit,
                padding='same',  )(blockInput)
    x1 = BatchNormalization()(x1)
    x1 = Activation(LeakyReLU(alpha=lrelu_alpha))(x1)

    x2 = Conv2D(num_filters//2, 3, use_bias=False, kernel_initializer=kinit,
                padding='same',  )(x1)
    x2 = BatchNormalization()(x2)
    x2 = Activation(LeakyReLU(alpha=lrelu_alpha))(x2)

    x3 = Conv2D(num_filters, 3, use_bias=False, kernel_initializer=kinit,
                padding='same',  )(x2)
    x3 = BatchNormalization()(x3)
    x3 = Activation(LeakyReLU(alpha=lrelu_alpha))(x3)

    x = Add()([x0, x3])

    return x

def ConvAFBN(x, nb_filters, kernal_size):
    path = Conv2D(nb_filters, (kernal_size, kernal_size), padding='same', strides=(1, 1),
                  use_bias=False, kernel_initializer=kinit, )(x)
    path = BatchNormalization()(path)
    path = Activation(LeakyReLU(alpha=lrelu_alpha))(path)

    return path

def TransConvAfBn(x, nb_filters, kernal_size):
    deconv4 = Conv2DTranspose(nb_filters, (kernal_size, kernal_size), strides=(2, 2),use_bias=False,
                              padding="same", kernel_initializer=kinit,  )(x)
    deconv4 = BatchNormalization()(deconv4)
    deconv4 = Activation(LeakyReLU(alpha=lrelu_alpha))(deconv4)
    return deconv4


def encoder128(input128,start_neurons,connect64,connect32,connect16,mid64):
#128
    conv1 = ConvAFBN(input128, start_neurons * 4, 3)
    conv1 = ConvAFBN(conv1, start_neurons * 2, 3)
    conv1 = BMA_Block(tiny_unet(conv1, start_neurons * 2),conv1,start_neurons * 2)
    pool1 = AveragePooling2D((2, 2))(conv1)
    conv1 = concatenate(
    [Conv2DTranspose(start_neurons * 2, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool1),
     conv1], axis=3)
#64
    conv2 = AveragePooling2D((2, 2))(conv1)
    conv2 = ConvAFBN(conv2, start_neurons * 8, 3)
    conv2 = ConvAFBN(conv2, start_neurons * 4, 3)
    conv2 = BMA_Block(tiny_unet(conv2, start_neurons * 4), conv2, start_neurons * 4)
    pool2 = AveragePooling2D((2, 2))(conv2)
    conv2 = concatenate(
    [Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool2),
     conv2], axis=3)
    conv2 = concatenate([conv2,connect64],axis=-1)
    #32
    conv3 = AveragePooling2D((2, 2))(conv2)
    conv3 = ConvAFBN(conv3, start_neurons * 16, 3)
    conv3 = ConvAFBN(conv3, start_neurons * 8, 3)
    conv3 = BMA_Block(tiny_unet(conv3, start_neurons * 8), conv3, start_neurons * 8)
    pool3 = AveragePooling2D((2, 2))(conv3)
    conv3 = concatenate(
    [Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool3),
     conv3], axis=3)
    conv3 = concatenate([conv3, connect32], axis=-1)

#16
    conv4 = AveragePooling2D((2, 2))(conv3)
    conv4 = ConvAFBN(conv4, start_neurons * 32, 3)
    conv4 = ConvAFBN(conv4, start_neurons * 16, 3)
    conv4 = BMA_Block(tiny_unet(conv4, start_neurons * 16), conv4, start_neurons * 16)
    pool4 = AveragePooling2D((2, 2))(conv4)
    conv4 = concatenate(
    [Conv2DTranspose(start_neurons * 16, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool4),
     conv4], axis=3)
    conv4 = concatenate([conv4, connect16], axis=-1)
#8
    mid128 = AveragePooling2D((2, 2))(conv4)
    mid128 = concatenate([mid128, mid64], axis=-1)
    return mid128, conv4, conv3, conv2, conv1


def encoder64(input64,start_neurons):
#64
    conv2 = ConvAFBN(input64, start_neurons * 8, 3)
    conv2 = ConvAFBN(conv2, start_neurons * 4, 3)
    conv2 = BMA_Block(tiny_unet(conv2, start_neurons * 4), conv2, start_neurons * 4)
    pool2 = AveragePooling2D((2, 2))(conv2)
    conv2 = concatenate(
    [Conv2DTranspose(start_neurons * 4, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool2),
     conv2], axis=3)
    #32
    conv3 = AveragePooling2D((2, 2))(conv2)
    conv3 = ConvAFBN(conv3, start_neurons * 16, 3)
    conv3 = ConvAFBN(conv3, start_neurons * 8, 3)
    conv3 = BMA_Block(tiny_unet(conv3, start_neurons * 8), conv3, start_neurons * 8)
    pool3 = AveragePooling2D((2, 2))(conv3)
    conv3 = concatenate(
    [Conv2DTranspose(start_neurons * 8, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool3),
     conv3], axis=3)
#16
    conv4 = AveragePooling2D((2, 2))(conv3)
    conv4 = ConvAFBN(conv4, start_neurons * 32, 3)
    conv4 = ConvAFBN(conv4, start_neurons * 16, 3)
    conv4 = BMA_Block(tiny_unet(conv4, start_neurons * 16), conv4, start_neurons * 16)
    pool4 = AveragePooling2D((2, 2))(conv4)
    conv4 = concatenate(
    [Conv2DTranspose(start_neurons * 16, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(pool4),
     conv4], axis=3)
#8
    mid64 = AveragePooling2D((2, 2))(conv4)

    return mid64, conv4, conv3, conv2



def deco128(input,start_neurons,connect64,connect32,connect16,conv4_128, conv3_128, conv2_128):
    # 16
    deconv4 = TransConvAfBn(input, start_neurons * 64, 3)
    x4 = BMA_Block(connect16,deconv4, 16)
    deconv4 = Dialeted_fusion_block((x4, conv4_128, deconv4), start_neurons * 32)
    deconv4 = residual_block(deconv4, start_neurons * 32)
    deconv4 = ConvAFBN(deconv4, start_neurons * 32, 3)

    # 32
    deconv3 = TransConvAfBn(deconv4, start_neurons * 32, 3)
    x3 = BMA_Block(connect32,deconv3, 16)
    deconv3 = Dialeted_fusion_block((x3, conv3_128, deconv3), start_neurons * 16)
    deconv3 = residual_block(deconv3, start_neurons * 16)
    deconv3 = ConvAFBN(deconv3, start_neurons * 16, 3)

    # 64
    deconv2 = TransConvAfBn(deconv3, start_neurons * 16, 3)
    x2 = BMA_Block(connect64,deconv2, 16)
    deconv2 = Dialeted_fusion_block((x2, conv2_128, deconv2), start_neurons * 8)
    deconv2 = residual_block(deconv2, start_neurons * 8)
    deconv2 = ConvAFBN(deconv2, start_neurons * 8, 3)

    # 128
    deconv1 = TransConvAfBn(deconv2, start_neurons * 8, 3)
    deconv1 = residual_block(deconv1, start_neurons * 4)
    deconv1 = ConvAFBN(deconv1, start_neurons * 4, 3)

    out128 = Conv2D(2, (1, 1), padding="same", activation="softmax", name='pred128',
                          kernel_initializer=kinit)(deconv1)

    return out128,deconv2,deconv3,deconv4


def deco64(input, start_neurons):
    # 16
    deconv4 = TransConvAfBn(input, start_neurons * 64, 3)
    deconv4 = residual_block(deconv4, start_neurons * 32)
    deconv4 = ConvAFBN(deconv4, start_neurons * 32, 3)

    # 32
    deconv3 = TransConvAfBn(deconv4, start_neurons * 32, 3)
    deconv3 = residual_block(deconv3, start_neurons * 16)
    deconv3 = ConvAFBN(deconv3, start_neurons * 16, 3)

    # 64
    deconv2 = TransConvAfBn(deconv3, start_neurons * 16, 3)
    deconv2 = residual_block(deconv2, start_neurons * 8)
    deconv2 = ConvAFBN(deconv2, start_neurons * 8, 3)

    out64 = Conv2D(2, (1, 1), padding="same", activation="softmax", name='pred64',
                    kernel_initializer=kinit)(deconv2)


    return out64, deconv3, deconv4

def Ms3LcUNet(input_size=(None, None, 3),input_size0=(None, None, 3),input_size1=(None, None, 3), start_neurons=8):
    input128 = Input(shape=input_size0)
    input64 = Input(shape=input_size1)

    mid64, conv4_64, conv3_64, conv2_64=encoder64(input64, start_neurons)
    mid128, conv4_128, conv3_128, conv2_128, conv1_128=encoder128(input128,start_neurons,conv2_64,conv3_64,conv4_64,mid64)

    backbone = EfficientNetB5(weights=INITIAL_WEIGHTS,
                              include_top=False,
                              input_shape=input_size)

    input = backbone.input

    ccc0=backbone.layers[0].output
    ccc1=backbone.get_layer("block2a_expand_activation").output
    ccc2=backbone.get_layer("block3a_expand_activation").output
    ccc3=backbone.get_layer("block4a_expand_activation").output
    ccc4=backbone.get_layer("block6a_expand_activation").output

    conv0 = BMA_Block(tiny_unet(ccc0, start_neurons * 1),ccc0,start_neurons * 1)
    conv1 = BMA_Block(tiny_unet(ccc1, start_neurons * 2),ccc1,start_neurons * 2)
    conv2 = BMA_Block(tiny_unet(ccc2, start_neurons * 4),ccc2,start_neurons * 4)
    conv3 = BMA_Block(tiny_unet(ccc3, start_neurons * 8),ccc3,start_neurons *8)
    conv4 = BMA_Block(tiny_unet(ccc4, start_neurons * 16),ccc4,start_neurons * 16)

    conv1 = concatenate([conv1,conv1_128], axis=-1)
    conv2 = concatenate([conv2,conv2_128], axis=-1)
    conv3 = concatenate([conv3,conv3_128], axis=-1)
    conv4 = concatenate([conv4,conv4_128], axis=-1)

    pool4 = AveragePooling2D((2, 2))(conv4)

    convmain=aspp(pool4, start_neurons * 64)
    convm128=aspp(mid128, start_neurons * 64)
    convm64=aspp(mid64, start_neurons * 64)
    avg=average([convmain,convm128,convm64])

    convm = concatenate([convmain, convm128], axis=-1)
    convm = concatenate([convm, convm64], axis=-1)
    convm = concatenate([convm, avg], axis=-1)
    convm = ConvAFBN(convm, start_neurons * 128, 3)

    out64, deconv3_64, deconv4_64=deco64(convm64, start_neurons)
    out128, deconv2_128, deconv3_128, deconv4_128=deco128(convm128,start_neurons,out64, deconv3_64, deconv4_64,conv4_128, conv3_128, conv2_128)

    # D4
    deconv4 = TransConvAfBn(convm, start_neurons * 64, 3)
    x4 = BMA_Block(deconv4_128,deconv4, 16)
    deconv4 = Dialeted_fusion_block((x4, conv4, deconv4), start_neurons * 32)
    deconv4 = residual_block(deconv4, start_neurons * 32)
    deconv4 = ConvAFBN(deconv4, start_neurons * 32, 3)

    # D3
    deconv3 = TransConvAfBn(deconv4, start_neurons * 32, 3)
    x3 = BMA_Block(deconv3_128,deconv3, 16)
    deconv3 = Dialeted_fusion_block((x3, conv3, deconv3), start_neurons * 16)
    deconv3 = residual_block(deconv3, start_neurons * 16)
    deconv3 = ConvAFBN(deconv3, start_neurons * 16, 3)

    # D2
    deconv2 = TransConvAfBn(deconv3, start_neurons * 16, 3)
    x2 = BMA_Block(deconv2_128,deconv2, 16)
    deconv2 = Dialeted_fusion_block((x2, conv2, deconv2), start_neurons * 8)
    deconv2 = residual_block(deconv2, start_neurons * 8)
    deconv2 = ConvAFBN(deconv2, start_neurons * 8, 3)

    # D1
    deconv1 = TransConvAfBn(deconv2, start_neurons * 8, 3)
    x1 = BMA_Block(out128,deconv1, 16)
    deconv1 = Dialeted_fusion_block((x1, conv1, deconv1), start_neurons * 4)
    deconv1 = residual_block(deconv1, start_neurons * 4)
    deconv1 = ConvAFBN(deconv1, start_neurons * 4, 3)

    # D0
    deconv0 = TransConvAfBn(deconv1, start_neurons * 4, 3)
    x0 = concatenate([BMA_Block(conv0,deconv0, 16),deconv0],axis=-1)
    deconv0 = residual_block(x0, start_neurons * 2)
    uconv0 = ConvAFBN(deconv0, start_neurons * 2, 3)
    output_layer = Conv2D(2, (1, 1), padding="same", activation="softmax", name='final',
                          kernel_initializer=kinit)(uconv0)
    model = Model(inputs=[input,input128,input64], outputs=[output_layer,out128,out64])

    return model

