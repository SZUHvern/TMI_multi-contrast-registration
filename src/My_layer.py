# -*- coding: utf-8 -*-
from keras import *
from keras.layers import *
import tensorflow as tf
kernel_regularizer = regularizers.l2(5e-4)
bias_regularizer = regularizers.l2(5e-4)
kernel_regularizer = None
bias_regularizer = None

def my_lstm(input1, input2, channel=256):
    lstm_input1 = Reshape((1, input1.shape.as_list()[1], input1.shape.as_list()[2], input1.shape.as_list()[3]))(input1)
    lstm_input2 = Reshape((1, input2.shape.as_list()[1], input2.shape.as_list()[2], input1.shape.as_list()[3]))(input2)

    lstm_input = My_concat(axis=1)([lstm_input1, lstm_input2])
    x = ConvLSTM2D(channel, (3, 3), strides=(1, 1), padding='same')(lstm_input)
    return x

def conv_2(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = None, bn=True):
    print('conv_2')
    
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_) if bn==True else conv_
    conv_ = Activation('relu')(conv_)
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(conv_)
    conv_ = BatchNormalization()(conv_) if bn==True else conv_
    conv_ = Activation('relu')(conv_)   
    return conv_

def conv_2_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1), bn=True):
    print('conv_2_init')
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = None, bn=bn) 

def conv_2_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_2(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4)) 

def conv_1(inputs, filter_num, kernel_size=(3,3), strides=(1,1), kernel_initializer='glorot_uniform', kernel_regularizer = None, bn=True):
    print('conv_1')
    conv_ = Conv2D(filter_num, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=kernel_initializer, kernel_regularizer = kernel_regularizer)(inputs)
    conv_ = BatchNormalization()(conv_) if bn==True else conv_
    conv_ = Activation('relu')(conv_)
    return conv_

def conv_1_init(inputs, filter_num, kernel_size=(3,3), strides=(1,1), bn=True):
    print('conv_1_init')
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = None, bn=bn) 

def conv_1_init_regularization(inputs, filter_num, kernel_size=(3,3), strides=(1,1)):
    return conv_1(inputs, filter_num, kernel_size=kernel_size, strides=strides, kernel_initializer='he_normal', kernel_regularizer = regularizers.l2(5e-4)) 

def DW_conv(inputs, filter_num, dilation_rate):
    conv_ = DepthwiseConv2D((3, 3), dilation_rate=dilation_rate, depthwise_initializer='he_normal', padding="same")(inputs)
    conv_ = BatchNormalization()(conv_)
    conv_ = Activation("relu")(conv_)
    conv_ = conv_1(conv_, filter_num, (1, 1))
    return conv_

class My_concat(Layer):

    def __init__(self, axis=-1, **kwargs):
        super(My_concat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.built = True
        super(My_concat, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        self.res = tf.concat(x, self.axis)

        return self.res

    def compute_output_shape(self, input_shape):
        # return (input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:]
        # print((input_shape[0][0],)+(len(input_shape),)+input_shape[0][2:])
        input_shapes = input_shape
        output_shape = list(input_shapes[0])

        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]

        return tuple(output_shape)


class BilinearUpsampling(Layer):
    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def res_pool(conv, pool, num, strides=(2,2)):
    conv_pool = Conv2D(num, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(conv)
    conv_pool = BatchNormalization()(conv_pool)
    conv_pool = Activation('relu')(conv_pool)  
    res_pool = Add()([conv_pool, pool])
    return res_pool


def aspp(x, input_shape, out_stride, conv1, conv2, conv3, conv4):
    b0 = Conv2D(256, (1, 1), padding="same")(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(2, 2), padding="same")(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same")(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(4, 4), padding="same")(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same")(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same")(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same")(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape0 = int(input_shape[0] / out_stride)
    out_shape1 = int(input_shape[1] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape0, out_shape1))(x)
    b4 = Conv2D(256, (1, 1), padding="same")(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape0, out_shape1))(b4)

    dense1 = Conv2D(256, (3, 3), strides=(16,16), padding='same', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(conv1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)  
    
    dense2 = Conv2D(256, (3, 3), strides=(8,8), padding='same', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(conv2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2) 
    
    dense3 = Conv2D(256, (3, 3), strides=(4,4), padding='same', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(conv3)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3) 
    
    dense4 = Conv2D(256, (3, 3), strides=(2,2), padding='same', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(conv4)
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4) 
    
    lstm_input0 = Reshape((1, b0.shape.as_list()[1], b0.shape.as_list()[2], 256))(dense1)
    lstm_input1 = Reshape((1, b1.shape.as_list()[1], b0.shape.as_list()[2], 256))(dense2)
    lstm_input2 = Reshape((1, b2.shape.as_list()[1], b0.shape.as_list()[2], 256))(dense3)
    lstm_input3 = Reshape((1, b3.shape.as_list()[1], b0.shape.as_list()[2], 256))(dense4)
    lstm_input4 = Reshape((1, b0.shape.as_list()[1], b0.shape.as_list()[2], 256))(b0)
    lstm_input5 = Reshape((1, b1.shape.as_list()[1], b0.shape.as_list()[2], 256))(b1)
    lstm_input6 = Reshape((1, b2.shape.as_list()[1], b0.shape.as_list()[2], 256))(b2)
    lstm_input7 = Reshape((1, b3.shape.as_list()[1], b0.shape.as_list()[2], 256))(b3)
    lstm_input8 = Reshape((1, b3.shape.as_list()[1], b0.shape.as_list()[2], 256))(b4)

    #x = Concatenate()([dense1, dense2, dense3, dense4, b0, b1, b2, b3, b4])
    lstm_input = My_concat(axis=1)([lstm_input0, lstm_input1, lstm_input2, lstm_input3, lstm_input4, lstm_input5, lstm_input6, lstm_input7, lstm_input8])
    x = ConvLSTM2D(256, (3, 3), strides=(1, 1), padding='same')(lstm_input)
    
    return x