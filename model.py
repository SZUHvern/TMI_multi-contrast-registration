from __future__ import print_function
from keras.models import Model
from keras.layers import *
from neuron.layers import SpatialTransformer
from src.models.utils import get_initial_weights
from src.models.layers import BilinearInterpolation

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
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def flowinverse(flow):
    flow0 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(flow)
    flow1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1], axis=-1))(flow)

    flow0 = SpatialTransformer(interp_method='linear', indexing='ij', name='inverse_stn1')([flow0, flow])
    flow1 = SpatialTransformer(interp_method='linear', indexing='ij', name='inverse_stn2')([flow1, flow])

    flow_inverse = Concatenate()([flow0, flow1])
    flow_inverse = Lambda(lambda x: x * -1.)(flow_inverse)
    return flow_inverse

def BN_block(filter_num, input, name, trainable=True):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_1',
               trainable=trainable)(input)
    x = BatchNormalization(name='BN' + name + '_1', trainable=trainable)(x)
    # x = LeakyReLU(name='LeakyReLU' + name + '_1')(x)
    x = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_2',
               trainable=trainable)(x)
    x = BatchNormalization(name='BN' + name + '_2', trainable=trainable)(x)
    # x = LeakyReLU(name='LeakyReLU' + name + '_2')(x)
    x = Activation('relu')(x)
    return x

def BN_block_leaky(filter_num, input, name, trainable=True):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_1',
               trainable=trainable)(input)
    x = BatchNormalization(name='BN' + name + '_1', trainable=trainable)(x)
    x = LeakyReLU(name='LeakyReLU' + name + '_1')(x)
    # x = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_2',
               trainable=trainable)(x)
    x = BatchNormalization(name='BN' + name + '_2', trainable=trainable)(x)
    x = LeakyReLU(name='LeakyReLU' + name + '_2')(x)
    # x = Activation('relu')(x)
    return x

def block_leaky_single(filter_num, input, name, trainable=True):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_1',
               trainable=trainable)(input)
    x = LeakyReLU(name='LeakyReLU' + name + '_1')(x)
    return x

def Affine_arch(input_stack, trainable=False, w=16):
    affine_net1 = BN_block_leaky(w, input_stack, name='affine_net1', trainable=trainable)
    pool1 = MaxPooling2D(pool_size=(2, 2))(affine_net1)
    affine_net2 = BN_block_leaky(w * 2, pool1, name='affine_net2', trainable=trainable)
    pool2 = MaxPooling2D(pool_size=(2, 2))(affine_net2)
    affine_net3 = BN_block_leaky(w * 2, pool2, name='affine_net3', trainable=trainable)
    pool3 = MaxPooling2D(pool_size=(2, 2))(affine_net3)
    affine_net4 = BN_block_leaky(w * 4, pool3, name='affine_net4', trainable=trainable)
    pool4 = MaxPooling2D(pool_size=(2, 2))(affine_net4)
    affine_net5 = BN_block_leaky(w * 4, pool4, name='affine_net5', trainable=trainable)
    affine_net5 = Dropout(0.3, name='affine_net5gdrop_1')(affine_net5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(affine_net5)
    affine_net6 = BN_block_leaky(w * 8, pool5, name='affine_net6', trainable=trainable)
    affine_net6 = Dropout(0.3, name='affine_net6gdrop_1')(affine_net6)

    affine_net = Flatten()(affine_net6)
    affine_net = Dense(32, activation='relu', name='affine_net_dense1', trainable=trainable)(affine_net)
    weights = get_initial_weights(32)
    affine_net = Dense(6, weights=weights, activation='linear', name='affine_net_dense2', trainable=trainable)(affine_net)

    return affine_net

def Unet_arch(input_stack, w=16, name='1'):
    conv1 = BN_block_leaky(w, input_stack, name=name + 'g1')
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BN_block_leaky(w*2, pool1, name=name + 'g2')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = BN_block_leaky(w*2, pool2, name=name + 'g3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #
    conv4 = BN_block_leaky(w * 4, pool3, name=name + 'g4')
    drop4 = Dropout(0.3, name=name + 'gdrop_1')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    conv5 = BN_block_leaky(w * 4, pool4, name=name + 'g5')
    drop5 = Dropout(0.3, name=name + 'gdrop_2')(conv5)

    up6 = Conv2D(w * 4, 2, padding='same', kernel_initializer='he_normal', name=name + 'gup6')(
        UpSampling2D(size=(2, 2))(drop5))
    up6 = BatchNormalization(name='BNup6')(up6)
    up6 = LeakyReLU(name='leakyup6')(up6)
    # up6 = Activation('relu')(up6)
    merge6 = Concatenate()([drop4, up6])
    conv6 = BN_block_leaky(w * 4, merge6, name=name + 'g6')
    #
    up7 = Conv2D(w * 2, 2, padding='same', kernel_initializer='he_normal', name=name + 'gup7')(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = BatchNormalization(name='BNup7')(up7)
    # up7 = Activation('relu')(up7)
    up7 = LeakyReLU(name='leakyup7')(up7)
    merge7 = Concatenate()([conv3, up7])
    conv7 = BN_block_leaky(w * 2, merge7, name=name + 'g7')
    #
    up8 = Conv2D(w*2, 2, padding='same', kernel_initializer='he_normal', name=name + 'gup8')(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = BatchNormalization(name='BNup8')(up8)
    up8 = LeakyReLU(name='leakyup8')(up8)
    # up8 = Activation('relu')(up8)
    merge8 = Concatenate()([conv2, up8])
    conv8 = BN_block_leaky(w*2, merge8, name=name +'g8')

    up9 = Conv2D(w, 2, padding='same', kernel_initializer='he_normal', name=name + 'gup9')(
        UpSampling2D(size=(2, 2))(conv8))
    up9 = BatchNormalization(name='BNup9')(up9)
    up9 = LeakyReLU(name='leakyup9')(up9)
    # up9 = Activation('relu')(up9)
    merge9 = Concatenate()([conv1, up9])
    conv9 = BN_block_leaky(w, merge9, name=name + 'g9')

    # x1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='conv' + name + '_1')(conv9)
    # x1 = BatchNormalization(name='BN' + 'end_1')(x1)
    # conv9 = Activation('tanh')(x1)

    return conv9

def Affine_net(w=16, trainable=False):
    src = Input(shape=(224, 224, 1))
    tgt = Input(shape=(224, 224, 1))

    input_stack = Concatenate()([src, tgt])
    affine_param = Affine_arch(input_stack, trainable=trainable, w=w)
    affine_result = BilinearInterpolation((224, 224), name='affine_result')([src, affine_param])

    model = Model(input=[src, tgt], output=[affine_result])
    return model

def dual_net(w):
    src = Input(shape=(224, 224, 1))
    tgt = Input(shape=(224, 224, 1))
    label = Input(shape=(224, 224, 1))

    input_stack = Concatenate()([src, tgt])
    affine_param = Affine_arch(input_stack, trainable=False)
    src_affine = BilinearInterpolation((224, 224), name='src_affine')([src, affine_param])

    flow_ori1 = Unet_arch(Concatenate()([src_affine, tgt]), w=w)
    flow = Conv2D(2, 3, name='flow', padding='same', activation='linear', kernel_initializer='he_normal')(flow_ori1)
    deformable = SpatialTransformer(interp_method='linear', indexing='ij', name='stn0')([src_affine, flow])

    flow_ori2 = flowinverse(flow)
    inverse_deformable = SpatialTransformer(interp_method='linear', indexing='ij', name='defo_iv')([deformable, flow_ori2])
    resduce = Lambda(lambda x: x[0] - x[1], name='rl')([src_affine, inverse_deformable])

    label_affine = BilinearInterpolation((224, 224), name='src_affine2')([label, affine_param])
    label_flow = SpatialTransformer(interp_method='linear', indexing='ij', name='stn4')([label_affine, flow])
    model = Model(input=[src, tgt, label], output=[flow, deformable, resduce, label_flow])

    return model
