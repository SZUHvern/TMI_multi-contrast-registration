import tensorflow as tf
from keras import backend as K
import numpy as np

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def MSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def residuce_loss(y_true, y_pred):
    return K.mean(K.square(y_pred))

def none_loss(y_true, y_pred):
    return tf.convert_to_tensor([0.])

def net_mi(x):
    x1 = x[0]
    x2 = x[1]
    return mi(x1, x2)

def mi(y_true, y_pred):
    bin_centers = np.linspace(0, 1, 100) # return specified interval numbers

    sigma_ratio = 1
    crop_background = False

    vol_bin_centers = K.variable(bin_centers)
    sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
    preterm = K.variable(1 / (2 * np.square(sigma)))

    y_pred = K.clip(y_pred, 0, 1)
    y_true = K.clip(y_true, 0, 1)

    if crop_background:
        # does not support variable batch size
        thresh = 0.0001
        padding_size = 20
        filt = tf.ones([padding_size, padding_size, 1, 1])

        smooth = tf.nn.conv2d(y_true, filt, [1, 1, 1, 1], "SAME")
        mask = smooth > thresh
        # mask = K.any(K.stack([y_true > thresh, y_pred > thresh], axis=0), axis=0)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = K.expand_dims(K.expand_dims(y_pred, 0), 2)
        y_true = K.expand_dims(K.expand_dims(y_true, 0), 2)

    else:
        # reshape: flatten images into shape (batch_size, heightxwidthxdepthxchan, 1)
        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)

    nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

    # reshape bin centers to be (1, 1, B)
    o = [1, 1, np.prod(vol_bin_centers.get_shape().as_list())]
    vbc = K.reshape(vol_bin_centers, o)

    # compute image terms
    I_a = K.exp(- preterm * K.square(y_true - vbc))
    I_a /= K.sum(I_a, -1, keepdims=True)

    I_b = K.exp(- preterm * K.square(y_pred - vbc))
    I_b /= K.sum(I_b, -1, keepdims=True)

    # compute probabilities
    I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
    pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
    pab /= nb_voxels
    pa = tf.reduce_mean(I_a, 1, keep_dims=True)
    pb = tf.reduce_mean(I_b, 1, keep_dims=True)

    papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
    mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)
    return - mi

class design_loss():

    def __init__(self, parameter=1, parameter_mi=1, win=9, parameter_threth = 0.1):
        self.parameter = parameter
        self.parameter_mi = parameter_mi
        self.win = [win, win]
        self.jl_threth = parameter_threth

    def _local_map(self, var):
        return tf.nn.conv2d(var, tf.ones([*self.win, 1, 1]), strides=[1, 1, 1, 1], padding='SAME') / (self.win[0] * self.win[1])

    def gradient(self, var):
        grad_var_nor = K.spatial_2d_padding(var, padding=((1, 1), (1, 1)), data_format=None)
        grad_var_1 = K.spatial_2d_padding(var, padding=((2, 0), (1, 1)), data_format=None)
        grad_var_2 = K.spatial_2d_padding(var, padding=((0, 2), (1, 1)), data_format=None)
        grad_var_3 = K.spatial_2d_padding(var, padding=((1, 1), (2, 0)), data_format=None)
        grad_var_4 = K.spatial_2d_padding(var, padding=((1, 1), (0, 2)), data_format=None)
        grad_var = K.abs(grad_var_nor - grad_var_1) + K.abs(grad_var_nor - grad_var_2) + \
                    K.abs(grad_var_nor - grad_var_3) + K.abs(grad_var_nor - grad_var_4)

        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[1] - 1), axis=1)
        grad_var = tf.gather(grad_var, tf.range(1, tf.shape(grad_var)[2] - 1), axis=2)

        return grad_var

    def _gradient_tf(self, var):
        grad_var = K.abs(tf.image.image_gradients(var))
        grad_var = grad_var[0, :, :, :, :] + grad_var[1, :, :, :, :]
        return grad_var

    def smooth(self, y_true, y_pred):
        grad_pred = self.gradient(y_pred)
        return K.mean(grad_pred * grad_pred)


    def L1(self, y_true, y_pred):
        return K.mean(K.abs(y_true - y_pred))

    def gl1(self, y_true, y_pred):
        grad_ture = self.gradient(y_true)
        grad_pred = self.gradient(y_pred)
        grad_ture = K.sigmoid(self.gradient(grad_ture))
        grad_pred = K.sigmoid(self.gradient(grad_pred))
        return K.mean(K.abs(grad_ture - grad_pred))

    def gl2(self, y_true, y_pred):
        grad_ture = self.gradient(y_true)
        grad_pred = self.gradient(y_pred)
        grad_ture = K.sigmoid(self.gradient(grad_ture))
        grad_pred = K.sigmoid(self.gradient(grad_pred))
        return MSE(grad_ture, grad_pred)

    def mi_gl1(self, y_true, y_pred):
        grad_ture = K.sigmoid(self.gradient(y_true))
        grad_pred = K.sigmoid(self.gradient(y_pred))
        return self.parameter*self.L1(grad_ture, grad_pred) + self.parameter_mi * mi(y_true, y_pred)

    def mi_gmi(self, y_true, y_pred):
        grad_ture = K.sigmoid(self.gradient(y_true))
        grad_pred = K.sigmoid(self.gradient(y_pred))
        return self.parameter * mi(grad_ture, grad_pred) + self.parameter_mi * mi(y_true, y_pred)

    def mi_gl2(self, y_true, y_pred):
        grad_ture = K.sigmoid(self.gradient(y_true))
        grad_pred = K.sigmoid(self.gradient(y_pred))
        return self.parameter*MSE(grad_ture, grad_pred) + self.parameter_mi *  mi(y_true, y_pred)

    def mi_gl2local(self, y_true, y_pred):
        grad_ture = self.gradient(y_true)
        grad_pred = self.gradient(y_pred)
        local_ture = self._local_map(grad_ture)
        local_pred = self._local_map(grad_pred)
        return self.parameter * MSE(local_ture, local_pred) + self.parameter_mi *  mi(y_true, y_pred)

    def mi_gl2local_mine(self, y_true, y_pred):
        grad_ture = self.gradient(y_true)
        grad_pred = self.gradient(y_pred)
        local_ture = self._local_map(grad_ture)
        local_pred = self._local_map(grad_pred)
        return self.parameter * MSE(local_ture, local_pred) + self.parameter_mi * mi(y_true, y_pred)

    def _clip(self, y_true):
        threth = self.jl_threth
        y_round = K.round((K.clip(y_true, 0, threth*2))/(threth*2))
        return y_round

    def mi_clipmse(self, y_true, y_pred):
        round = self._clip(y_true)
        return self.parameter * MSE((1-round)*y_true, (1-round)*y_pred) + self.parameter_mi * mi(y_true, y_pred)






