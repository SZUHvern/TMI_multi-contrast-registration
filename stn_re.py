from src.models.utils import get_initial_weights
from src.models.layers import BilinearInterpolation
from model import *

def getSTN(input_shape_other, name):
    
    sampling_size = (224, 224)
    image = Input(shape=input_shape_other)
    image_mask = Input(shape=input_shape_other)
    image_mask_DWI = Input(shape=input_shape_other)
    input_ = Concatenate()([image_mask, image_mask_DWI])
    locnet = origin_block(32, input_, 0)
    locnet = Flatten()(locnet)
    locnet = Dense(128)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(128)
    locnet = Dense(6, weights=weights, name='stn_w'+name)(locnet)
    
    stn_out1 = BilinearInterpolation(sampling_size, name='stn_out'+name)([image_mask, locnet])
    stn_out12 = BilinearInterpolation(sampling_size, name='stn_out'+name+'2')([image, locnet])
    
    return image, image_mask, image_mask_DWI, stn_out1, stn_out12 