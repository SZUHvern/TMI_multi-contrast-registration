import os
from model import *
from keras.preprocessing.image import ImageDataGenerator
import time
import h5py
import math
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
import shutil
from losses import *

def generator_train_stn(input, label, batch_size=32):
    # gen_image = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, rotation_range=5, fill_mode='constant')
    gen_image = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=0.1,
                                   featurewise_center=True, featurewise_std_normalization=True,
                                   horizontal_flip=True, fill_mode='constant')
    gen_mask = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=0.1,
                                   featurewise_center=True, featurewise_std_normalization=True,
                                   horizontal_flip=True, fill_mode='constant')

    train_image = gen_image.flow(input, batch_size=batch_size, shuffle=True, seed=1)
    train_mask = gen_mask.flow(label, batch_size=batch_size, shuffle=True, seed=1)

    while True:
        next_train = next(train_image)
        next_mask = next(train_mask)
        yield [next_train, next_mask], [next_mask]

def generator_val_stn(input, label, batch_size=32):
    gen_image2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    gen_mask2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    train_image = gen_image2.flow(input, batch_size=batch_size, shuffle=False)
    train_mask = gen_mask2.flow(label, batch_size=batch_size, shuffle=False)

    while True:
        next_train = next(train_image)
        next_mask = next(train_mask)
        yield [next_train, next_mask], [next_mask]

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path_h5_save = './ori_data/deal/'
    output_path = './model/'
    load_weight = ''
    mode = 'train'
    batch_size = 64
    lr = 1e-4
    w = 16

    h5_name = 'affine'
    subdir_savepic = h5_name
    output_path += h5_name + '/'
    #

    if mode == 'train':

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        model = Affine_net(w=w, trainable=True)
        model.compile(optimizer=Adam(lr=lr),
                      loss=design_loss().mi_clipmse,
                      metrics={'affine_result': dice_coef})
        model.summary()

        if load_weight != '':
            print('loading：', load_weight)
            model.load_weights(load_weight, by_name=True)
        else:
            print('no loading weight!')

        time_start = time.time()
        print(path_h5_save + 'train')
        h5 = h5py.File(path_h5_save + 'train')
        train_F = h5['F']
        train_DWI = h5['DWI']

        h5 = h5py.File(path_h5_save + 'test')
        val_F = h5['F']
        val_DWI = h5['DWI']
        label_dwi = h5['label_dwi']
        label_flair = h5['label_flair']

        num_train_steps = math.floor(len(train_F) / batch_size)
        num_val_steps = math.floor(len(val_F) / batch_size)

        print('training data:' + str(len(train_F)) + '  validation data:' + str(len(val_F)))
        print('used:', str(time.time() - time_start) + 's\n')
        time_start = time.time()

        train_data = generator_train_stn(train_F, train_DWI, batch_size=batch_size)
        val_data = generator_val_stn(val_F, val_DWI, batch_size=batch_size)

        earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        tensorboard = TensorBoard(log_dir='./tensorboard/' + h5_name + '/', batch_size=batch_size)

        checkpointer = ModelCheckpoint(output_path + 'epoch{epoch:03d}-{val_dice_coef:.2f}.h5',
                                       monitor='val_loss', mode='min', verbose=1,
                                       save_best_only=True)
        history = model.fit_generator(train_data, epochs=100, initial_epoch=0, steps_per_epoch=num_train_steps, shuffle=True,
                            callbacks=[checkpointer, tensorboard, earlystop], validation_data=val_data,
                            validation_steps=num_val_steps, verbose=2)
