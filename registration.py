import os
import plot
from model import *
from losses import *
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import time
import h5py
import math
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
import shutil
import cv2
import numpy as np
import os
import pystrum.pynd.ndutils as nd

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def cal_patient(gt, predict, flow):
    patient = 0
    dice_PGDC = 0
    recall_PGDC = 0
    precision_PGDC = 0
    patient_continue = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(gt)):
        if gt[i].max() > 0:
            TP += plot.cal_seg(gt[i], predict[i]).TP()
            FP += plot.cal_seg(gt[i], predict[i]).FP()
            TN += plot.cal_seg(gt[i], predict[i]).TN()
            FN += plot.cal_seg(gt[i], predict[i]).FN()
            if patient_continue != i-1 or i == len(gt):
                dice_PGDC += 2*TP/(2*TP+FP+FN)
                recall_PGDC += TP/(TP+FN)
                precision_PGDC += TP/(TP+FP)
                patient_continue = i
                patient += 1
                TP = 0
                FP = 0
                TN = 0
                FN = 0
    print(str(h5_name))
    jac_list = jac_from_output(flow)
    jac_mean = int(sum(jac_list)/len(jac_list))
    jac_standard = int(np.std(jac_list))

    file_name = '/home/siat/桌面/affine.txt'
    with open(file_name, 'a') as file_obj:
        file_obj.write('%-30s' % str(h5_name))
        file_obj.write('DICE: %.5f' % (dice_PGDC / patient) + '     Recall: %.5f'%(recall_PGDC / patient)+'     precision: %.5f'%(precision_PGDC / patient)+'     jac: ' + str(
            jac_mean) + '+'+str(jac_standard) + '\n\n')
    print('DICE: ' + str(dice_PGDC / patient))

def jac_from_output(output):
    a = []
    for i in output:
        jac = jacobian_determinant(i)
        jac_negative = jac[jac < 0]
        a.append(jac_negative.shape[0])
    return a

def generator_train(src, tgt, batch_size=32):
    gen_src = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=0.1,
                                   featurewise_center=True, featurewise_std_normalization=True,
                                   horizontal_flip=True, fill_mode='constant')
    gen_tgt = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=5, zoom_range=0.1,
                                   featurewise_center=True, featurewise_std_normalization=True,
                                   horizontal_flip=True, fill_mode='constant')

    train_src = gen_src.flow(src, batch_size=batch_size, shuffle=True, seed=1)
    train_tgt = gen_tgt.flow(tgt, batch_size=batch_size, shuffle=True, seed=1)


    while True:
        next_src = next(train_src)
        next_tgt = next(train_tgt)
        yield [next_src, next_tgt, next_tgt], [next_tgt, next_tgt, next_tgt, next_tgt]

def generator_val(src, tgt, fl, dl, batch_size=32):
    gen_src2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    gen_tgt2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    fl2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    dl2 = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    train_src = gen_src2.flow(src, batch_size=batch_size, shuffle=False)
    train_tgt = gen_tgt2.flow(tgt, batch_size=batch_size, shuffle=False)
    fl22 = fl2.flow(fl, batch_size=batch_size, shuffle=False)
    dl22 = dl2.flow(dl, batch_size=batch_size, shuffle=False)

    while True:
        next_src = next(train_src)
        next_tgt = next(train_tgt)
        next_fl = next(fl22)
        next_dl = next(dl22)
        yield [next_src, next_tgt, next_fl], [next_tgt, next_tgt, next_tgt, next_dl]

def detection(output_path, path_h5_save, h5_name, load_weight=''):
    if not os.path.exists(output_path+h5_name+'/'):
        os.makedirs(output_path + 'gt_moving/')
        os.makedirs(output_path + 'result_moving/')
        os.makedirs(output_path + 'DWI/')
        os.makedirs(output_path + 'Flair/')
        os.makedirs(output_path + 'Field/')
    if not os.path.exists('./reimplement/' + h5_name+'/'):
        os.makedirs('./reimplement/' + h5_name+'/')

    if load_weight != '':
        model.load_weights(load_weight, by_name=True)
    else:
        load_weight = os.listdir(output_path+'model/')
        load_weight.sort()
        load_weight = load_weight[len(load_weight) - 1]
        print('loading：', load_weight)
        model.load_weights(output_path +'model/'+ load_weight, by_name=True)

    h5 = h5py.File(path_h5_save + 'test')
    val_F = h5['F']
    val_DWI = h5['DWI']
    label_flair = h5['label_flair']
    label_dwi = h5['label_dwi']
    print('load data done!')
    strat = time.time()
    predict = model.predict(x=[val_F, val_DWI, label_flair], verbose=1)
    print('time==', str((time.time()-strat)/len(val_F)))
    cal_patient(gt=label_dwi, predict=predict[3], flow=predict[0])

    field_visualization(predict[0], path=output_path+'Field/')
    for i in range(len(predict[0])):
        if label_flair[i].max() > 0:
            cv2.imwrite(output_path + 'result_moving/' + str(i) + '.png', predict[1][i, :, :, 0] * 255)
            cv2.imwrite(output_path + 'gt_moving/'+str(i)+'.png', predict[3][i, :, :, 0] * 255)
            cv2.imwrite(output_path + 'DWI/' + str(i) + '.png', val_DWI[i, :, :, 0] * 255)
            cv2.imwrite(output_path + 'Flair/'+str(i)+'.png', val_F[i, :, :, 0] * 255)
            cv2.imwrite('./reimplement/' + h5_name + '/' + str(i) + '.png', predict[3][i, :, :, 0] * 255)

def field_visualization(field, path=None):
    tem = np.zeros((224, 224, 3), dtype='float')
    for num in range(len(field)):
        x = np.abs(copy.deepcopy(field[num, :, :, 0]))
        y = np.abs(copy.deepcopy(field[num, :, :, 1]))
        tem[:, :, 0] = 0
        tem[:, :, 1] = y/5
        tem[:, :, 2] = x/5
        if path is not None:
            cv2.imwrite(path + str(num) + '.png', tem * 255)

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path_h5_save = './ori_data/deal/'
    load_weight = './model/affine/epoch003-0.76.h5' #should load the affine weight
    mode = 'train'

    batch_size = 32
    lr = 1e-2
    w = 32
    model = dual_net(w)
    para_mi = 4    #'Alpha' mentioned in the paper
    para_jl = 100  #'Beta'
    para_rl = 100  #'Lambda_4'

    output_path = './model/'
    h5_name = 'dual_net'
    print(h5_name)
    subdir_savepic = h5_name
    output_path += h5_name + '/'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    os.makedirs(output_path + 'model/')
    model.compile(optimizer=Adam(lr=lr),
                  loss=[design_loss().smooth, design_loss(parameter_mi=para_mi, parameter=para_jl).mi_clipmse, residuce_loss, none_loss],
                  loss_weights=[1, 1, para_rl, 0], metrics={'stn4': dice_coef})

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

    train_data = generator_train(train_F, train_DWI, batch_size=batch_size)
    val_data = generator_val(val_F, val_DWI, label_flair, label_dwi, batch_size=batch_size)

    earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    tensorboard = TensorBoard(log_dir='./tensorboard/' + h5_name + '/', batch_size=batch_size)

    checkpointer = ModelCheckpoint(output_path + 'model/epoch{epoch:03d}-{val_stn4_dice_coef:.2f}.h5',
                                   monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    model.fit_generator(train_data, epochs=100, initial_epoch=0, steps_per_epoch=num_train_steps, shuffle=True,
                        callbacks=[checkpointer, tensorboard, earlystop], validation_data=val_data,
                        validation_steps=num_val_steps, verbose=2)
    detection(output_path=output_path, path_h5_save=path_h5_save, h5_name=h5_name)
