import argparse
import h5py
import numpy as np
import os, sys
import time
import matplotlib.pyplot as plt
import scipy.io as scio
import cv2
import pickle as pkl
import scipy.io as scio
import tensorflow as tf
import gym
import warnings
from CustomizeUtils import CommonUtils as ComUtils

np.set_printoptions(threshold=sys.maxsize)
cur_path = os.getcwd()


def get_args_for_FC_dynamics():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CarSim_v5'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=3, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    # network = [input_dim + u_dim, 32, 64, 128, 128, 128, 128, 64, 32, s_dim]
    if not is_random_train:
        network = [input_dim + u_dim, 32, 64, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', '']
    else:
        network = [input_dim + u_dim, 32, 64, 128, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', '']
    # act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    if (not len(act_type) == len(network) - 1):
        raise Exception('the length of eact_type is wrong')
    parser.add_argument('--act_type', type=str, default=act_type, help='the type of activation')
    parser.add_argument('--network', type=list, default=network, help='the NN construction')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v5_FC/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v5/' + 'data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_04_04_17_54_h11/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_args_FC_local_v():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    if version == '6':
        state_bound = np.array([[-0.2, -2.7, -1.2], [27.3, 1.9, 1.1]])
        action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 9.1]])  # 方向、油门、刹车
    elif version == '7':
        state_bound = np.array([[-0.1, -1.1, -0.8], [24.1, 1.1, 0.8]])
        action_bound = np.array([[-7.0, 0.0, 0.0], [6.9, 0.2, 9.0]])  # 方向、油门、刹车
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    # network = [input_dim + u_dim, 32, 64, 128, 128, 128, 128, 64, 32, s_dim]
    # network = [input_dim + u_dim, 16, 32, 32, 16, s_dim]
    if not is_random_train:
        network = [input_dim + u_dim, 32, 64, 128, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', '']
    else:
        network = [input_dim + u_dim, 32, 64, 128, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', '']
    # act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    if (not len(act_type) == len(network) - 1):
        raise Exception('the length of eact_type is wrong')
    parser.add_argument('--act_type', type=str, default=act_type, help='the type of activation')
    parser.add_argument('--network', type=list, default=network, help='the NN construction')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 68  # 130
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    predicted_step = 1005
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=5., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v' + version + '_FC/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_local_v/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v' + version + '/data_local_v/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_local_v/' + 'carsim_2020_06_15_11_14_FC_h21/model_37min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_args_FC_v6_0_1():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = True
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    state_bound = np.array([[-28.4, -26.4, -1.1], [26.2, 24.6, 1.1]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 10.7]])  # 方向、油门、刹车
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    # network = [input_dim + u_dim, 32, 64, 128, 128, 128, 128, 64, 32, s_dim]
    if not is_random_train:
        network = [input_dim + u_dim, 32, 64, 128, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', '']
    else:
        network = [input_dim + u_dim, 32, 64, 128, 128, 64, 32, s_dim]
        act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', '']
    # act_type = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    if (not len(act_type) == len(network) - 1):
        raise Exception('the length of eact_type is wrong')
    parser.add_argument('--act_type', type=str, default=act_type, help='the type of activation')
    parser.add_argument('--network', type=list, default=network, help='the NN construction')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 88  # 130
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=5., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v' + version + '_FC/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_0_1/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--path', type=str, default=path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v' + version + '/data_0_1/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_0_1/' + 'carsim_2020_04_04_17_54_h11/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_con_mountaincar_args():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'MountainCarContinuous-v0'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 80, 80
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 2, 1
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['x', 'x_dot']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['force']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # min_position = -1.2
    # max_position = 0.6
    # max_speed = 0.07
    # goal_position = 0.45
    state_bound = np.array([[-1.2, -0.07], [0.6, 0.07]])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-1], [1]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    conca_num = 2
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=10,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=78, help='the num of train dataset file')
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    if input_mode == 0:
        conv_size = [[8, 2, conca_num + 1, 32], [4, 2, 32, 64], [3, 2, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 2], [2, 2], [2, 2]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [num_neural, lift_dim]
        eact_type = ['']
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        eact_type = ['relu', 'relu', 'relu', 'relu']
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3 = 16, 32, 64
    decoder_widths = [lift_dim, h3, h2, h1, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['tn', 'tn', 'tn', 'tn']
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 30  # 130
    num_koopman_shifts = 1
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    # 各个状态计算损失的权重，根据训练情况来设定
    parser.add_argument('--state_weight', type=list, default=np.array([1] * s_dim),
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    predicted_step = 100
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=0.3, help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.95, help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/conmountaincar/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'Mountain_%d_' % input_mode + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = '/home/BackUp/data/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'ConMountain_450_traj_%d_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'ConMountain_450_traj_7_9.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'ConMountain_450_traj_7_8.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'Mountain_2020_01_13_16_07/model_9min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_acrobot_args(input_mode=None):
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'Acrobot-v1'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 80, 80
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 6, 1
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['cos(th1)', 'sin(th1)', 'cos(th2)', 'sin(th2)', 'th1_dot', 'th2_dot']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['torque']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # min_position = -1.2
    # max_position = 0.6
    # max_speed = 0.07
    # goal_position = 0.45
    state_bound = [1, 1, 1, 1, 4 * np.pi, 9 * np.pi]
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = [-1, 1]
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=5,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=91, help='the num of train dataset file')
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    if input_mode is not None:
        input_mode = input_mode
    else:
        input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    if input_mode == 0:
        conv_size = [[8, 2, conca_num + 1, 32], [4, 2, 32, 64], [3, 2, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 2], [2, 2], [2, 2]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [num_neural, lift_dim]
        eact_type = ['']
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        eact_type = ['relu', 'relu', 'relu', '']
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3 = 16, 32, 64
    decoder_widths = [lift_dim, h3, h2, h1, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl']
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    dact_type = ['relu', 'relu', 'relu', 'tanh']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=True, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 30  # 130
    num_koopman_shifts = 1
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    # 各个状态计算损失的权重，根据训练情况来设定
    parser.add_argument('--state_weight', type=list, default=np.array([1] * s_dim),
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    predicted_step = 100
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=0.01, help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=.5, help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.97, help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/acrobot/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    date_day = time.strftime("%Y_%m_%d", time.localtime())
    path = base_path + 'model/' + date_day + '/Acrobot_%d_' % (input_mode) + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = '/media/buduo/Data/Acrobot_Data/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'Acrobot_500_250_251_traj_%d_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'Acrobot_500_250_251_traj_9_3.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'Acrobot_500_250_251_traj_9_2.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + date_day + '/Acrobot_0_2020_01_15_12_37/model_30min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_cartpole_args():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CartPole-v0'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 40, 240
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 4, 1
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['x', 'x_dot', 'theta', 'theta_dot']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['force']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # min_position = -1.2
    # max_position = 0.6
    # max_speed = 0.07
    # goal_position = 0.45
    state_bound = [2.0, 4, 0.44, 3]
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = [10.]
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    conca_num = 2
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=5,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=249, help='the num of train dataset file')
    lift_dim = 128
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 0
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
        eact_type = ['']
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim]
        eact_type = ['relu', 'relu', 'relu']
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3 = 16, 32, 64
    decoder_widths = [lift_dim, h3, h2, h1, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['tn', 'tn', 'tn', 'tn', 'tn']
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    dact_type = ['relu', 'relu', 'relu', 'tanh']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 50  # 130
    num_koopman_shifts = 1
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    # 各个状态计算损失的权重，根据训练情况来设定
    parser.add_argument('--state_weight', type=list, default=np.array([1] * s_dim),
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    predicted_step = 80
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=0.3, help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/cartpole/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'Cartpole_%d_' % input_mode + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = '/home/BackUp/cartpole_data/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'CartPole_1000_55_traj_%d_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'CartPole_80_traj_31.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'CartPole_1000_55_traj_24_9.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'Cartpole_2020_01_08_14_24/model_76min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_cartpolev2_args():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CartPole-v2'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 40, 240
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 4, 1
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['x', 'x_dot', 'theta', 'theta_dot']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['force']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # min_position = -1.2
    # max_position = 0.6
    # max_speed = 0.07
    # goal_position = 0.45
    state_bound = [2., 2.4, 0.44, 3]
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = [10.]
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=5, help='the num of train dataset file')
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
        eact_type = ['']
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        eact_type = ['relu', 'relu', 'relu', 'relu']
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3, h4 = 16, 32, 64, 128
    decoder_widths = [lift_dim, h4, h3, h2, h1, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['tn', 'tn', 'tn', 'tn', 'tn']
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    dact_type = ['relu', 'relu', 'tanh', 'relu', 'tanh']
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 130  # 130
    num_koopman_shifts = 1
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = [1, 1, 1, 1]
    parser.add_argument('--state_weight', type=list, default=np.array(state_weight),
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    predicted_step = 200
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=0.03, help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/cartpole_v2/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'Cartpole_%d_' % input_mode + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.h5')
    # path of the training, testing and validation data
    data_path = '/home/buduo/myPythonDemo/myKoopman/DEDMDP/cartpole_v2/data/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'CartPole_1000_301_traj_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'CartPole_1000_301_traj_5.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'CartPole_1000_301_traj_6.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'Cartpole_1_2020_01_16_18_05/model_118min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v5_args():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CarSim_v5'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 4
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=10, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    # lift_dim = 125
    lift_dim = 8
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        # encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim, lift_dim]
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 256, lift_dim]
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    # decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    # decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    decoder_widths = [lift_dim + s_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    eact_type = ['relu', 'relu', 'relu', 'tanh']
    # dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    dact_type = ['']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v5/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = base_path + 'data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_04_20_23_12_h2/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v6_args_minus_1_1():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 4
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    state_bound = np.array([[-28.4, -26.4, -1.1], [26.2, 24.6, 1.1]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 10.7]])  # 方向、油门、刹车
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    # lift_dim = 125
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        # encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim, lift_dim]
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 256, lift_dim]
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    # decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    eact_type = ['relu', 'relu', 'relu', 'tanh']
    dact_type = ['relu', 'relu', 'relu', 'tanh']
    # dact_type = ['']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v' + version + '/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_minus_1_1/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = base_path + 'data_minus_1_1/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_minus_1_1/' + 'carsim_2020_06_07_22_13/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v6_args_0_1():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 4
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    state_bound = np.array([[-28.4, -26.4, -1.1], [26.2, 24.6, 1.1]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 10.7]])  # 方向、油门、刹车
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    # lift_dim = 125
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        # encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim, lift_dim]
        encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 256, lift_dim]
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    # decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    eact_type = ['relu', 'relu', 'relu', 'tanh']
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    # dact_type = ['']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v' + version + '/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_0_1/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = base_path + 'data_0_1/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_0_1/' + 'carsim_2020_06_07_22_13/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_args_local_v():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    if version == '6':
        state_bound = np.array([[-0.2, -2.7, -1.2], [27.3, 1.9, 1.1]])
        action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 9.1]])  # 方向、油门、刹车
    elif version == '7':
        state_bound = np.array([[-0.1, -1.1, -0.8], [24.1, 1.1, 0.8]])
        action_bound = np.array([[-7.0, 0.0, 0.0], [6.9, 0.2, 9.0]])  # 方向、油门、刹车
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    # lift_dim = 125
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 256, lift_dim]
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    # decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    eact_type = ['relu', 'relu', 'relu', 'tanh']
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    # dact_type = ['']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 88  # 130
    num_koopman_shifts = 86
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 50
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = ComUtils.get_upLevel_dir(os.getcwd())
    base_path = cur_path + '/carsim_v' + version + '/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_local_v/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = base_path + 'data_local_v/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_local_v/' + 'carsim_2020_06_11_19_09_l11/model_296min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_args_local_v_attention():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=36, help='the num of train dataset file')
    parser.add_argument('--test_file_num', type=int, default=2, help='the num of train dataset file')
    parser.add_argument('--val_file_num', type=int, default=2, help='the num of train dataset file')
    if version == '6':
        state_bound = np.array([[-0.2, -2.7, -1.2], [27.3, 1.9, 1.1]])
        action_bound = np.array([[-7.9, 0., 0.], [7.9, 0.2, 9.1]])  # 方向、油门、刹车
    elif version == '7':
        state_bound = np.array([[-0.1, -1.1, -0.8], [24.1, 1.1, 0.8]])
        action_bound = np.array([[-7.0, 0.0, 0.0], [6.9, 0.2, 9.0]])  # 方向、油门、刹车
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    # lift_dim = 125
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        encoder_widths = [(conca_num + 1) * s_dim, 32, 64, lift_dim, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 64, lift_dim]
        # encoder_widths = [(conca_num + 1) * s_dim, 16, 32, 256, lift_dim]
    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    # decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    eact_type = ['relu', 'relu', 'relu', 'tanh']
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    # dact_type = ['']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 88  # 130
    num_koopman_shifts = 86
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 50
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.path.split(os.getcwd())[0]
    base_path = cur_path + '/carsim_v' + version + '/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model_local_v_attention/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = base_path + 'data_local_v/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_v' + version + '_test_%d.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_v' + version + '_val_%d.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model_local_v/' + 'carsim_2020_06_11_19_09_l11/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v5_args_for_no_encoder():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    version = '6'
    domain_name = 'CarSim_v' + version
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=1,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=10, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 125
    # lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    parser.add_argument('--input_dim', type=int, default=(conca_num + 1) * s_dim, help='the dim of input')
    decoder_widths = [lift_dim + s_dim, 128, 64, 32, s_dim]
    # decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    if not len(dact_type) == len(decoder_widths) - 1:
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = False
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v' + version + '_no_encoder/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')
    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v' + version + '/data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_05_15_09_33/model_202min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    if is_predict:
        args_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/args.pkl'
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        matlab_file_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/params_for_matlab.mat'
    else:
        args_file_path = path + '/args.pkl'
        test_image_path = path + '/test/'
        matlab_file_path = path + '/params_for_matlab.mat'
    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    parser.add_argument('--args_file_path', type=str, default=args_file_path, help='save dict args for restore')
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                        help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v5_args_for_D_EDMD():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CarSim_v5'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    is_random_train = False
    parser.add_argument('--is_random_train', type=bool, default=is_random_train,
                        help='whether to train with a random layer')
    if is_random_train:
        random_weight_layer = 3
    else:
        random_weight_layer = 100
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=2,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=10, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 16
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--input_dim', type=int, default=input_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        # width = [input_dim, 16, 32, 32]
        width = [input_dim, 16, 32]
        # width = []
        encoder_widths = [32, 16, 1]
        # encoder_widths = [3, 16, 32, 16, 1]

    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--width', default=width, help='the Feature extraction layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3, h4, h5 = 32, 64, 128, 64, 32
    decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, 32, 32, s_dim]
    # decoder_widths = [lift_dim, 64, 64, 32, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    # temp_type = ['relu', 'relu', 'tanh']
    temp_type = ['relu', 'tanh']
    # temp_type = []
    eact_type = ['relu', 'tanh']
    # eact_type = ['relu', 'relu', 'relu', 'relu']
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    if len(width) != 0 and (not len(temp_type) == len(width) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--temp_type', type=str, default=temp_type,
                        help='the type of activation of the feature extraction layers')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v5_DEDMD/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v5/data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_04_14_15_46/model_403min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    matlab_file_path = base_path + 'model/' + 'carsim_2020_04_14_15_46/params_for_matlab.mat'
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v5_args_for_D_EDMD2():  # for DEDMDP6.py
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CarSim_v5'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=2,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=10, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 10
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--input_dim', type=int, default=input_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        width = [input_dim, 16, 32, 16]
        # width = []
        encoder_widths = [8, 1]
        # encoder_widths = [3, 16, 32, 16, 1]

    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--width', default=width, help='the Feature extraction layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    h1, h2, h3, h4, h5 = 32, 64, 128, 64, 32
    # decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    decoder_widths = [lift_dim + s_dim, 32, 32, s_dim]
    # decoder_widths = [lift_dim, 64, 64, 32, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    temp_type = ['relu', 'relu', 'tanh']
    # temp_type = []
    eact_type = ['tanh']
    # eact_type = ['relu', 'relu', 'relu', 'relu']
    dact_type = ['relu', 'relu', 'sigmoid']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--temp_type', type=str, default=temp_type,
                        help='the type of activation of the feature extraction layers')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 41  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v5_DEDMD/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v5/data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_04_18_16_20/model_min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    matlab_file_path = base_path + 'model/' + 'carsim_2020_04_18_16_20/params_for_matlab.mat'
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def get_carsim_v5_args_for_ELM_EDMD():
    parser = argparse.ArgumentParser()
    # ------------------------------ environment --------------------
    domain_name = 'CarSim_v5'
    parser.add_argument('--domain_name', type=str, default=domain_name, help='the name of the system')
    image_height, image_width = 1, 1
    parser.add_argument('--image_width', type=int, default=image_width, help='the width of training image')
    parser.add_argument('--image_height', type=int, default=image_height, help='the height of training image')
    parser.add_argument('--is_gym', type=bool, default=True, help='flag of gym env')
    s_dim, u_dim = 3, 2
    parser.add_argument('--s_dim', type=int, default=s_dim, help='the dim of the state of the forced dynamics')
    parser.add_argument('--u_dim', type=int, default=u_dim, help='the dim of the action of the forced dynamics')
    state_name = ['vx', 'vy', 'Yaw_rate']
    parser.add_argument('--state_name', type=list, default=state_name, help='the name of each state')
    action_name = ['steer_SW', 'throttle']
    parser.add_argument('--action_name', type=list, default=action_name, help='the name of each action')
    # -------------- for data process -------------------------------------------
    random_weight_layer = 2
    parser.add_argument('--random_weight_layer', type=int, default=random_weight_layer, help='the random weight layer')
    conca_num = 0
    parser.add_argument('--conca_num', type=int, default=conca_num, help='concatenate conca_num states to one input')
    parser.add_argument('--interval', type=int, default=2,
                        help='sample intervals when sampling data from train dataset')
    parser.add_argument('--train_file_num', type=int, default=10, help='the num of train dataset file')
    state_bound = np.array([[-0.2, -5.6, -0.9], [83.1, 5.6, 0.9]])
    state_bound = np.reshape(np.tile(state_bound, (1, conca_num + 1)), [2, -1])
    parser.add_argument('--state_bound', type=list, default=state_bound, help='the bound of each states')
    action_bound = np.array([[-8.8, -0.2], [8.8, 0.2]])
    parser.add_argument('--action_bound', type=list, default=action_bound, help='the bound of each action')
    # ---------------------- construction ------------------------
    # 编码和解码部分的隐藏层神经元个数
    lift_dim = 16
    # # conv 和 pool 的padding 方式默认都采用 'SAME'
    # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    # conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
    # mode --- 训练模式标志位，当mode=0时，表示用图像作为输入， 当mode = 1时，使用状态作为输入
    input_mode = 1
    parser.add_argument('--input_mode', type=int, default=input_mode,
                        help='the mode decide whether to use image as input')
    input_dim = (conca_num + 1) * s_dim
    parser.add_argument('--input_dim', type=int, default=input_dim, help='the dim of input')
    if input_mode == 0:
        conv_size = [[4, 1, conca_num + 1, 32], [4, 1, 32, 64], [3, 1, 64, 64]]
        # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
        pool_size = [[2, 1], [2, 1], [2, 1]]
        # # encoder 全连接层
        # 计算经过conv和pool后的图片尺寸
        # _, num_neural = cal_size([image_height, image_width, conca_num + 1], conv_size, pool_size)
        encoder_widths = [256, lift_dim]
    elif input_mode == 1:
        conv_size = []
        pool_size = []
        width = [input_dim, 16, 32]
        # width = []
        encoder_widths = [32, 16, 1]
        # encoder_widths = [3, 16, 32, 16, 1]

    parser.add_argument('--conv', type=list, default=conv_size, help='the conv2d layers info')
    parser.add_argument('--pool', type=list, default=pool_size, help='the pooling layers info')
    parser.add_argument('--width', default=width, help='the Feature extraction layers info')
    parser.add_argument('--encoder_widths', default=encoder_widths, help='the fc layers info')
    decoder_widths = [lift_dim + s_dim, 64, 32, 16, s_dim]
    # decoder_widths = [lift_dim + s_dim, 64, 32, s_dim]
    # decoder_widths = [lift_dim, 64, 64, 32, s_dim]
    parser.add_argument('--decoder_widths', type=list, default=decoder_widths,
                        help='the construction of the auto-encoder')
    parser.add_argument('--lift_dim', type=int, default=lift_dim, help='the lifted dim')
    parser.add_argument('--scale', type=float, default=0.1, help='scale for initializing weights')
    e_weight_dist = ['dl', 'dl', 'dl', 'dl', 'dl', 'dl']  # shape = len(encoder_widths) - 1
    parser.add_argument('--edist_weights', type=list, default=e_weight_dist, help='random type of weights')
    parser.add_argument('--edist_biases', type=list, default=[''] * (len(conv_size) + len(encoder_widths) - 1),
                        help='random type of biases ')
    d_weight_dist = 'dl'
    parser.add_argument('--ddist_weights', type=list, default=[d_weight_dist] * (len(decoder_widths) - 1),
                        help='random type of weights')
    parser.add_argument('--ddist_biases', type=list, default=[''] * (len(decoder_widths) - 2),
                        help='random type of biases ')
    temp_type = ['relu', 'tanh']
    # temp_type = []
    eact_type = ['relu', 'tanh']
    # eact_type = ['relu', 'relu', 'relu', 'relu']
    dact_type = ['relu', 'relu', 'relu', 'sigmoid']
    if (not len(eact_type) == len(encoder_widths) - 1) or (not len(dact_type) == len(decoder_widths) - 1):
        raise Exception('the length of eact_type or dact_type is wrong')
    parser.add_argument('--eact_type', type=str, default=eact_type, help='the type of activation')
    parser.add_argument('--dact_type', type=str, default=dact_type, help='the type of activation')
    parser.add_argument('--temp_type', type=str, default=temp_type,
                        help='the type of activation of the feature extraction layers')
    parser.add_argument('--batch_flag', type=bool, default=False, help='the flag of batch normalization')
    # 动作部分的网络结构
    uwidths = [u_dim, lift_dim]
    parser.add_argument('--uwidths', type=list, default=uwidths, help='the construction of the auto-encoder')
    uweight_dist = 'dl'
    parser.add_argument('--uscale', type=float, default=0.1, help='scale for initializing weights')
    parser.add_argument('--udist_weights', type=list, default=[uweight_dist] * len(uwidths),
                        help='random type of weights')
    parser.add_argument('--udist_biases', type=list, default=[''] * len(uwidths), help='random type of biases ')
    parser.add_argument('--uact_type', type=str, default='relu', help='the type of activation')
    parser.add_argument('--ubatch_flag', type=bool, default=False, help='the flag of batch normalization')
    # # ------------------------ data attribute ---------------------
    # file_num，step_num, traj_num, 分别表示训练文件的数量，每个训练文件数据的trajectory数量，每条trajectory的数据长度
    # ------------------------ loss function ------------
    is_predict = True
    parser.add_argument('--is_predict', type=bool, default=is_predict, help='the flag of running model')
    num_shifts = 40  # 130
    num_koopman_shifts = 40
    parser.add_argument('--num_shifts', type=int, default=num_shifts,
                        help='the steps for calculating the predicted loss')
    parser.add_argument('--num_koopman_shifts', type=int, default=num_koopman_shifts,
                        help='the steps for koopman operator')
    predicted_step = 400
    # 各个状态计算损失的权重，根据训练情况来设定
    state_weight = np.array([1, 1, 1])
    parser.add_argument('--state_weight', type=list, default=state_weight,
                        help='the weight of each state to calculate reconstruction loss and multi-steps prediction loss')
    parser.add_argument('--predicted_step', type=int, default=predicted_step, help='the num of the predicted step')
    parser.add_argument('--koopman_lam', type=float, default=1., help='the coefficient of koopman linear loss')
    parser.add_argument('--L2_lam', type=float, default=10 ** (-9), help='the coefficient of L2 regulation')
    parser.add_argument('--recon_lam', type=float, default=1., help='the coefficient of reconstruction loss')
    parser.add_argument('--L1_lam', type=float, default=0.0, help='the coefficient of L1 regulation')
    parser.add_argument('--Linf_lam', type=float, default=10 ** (-9), help='the coefficient of infinite norm')
    parser.add_argument('--opt_alg', type=str, default='adam', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='the size of each batch')
    parser.add_argument('--learning_rate', type=float, default=10 ** (-4), help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0., help='decay rate of learning rate')
    parser.add_argument('--dropout_rate', type=float, default=1., help='keep prob')
    # -------------- path ---------------------
    # date is the top folder of the current experiment. the model,pictures and params are the data during training
    cur_path = os.getcwd()
    base_path = cur_path + '/carsim_v5_ELM/'
    date = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    path = base_path + 'model/' + 'carsim_' + str(date)
    if (not os.path.exists(path)) and (not is_predict):
        os.makedirs(path)
    parser.add_argument('--model_path', type=str, default=path + '/model_%dmin.ckpt',
                        help='the path for saving trained model')
    parser.add_argument('--image_count', type=int, default=1, help='the sequence start for saving images')
    parser.add_argument('--image_path', type=str,
                        default=path + '/result_%smin_' + str(predicted_step) + 'steps_%s.png',
                        help='the path for saving the predicted picture result')

    # test_image_folder = base_path + 'model/test/'
    # if not os.path.exists(test_image_folder):
    #     os.makedirs(test_image_folder)
    parser.add_argument('--params_path', type=str, default=path + '/params.txt')
    parser.add_argument('--loss_path', type=str, default=path + '/train_val_losses.mat')
    # path of the training, testing and validation data
    data_path = cur_path + '/carsim_v5/data/data_without_noise/'
    parser.add_argument('--train_data_path', type=str,
                        default=data_path + 'carsim_train_9001_50_%d.pkl', help='the path of the training data')
    parser.add_argument('--test_data_path', type=str,
                        default=data_path + 'carsim_test_9001_50_1.pkl', help='the path of testing data')
    parser.add_argument('--val_data_path', type=str, default=data_path + 'carsim_val_9001_50_1.pkl',
                        help='the path of the validation data')
    restore_model_path = base_path + 'model/' + 'carsim_2020_04_18_22_19/model_16min.ckpt'
    parser.add_argument('--restore_model_path', type=str, default=restore_model_path, help='the path of trained model')
    matlab_file_path = base_path + 'model/' + 'carsim_2020_04_18_22_19/params_for_matlab.mat'
    parser.add_argument('--matlab_file_path', type=str, default=matlab_file_path, help='for matlab prediction')
    # 如果是预测模式，且预测模型路径存在，则创建用于存储预测图片的文件夹
    if is_predict and os.path.exists(''.join(os.path.split(restore_model_path)[:-1])):
        test_image_path = ''.join(os.path.split(restore_model_path)[:-1]) + '/test/'
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        parser.add_argument('--test_image_path', type=str, default=test_image_path + str(date) + '_%s.png',
                            help='the path for saving the testing images')
    args = parser.parse_args()
    return args


def plot_X_and_Y(args, X_true, X_koop, U, state_name, action_name, title=None, index=None,
                 is_save=False, save_dir=None, is_show=False, is_reference=False, predict_step=100):
    '''
    如果index＝None，则plot出X和Y的各个维度的数据，如果index!=0，则plot出index指定的X、Y的维度
    :param X_true: the real states data
    :param X_koop: the approximate data with Koopman
    :param U: the control
    :param index: List, denotes the indexes of X and Y needed to plot
    :return: no return
    '''
    X_input = X_true
    X_true = X_true[-args['s_dim']:, :]
    # diff_level = [[0.02, 0.05], [0.05, 0.1], [0.02, 0.05], [0.05, 0.1]]
    diff = np.array([[100, 100], [100, 100], [100, 100], [100, 100]], dtype=np.float64)
    predict_step = predict_step
    if index is None:
        index = np.arange(X_true.shape[0])
    # 用来绘制参考线
    squence = X_true.shape[1]
    if np.ndim(U) == 1:
        u_dim = 1
    else:
        u_dim = np.min(U.shape)
    figure_num = len(index) + u_dim
    col_num = 2
    row_num = int(np.ceil(figure_num / col_num))
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.2, hspace=1.2)
    if title is None:
        fig.suptitle("the result of Testing")
    else:
        fig.suptitle(title)
    # plot the states
    for i in range(len(index)):
        ax = plt.subplot(row_num, col_num, i + 1)
        ax.plot(X_true[index[i], :], 'c', label='True')
        ax.plot(X_koop[index[i], :], 'r', label='DEDMD')
        # if np.ndim(args.state_bound) == 1:
        #     low_state = -np.array(args.state_bound)
        #     high_state = np.array(args.state_bound)
        #     # low_u = -np.array(args.action_bound)
        #     # high_u = np.array(args.action_bound)
        #     ax.axis(ymin=low_state[i], ymax=high_state[i])
        if (i == 0) and is_reference is True:
            # 画角度的参考线
            min = np.min([np.min(X_true[i, :]), np.min(X_koop[i, :])])
            max = np.max([np.max(X_true[i, :]), np.max(X_koop[i, :])])
            x1 = np.array([predict_step - 21, predict_step - 21])
            y1 = np.array([min, max])
            ax.plot(x1, y1, color='gray', linestyle='--')
            true_end = X_true[i, -1]
            koopman_end = X_koop[i, -1]
            x = np.array([0, squence])
            y = np.array([X_true[i, predict_step - 21], X_true[i, predict_step - 21]])
            ax.plot(x, y, color='gray', linestyle='--')
            if true_end >= koopman_end:
                ax.plot(x, y - 0.02, color='green', linestyle='--')
                ax.plot(x, y - 0.05, color='red', linestyle='--')
            else:
                ax.plot(x, y + 0.02, color='green', linestyle='--')
                ax.plot(x, y + 0.05, color='red', linestyle='--')
            diff[i, 0] = abs(X_true[i, predict_step - 21] - X_koop[i, predict_step - 21])
            diff[i, 1] = abs(X_true[i, predict_step - 1] - X_koop[i, predict_step - 1])
            ax.set_title('state_%d: %s, offset_%d: %.3f, offset_%d: %.3f' % (i, state_name[i], predict_step - 20,
                                                                             diff[i, 0], predict_step, diff[i, 1]))
        else:
            ax.set_title('state: %s' % (state_name[i]))
        ax.legend()
        # plot the action
    for i in range(u_dim):
        ax = plt.subplot(row_num, col_num, len(index) + i + 1)
        # print(U.shape)
        ax.plot(np.squeeze(U[i, :]))
        ax.set_title("action_%d: %s" % (i, action_name[i]))
    if is_save is True:
        if save_dir is None:
            raise Exception("you should give the saving path")
        plt.savefig(save_dir)
    if args['is_predict'] is True:
        if save_dir is None:
            raise Exception("you should give the saving path")
        save_dict_as_mat({'true': X_input, 'predict': X_koop, 'action': U}, save_dir.replace('png', 'mat'))
    if is_show is True:
        plt.show()
    plt.close()
    if is_reference is True:
        return np.array(diff[0, :][:, np.newaxis])


def save_dict_as_txt(dict, save_dir):
    with open(save_dir, 'w') as fw:
        for key in dict.keys():
            fw.writelines(key + ': ' + str(dict.get(key)) + '\n')
        fw.close()


def check_progress(start, best_error, params, batch_size):
    """Check on the progress of the network training and decide if it's time to stop.

    Arguments:
        start -- time that experiment started
        best_error -- best error so far in training
        params -- dictionary of parameters for experiment

    Returns:
        finished -- 0 if should continue training, 1 if should stop training
        save_now -- 0 if don't need to save results, 1 if should save results

    Side effects:
        May update params dict: stop_condition, been5min, been20min, been40min, been1hr, been2hr, been3hr, been4hr,
        beenHalf
    """
    finished = 0
    save_now = 0

    current_time = time.time()
    if not params['been3min']:
        if current_time - start > 3 * 60:
            params['been3min'] = best_error
            params['samples3min'] = params['train_times'] * batch_size
            print('------------------ 3 minutes -----------------')

    if not params['been5min']:
        # only check 5 min progress once
        if current_time - start > 5 * 60:
            params['been5min'] = best_error
            params['samples5min'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 5 minutes -----------------')
    if not params['been20min']:
        # only check 20 min progress once
        if current_time - start > 20 * 60:
            params['been20min'] = best_error
            params['samples20min'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 20 minutes -----------------')
    if not params['been40min']:
        # only check 40 min progress once
        if current_time - start > 40 * 60:
            params['been40min'] = best_error
            params['samples40min'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 40 minutes -----------------')
    if not params['been1hr']:
        # only check 1 hr progress once
        if current_time - start > 60 * 60:
            params['been1hr'] = best_error
            params['samples1hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 1 hour -----------------')
    if not params['been2hr']:
        # only check 2 hr progress once
        if current_time - start > 2 * 60 * 60:
            params['been2hr'] = best_error
            params['samples2hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 2 hour -----------------')
    if not params['been3hr']:
        # only check 3 hr progress once
        if current_time - start > 3 * 60 * 60:
            params['been3hr'] = best_error
            params['samples3hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 3 hour -----------------')
    if not params['been4hr']:
        # only check 4 hr progress once
        if current_time - start > 4 * 60 * 60:
            params['been4hr'] = best_error
            params['samples4hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 4 hour -----------------')
    if not params['been5hr']:
        # only check 4 hr progress once
        if current_time - start > 5 * 60 * 60:
            params['been5hr'] = best_error
            params['samples5hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 5 hour -----------------')
    if not params['been6hr']:
        # only check 4 hr progress once
        if current_time - start > 6 * 60 * 60:
            params['been6hr'] = best_error
            params['samples6hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 6 hour -----------------')
    if not params['been7hr']:
        # only check 4 hr progress once
        if current_time - start > 7 * 60 * 60:
            params['been7hr'] = best_error
            params['samples7hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 7 hour -----------------')
    if not params['been8hr']:
        # only check 4 hr progress once
        if current_time - start > 8 * 60 * 60:
            params['been8hr'] = best_error
            params['samples8hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 8 hour -----------------')
    if not params['been9hr']:
        # only check 4 hr progress once
        if current_time - start > 9 * 60 * 60:
            params['been9hr'] = best_error
            params['samples9hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 9 hour -----------------')
    if not params['been10hr']:
        # only check 4 hr progress once
        if current_time - start > 10 * 60 * 60:
            params['been10hr'] = best_error
            params['samples10hr'] = params['train_times'] * batch_size
            save_now = 1
            print('------------------ 10 hour -----------------')

    if current_time - start > params['max_time']:
        print('------------------ game over -----------------')
        params['samples10hr'] = params['train_times'] * batch_size
        finished = 1
        save_now = 1
    return finished, save_now


def get_gym_image_data(name, image):
    if name == 'CartPole-v1':
        image_width, image_height = 240, 40
        image = image[230:310, :]
    if name == 'MountainCar-v0':
        image_width, image_height = 80, 80
    image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    result = np.reshape(image, (1, image_width * image_height))
    result = result / 255
    return result


def save_dict_as_mat(dict, save_dir):
    scio.savemat(save_dir, dict)


def save_dict_as_h5(dict, savePath):
    '''
    save the dict data as a .h5 file
    :param dict:
    :param savePath: the saving path
    '''
    with h5py.File(savePath, 'w') as f:
        for key in dict.keys():
            f[key] = dict.get(key)
        f.close()
    print('save the datas as h5 successfully!!!')


def read_h5(savePath, keys=None):
    '''
    read h5 file
    savePath: the path of the h5 file
    keys: a list or a numpy string array of keys, such as ['train_x', 'train_y', 'test_x', 'test_y']
    return: return a dictionary format data contains the datas asked by keys
    '''
    if not os.path.exists(savePath):
        raise Exception('The file is not exist!!!')
    dic = {}
    with h5py.File(savePath, 'r') as f:
        for key in f.keys():
            if keys is not None:
                if key in keys:
                    dic[key] = f[key].value
            else:
                dic[key] = f[key].value
        f.close()
    return dic


def save_dict_as_pkl(dict, savePath):
    file = open(savePath, 'wb+')
    pkl.dump(dict, file)
    file.close()
    print('save %s successfully!!!' % savePath)


def read_pkl_as_dict(file_path):
    if not os.path.exists(file_path):
        raise Exception('The .pkl file is not exist!!! ----> %s ' % file_path)
    with open(file_path, 'rb') as f:
        dict = pkl.load(f)
    return dict


def process_data(data, num_shifts, conca, interval=1, mode=0):
    """Stack state data from a 2D array into a 3D array. 与DeepKoopman下的stack_data不同的是，这里处理的数据中，每个episode的长度
        有可能都不一样，即seq_length不同
    Arguments:
        data -- 字典数据，其中包括三个list的数据，他们的key分别是'X_images', 'X_states', 'U',分别表示：
                X_images : list, X_images[i].shape=[j, image_width*image_height],即X_images的每一个数据为一个二维数组，数组每
                            一行表示一张图片数据
                X_states: list，和X_images一样，其每一个数据为一个二维数组,shape=[j, s_dime]，图片对应的状态值
                U: list, 长度比X_images小1, 其每一个数据为一个二维数组，shape=[j, u_dim]，由j状态-->j+1状态的动作值
        num_shifts -- number of shifts (time steps) that losses will use (maximum is len_time - 1)
        len_time -- number of time steps in each trajectory in data
        conca -- 取当前状态和前conca帧状态串联作为神经网络的输入，例如当conca=0时，神经网络的输入维度=s_dim，当conca>0时，神经网络的输入
                 维度为(conca+1)*s_dim，t0和t1两个时刻的状态串联作为t1时刻的状态
        interval -- 间隔interval个shifts采一个样本，例如长度为251的数据，当num_shifts=30,interval=1时，这条数据可以采集成
                 ((len_time - num_shifts) / interval + 1) 条训练数据
        mode -- 表示当前的训练模式，mode=0表示使用图像作为输入；mode=1表示使用状态作为输入，则不需要返回图像数据
    Returns:
        data_tensor -- data reshaped into 3D array, shape=[num_shifts + 1, num_traj * num_data_each_traj, s_dim*conca+1],
                        此时的num_data_each_traj = (len_time - num_shifts -1) / interval + 1
                        如果conca>0,则每条traj数据的前conca将不直接用于生成训练数据，此时用于计算的len_time=len_time-conca，
                        返回的数据shape=num_shifts+1, num_traj * num_data_each_traj, s_dim*conca (即状态数变为conca*s_dim)，
                        此时的num_data_each_traj=(len_time - conca - num_shifts - 1) / interval + 1
    """
    states = data['X_states']
    u = data['U']
    state_dim = states[0].shape[1]
    u_dim = u[0].shape[1]
    # 每一条训练数据的长度：因为shift num_shifts步，数据长度为num_shifts+1
    len_each_train = num_shifts + 1
    num_traj = len(states)
    # 计算总共可以分割成num_train_data条训练数据
    num_train_data = 0
    for i in np.arange(num_traj):
        # 当前num_traj可以分割成num_data_each_traj条训练数据
        len_time = states[i].shape[0]
        num_data_each_traj = int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        if num_data_each_traj > 0:
            num_train_data += num_data_each_traj
    if mode == 0:
        images = data['X_images']
        image_dim = images[0].shape[1]
        image_tensor = np.zeros([len_each_train, num_train_data, image_dim * (conca + 1)])
    state_tensor = np.zeros([len_each_train, num_train_data, state_dim * (conca + 1)])
    u_tensor = np.zeros([num_shifts, num_train_data, u_dim])
    start = 0
    end = 0
    for i in np.arange(num_traj):
        # 取出第i条traj的数据
        len_time = states[i].shape[0]
        num_data_each_traj = int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        if num_data_each_traj <= 0:
            continue
        end = start + int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        if mode == 0:
            image_tensor[:, np.arange(start, end), :] = stack_data(images[i], num_shifts, len_time, conca, interval)
        state_tensor[:, np.arange(start, end), :] = stack_data(states[i], num_shifts, len_time, conca, interval)
        if conca > 0:
            temp_u = u[i][conca:, :]
            u_tensor[:, np.arange(start, end), :] = stack_data(temp_u, num_shifts - 1, len_time - conca - 1, 0,
                                                               interval)
        else:
            u_tensor[:, np.arange(start, end), :] = stack_data(u[i], num_shifts - 1, len_time - 1, 0, interval)
        start += int(np.floor((len_time - conca - len_each_train) / interval)) + 1
    if not mode == 0:
        image_tensor = []
    return image_tensor, state_tensor, u_tensor


def carsim_data_process(data, num_shifts, conca, interval, mode=0):
    ''' 此函数与处理gym数据的process_data()函数几乎一模一样，只是数据有一些小的区别，还是写成两个函数，以求代码更清晰一些
    carsim数据和gym数据是有区别的，carsim数据没有图像输入模式，而又比gym数据多一种模式，就是Delta数据
    :param data: 与process_data有一样的格式，data为字典数据,由carsim_data_preprocess()函数处理而成，
                 包括X = data['X'], U = data['U'], Delta = data['Delta']，X， U， Delta数据结构相同，X数据为list数据，
                 数据的条数traj_num = len(X), 每条数据的step_num = X[0].shape[0] - 1
    :param num_shifts: 计算num_shifts步的预测损失
    :param conca: 取前conca+1个状态一起作为输入
    :param interval: 在同一条数据上，每帧数据之间的采样间隔
    :param mode: mode=1表示网络不输出Delta_Yaw, Delta_X, Delta_Y， mode=0则表示输出这些值
    :return: state_tensor, u_tensor, delta_tensor
    '''
    states, u = data['X'], data['U']
    state_dim, u_dim = states[0].shape[1], u[0].shape[1]
    # 每一条训练数据的长度：因为shift num_shifts步，数据长度为num_shifts+1
    len_each_train = num_shifts + 1
    num_traj = len(states)
    # 计算总共可以分割成num_train_data条训练数据
    num_train_data = 0
    for i in np.arange(num_traj):
        # 当前num_traj可以分割成num_data_each_traj条训练数据
        len_time = states[i].shape[0]
        num_data_each_traj = int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        if num_data_each_traj > 0:
            num_train_data += num_data_each_traj
    # if mode == 0:
    #     delta = data['Delta']
    #     delta_dim = delta[0].shape[1]
    #     delta_tensor = np.zeros([num_shift
    delta = data['Delta']
    delta_dim = delta[0].shape[1]
    delta_tensor = np.zeros([num_shifts, num_train_data, delta_dim])
    state_tensor = np.zeros([len_each_train, num_train_data, state_dim * (conca + 1)])
    u_tensor = np.zeros([num_shifts, num_train_data, u_dim])
    start = 0
    end = 0
    for i in np.arange(num_traj):
        # 取出第i条数据,
        len_time = states[i].shape[0]
        num_data_each_traj = int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        if num_data_each_traj <= 0:
            continue
        end = start + int(np.floor((len_time - conca - len_each_train) / interval)) + 1
        state_tensor[:, np.arange(start, end), :] = stack_data(states[i], num_shifts, len_time, conca, interval)
        if conca > 0:
            temp_u = u[i][conca:, :]
        else:
            temp_u = u[i]
        u_tensor[:, np.arange(start, end), :] = stack_data(temp_u, num_shifts - 1, len_time - conca - 1, 0, interval)
        # if mode == 0:
        if conca > 0:
            temp_delta = delta[i][conca:, :]
        else:
            temp_delta = delta[i]
        delta_tensor[:, np.arange(start, end), :] = stack_data(temp_delta, num_shifts - 1, len_time - conca - 1, 0,
                                                               interval)
        start += int(np.floor((len_time - conca - len_each_train) / interval)) + 1
    # if not mode == 0:
    #     delta_tensor = []
    return delta_tensor, state_tensor, u_tensor


def stack_data(data, num_shifts, len_time, conca, interval=1):
    '''
    传入一条数据，注意只是一条数据
    :param data: 一条数据，即一个episode的数据
    :param num_shifts: shifts的步数，计算shifts步的预测损失
    :param len_time: 该条episode数据的长度
    :param conca: 采用(conca + 1)帧的数据作为输入，即当前帧和前conca帧数据串联作为输入
    :param interval: 每帧训练数据之间的间隔
    :return: 返回当前traj按照num_shifts分割后的数据,
             shape = [num_shifts+1, (len_time - conca - num_shifts - 1) / interval + 1, s_dim * (conca + 1)]
    '''
    n = data.shape[1]
    # 每一条训练数据的长度：因为shift num_shifts步，数据长度为num_shifts+1
    len_each_train = num_shifts + 1
    # num_traj = int(data.shape[0] / len_time)
    # 每一条num_traj可以分割成num_data_each_traj条训练数据
    num_data_each_traj = int(np.floor((len_time - conca - len_each_train) / interval)) + 1
    # 生成num_data_each_traj条数据之外多余的数据个数
    spare_num = int(np.mod(len_time - conca - len_each_train, interval))
    data_tensor = np.zeros([len_each_train, num_data_each_traj, n * (conca + 1)])
    if conca > 0:
        temp_tensor = np.zeros([len_time - conca, n * (conca + 1)])
        for k in np.arange(len_time - conca):
            for h in np.arange(conca + 1):
                temp_tensor[k, n * h:(h + 1) * n] = data[k + h, :]
    else:
        temp_tensor = data
    if spare_num > 0:
        init_index = np.random.randint(0, spare_num)
    else:
        init_index = 0
    for j in np.arange(num_data_each_traj):
        data_tensor_range = np.arange(j * interval + init_index, j * interval + len_each_train + init_index)
        data_tensor[:, j, :] = temp_tensor[data_tensor_range, :]
    return data_tensor


def split_pkl_file(path, n):
    '''
    spile a big .pkl file to n pieces of .pkl files
    :param path: the file path of the file need to be spilt
    :param n: spilt to n pieces
    :return: save n .pkl file the dir of path
    '''
    dic = read_pkl_as_dict(path)
    num_traj = len(dic['X_images'])
    # shuffle the sequence
    temp = list(zip(dic['X_images'], dic['X_states'], dic['U']))
    np.random.shuffle(temp)
    dic['X_images'], dic['X_states'], dic['U'] = zip(*temp)
    num_traj_each_file = int(np.floor(num_traj / n))
    for i in np.arange(n):
        dict_ = dict()
        start_index = i * num_traj_each_file
        # 最后一个文件，将不足num_traj_each_file条的数据也包括进来
        if (i < n - 1):
            end_index = (i + 1) * num_traj_each_file
        else:
            end_index = num_traj
        dict_['X_images'] = dic['X_images'][start_index:end_index]
        dict_['X_states'] = dic['X_states'][start_index:end_index]
        dict_['U'] = dic['U'][start_index:end_index]

        ab_path = os.path.split(path)[0]
        file_name = os.path.split(path)[1].split('.')
        new_path = ab_path + '/' + file_name[0] + '_%d.' % i + file_name[-1]
        save_dict_as_pkl(dict_, new_path)


def cal_cnn_pool_size(images_size, kernel_size, strides=1, padding='SAME'):
    '''
    计算经过单层卷积后的尺寸
    :param images_size: 输入图片尺寸， shape=[height, width, channel]
    :param kernel_size: 卷积核大小,format = [k_size, out_channel]
    :param strides: int, strides步长, [1, strides, strides, 1]
    :param padding: 'SAME', 'VALID'
    :return: [heigth, width, channel]
    '''
    height, width = images_size[0], images_size[1]
    kernel_width, out_channel = kernel_size[0], kernel_size[-1]
    if padding == 'SAME':
        if not (np.mod(height, strides) == 0):
            height = np.floor(height / strides) + 1
        else:
            height = np.floor(height / strides)
        if not (np.mod(width, strides) == 0):
            width = np.floor(width / strides) + 1
        else:
            width = np.floor(width / strides)
    elif padding == 'VALID':
        if not (np.mod(height - kernel_width, strides) == 0):
            height = np.floor((height - kernel_width) / strides) + 1
        else:
            height = np.floor((height - kernel_width) / strides)
        if not (np.mod(width - kernel_width, strides) == 0):
            width = np.floor((width - kernel_width) / strides) + 1
        else:
            width = np.floor((width - kernel_width) / strides)
    else:
        raise Exception('Please give the correct parameter of padding !!!')
    return [height, width, out_channel]


def cal_size(images_size, conv_size, pool_size):
    '''
    calculate the image size after a series of conv and pool operator. the padding is 'SAME' when do operator conv and pool
    :param images_size: [image_height, image_width, channel]
    :param conv_size: format: [[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
    :param pool_size: format: [[k_size1, strides1], [k_size2, strides2], ...]
    :return:
    '''
    size = images_size
    k_size = conv_size[0:2]
    for i in np.arange(len(conv_size)):
        k_size, strides, out_channel = conv_size[i][0], conv_size[i][1], conv_size[i][-1]
        size = cal_cnn_pool_size(size, [k_size, out_channel], strides)
        k_size, strides, out_channel = pool_size[i][0], pool_size[i][1], size[-1]
        size = cal_cnn_pool_size(size, [k_size, out_channel], strides)
    return size, int(size[0] * size[1] * size[2])
    print('After conv and pool, the image_size is :', size)


def choose_optimizer(args, regularized_loss, trainable_var):
    """Choose which optimizer to use for the network training.

    Arguments:
        regularized_loss -- loss, including regularization
        trainable_var -- list of trainable TensorFlow variables

    Returns:
        optimizer -- optimizer from TensorFlow Class optimizer

    Side effects:
        None

    Raises ValueError if params['opt_alg'] is not 'adam', 'adadelta', 'adagrad', 'adagradDA', 'ftrl', 'proximalGD',
        'proximalAdagrad', or 'RMS'
    """
    opt_alg = args['opt_alg']
    learning_rate = args['learning_rate']
    decay_rate = args['decay_rate']
    if opt_alg == 'adam':
        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(regularized_loss, var_list=trainable_var)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(regularized_loss, var_list=trainable_var)
    elif opt_alg == 'adadelta':
        if decay_rate > 0:
            optimizer = tf.train.AdadeltaOptimizer(learning_rate, decay_rate).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # defaults 0.001, 0.95
            optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(regularized_loss,
                                                                           var_list=trainable_var)
    elif opt_alg == 'adagrad':
        # also has initial_accumulator_value parameter
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(regularized_loss,
                                                                      var_list=trainable_var)
    elif opt_alg == 'adagradDA':
        # Be careful when using AdagradDA for deep networks as it will require careful initialization of the gradient
        # accumulators for it to train.
        optimizer = tf.train.AdagradDAOptimizer(learning_rate, tf.get_global_step()).minimize(
            regularized_loss,
            var_list=trainable_var)
    elif opt_alg == 'ftrl':
        # lots of hyperparameters: learning_rate_power, initial_accumulator_value,
        # l1_regularization_strength, l2_regularization_strength
        optimizer = tf.train.FtrlOptimizer(learning_rate).minimize(regularized_loss, var_list=trainable_var)
    elif opt_alg == 'proximalGD':
        # can have built-in reg.
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(regularized_loss,
                                                                                      var_list=trainable_var)
    elif opt_alg == 'proximalAdagrad':
        # initial_accumulator_value, reg.
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate).minimize(regularized_loss,
                                                                              var_list=trainable_var)
    elif opt_alg == 'RMS':
        # momentum, epsilon, centered (False/True)
        if decay_rate > 0:
            optimizer = tf.train.RMSPropOptimizer(learning_rate, decay_rate).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # default decay_rate 0.9
            optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(regularized_loss,
                                                                          var_list=trainable_var)
    else:
        raise ValueError("chose invalid opt_alg %s in params dict" % opt_alg)
    return optimizer


def check_data():
    '''
    在训练期间，显示采集的数据中u_tensor的最小值小于-1，这对于MountainCar是不对的，特此检查
    :return:
    '''
    path = '/home/BackUp/data/ConMountain_450_traj_0.pkl'
    dict = read_pkl_as_dict(path)
    image_tensor, state_tensor, u_tensor = process_data(dict, 30, 0, 10)
    print('max_u:', np.max(u_tensor), 'min_u:', np.min(u_tensor))
    print('max_state_x:', np.max(state_tensor[:, :, 0]), 'min_state_x:', np.min(state_tensor[:, :, 0]))
    print('max_state_xdot:', np.max(state_tensor[:, :, 1]), 'min_state_xdot:', np.min(state_tensor[:, :, 1]))
    img = image_tensor[10, 100, :] * 255
    img = np.reshape(img, [80, 80])
    plt.imshow(img)
    # plt.show()


def carsim_data_preprocess(is_noise=False):
    '''
    操作说明： 先读取文件求最大值和最小值，即将ix_max_min=True，然后再将形参is_max_min=False，重新调用该函数
    carsim matlab数据 --> pkl格式下的list格式，即pkl字典数据中每一个训练数据对应一个list，list中的每一个位对应一条traj数据
    carsim 采下的原始数据格式：state_name=Vx, Vy, AVz(偏航率), Steer_L1(左前轮转角)，Yaw, X, Y,  AAz(偏航加速度),
        Beta(vehicle slip angle车辆重心滑移角),steer_SW(方向盘转角), Steer_L2(右前轮转角)，Ax_SM(纵向加速度), Ay_SM,
        steer_SW(控制输入的方向盘转角，比状态中的steer_SW早一步), throttle, time(时间)
        数据shape = [step_num+1, dim * traj_num] 其中dim=所有数据的种类，包括时间（16维）
    :return:
    '''
    cur_path = os.getcwd()
    # train data
    train_step_num = 9001
    train_traj_num = 50
    train_file_num = 10

    # val data
    val_step_num = 9001
    val_traj_num = 50
    val_file_num = 1

    # test data
    test_step_num = 9001
    test_traj_num = 50
    test_file_num = 1
    interval = 0.01

    file_num = train_file_num + val_file_num + test_file_num
    dict_data = dict()
    # #### ----------------- 求最大值和最小值时取消注释 -----------------
    max_X, min_X = [], []
    max_U, min_U = [], []
    max_Delta, min_Delta = [], []
    # ---------------------------------------------------------------
    for i in np.arange(file_num):
        if i < test_file_num:
            flag = 1
            step_num = test_step_num
            traj_num = test_traj_num
            file_count = i + 1
        elif i >= test_file_num and i < (test_file_num + val_file_num):
            flag = 2
            step_num = val_step_num
            traj_num = val_traj_num
            file_count = i + 1 - test_file_num
        elif i >= (test_file_num + val_file_num):
            flag = 3
            step_num = train_step_num
            traj_num = train_traj_num
            file_count = i - test_file_num - val_file_num + 1
        if flag == 1:
            data_type = 'test'
        elif flag == 2:
            data_type = 'val'
        elif flag == 3:
            data_type = 'train'
        file_path = cur_path + '/carsim_v5/data/carsim_%s_%d_%d_%d.mat' % (
            data_type, step_num, traj_num, file_count)
        save_path = cur_path + '/carsim_v5/data/carsim_%s_%d_%d_%d.pkl' % (
            data_type, step_num, traj_num, file_count)
        max_min_path = cur_path + '/carsim_v5/data/max_min.txt'
        data = scio.loadmat(file_path)['data']
        X, U, Delta = [], [], []
        data_dim = 16

        for m in np.arange(traj_num):
            index = m * data_dim
            temp_X = data[1:step_num, index:(index + 3)]
            temp_X[:, 2] = temp_X[:, 2] * np.pi / 180
            temp_delta = data[2:step_num, index + 4:index + 7] - data[1:step_num - 1, index + 4:index + 7]
            temp_U = data[1:step_num - 1, index + 13:index + 15]
            temp_U[:, 0] = temp_U[:, 0] * np.pi / 180
            # #### ----------------- 将数据放到一起 -----------------
            max_X.append(np.max(temp_X, axis=0))
            min_X.append(np.min(temp_X, axis=0))
            max_U.append(np.max(temp_U, axis=0))
            min_U.append(np.min(temp_U, axis=0))
            max_Delta.append(np.max(temp_delta, axis=0))
            min_Delta.append(np.min(temp_delta, axis=0))

    # # --------------------------------- 求最大值与最小值 ------------------------
    f = open(max_min_path, 'w')
    max_X, min_X = np.array(max_X), np.array(min_X)
    max_X, min_X = np.max(max_X, axis=0), np.min(min_X, axis=0)
    max_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_X]
    min_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_X]
    f.writelines('[vx (km/h), vy (km/h), yaw_rate (rad/s)] \n')
    f.writelines(
        ['max_X: [', ', '.join(str(x) for x in max_X), '] min_X: [', ', '.join(str(x) for x in min_X), ']\n'])
    max_U, min_U = np.array(max_U), np.array(min_U)
    max_U, min_U = np.max(max_U, axis=0), np.min(min_U, axis=0)
    max_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_U]
    min_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_U]
    f.writelines('[steer_SW (rad), throttle] \n')
    f.writelines(['max_U: [', ', '.join(str(x) for x in max_U), '] min_U: [', ', '.join(str(x) for x in min_U), ']\n'])
    max_Delta, min_Delta = np.array(max_Delta), np.array(min_Delta)
    max_Delta, min_Delta = np.max(max_Delta, axis=0), np.min(min_Delta, axis=0)
    max_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_Delta]
    min_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_Delta]
    f.writelines('[delta_yaw (rad), delta_x (m), delta_y (m)] \n')
    f.writelines(
        ['max_Delta: [', ', '.join(str(x) for x in max_Delta), '] min_Delta: [', ', '.join(str(x) for x in min_Delta),
         ']\n'])
    f.writelines(['每一行分别对应采集的数据的最大值与最小值，用于归一化与反归一化'])
    f.close()
    # ---------------------------------------------------------------------------------
    max_X, min_X = np.array(max_X)[np.newaxis, :], np.array(min_X)[np.newaxis, :]
    max_U, min_U = np.array(max_U)[np.newaxis, :], np.array(min_U)[np.newaxis, :]
    max_Delta, min_Delta = np.array(max_Delta)[np.newaxis, :], np.array(min_Delta)[np.newaxis, :]
    for i in np.arange(file_num):
        if i < test_file_num:
            flag = 1
            step_num = test_step_num
            traj_num = test_traj_num
            file_count = i + 1
        elif i >= test_file_num and i < (test_file_num + val_file_num):
            flag = 2
            step_num = val_step_num
            traj_num = val_traj_num
            file_count = i + 1 - test_file_num
        elif i >= (test_file_num + val_file_num):
            flag = 3
            step_num = train_step_num
            traj_num = train_traj_num
            file_count = i - test_file_num - val_file_num + 1
        if flag == 1:
            data_type = 'test'
        elif flag == 2:
            data_type = 'val'
        elif flag == 3:
            data_type = 'train'
        file_path = cur_path + '/carsim_v5/data/carsim_%s_%d_%d_%d.mat' % (
            data_type, step_num, traj_num, file_count)
        if not is_noise:
            save_path = cur_path + '/carsim_v5/data/carsim_%s_%d_%d_%d.pkl' % (
                data_type, step_num, traj_num, file_count)
        else:
            save_path = cur_path + '/carsim_v5/data/carsim_noise_%s_%d_%d_%d.pkl' % (
                data_type, step_num, traj_num, file_count)
        data = scio.loadmat(file_path)['data']
        X, U, Delta = [], [], []
        data_dim = 16

        for m in np.arange(traj_num):
            index = index = m * data_dim
            temp_X = data[1:step_num, index:(index + 3)]
            temp_X[:, 2] = temp_X[:, 2] * np.pi / 180
            temp_delta = data[2:step_num, index + 4:index + 7] - data[1:step_num - 1, index + 4:index + 7]
            temp_U = data[1:step_num - 1, index + 13:index + 15]
            temp_U[:, 0] = temp_U[:, 0] * np.pi / 180
            temp_X = (temp_X - min_X) / (max_X - min_X)
            if is_noise:
                # 加入控制量上的噪声，throttle: [-0.01, 0.01], steering angle: [-1, 1]
                throttle_noise = np.random.uniform(-0.01, 0.01, [temp_U.shape[0], 1])
                angle_noise = np.random.uniform(-1, 1, [temp_U.shape[0], 1])
                noise = np.concatenate([throttle_noise, angle_noise], axis=1)
                temp_U = temp_U - noise
            temp_U = (temp_U - min_U) / (max_U - min_U)
            temp_delta = (temp_delta - min_Delta) / (max_Delta - min_Delta)
            X.append(temp_X)
            U.append(temp_U)
            Delta.append(temp_delta)
        print(len(X), len(U), len(Delta))
        dict_data['X'] = X
        dict_data['U'] = U
        dict_data['Delta'] = Delta
        save_dict_as_pkl(dict_data, save_path)


def acrobot_data_process():
    '''
    normalization处理
    :return:
    '''
    env = gym.make('Acrobot-v1').unwrapped
    env.reset()
    path = '/media/buduo/Data/Acrobot_Data/Acrobot_500_250_251_traj_%d_%d.pkl'
    for i in np.arange(100):
        path_ = path % (i / 10, np.mod(i, 10))
        data_dict = read_pkl_as_dict(path_)
        state = data_dict['X_states']
        for j in np.arange(len(state)):
            state[j][:, 4] = state[j][:, 4] / (4 * np.pi)
            state[j][:, 5] = state[j][:, 5] / (9 * np.pi)
        data_dict['X_states'] = state
        save_dict_as_pkl(data_dict, path_)


def h5_to_mat(h5path):
    '''
    transform a h5 file to a corresponding .mat file for MATLAB
    :param h5path: path of the .h5 file
    :return: None
    '''
    path = h5path.split('.')
    mat_path = path[0] + '.mat'
    dict = read_h5(h5path)
    save_dict_as_mat(dict, mat_path)


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def carsim_v6_preprocess_minus_1_1():
    ''' function: 该函数是用来处理carsim_v6版本的数据，即人工通过罗技方向盘采集，另外，此归一化的范围为[-1, 1]
    :return:
    '''
    cur_path = os.getcwd()
    file_num = 40
    file_name = 'D:\matlabDemo/carsim_sample/carsim_v6/carsim_v6_unit_%d.mat'
    max_min_path = cur_path + '/carsim_v6/data_minus_1_1/max_min.txt'
    dict_data = dict()
    # #### ----------------- 求最大值和最小值时取消注释 -----------------
    max_X, min_X = [], []
    max_U, min_U = [], []
    max_Delta, min_Delta = [], []
    for i in np.arange(file_num):
        file_count = i + 1
        data = scio.loadmat(file_name % file_count)['data']
        X, U, Delta = [], [], []
        # 全局的Vx, Vy
        temp_X = data[:, 7:9]
        # 加上 yaw rate
        temp_X = np.concatenate((temp_X, data[:, 6][:, np.newaxis]), axis=1)
        # temp_U 用来装 方向盘转角，油门，刹车
        temp_U = data[:, -3:]
        # temp_delta 用来装前后两个相邻时刻位置的差值
        temp_delta = data[1:, 1: 4] - data[0:-1, 1:4]
        # ================ 将所有数据文件中的最大最小值放到一起，最后起最值，用于归一和反归一化 ================
        max_X.append(np.max(temp_X, axis=0))
        min_X.append(np.min(temp_X, axis=0))
        max_U.append(np.max(temp_U, axis=0))
        min_U.append(np.min(temp_U, axis=0))
        max_Delta.append(np.max(temp_delta, axis=0))
        min_Delta.append(np.min(temp_delta, axis=0))

    # # --------------------------------- 求最大值与最小值 ------------------------
    f = open(max_min_path, 'w')
    max_X, min_X = np.array(max_X), np.array(min_X)
    max_X, min_X = np.max(max_X, axis=0), np.min(min_X, axis=0)
    max_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_X]
    min_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_X]
    max_X = [abs(x) if abs(x) > abs(y) else abs(y) for x, y in zip(max_X, min_X)]
    min_X = [-x for x in max_X]
    f.writelines('[global vx (km/h), vy (km/h), yaw_rate (rad/s)] \n')
    f.writelines(
        ['max_X: [', ', '.join(str(x) for x in max_X), '] min_X: [', ', '.join(str(x) for x in min_X), ']\n'])
    max_U, min_U = np.array(max_U), np.array(min_U)
    max_U, min_U = np.max(max_U, axis=0), np.min(min_U, axis=0)
    max_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_U]
    min_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_U]
    f.writelines('[steer_SW (rad), throttle] \n')
    f.writelines(['max_U: [', ', '.join(str(x) for x in max_U), '] min_U: [', ', '.join(str(x) for x in min_U), ']\n'])
    max_Delta, min_Delta = np.array(max_Delta), np.array(min_Delta)
    max_Delta, min_Delta = np.max(max_Delta, axis=0), np.min(min_Delta, axis=0)
    max_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_Delta]
    min_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_Delta]
    max_Delta = [abs(x) if abs(x) > abs(y) else abs(y) for x, y in zip(max_Delta, min_Delta)]
    min_Delta = [-x for x in max_Delta]
    f.writelines('[delta_yaw (rad), delta_x (m), delta_y (m)] \n')
    f.writelines(
        ['max_Delta: [', ', '.join(str(x) for x in max_Delta), '] min_Delta: [', ', '.join(str(x) for x in min_Delta),
         ']\n'])
    f.writelines(['每一行分别对应采集的数据的最大值与最小值，用于归一化与反归一化'])
    f.close()
    # ---------------------------------------------------------------------------------
    max_X, min_X = np.array(max_X)[np.newaxis, :], np.array(min_X)[np.newaxis, :]
    max_U, min_U = np.array(max_U)[np.newaxis, :], np.array(min_U)[np.newaxis, :]
    max_Delta, min_Delta = np.array(max_Delta)[np.newaxis, :], np.array(min_Delta)[np.newaxis, :]
    train_count, test_count, val_count = 0, 0, 0
    test_file = [7, 36]
    val_file = [8, 25]
    for i in np.arange(file_num):
        if i + 1 == test_file[test_count]:
            save_path = cur_path + '/carsim_v6/data_minus_1_1/carsim_v6_test_%d.pkl' % test_count
            test_count = test_count + 1 if test_count < len(test_file) - 1 else test_count
        elif i + 1 == val_file[val_count]:
            save_path = cur_path + '/carsim_v6/data_minus_1_1/carsim_v6_val_%d.pkl' % val_count
            val_count = val_count + 1 if val_count < len(val_file) - 1 else val_count
        else:
            save_path = cur_path + '/carsim_v6/data_minus_1_1/carsim_v6_%d.pkl' % train_count
            train_count += 1
        data = scio.loadmat(file_name % (i + 1))['data']
        X, U, Delta = [], [], []
        # 全局的Vx, Vy
        temp_X = data[:, 7:9]
        # 加上 yaw rate
        temp_X = np.concatenate((temp_X, data[:, 6][:, np.newaxis]), axis=1)
        # temp_U 用来装 方向盘转角，油门，刹车
        temp_U = data[0:-1, -3:]
        # temp_delta 用来装前后两个相邻时刻位置的差值
        temp_delta = data[1:, 1: 4] - data[0:-1, 1:4]
        # temp_X = (temp_X - min_X) / (max_X - min_X)
        temp_X = temp_X / abs(max_X)
        temp_U[:, 0] = temp_U[:, 0] / abs(max_U[:, 0])
        temp_U[:, 1:3] = (temp_U[:, 1:3] - min_U[:, 1:3]) / (max_U[:, 1:3] - min_U[:, 1:3])
        # temp_delta = (temp_delta - min_Delta) / (max_Delta - min_Delta)
        temp_delta = temp_delta / abs(max_Delta)
        # 融合油门和刹车，用一个变量表示，当油门为0，刹车不为0时，用负的刹车值表示油门
        for i in np.arange(np.shape(temp_U)[0]):
            if temp_U[i, 1] == 0 and temp_U[i, 2] != 0:
                temp_U[i, 1] = -temp_U[i, 2]
        print(np.shape(temp_X)[0], np.shape(temp_U)[0], np.shape(temp_delta)[0])
        X.append(temp_X)
        U.append(temp_U[:, 0:2])
        Delta.append(temp_delta)
        dict_data['X'] = X
        dict_data['U'] = U
        dict_data['Delta'] = Delta
        save_dict_as_pkl(dict_data, save_path)


def carsim_v6_preprocess_0_1(is_local_v=False):
    ''' function: 该函数是用来处理carsim_v6版本的数据，即人工通过罗技方向盘采集，另外，此归一化的范围为[0, 1]
    :return:
    '''
    version = '6'
    data_dir = '/data_0_1/' if not is_local_v else '/data_local_v/'
    cur_path = os.getcwd()
    file_num = 40
    file_name = 'D:\matlabDemo/carsim_sample/carsim_v' + version + '/carsim_v6_unit_%d.mat'
    max_min_path = cur_path + '/carsim_v' + version + data_dir + 'max_min.txt'
    dict_data = dict()
    # #### ----------------- 求最大值和最小值时取消注释 -----------------
    max_X, min_X = [], []
    max_U, min_U = [], []
    max_Delta, min_Delta = [], []
    for i in np.arange(file_num):
        file_count = i + 1
        data = scio.loadmat(file_name % file_count)['data']
        X, U, Delta = [], [], []
        if not is_local_v:
            # 全局的Vx, Vy
            temp_X = data[:, 7:9]
        else:
            # local velocity
            temp_X = data[:, 4:6]
        # 加上 yaw rate
        temp_X = np.concatenate((temp_X, data[:, 6][:, np.newaxis]), axis=1)
        # temp_U 用来装 方向盘转角，油门，刹车
        temp_U = data[:, -3:]
        # temp_delta 用来装前后两个相邻时刻位置的差值
        temp_delta = data[1:, 1: 4] - data[0:-1, 1:4]
        # ================ 将所有数据文件中的最大最小值放到一起，最后起最值，用于归一和反归一化 ================
        max_X.append(np.max(temp_X, axis=0))
        min_X.append(np.min(temp_X, axis=0))
        max_U.append(np.max(temp_U, axis=0))
        min_U.append(np.min(temp_U, axis=0))
        max_Delta.append(np.max(temp_delta, axis=0))
        min_Delta.append(np.min(temp_delta, axis=0))

    # # --------------------------------- 求最大值与最小值 ------------------------
    f = open(max_min_path, 'w')
    max_X, min_X = np.array(max_X), np.array(min_X)
    max_X, min_X = np.max(max_X, axis=0), np.min(min_X, axis=0)
    max_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_X]
    min_X = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_X]
    if not is_local_v:
        max_X = [abs(x) if abs(x) > abs(y) else abs(y) for x, y in zip(max_X, min_X)]
        min_X = [-x for x in max_X]
    f.writelines('[local vx (m/s), vy (m/s), yaw_rate (rad/s)] \n')
    f.writelines(
        ['max_X: [', ', '.join(str(x) for x in max_X), '] min_X: [', ', '.join(str(x) for x in min_X), ']\n'])
    max_U, min_U = np.array(max_U), np.array(min_U)
    max_U, min_U = np.max(max_U, axis=0), np.min(min_U, axis=0)
    max_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_U]
    min_U = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_U]
    f.writelines('[steer_SW (rad), throttle] \n')
    f.writelines(['max_U: [', ', '.join(str(x) for x in max_U), '] min_U: [', ', '.join(str(x) for x in min_U), ']\n'])
    max_Delta, min_Delta = np.array(max_Delta), np.array(min_Delta)
    max_Delta, min_Delta = np.max(max_Delta, axis=0), np.min(min_Delta, axis=0)
    max_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in max_Delta]
    min_Delta = [(np.ceil(x * 10) if (x > 0) else np.floor(x * 10)) / 10 for x in min_Delta]
    max_Delta = [abs(x) if abs(x) > abs(y) else abs(y) for x, y in zip(max_Delta, min_Delta)]
    min_Delta = [-x for x in max_Delta]
    f.writelines('[delta_yaw (rad), delta_x (m), delta_y (m)] \n')
    f.writelines(
        ['max_Delta: [', ', '.join(str(x) for x in max_Delta), '] min_Delta: [', ', '.join(str(x) for x in min_Delta),
         ']\n'])
    f.writelines(['每一行分别对应采集的数据的最大值与最小值，用于归一化与反归一化'])
    f.close()
    # ---------------------------------------------------------------------------------
    max_X, min_X = np.array(max_X)[np.newaxis, :], np.array(min_X)[np.newaxis, :]
    max_U, min_U = np.array(max_U)[np.newaxis, :], np.array(min_U)[np.newaxis, :]
    max_Delta, min_Delta = np.array(max_Delta)[np.newaxis, :], np.array(min_Delta)[np.newaxis, :]
    train_count, test_count, val_count = 0, 0, 0
    test_file = [7, 36]
    val_file = [8, 25]
    for i in np.arange(file_num):
        base_path = cur_path + '/carsim_v' + version + data_dir + 'carsim_v' + version
        if i + 1 == test_file[test_count]:
            save_path = base_path + '_test_%d.pkl' % test_count
            test_count = test_count + 1 if test_count < len(test_file) - 1 else test_count
        elif i + 1 == val_file[val_count]:
            save_path = base_path + '_val_%d.pkl' % val_count
            val_count = val_count + 1 if val_count < len(val_file) - 1 else val_count
        else:
            save_path = base_path + '_%d.pkl' % train_count
            train_count += 1
        data = scio.loadmat(file_name % (i + 1))['data']
        X, U, Delta = [], [], []
        if not is_local_v:
            # 全局的Vx, Vy
            temp_X = data[:, 7:9]
        else:
            # local velocity
            temp_X = data[:, 4:6]
        # 加上 yaw rate
        temp_X = np.concatenate((temp_X, data[:, 6][:, np.newaxis]), axis=1)
        # temp_U 用来装 方向盘转角，油门，刹车
        temp_U = data[0:-1, -3:]
        # temp_delta 用来装前后两个相邻时刻位置的差值
        temp_delta = data[1:, 1: 4] - data[0:-1, 1:4]
        temp_X = (temp_X - min_X) / (max_X - min_X)
        # temp_X = temp_X / abs(max_X)
        temp_U[:, 0] = temp_U[:, 0] / abs(max_U[:, 0])
        temp_U[:, 1:3] = (temp_U[:, 1:3] - min_U[:, 1:3]) / (max_U[:, 1:3] - min_U[:, 1:3])
        temp_delta = (temp_delta - min_Delta) / (max_Delta - min_Delta)
        # temp_delta = temp_delta / abs(max_Delta)
        # 融合油门和刹车，用一个变量表示，当油门为0，刹车不为0时，用负的刹车值表示油门
        for i in np.arange(np.shape(temp_U)[0]):
            if temp_U[i, 1] == 0 and temp_U[i, 2] != 0:
                temp_U[i, 1] = -temp_U[i, 2]
        print(np.shape(temp_X)[0], np.shape(temp_U)[0], np.shape(temp_delta)[0])
        X.append(temp_X)
        U.append(temp_U[:, 0:2])
        Delta.append(temp_delta)
        dict_data['X'] = X
        dict_data['U'] = U
        dict_data['Delta'] = Delta
        save_dict_as_pkl(dict_data, save_path)


def get_carsim_v6_data_for_matlab(is_train_data=True):
    ''' 导出数据到.mat格式，用于matlab下的EDMD测试
    :return:
    '''
    root_path = ComUtils.get_upLevel_dir(cur_path, 1)
    max_s, min_s = np.array([[27.3], [1.7], [1.1]]), np.array([[-0.2], [-2.0], [-1.1]])
    max_a, min_a = np.array([[7.9], [0.2], [0.1]]), np.array([[-7.9], [-0.], [0.]])
    file_num = 36 if is_train_data else 2
    X, Y, U = [], [], []
    file_len = np.zeros([file_num])
    for i in np.arange(file_num):
        if is_train_data:
            file_path = root_path + '/carsim_v6/data_0_1/carsim_v6_%d.pkl' % i
        else:
            file_path = root_path + '/carsim_v6/data_0_1/carsim_v6_test_%d.pkl' % i
        data_dict = read_pkl_as_dict(file_path % i)
        # X_array: len x 3 ; U_array: len-1 x 2
        X_array, U_array = data_dict['X'][0], data_dict['U'][0]
        X_temp, Y_temp, U_temp = X_array[:-1, :], X_array[1:, :], U_array
        ep_len = X_temp.shape[0]
        file_len[i] = ep_len
        if X == []:
            # self.X self.Y ---- shape =(None, s_dim)
            X, Y = np.transpose(X_temp), np.transpose(Y_temp)
            # self.U ------ shape = (a_dim, None)
            U = np.transpose(U_temp)
        else:
            X = np.concatenate((X, np.transpose(X_temp)), axis=1)
            Y = np.concatenate((Y, np.transpose(Y_temp)), axis=1)
            U = np.concatenate((U, np.transpose(U_temp)), axis=1)
    # # calculate the mean and standard deviation of each state dim  归一化
    mean_s = np.mean(X, axis=1)[:, np.newaxis]
    std_s = np.std(X, axis=1)[:, np.newaxis]
    mean_a = np.mean(U, axis=1)[:, np.newaxis]
    std_a = np.std(U, axis=1)[:, np.newaxis]
    training_data = {
        'X': X,
        'Y': Y,
        'U': U,
        'max_s': max_s,
        'min_s': min_s,
        'max_a': max_a,
        'min_a': min_a,
        'mean_s': mean_s,
        'std_s': std_s,
        'mean_a': mean_a,
        'std_a': std_a,
        'file_len': file_len
    }
    if is_train_data:
        # save_path = 'D:\matlabDemo\DEDMD_draw\draw_predictions/carsim_v6_normed_training_data.mat'
        save_path = os.path.join(root_path, 'carsim_v6/data_0_1/carsim_v6_normed_training_data.mat')
    else:
        # save_path = 'D:\matlabDemo\DEDMD_draw\draw_predictions/carsim_v6_normed_testing_data.mat'
        save_path = os.path.join(root_path, 'carsim_v6/data_0_1/carsim_v6_normed_testing_data.mat')
    # save_path = self.cur_path + '/carsim_v5/model/carsim_2020_02_19_17_27/test/excellent/training_data_after_norm.mat'
    # save_path = 'D:\matlabDemo\D_EDMD/train_data.mat'
    scio.savemat(save_path, training_data)
    print('shape of the X: %d, Y: %d, U: %d', (X.shape, Y.shape, U.shape))
    print('================= save training data successfully !!! ===============')


print('current path ---------------------------------------------', os.getcwd())
# get_carsim_v6_data_for_matlab(is_train_data=False)
# carsim_v6_preprocess_0_1(is_local_v=True)
# get_carsim_v6_data_for_matlab(is_local_v=True, is_train_data=False)
# carsim_data_preprocess(is_noise=True)
# acrobot_data_process()

# data_path = os.getcwd() + '/carsim_v5/data/carsim_train_3000_100_1.pkl'
# data = read_pkl_as_dict(data_path)
# delta_tensor, state_tensor, u_tensor = carsim_data_process(data, 30, 2, 1, mode=0)
# print(delta_tensor.shape, state_tensor.shape, u_tensor.shape)
# check_data()
# get_con_mountaincar_args()

# conv_size = [[8, 2, 2, 32], [4, 2, 32, 64], [3, 2, 64, 64]]
# # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
# pool_size = [[2, 2], [2, 2], [2, 2]]
# cal_size([80, 80, 2], conv_size, pool_size)

# path = '/home/BackUp/data/ConMountain_450_traj_%d.pkl'
# path = '/home/BackUp/cartpole_data/CartPole_1000_traj_%d.pkl'
# for i in np.arange(10):
#     path_ = path % (i)
#     split_pkl_file(path_, 10)

##
# path = '/home/BackUp/data/ConMountain_450_traj_7_9.pkl'
# dict = read_pkl_as_dict(path)
# image_tensor, state_tensor, u_tensor = process_data(dict, 30, 2, 7)
# print('wait')
## 计算尺寸
# conv_size = [[8, 2, 2 + 1, 16], [4, 1, 16, 32], [3, 1, 32, 32]]
#     # 格式：[[k_size1, strides1], [k_size2, strides2], ...]
# pool_size = [[2, 2], [2, 2], [2, 2]]
# _, num_neural = cal_size([40, 240, 2 + 1], conv_size, pool_size)
# print(num_neural)
# h5path = 'D:\myPythonDemo\myKoopman\DEDMD\carsim_v5\model\carsim_2020_02_19_17_27/train_val_losses.h5'
# h5_to_mat(h5path)
