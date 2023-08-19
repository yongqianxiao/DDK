''' 状态归一化到[0, 1]
Author: Yongqian, Xiao
Specification: 一个新的版本，与DEDMDP2_1.py 类似，不同之处在于训练数据不同，代码不同在于归一化等不同
Date: 2019-12-23
'''
import tensorflow as tf
import numpy as np
import helperfuns_DEDMDP as helper
import time
import os, sys
import gym
from sample_gym_image_data import SampleData
import scipy.io as scio
import CustomizeUtils.CommonUtils as Comutils
from CustomizeUtils import VisualizeUtils

# 当数组维度比较大时，会出现省略号，这条语句可以避免这种情况
np.set_printoptions(threshold=sys.maxsize)


class DeepEDMDwithPixel():
    def __init__(self, args):
        self.test_flag = False
        self.train_AB_only = False
        # self.for_program_testing(args, 1)
        self.random_weight_layer = args['random_weight_layer']
        self.s_dim, self.u_dim, self.lift_dim = args['s_dim'], args['u_dim'], args['lift_dim']
        self.save_now = False
        self.args = args
        self.is_predict = args['is_predict']
        self.state_bound = np.array(args['state_bound'])[:, np.newaxis]
        self.action_bound = np.array(args['action_bound'])[:, np.newaxis]
        self.image_height, self.image_width = args['image_height'], args['image_width']
        self.conca, self.interval = args['conca_num'], args['interval']
        # flag of training or testing. True means training, or it's testing
        self.phase = tf.placeholder(tf.bool, name='phase')
        # the rate of dropout
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        # self.predicted_step = args['predicted_step']
        self.predicted_step = args['num_shifts']
        self.num_shifts = args['num_shifts']
        self.num_koopman_shifts = args['num_koopman_shifts']
        # for control
        self.control_init_flag = False  # 控制参数初始化flag

        # 各个状态计算损失的权重
        self.state_weight = args['state_weight'][np.newaxis, :]
        # create a complete koopman net
        self.create_koopman_net(phase=self.phase, keep_prob=self.keep_prob, args=args)

        # create the predicted network
        if args['input_mode'] == 0:
            self.x0 = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.conca + 1])
        elif args['input_mode'] == 1:
            self.x0 = tf.placeholder(tf.float32, shape=[1, (self.conca + 1) * self.s_dim], name='x0')
        if not self.test_flag:
            self.U = tf.placeholder(tf.float32, shape=[self.predicted_step, self.u_dim], name='U')
        else:
            self.U = tf.placeholder(tf.float32, shape=[self.predicted_step - self.conca, self.u_dim],
                                    name='U')
        # y_koopman denotes the predicted states
        self.y_koopman = self.get_Koopman_prediction(args, self.x0, self.U,
                                                     None if not self.test_flag else self.predicted_step - args[
                                                         'conca_num'])

        # define loss function
        trainable_var = tf.trainable_variables()
        self.loss1, self.loss2, self.loss3, self.loss_Linf, self.loss = self.define_loss(args, self.x, self.y,
                                                                                         self.g_list, self.weights,
                                                                                         self.biases,
                                                                                         phase=self.phase,
                                                                                         keep_prob=self.keep_prob)
        self.loss_L2, self.regularized_loss, self.regularized_loss1 = self.define_regularization(args, trainable_var,
                                                                                                 self.loss, self.loss1)

        # choose optimization algorithm
        if not self.is_predict:
            self.optimizer = helper.choose_optimizer(args, self.regularized_loss, trainable_var)
            self.optimizer_autoencoder = helper.choose_optimizer(args, self.regularized_loss1, trainable_var)

        self.sess = tf.Session()
        vl = [v for v in tf.global_variables() if "Adam" not in v.name]
        max_saver = 50 if args['is_random_train'] is True else 5
        self.saver = tf.train.Saver(var_list=vl, max_to_keep=max_saver)
        self.sess.run(tf.global_variables_initializer())
        # 获取test 数据, 将val的数据也用于测试
        test_data_path = args['test_data_path']
        val_data_path = args['val_data_path']
        self.total_test_delta_list, self.total_test_data_list, self.total_test_u_list = [], [], []
        # for i in np.arange(args['test_file_num'] + args['val_file_num']):
        for i in np.arange(1):
            if i < args['test_file_num']:
                temp_path = test_data_path % i
            else:
                temp_path = val_data_path % (i - args['test_file_num'])
            Delta_tensor, state_tensor, u_tensor = self.get_train_test_val(args, 'test', temp_path)
            self.total_test_delta_list.append(Delta_tensor)
            self.total_test_data_list.append(state_tensor)
            self.total_test_u_list.append(u_tensor)

        # print(x.shape, len(y), len(g_list))
        is_restore = self.restore_model(args)
        if not self.is_predict:
            if is_restore:
                print('======================= train continuously !!! ====================')
            else:
                print('======================= restart training !!! ====================')
            helper.save_dict_as_pkl(args, args['args_file_path'])
            self.training(args)
        else:
            if not is_restore:
                Exception('The model does not exist, Please check the model path !!!')
            else:
                if not os.path.exists(args['args_file_path']):
                    helper.save_dict_as_pkl(args, args['args_file_path'])
                # self.lqr_control(args)
                self.save_parameters_for_matlab(args)
                pre_dict = dict()
                self.latent_error = []
                latent_error_savedir = Comutils.get_upLevel_dir(args['test_image_path'], 2) + "/latent_error/mean_latent_error.png"
                for i in range(3000):
                    if self.test_flag:
                        # index = np.mod(i, 2) + 1
                        # self.for_program_testing(args, i + 1)
                        self.random_result_ave(args, i + 1)
                    else:
                        pre_dict['x_true_%d' % i], pre_dict['U_%d' % i], pre_dict['x_Koopman_%d' % i] = self.prediction(args, str(i))
                self.latent_error = Comutils.list2array(self.latent_error, dim=0, add_dim=True)
                self.latent_error = np.mean(self.latent_error, axis=0)
                VisualizeUtils.draw_latent_error(None, [abs(self.latent_error)], ["abs_latent_error"],
                                                 is_show=True, is_save=True, save_dir=latent_error_savedir)

    def random_result_ave(self, args, k):
        """
        带随机权重层的测试, l22
        Args:
            args:
            k:

        Returns:

        """
        random_layer = []
        times = 100
        predicted_step = self.predicted_step
        pred_dict = dict()
        pred_x = np.zeros([times, self.s_dim, predicted_step], dtype=np.float32)
        pred_xmin = np.zeros([self.s_dim, predicted_step], dtype=np.float32)
        pred_xmax = np.zeros([self.s_dim, predicted_step], dtype=np.float32)
        pred_xave = np.zeros([self.s_dim, predicted_step], dtype=np.float32)
        conca = self.conca
        x_min = args['state_bound'][0, :][:, np.newaxis]
        x_max = args['state_bound'][1, :][:, np.newaxis]
        U_min = args['action_bound'][0, :][:, np.newaxis]
        U_max = args['action_bound'][1, :][:, np.newaxis]
        save_path = os.getcwd() + '/carsim_v6/model_local_v/carsim_2020_06_11_19_09_l11/test/selected_DEDMD/%d.mat' % k
        test_dict = scio.loadmat(save_path)
        x_true = (test_dict['true'] - x_min[-self.s_dim:, :]) / (x_max[-self.s_dim:, :] - x_min[-self.s_dim:, :])
        x0 = np.squeeze(x_true[:, 0])[np.newaxis, :]
        for i in np.arange(conca):
            temp_x0 = np.squeeze(x_true[:, i + 1])[np.newaxis, :]
            x0 = np.concatenate((x0, temp_x0), axis=1)
        U = test_dict['action']
        U[0, :] = U[0, :] / abs(U_max[0, :])
        for q in np.arange(U.shape[1]):
            U[1, q] = U[1, q] / U_max[1, 0] if U[1, q] >= 0 else -U[1, q] / U_max[2, 0]
        u_test_tensor = np.transpose(U[:, conca:])
        feed_test_dict = {self.x0: x0, self.U: u_test_tensor, self.phase: 0, self.keep_prob: 1.}
        for i in np.arange(times):
            y_koopman = self.sess.run(self.y_koopman, feed_dict=feed_test_dict)
            x_Koopman = np.squeeze(np.transpose(np.array(y_koopman)))
            xmax_minus_xmin = np.tile((x_max[-self.s_dim:, :] - x_min[-self.s_dim:, :]),
                                      (1, predicted_step + 1 - conca))
            x_Koopman = np.add(np.multiply(x_Koopman, xmax_minus_xmin), x_min[:self.s_dim, :])
            pred_x[i, :, :] = x_Koopman[:, 1:]
            temp_random_layer = self.sess.run(self.weights['WEF3'])
            random_layer.append(temp_random_layer)
        random_layer = np.array(random_layer)
        pred_xmin = np.min(pred_x, axis=0)
        pred_xmax = np.max(pred_x, axis=0)
        pred_xave = np.mean(pred_x, axis=0)
        pred_dict['rdedmd_pred'] = pred_x[0, :, :]
        pred_dict['rdedmd_pred_xmin'] = pred_xmin
        pred_dict['rdedmd_pred_xmax'] = pred_xmax
        pred_dict['rdedmd_pred_xave'] = pred_xave
        pred_dict['rdedmd_true'] = test_dict['true'][:, conca:]
        pred_dict['rdedmd_action'] = test_dict['action']
        pred_dict['random_layer'] = random_layer
        save_path = os.getcwd() + '/carsim_v6/model_local_v/carsim_2020_06_14_03_37_l22/test/from_DEDMD/%d_RDEDMD_l22.mat' % k
        helper.save_dict_as_mat(pred_dict, save_path)

    def training(self, args):
        # ---------------------- training params -------------------------------------
        self.params = self.get_training_params()
        count = args['image_count']
        best_error = 10000
        batch_size = args['batch_size']
        self.start = time.time()
        self.error_recorder = []
        self.loss_dict = dict()
        # 获取 validation 数据
        val_data_path = args['val_data_path']
        self.total_val_delta_list, self.total_val_state_list, self.total_val_u_tensor = [], [], []
        for i in np.arange(args['val_file_num']):
            temp_path = val_data_path % i
            Delta_tensor, state_tensor, u_tensor = self.get_train_test_val(args, 'val', temp_path)
            self.total_val_delta_list.append(Delta_tensor)
            self.total_val_state_list.append(state_tensor)
            self.total_val_u_tensor.append(u_tensor)
        train_val_error = np.zeros([14, ])
        # -------------------training -------------------------
        times_each_file = 200  # training times of each train dataset file
        finished = 0
        for i in range(args['train_file_num'] * times_each_file):
            i = np.mod(i, args['train_file_num'])
            if self.is_predict:
                break
            # reading training data
            cur_path = os.getcwd()
            if 'CarSim' in args['domain_name'] or (args['domain_name'] == 'CartPole-v2'):
                train_data_path = args['train_data_path'] % (int(np.mod(i, args['train_file_num'])))
            else:
                train_data_path = args['train_data_path'] % (int(i / 10), int(np.mod(i, 10)))
            train_image_tensor, train_data_tensor, train_u_tensor = self.get_train_test_val(args, 'train',
                                                                                            train_data_path)
            # the number of train data of the current train set
            num_examples = train_data_tensor.shape[1]
            num_batches = int(np.floor(num_examples / args['batch_size']))
            # shuffle the sequence
            ind = np.arange(num_examples)
            np.random.shuffle(ind)
            train_data_tensor = train_data_tensor[:, ind, :]
            train_U_tensor = train_u_tensor[:, ind, :]
            if args['input_mode'] == 0:
                train_image_tensor = train_image_tensor[:, ind, :]
            # count the times don't get a better result. if it doesn't get a better result more than 40 times
            # continuously, then break the training
            for step in range(num_batches):
                self.params['train_times'] += 1
                if args['batch_size'] < train_data_tensor.shape[1]:
                    # 记录训练集已经训练过样本的index
                    offset = (step * batch_size) % (num_examples - batch_size)
                else:
                    offset = 0
                batch_data_train = train_data_tensor[:, offset:(offset + batch_size), :]
                batch_U_train = train_U_tensor[:, offset:(offset + batch_size), :]
                # 随机选择一个validation 文件来提供validation 数据
                val_file_index = np.random.randint(low=0, high=args['val_file_num'])
                val_data_tensor = self.total_val_state_list[val_file_index]
                val_u_tensor = self.total_val_u_tensor[val_file_index]
                val_index = np.random.randint(low=0, high=val_data_tensor.shape[1] - 1, size=[100])
                if args['input_mode'] == 0:
                    val_image_tensor = self.total_val_delta_list[val_file_index]
                    # 训练数据
                    batch_image_train = train_image_tensor[:, offset:(offset + batch_size), :]
                    batch_image_train = np.reshape(batch_image_train,
                                                   [self.num_shifts + 1, batch_size, self.image_height,
                                                    self.image_width, self.conca + 1])
                    feed_dict_train = {self.x: batch_data_train, self.u: batch_U_train, self.phase: True,
                                       self.keep_prob: args['dropout_rate'], self.images: batch_image_train}
                    feed_dict_train_loss = {self.x: batch_data_train, self.u: batch_U_train, self.phase: True,
                                            self.keep_prob: 1., self.images: batch_image_train}
                    # 校验数据
                    temp_val_image_tensor = val_image_tensor[:, val_index, :]
                    temp_val_image_tensor = np.reshape(temp_val_image_tensor,
                                                       [self.num_shifts + 1, -1, self.image_height,
                                                        self.image_width, self.conca + 1])
                    feed_dict_val = {self.x: val_data_tensor[:, val_index, :], self.u: val_u_tensor[:, val_index, :],
                                     self.phase: False, self.keep_prob: 1.0, self.images: temp_val_image_tensor}
                elif args['input_mode'] == 1:
                    # 训练数据
                    feed_dict_train = {self.x: batch_data_train, self.u: batch_U_train, self.phase: True,
                                       self.keep_prob: args['dropout_rate']}
                    feed_dict_train_loss = {self.x: batch_data_train, self.u: batch_U_train, self.phase: True,
                                            self.keep_prob: 1.}
                    # 校验数据
                    feed_dict_val = {self.x: val_data_tensor[:, val_index, :], self.u: val_u_tensor[:, val_index, :],
                                     self.phase: False, self.keep_prob: 1.0}

                # ------------------------ train -------------------
                # if not self.params['been3min']:
                #     self.sess.run(self.optimizer_autoencoder, feed_dict=feed_dict_train)
                # else:
                #     self.sess.run(self.optimizer, feed_dict=feed_dict_train)
                self.sess.run(self.optimizer, feed_dict=feed_dict_train)
                if step % 20 == 0:
                    train_error = self.sess.run(self.loss, feed_dict=feed_dict_train_loss)
                    val_error = self.sess.run(self.loss, feed_dict=feed_dict_val)
                    # self.prediction(args, str(count))

                    if val_error < (best_error - best_error * (10 ** (-4))):
                        self.params['current_samples'] = self.params['train_times'] * batch_size
                        self.prediction(args, str(count))
                        count += 1
                        count_better_val = 0
                        best_error = val_error.copy()
                        # if self.params['been3min']:
                        #     for k in range(3):
                        #         self.prediction(args, str(count) + '_%d' % k)
                        #     count += 1
                        reg_train_err = self.sess.run(self.regularized_loss, feed_dict=feed_dict_train_loss)
                        reg_val_err = self.sess.run(self.regularized_loss, feed_dict=feed_dict_val)

                        print(
                            "The %dth train file, New best val error %f (with reg. train err %f and reg. val err %f)" % (
                                i, best_error, reg_train_err, reg_val_err))
                    else:
                        print(
                            'The %dth train file, 20 / %d steps passed, it did not become better,best val error %f ,err %f)' % (
                                i, count_better_val, best_error, val_error))

                    train_val_error[0] = train_error.copy()
                    train_val_error[1] = val_error.copy()
                    train_val_error[2] = self.sess.run(self.regularized_loss, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[3] = self.sess.run(self.regularized_loss, feed_dict=feed_dict_val).copy()
                    train_val_error[4] = self.sess.run(self.loss1, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[5] = self.sess.run(self.loss1, feed_dict=feed_dict_val).copy()
                    train_val_error[6] = self.sess.run(self.loss2, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[7] = self.sess.run(self.loss2, feed_dict=feed_dict_val).copy()
                    train_val_error[8] = self.sess.run(self.loss3, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[9] = self.sess.run(self.loss3, feed_dict=feed_dict_val).copy()
                    train_val_error[10] = self.sess.run(self.loss_Linf, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[11] = self.sess.run(self.loss_Linf, feed_dict=feed_dict_val).copy()
                    if np.isnan(val_error):
                        self.params['stop_condition'] = 'total loss is nan'
                        finished = 1
                        break
                    if np.isnan(train_val_error[10]):
                        self.params['stop_condition'] = 'loss_Linf is nan'
                        finished = 1
                        break
                    train_val_error[12] = self.sess.run(self.loss_L2, feed_dict=feed_dict_train_loss).copy()
                    train_val_error[13] = self.sess.run(self.loss_L2, feed_dict=feed_dict_val).copy()
                    self.error_recorder.append(train_val_error)
                    train_val_error = np.zeros([14, ])
                    finished, save_now = helper.check_progress(self.start, best_error, self.params, batch_size)
                    if args['is_random_train'] is True and (time.time() - self.start > 3 * 60 * 60):
                        finished = 1
                    # if save_now:
                    #     self.prediction(args, str(count))
                    count_better_val += 1
                    if finished == 1 or best_error == 0.0:
                        break
                    if count_better_val > 2000 or (count_better_val > 2000 and self.params['train_times'] > 12000 * 20):
                        finished = 1
                        break
            if finished == 1 or best_error == 0.0:
                self.params['latest'] = best_error
                self.params['latest_samples'] = self.params['train_times'] * batch_size
                self.prediction(args, 'lastest')
                break

    def for_program_testing(self, args, k):
        conca = self.conca
        x_min = args['state_bound'][0, :][:, np.newaxis]
        x_max = args['state_bound'][1, :][:, np.newaxis]
        U_min = args['action_bound'][0, :][:, np.newaxis]
        U_max = args['action_bound'][1, :][:, np.newaxis]
        save_path = 'D:\myPythonDemo\myKoopman\DEDMD\carsim_v6\model_local_v/carsim_2020_06_12_09_38_l12/test/%d.mat' % k
        if not os.path.exists(save_path):
            return
        test_dict = scio.loadmat(save_path)
        x_true = (test_dict['true'] - x_min[-self.s_dim:, :]) / (x_max[-self.s_dim:, :] - x_min[-self.s_dim:, :])
        x0 = np.squeeze(x_true[:, 0])[np.newaxis, :]
        for i in np.arange(conca):
            temp_x0 = np.squeeze(x_true[:, i + 1])[np.newaxis, :]
            x0 = np.concatenate((x0, temp_x0), axis=1)
        # u_test_tensor = np.transpose((test_dict['action'] - U_min) / (U_max - U_min))[conca:, :]
        U = test_dict['action']
        U[0, :] = U[0, :] / abs(U_max[0, :])
        for q in np.arange(U.shape[1]):
            U[1, q] = U[1, q] / U_max[1, 0] if U[1, q] >= 0 else -U[1, q] / U_max[2, 0]
        u_test_tensor = np.transpose(U[:, conca:])
        # feed_test_dict = {self.x0: x0, self.U: u_test_tensor, self.phase: 0, self.keep_prob: 1.}
        # encoder, koopman, y_koopman = self.sess.run([self.encoder_x0, self.koopman, self.y_koopman], feed_dict=feed_test_dict)
        feed_test_dict = {self.x: x_true, self.U: u_test_tensor, self.phase: 0, self.keep_prob: 1.}
        g_list, koopman, y_koopman = self.sess.run([self.g_list, self.koopman, self.y_koopman], feed_dict=feed_test_dict)
        x_Koopman = np.squeeze(np.transpose(np.array(y_koopman)))
        xmax_minus_xmin = np.tile((x_max[-self.s_dim:, :] - x_min[-self.s_dim:, :]),
                                  (1, self.predicted_step + 1 - conca))
        x_Koopman = np.add(np.multiply(x_Koopman, xmax_minus_xmin), x_min[:self.s_dim, :])
        # save the picture result
        images_dir = args['test_image_path'] % k
        # print("x_true: ", test_dict['true'].shape, "U: ", test_dict['action'].shape, "x_Koopman: ", x_Koopman.shape)
        offset = helper.plot_X_and_Y(args, test_dict['true'][:, conca:], x_Koopman, test_dict['action'],
                                     is_reference=False, is_save=True, state_name=args['state_name'],
                                     action_name=args['action_name'], predict_step=self.predicted_step,
                                     save_dir=images_dir, is_show=False,
                                     title=args['domain_name'] + ' with no normalization')
        weights = self.sess.run(self.weights)

    def prediction(self, args, sequence_str):
        '''
        Predict predicted_step steps with the current weights, and save the predicted picture, the args and the params
        :param sequence_str: part of the images saving path string
        :param predicted_step: the num of the step be going to be predicted
        :param save_dir: save the picture and the dict contains the args and the params to the specified dir.
        :return:
        '''
        # the passed time since the program begun
        if not self.is_predict:
            passed_time = int(np.ceil((time.time() - self.start) / 60))
            # self.save_weight_txt(passed_time)
        # 从多个测试集文件中随机告白选择一个用于测试
        # test_file_index = np.random.randint(low=0, high=args['test_file_num'] + args['val_file_num'])
        test_file_index = 0
        test_image_tensor = self.total_test_delta_list[test_file_index]
        test_data_tensor = self.total_test_data_list[test_file_index]
        u_test_tensor = self.total_test_u_list[test_file_index]
        test_num = np.random.randint(low=0, high=test_data_tensor.shape[1] - 1)
        # print('s_test_tensor:, ', test_data_tensor.shape)
        # print("u_test_tensor:, ", u_test_tensor.shape)
        # randomly to choose a testing vector

        x_true = np.transpose(test_data_tensor[:, test_num, :])
        # input_mode = 0 表示当前是图像输入模式，input_mode=1 表示是状态作为输入模式
        if args['input_mode'] == 0:
            x0 = np.squeeze(test_image_tensor[0, test_num, :])[np.newaxis, :]
            x0 = np.reshape(x0, [1, self.image_height, self.image_width, self.conca + 1])
        elif args['input_mode'] == 1:
            x0 = np.squeeze(test_data_tensor[0, test_num, :])[np.newaxis, :]
        feed_test_dict = {self.x0: x0, self.U: u_test_tensor[:, test_num, :], self.phase: 0, self.keep_prob: 1.}
        y_lift = self.sess.run(self.y_koopman, feed_dict=feed_test_dict)
        feed_test_dict1 = {self.x0: x0, self.U: u_test_tensor[:, test_num, :], self.phase: 0, self.keep_prob: 1.}
        koopman, y_koopman = self.sess.run([self.koopman, self.y_koopman], feed_dict=feed_test_dict1)
        x_tmp = x_true.T[:, np.newaxis, :]
        feed_test_dict2 = {self.x0: x0, self.x: x_tmp, self.phase: 0, self.keep_prob: 1.}
        g_list = self.sess.run(self.g_list, feed_dict=feed_test_dict2)
        # print('the length of y_lift: ', len(y_lift))
        # print('the shape of test x_true: ', x_true.shape)
        x_Koopman = np.squeeze(np.transpose(np.array(y_lift)))
        g_list = Comutils.list2array(g_list[1:], dim=0)
        koopman = Comutils.list2array(koopman, dim=0)
        save_path = Comutils.get_upLevel_dir(args['test_image_path'], 2) + "/latent_error/LatentError_" + sequence_str + ".png"
        is_save = True if np.mod(int(sequence_str), 100) == 0 else False
        VisualizeUtils.draw_latent_error(g_list, [koopman], ["DeepEDMD"], is_show=False, is_save=is_save, save_dir=save_path)
        self.latent_error.append(g_list - koopman)
        # print('the shape of x_koopman: ', x_Koopman.shape)
        U = np.transpose(np.squeeze(u_test_tensor[:, test_num, :]))
        if (np.ndim(U) == 1):
            U = U[np.newaxis, :]
        x_min = args['state_bound'][0, :][:, np.newaxis]
        x_max = args['state_bound'][1, :][:, np.newaxis]
        U_min = args['action_bound'][0, :][:, np.newaxis]
        U_max = args['action_bound'][1, :][:, np.newaxis]
        xmax_minus_xmin = np.tile((x_max - x_min), (1, self.predicted_step + 1))
        x_true = np.add(np.multiply(x_true, xmax_minus_xmin), x_min)
        xmax_minus_xmin = np.tile((x_max[-self.s_dim:, :] - x_min[-self.s_dim:, :]),
                                  (1, self.predicted_step + 1))
        x_Koopman = np.add(np.multiply(x_Koopman, xmax_minus_xmin), x_min[:self.s_dim, :])
        U[0, :] = U[0, :] * U_max[0, 0]
        U[1, :] = [x * U_max[1, 0] if x >= 0 else x * (U_max[2, 0]) for x in U[1, :]]
        return x_true, U, x_Koopman
        # save the picture result
        if not self.is_predict:
            images_dir = args['image_path'] % (str(passed_time), sequence_str)
        else:
            images_dir = args['test_image_path'] % sequence_str
        # print("x_true: ", x_true.shape, "U: ", U.shape, "x_Koopman: ", x_Koopman.shape)
        offset = helper.plot_X_and_Y(args, x_true[:, 1:], x_Koopman[:, 1:], U, is_reference=False, is_save=True,
                                     state_name=args['state_name'],
                                     action_name=args['action_name'], predict_step=self.predicted_step,
                                     save_dir=images_dir, is_show=False,
                                     title=args['domain_name'] + ' with no normalization')

        # save the neural network, params and losses
        if not self.is_predict:
            self.saver.save(self.sess, args['model_path'] % passed_time)
            args_dict = args
            args_dict.update(self.params)

            # save the params
            helper.save_dict_as_txt(args_dict, args['params_path'])
            helper.save_dict_as_mat(args_dict, args['params_path'].replace('.txt', '.mat'))

            # save each kind of loss with training and validation data
            self.loss_dict['losses'] = np.array(self.error_recorder)
            print('shape of error_recorder: ', np.array(self.error_recorder).shape)
            helper.save_dict_as_mat(self.loss_dict, args['loss_path'])
        return x_true, U, x_Koopman
        print('---------------- it has been saved ----------------')

    def get_Koopman_prediction(self, args, x0, U, steps=None):
        '''
        # This function is mainly for testing, and the U is sampled from testing or validation dataset
        get the predicted koopman sequence according to the original state x0 and the action sequence U
        :param x0: the original state, [1, s_dim]
        :param U: the action sequence , [predicted_step, u_dim]
        :return: return the predicted sequence
        '''
        if steps == None:
            steps = self.predicted_step
        y = []
        koopman = []
        encoder_weights_num = len(args['encoder_widths']) - 1
        decoder_weights_num = len(args['decoder_widths']) - 1
        encoded_layer = self.encoder_apply_one_shift(x0, self.weights, self.biases, args['eact_type'],
                                                     args['batch_flag'], self.phase, self.keep_prob, 'E',
                                                     encoder_weights_num)
        self.encoder_x0 = encoded_layer
        y.append(
            self.decoder_apply(encoded_layer, self.weights, self.biases, args['dact_type'], args['batch_flag'],
                               self.phase, self.keep_prob, decoder_weights_num))

        advanced_layer = self.koopman_net(args, self.weights, self.biases, encoded_layer, U[0, :][np.newaxis, :],
                                          self.keep_prob)
        koopman.append(advanced_layer)
        # multi-step prediction
        for i in np.arange(steps):
            y.append(self.decoder_apply(advanced_layer, self.weights, self.biases, args['dact_type'],
                                        args['batch_flag'], self.phase, self.keep_prob, decoder_weights_num))
            if (i + 1) < self.predicted_step:
                advanced_layer = self.koopman_net(args, self.weights, self.biases, advanced_layer,
                                                  U[i + 1, :][np.newaxis, :], self.keep_prob)
                koopman.append(advanced_layer)
        self.koopman = koopman
        return y

    def define_loss(self, args, x, y, g_list, weight, biases, phase, keep_prob):
        """Define the (unregularized) loss functions for the training.

            Arguments:
                x -- placeholder for input
                y -- list of outputs of network for each shift (each prediction step)
                g_list -- list of output of encoder for each shift (encoding each step in x)
                weights -- dictionary of weights for all networks
                biases -- dictionary of biases for all networks
                phase -- boolean placeholder for dropout: training phase or not training phase
                keep_prob -- probability that weight is kept during dropout

            Returns:
                loss1 -- autoencoder loss function
                loss2 -- dynamics/prediction loss function
                loss3 -- linearity loss function
                loss_Linf -- inf norm on autoencoder loss and one-step prediction loss
                loss -- sum of above four losses
        """
        # autoencoder loss: reconstruction loss
        y1 = []
        decoder_weights_num = len(args['decoder_widths']) - 1
        denominator_nonzero = 10 ** (-5)
        loss1_denominator = tf.to_float(1.0)
        mean_squared_error = tf.reduce_mean(
            tf.reduce_mean(tf.square((y[0] - tf.squeeze(x[0, :, -self.s_dim:])) * self.state_weight), 1))
        loss1 = tf.truediv(mean_squared_error, loss1_denominator)
        for i in np.arange(self.num_koopman_shifts - 1):
            temp_y = self.decoder_apply(g_list[i + 1], self.weights, self.biases, args['dact_type'],
                                        args['batch_flag'], self.phase, self.keep_prob, decoder_weights_num)
            loss1 = loss1 + tf.reduce_mean(
                tf.reduce_mean(tf.square((temp_y - tf.squeeze(x[i + 1, :, -self.s_dim:])) * self.state_weight), 1))
        loss1 = args['recon_lam'] * loss1 / self.num_koopman_shifts

        # gets predicted loss
        loss2_denominator = tf.to_float(1.0)
        loss2 = tf.zeros([1, ], dtype=tf.float32)
        for i in np.arange(self.num_shifts):
            shift = i + 1
            loss2 = loss2 + args['recon_lam'] * tf.truediv(
                tf.reduce_mean(tf.reduce_mean(tf.square((y[shift] - tf.squeeze(x[shift, :, -self.s_dim:])) * self.state_weight), 1)),
                loss2_denominator)
        loss2 = loss2 / self.num_shifts

        # linear loss
        loss3 = tf.zeros([1, ], dtype=tf.float32)
        count_shift = 0
        y_lift = self.koopman_net(args, self.weights, self.biases, g_list[0], self.u[0, :, :], self.keep_prob)
        for i in np.arange(self.num_koopman_shifts):
            loss3_denominator = tf.to_float(1.0)
            loss3 = loss3 + args['koopman_lam'] * tf.truediv(
                tf.reduce_mean(tf.reduce_mean(tf.square(y_lift - g_list[count_shift + 1]), 1)), loss3_denominator)
            count_shift += 1
            if (i + 1) < self.num_koopman_shifts:
                y_lift = self.koopman_net(args, self.weights, self.biases, y_lift, self.u[i + 1, :, :], self.keep_prob)
        loss3 = loss3 / self.num_koopman_shifts

        # 无穷范数
        Linf1_den = tf.to_float(1.0)
        Linf2_den = tf.to_float(1.0)
        Linf1_penalty = tf.truediv(
            tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, -self.s_dim:]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
        Linf2_penalty = tf.truediv(
            tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, -self.s_dim:]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
        loss_Linf = args['Linf_lam'] * (Linf1_penalty + Linf2_penalty)
        loss = loss1 + loss2 + loss3 + loss_Linf
        return loss1, loss2, loss3, loss_Linf, loss

    def define_regularization(self, args, trainable_var, loss, loss1):
        '''Define the regularization and add to loss.

            Arguments:
                trainable_var -- list of trainable TensorFlow variables
                loss -- the unregularized loss
                loss1 -- the autoenocder component of the loss

            Returns:
                loss_L1 -- L1 regularization on weights W and b
                loss_L2 -- L2 regularization on weights W
                regularized_loss -- loss + regularization
                regularized_loss1 -- loss1 (autoencoder loss) + regularization
        '''
        l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trainable_var if 'b' not in v.name])
        loss_L2 = args['L2_lam'] * l2_regularizer
        regularized_loss = loss + loss_L2
        regularized_loss1 = loss1 + loss_L2
        return loss_L2, regularized_loss, regularized_loss1

    def create_koopman_net(self, phase, keep_prob, args):
        '''
        Create a Koopman network, including encodes, decodes, matrix A, B and C.
        Y_lift = A X_lift + B U
        :param phase:
        :param keep_prob:
        '''
        # the number of layers of the encoder and decoder
        encoder_widths = args['encoder_widths']  # 这里的encoder_widths 只包括全连接层
        self.encoder_init(args, widths=encoder_widths, dist_weights=args['edist_weights'],
                          dist_biases=args['edist_biases'])
        num_encoder_weights = len(encoder_widths) - 1
        if args['input_mode'] == 0:
            g_list = self.encoder_apply(self.images, self.weights, self.biases, args['eact_type'], args['batch_flag'],
                                        phase=phase, keep_prob=keep_prob, shifts_middle=self.num_shifts,
                                        num_encoder_weights=num_encoder_weights)
        elif args['input_mode'] == 1:
            g_list = self.encoder_apply(self.x, self.weights, self.biases, args['eact_type'], args['batch_flag'],
                                        phase=phase,
                                        keep_prob=keep_prob, shifts_middle=self.num_shifts,
                                        num_encoder_weights=num_encoder_weights)
        # the net for action u
        uwidths = args['uwidths']
        self.koopman_net_init(args, args['uwidths'], args['udist_weights'], args['udist_biases'])
        koopman_net = self.koopman_net(args, self.weights, self.biases, g_list[0], self.u[0, :, :], self.keep_prob)
        decoder_widths = args['decoder_widths']
        self.decoder_init(decoder_widths, args['ddist_weights'], args['ddist_biases'], args['scale'])
        y = []
        z = []
        encoded_layer = g_list[0]
        decoder_weights_num = len(decoder_widths) - 1
        # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
        y.append(
            self.decoder_apply(encoded_layer, self.weights, self.biases, args['dact_type'], args['batch_flag'], phase,
                               keep_prob, decoder_weights_num))
        advanced_layer = self.koopman_net(args, self.weights, self.biases, encoded_layer, self.u[0, :, :],
                                          self.keep_prob)
        # multi-step prediction
        for i in np.arange(self.num_shifts):
            y.append(
                self.decoder_apply(advanced_layer, self.weights, self.biases, args['dact_type'], args['batch_flag'],
                                   phase, keep_prob, decoder_weights_num))
            if (i + 1) < self.num_shifts:
                advanced_layer = self.koopman_net(args, self.weights, self.biases, advanced_layer, self.u[i + 1, :, :],
                                                  self.keep_prob)
        if len(y) != self.num_shifts + 1:
            print("messed up looping over shifts! %r" % self.num_shifts)
            raise ValueError(
                'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')
        self.y = y
        self.g_list = g_list
        return self.x, self.y, self.g_list, self.weights, self.biases

    def encoder_init(self, args, widths, dist_weights, dist_biases):
        '''
        initialize an encoder including the weights and bias of the lifting network, meanwhile create a placeholder of the input
        :param widths: neural numbers of each layer
        :param dist_weights: list; the method for initializing the weights of each layers
        :return: dist_biases: list; teh method for initializing the biases of each layers
            weights: the dictionary of weights
            biases: the dictionary of biases
            x: a placeholder for a input
            +
        '''
        conca = self.conca
        self.x = tf.placeholder(tf.float32, [self.num_shifts + 1, None, self.s_dim * (self.conca + 1)])
        image_height, image_width = self.image_height, self.image_width
        self.images = tf.placeholder(tf.float32, [self.num_shifts + 1, None, image_height, image_width, conca + 1])
        num_cnn_layers = 3
        num_fc_layers = 2
        scale = args['scale']
        weights = dict()
        biases = dict()
        # 初始化卷积层权重和偏置
        image_size = [image_height, image_width, conca + 1]
        ## conv 和 pool 的padding 方式默认都采用 'SAME'
        # 格式：[[k_size1, strides1, in_channel, out_channel], [k_size2, strides2, in_channel, out_channel], ...]
        conv_size = args['conv']
        for i in np.arange(len(conv_size)):
            conv = conv_size[i]
            k_size, strides, in_channel, out_channel = conv[0], conv[1], conv[2], conv[3]
            weights['WEC%d' % (i + 1)] = self.weight_variable([k_size, k_size, in_channel, out_channel],
                                                              var_name='WEC%d' % (i + 1), distribution=dist_weights[i],
                                                              scale=scale)
            biases['bEC%d' % (i + 1)] = self.bias_variable([out_channel, ], var_name='bEC%d' % (i + 1), distribution='')
        # 初始化全连接层权重和偏置
        widths = args['encoder_widths']
        # 最后一层没有激活函数和集团
        trainable = not self.train_AB_only
        for i in np.arange(len(widths) - 1):
            weights['WEF%d' % (i + 1)] = self.weight_variable([widths[i], widths[i + 1]], var_name='WEF%d' % (i + 1),
                                                              distribution=dist_weights[i + len(conv_size)],
                                                              scale=args['scale'], trainable=trainable)
            if (i < len(widths) - 2):
                biases['bEF%d' % (i + 1)] = self.bias_variable([widths[i + 1], ], var_name='bEF%d' % (i + 1),
                                                               distribution=dist_biases[i], trainable=trainable)
        self.weights = weights
        self.biases = biases
        return weights, biases

    def encoder_apply(self, x, weights, biases, act_type, batch_flag, phase, shifts_middle, keep_prob, name='E',
                      num_encoder_weights=1):
        """Apply an encoder to data x.

        Arguments:
            :param images -- placeholder for input, format: [self.num_shifts + 1, None, image_height, image_width, conca + 1]
            :param weights -- dictionary of weights
            :param biases -- dictionary of biases
            :param act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            :param batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
            :param phase -- boolean placeholder for dropout: training phase or not training phase
            :param shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss
            :param keep_prob -- probability that weight is kept during dropout
            :param name -- string for prefix on weight matrices (default 'E' for encoder)
            :param num_encoder_weights -- number of weight matrices (layers) in encoder network (default 1)

        Returns:
            y -- list, output of encoder network applied to each time shift in input x

        Side effects:
            None
        """
        y = []
        num_shifts_middle = shifts_middle
        for j in np.arange(num_shifts_middle + 1):
            if self.args['input_mode'] == 0:
                # format: [self.num_shifts + 1, None, image_height, image_width, conca + 1]
                x_shift = x[j, :, :, :]
            elif self.args['input_mode'] == 1:
                shift = j
                if isinstance(x, (list,)):
                    x_shift = x[shift]
                else:
                    if x.get_shape()[1] > 1:
                        x_shift = tf.squeeze(x[shift, :, :])
                    else:
                        x_shift = x[shift, :, :]
            # 对每一次shift的一个batch的数据都经过一下编码网络，即得到第j步时所有的状态的观测值\psi(x_j)
            y.append(
                self.encoder_apply_one_shift(x_shift, weights, biases, act_type, batch_flag, phase, keep_prob, name,
                                             num_encoder_weights))
        return y

    def encoder_apply_one_shift(self, prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, name='E',
                                num_encoder_weights=1):
        """Apply an encoder to data for only one time step (shift).

        Arguments:
            prev_layer -- images input, [None, image_height, image_width, conca+1]
            weights -- dictionary of weights
            biases -- dictionary of biases
            act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
            phase -- boolean placeholder for dropout: training phase or not training phase
            keep_prob -- probability that weight is kept during dropout
            name -- string for prefix on weight matrices (default 'E' for encoder)
            num_encoder_weights -- 全连接层的层数，不包括卷积层

        Returns:
            final -- output of encoder network applied to input prev_layer (a particular time step / shift)
        """
        if self.args['input_mode'] == 0:
            for i in np.arange(len(self.args['conv'])):
                prev_layer = tf.nn.relu(
                    self.conv2d(prev_layer, weights['WEC%d' % (i + 1)], stride=2) + biases['bEC%d' % (i + 1)])
                prev_layer = self.max_pool_2x2(prev_layer)
            # 将卷积层拉直
            prev_layer = tf.layers.flatten(prev_layer)
        temp_input1 = prev_layer[:, -self.s_dim:]
        for i in np.arange(num_encoder_weights):
            if (i < num_encoder_weights - 1):
                if i == self.random_weight_layer - 1:
                    # print('========================random weight================================')
                    # temp_weight = tf.random.uniform([args['encoder_widths'][i], arga['encoder_widths'][i + 1]], -1., 1.,
                    #                                 dtype=tf.float32)
                    temp_weight = tf.random.normal([args['encoder_widths'][i], args['encoder_widths'][i + 1]],
                                                   dtype=tf.float32)
                    temp_bias = tf.random.normal([args['encoder_widths'][i + 1]], dtype=tf.float32)
                    h1 = tf.matmul(prev_layer, temp_weight) + temp_bias
                else:
                    h1 = tf.matmul(prev_layer, weights['WEF%d' % (i + 1)]) + biases['bEF%d' % (i + 1)]
                if batch_flag:
                    h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
            else:
                if i == self.random_weight_layer - 1:
                    # print('========================random weight================================')
                    # temp_weight = tf.random.uniform([args['encoder_widths'][i], args['encoder_widths'][i + 1]], -1., 1.,
                    #                                 dtype=tf.float32)
                    temp_weight = tf.random.normal([args['encoder_widths'][i], args['encoder_widths'][i + 1]],
                                                   dtype=tf.float32)
                    h1 = tf.matmul(prev_layer, temp_weight)
                else:
                    h1 = tf.matmul(prev_layer, weights['WEF%d' % (i + 1)])
            if (i < num_encoder_weights - 1):
                if act_type[i] == 'sigmoid':
                    h1 = tf.sigmoid(h1)
                elif act_type[i] == 'relu':
                    h1 = tf.nn.relu(h1)
                elif act_type[i] == 'elu':
                    h1 = tf.nn.elu(h1)
                elif act_type[i] == 'sin':
                    h1 = tf.sin(h1)
                elif act_type[i] == 'cos':
                    h1 = tf.cos(h1)
                elif act_type[i] == 'tanh':
                    h1 = tf.tanh(h1)
                prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)
            else:
                prev_layer = h1
        prev_layer = tf.concat((temp_input1, prev_layer), axis=1)
        return prev_layer

    def koopman_net_init(self, args, uwidths, udist_weights, udist_biases):
        '''
        initialize the koopman net, and return the relative weights
        Arguments:
        :param uwidths: the number of each layer of the neural network
        :param udist_weights: the type to randomly generate the weights
        :param udist_biases: the type to randomly generate the biases
        Returns:
        weights -- dictionary of weights
        biases -- dictionary of biases
        '''
        self.u = tf.placeholder(tf.float32, [self.num_shifts, None, self.u_dim], name='U')
        self.t = tf.placeholder(tf.float32, [None, 1], name='T')
        weights = dict()
        biases = dict()
        weight_K = self.weight_variable([self.lift_dim + self.s_dim, self.lift_dim + self.s_dim], 'WK',
                                        distribution='dl', scale=args['scale'])
        weights['WK'] = weight_K
        weights['WU'] = self.weight_variable([self.u_dim, self.lift_dim + self.s_dim], var_name='WU', distribution='dl',
                                             scale=args['uscale'])
        self.weights.update(weights)
        self.biases.update(biases)
        return weights

    def koopman_net(self, args, weights, biases, x_lift, u, keep_prob):
        '''
        :param x_lift: the states after lifting with lift_net
        :param u: the action
        :return: return the lifted state of next sample time
        '''
        y_lift = tf.matmul(x_lift, weights['WK']) + tf.matmul(u, weights['WU'])
        return y_lift

    def decoder_init(self, widths, dist_weights, dist_biases, scale, name='D', first_guess=0):
        """Create a decoder network: a dictionary of weights and a dictionary of biases.

        Arguments:
            widths -- array or list of widths for layers of network
            dist_weights -- array or list of strings for distributions of weight matrices
            dist_biases -- array or list of strings for distributions of bias vectors
            scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
            name -- string for prefix on weight matrices (default 'D' for decoder)
            first_guess -- (for tn dist. of weight matrices): array of first guess for weight matrix, added to tn dist.
                (default 0)

        Returns:
            weights -- dictionary of weights
            biases -- dictionary of biases

        Side effects:
            None
        """
        weights = dict()
        biases = dict()
        trainable = not self.train_AB_only
        for i in np.arange(len(widths) - 1):
            ind = i + 1
            weights['W%s%d' % (name, ind)] = self.weight_variable([widths[i], widths[i + 1]],
                                                                  var_name='W%s%d' % (name, ind),
                                                                  distribution=dist_weights[i], scale=scale,
                                                                  first_guess=first_guess, trainable=trainable)
            biases['b%s%d' % (name, ind)] = self.bias_variable([widths[i + 1], ], var_name='b%s%d' % (name, ind),
                                                               trainable=trainable)
        self.weights.update(weights)
        self.biases.update(biases)
        return weights, biases

    def decoder_apply(self, prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, num_decoder_weights):
        """Apply a decoder to data prev_layer

        Arguments:
            prev_layer -- input to decoder network
            weights -- dictionary of weights
            biases -- dictionary of biases
            act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
            batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
            phase -- boolean placeholder for dropout: training phase or not training phase
            keep_prob -- probability that weight is kept during dropout
            num_decoder_weights -- number of weight matrices (layers) in decoder network

        Returns:
            output of decoder network applied to input prev_layer

        Side effects:
            None
        """
        num_decoder_weights = int(num_decoder_weights)
        for i in np.arange(num_decoder_weights):
            # if (i < num_decoder_weights - 1):
            h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
            # else:
            #     h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)])
            if batch_flag:
                h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
            if act_type[i] == 'sigmoid':
                h1 = tf.sigmoid(h1)
            elif act_type[i] == 'relu':
                h1 = tf.nn.relu(h1)
            elif act_type[i] == 'elu':
                h1 = tf.nn.elu(h1)
            elif act_type[i] == 'sin':
                h1 = tf.sin(h1)
            elif act_type[i] == 'cos':
                h1 = tf.cos(h1)
            elif act_type[i] == 'tanh':
                h1 = tf.tanh(h1)
            # the last layer don't need the dropout op
            if (i < num_decoder_weights - 1):
                prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)
            else:
                prev_layer = h1
        # apply last layer without any nonlinearity
        # last_layer = tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]
        return prev_layer

    def restore_model(self, args):
        # restore_model_path = cur_path + '/model/CartPole_2019_09_17_11_52/min_model.ckpt'
        restore_model_path = args['restore_model_path']
        if os.path.exists(restore_model_path + '.meta'):
            self.saver.restore(self.sess, restore_model_path)
            print('======================= restore model successfully !!! ====================')
            return True
        else:
            print('======================== The model does not exist !!! ======================')
            return False

    def weight_variable(self, shape, var_name, distribution='tn', scale=0.1, first_guess=0, trainable=True):
        """Create a variable for a weight matrix.

        Arguments:
            shape -- array giving shape of output weight variable
            var_name -- string naming weight variable
            distribution -- string for which distribution to use for random initialization (default 'tn')
            scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)
            first_guess -- (for tn distribution): array of first guess for weight matrix, added to tn dist. (default 0)

        Returns:
            a TensorFlow variable for a weight matrix

        Side effects:
            None

        Raises ValueError if distribution is filename but shape of data in file does not match input shape
        """
        if distribution == 'tn':
            initial = tf.truncated_normal(shape, stddev=scale, dtype=tf.float32)
        elif distribution == 'xavier':
            scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
            initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
        elif distribution == 'dl':
            # see page 295 of Goodfellow et al's DL book
            # divide by sqrt of m, where m is number of inputs
            scale = 1.0 / np.sqrt(shape[0])
            initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
        elif distribution == 'he':
            # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
            # divide by m, where m is number of inputs
            scale = np.sqrt(2.0 / shape[0])
            initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float32)
        elif distribution == 'glorot_bengio':
            # see page 295 of Goodfellow et al's DL book
            scale = np.sqrt(6.0 / (shape[0] + shape[1]))
            initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
        else:
            initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
            if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
                raise ValueError(
                    'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                        var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
        return tf.Variable(initial, name=var_name, trainable=trainable)

    def bias_variable(self, shape, var_name, distribution='', trainable=True):
        """Create a variable for a bias vector.

        Arguments:
            shape -- array giving shape of output bias variable
            var_name -- string naming bias variable
            distribution -- string for which distribution to use for random initialization (file name) (default '')

        Returns:
            a TensorFlow variable for a bias vector

        Side effects:
            None
        """
        if distribution:
            initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float32)
        else:
            initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=var_name, trainable=trainable)

    def conv2d(self, x, W, stride=1, padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    def max_pool_2x2(self, x, k_size=2, strides=2):
        return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, strides, strides, 1], padding='SAME')

    def get_train_test_val(self, args, which_data, path=None):
        '''
        get the validation data or testing data
        :param which_data: {'val', 'test', 'train'}, 'val': return validation data, 'test': return testing data, 'train': return train data
        :return: return the corresponding data
        '''
        input_mode = args['input_mode']
        if 'test' in which_data:
            num_shift = self.predicted_step
        else:
            num_shift = self.num_shifts
        if path is None:
            raise Exception('You have to tell the path of train data file')
        file_type = ''.join(path.split('.')[-1])
        if ('h5' in file_type):
            data_dict = helper.read_h5(path)
        elif ('pkl' in file_type):
            data_dict = helper.read_pkl_as_dict(path)
        conca = self.conca
        interval = self.interval
        if 'CarSim' in args['domain_name']:
            delta_tensor, state_tensor, u_tensor = helper.carsim_data_process(data_dict, num_shift, conca, interval, mode=1)
            return delta_tensor, state_tensor, u_tensor
        else:
            image_tensor, state_tensor, u_tensor = helper.process_data(data_dict, num_shift, conca, interval,
                                                                       mode=input_mode)
            return image_tensor, state_tensor, u_tensor

    def save_weight_txt(self, pass_time=0):
        weight = dict()
        bias = dict()
        for key in self.weights.keys():
            weight[key] = self.sess.run(self.weights[key])
        for key in self.biases.keys():
            bias[key] = self.sess.run(self.biases[key])
        if not self.is_predict:
            weight_path = ''.join(os.path.split(self.args['model_path'])[:-1]) + '/weights_%dmin.txt' % pass_time
            bias_path = ''.join(os.path.split(self.args['model_path'])[:-1]) + '/bias_%dmin.txt' % pass_time
        else:
            weight_path = ''.join(
                os.path.split(self.args['restore_model_path'])[:-1]) + '/weights_%dmin.txt' % pass_time
            bias_path = ''.join(os.path.split(self.args['restore_model_path'])[:-1]) + '/bias_%dmin.txt' % pass_time
        helper.save_dict_as_txt(weight, weight_path)
        helper.save_dict_as_txt(bias, bias_path)

    def get_training_params(self):
        params = dict()
        # settings related to timing
        params['max_time'] = 10.1 * 60 * 60  # 6 hours
        params['been3min'] = 0
        params['been5min'] = 0
        params['been20min'] = 0
        params['been40min'] = 0
        params['been1hr'] = 0
        params['been2hr'] = 0
        params['been3hr'] = 0
        params['been4hr'] = 0
        params['been5hr'] = 0
        params['been6hr'] = 0
        params['been7hr'] = 0
        params['been8hr'] = 0
        params['been9hr'] = 0
        params['been10hr'] = 0
        params['lastest'] = 0
        # 记录用过了的训练样本数
        params['samples3min'] = 0
        params['samples5min'] = 0
        params['samples20min'] = 0
        params['samples40min'] = 0
        params['samples1hr'] = 0
        params['samples2hr'] = 0
        params['samples3hr'] = 0
        params['samples4hr'] = 0
        params['samples5hr'] = 0
        params['samples6hr'] = 0
        params['samples7hr'] = 0
        params['samples8hr'] = 0
        params['samples9hr'] = 0
        params['samples10hr'] = 0
        params['current_samples'] = 0
        params['lastest_samples'] = 0
        params['loss_name'] = ['train_error', 'val_error', 'train_autoencoder_loss',
                               'train_regularized_loss', 'val_regularized_loss',
                               'val_autoencoder_loss', 'train_reconstruction_loss',
                               'val_reconstruction_loss', 'train_koopman_operator_loss',
                               'val_koopman_operator_loss', 'train_inf_loss', 'val_inf_loss',
                               'train_L2_loss', 'val_L2_loss']
        # 记录用过了的训练样本数
        params['train_times'] = 0
        return params

    def save_parameters_for_matlab(self, args):
        '''
        save the weights, biases, and activations to rebuild the network with MATLAB
        :return: None
        '''
        save_path = args['matlab_file_path']
        if os.path.exists(save_path):
            print('================= params_for_matlab.mat has existed !!!! ======================')
            return
        nn = dict()
        nn['weights'] = self.sess.run(self.weights)
        nn['biases'] = self.sess.run(self.biases)
        nn['encoder_widths'] = args['encoder_widths']
        nn['decoder_widths'] = args['decoder_widths']
        nn['eact_type'] = args['eact_type']
        nn['dact_type'] = args['dact_type']
        nn['s_dim'] = self.s_dim
        nn['u_dim'] = self.u_dim
        nn['conca_num'] = self.conca
        nn['lift_dim'] = self.lift_dim
        nn['state_bound'] = args['state_bound']
        nn['action_bound'] = args['action_bound']
        nn['num_shifts'] = self.num_shifts
        nn['num_koopman_shifts'] = self.num_koopman_shifts
        nn['is_random_train'] = args['is_random_train']
        nn['random_weight_layer'] = args['random_weight_layer']
        helper.save_dict_as_mat(nn, save_path)
        helper.save_dict_as_pkl(nn, save_path.replace('.mat', '.pkl'))
        print('====== save the file to %s successfully =======' % save_path)


if __name__ == '__main__':
    # --------------- 先采数据 --------------
    # helper.sample_gym_data(10, start=0, min_steps=55, epochs=1000)
    # helper.sample_gym_data(1, start=50, min_steps=85, epochs=100)
    # #  -------------- 对数据文件进行分割 -----------------
    # path = '/home/BackUp/cartpole_data/CartPole_1000_55_traj_%d.pkl'
    # path = '/home/SELF/Acrobot_data/Acrobot_100_100_2001_traj_%d.pkl

    # path = '/media/buduo/Data/Acrobot_Data/Acrobot_500_250_251_traj_%d.pkl'
    # for i in np.arange(10):
    #     path_ = path % (i)
    #     helper.split_pkl_file(path_, 10)
    # helper.acrobot_data_process()
    # args = helper.get_con_mountaincar_args()
    # args = helper.get_cartpole_args()
    args = helper.get_carsim_args_local_v()
    args_path = args.args_file_path
    restore_model_path = args.restore_model_path
    test_image_path = args.test_image_path
    test_data_path = args.test_data_path
    matlab_file_path = args.matlab_file_path
    is_predict = args.is_predict
    predicted_step = args.predicted_step
    if os.path.exists(args_path) and os.path.exists(restore_model_path + '.meta'):
        print('=================== args file exist !=====================')
        args = helper.read_pkl_as_dict(args_path)
        args['is_predict'] = is_predict
        args['predicted_step'] = predicted_step
        args['restore_model_path'] = restore_model_path
        args['test_image_path'] = test_image_path
        args['test_data_path'] = test_data_path
        args['args_file_path'] = args_path
        args['matlab_file_path'] = matlab_file_path
    else:
        print('=================== args file does not exist ===============')
        args = vars(args)
    # args = helper.args_update(args, path)
    # args = helper.get_acrobot_args(input_mode=0)
    # args = helper.get_cartpolev2_args()

    DeepEDMDwithPixel(args=args)
    # args = helper.get_acrobot_args(input_mode=0)
    # DeepEDMDwithPixel(args=args)
