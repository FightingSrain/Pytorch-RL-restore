#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import h5py as h5
import cv2
import os
from utils import Cal_psnr, step_psnr_reward, load_imgs, data_reformat

class Env(object):
    def __init__(self, train_dir, vali_dir, test_dir, dataset):
        self.reward = 0
        self.terminal = True
        self.stop_step = 3
        self.reward_func = 'step_psnr_reward'
        self.is_train = True
        self.learn_start = 1000
        self.count = 0 # 恢复步数
        self.psnr = 0.
        self.psnr_pre = 0.
        self.psnr_init = 0.
        self.train_dir = train_dir
        self.vali_dir = vali_dir
        self.test_dir = test_dir
        self.dataset = dataset
        if self.is_train:
            self.train_list = []
            for file in os.listdir(self.train_dir):
                if file.endswith('.h5'):
                    self.train_list.append(self.train_dir+file)

            self.train_cur = 0
            self.train_max = len(self.train_list)

            # 训练集
            train_file = h5.File(self.train_list[self.train_cur], "r")
            self.data = train_file["data"].value # (2752, 63, 63, 3)
            self.label = train_file["label"].value # (2752, 63, 63, 3)
            train_file.close()

            self.data_index = 0
            self.data_len = len(self.data)

            # 验证集
            vali_file = h5.File(self.vali_dir + os.listdir(self.vali_dir)[0], "r")
            self.data_test = vali_file["data"].value
            self.label_test = vali_file["label"].value
            vali_file.close()

            self.data_all = self.data_test
            self.label_all = self.label_test

        else:
            if self.dataset == "mine":
                self.my_img_dir = self.test_dir + "mine/"
                self.my_img_list = os.listdir(self.my_img_dir)
                self.my_img_list.sort()
                self.my_img_idx = 0
            elif self.dataset in ["mild", "moderate", "severe"]:
                self.test_batch = self.test_batch
                self.test_in = self.test_dir + self.dataset + "_in/"
                self.test_gt = self.test_dir + self.dataset + "_gt/"
                list_in = [self.test_in + name for name in os.listdir(self.test_in)]
                list_in.sort()
                list_gt = [self.test_gt + name for name in os.listdir(self.test_gt)]
                list_gt.sort()

                self.name_list = [os.path.splitext(os.path.basename(file))[0] for file in list_in]
                self.data_all, self.label_all = load_imgs(list_in, list_gt)
                self.test_total = len(list_in)
                self.test_cur = 0

                self.data_all = data_reformat(self.data_all)
                self.label_all = data_reformat(self.label_all)
                self.data_test = self.data_all[0: min(self.test_batch, self.test_total), ...]
                self.label_test = self.label_all[0: min(self.test_batch, self.test_total), ...]

            else:
                raise ValueError("数据集无效")

        if self.is_train or self.dataset!='mine':
            # input PSNR
            self.base_psnr = 0.
            for k in range(len(self.data_test)):
                self.base_psnr += Cal_psnr(self.data_test[k, ...], self.label_test[k, ...])
            self.base_psnr /= len(self.data_test)

            # reward functions
            self.rewards = {'step_psnr_reward': step_psnr_reward}
            self.reward_function = self.rewards[self.reward_func]

        self.action_size = 13 # 12 + 1
        toolbox_path = "toolbox/"
        self.graphs = []
        self.sessions = []
        self.inputs = []
        self.outputs = []

        for idx in range(12):
            g = tf.Graph()
            with g.as_default():
                saver = tf.train.import_meta_graph(toolbox_path+'tool%02d' %(idx + 1)+".meta")

                input_data = g.get_tensor_by_name("Placeholder:0")
                self.inputs.append(input_data)

                output_data = g.get_tensor_by_name("sum:0")
                self.outputs.append(output_data)

                self.graphs.append(g)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
            sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=True))
            with g.as_default():
                with sess.as_default():
                    saver.restore(sess, toolbox_path + "tool%02d" % (idx + 1))
                    self.sessions.append(sess)


    def new_image(self):
        self.terminal = False

        while self.data_index < self.data_len:
            # 从数据集中取出一张图片
            self.img = self.data[self.data_index: self.data_index+1, ...]
            self.img_gt = self.label[self.data_index: self.data_index+1, ...]
            self.psnr = Cal_psnr(self.img, self.img_gt)

            # 太平滑的图片跳过
            if self.psnr > 50:
                self.data_index += 1
            else:
                break

        # 更新训练文件(如果有多个训练文件)
        if self.data_index >= self.data_len:
            if self.train_cur > 1:
                self.train_cur += 1
                if self.train_cur >= self.train_max:
                    self.train_cur = 0


                print("loading file No. %d" % (self.train_cur + 1))

                train_file = h5.File(self.train_list[self.train_cur], "r")
                self.data = train_file["data"].value  # (2752, 63, 63, 3)
                self.label = train_file["label"].value  # (2752, 63, 63, 3)
                self.data_len = len(self.data)
                train_file.close()

            self.data_index = 0

            while True:
                self.img = self.data[self.data_index: self.data_index+1, ...]
                self.img_gt = self.label[self.data_index: self.data_index+1, ...]
                self.psnr = Cal_psnr(self.img, self.img_gt)
                if self.psnr > 50:
                    self.data_index += 1
                else:
                    break

        self.reward = 0
        self.count = 0
        self.psnr_init = self.psnr
        self.data_index += 1

        return self.img, self.reward, 0, self.terminal

    def step(self, action):
        self.psnr_pre = self.psnr
        if action == self.action_size - 1:
            self.terminal = True
        else:
            feed_dict = {self.inputs[action]: self.img}
            with self.graphs[action].as_default():
                with self.sessions[action].as_default():
                    im_out = self.sessions[action].run(self.outputs[action], feed_dict=feed_dict)
            self.img = im_out
        self.psnr = Cal_psnr(self.img, self.img_gt)

        if self.count >= self.stop_step - 1:
            self.terminal = True

        self.reward = self.reward_function(self.psnr, self.psnr_pre)
        self.count += 1
        return self.img, self.reward, self.terminal

    def step_test(self, action, step=0):
        reward_all = np.zeros(action.shape)
        psnr_all = np.zeros(action.shape)

        if step == 0:
            self.test_imgs = self.data_test.copy()
            self.test_temp_imgs = self.data_test.copy()
            self.test_pre_imgs = self.data_test.copy()
            self.test_steps = np.zeros(len(action), dtype=int)

        for k in range(len(action)):
            img_in = self.data_test[k:k+1,...].copy() if step==0 else self.test_imgs[k:k+1,...].copy()
            img_label = self.label_test[k:k+1,...].copy()

            self.test_temp_imgs[k:k+1,...].copy()

            psnr_pre = Cal_psnr(img_in, img_label)
            if action[k][0] == self.action_size- 1 or self.test_steps[k] == self.stop_step:
                img_out = img_in.copy()
                self.test_steps[k] = self.stop_step
            else:
                feed_dict = {self.inputs[action[k][0]]: img_in}
                with self.graphs[action[k][0]].as_default():
                    with self.sessions[action[k][0]].as_default():
                        with tf.device("/gpu:0"):
                            img_out = self.sessions[action[k][0]].run(self.outputs[action[k][0]], feed_dict=feed_dict)
                self.test_steps[k] += 1
            self.test_pre_imgs[k:k+1,...] = self.test_temp_imgs[k:k+1,...].copy()
            self.test_imgs[k:k+1,...] = img_out.copy()

            psnr = Cal_psnr(img_out, img_label)
            reward = self.reward_function(psnr, psnr_pre=psnr_pre)
            psnr_all[k] = psnr
            reward_all[k] = reward

        return reward_all.mean(), psnr_all.mean(), self.base_psnr, reward_all


    def get_test_imgs(self):
        return self.test_imgs.copy()


    def get_test_steps(self):
        return self.test_steps.copy()


    def get_data_test(self):
        return self.data_test.copy()


    def get_test_info(self):
        return self.test_cur, len(self.data_test)




















