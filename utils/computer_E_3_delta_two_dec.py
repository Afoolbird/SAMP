
import os,sys
from .model.models import SAFA_delta

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tf_slim as slim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .dataloader.dataloader import DataLoader
import random
from PIL import Image
import utils as ut
np.set_printoptions(threshold=np.inf)

#root='./data/VIGOR'

# delta generation and RGradCAM visualization from https://github.com/Jeff-Zilence/Explain_Metric_Learning
class ExplanationGenerator:

    def __init__(self, mode='CVUSA',delta=False,path=None):
        self.mode = mode
        self.size_sat = [320, 320]
        self.size_grd = [320, 640]
        self.ori_size_sat = [640,640]
        self.ori_size_grd = [1024,2048]
        #self.load_model(delta=delta,path=path)
        self.load_model_heatmap(path=path)
        self.Decomposition = None


##############################################################################################################################

    def load_model_heatmap(self, path = None):

        self.sat_x = tf.placeholder(tf.float32, [None, self.size_sat[0], self.size_sat[1], 3], name='sat_x')
        self.grd_x = tf.placeholder(tf.float32, [None, self.size_grd[0], self.size_grd[1], 3], name='grd_x')
        self.sat_x_semi = tf.placeholder(tf.float32, [None, self.size_sat[0], self.size_sat[1], 3], name='sat_x_semi')  
        # self.delta_2=tf.placeholder(tf.float32,[None, 2],name='delta_2')
        # self.keep_prob = tf.placeholder(tf.float32)
        # self.index = tf.placeholder(tf.float32, [4096], name='grd_x')

        self.sat_global, self.sat_global_semi, self.grd_global, self.sat_return, self.grd_return, self.heatmap,self.heatmap_semi,self.L,self.L_semi= SAFA_delta(self.sat_x, self.sat_x_semi, self.grd_x, out_dim= 2048)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # rename_dict = {
        # 'L_semi': 'new_name_for_L_semi'
        # }
        # variables_to_restore=['vgg_sat','vgg_grd','spatial_grd','spatial_sat','train','Variable','decoder','L','L_semi',]
        # variables_to_restore = slim.get_variables_to_restore(include=variables_to_restore)
        # saver = tf.train.Saver({rename_dict.get(var.op.name, var.op.name): var for var in variables_to_restore})

        #saver = tf.train.Saver(variables_to_restore)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
        print('load model...')
        saver.restore(self.sess, path)
        print("Model loaded from: %s" % path)

##############################################################################################################################


    #Calculate uncertainty values for positive and semi-positive samples
    def get_pre_gt(self,name1='',name2='', batch_size=10,index_semi=None,mode='train_SAFA_same_continuous_delta'):

        data_loader=DataLoader(mode=mode, same_area=True)
        # batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch(10)
        if os.path.exists(name1):
            return name1,name2
         
        else:
            delta_dic = {}
            delta_semi_dic={}
            delta_semi=ut.get_uncertainty(self.L_semi)
            delta=ut.get_uncertainty(self.L)
            break_iter = int(data_loader.test_data_size / batch_size)  ################    train   test
            print(break_iter)
            iter=0
            index=0
            # print(pre_gt.shape)
            while True:
                batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch_two_dec_test(batch_size,index_semi=index_semi)   #   get_next_batch_two_dec_test    get_next_batch_two_dec
                if iter==break_iter:  #data_loader.img_idx==data_loader.train_data_size-1
                    break
                
                # if batch_list[0] % 100 == 0:
                #     print(batch_list[0])
                
                feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
                                }
                delta_semi_val = self.sess.run(delta_semi,feed_dict=feed_dict)
                # feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
                #                 }
                delta_val = self.sess.run(delta,feed_dict=feed_dict)
                for i in range(batch_size):
                    delta_dic[batch_list[i]]=delta_val[i]
                    delta_semi_dic[batch_list[i]]=delta_semi_val[i]
                    index+=1
                    print(index)

                # print(iter)
                iter+=1

            np.save(name1,delta_semi_dic)
            np.save(name2,delta_dic)
            return name1,name2

    #Calculate the regression error for semi-positive sample 
    def get_pre_gt_uncer_error(self,name1='', batch_size=10,index_semi=None,mode='train_SAFA_same_continuous_delta'):

        data_loader=DataLoader(mode=mode, same_area=True)
        # batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch(10)
        if os.path.exists(name1):
            return name1
         
       
        # delta_dic = {}
        # delta_semi_dic={}
        error_dic={}

        # delta_semi=ut.get_uncertainty(self.L_semi)
        # delta=ut.get_uncertainty(self.L)
        pixel_x_semi,pixel_y_semi=ut.get_spatial_mean(self.heatmap_semi)
        pixel_x,pixel_y=ut.get_spatial_mean(self.heatmap)
        break_iter = int(data_loader.train_data_size / batch_size)
        # print(break_iter)
        iter=0
        index=0
        # feated_dis=feature_distance(self.sat_global)
        # print(pre_gt.shape)
        while True:
            batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch_two_dec_test(batch_size,index_semi=index_semi)
            if iter==break_iter:  #data_loader.img_idx==data_loader.train_data_size-1
                break
            
            # if batch_list[0] % 100 == 0:
            #     print(batch_list[0])
            
            feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
                            }
            semi_pixel_x_, semi_pixel_y_ = self.sess.run([pixel_x_semi,pixel_y_semi], feed_dict=feed_dict)
            # feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
            #                 }
            # delta_val = self.sess.run(delta,feed_dict=feed_dict)
            for i in range(batch_size):
                try:
                    pre_sat_name_delta=data_loader.test_delta[batch_list[i]][index_semi].astype(float)
                except IndexError:
                    break
                x=pre_sat_name_delta[0]
                y=pre_sat_name_delta[1]
                gt_x=x+320.0
                gt_y=319.0-y
                error=np.sqrt(np.square(gt_x-semi_pixel_x_[i])+np.square(gt_y-semi_pixel_y_[i]))

                # delta_dic[batch_list[i]]=delta_val[i]
                # delta_semi_dic[batch_list[i]]=delta_semi_val[i]
                error_dic[batch_list[i]]=error
                # pre_gt_dic[batch_list[i]]=pre_gt_val[i]
                # delta_dic[batch_list[i]]=delta_val[i]
                # print(batch_list[i])
                index+=1
                print(index)

            # print(iter)
            iter+=1

        np.save(name1,error_dic)
        # np.save(name2,delta_dic)
        # np.save(name3,data_loader.semi_inx)
        return name1
        
    #Calculate the regression error for positive sample 
    def get_pre_gt_uncer_error_pos(self,name1='', batch_size=10,index_semi=1,mode='train_SAFA_same_continuous_delta'):

            data_loader=DataLoader(mode=mode, same_area=True)
            # batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch(10)
            if os.path.exists(name1):
                return name1
            
        
            # delta_dic = {}
            # delta_semi_dic={}
            error_dic={}

            # delta_semi=ut.get_uncertainty(self.L_semi)
            # delta=ut.get_uncertainty(self.L)
            pixel_x_semi,pixel_y_semi=ut.get_spatial_mean(self.heatmap_semi)
            pixel_x,pixel_y=ut.get_spatial_mean(self.heatmap)
            break_iter = int(data_loader.train_data_size / batch_size)
            # print(break_iter)
            iter=0
            index=0
            # feated_dis=feature_distance(self.sat_global)
            # print(pre_gt.shape)
            while True:
                batch_sat, batch_grd, batch_list, delta_list = data_loader.get_next_batch_two_dec_test(batch_size,index_semi=index_semi)
                if iter==break_iter:  #data_loader.img_idx==data_loader.train_data_size-1
                    break
                
                # if batch_list[0] % 100 == 0:
                #     print(batch_list[0])
                
            
                feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
                                }
                pixel_x_, pixel_y_ = self.sess.run([pixel_x,pixel_y], feed_dict=feed_dict)
                # feed_dict = {self.sat_x: batch_sat[:batch_size], self.sat_x_semi: batch_sat[batch_size:], self.grd_x: batch_grd,    
                #                 }
                # delta_val = self.sess.run(delta,feed_dict=feed_dict)
                for i in range(batch_size):
                    try:
                        pre_sat_name_delta=data_loader.test_delta[batch_list[i]][0].astype(float)
                    except IndexError:
                        break
                    x=pre_sat_name_delta[0]
                    y=pre_sat_name_delta[1]
                    gt_x=x+320.0
                    gt_y=319.0-y
                    error=np.sqrt(np.square(gt_x-pixel_x_[i])+np.square(gt_y-pixel_y_[i]))

                    # delta_dic[batch_list[i]]=delta_val[i]
                    # delta_semi_dic[batch_list[i]]=delta_semi_val[i]
                    error_dic[batch_list[i]]=error
                    # pre_gt_dic[batch_list[i]]=pre_gt_val[i]
                    # delta_dic[batch_list[i]]=delta_val[i]
                    # print(batch_list[i])
                    index+=1
                    print(index)

                # print(iter)
                iter+=1

            np.save(name1,error_dic)
            # np.save(name2,delta_dic)
            # np.save(name3,data_loader.semi_inx)
            return name1
            
def feature_distance(sat_global,grd_global):

    #Compute the Euclidean distance matrix dist_array between sat_global and grd_global.
    dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)

    #Extract the values on the diagonal of dist_array to get the distance between positive samples pos_dist.
    pos_dist = tf.diag_part(dist_array)
    
    # dist_array_semi = 2 - 2 * tf.matmul(sat_global_semi, grd_global, transpose_b=True)
    # pos_dist_semi = tf.diag_part(dist_array_semi)

    return pos_dist


def get_key_by_value(dictionary, target_value,defalt=None):
        for key, value in dictionary.items():
            if value == target_value:
                return key
        return None

def meter_level_localization_from_npy(load_path=None,model='',mode='train_SAFA_same_continuous_delta'):
    semi_dic={}
    # data_loader = DataLoader(same_area=True)
    # data_loader=DataLoader(mode=mode, same_area=True)
    
    dataset=np.array(mode.split('_'))
    dataset=dataset[0]
    
    # if os.path.exists(path1):
    #     return path1,path2,path1_1,path1_2,path1_3

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    # uncomment this if you want to generate delta from checkpoints
    generator = ExplanationGenerator(path=f'{load_path}{model}/model.ckpt')

    
    path1_1,path2_1=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_1.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_1.npy',index_semi=1)
    path1_2,path2_2=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_2.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_2.npy',index_semi=2)
    path1_3,path2_3=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_3.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_3.npy',index_semi=3)
  
    path1_1_val=np.load(path1_1,allow_pickle=True).item()
    path1_2_val=np.load(path1_2,allow_pickle=True).item()
    path1_3_val=np.load(path1_3,allow_pickle=True).item()

    path2_3_val=np.load(path2_3,allow_pickle=True).item()
    path2_3_val_ = [value for key, value in path2_3_val.items()]
    
    keys=[key for key, value in path1_1_val.items()]
    for key in keys:
        item_list=[]
        item_list.append(path1_1_val[key])
        item_list.append(path1_2_val[key])
        item_list.append(path1_3_val[key])
        semi_dic[key]=item_list
    semi_dic_array=[value for key, value in semi_dic.items()]

    path1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu.npy'
    path1_=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu.txt'
    path2_3_=f'{load_path}L_delta_{model}_{dataset}_shunxu.txt'
    # path2=f'{load_path}L_delta_{model}_{dataset}_shunxu.npy'
    path2=path2_3
    np.save(path1,semi_dic)
    np.savetxt(path1_,semi_dic_array)
    np.savetxt(path2_3_,path2_3_val_)

    return path1,path2,path1_1,path1_2,path1_3


def comp_uncer_error_from_npy(load_path=None,model='',mode='train_SAFA_same_continuous_delta'):
    semi_dic={}
    # data_loader = DataLoader(same_area=True)
    # data_loader=DataLoader(mode=mode, same_area=True)
    

    
    # dataset=np.array(mode.split('_'))
    # dataset=dataset[0]
    # dataset='train'
    dataset='test'
    
    # if os.path.exists(path1):
    #     return path1,path2,path1_1,path1_2,path1_3

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95

    generator = ExplanationGenerator(path=f'{load_path}{model}/model.ckpt')

    
    path1_1,path2_1=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_1.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_1.npy',index_semi=1)
    path1_2,path2_2=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_2.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_2.npy',index_semi=2)
    path1_3,path2_3=generator.get_pre_gt(name1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu_3.npy',name2=f'{load_path}L_delta_{model}_{dataset}_shunxu_3.npy',index_semi=3)
    error_path_semi_1=generator.get_pre_gt_uncer_error(name1=f'{load_path}error_semi_{model}_{dataset}_shunxu_1.npy',index_semi=1)
    error_path_semi_2=generator.get_pre_gt_uncer_error(name1=f'{load_path}error_semi_{model}_{dataset}_shunxu_2.npy',index_semi=2)
    error_path_semi_3=generator.get_pre_gt_uncer_error(name1=f'{load_path}error_semi_{model}_{dataset}_shunxu_3.npy',index_semi=3)
    error_path_pos=generator.get_pre_gt_uncer_error_pos(name1=f'{load_path}error_pos_{model}_{dataset}_shunxu.npy')
    # error_path_semi_1=generator.get_pre_gt_uncer_error(name1=f'{load_path}feated_error_semi_{model}_{dataset}_shunxu_1.npy',index_semi=1)
    # error_path_semi_2=generator.get_pre_gt_uncer_error(name1=f'{load_path}feated_error_semi_{model}_{dataset}_shunxu_2.npy',index_semi=2)
    # error_path_semi_3=generator.get_pre_gt_uncer_error(name1=f'{load_path}feated_error_semi_{model}_{dataset}_shunxu_3.npy',index_semi=3)



    path1_1_val=np.load(path1_1,allow_pickle=True).item()
    path1_2_val=np.load(path1_2,allow_pickle=True).item()
    path1_3_val=np.load(path1_3,allow_pickle=True).item()

    path2_3_val=np.load(path2_3,allow_pickle=True).item()
    path2_3_val_ = [value for key, value in path2_3_val.items()]
    
    keys=[key for key, value in path1_1_val.items()]
    for key in keys:
        item_list=[]
        item_list.append(path1_1_val[key])
        item_list.append(path1_2_val[key])
        item_list.append(path1_3_val[key])
        semi_dic[key]=item_list
    semi_dic_array=[value for key, value in semi_dic.items()]

    path1=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu.npy'
    path1_=f'{load_path}L_semi_delta_{model}_{dataset}_shunxu.txt'
    path2_3_=f'{load_path}L_delta_{model}_{dataset}_shunxu.txt'
    # path2=f'{load_path}L_delta_{model}_{dataset}_shunxu.npy'
    path2=path2_3
    np.save(path1,semi_dic)
    np.savetxt(path1_,semi_dic_array)
    np.savetxt(path2_3_,path2_3_val_)

    return path1,path2,path1_1,path1_2,path1_3,error_path_semi_1,error_path_semi_2,error_path_semi_3,error_path_pos





if __name__=='__main__':

    gpu_visible = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    path1,path2,path1_1,path1_2,path1_3=meter_level_localization_from_npy(load_path = './data-E-2-2-gaimodel-posdelta/',model='80',mode='train_SAFA_same_continuous_delta')
#     path1,path2=meter_level_localization_from_npy(load_path = './data-E-2-2-zuobiao/',model='80',mode='train_SAFA_same_continuous_delta')
#     # tf.reset_default_graph()
#     # meter_level_localization_from_npy(load_path = './data-E-2-2-zuobiao/',model='80',mode='test_SAFA_same_continuous_delta')
#     # tf.reset_default_graph()




