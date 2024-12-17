from matplotlib import pyplot as plt
from .model.models import SAFA_delta
from .dataloader.dataloader import DataLoader
import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import time
from utils import get_cholesky
from scipy.linalg import eig
import tf_slim as slim
import sympy as sp
from .utils.computer_E_3_delta_two_dec import meter_level_localization_from_npy,delta_fenji

is_training = True
keep_prob_val = 0.8
a3,b3,c3=0.3,75.0,2.0

# -------------------------------------------------------- #
def validate(grd_descriptor, sat_descriptor):
    # Used to compute the similarity between a pair of descriptors (feature vectors) and determine the accuracy of validation based on the similarity
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    return accuracy

def validate_top(grd_descriptor, sat_descriptor, dataloader):
    accuracy = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))

    top1_percent = int(dist_array.shape[1] * 0.01) + 1
    top1 = 1
    top5 = 5
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, dataloader.test_label[i][0]]
        prediction = np.sum(dist_array[i, :] < gt_dist)

        dist_temp = np.ones(dist_array[i, :].shape[0])
        dist_temp[dataloader.test_label[i][1:]] = 0
        prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

        if prediction < top1_percent:
            accuracy += 1.0
        if prediction < top1:
            accuracy_top1 += 1.0
        if prediction < top5:
            accuracy_top5 += 1.0
        if prediction_hit < top1:
            accuracy_hit += 1.0
        data_amount += 1.0
    accuracy /= data_amount
    accuracy_top1 /= data_amount
    accuracy_top5 /= data_amount
    accuracy_hit /= data_amount
    return accuracy, accuracy_top1, accuracy_top5, accuracy_hit


def compute_loss(sat_global, grd_global, batch_hard_count=0, batch_size = 14, loss_weight = 10.0):
    '''
    Compute the weighted soft-margin triplet loss计算加权软边际三重损失
    :param sat_global: the satellite image global descriptor卫星图像全局描述符
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    批次内最高硬负对的数量。如果为 0，则不进行批内硬负挖掘
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)     
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0
    return loss

def compute_loss_continuous_IOU(sat_global_1, sat_global_2, grd_global, ratio, W_, batch_size = 10):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''

    loss_1 = compute_loss(sat_global_1, grd_global,batch_size=batch_size)
    loss_2 = compute_loss(sat_global_2, grd_global,batch_size=batch_size)
    sim_1 = tf.reduce_sum(sat_global_1 * grd_global, axis=1)
    sim_2 = tf.reduce_sum(sat_global_2 * grd_global, axis=1)
    convince=W_* (sim_2/sim_1)
    error = convince - ratio
    loss_3 = tf.reduce_mean(error*error)
    loss = loss_1 + loss_3
    
    return loss, loss_1, loss_3,convince


def distance_score(delta_1, delta_2, mode = 'IOU', L=640.):
    if mode == 'DIS':
        distance_1 = np.sqrt(delta_1[:, 0] * delta_1[:, 0] + delta_1[:, 1] * delta_1[:, 1])
        distance_2 = np.sqrt(delta_2[:, 0] * delta_2[:, 0] + delta_2[:, 1] * delta_2[:, 1])
        ratio = distance_1/distance_2
    elif mode == 'IOU':
        IOU_1 = 1. / (1. - (1 - np.abs(delta_1[:, 0]) / L) * (1. - np.abs(delta_1[:, 1]) / L) / 2.) - 1
        IOU_2 = 1. / (1. - (1 - np.abs(delta_2[:, 0]) / L) * (1. - np.abs(delta_2[:, 1]) / L) / 2.) - 1
        ratio = IOU_2/ IOU_1
    return ratio


# def compute_loss_work_2_1(sat_global_semi, grd_global, W_, batch_hard_count=0, batch_size = 14, loss_weight = 10.0):
#     '''
#     Compute the weighted soft-margin triplet loss
#     :param sat_global: the satellite image global descriptor
#     :param grd_global: the ground image global descriptor
#     :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
#     :return: the loss
#     '''
#     with tf.name_scope('weighted_soft_margin_triplet_loss'):
        
#         dist_array = 2 - 2 * tf.matmul(sat_global_semi, grd_global, transpose_b=True)
#         pos_dist = tf.diag_part(dist_array)
#         if batch_hard_count == 0:
    
#             triplet_dist_semi = W_ * pos_dist

#             loss_semi = tf.reduce_mean(tf.log(1 + tf.exp(triplet_dist_semi * loss_weight))) 

#     return loss_semi/10.0

def get_weights(sat_global_1, sat_global_2, grd_global, ratio):
    
    sim_1 = tf.reduce_sum(sat_global_1 * grd_global, axis=1)
    sim_2 = tf.reduce_sum(sat_global_2 * grd_global, axis=1)
    convince=sim_1/sim_2
    W=ratio*convince - 1
    return W

def train(start_epoch=1, mode='', model_dir='', load_model_path='', load_mining_path='', break_iter=None,
          dim=4096, number_of_epoch = 30, learning_rate_val = 1e-5, batch_size = 14, mining_start= -1,save_dir='',path=''):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''
    

    # import data
    input_data = DataLoader(mode=mode, dim=dim, same_area=True if 'same' in mode else False)
    if break_iter is None:
        break_iter = int(input_data.train_data_size / batch_size)
    if 'continuous' in mode:
        batch_size = int(np.ceil(batch_size / 1.5) // 2 * 2)
        #batch_size=20

    sat_x = tf.placeholder(tf.float32, [None, input_data.sat_size[0], input_data.sat_size[1], 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, input_data.grd_size[0], input_data.grd_size[1], 3], name='grd_x')

    
    W_=tf.placeholder(tf.float32, [None])
    ratio=tf.placeholder(tf.float32, [None])

    
    delta_2=tf.placeholder(tf.float32,[None, 2],name='delta_2')
    delta_1=tf.placeholder(tf.float32,[None, 2],name='delta_1')
    global_step = tf.Variable(0, trainable=False)
    
    # build model
    sat_x_semi = tf.placeholder(tf.float32, [None, input_data.sat_size[0], input_data.sat_size[1], 3], name='sat_x_semi')   
    # sat_global, sat_global_semi, grd_global, sat_return, grd_return, mlp_bottleneck, logits, delta_regression= SAFA_delta(sat_x, sat_x_semi, grd_x, out_dim= 2048)
    sat_global, sat_global_semi, grd_global, sat_return, grd_return,heatmap,heatmap_semi,L,L_semi= SAFA_delta(sat_x, sat_x_semi, grd_x, out_dim= 2048)
    
    # choose loss
    with tf.name_scope('loss'):
        loss,loss_tri,loss_iou,convince=compute_loss_continuous_IOU(sat_global, sat_global_semi, grd_global, ratio, W_, batch_size = batch_size)
        L_semi_delta_dict=np.load(path,allow_pickle=True).item()
        
        tf.summary.scalar('loss_tri', loss_tri)
        tf.summary.scalar('loss_iou', loss_iou)
        tf.summary.scalar('zong_loss', loss)
        tf.summary.scalar('convince', tf.reduce_mean(convince))
        
    
    L_semi_weight_dict={}
    for i in range(input_data.train_data_size):
        temp=[]
        for j in range(0,3):
            weight=1/(1+c3*np.exp(a3*(b3-L_semi_delta_dict[i][j])))
            temp.append(weight)
        L_semi_weight_dict[i]=temp

    W_z=get_weights(sat_global,sat_global_semi,grd_global,ratio)

    # set training
    rate = tf.train.exponential_decay(learning_rate_val, global_step, break_iter, 0.9)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver([v for v in tf.global_variables() if 'Variable' not in v.name], max_to_keep=None)
    #assign_op = tf.assign(global_step, 0)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    merged = tf.summary.merge_all()
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(assign_op)
        # Continue from breakpoint
        writer = tf.summary.FileWriter(f'./{save_dir}/logs/', sess.graph)
        
        #if start_epoch != 1:
        print('load model...')
        saver.restore(sess, os.path.join(load_model_path , 'model.ckpt'))
        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')
        if 'mining' in mode and len(load_mining_path) > 0:
            input_data.sat_global_train = np.load(os.path.join(load_mining_path , 'sat_global_train.npy'))
            input_data.grd_global_train = np.load(os.path.join(load_mining_path , 'grd_global_train.npy'))
            input_data.mining_save = np.load(os.path.join(load_mining_path, 'mining_save.npy'))
            input_data.mining_pool_ready = True
            input_data.cal_ranking_train_limited()
            print('load train global done!')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            # L_semi_dic=[]
            # L_dic=[]
            while True:
                t1 = time.time()
                # train
                batch_sat, batch_grd, batch_list, delta_list ,randx_list = input_data.get_next_batch_E_3(batch_size)

                if batch_sat is None or iter == break_iter:
                    if 'mining' in mode:
                        input_data.cal_ranking_train_limited()
                        print('refreshed and sorted..')
                        if epoch >= (mining_start - 1):
                            if not os.path.exists(model_dir):
                                os.makedirs(model_dir)
                            np.save(os.path.join(model_dir , 'sat_global_train.npy'), input_data.sat_global_train)
                            np.save(os.path.join(model_dir , 'grd_global_train.npy'), input_data.grd_global_train)
                            np.save(os.path.join(model_dir, 'mining_save.npy'), input_data.mining_save)
                            print('mining saved in:', model_dir)
                            input_data.mining_pool_ready = True
                            print('start mining at epoch:', epoch + 1)
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat[:batch_size], sat_x_semi: batch_sat[batch_size:], grd_x: batch_grd,
                             delta_2:delta_list[batch_size:],
                             delta_1:delta_list[:batch_size],
                             ratio: distance_score(delta_list[:batch_size], delta_list[batch_size:])
                             }
                W_m=[]
                for i in range(batch_size):
                    weight=L_semi_weight_dict[batch_list[i]][randx_list[i]-1]
                    W_m.append(weight)
                W_z_val=sess.run(W_z,feed_dict=feed_dict)
                W=1+W_z_val*W_m
            
                feed_dict = {sat_x: batch_sat[:batch_size], sat_x_semi: batch_sat[batch_size:], grd_x: batch_grd,
                             delta_2:delta_list[batch_size:],
                             delta_1:delta_list[:batch_size],
                             ratio: distance_score(delta_list[:batch_size], delta_list[batch_size:]),
                             W_:W
                            }
                
                loss_tri_val,loss_iou_val,_,sat_global_iter, grd_global_iter, loss_val,summary = sess.run([loss_tri,loss_iou,train_step,sat_global,grd_global, loss, merged], feed_dict=feed_dict)
                t2 = time.time()
                print('loss_tri_val:',loss_tri_val)
                print('loss_iou_val:',loss_iou_val)
                print('global %d, epoch %d, iter %d: loss : %.12f , time: %.4f' % (global_step_val, epoch, iter, loss_val, t2 - t1))
                
                if 'mining' in mode:
                    batch_list_sat = input_data.train_label[batch_list.astype(np.int),0].astype(np.int)
                    input_data.sat_global_train[batch_list_sat, :] = sat_global_iter
                    input_data.grd_global_train[batch_list.astype(np.int), :] = grd_global_iter
                
                iter += 1
                writer.add_summary(summary, global_step_val)
           
            # ---------------------- validation ----------------------
            if epoch % 10 == 0 :
                print('validate...')
                print('   compute global descriptors')
                input_data.reset_scan()
                sat_global_descriptor = np.zeros([input_data.test_sat_data_size, dim])
                grd_global_descriptor = np.zeros([input_data.test_data_size, dim]) 
                # compute sat descriptors
                val_i = 0
                while True:
                    if val_i % 100 == 0:
                        print('sat      progress %d' % val_i)

                    if 'continuous' in mode:
                        batch_sat = input_data.next_sat_scan(32)
                        if batch_sat is None:
                            break
                        feed_dict = {sat_x: batch_sat, sat_x_semi: batch_sat} # semi is not used, load for the static graph
                    else:
                        batch_sat = input_data.next_sat_scan(64)
                        if batch_sat is None:
                            break
                        feed_dict = {sat_x: batch_sat}

                    sat_global_val = sess.run(sat_global, feed_dict=feed_dict)
                    sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                    val_i += sat_global_val.shape[0]
                # compute grd descriptors
                val_i = 0
                while True:
                    if val_i % 100 == 0:
                        print('grd      progress %d' % val_i)
                    batch_grd = input_data.next_grd_scan(32)
                    if batch_grd is None:
                        break
                    feed_dict = {grd_x: batch_grd}
                    grd_global_val = sess.run(grd_global, feed_dict=feed_dict)
                    grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                    val_i += grd_global_val.shape[0]

                print('   compute accuracy')
                val_accuracy, val_accuracy_top1, val_accuracy_top5, hit_rate = validate_top(grd_global_descriptor,
                                                                                sat_global_descriptor, input_data)
                print('Evaluation epoch %d: accuracy = %.1f%% , top1: %.1f%%, top5: %.1f%%, hit_rate: %.1f%%' % (
                epoch, val_accuracy * 100.0, val_accuracy_top1 * 100.0, val_accuracy_top5 * 100.0, hit_rate * 100.0))

                
                np.save(os.path.join(model_dir, 'sat_global_descriptor_60.npy'), sat_global_descriptor)
                np.save(os.path.join(model_dir, 'grd_global_descriptor_60.npy'), grd_global_descriptor)
                del(sat_global_descriptor)
                del(grd_global_descriptor)
                
            model_save_dir = os.path.join(model_dir, str(epoch))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            save_path = saver.save(sess, os.path.join(model_save_dir, 'model.ckpt'))
            print(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))
            print("Model saved in file: {}".format(save_path))
            # ---------------------------------------------------------

def return_rate_val():
    data_loader=DataLoader(mode='train_SAFA_same_continuous_delta', same_area=True)
    path1,path2,path1_1,path1_2,path1_3=meter_level_localization_from_npy(load_path = './data-E-2-2/',model='80',mode='train_SAFA_same_continuous_delta')
    semi_delta=np.load(path1,allow_pickle=True).item()
    delta=np.load(path2,allow_pickle=True).item()
    semi_delta1_1=np.load(path1_1,allow_pickle=True).item()
    semi_delta1_2=np.load(path1_2,allow_pickle=True).item()
    semi_delta1_3=np.load(path1_3,allow_pickle=True).item()
    item_list=[]

    keys=[key for key, value in delta.items()]
    for key in keys:
        item_list.append(delta[key])

    keys=[key for key, value in semi_delta1_1.items()]
    for key in keys:
        item_list.append(semi_delta1_1[key])

    keys=[key for key, value in semi_delta1_2.items()]
    for key in keys:
        item_list.append(semi_delta1_2[key])

    keys=[key for key, value in semi_delta1_3.items()]
    for key in keys:
        item_list.append(semi_delta1_3[key])

    path='./data-E-2-2/delta_semi_delta_list.npy'
    np.save(path,item_list)
    
    return path


if __name__ == '__main__':
    save_dir='data-E-3'
    gpu_visible = "0"
    #mode = 'train_SAFA_mining_same_CVM-loss'
    # mode = 'train_SAFA_mining_same_continuous_delta'
    mode = 'train_SAFA_same_continuous_delta'
    start_epoch = 1
    mining_start = 2
    number_of_epoch = 150
    learning_rate_val = 1e-6
    batch_size = 14
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    model_load_dir = f'./data-E-2-2-gaimodel-posdelta/80/'
    model_save_dir = f'./{save_dir}/'
    load_mining_path = ''
    
    path1,path2,path1_1,path1_2,path1_3=meter_level_localization_from_npy(load_path = './data-E-2-2/',model='80',mode='train_SAFA_same_continuous_delta')
    # path1_,path2_=delta_fenji(path2,path1_1,path1_2,path1_3,num=10,save_dir=model_save_dir)
    # path=return_rate_val()
    tf.reset_default_graph()
    train(start_epoch=start_epoch, mode=mode, model_dir=model_save_dir, load_model_path=model_load_dir,
          number_of_epoch=number_of_epoch, learning_rate_val=learning_rate_val, batch_size=batch_size,
          mining_start=mining_start, load_mining_path=load_mining_path,save_dir=save_dir,path=path1)
    
# ==========================================================================================================
# ### for cross area
# if __name__ == '__main__':
#     gpu_visible = "0"
#     mode = 'train_SAFA_mining_CVM-loss'
#     start_epoch = 1
#     mining_start = 2
#     number_of_epoch = 11
#     learning_rate_val = 1e-5
#     batch_size = 14
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
#     model_load_dir = ''
#     model_save_dir = './data/'
#     load_mining_path = ''
#
#     train(start_epoch=start_epoch, mode=mode, model_dir=model_save_dir, load_model_path=model_load_dir,
#           number_of_epoch=number_of_epoch, learning_rate_val=learning_rate_val, batch_size=batch_size,
#           mining_start=mining_start, load_mining_path=load_mining_path)
#     tf.reset_default_graph()
#
#     # training with regression from 10 epochs
#     mode = 'train_SAFA_mining_continuous_delta'
#     model_load_dir = './data/11'
#     model_save_dir = './data/'
#     load_mining_path = './data/'
#     start_epoch = 12
#     number_of_epoch = 10
#     train(start_epoch=start_epoch, mode=mode, model_dir=model_save_dir, load_model_path=model_load_dir,
#           number_of_epoch=number_of_epoch, learning_rate_val=learning_rate_val, batch_size=batch_size,
#           mining_start=mining_start, load_mining_path=load_mining_path)
