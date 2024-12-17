from backbone.VGG import Vgg16
import tensorflow as tf2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import loupe as lp
from output_head import decoder
from utils import get_cholesky

# the clean siamese network used in https://openaccess.thecvf.com/content/WACV2021/papers/Zhu_Revisiting_Street-to-Aerial_View_Image_Geo-Localization_and_Orientation_Estimation_WACV_2021_paper.pdf



def SAFA_delta(x_sat, x_sat_semi, x_grd, dimension=8, trainable=True, out_dim=2048, original=False):
    with tf.variable_scope('vgg_grd'):
        vgg_grd = Vgg16()
        _,_,_,_, grd_local = vgg_grd.build(x_grd)
        grd_local_out = grd_local
        grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    # print('grd_w.shape:',grd_w.shape)
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])
    # print('grd_local.shape:',grd_local.shape)
    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    # print('grd_global.shape:',grd_global.shape)
    grd_global = tf.reshape(grd_global, [-1, dimension * channel])
    # print('grd_global.shape:',grd_global.shape)

    with tf.variable_scope('vgg_sat') as scope:
        vgg_sat = Vgg16()
        conv1_2,conv2_2,conv3_3,conv4_3, sat_local = vgg_sat.build(tf.concat([x_sat, x_sat_semi], axis=0))
        sat_local_out = sat_local
        
        conv1_2_de , conv1_2_de_semi= tf.split(conv1_2, 2, axis=0)
        conv2_2_de , conv2_2_de_semi= tf.split(conv2_2, 2, axis=0)
        conv3_3_de , conv3_3_de_semi = tf.split(conv3_3, 2, axis=0)
        conv4_3_de , conv4_3_de_semi= tf.split(conv4_3, 2, axis=0)
        sat_local_de , sat_local_de_semi = tf.split(sat_local, 2, axis=0)

        print(conv1_2_de_semi.shape,conv2_2_de_semi.shape,conv3_3_de_semi.shape,conv4_3_de_semi.shape,sat_local_de_semi.shape)

        sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    # print('sat_w.shape:',sat_w.shape)
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])
    # print('sat_local.shape:',sat_local.shape)
    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    # print('sat_global.shape:',sat_global.shape)
    sat_global = tf.reshape(sat_global, [-1, dimension * channel])
    # print('sat_global.shape:',sat_global.shape)
    
    sat_global_split_1, sat_global_split_2 = tf.split(sat_global, 2, axis=0)

    with tf.variable_scope('L') as scope:
        both_feature = tf.concat([sat_global_split_1, grd_global], axis=-1)
        print('both_feature.shape:',both_feature.shape)
        both_feature_fc = fc_layer(both_feature, 4096 * 2, 4096, 0.005, 0.1, 'fc_delta_1')
        #delta_regression=fc_layer(both_feature_fc, 4096, 2, 0.005, 0.1, 'fc_delta_2', activation_fn=None)
        mlp_bottleneck = fc_layer(both_feature_fc, 4096, out_dim, 0.005, 0.1, 'mlp_bottleneck', activation_fn=None)
        L=get_cholesky(mlp_bottleneck)

    with tf.variable_scope('L_semi') as scope:
        both_feature_semi = tf.concat([sat_global_split_2, grd_global], axis=-1)
        print('both_feature.shape:',both_feature.shape)
        both_feature_fc_semi = fc_layer(both_feature_semi, 4096 * 2, 4096, 0.005, 0.1, 'fc_delta_1_semi')
        mlp_bottleneck_semi = fc_layer(both_feature_fc_semi, 4096, out_dim, 0.005, 0.1, 'mlp_bottleneck_semi', activation_fn=None)
        L_semi=get_cholesky(mlp_bottleneck_semi)

    with tf.variable_scope('decoder_semi') as scope:
        mlp_decoder=decoder()
        mlp_bottleneck_reshape_semi=tf.expand_dims(mlp_bottleneck_semi,1)
        mlp_bottleneck_reshape_semi=tf.expand_dims(mlp_bottleneck_reshape_semi,1)
        #mlp_bottleneck_reshape_semi = tf2.image.resize_with_pad(mlp_bottleneck_reshape_semi,5,5)
        
        logits_semi = mlp_decoder.decode(mlp_bottleneck_reshape_semi, trainable, 'decode_semi', conv1_2_de_semi, conv2_2_de_semi, conv3_3_de_semi, conv4_3_de_semi, sat_local_de_semi)
        
        logits_reshaped_semi = tf.reshape(logits_semi, [tf.shape(logits_semi)[0], 640*640])
        heatmap_semi = tf.reshape(tf.nn.relu(logits_reshaped_semi), tf.shape(logits_semi))

    with tf.variable_scope('decoder') as scope:
        mlp_decoder=decoder()
        mlp_bottleneck_reshape=tf.expand_dims(mlp_bottleneck,1)
        mlp_bottleneck_reshape=tf.expand_dims(mlp_bottleneck_reshape,1)
        #mlp_bottleneck_reshape = tf2.image.resize_with_pad(mlp_bottleneck_reshape,5,5)
        
        logits = mlp_decoder.decode(mlp_bottleneck_reshape, trainable, 'decode', conv1_2_de, conv2_2_de, conv3_3_de, conv4_3_de, sat_local_de)
        
        logits_reshaped = tf.reshape(logits, [tf.shape(logits)[0], 640*640])
        # print(tf.nn.softmax(logits_reshaped).shape)
        heatmap = tf.reshape(tf.nn.relu(logits_reshaped), tf.shape(logits))

    if original:
        return tf.nn.l2_normalize(sat_global_split_1, dim=1), tf.nn.l2_normalize(sat_global_split_2, dim=1),\
           tf.nn.l2_normalize(grd_global, dim=1), sat_global_split_1, grd_global, sat_local, grd_local, sat_local_out, grd_local_out, heatmap,heatmap_semi, L,L_semi

    return tf.nn.l2_normalize(sat_global_split_1, dim=1), tf.nn.l2_normalize(sat_global_split_2, dim=1),\
           tf.nn.l2_normalize(grd_global, dim=1), sat_local, grd_local,heatmap,heatmap_semi, L,L_semi


def stable_softmax(logits, axis=-1):
    # Translate the input to set the maximum value to 0
    shift_logits = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    #exp_logits = tf.exp(shift_logits)
    #softmax = exp_logits / tf.reduce_sum(exp_logits, axis=axis, keepdims=True)
    return tf.nn.softmax(shift_logits)

# supportive blocks
def fc_layer(x, input_dim, output_dim, init_dev, init_bias, name='fc_layer', activation_fn=tf.nn.relu,
             reuse=tf.compat.v1.AUTO_REUSE):
    '''
    This function is used to create a fully connected layer neuron layer. 
    It creates weights and biases in a given variable scope and multiplies the input data with the weights, 
    adds the biases, and then activates it with the activation function
    '''

    with tf.variable_scope(name, reuse=reuse):
        weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                 trainable=True,
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=init_dev))
        bias = tf.get_variable(name='biases', shape=[output_dim],
                               trainable=True, initializer=tf.constant_initializer(init_bias))

        if activation_fn is not None:
            out = tf.nn.xw_plus_b(x, weight, bias)
            out = activation_fn(out)
        else:
            out = tf.nn.xw_plus_b(x, weight, bias)

    return out



#spatial aware, SAFA module
def spatial_aware(input_feature, dimension, trainable, name):
    batch, height, width, channel = input_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_max(input_feature, axis=-1), [-1, height * width])

    with tf.variable_scope(name):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width / 2), dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.keras.regularizers.l2(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width / 2), dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.keras.regularizers.l1(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1

        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.keras.regularizers.l2(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.keras.regularizers.l1(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2

        return vec3
