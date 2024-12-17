import time
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from models import SAFA_delta
import tensorflow as tf2
import numpy as np
from .dataloader.dataloader import DataLoader
import Constants as constants

class CholeskyBlock(tf.keras.layers.Layer):
    """
    A helper block which converts the output of neck block to 
    class_num*3 coefficients
    """
    #mlp_tot_layers=1
    def __init__(self, mlp_tot_layers=1, mlp_hidden_units=512):
        super(CholeskyBlock, self).__init__()
        self.mlp_tot_layers=mlp_tot_layers

        num_coefficients_to_output = 3

        if mlp_tot_layers > 1:
            self.fc_1 = tf.keras.layers.Dense(units=mlp_hidden_units, activation='sigmoid')
            
            for i in range(mlp_tot_layers - 2):
                setattr(self, f'fc_{i+2}', tf.keras.layers.Dense(units=mlp_hidden_units, activation='sigmoid'))

            self.fc_output = tf.keras.layers.Dense(units=num_coefficients_to_output)
        else:
            self.fc_output = tf.keras.layers.Dense(units=num_coefficients_to_output)

    def call(self, inputs):
        x = tf.reshape(inputs, (-1, 2048))  # Reshape the input
        #x=inputs
        if hasattr(self, 'fc_1'):
            x = self.fc_1(x)
            for i in range(2, self.mlp_tot_layers):
                x = getattr(f'fc_{i}')(x)

        output = self.fc_output(x)
        return output

def get_cholesky(bottleneck):

        cholesky=CholeskyBlock(mlp_tot_layers= 2, mlp_hidden_units= 512 )
        y = cholesky(bottleneck)
        if tf.test.is_gpu_available():
            y = tf.where(tf.test.is_gpu_available(), y, y)
    
        update_y=elu_plus(y)  
        #tf.nn.sigmoid

        return update_y

def elu_plus(x):
    return tf.nn.elu(x) + 1.0

def post_process_and_normalize(htp, use_softmax=False, tau=0.02):
    """
        Post process and then normalize to sum to 1
    """    

    # Extract dimensions of the input heatmaps
    batch_size, height, width,_ = htp.get_shape().as_list()

    # Apply post-processing
    # htp = post_process_input(heatmaps, postprocess)

    if use_softmax:
        # Small tau can blow up the numerical values. Use numerically stable
        # softmax by first subtracting the max values from the individual heatmap
        # Use https://stackoverflow.com/a/49212689
        m = tf.reshape(htp, [batch_size, height * width])
        m = tf.reduce_max(m, axis=1)                                              
        m = tf.reshape(m, [batch_size])                                   
        # m = tf.reshape(m, [batch_size])                            
        htp = htp - m        
        htp = tf.exp(htp/tau) 

    # Add a small EPSILON for the case where the sum of entries is all zero
    # EPSILON = 1e-10  # Define EPSILON value
    sum2 = get_channel_sum(htp) + constants.EPSILON
    # Get the normalized heatmaps
    htp = htp / sum2

    return htp

def get_spatial_mean(htp):
    """
    Gets the spatial mean of each heatmap from the batch of normalized heatmaps.
    Input : htp          = batch_size x height x width x 1
    Output: means        = batch_size x 2

    Convention:
    |----> X (0th coordinate)
    |
    |
    V Y (1st coordinate)
    """
    # batch_size, height, width, _ = htp.shape

    height=htp.shape[1].value
    width=htp.shape[2].value

    # htp is the normalized heatmap
    sum_htp = get_channel_sum(htp)  # batch_size x 1

    xv, yv = generate_grid(height, width)
    xv = tf.constant(xv, dtype=tf.float32)
    yv = tf.constant(yv, dtype=tf.float32)

    x_pts = get_spatial_mean_along_axis(xv, htp, sum_htp)
    y_pts = get_spatial_mean_along_axis(yv, htp, sum_htp)

    # means = tf.concat([x_pts, y_pts], axis=0)  # batch_size x 2
    return x_pts,y_pts

def get_channel_sum(input):
    """
    Generates the sum of the input tensor along the channel axis.
    input  = batch_size x height x width x 1
    output = batch_size x 1
    """
    temp = tf.reduce_sum(input, axis=[1, 2, 3])
    return temp

def generate_grid(height, width):
    """
    Generates an equally spaced grid with coordinates as integers with the
    size same as the input heatmap.

    Convention of axis:
    |----> X
    |
    |
    V Y
    """
    x = np.linspace(0, width - 1, num=width)
    # print('x.shape：',x.shape)
    y = np.linspace(0, height - 1, num=height)
    # print('y.shape：',y.shape)
    yv, xv = np.meshgrid(x, y)      ##########Here to match the VIGOR coordinate system
    # print('xv.shape：',xv.shape)
    # print('yv.shape：',yv.shape)
    return xv, yv

def get_spatial_mean_along_axis(grid, htp, sum_htp):
    """
    Gets spatial mean along one of the axis.
    Input : grid         = height x width grid
            htp          = batch_size x height x width x 1
            sum_htp      = batch_size x 1
    Output: means        = batch_size x 1
    """
    batch_size, height, width, _ = htp.shape

    # grid * heatmap
    grid_times_htp = grid[:, :, np.newaxis] * htp
    
    # sum of grid times heatmap
    s = get_channel_sum(grid_times_htp)
   
    # predicted pts
    # Add a small nudge when sum_htp is all zero
    pts = s / (sum_htp + constants.EPSILON)

    return pts

# Example usage
# htp = your_normalized_heatmaps_with_shape_(10, 640, 640, 1)
# means = get_spatial_mean(htp)

def get_mahalanobis_distance(x, Sigma_inverse, y):
        """
        Returns x^T Sigma_inverse y
        :param x:             batch_size x 2
        :param Sigma_inverse: batch_size x 2 x 2
        :param y:             batch_size x 2
            
        :return: product of size batch_size x 68
        """
        batch_size = tf.shape(Sigma_inverse)[0]  # batch_size
        

        x_vec = tf.reshape(x, (batch_size , 2, 1))  # batch_size  x 2 x 1
        y_vec = tf.reshape(y, (batch_size , 2, 1))  # batch_size  x 2 x 1
        Sigma_inverse = tf.reshape(Sigma_inverse, (batch_size, 2, 2))  # batch_size x 2 x 2

        # TensorFlow batch matrix multiplication
        # https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
        product = tf.matmul(tf.transpose(x_vec, perm=[0, 2, 1]), tf.matmul(Sigma_inverse, y_vec))  # batch_size * 68 x 1 x 1
        product = tf.squeeze(product, axis=[-1, -2])  # batch_size * 68
        # print('product shape:',product.shape)
        # product = tf.reshape(product, (batch_size))  # batch_size x 68
        # print('product shape:',product.shape)
        # Sigma_inverse = tf.reshape(Sigma_inverse, (batch_size, 2, 2))  # batch_size x 68 x 2 x 2
        # x_vec = tf.reshape(x, (batch_size, 2))
        # y_vec = tf.reshape(y, (batch_size, 2))
        return product

def get_zero_variable(shape, type_like_input):
    """
    Returns a zero-initialized variable whose shape is similar to the shape but type is 
    like the variable type_like_input.
    """
    # zero_initializer = tf.keras.initializers(dtype=type_like_input.dtype)
    return tf.Variable(tf.zeros(shape, dtype=type_like_input.dtype), trainable=False)

def get_sigma(L): 
    
    zero_column = tf.zeros((tf.shape(L)[0], 1))
    tensor_b = tf.concat([L[:, :1], zero_column, L[:, 1:]], axis=1)
    L_mat=tf.reshape(tensor_b,shape=(-1, 2, 2))
   
    # Covariance matrix Sigma = LL^T
    Sigma = tf.matmul(L_mat, tf.transpose(L_mat, perm=[0, 2, 1]))

    return Sigma

def get_uncertainty(L):

    Sigma=get_sigma(L)

    # # print('Sigma shape:',Sigma.shape)

    eigenvalues = tf2.linalg.eigvals(Sigma)
    # # print('eigenvalues shape:',eigenvalues.shape)

    eigenvalues = tf.cast(eigenvalues, dtype=tf.float32)

    major_axis_length =2*tf.sqrt(tf.maximum(eigenvalues[:,0],eigenvalues[:,1]))
    minor_axis_length =2*tf.sqrt(tf.minimum(eigenvalues[:,0],eigenvalues[:,1]))
    # det_Sigma = Sigma[:, 0, 0] * Sigma[:, 1, 1] - Sigma[:, 0, 1] * Sigma[:, 1, 0]

    # return tf.sqrt(det_Sigma)
    return major_axis_length+minor_axis_length
# tf.sqrt(det_Sigma)    major_axis_length+minor_axis_length

