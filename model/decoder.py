import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class decoder:
   
    def conv2d(self, x, W):
        
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1],
                            padding='SAME')

    def deconv2d(self, x, W):

        #print(x.get_shape().as_list())
        
        _, height, width, _ = x.get_shape().as_list()
        channel = W.get_shape().as_list()[2]
        
        return tf.nn.conv2d_transpose(x, W, output_shape=[tf.shape(x)[0], height*2, width*2, channel], strides=[1, 2, 2, 1], padding='SAME')
    
    def deconv2d_x5(self, x, W):

        #print(x.get_shape().as_list())
        
        _, height, width, _ = x.get_shape().as_list()
        channel = W.get_shape().as_list()[2]
        
        return tf.nn.conv2d_transpose(x, W, output_shape=[tf.shape(x)[0], height*5, width*5, channel], strides=[1, 5, 5, 1], padding='SAME')



    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name): 
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.keras.initializers.glorot_normal)
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.keras.initializers.glorot_normal)

            if activated:
                out = activation_function(self.conv2d(x, weight) + bias)
            else:
                out = self.conv2d(x, weight) + bias

            return out
    
        
    def deconv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                     name='layer_deconv', activation_function=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, output_dim, input_dim],
                                     trainable=trainable, initializer=tf.keras.initializers.glorot_normal)
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.keras.initializers.glorot_normal)
            
            if activated:
                out = activation_function(self.deconv2d(x, weight) + bias)
            else:
                out = self.deconv2d(x, weight) + bias
            
            return out
        
    def deconv_layer_x5(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                     name='layer_deconv', activation_function=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, output_dim, input_dim],
                                     trainable=trainable, initializer=tf.keras.initializers.glorot_normal)
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.keras.initializers.glorot_normal)
            
            if activated:
                out = activation_function(self.deconv2d_x5(x, weight) + bias)
            else:
                out = self.deconv2d_x5(x, weight) + bias
            
            return out


    def decode(self, x, trainable, name, sat512, sat256, sat128, sat64, sat32, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(name, reuse=reuse):
            ''''''
            deconv0 = self.deconv_layer_x5(x, 3, 2048, 2048, trainable, True, 'deconv0')
            conv0_1 = self.conv_layer(deconv0, 3, 2048, 2048, trainable, True, 'conv0_1')
            #conv0_2 = self.conv_layer(conv0_1, 3, 2048, 2048, trainable, True, 'conv0_2')


            # deconv1 and conv1, height, width: 10*10
            deconv1 = self.deconv_layer(conv0_1, 3, 2048, 1024, trainable, True, 'deconv1')
            conv1_1 = self.conv_layer(deconv1, 3, 1024, 512, trainable, True, 'conv1_1')
            conv1_2 = self.conv_layer(conv1_1, 3, 512, 512, trainable, True, 'conv1_2')
            
            # deconv2 and conv2, height, width: 20*20
            deconv2 = self.deconv_layer(conv1_2, 3, 512, 256, trainable, True, 'deconv2')
            #print('@@@@@@:',sat32.shape)
            conv2_1 = self.conv_layer(tf.concat([deconv2, sat32], 3), 3, 768, 256, trainable, True, 'conv2_1')
            conv2_2 = self.conv_layer(conv2_1, 3, 256, 256, trainable, True, 'conv2_2')
            
            # deconv3 and conv3, height, width: 40*40
            deconv3 = self.deconv_layer(conv2_2, 3, 256, 128, trainable, True, 'deconv3')
            conv3_1 = self.conv_layer(tf.concat([deconv3, sat64], 3), 3, 640, 128, trainable, True, 'conv3_1')
            conv3_2 = self.conv_layer(conv3_1, 3, 128, 128, trainable, True, 'conv3_2')
            
            # deconv4 and conv4, height, width: 80*80
            deconv4 = self.deconv_layer(conv3_2, 3, 128, 64, trainable, True, 'deconv4')
            conv4_1 = self.conv_layer(tf.concat([deconv4, sat128], 3), 3, 320, 64, trainable, True, 'conv4_1')
            conv4_2 = self.conv_layer(conv4_1, 3, 64, 64, trainable, True, 'conv4_2')
            
            # deconv5 and conv5, height, width: 160*160
            deconv5 = self.deconv_layer(conv4_2, 3, 64, 32, trainable, True, 'deconv5')
            conv5_1 = self.conv_layer(tf.concat([deconv5, sat256], 3), 3, 160, 32, trainable, True, 'conv5_1')
            conv5_2 = self.conv_layer(conv5_1, 3, 32, 32, trainable, True, 'conv5_2')
            
            # deconv6 and conv6, height, width: 320*320
            deconv6 = self.deconv_layer(conv5_2, 3, 32, 16, trainable, True, 'deconv6')
            conv6_1 = self.conv_layer(tf.concat([deconv6, sat512], 3), 3, 80, 16, trainable, True, 'conv6_1')
            conv6_2 = self.conv_layer(conv6_1, 3, 16, 16, trainable, True, 'conv6_2')

            # deconv6 and conv7, height, width: 640*640
            deconv7 = self.deconv_layer(conv6_2, 3, 16, 8, trainable, True, 'deconv7')
            conv7_1 = self.conv_layer(deconv7, 3, 8, 4, trainable, True, 'conv7_1')
            conv7_2 = self.conv_layer(conv7_1, 3, 4, 1, trainable, False, 'conv7_2')

            return conv7_2
