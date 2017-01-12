import tensorflow as tf
import numpy as np
from model import Model

class ModelA3C(Model):
    def build_network_nature(self, name, trainable, num_actions):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x_in = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x_in)
        
            x_normalized = tf.to_float(x_in) / 255.0
            self.print_log(x_normalized)
    
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, self.history_len, 32], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            self.print_log(h_conv1)
    
            # Third layer convolves 64 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 32, 64], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            self.print_log(h_conv2)
    
            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            W_conv3, b_conv3 = self.make_layer_variables([3, 3, 64, 64], trainable, "conv3")
    
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            self.print_log(h_conv3)
            
            conv_out_size = np.prod(h_conv3._shape[1:]).value
    
            h_conv3_flat = tf.reshape(h_conv3, [-1, conv_out_size], name="h_conv3_flat")
            self.print_log(h_conv3_flat)
    
            # Fifth layer is fully connected with 512 relu units
            W_fc1, b_fc1 = self.make_layer_variables([conv_out_size, 512], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            self.print_log(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([512, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            self.print_log(y)
            
            y_class = tf.nn.softmax(y)
            
            W_fc3, b_fc3 = self.make_layer_variables([512, 1], trainable, "fc3")
            v_ = tf.matmul(h_fc1, W_fc3) + b_fc3
            v = tf.reshape(v_, [-1] )
        
        self.x = x_in
        self.y = y
        self.y_class = y_class
        self.v = v        
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x_in = tf.placeholder(tf.float32, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x_in)
        
            x_normalized = tf.to_float(x_in) / 255.0
            self.print_log(x_normalized)
    
            # Second layer convolves 16 8x8 filters with stride 4 with relu
            W_conv1, b_conv1 = self.make_layer_variables([8, 8, self.history_len, 16], trainable, "conv1")
    
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            self.print_log(h_conv1)
    
            # Third layer convolves 32 4x4 filters with stride 2 with relu
            W_conv2, b_conv2 = self.make_layer_variables([4, 4, 16, 32], trainable, "conv2")
    
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            self.print_log(h_conv2)

            conv_out_size = np.prod(h_conv2._shape[1:]).value
    
            h_conv2_flat = tf.reshape(h_conv2, [-1, conv_out_size], name="h_conv2_flat")
            self.print_log(h_conv2_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_fc1, b_fc1 = self.make_layer_variables([conv_out_size, 256], trainable, "fc1")
    
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1, name="h_fc1")
            self.print_log(h_fc1)
    
            W_fc2, b_fc2 = self.make_layer_variables([256, num_actions], trainable, "fc2")
    
            y = tf.matmul(h_fc1, W_fc2) + b_fc2
            self.print_log(y)
            
            y_class = tf.nn.softmax(y)
            
            W_fc3, b_fc3 = self.make_layer_variables([256, 1], trainable, "fc3")
            v_ = tf.matmul(h_fc1, W_fc3) + b_fc3
            v = tf.reshape(v_, [-1] )
        
        self.x = x_in
        self.y = y
        self.y_class = y_class
        self.v = v
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3]
