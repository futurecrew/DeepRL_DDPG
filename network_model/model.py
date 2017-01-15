import tensorflow as tf
import numpy as np
import math

def new_session(graph=None):
    config = tf.ConfigProto()
    # Use 20% of GPU memory to prevent from using it all 
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return tf.Session(config=config, graph=graph)

class Model(object):
    def __init__(self, args, name, trainable, action_type_no, thread_no):
        self.args = args
        self.network = args.network
        self.screen_height = args.screen_height
        self.screen_width = args.screen_width 
        self.history_len = args.screen_history
        self.action_type_no = action_type_no
        self.thread_no = thread_no
        with tf.device(args.device):
            self.build_network(name, args.network, trainable, action_type_no)
    
    def make_layer_variables(self, shape, trainable, name_suffix, weight_range=-1):
        if weight_range == -1:
            weight_range = 1.0 / math.sqrt(np.prod(shape[0:-1]))
        weights = tf.Variable(tf.random_uniform(shape, minval=-weight_range, maxval=weight_range), trainable=trainable, name='W_' + name_suffix)
        biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-weight_range, maxval=weight_range), trainable=trainable, name='b_' + name_suffix)
        return weights, biases
    
    def print_log(self, log):
        if self.thread_no == 0:
            print log
            
    def build_network(self, name, network, trainable, num_actions):
        if network == 'nips':
            self.build_network_nips(name, trainable, num_actions)
        else:
            self.build_network_nature(name, trainable, num_actions)
        
    def build_network_nature(self, name, trainable, num_actions):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x)
        
            x_normalized = tf.to_float(x) / 255.0
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
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
        
    def build_network_nips(self, name, trainable, num_actions):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # First layer takes a screen, and shrinks by 2x
            x = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x)
        
            x_normalized = tf.to_float(x) / 255.0
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
        
        self.x = x
        self.y = y
        self.variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

    def get_vars(self):
        return self.variables
    
    def prepare_global(self, rms_decay, rms_epsilon):
        global_sess = new_session()
        global_learning_rate = tf.placeholder("float")
        global_optimizer = tf.train.RMSPropOptimizer(global_learning_rate, decay=rms_decay, epsilon=rms_epsilon)
        global_vars = self.get_vars()
    
        return global_sess, global_vars, global_optimizer, global_learning_rate
    
    def init_global(self, global_sess):
        global_sess.run(tf.initialize_all_variables())
