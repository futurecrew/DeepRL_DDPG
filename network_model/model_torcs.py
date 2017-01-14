import tensorflow as tf
import numpy as np
from model import Model

class ModelTorcs(Model):
    def build_network(self, name, network, trainable, action_type_no):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # Actor network
            input_img_size = self.screen_height * self.screen_width
            x_in = tf.placeholder(tf.uint8, shape=[None, self.screen_height, self.screen_width, self.history_len], name="screens")
            self.print_log(x_in)
        
            self.x_normalized = tf.to_float(x_in) / 255.0
            self.print_log(self.x_normalized)
    
            # Second layer convolves 16 6x6 filters with stride 3 with relu
            W_actor_conv1, b_actor_conv1 = self.make_layer_variables([6, 6, self.history_len, 16], trainable, "actor_conv1")
            h_actor_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_normalized, W_actor_conv1, strides=[1, 3, 3, 1], padding='VALID') + b_actor_conv1, name="h_actor_conv1")
            self.print_log(h_actor_conv1)
    
            # Third layer convolves 16 3x3 filters with stride 2 with relu
            W_actor_conv2, b_actor_conv2 = self.make_layer_variables([3, 3, 16, 16], trainable, "actor_conv2")
            h_actor_conv2 = tf.nn.relu(tf.nn.conv2d(h_actor_conv1, W_actor_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_actor_conv2, name="h_actor_conv2")
            self.print_log(h_actor_conv2)
    
            # Third layer convolves 32 3x3 filters with stride 2 with relu
            W_actor_conv3, b_actor_conv3 = self.make_layer_variables([3, 3, 16, 32], trainable, "actor_conv3")
            h_actor_conv3 = tf.nn.relu(tf.nn.conv2d(h_actor_conv2, W_actor_conv3, strides=[1, 2, 2, 1], padding='VALID') + b_actor_conv3, name="h_actor_conv3")
            self.print_log(h_actor_conv3)
    
            conv_out_size = np.prod(h_actor_conv3._shape[1:]).value
    
            h_actor_conv3_flat = tf.reshape(h_actor_conv3, [-1, conv_out_size], name="h_actor_conv3_flat")
            self.print_log(h_actor_conv3_flat)
    
            # Fourth layer is fully connected with 256 relu units
            W_actor_fc1, b_actor_fc1 = self.make_layer_variables([conv_out_size, 600], trainable, "actor_fc1")
    
            h_actor_fc1 = tf.nn.relu(tf.matmul(h_actor_conv3_flat, W_actor_fc1) + b_actor_fc1, name="h_actor_fc1")
            self.print_log(h_actor_fc1)
            
            weight_range = 3 * 10 ** -3
            W_steering, b_steering = self.make_layer_variables([600, 1], trainable, "steering", weight_range)    
            h_steering = tf.tanh(tf.matmul(h_actor_fc1, W_steering) + b_steering, name="h_steering")
            self.print_log(h_steering)
            
            W_acc, b_acc = self.make_layer_variables([600, 1], trainable, "acc", weight_range)    
            h_acc = tf.sigmoid(tf.matmul(h_actor_fc1, W_acc) + b_acc, name="h_acc")
            self.print_log(h_acc)
            
            W_brake, b_brake = self.make_layer_variables([600, 1], trainable, "brake", weight_range)    
            h_brake = tf.sigmoid(tf.matmul(h_actor_fc1, W_brake) + b_brake, name="h_brake")
            self.print_log(h_brake)

            if action_type_no == 3:
                actor_y = tf.concat(1, [h_steering, h_acc, h_brake])
            else:
                actor_y = tf.concat(1, [h_steering, h_acc])
            self.print_log(actor_y)


            # Critic network    
            action_in = tf.placeholder(tf.float32, shape=[None, self.action_type_no], name="actions")

            # Second layer convolves 16 6x6 filters with stride 3 with relu
            W_critic_conv1, b_critic_conv1 = self.make_layer_variables([6, 6, self.history_len, 16], trainable, "critic_conv1")
            self.h_critic_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_normalized, W_critic_conv1, strides=[1, 3, 3, 1], padding='VALID') + b_critic_conv1, name="h_critic_conv1")
            self.print_log(self.h_critic_conv1)
    
            # Third layer convolves 16 3x3 filters with stride 2 with relu
            W_critic_conv2, b_critic_conv2 = self.make_layer_variables([3, 3, 16, 16], trainable, "critic_conv2")
            self.h_critic_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_critic_conv1, W_critic_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_critic_conv2, name="h_critic_conv2")
            self.print_log(self.h_critic_conv2)
    
            # Third layer convolves 32 3x3 filters with stride 2 with relu
            W_critic_conv3, b_critic_conv3 = self.make_layer_variables([3, 3, 16, 32], trainable, "critic_conv3")
            self.h_critic_conv3 = tf.nn.relu(tf.nn.conv2d(self.h_critic_conv2, W_critic_conv3, strides=[1, 2, 2, 1], padding='VALID') + b_critic_conv3, name="h_critic_conv3")
            self.print_log(self.h_critic_conv3)
    
            conv_out_size = np.prod(self.h_critic_conv3._shape[1:]).value
    
            self.h_critic_conv3_flat = tf.reshape(self.h_critic_conv3, [-1, conv_out_size], name="h_critic_conv3_flat")
            self.print_log(self.h_critic_conv3_flat)
    
            self.h_critic_concat = tf.concat(1, [self.h_critic_conv3_flat, action_in])
    
            # Fourth layer is fully connected with 600 relu units
            W_critic_fc1, b_critic_fc1 = self.make_layer_variables([conv_out_size + action_type_no, 600], trainable, "critic_fc1")
    
            self.h_critic_fc1 = tf.nn.relu(tf.matmul(self.h_critic_concat, W_critic_fc1) + b_critic_fc1, name="h_critic_fc1")
            self.print_log(self.h_critic_fc1)
    
            W_critic_fc2, b_critic_fc2 = self.make_layer_variables([600, self.action_type_no], trainable, "critic_fc2")    
            #critic_y = tf.nn.relu(tf.matmul(self.h_critic_fc1, W_critic_fc2) + b_critic_fc2, name="h_critic_fc2")
            critic_y = tf.matmul(self.h_critic_fc1, W_critic_fc2) + b_critic_fc2
            self.print_log(critic_y)

                        
        self.x_in = x_in
        self.action_in = action_in
        self.actor_y = actor_y
        self.critic_y = critic_y
        self.actor_vars = [W_actor_conv1, b_actor_conv1, W_actor_conv2, b_actor_conv2, W_actor_fc1, b_actor_fc1, W_steering, b_steering, W_acc, b_acc, W_brake, b_brake]
        self.critic_vars = [W_critic_conv1, b_critic_conv1, W_critic_conv2, b_critic_conv2, W_critic_fc1, b_critic_fc1, W_critic_fc2, b_critic_fc2]
                          
