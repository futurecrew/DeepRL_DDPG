import tensorflow as tf
import numpy as np
from model import Model

class ModelTorcs2(Model):
    def build_network(self, name, network, trainable, action_type_no):
        self.print_log("Building network for %s trainable=%s" % (name, trainable))
    
        with tf.variable_scope(name):
            # Actor network
            input_img_size = self.screen_height * self.screen_width
            x_in = tf.placeholder(tf.float32, shape=[None, input_img_size], name="screens")
            self.print_log(x_in)
        
            W_actor_fc1, b_actor_fc1 = self.make_layer_variables([input_img_size, 300], trainable, "actor_fc1")    
            h_actor_fc1 = tf.nn.relu(tf.matmul(x_in, W_actor_fc1) + b_actor_fc1, name="h_actor_fc1")
            self.print_log(h_actor_fc1)
    
            W_actor_fc2, b_actor_fc2 = self.make_layer_variables([300, 600], trainable, "actor_fc2")    
            h_actor_fc2 = tf.nn.relu(tf.matmul(h_actor_fc1, W_actor_fc2) + b_actor_fc2, name="h_actor_fc2")
            self.print_log(h_actor_fc2)
            
            weight_range = 3 * 10 ** -3
            W_steering, b_steering = self.make_layer_variables([600, 1], trainable, "steering", weight_range)    
            h_steering = tf.tanh(tf.matmul(h_actor_fc2, W_steering) + b_steering, name="h_steering")
            self.print_log(h_steering)
            
            W_acc, b_acc = self.make_layer_variables([600, 1], trainable, "acc", weight_range)
            h_acc = tf.sigmoid(tf.matmul(h_actor_fc2, W_acc) + b_acc, name="h_acc")
            self.print_log(h_acc)
            
            W_brake, b_brake = self.make_layer_variables([600, 1], trainable, "brake", weight_range)    
            h_brake = tf.tanh(tf.matmul(h_actor_fc2, W_brake) + b_brake, name="h_brake")
            self.print_log(h_brake)

            #actor_y = tf.concat(1, [h_steering, h_acc, h_brake])
            actor_y = tf.concat(1, [h_steering, h_acc])
            self.print_log(actor_y)


            # Critic network    
            action_in = tf.placeholder(tf.float32, shape=[None, self.action_type_no], name="actions")

            W_critic_fc1, b_critic_fc1 = self.make_layer_variables([input_img_size, 300], trainable, "critic_fc1")    
            h_critic_fc1 = tf.nn.relu(tf.matmul(x_in, W_critic_fc1) + b_critic_fc1, name="h_critic_fc1")
            self.print_log(h_critic_fc1)

            h_concat = tf.concat(1, [h_critic_fc1, action_in])
        
            W_critic_fc2, b_critic_fc2 = self.make_layer_variables([300 + self.action_type_no, 600], trainable, "critic_fc2")    
            h_critic_fc2 = tf.nn.relu(tf.matmul(h_concat, W_critic_fc2) + b_critic_fc2, name="h_critic_fc2")
            self.print_log(h_critic_fc2)
    
            W_critic_fc3, b_critic_fc3 = self.make_layer_variables([600, self.action_type_no], trainable, "critic_fc3")
            critic_y = tf.nn.relu(tf.matmul(h_critic_fc2, W_critic_fc3) + b_critic_fc3, name="h_critic_fc3")
            self.print_log(critic_y)

                        
        self.x_in = x_in
        self.action_in = action_in
        self.actor_y = actor_y
        self.critic_y = critic_y
        self.actor_vars = [W_actor_fc1, b_actor_fc1, W_actor_fc2, b_actor_fc2, W_steering, b_steering, W_acc, b_acc, W_brake, b_brake]
        self.critic_vars = [W_critic_fc1, b_critic_fc1, W_critic_fc2, b_critic_fc2, W_critic_fc3, b_critic_fc3]
                          
