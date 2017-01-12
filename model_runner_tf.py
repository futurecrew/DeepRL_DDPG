import numpy as np
import os
import random
import math
import time
import threading
import traceback
import pickle
import tensorflow as tf
from network_model.model import Model, new_session
from network_model.model_torcs import ModelTorcs
from network_model.model_torcs2 import ModelTorcs2

class ModelRunnerTF(object):
    def __init__(self, args,  action_type_no, batch_dimension, thread_no):
        self.args = args
        learning_rate = args.learning_rate
        rms_decay = args.rms_decay
        rms_epsilon =  args.rms_epsilon
        self.network = args.network
        self.thread_no = thread_no
        
        self.train_batch_size = args.train_batch_size
        self.discount_factor = args.discount_factor
        self.action_type_no = action_type_no
        self.be = None
        self.action_mat = np.zeros((self.train_batch_size, self.action_type_no))
        tf.logging.set_verbosity(tf.logging.WARN)
        
        self.sess = new_session()
        self.init_models(self.network, action_type_no, learning_rate, rms_decay, rms_epsilon)

    def init_models(self, network, action_type_no, learning_rate, rms_decay, rms_epsilon):        
        with tf.device(self.args.device):
            model_policy = ModelTorcs2(self.args, "policy", True, action_type_no, self.thread_no)
            model_target = ModelTorcs2(self.args, "target", False, action_type_no, self.thread_no)
            
            self.model_policy = model_policy
    
            self.x_in = model_policy.x_in
            self.action_in = model_policy.action_in
            self.actor_y = model_policy.actor_y
            self.critic_y = model_policy.critic_y
            self.actor_vars = model_policy.actor_vars
            self.critic_vars = model_policy.critic_vars
            
            self.x_in_target = model_target.x_in
            self.action_in_target = model_target.action_in
            self.actor_y_target = model_target.actor_y
            self.critic_y_target = model_target.critic_y
            self.actor_vars_target = model_target.actor_vars
            self.critic_vars_target = model_target.critic_vars

            # build the variable copy ops
            self.update_t = tf.placeholder(tf.float32, 1)
            self.update_target_list = []
            for i in range(0, len(self.actor_vars_target)):
                self.update_target_list.append(self.actor_vars_target[i].assign(self.update_t * self.actor_vars[i] + (1-self.update_t) * self.actor_vars_target[i]))
            for i in range(0, len(self.critic_vars_target)):
                self.update_target_list.append(self.critic_vars_target[i].assign(self.update_t * self.critic_vars[i] + (1-self.update_t) * self.critic_vars_target[i]))
            self.update_target = tf.group(*self.update_target_list)
    
            self.critic_y_ = tf.placeholder(tf.float32, [None, action_type_no])
            #self.critic_y_ = tf.placeholder(tf.float32, [None, 1])
            self.critic_grads_in = tf.placeholder(tf.float32, [None, action_type_no])
            
            critic_loss = tf.reduce_mean(tf.square(self.critic_y_ - self.critic_y))
    
            #optimizer_critic = tf.train.RMSPropOptimizer(0.001, decay=rms_decay, epsilon=rms_epsilon)
            optimizer_critic = tf.train.AdamOptimizer(0.001)
            
            #self.critic_grads = optimizer_critic.compute_gradients(self.critic_y, self.action_in)
            self.critic_grads = tf.gradients(self.critic_y, self.action_in)
            #self.critic_grads2 = tf.gradients(self.critic_y, self.x_in)
            self.critic_step = optimizer_critic.minimize(loss = critic_loss, var_list = self.critic_vars)

            #optimizer_actor = tf.train.RMSPropOptimizer(0.0001, decay=rms_decay, epsilon=rms_epsilon)
            optimizer_actor = tf.train.AdamOptimizer(0.0001)
            
            gvs = optimizer_actor.compute_gradients(self.actor_y, var_list=self.actor_vars, grad_loss=-1 * self.critic_grads_in)
            self.actor_step = optimizer_actor.apply_gradients(gvs)
            
            self.saver = tf.train.Saver(max_to_keep=100)
            self.sess.run(tf.initialize_all_variables())
            self.sess.run(self.update_target, feed_dict={
                self.update_t: [1.0]
            })

    def clip_reward(self, reward):
        if reward > self.args.clip_reward_high:
            return self.args.clip_reward_high
        elif reward < self.args.clip_reward_low:
            return self.args.clip_reward_low
        else:
            return reward

    def predict(self, history_buffer):
        return self.sess.run(self.actor_y, {self.x_in: history_buffer})[0]
    
    def print_weights(self):
        for var in self.actor_vars:
            print ''
            print '[ ' + var.name + ']'
            print self.sess.run(var)
        
    def train(self, minibatch, replay_memory, learning_rate, debug):
        global global_step_no

        prestates, actions, rewards, poststates, terminals = minibatch
        
        actions_post = self.sess.run(self.actor_y_target, feed_dict={
                self.x_in_target: poststates
            })
        
        y2 = self.sess.run(self.critic_y_target, feed_dict={
            self.x_in_target: poststates, 
            self.action_in_target: actions_post
        })
        
        y_ = np.zeros((self.train_batch_size, self.action_type_no))
        #y_ = np.zeros((self.train_batch_size, 1))
        
        for i in range(self.train_batch_size):
            if self.args.clip_reward:
                reward = self.clip_reward(rewards[i])
            else:
                reward = rewards[i]
            if terminals[i]:
                y_[i] = reward
            else:
                y_[i] = reward + self.discount_factor * y2[i]

        #_, critic_y_value, x_norm_value, h_critic_conv1_value, h_critic_conv2_value, h_critic_fc1_value, h_critic_concat_value = self.sess.run([self.critic_step, self.critic_y, self.model_policy.x_normalized, self.model_policy.h_critic_conv1, self.model_policy.h_critic_conv2, self.model_policy.h_critic_fc1, self.model_policy.h_critic_concat], feed_dict={
        self.sess.run([self.critic_step], feed_dict={
            self.x_in: prestates,
            self.action_in: actions,
            self.critic_y_: y_
        })
        
        actor_y_value = self.sess.run(self.actor_y, feed_dict={
            self.x_in: prestates,
        })

        critic_grads_value = self.sess.run(self.critic_grads, feed_dict= {
            self.x_in: prestates,
            self.action_in: actor_y_value
        })
        
        self.sess.run(self.actor_step, feed_dict={
            self.x_in: prestates,
            self.critic_grads_in: critic_grads_value[0]
        })
        pass

    def update_model(self):
        self.sess.run(self.update_target, feed_dict={
            self.update_t: [0.001]
        })
        
    def load(self, fileName):
        self.saver.restore(self.sess, fileName)
        self.update_model()
        
    def save(self, fileName):
        self.saver.save(self.sess, fileName)
        

