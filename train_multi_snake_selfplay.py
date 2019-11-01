'''
   Copyright 2019 Kimbring2
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

# train a single snake agent using the baseline PPO algorithm
import sys

import tensorflow as tf
import gym

from multi_snake import MultiSnake

import numpy as np
import subprocess as sp

import os
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random


class Qnetwork_init():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.imageIn = state_pov = tf.placeholder(shape=[None,15,15,4], dtype=tf.float32)
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=16, kernel_size=[3,3], stride=[1,1], 
                                 padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=[3,3], stride=[1,1], 
                                 padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=32, kernel_size=[3,3], stride=[1,1], 
                                 padding='VALID', biases_initializer=None)
            
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,4]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
            
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
            
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)
            
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
            
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class Qnetwork():
    def __init__(self, h_size, scope):
        with tf.variable_scope(scope):
            #The network recieves a frame from the game, flattened into an array.
            #It then resizes it and processes it through four convolutional layers.
            self.imageIn = state_pov = tf.placeholder(shape=[None,15,15,4], dtype=tf.float32)
            self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=16, kernel_size=[3,3], stride=[1,1], 
                                     padding='VALID', biases_initializer=None)
            self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=[3,3], stride=[1,1], 
                                     padding='VALID', biases_initializer=None)
            self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=32, kernel_size=[3,3], stride=[1,1], 
                                     padding='VALID', biases_initializer=None)
            
            #We take the output from the final convolutional layer and split it into separate advantage and value streams.
            self.streamAC, self.streamVC = tf.split(self.conv3, 2, 3)
            self.streamA = slim.flatten(self.streamAC)
            self.streamV = slim.flatten(self.streamVC)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.AW = tf.Variable(xavier_init([h_size//2,4]))
            self.VW = tf.Variable(xavier_init([h_size//2,1]))
            self.Advantage = tf.matmul(self.streamA, self.AW)
            self.Value = tf.matmul(self.streamV, self.VW)
            
            #Then combine them together to get our final Q-values.
            self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            self.predict = tf.argmax(self.Qout, 1)
            
            #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)
            
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
            
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer,size)), [size,5])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)


batch_size = 32 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 0.1 # Starting chance of random action
endE = 0.0001 # Final chance of random action
annealing_steps = 500000. # How many steps of training to reduce startE to endE.
num_episodes = 500000 # How many episodes of game environment to train network with.
pre_train_steps = 512 # How many steps of random actions before training begins.
max_epLength = 500 # The max allowed length of our episode.
load_model = False # Whether to load a saved model.
saving_path = "./dqn_multi" # The path to save our model to.
loding_path = "./dqn_single"
h_size = 1296 * 2 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network


def main():
    spacing = 22
    grid_dim = 15
    history = 4
    save_gif = False
    env = MultiSnake(num_agents=2, num_fruits=3, spacing=spacing, grid_dim=grid_dim, flatten_states=False,
                     reward_killed=-1.0, history=history, save_gif=save_gif)
    env.reset()
    #env = makegymwrapper(env, visualize=test_model)

    tf.reset_default_graph()
    mainQN = Qnetwork_init(h_size=h_size)
    targetQN = Qnetwork_init(h_size=h_size)

    mainQN_new = Qnetwork(h_size=h_size, scope="main_new")
    targetQN_new = Qnetwork(h_size=h_size, scope="target_new")

    mainQN_old = Qnetwork(h_size=h_size, scope="main_old")
    targetQN_old = Qnetwork(h_size=h_size, scope="target_old")

    init = tf.global_variables_initializer()

    trainables = tf.trainable_variables()
    variables_init_restore = [v for v in trainables if v.name.split('/')[0] not in ['main_new', 'target_new', 'main_old', 'target_old']]
    variables_new_restore = [v for v in trainables if v.name.split('/')[0] in ['main_new', 'target_new']]
    variables_old_restore = [v for v in trainables if v.name.split('/')[0] in ['main_old', 'target_old']]

    init_weights = [tf.assign(old, init) for (old, init) in zip(variables_old_restore, variables_init_restore)]
    update_weights = [tf.assign(old, new) for (old, new) in zip(variables_old_restore, variables_new_restore)]
    saver_new_model = tf.train.Saver(variables_new_restore)

    targetOps_new = updateTargetGraph(variables_new_restore, tau)

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE) / annealing_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList_agent_old = []
    rList_agent_new = []

    #Make a path for our model to be saved in.
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    with tf.Session() as sess:
        sess.run(init)

        saver_init = tf.train.Saver(variables_init_restore)
        ckpt_init = tf.train.get_checkpoint_state(loding_path)
        saver_init.restore(sess, ckpt_init.model_checkpoint_path)

        #print('Loading Initial Model...')
        #sess.run(init_weights)
        sess.run(update_weights)

        if load_model == True:
            print('Loading Saving Model...')
            ckpt = tf.train.get_checkpoint_state(saving_path)
            saver_new_model.restore(sess, ckpt.model_checkpoint_path)
            sess.run(update_weights)

        myBuffer = experience_buffer()
        total_steps = 0

        for i in range(num_episodes):
            win_num = [0, 0]

            for k in range(0, 100):
                episodeBuffer = [experience_buffer(), experience_buffer()]

                #Reset environment and get first new observation
                s = env.reset()

                s_agent_old = s[1]
                s_agent_new = s[0]
                s1_agent_old = s_agent_old
                s1_agent_new = s_agent_new
                d_agent_old = False
                d_agent_new = False
                d = [False, False]
                pre_d = [False, False]
                win_index = None

                rAll_agent_old = 0
                rAll_agent_new = 0
                j = 0

                #The Q-Network
                while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                    #env.render()
                    j += 1

                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                    #if np.random.rand(1) < e:
                        a_agent_old = np.random.randint(0,4)
                    else:
                        a_agent_old = sess.run(mainQN_old.predict, feed_dict = {mainQN_old.imageIn:[s_agent_old / 3.0]})[0]

                    if np.random.rand(1) < e or total_steps < pre_train_steps:
                    #if np.random.rand(1) < e:
                        a_agent_new = np.random.randint(0,4)
                    else:
                        a_agent_new = sess.run(mainQN_new.predict, feed_dict = {mainQN_new.imageIn:[s_agent_new / 3.0]})[0]

                    s1, r, d, d_common = env.step([a_agent_old, a_agent_new])

                    r_agent_old = r[0]
                    r_agent_new = r[1]

                    d_agent_old = d[0]
                    d_agent_new = d[1]

                    if ( (d[0] == False) & (d[1] == False) ):
                        s1_agent_old = s1[1]
                        s1_agent_new = s1[0]
                    elif (d[0] == False):
                        s1_agent_old = s1[1]
                        s1_agent_new = s1[0]
                    elif (d[0] == True):
                        s1_agent_old = s1[0]
                        s1_agent_new = s1[1]

                    total_steps += 1
                    if d_agent_old == False:
                        episodeBuffer[0].add(np.reshape(np.array([s_agent_old,a_agent_old,r_agent_old,s1_agent_old,d_agent_old]),[1,5]))
                    if d_agent_new == False:
                        episodeBuffer[1].add(np.reshape(np.array([s_agent_new,a_agent_new,r_agent_new,s1_agent_new,d_agent_new]),[1,5]))
                    
                    rAll_agent_old += r_agent_old
                    rAll_agent_new += r_agent_new

                    s_agent_old = s1_agent_old
                    s_agent_new = s1_agent_new
                    if d_common == True:
                        #env.write_gif('./video/play_' + str(k) + '.gif')
                        win_index = pre_d.index(False)
                        break

                    if j == max_epLength:
                        win_index = np.random.randint(0,2)

                    #print("pre_d: " + str(pre_d))
                    pre_d[0] = d[0]
                    pre_d[1] = d[1]

                win_num[win_index] = win_num[win_index] + 1

                print("total_steps: " + str(total_steps))
                print("len(myBuffer.buffer): " + str(len(myBuffer.buffer)))
                if total_steps > pre_train_steps:
                    for q in range(0, 16):
                        trainBatch = myBuffer.sample(batch_size) # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN_new.predict, feed_dict={mainQN_new.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        Q2 = sess.run(targetQN_new.Qout, feed_dict={targetQN_new.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size), Q1]

                        targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        _ = sess.run(mainQN_new.updateModel, feed_dict={mainQN_new.imageIn:np.stack(trainBatch[:,0] / 3.0),
                                                                        mainQN_new.targetQ:targetQ, 
                                                                        mainQN_new.actions:trainBatch[:,1]})
                        #print("Training New Model")
                        updateTarget(targetOps_new, sess) #Update the target network toward the primary network.

                myBuffer.add(episodeBuffer[1].buffer)
                jList.append(j)

                print("win_num: " + str(win_num))
                print("rAll_agent_old: " + str(rAll_agent_old))
                print("rAll_agent_new: " + str(rAll_agent_new))
                print("")
                rList_agent_old.append(rAll_agent_old)
                rList_agent_new.append(rAll_agent_new)

            if ( (win_num[1] > win_num[0]) & (win_num[1] - win_num[0] >= 5) ):
                print('Updating Weight...')
                sess.run(update_weights)

                myBuffer = experience_buffer()
                if e > endE:
                    e -= stepDrop

                total_steps = 0

            # Periodically save the model. 
            if i % 100 == 0:
                saver_new_model.save(sess, saving_path + '/model-' + str(i) + '.ckpt')
                print("Saved Model")
            
            if len(rList_agent_old) % 10 == 0:
                print(total_steps, "agent_old", np.mean(rList_agent_old[-10:]), e)

            if len(rList_agent_new) % 10 == 0:
                print(total_steps, "agent_new", np.mean(rList_agent_new[-10:]), e)
            
        saver_new_model.save(sess, saving_path + '/model-' + str(i) + '.ckpt')

    print("Percent of succesful episodes: " + str(sum(rList_agent_old) / num_episodes) + "%")
    print("Percent of succesful episodes: " + str(sum(rList_agent_new) / num_episodes) + "%")

if __name__ == '__main__':
    main()