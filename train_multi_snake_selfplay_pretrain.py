# train a single snake agent using the baseline PPO algorithm
import sys

import tensorflow as tf
import gym

from multi_snake_2 import MultiSnake

import numpy as np
import subprocess as sp

import os
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

class Qnetwork():
    def __init__(self, h_size):
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


batch_size = 512 # How many experiences to use for each training step.
update_freq = 4 # How often to perform a training step.
y = .99 # Discount factor on the target Q-values
startE = 0.0001 # Starting chance of random action
endE = 0.0001 # Final chance of random action
annealing_steps = 500000. # How many steps of training to reduce startE to endE.
num_episodes = 500000 # How many episodes of game environment to train network with.
pre_train_steps = 50000 # How many steps of random actions before training begins.
max_epLength = 500 # The max allowed length of our episode.
load_model = True # Whether to load a saved model.
pretrained_model_path = "./dqn_single"
new_model_path = "./dqn_multi" # The path to save our model to.
h_size = 1296*2 # The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 # Rate to update target network toward primary network

def main():
    spacing = 22
    grid_dim = 15
    history = 4
    save_gif = True
    env = MultiSnake(num_agents=2, num_fruits=3, spacing=spacing, grid_dim=grid_dim, flatten_states=False,
                     reward_killed=-1.0, history=history, save_gif=save_gif)
    env.reset()
    #env = makegymwrapper(env, visualize=test_model)

    tf.reset_default_graph()
    mainQN = Qnetwork(h_size)
    targetQN = Qnetwork(h_size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables, tau)
    myBuffer = experience_buffer()

    #Set the rate of random action decrease. 
    e = startE
    stepDrop = (startE - endE) / annealing_steps

    #create lists to contain total rewards and steps per episode
    jList = []
    rList_agent1 = []
    rList_agent2 = []
    total_steps = 0

    #Make a path for our model to be saved in.
    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Pretrained Model...')
            ckpt = tf.train.get_checkpoint_state(new_model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            episodeBuffer = [experience_buffer(), experience_buffer()]

            #Reset environment and get first new observation
            s = env.reset()

            s_agent1 = s[1]
            s_agent2 = s[0]
            d_agent1 = False
            d_agent2 = False
            d = [False, False]
            pre_d = [False, False]
            win_index = None

            rAll_agent1 = 0
            rAll_agent2 = 0
            j = 0

            #The Q-Network
            while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                env.render()
                j += 1

                #print("sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent1 / 3.0]}): " + str(sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent1 / 3.0]})))
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    #a_agent1 = np.random.randint(0,4)
                    a_agent1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent1 / 3.0]})[0]
                else:
                    a_agent1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent1 / 3.0]})[0]

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    #a_agent2 = np.random.randint(0,4)
                    a_agent2 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent2 / 3.0]})[0]
                else:
                    a_agent2 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent2 / 3.0]})[0]

                s1, r, d, d_common = env.step([a_agent1, a_agent2])
                r_agent1 = r[0]
                r_agent2 = r[1]

                d_agent1 = d[0]
                d_agent2 = d[1]

                s1_agent1 = s1[1]
                s1_agent2 = s1[0]

                total_steps += 1
                if d_agent1 == False:
                    episodeBuffer[0].add(np.reshape(np.array([s_agent1,a_agent1,r_agent1,s1_agent1,d_agent1]),[1,5]))
                if d_agent2 == False:
                    episodeBuffer[1].add(np.reshape(np.array([s_agent2,a_agent2,r_agent2,s1_agent2,d_agent2]),[1,5]))
                '''
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) # Get a random batch of experiences.

                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size), Q1]

                        targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0] / 3.0),
                                                                    mainQN.targetQ:targetQ, 
                                                                    mainQN.actions:trainBatch[:,1]})
                        #print("Training Model")
                        updateTarget(targetOps, sess) #Update the target network toward the primary network.
                '''
                rAll_agent1 += r_agent1
                rAll_agent2 += r_agent2

                s_agent1 = s1_agent1
                s_agent2 = s1_agent2
                if d_common == True:
                    env.write_gif('./video/play_' + str(i) + '.gif')
                    win_index = pre_d.index(False)
                    break

                if j == max_epLength:
                    win_index = np.random.randint(0,2)

                pre_d[0] = d[0]
                pre_d[1] = d[1]

            myBuffer.add(episodeBuffer[win_index].buffer)
            jList.append(j)

            print("rAll_agent1: " + str(rAll_agent1))
            print("rAll_agent2: " + str(rAll_agent2))
            rList_agent1.append(rAll_agent1)
            rList_agent2.append(rAll_agent2)

            #Periodically save the model. 
            #if i % 500 == 0:
            #    saver.save(sess, new_model_path + '/model-' + str(i) + '.ckpt')
            #    print("Saved New Model")
            
        #saver.save(sess, new_model_path + '/model-' + str(i) + '.ckpt')

    print("Percent of succesful episodes: " + str(sum(rList_agent1)/num_episodes) + "%")
    print("Percent of succesful episodes: " + str(sum(rList_agent2)/num_episodes) + "%")


if __name__ == '__main__':
    main()