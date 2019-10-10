# train a single snake agent using the baseline PPO algorithm

import sys
from baselines import logger
# from baselines.ppo2 import ppo2
from ppo import ppo2
import tensorflow as tf
import gym
from baselines.a2c.utils import fc, conv, conv_to_fc
from baselines.common.distributions import make_pdtype
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines import bench, logger
from multi_snake import MultiSnake
from handle_args import handle_args

import numpy as np
import subprocess as sp

import os
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

# set environment variable to free gpu device
out = sp.check_output('./free_gpus.sh').split()[0]
if isinstance(out, bytes):
    out = out.decode()

os.environ["CUDA_VISIBLE_DEVICES"] = out

save_int = 1000
trainhistdir = 'train_multi_snake_selfplay/'
save_gif = False
test_model = True
statlist = []
stat = np.array([])
reloadlist = []
numepisodes = 0
lastnumepisodes = 0
maxepisodes = int(6e5)
max_episodes_timestep = 1000
tol_frac = .9
gamma = .9
lam = 1.0
buf_size = 2048
minibatch_size = 32
nminibatches = buf_size // minibatch_size

num_agents = 2
best_avg_reward = -1 * num_agents
min_reward = -1 * num_agents

giffn = trainhistdir + 'video.gif'

# handle command line arguments
test_model, save_gif = handle_args(test_model, save_gif)

if save_gif:
    maxepisodes = 3


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def findlastepisode(traindir):
    # list directory contents
    errstatfiles = os.listdir(traindir)
    
    # find files in directory that look like reward statistics files and extract their numbers
    errstatnumbers = []
    for x in errstatfiles:
        matchobj = re.match(r'(?:Reward_stat_)(\d+)(?:\.npz)',x)
        if matchobj is not None:
            errstatnumbers.append(int(matchobj.group(1)))
        
    # find highest number
    lastfilenumber = np.array(errstatnumbers).max().astype(int)
    
    return lastfilenumber    


class Qnetwork():
    def __init__(self, h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.imageIn = state_pov = tf.placeholder(shape=[None,15,15,4], dtype=tf.float32)
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=16, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=32, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2, num_outputs=32, kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv3, 2, 3)
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
    def __init__(self, buffer_size = 250000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


batch_size = 512 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 50000. #How many steps of training to reduce startE to endE.
num_episodes = 50000 #How many episodes of game environment to train network with.
pre_train_steps = 5000 #How many steps of random actions before training begins.
max_epLength = 100 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 1296*2 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

    
def main():
    # disable logging during testing
    if test_model:
        log_interval = int(1e20)
    else:
        log_interval = 1
    
    spacing = 22
    grid_dim = 15
    history = 4
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
    targetOps = updateTargetGraph(trainables,tau)
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
    if not os.path.exists(path):
        os.makedirs(path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            episodeBuffer = [experience_buffer(), experience_buffer()]

            #Reset environment and get first new observation
            s = env.reset()

            s_agent1 = s[0]
            s_agent2 = s[1]
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
                #env.render()
                j += 1

                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a_agent1 = np.random.randint(0,4)
                else:
                    a_agent1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent1 / 3.0]})[0]

                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a_agent2 = np.random.randint(0,4)
                else:
                    a_agent2 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:[s_agent2 / 3.0]})[0]

                s1, r, d, d_common = env.step([a_agent1, a_agent2])

                r_agent1 = r[0]
                r_agent2 = r[1]
                d_agent1 = d[0]
                d_agent2 = d[1]

                s1_agent1 = s1[0]
                s1_agent2 = s1[1]
                total_steps += 1

                if d_agent1 == False:
                    episodeBuffer[0].add(np.reshape(np.array([s_agent1,a_agent1,r_agent1,s1_agent1,d_agent1]),[1,5])) #Save the experience to our episode buffer.
                if d_agent2 == False:
                    episodeBuffer[1].add(np.reshape(np.array([s_agent2,a_agent2,r_agent2,s1_agent2,d_agent2]),[1,5]))

                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict, feed_dict={mainQN.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.imageIn:np.stack(trainBatch[:,3] / 3.0)})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        doubleQ = Q2[range(batch_size),Q1]
                        targetQ = trainBatch[:,2] + (y * doubleQ * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateModel, \
                            feed_dict={mainQN.imageIn:np.stack(trainBatch[:,0] / 3.0), mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                        #print("Training Model")
                        updateTarget(targetOps, sess) #Update the target network toward the primary network.

                rAll_agent1 += r_agent1
                rAll_agent2 += r_agent2
                s_agent1 = s1_agent1
                s_agent2 = s1_agent2

                if d_common == True:
                    win_index = pre_d.index(False)
                    break

                if j == max_epLength:
                    win_index = pre_d.index(False)

                pre_d[0] = d[0]
                pre_d[1] = d[1]

            myBuffer.add(episodeBuffer[win_index].buffer)
            jList.append(j)
            rList_agent1.append(rAll_agent1)
            rList_agent2.append(rAll_agent2)

            #Periodically save the model. 
            if i % 1000 == 0:
                saver.save(sess, path + '/model-' + str(i) + '.ckpt')
                print("Saved Model")

            if len(rList_agent1) % 10 == 0:
                print(total_steps, "agent1", np.mean(rList_agent1[-10:]), e)
            if len(rList_agent2) % 10 == 0:
                print(total_steps, "agent2", np.mean(rList_agent2[-10:]), e)
        print("")
        saver.save(sess, path + '/model-' + str(i) + '.ckpt')

    print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")


if __name__ == '__main__':
    main()