import os
import numpy as np
import tensorflow as tf
from .DDPG import Actor, Critic, Memory

ACTION_DIM = 1
ACTION_BOUND = [-1, 1]
STATE_DIM = int(os.environ['NUM_SENSORS'])
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
GAMMA = 0.9  # reward discount
MEMORY_CAPACITY = 5000



class Car:
	def __init__(self, name, iden):
		# with tf.name_scope('S'):
		# 	S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
		# with tf.name_scope('R'):
		# 	R = tf.placeholder(tf.float32, [None, 1], name='r')
		# with tf.name_scope('S_'):
		# 	S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')
		self.id = iden
		self.name = name
		self.path = ""
		self.tmp_path = ""
		self.actor = {}
		self.critic = {}
		self.saver = {}
		self.sess = {}
		# print(ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
		# with tf.variable_scope(self.name):
		# 	self.actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, "Actor_"+self.name)
		# 	self.critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, self.actor.a, self.actor.a_, "Critic_"+self.name)
		# 	self.actor.add_grad_to_graph(self.critic.a_grads)

	def build_train(self):
		self.s = np.array([1,1,1,1,1])
		self.r_ = 0
		self.ep_step = 0
		self.var = 2.
		self.max_point = 3
		self.count_finish = 0

		# self.sess.run(tf.global_variables_initializer())
		# self.actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, "Actor_"+self.name)
		# self.critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, self.actor.a, self.actor.a_, "Critic_"+self.name)
		# self.actor.add_grad_to_graph(self.critic.a_grads)
		self.M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

	def restore(self, path):
		# self.save = tf.train.Saver()
		self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
		pass

if __name__ == '__main__':
	car = Car()