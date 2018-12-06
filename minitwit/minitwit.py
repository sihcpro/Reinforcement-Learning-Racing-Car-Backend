# -*- coding: utf-8 -*-
"""
	MiniTwit
	~~~~~~~~

	A microblogging application written with Flask and sqlite3.

	:copyright: © 2010 by the Pallets team.
	:license: BSD, see LICENSE for more details.
"""

import time, sys, os, shutil, json
import numpy as np
import tensorflow as tf
from .DDPG import Actor, Critic, Memory
from .car import Car

from sqlite3 import dbapi2 as sqlite3
from hashlib import md5
from datetime import datetime
from flask import Flask, request, session, url_for, redirect, \
	 render_template, abort, g, flash, _app_ctx_stack
from werkzeug import check_password_hash, generate_password_hash


# # configuration
# DATABASE = '/tmp/minitwit.db'
# PER_PAGE = 30
# DEBUG = False
# SECRET_KEY = b'_5#y2L"F4Q8z\n\xec]/'

# create our little application :)
app = Flask('minitwit')
app.config.from_object(__name__)
app.config.from_envvar('MINITWIT_SETTINGS', silent=True)


# def get_db():
# 	"""Opens a new database connection if there is none yet for the
# 	current application context.
# 	"""
# 	top = _app_ctx_stack.top
# 	if not hasattr(top, 'sqlite_db'):
# 		top.sqlite_db = sqlite3.connect(app.config['DATABASE'])
# 		top.sqlite_db.row_factory = sqlite3.Row
# 	return top.sqlite_db


# @app.teardown_appcontext
# def close_database(exception):
# 	"""Closes the database again at the end of the request."""
# 	top = _app_ctx_stack.top
# 	if hasattr(top, 'sqlite_db'):
# 		top.sqlite_db.close()


# def init_db():
# 	"""Initializes the database."""
# 	db = get_db()
# 	with app.open_resource('schema.sql', mode='r') as f:
# 		db.cursor().executescript(f.read())
# 	db.commit()


# @app.cli.command('initdb')
# def initdb_command():
# 	"""Creates the database tables."""
# 	init_db()
# 	print('Initialized the database.')


# def query_db(query, args=(), one=False):
# 	"""Queries the database and returns a list of dictionaries."""
# 	cur = get_db().execute(query, args)
# 	rv = cur.fetchall()
# 	return (rv[0] if rv else None) if one else rv


# def get_user_id(username):
# 	"""Convenience method to look up the id for a username."""
# 	rv = query_db('select user_id from user where username = ?',
# 				  [username], one=True)
# 	return rv[0] if rv else None


# def format_datetime(timestamp):
# 	"""Format a timestamp for display."""
# 	return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d @ %H:%M')


# def gravatar_url(email, size=80):
# 	"""Return the gravatar image for the given email address."""
# 	return 'https://www.gravatar.com/avatar/%s?d=identicon&s=%d' % \
# 		(md5(email.strip().lower().encode('utf-8')).hexdigest(), size)


# @app.before_request
# def before_request():
# 	g.user = None
# 	if 'user_id' in session:
# 		# g.user = query_db('select * from user where user_id = ?',
# 		#                   [session['user_id']], one=True)
# 		next


# @app.route('/')
# def timeline():
# 	"""Shows a users timeline or if no user is logged in it will
# 	redirect to the public timeline.  This timeline shows the user's
# 	messages as well as all the messages of followed users.
# 	"""
# 	if not g.user:
# 		return redirect(url_for('public_timeline'))
# 	return render_template('timeline.html', messages=query_db('''
# 		select message.*, user.* from message, user
# 		where message.author_id = user.user_id and (
# 			user.user_id = ? or
# 			user.user_id in (select whom_id from follower
# 									where who_id = ?))
# 		order by message.pub_date desc limit ?''',
# 		[session['user_id'], session['user_id'], PER_PAGE]))


# @app.route('/public')
# def public_timeline():
# 	"""Displays the latest messages of all users."""
# 	return render_template('timeline.html', messages=query_db('''
# 		select message.*, user.* from message, user
# 		where message.author_id = user.user_id
# 		order by message.pub_date desc limit ?''', [PER_PAGE]))


# @app.route('/<username>')
# def user_timeline(username):
# 	"""Display's a users tweets."""
# 	profile_user = query_db('select * from user where username = ?',
# 							[username], one=True)
# 	if profile_user is None:
# 		abort(404)
# 	followed = False
# 	if g.user:
# 		followed = query_db('''select 1 from follower where
# 			follower.who_id = ? and follower.whom_id = ?''',
# 			[session['user_id'], profile_user['user_id']],
# 			one=True) is not None
# 	return render_template('timeline.html', messages=query_db('''
# 			select message.*, user.* from message, user where
# 			user.user_id = message.author_id and user.user_id = ?
# 			order by message.pub_date desc limit ?''',
# 			[profile_user['user_id'], PER_PAGE]), followed=followed,
# 			profile_user=profile_user)


# @app.route('/<username>/follow')
# def follow_user(username):
# 	"""Adds the current user as follower of the given user."""
# 	if not g.user:
# 		abort(401)
# 	whom_id = get_user_id(username)
# 	if whom_id is None:
# 		abort(404)
# 	db = get_db()
# 	db.execute('insert into follower (who_id, whom_id) values (?, ?)',
# 			  [session['user_id'], whom_id])
# 	db.commit()
# 	flash('You are now following "%s"' % username)
# 	return redirect(url_for('user_timeline', username=username))


# @app.route('/<username>/unfollow')
# def unfollow_user(username):
# 	"""Removes the current user as follower of the given user."""
# 	if not g.user:
# 		abort(401)
# 	whom_id = get_user_id(username)
# 	if whom_id is None:
# 		abort(404)
# 	db = get_db()
# 	db.execute('delete from follower where who_id=? and whom_id=?',
# 			  [session['user_id'], whom_id])
# 	db.commit()
# 	flash('You are no longer following "%s"' % username)
# 	return redirect(url_for('user_timeline', username=username))


# @app.route('/add_message', methods=['POST'])
# def add_message():
# 	"""Registers a new message for the user."""
# 	if 'user_id' not in session:
# 		abort(401)
# 	if request.form['text']:
# 		db = get_db()
# 		db.execute('''insert into message (author_id, text, pub_date)
# 		  values (?, ?, ?)''', (session['user_id'], request.form['text'],
# 								int(time.time())))
# 		db.commit()
# 		flash('Your message was recorded')
# 	return redirect(url_for('timeline'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
# 	"""Logs the user in."""
# 	if g.user:
# 		return redirect(url_for('timeline'))
# 	error = None
# 	if request.method == 'POST':
# 		user = query_db('''select * from user where
# 			username = ?''', [request.form['username']], one=True)
# 		if user is None:
# 			error = 'Invalid username'
# 		elif not check_password_hash(user['pw_hash'],
# 									 request.form['password']):
# 			error = 'Invalid password'
# 		else:
# 			flash('You were logged in')
# 			session['user_id'] = user['user_id']
# 			return redirect(url_for('timeline'))
# 	return render_template('login.html', error=error)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
# 	"""Registers the user."""
# 	if g.user:
# 		return redirect(url_for('timeline'))
# 	error = None
# 	if request.method == 'POST':
# 		if not request.form['username']:
# 			error = 'You have to enter a username'
# 		elif not request.form['email'] or \
# 				'@' not in request.form['email']:
# 			error = 'You have to enter a valid email address'
# 		elif not request.form['password']:
# 			error = 'You have to enter a password'
# 		elif request.form['password'] != request.form['password2']:
# 			error = 'The two passwords do not match'
# 		elif get_user_id(request.form['username']) is not None:
# 			error = 'The username is already taken'
# 		else:
# 			db = get_db()
# 			db.execute('''insert into user (
# 			  username, email, pw_hash) values (?, ?, ?)''',
# 			  [request.form['username'], request.form['email'],
# 			   generate_password_hash(request.form['password'])])
# 			db.commit()
# 			flash('You were successfully registered and can login now')
# 			return redirect(url_for('login'))
# 	return render_template('register.html', error=error)


# @app.route('/logout')
# def logout():
# 	"""Logs the user out."""
# 	flash('You were logged out')
# 	session.pop('user_id', None)
# 	return redirect(url_for('public_timeline'))


# # add some filters to jinja
# app.jinja_env.filters['datetimeformat'] = format_datetime
# app.jinja_env.filters['gravatar'] = gravatar_url










###############################################################################


CAR = {}
COLUMN_NAME = []
PRE_COLUMNS = ["save", "car_name", "move", "rad", "x", "y", "speed", "point", "status"]
AFT_COLUMS = ["max_leng_sensor"]

NUM_SENSORS = int(os.environ['NUM_SENSORS'])
NUM_VARIABLES = NUM_SENSORS + (len(PRE_COLUMNS) + len(AFT_COLUMS))

def make_column_name(num):
	global NUM_SENSORS
	global NUM_VARIABLES
	global COLUMN_NAME
	global PRE_COLUMNS
	global AFT_COLUMS

	column_rest = num - (len(PRE_COLUMNS) + len(AFT_COLUMS))

	sensor_columns = []
	if column_rest % 2 == 1:
		sensor_columns.append("m")
	for i in range(round(column_rest/2)):
		index = str(i+1)
		sensor_columns = ["l"+index] + sensor_columns
		sensor_columns.append( "r"+index )

	COLUMN_NAME = PRE_COLUMNS + sensor_columns + AFT_COLUMS
	NUM_VARIABLES = num
	NUM_SENSORS = NUM_VARIABLES - (len(PRE_COLUMNS) + len(AFT_COLUMS))
	os.environ['NUM_SENSORS'] = str(NUM_SENSORS)
	print(NUM_SENSORS, " sensor")

make_column_name(NUM_VARIABLES)


ACTION_DIM = 1
ACTION_BOUND = [-1, 1]
STATE_DIM = int(os.environ['NUM_SENSORS'])
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
GAMMA = 0.9  # reward discount
MEMORY_CAPACITY = 5000

PATH = './save'
TMP_PATH = os.path.join(PATH, "tmp")
mark_point = [20, 30, 40, 50, 60]
MAX_CAR = 1

graph = tf.get_default_graph()
# with graph.as_default():
# 	sess = tf.Session()
# 	actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, 'Actor')
# 	critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_, 'Critic')
# 	actor.add_grad_to_graph(critic.a_grads)
# 	sess.run(tf.global_variables_initializer())
# 	saver = tf.train.Saver()

def initPath(car):
	global mark_point
	global PATH

	car.path = os.path.join(PATH, car.name)
	car.tmp_path = os.path.join(PATH, car.name, "tmp")

	if not os.path.isdir(car.path):
		os.mkdir(car.path)
	if not os.path.isdir(car.tmp_path):
		os.mkdir(car.tmp_path)

	for mark in mark_point:
		mark_path = os.path.join(car.path, str(mark))
		if not os.path.isdir(mark_path):
			os.mkdir(mark_path)

# sess = tf.Session()
# actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, "Actor")
def initInviroment(car):
	global sess
	# global actor
	# global critic
	global saver

	if car.name == 'car1':
		actor_name = 'Actor'
		critic_name = 'Critic'
	else:
		actor_name = "Actor_"+car.name
		critic_name = "Critic_"+car.name

	# print('initInviroment', actor_name)

	with graph.as_default():
		car.sess = tf.Session()
		car.actor = Actor(car.sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, actor_name)
		car.critic = Critic(car.sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, car.actor.a, car.actor.a_, critic_name)
		car.actor.add_grad_to_graph(car.critic.a_grads)
		car.sess.run(tf.global_variables_initializer())
		car.saver = tf.train.Saver()
		# car.sess.run(tf.global_variables_initializer())
# car = Car('Default', -1)
# initInviroment(car)

@app.route('/init/<car_name>', methods=['GET'])
def initCar(car_name):
	global CAR

	if car_name in CAR.keys():
		print('Car exist :', CAR[car_name].name)
		return json.dumps({'message': 'Init car', 'exist': True, 'readyState': 4, 'status': 200})
	else:
		print('Init car  :', car_name)
		car = Car(car_name, len(CAR)-1)
		initPath(car)
		initInviroment(car)
		CAR[car_name] = car

		for key in CAR:
			print('init', 'car', CAR[key], 'actor', CAR[key].actor)

		return json.dumps({'message': 'Init car', 'exist': False, 'readyState': 4, 'status': 200})

@app.route('/get-train/<car_name>', methods=['GET'])
def getTrain(car_name):
	global CAR

	car = {}
	if not car_name in CAR.keys():
		initCar(car_name)
	car = CAR[car_name]

	car.build_train()

	return json.dumps({'message': 'Train car', 'exist': False, 'readyState': 4, 'status': 200})

VAR_MIN = 0.1
STATE_DIM = 5
BATCH_SIZE = 16
@app.route('/train/<data>', methods=['GET'])
def train(data):
	global mark_point

	global CAR
	global ACTION_BOUND
	global BATCH_SIZE
	global VAR_MIN
	global STATE_DIM
	global MEMORY_CAPACITY

	with graph.as_default():
		tmp = data.split(',')
		car_name = tmp[1]
		car = CAR[car_name]

		sensors = tmp[len(PRE_COLUMNS):NUM_VARIABLES - len(AFT_COLUMS)]
		max_leng_sensor = int(tmp[-1])
		for i in range(len(sensors)):
			sensors[i] = float(sensors[i]) / max_leng_sensor
		# print(sensors, "/", max_leng_sensor)

		# control exploration
		# Added exploration noise
		a = car.actor.choose_action(car.s)
		# add randomness to action selection for exploration
		a = np.clip(np.random.normal(a, car.var), *ACTION_BOUND)
		s_ = np.array(sensors)
		done = False


		if (tmp[8] == '0'):
			r = -1
			done = True
		else:
			if int(tmp[7]) - car.r_ >= 0:
				r = 0
				car.r_ = int(tmp[7])
				if car.r_ >= car.max_point:
					done = True
					car.count_finish += 1
					if car.count_finish == 10:
						car.max_point += 1
						car.count_finish = 0
						if car.r_ in mark_point:
							file_path = os.path.join(car.path, str(car.r_), "save"+str(car.r_))
							car.saver.save(car.sess, file_path, write_meta_graph=False)
							print(" *******************  Save", car.r_," *****************")
							time.sleep(1)
			else:
				r = -1
				done = True

		if tmp[0] == "1":
			file_name = str(car.max_point) + "." + time.strftime("%a,%d-%b-%Y,%H:%M:%S,", time.gmtime())
			car.saver.save(car.sess, os.path.join(car.tmp_path, file_name), write_meta_graph=False)
			print(" *******************  Save tmp ", file_name," *****************")
			time.sleep(1)

		car.M.store_transition(car.s, a, r, s_)

		if car.M.pointer > MEMORY_CAPACITY:
			car.var = max([car.var*.9995, VAR_MIN])    # decay the action randomness
			b_M = car.M.sample(BATCH_SIZE)
			b_s = b_M[:, :STATE_DIM]
			b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
			b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
			b_s_ = b_M[:, -STATE_DIM:]

			car.critic.learn(b_s, b_a, b_r, b_s_)
			car.actor.learn(b_s)

		car.s = s_
		car.ep_step += 1

		print(r, '|', car.r_, '|', car.max_point, '|', car.count_finish, '| Return : %2.2f' % a, '| Steps: %i' % int(car.ep_step), '| Memory: %.d' % car.M.pointer, '| Explore: %.2f' % car.var )
		print()

		message = '| Memory: %.d' % car.M.pointer + '| Explore: %.2f' % car.var

		if not done:
			return json.dumps({'move': float(a[0]), 'message': message, 'readyState': 4, 'status': 200})
		else:
			car.r_ = 0
			car.ep_step = 0
			return json.dumps({'move': -2, 'message': 'reset', 'readyState': 4, 'status': 200})


@app.route('/get-load/<car_name>', methods=['GET'])
def getLoad(car_name):
	global CAR

	returnPackage = {'readyState': 4}

	with graph.as_default():
		if car_name in CAR.keys():
			car = CAR[car_name]
			if not os.path.isdir(car.path):
				os.mkdir(car.path)
				returnPackage['file'] = []
				print("create dir")
			else:
				print('path', car.path)
				print('file', os.listdir(car.path))
				file = [f for f in os.listdir(car.path)]
				tmp = []
				for f in file:
					if len(os.listdir(os.path.join(car.path, f))) > 0:
						tmp.append({f: f})
				print("load dir", tmp)
				returnPackage['file'] = tmp
		else:
			initCar(car_name)

		return json.dumps(returnPackage)

@app.route('/load/<data>', methods=['GET'])
def load(data):
	global CAR

	with graph.as_default():
		data = data.split(',')
		car = CAR[data[0]]
		print(car.name, car.sess, car)
		car.saver.restore(car.sess, tf.train.latest_checkpoint(car.path+'/'+data[1]))
		return json.dumps({'message': 'done', 'readyState': 4, 'status': 200})

@app.route('/run/<data>', methods=['GET'])
def run(data):
	global CAR

	data = data.split(',')
	car = CAR[data[0]]
	data = [ float(x) for x in data[1:] ]
	car.s = np.array(data)
	a = car.actor.choose_action(car.s)

	return json.dumps({'move': float(a[0]), 'readyState': 4, 'status': 200})



