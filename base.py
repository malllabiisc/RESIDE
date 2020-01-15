from helper import *
import tensorflow as tf

class Model(object):
	"""Abstracts a Tensorflow graph for a learning task.
	We use various Model classes as usual abstractions to encapsulate tensorflow
	computational graphs. Each algorithm you will construct in this homework will
	inherit from a Model object.
	"""
	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		self.p  = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p)); pprint(vars(self.p))
		self.p.batch_size = self.p.batch_size

		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.add_placeholders()

		nn_out, self.accuracy = self.add_model()

		self.loss      	= self.add_loss(nn_out)
		self.logits  	= tf.nn.softmax(nn_out)
		self.train_op   = self.add_optimizer(self.loss)

		tf.summary.scalar('accmain', self.accuracy)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None


	def splitBags(self, data, chunk_size):
		"""
		Split bags which are too big (contains greater than chunk_size sentences)

		Parameters
		----------
		data:	Dataset as list of bags

		Returns
		-------
		Data after preprocessing
		"""

		for dtype in ['train']:

			for i in range(len(data[dtype])-1, -1, -1):
				bag = data[dtype][i]

				if len(bag['X']) > chunk_size:
					del data[dtype][i]
					chunks = getChunks(range(len(bag['X'])), chunk_size)

					for chunk in chunks:
						res = {
							'Y': 		bag['Y'],
							'SubType':	bag['SubType'],
							'ObjType':	bag['ObjType']
						}

						res['X']    	 = [bag['X'][j]    	for j in chunk]
						res['Pos1'] 	 = [bag['Pos1'][j] 	for j in chunk]
						res['Pos2'] 	 = [bag['Pos2'][j] 	for j in chunk]
						res['DepEdges']  = [bag['DepEdges'][j]  for j in chunk]
						res['ProbY']  	 = [bag['ProbY'][j]  	for j in chunk]

						data[dtype].append(res)
		return data


	def getPdata(self, data):
		"""
		Creates data required for P@N metric evaluation

		Parameters
		----------
		data:	Dataset as list of bags

		Returns
		-------
		p_one and p_two are dataset for P@100 and P@200 evaluation. P@All is the original data itself
		"""

		p_one = []
		p_two = []

		for bag in data['test']:
			if len(bag['X']) < 2: continue

			indx = list(range(len(bag['X'])))
			random.shuffle(indx)

			p_one.append({
				'X':    	[bag['X'][indx[0]]],
				'Pos1': 	[bag['Pos1'][indx[0]]],
				'Pos2': 	[bag['Pos2'][indx[0]]],
				'DepEdges': 	[bag['DepEdges'][indx[0]]],
				'ProbY': 	[bag['ProbY'][indx[0]]],
				'Y':    	bag['Y'],
				'SubType':	bag['SubType'],
				'ObjType':	bag['ObjType']
			})

			p_two.append({
				'X':    	[bag['X'][indx[0]], bag['X'][indx[1]]],
				'Pos1': 	[bag['Pos1'][indx[0]], bag['Pos1'][indx[1]]],
				'Pos2': 	[bag['Pos2'][indx[0]], bag['Pos2'][indx[1]]],
				'DepEdges': 	[bag['DepEdges'][indx[0]], bag['DepEdges'][indx[1]]],
				'ProbY': 	[bag['ProbY'][indx[0]], bag['ProbY'][indx[1]]],
				'Y':   	 	bag['Y'],
				'SubType':	bag['SubType'],
				'ObjType':	bag['ObjType']
			})

		return p_one, p_two

	def load_data(self):
		"""
		Reads the data from pickle file

		Parameters
		----------
		self.p.dataset: The path of the dataset to be loaded

		Returns
		-------
		self.voc2id:		Mapping of word to its unique identifier
		self.Id2voc:		Inverse of self.voc2id
		self.type2id:		Mapping of entity type to its unique identifier
		self.type_num:		Total number of entity types 
		self.max_pos:		Maximum positional embedding
		self.num_class:		Total number of relations to be predicted
		self.num_deLabel:	Number of dependency labels
		self.wrd_list:		Words in vocabulary
		self.test_one:		Data required for P@100 evaluation
		self.test_two:		Data required for P@200 evaluation
		self.data:		Datatset as a list of bags, where each bag is a dictionary as described 
		"""
		data = pickle.load(open(self.p.dataset, 'rb'))

		self.voc2id 	   = data['voc2id']
		self.id2voc 	   = data['id2voc']
		self.type2id 	   = data['type2id']
		self.type_num	   = len(data['type2id'])
		self.max_pos 	   = data['max_pos']						# Maximum position distance
		self.num_class     = len(data['rel2id'])
		self.num_deLabel   = 1

		# Get Word List
		self.wrd_list 	   = list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list,_    = zip(*self.wrd_list)

		self.test_one,\
		self.test_two	   = self.getPdata(data)

		self.data 	   = data
		# self.data	   = self.splitBags(data, self.p.chunk_size)			# Activate if bag sizes are too big

		self.logger.info('Document count [{}]: {}, [{}]: {}'.format('train', len(self.data['train']), 'test', len(self.data['test'])))


	def padData(self, data, seq_len):
		"""
		Pads the data in a batch | Used as a helper function by pad_dynamic

		Parameters
		----------
		data:		batch to be padded
		seq_len:	maximum number of words in the batch

		Returns
		-------
		Padded data and mask
		"""
		pad_data = np.zeros((len(data), seq_len), np.int32)
		mask     = np.zeros((len(data), seq_len), np.float32)

		for i, ele in enumerate(data):
			pad_data[i, :len(ele)] = ele[:seq_len]
			mask    [i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

		return pad_data, mask

	def getOneHot(self, data, num_class, isprob=False):
		"""
		Generates the one-hot representation

		Parameters
		----------
		data:		Batch to be padded
		num_class:	Total number of relations 

		Returns
		-------
		One-hot representation of batch
		"""
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			for rel in ele:
				if isprob:	temp[i, rel-1] = 1
				else:		temp[i, rel]   = 1
		return temp

	def add_placeholders(self):
		"""
		Adds placeholder variables to tensorflow computational graph.
		Tensorflow uses placeholder variables to represent locations in a
		computational graph where data is inserted.  These placeholders are used as
		inputs by the rest of the model building code and will be fed data during
		training.
		See for more information:
		https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def create_feed_dict(self, input_batch, label_batch):
		"""
		Creates the feed_dict for training the given step.
		A feed_dict takes the form of:
		feed_dict = {
				<placeholder>: <tensor of values to be passed for placeholder>,
				....
		}
	
		If label_batch is None, then no labels are added to feed_dict.
		Hint: The keys for the feed_dict should be a subset of the placeholder
					tensors created in add_placeholders.
		
		Args:
			input_batch: A batch of input data.
			label_batch: A batch of label data.
		Returns:
			feed_dict: The feed dictionary mapping from placeholders to values.
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_model(self, input_data):
		"""
		Implements core of model that transforms input_data into predictions.
		The core transformation for this model which transforms a batch of input
		data into a batch of predictions.
		Args:
			input_data: A tensor of shape (batch_size, n_features).
		Returns:
			out: A tensor of shape (batch_size, n_classes)
		"""
		raise NotImplementedError("Each Model must re-implement this method.")

	def add_loss(self, nn_out):
		"""
		Computes loss based on logits and actual labels

		Parameters
		----------
		nn_out:		Logits for each bag in the batch

		Returns
		-------
		loss:		Computes loss based on prediction and actual labels of the bags
		"""

		with tf.name_scope('Loss_op'):
			loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return loss

	def add_optimizer(self, loss):
		"""
		Add optimizer for training variables

		Parameters
		----------
		loss:		Computed loss

		Returns
		-------
		train_op:	Training optimizer
		"""
		with tf.name_scope('Optimizer'):
			if self.p.opt == 'adam' and not self.p.restore:
				optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:		
				optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		return train_op


	def predict(self, sess, data, wLabels=True, shuffle=False, label='Evaluating on Test'):
		"""
		Evaluate model on valid/test data

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to evaluate on
		wLabels:	Does data include labels or not
		shuffle:	Shuffle data while before creates batches
		label:		Log label to be used while logging

		Returns
		-------
		losses:		Loss over the entire data
		accuracies:	Overall Accuracy
		y: 		Actual label
		y_pred:		Predicted labels
		logit_list:	Logit list for each bag in the data
		y_actual_hot:	One hot represetnation of actual label for each bag in the data

		"""
		losses, accuracies, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):

			loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, split='test'))
			losses.    append(loss)
			accuracies.append(accuracy)

			pred_ind      = logits.argmax(axis=1)
			logit_list   += logits.tolist()
			y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
			y_pred       += pred_ind.tolist()
			y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
			bag_cnt      += len(batch['sent_num'])

			if step % 100 == 0:
				self.logger.info('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['test']), np.mean(accuracies)*100, np.mean(losses), self.p.name))

		self.logger.info('Test Accuracy: {}'.format(accuracy))

		return np.mean(losses), np.mean(accuracies)*100, y, y_pred, logit_list, y_actual_hot


	def run_epoch(self, sess, data, epoch, shuffle=True):
		"""
		Runs one epoch of training

		Parameters
		----------
		sess:		Session of tensorflow
		data:		Data to train on
		epoch:		Epoch number
		shuffle:	Shuffle data while before creates batches

		Returns
		-------
		losses:		Loss over the entire data
		Accuracy:	Overall accuracy
		"""
		losses, accuracies = [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			summary_str, loss, accuracy, _ = sess.run([self.merged_summ, self.loss, self.accuracy, self.train_op], feed_dict=feed)

			losses.    append(loss)
			accuracies.append(accuracy)

			bag_cnt += len(batch['sent_num'])

			if step % 10 == 0:
				self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, len(self.data['train']), np.mean(accuracies)*100, np.mean(losses), self.p.name, self.best_train_acc))
				self.summ_writer.add_summary(summary_str, epoch*len(self.data['train']) + bag_cnt)

		accuracy = np.mean(accuracies) * 100.0
		self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
		return np.mean(losses), accuracy

	def calc_prec_recall_f1(self, y_actual, y_pred, none_id):
		"""
		Calculates precision recall and F1 score

		Parameters
		----------
		y_actual:	Actual labels
		y_pred:		Predicted labels
		none_id:	Identifier used for denoting NA relation

		Returns
		-------
		precision:	Overall precision
		recall:		Overall recall
		f1:		Overall f1
		"""
		pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0

		for i in range(len(y_actual)):
			if y_actual[i] != none_id:
				pos_gt += 1.0

		for i in range(len(y_pred)):
			if y_pred[i] != none_id:
				pos_pred += 1.0					# classified as pos example (Is-A-Relation)
				if y_pred[i] == y_actual[i]:
					true_pos += 1.0

		precision 	= true_pos / (pos_pred + self.p.eps)
		recall 		= true_pos / (pos_gt + self.p.eps)
		f1 		= 2 * precision * recall / (precision + recall + self.p.eps)

		return precision, recall, f1

	def getPscore(self, sess, data, label='P@N Evaluation'):
		"""
		Computes P@N for N = 100, 200, and 300

		Parameters
		----------
		data:		Data for P@N evaluation
		label:		Log label to be used while logging

		Returns
		-------
		P@100		Precision @ 100
		P@200		Precision @ 200
		P@300		Precision @ 300
		"""
		test_loss, test_acc, y, y_pred, logit_list, y_hot = self.predict(sess, data, label)

		y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
		y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))

		allprob = np.reshape(np.array(y_scores), (-1))
		allans  = np.reshape(y_true, (-1))
		order   = np.argsort(-allprob)

		def p_score(n):
			corr_num = 0.0
			for i in order[:n]:
				corr_num += 1.0 if (allans[i] == 1) else 0
			return corr_num / n

		return p_score(100), p_score(200), p_score(300)

	def fit(self, sess):
		"""
		Trains the model and finally evaluates on test

		Parameters
		----------
		sess:		Tensorflow session object

		Returns
		-------
		"""
		self.summ_writer = tf.summary.FileWriter('tf_board/{}'.format(self.p.name), sess.graph)
		saver     = tf.train.Saver()
		save_dir  = 'checkpoints/{}/'.format(self.p.name); make_dir(save_dir)
		res_dir   = 'results/{}/'.format(self.p.name);     make_dir(res_dir)
		save_path = os.path.join(save_dir, 'best_model')
		
		# Restore previously trained model
		if self.p.restore: 
			saver.restore(sess, save_path)

		''' Train model '''
		if not self.p.only_eval:
			self.best_train_acc = 0.0
			for epoch in range(self.p.max_epochs):
				train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
				self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}\n'.format(epoch, train_loss, train_acc))

				# Store the model with least train loss
				if train_acc > self.best_train_acc:
					self.best_train_acc = train_acc
					saver.save(sess=sess, save_path=save_path)
		
		''' Evaluation on Test '''
		saver.restore(sess, save_path)
		test_loss, test_acc, y, y_pred, logit_list, y_hot = self.predict(sess, self.data['test'])
		test_prec, test_rec, test_f1 			  = self.calc_prec_recall_f1(y, y_pred, 0)	# 0: ID for 'NA' relation

		y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
		y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
		area_pr  = average_precision_score(y_true, y_scores)

		self.logger.info('Final results: Prec:{} | Rec:{} | F1:{} | Area:{}'.format(test_prec, test_rec, test_f1, area_pr))
		# Store predictions
		pickle.dump({'logit_list': logit_list, 'y_hot': y_hot}, open("results/{}/precision_recall.pkl".format(self.p.name), 'wb'))

		''' P@N Evaluation '''

		# P@1
		one_100, one_200, one_300 = self.getPscore(sess, self.test_one, label='P@1 Evaluation')
		self.logger.info('TEST_ONE: P@100: {}, P@200: {}, P@300: {}'.format(one_100, one_200, one_300))
		one_avg = (one_100 + one_200 + one_300)/3

		# P@2
		two_100, two_200, two_300 = self.getPscore(sess, self.test_two, label='P@2 Evaluation')
		self.logger.info('TEST_TWO: P@100: {}, P@200: {}, P@300: {}'.format(two_100, two_200, two_300))
		two_avg = (two_100 + two_200 + two_300)/3

		# P@All
		all_100, all_200, all_300 = self.getPscore(sess, self.data['test'], label='P@All Evaluation')
		self.logger.info('TEST_THREE: P@100: {}, P@200: {}, P@300: {}'.format(all_100, all_200, all_300))
		all_avg = (all_100 + all_200 + all_300)/3

		pprint ({
				'one_100':  one_100,
				'one_200':  one_200,
				'one_300':  one_300,
				'mean_one': one_avg,
				'two_100':  two_100,
				'two_200':  two_200,
				'two_300':  two_300,
				'mean_two': two_avg,
				'all_100':  all_100,
				'all_200':  all_200,
				'all_300':  all_300,
				'mean_all': all_avg,
		})
