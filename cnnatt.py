from base import Model
from helper import *
import tensorflow as tf

"""
Abbreviations used in variable names:
	Type:  Entity type side informatoin
	ProbY, RelAlias: Relation alias side information
Recommendation: View this file with tab size 8.
"""

class PCNNATT(Model):

	def getBatches(self, data, shuffle = True):
		"""
	        Generates batches of multiple bags

	        Parameters
	        ----------
	        data:		Data to be used for creating batches. Dataset as list of bags where each bag is a dictionary
	        shuffle:	Decides whether to shuffle the data or not.
	        
	        Returns
	        -------
	        Generator for creating batches. 
	        """
		if shuffle: random.shuffle(data)

		def get_sent_part(sub_pos, obj_pos, sents):
			assert len(sub_pos) == len(obj_pos)

			part_pos = []
			for i in range(len(sub_pos)):
				sent = sents[i]
				pos1, pos2 = sub_pos[i], obj_pos[i]
				pos1, pos2 = min(pos1, pos2), max(pos1, pos2)
				if pos1 == pos2 or pos1 == 0 or pos2 == len(sent)-1:
					pos1 = len(sent) // 4 
					pos2 = pos1 + len(sent) // 2

				part_pos.append([pos1, pos2])

			return part_pos


		for chunk in getChunks(data, self.p.batch_size):			# chunk = batch
			batch = ddict(list)

			num = 0
			for i, bag in enumerate(chunk):

				batch['X']		+= bag['X']
				batch['Pos1']		+= bag['Pos1']
				batch['Pos2']		+= bag['Pos2']
				batch['PartPos']	+= get_sent_part(bag['SubPos'], bag['ObjPos'], bag['X'])

				batch['Y'].append(bag['Y'])
				old_num  = num
				num 	+= len(bag['X'])

				batch['sent_num'].append([old_num, num, i])

			batch = dict(batch)

			yield batch


	def add_placeholders(self):
		"""
		Defines the placeholder required for the model
		"""

		self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')			# Tokens ids of sentences
		self.input_y 		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_labels')			# Actual relation of the bag
		self.input_pos1 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos1')			# Position ids wrt entity 1
		self.input_pos2 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos2')			# Position ids wrt entity 2
		self.part_pos 		= tf.placeholder(tf.int32,   shape=[None, 2],      name='input_part_pos') 		# Positions where sentence needs to be partitioned

		self.x_len		= tf.placeholder(tf.int32,   shape=[None],         name='input_len')			# Number of words in sentences in a batch
		self.sent_num 		= tf.placeholder(tf.int32,   shape=[None, 3], 	   name='sent_num')			# Stores which sentences belong to which bag | [start_index, end_index, bag_number]
		self.seq_len 		= tf.placeholder(tf.int32,   shape=(), 		   name='seq_len')			# Max number of tokens in sentences in a batch
		self.total_bags 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_bags')			# Total number of bags in a batch
		self.total_sents 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_sents')			# Total number of sentences in a batch

		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')		# Dropout used in GCN Layer


	def pad_dynamic(self, X, pos1, pos2):
		"""
		Pads each batch during runtime.

		Parameters
		----------
		X:		For each sentence in the bag, list of words
		pos1:		For each sentence in the bag, list position of words with respect to subject
		pos2:		For each sentence in the bag, list position of words with respect to object

		Returns
		-------
		x_pad		Padded words 
		x_len		Number of sentences in each sentence, 
		pos1_pad	Padded position 1
		pos2_pad	Padded position 2
		seq_len 	Maximum sentence length in the batch
		"""

		seq_len, max_et = 0, 0
		x_len = np.zeros((len(X)), np.int32)

		for i, x in enumerate(X):
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		x_pad,  _ 	= self.padData(X, seq_len)
		pos1_pad,  _ 	= self.padData(pos1, seq_len)
		pos2_pad,  _ 	= self.padData(pos2, seq_len)

		return x_pad, x_len, pos1_pad, pos2_pad, seq_len


	def create_feed_dict(self, batch, wLabels=True, split='train'):
		X, Y, pos1, pos2, sent_num, part_pos = batch['X'], batch['Y'], batch['Pos1'], batch['Pos2'], batch['sent_num'], batch['PartPos']

		total_sents = len(batch['X'])
		total_bags  = len(batch['Y'])
		x_pad, x_len, pos1_pad, pos2_pad, seq_len = self.pad_dynamic(X, pos1, pos2)

		y_hot = self.getOneHot(Y, self.num_class)

		feed_dict = {}
		feed_dict[self.input_x] 		= np.array(x_pad)
		feed_dict[self.input_pos1]		= np.array(pos1_pad)
		feed_dict[self.input_pos2]		= np.array(pos2_pad)
		feed_dict[self.x_len] 			= np.array(x_len)
		feed_dict[self.seq_len]			= seq_len
		feed_dict[self.total_sents]		= total_sents
		feed_dict[self.total_bags]		= total_bags
		feed_dict[self.sent_num]		= sent_num
		feed_dict[self.part_pos] 		= np.array(part_pos)

		if wLabels: feed_dict[self.input_y] 	= y_hot

		if split != 'train': 	feed_dict[self.dropout]     = 1.0
		else: 			feed_dict[self.dropout]     = self.p.dropout

		return feed_dict

	def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
		"""
		Stores the adjacency matrix as indices and data for feeding to TensorFlow

		Parameters
		----------
		edgeList:	List of list of edges 
		batch_size:	Number of bags in a batch
		max_nodes:	Maximum number of nodes in the graph
		max_labels:	Maximum number of edge labels in the graph 

		Returns
		-------
		adj_mat_ind 	indices of adjacency matrix
		adj_mat_data	data of adjacency matrix
		"""
		max_edges = 0
		for edges in edgeList:
			max_edges = max(max_edges, len(edges))

		adj_mat_ind  = np.zeros((max_labels, batch_size, max_edges, 2), np.int64)
		adj_mat_data = np.zeros((max_labels, batch_size, max_edges), 	np.float32)

		for lbl in range(max_labels):
			for i, edges in enumerate(edgeList):
				in_ind_temp,  in_data_temp  = [], []
				for j, (src, dest, _, _) in enumerate(edges):
					adj_mat_ind [lbl, i, j] = (src, dest)
					adj_mat_data[lbl, i, j] = 1.0

		return adj_mat_ind, adj_mat_data

	def add_model(self):
		"""
		Creates the Computational Graph

		Parameters
		----------

		Returns
		-------
		nn_out:		Logits for each bag in the batch
		accuracy:	accuracy for the entire batch
		"""

		in_wrds, in_pos1, in_pos2 = self.input_x, self.input_pos1, self.input_pos2

		with tf.variable_scope('Embeddings') as scope:
			embed_init 	= getEmbeddings(self.wrd_list, self.p.embed_dim, self.p.embed_loc)
			_wrd_embeddings = tf.get_variable('embeddings',      initializer=embed_init, trainable=True, 		regularizer=self.regularizer)  # Word Embeddings
			wrd_pad  	= tf.zeros([1, self.p.embed_dim])
			wrd_embeddings  = tf.concat([wrd_pad, _wrd_embeddings], axis=0)

			pos1_embeddings = tf.get_variable('pos1_embeddings', [self.max_pos, self.p.pos_dim], trainable=True,  	regularizer=self.regularizer)
			pos2_embeddings = tf.get_variable('pos2_embeddings', [self.max_pos, self.p.pos_dim], trainable=True, 	regularizer=self.regularizer)

		wrd_embed  = tf.nn.embedding_lookup(wrd_embeddings,  self.input_x)
		pos1_embed = tf.nn.embedding_lookup(pos1_embeddings, self.input_pos1)
		pos2_embed = tf.nn.embedding_lookup(pos2_embeddings, self.input_pos2)
		conv_in    = tf.expand_dims(tf.concat([wrd_embed, pos1_embed, pos2_embed], axis=2), axis=3)

		conv_in_dim = self.p.embed_dim + 2 * self.p.pos_dim

		with tf.variable_scope('Convolution') as scope:
			padding	= tf.constant([[0,0], [1,1], [0,0], [0,0]])
			kernel	= tf.get_variable('kernel', [self.p.filt_size, conv_in_dim, 1, self.p.num_filt], initializer=tf.truncated_normal_initializer(), regularizer=self.regularizer)
			biases	= tf.get_variable('biases', [self.p.num_filt],					 initializer=tf.random_normal_initializer(), 	regularizer=self.regularizer)

			conv_in = tf.pad(conv_in, padding, 'CONSTANT')
			conv  	= tf.nn.conv2d(conv_in, kernel, strides=[1, 1, 1, 1], padding='VALID')						
			convRes = tf.nn.relu(conv + biases, name=scope.name)
			convRes = tf.squeeze(convRes, 2)

		sent_rep	= tf.reduce_max(convRes, axis = 1)
		sent_rep_dim	= self.p.num_filt
		
		with tf.variable_scope('Sentence_attention') as scope:
			sent_atten_q = tf.get_variable('sent_atten_q', [sent_rep_dim, 1] )

			def getSentAtten(num):
				bag_sents   	= sent_rep[num[0]: num[1]]
				num_sents  	= num[1] - num[0]
				if self.p.inc_attn:
					sent_atten_wts  = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(bag_sents), sent_atten_q), [-1]))
					bag_rep_ 	= tf.reshape(tf.matmul(tf.expand_dims(sent_atten_wts, 0), bag_sents), [sent_rep_dim])
				else:
					bag_rep_ 	= tf.reduce_mean(bag_sents, axis=0)

				return bag_rep_

			bag_rep = tf.map_fn(getSentAtten, self.sent_num, dtype=tf.float32)

		with tf.variable_scope('FC1') as scope:
			w_rel   = tf.get_variable('w_rel', [sent_rep_dim, self.num_class], 	initializer=tf.truncated_normal_initializer(),	regularizer=self.regularizer)
			b_rel   = tf.get_variable('b_rel', [self.num_class], 			initializer=tf.constant_initializer(0.0), 	regularizer=self.regularizer)
			nn_out  = tf.nn.xw_plus_b(bag_rep, w_rel, b_rel)


		with tf.name_scope('Accuracy') as scope:
			prob     = tf.nn.softmax(nn_out)
			y_pred   = tf.argmax(prob, 	   axis=1)
			y_actual = tf.argmax(self.input_y, axis=1)
			accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))

		''' Debugging command:
			Put the below command anywhere and get the values of the tensors (Use TF like PyTorch!)
			res  = debug_nn([de_out], self.create_feed_dict( next(self.getBatches(self.data['train'])) ) ); pdb.set_trace()
		'''

		return nn_out, accuracy

	def __init__(self, params):
		"""
		Constructor for the main function. Loads data and creates computation graph. 

		Parameters
		----------
		params:		Hyperparameters of the model

		Returns
		-------
		"""
		super(PCNNATT, self).__init__(params)



if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Improving Distantly-Supervised Neural Relation Extraction using Side Information')

	parser.add_argument('-data', 	 dest="dataset", 	required=True,							help='Dataset to use')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',							help='GPU to use')
	parser.add_argument('-nGate', 	 dest="wGate", 		action='store_false',   					help='Include edgewise-gating in GCN')

	parser.add_argument('-pos_dim',  dest="pos_dim", 	default=16, 			type=int, 			help='Dimension of positional embeddings')
	parser.add_argument('-filt_size',dest="filt_size", 	default=3, 			type=int, 			help='Size of filters used in Convolution Layer')
	parser.add_argument('-num_filt', dest="num_filt", 	default=100, 			type=int, 			help='Number of filters used in Convolution Layer')
	parser.add_argument('-drop',	 dest="dropout", 	default=0.8,  			type=float,			help='Dropout for full connected layer')
	parser.add_argument('-attn',	 dest="inc_attn", 	action='store_true', 						help='Include attention during instance aggregation')

	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,			help='Learning rate')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 			help='L2 regularization')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=2,   			type=int, 			help='Max epochs')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=32,   			type=int, 			help='Batch size')
	parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 			help='Chunk size')
	parser.add_argument('-only_eval',dest="only_eval", 	action='store_true', 						help='Only Evaluate the pretrained model (skip training)')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 						help='Restore from the previous best saved model')
	parser.add_argument('-opt',	 dest="opt", 		default='adam', 						help='Optimizer to use for training')

	parser.add_argument('-eps', 	 dest="eps", 		default=0.00000001,  		type=float, 			help='Value of epsilon')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),				help='Name of the run')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,			help='Seed for randomization')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 						help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 						help='Config directory')
	parser.add_argument('-embed_loc',dest="embed_loc", 	default='./glove/glove.6B.50d.txt', 				help='Log directory')
	parser.add_argument('-embed_dim',dest="embed_dim", 	default=50, type=int,						help='Dimension of embedding')
	args = parser.parse_args()

	if not args.restore: args.name = args.name

	# Set GPU to use
	set_gpu(args.gpu)

	# Set seed 
	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	# Create model computational graph
	model  = PCNNATT(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)