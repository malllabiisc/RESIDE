from models import *
from helper import *
import tensorflow as tf

"""
Abbreviations used in variable names:
	Type:  Entity type side informatoin
	ProbY: Relation alias side information
"""

class RE_NN(Model):

	# Pads the data in a batch
	def padData(self, data, seq_len):
		temp = np.zeros((len(data), seq_len), np.int32)
		mask = np.zeros((len(data), seq_len), np.float32)

		for i, ele in enumerate(data):
			temp[i, :len(ele)] = ele[:seq_len]
			mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

		return temp, mask

	# Generates the one-hot representation
	def getOneHot(self, data, num_class, isprob=False):
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			for rel in ele:
				if isprob:	temp[i, rel-1] = 1
				else:		temp[i, rel]   = 1
		return temp

	# Generates batches of multiple bags
	def getBatches(self, data, shuffle = True):
		if shuffle: random.shuffle(data)

		for chunk in getChunks(data, self.p.batch_size):			# chunk = batch
			batch = ddict(list)

			num = 0
			for i, bag in enumerate(chunk):

				batch['X']    	   += bag['X']
				batch['Pos1'] 	   += bag['Pos1']
				batch['Pos2'] 	   += bag['Pos2']
				batch['DepEdges']  += bag['DepEdges']
				batch['ProbY']	   += bag['ProbY']

				batch['SubType'].append(bag['SubType'])
				batch['ObjType'].append(bag['ObjType'])

				batch['Y'].append(bag['Y'])
				old_num  = num
				num 	+= len(bag['X'])

				batch['sent_num'].append([old_num, num, i])

			yield batch

	# Split bags which are too big (contains greater than chunk_size sentences)
	def splitBags(self, data, chunk_size):
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
		data = pickle.load(open(self.p.dataset, 'rb'))

		self.voc2id 	   = data['voc2id']
		self.id2voc 	   = data['id2voc']
		self.type2id 	   = data['type2id']
		self.type_num	   = len(data['type2id'])
		self.max_pos 	   = data['max_pos']

		# self.rel2id        = json.loads(open(self.p.rel2id_map).read())
		self.num_class     = self.p.num_class
		self.num_deLabel   = 1

		# Get Word List
		self.wrd_list 	   = list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list,_    = zip(*self.wrd_list)

		self.test_one,\
		self.test_two	   = self.getPdata(data)

		self.data 	   = data
		# self.data	   = self.splitBags(data, self.p.chunk_size)

		self.logger.info('Document count [{}]: {}, [{}]: {}'.format('train', len(self.data['train']), 'test', len(self.data['test'])))


	def add_placeholders(self):
		self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')			# Tokens ids of sentences
		self.input_y 		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_labels')			# Actual relation of the bag
		self.input_pos1 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos1')			# Position ids wrt entity 1
		self.input_pos2 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos2')			# Position ids wrt entity 2

		# Entity Type Side Information
		self.input_subtype 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_subtype')		# Entity type information of entity 1
		self.input_objtype 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_objtype')		# Entity type information of entity 2
		self.input_subtype_len 	= tf.placeholder(tf.float32, shape=[None],   	   name='input_subtype_len')		# Max number of types of entity 1
		self.input_objtype_len 	= tf.placeholder(tf.float32, shape=[None],   	   name='input_objtype_len')		# Max number of types of entity 2 

		# Relation Alias Side Information
		self.input_proby 	= tf.placeholder(tf.float32, shape=[None, None],   name='input_prob_y')			# Probable relation match
		self.input_proby_ind 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_prob_ind')
		self.input_proby_len 	= tf.placeholder(tf.float32, shape=[None],   	   name='input_prob_len')		# Max number of relation matched

		self.x_len		= tf.placeholder(tf.int32,   shape=[None],         name='input_len')			# Number of words in sentences in a batch
		self.sent_num 		= tf.placeholder(tf.int32,   shape=[None, 3], 	   name='sent_num')			# Stores which sentences belong to which bag
		self.seq_len 		= tf.placeholder(tf.int32,   shape=(), 		   name='seq_len')			# Max number of tokens in sentences in a batch
		self.total_bags 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_bags')			# Total number of bags in a batch
		self.total_sents 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_sents')			# Total number of sentences in a batch

		self.de_adj_ind 	= tf.placeholder(tf.int64,   shape=[self.num_deLabel, None, None, 2], name='de_adj_ind')# Dependency graph information (Storing only indices and data)
		self.de_adj_data 	= tf.placeholder(tf.float32, shape=[self.num_deLabel, None, None], name='de_adj_data')

		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')		# Dropout used in GCN Layer
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')		# Dropout used in Bi-LSTM


	def pad_dynamic(self, X, pos1, pos2, sub_type, obj_type, prob_y):
		seq_len, max_et, max_type, max_proby = 0, 0, 0, 0
		subtype_len, objtype_len, proby_len = [], [], []

		x_len = np.zeros((len(X)), np.int32)

		for py in prob_y:
			proby_len.append(len(py))
			max_proby = max(max_proby, len(py))

		for typ in sub_type:
			subtype_len.append(len(typ))
			max_type = max(max_type, len(typ))

		for typ in obj_type:
			objtype_len.append(len(typ))
			max_type = max(max_type, len(typ))

		for i, x in enumerate(X):
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		x_pad,  _ 	= self.padData(X, seq_len)
		pos1_pad,  _ 	= self.padData(pos1, seq_len)
		pos2_pad,  _ 	= self.padData(pos2, seq_len)
		subtype, _ 	= self.padData(sub_type, max_type)
		objtype, _ 	= self.padData(obj_type, max_type)
		proby_ind, _ 	= self.padData(prob_y, max_proby)

		return x_pad, x_len, pos1_pad, pos2_pad, seq_len, subtype, subtype_len, objtype, objtype_len, proby_ind, proby_len


	def create_feed_dict(self, batch, wLabels=True, dtype='train'):									# Where putting dropout for train?
		X, Y, pos1, pos2, sent_num, sub_type, obj_type, prob_y = batch['X'], batch['Y'], batch['Pos1'], batch['Pos2'], batch['sent_num'], batch['SubType'], batch['ObjType'], batch['ProbY']
		total_sents = len(batch['X'])
		total_bags  = len(batch['Y'])
		x_pad, x_len, pos1_pad, pos2_pad, seq_len, subtype, subtype_len, objtype, objtype_len, proby_ind, proby_len = self.pad_dynamic(X, pos1, pos2, sub_type, obj_type, prob_y)

		y_hot = self.getOneHot(Y, 	self.num_class)
		proby = self.getOneHot(prob_y, 	self.num_class-1, isprob=True)	# -1 because NA cannot be in proby

		feed_dict = {}
		feed_dict[self.input_x] 		= np.array(x_pad)
		feed_dict[self.input_pos1]		= np.array(pos1_pad)
		feed_dict[self.input_pos2]		= np.array(pos2_pad)
		feed_dict[self.input_subtype]		= np.array(subtype)
		feed_dict[self.input_objtype]		= np.array(objtype)
		feed_dict[self.input_proby]		= np.array(proby)
		feed_dict[self.x_len] 			= np.array(x_len)
		feed_dict[self.seq_len]			= seq_len
		feed_dict[self.total_sents]		= total_sents
		feed_dict[self.total_bags]		= total_bags
		feed_dict[self.sent_num]		= sent_num
		feed_dict[self.input_subtype_len] 	= np.array(subtype_len) + 0.00000001
		feed_dict[self.input_objtype_len] 	= np.array(objtype_len) + 0.00000001
		feed_dict[self.input_proby_ind] 	= np.array(proby_ind)
		feed_dict[self.input_proby_len] 	= np.array(proby_len) + 0.00000001

		if wLabels: feed_dict[self.input_y] 	= y_hot

		feed_dict[self.de_adj_ind], \
		feed_dict[self.de_adj_data] 		= self.get_adj(batch['DepEdges'], total_sents, seq_len, self.num_deLabel)

		if dtype != 'train':
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0
		else:
			feed_dict[self.dropout]     = self.p.dropout
			feed_dict[self.rec_dropout] = self.p.rec_dropout

		return feed_dict

	# Stores the adjacency matrix as indices and data for feeding to TensorFlow
	def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
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

	# GCN Layer Implementation 
	def GCNLayer(self, 	gcn_in, 	# Input to GCN Layer
				in_dim, 	# Dimension of input to GCN Layer 
				gcn_dim, 	# Hidden state dimension of GCN
				batch_size, 	# Batch size
				max_nodes, 	# Maximum number of nodes in graph
				max_labels, 	# Maximum number of edge labels
				adj_ind, 	# Adjacency matrix indices
				adj_data, 	# Adjacency matrix data (all 1's)
				w_gating=True,  # Whether to include gating in GCN
				num_layers=1, 	# Number of GCN Layers
				name="GCN"):
		out = []
		out.append(gcn_in)

		for layer in range(num_layers):
			gcn_in    = out[-1]				# out contains the output of all the GCN layers, intitally contains input to first GCN Layer
			if len(out) > 1: in_dim = gcn_dim 		# After first iteration the in_dim = gcn_dim

			with tf.name_scope('%s-%d' % (name,layer)):
				act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
				for lbl in range(max_labels):

					with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer), reuse=tf.AUTO_REUSE) as scope:
						w_in   = tf.get_variable('w_in')
						b_in   = tf.get_variable('b_in')
						w_out  = tf.get_variable('w_out')
						b_out  = tf.get_variable('b_out')
						w_loop = tf.get_variable('w_loop')

						if w_gating:
							w_gin  = tf.get_variable('w_gin')
							b_gin  = tf.get_variable('b_gin')
							w_gout = tf.get_variable('w_gout')
							b_gout = tf.get_variable('b_gout')
							w_gloop = tf.get_variable('w_gloop')


					with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_in  = tf.tensordot(gcn_in, w_in, axes=[2,0]) + tf.expand_dims(b_in, axis=0)

						def map_func1(i):
							adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
							adj_mat = tf.sparse_transpose(adj_mat)
							return tf.sparse_tensor_dense_matmul(adj_mat, inp_in[i])
						in_t = tf.map_fn(map_func1, tf.range(batch_size), dtype=tf.float32)

						if self.p.dropout != 1.0: in_t = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

						if w_gating:
							inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2,0]) + tf.expand_dims(b_gin, axis=0)

							def map_func2(i):
								adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
								adj_mat = tf.sparse_transpose(adj_mat)
								return tf.sparse_tensor_dense_matmul(adj_mat, inp_gin[i])
							in_gate = tf.map_fn(map_func2, tf.range(batch_size), dtype=tf.float32)
							in_gsig = tf.sigmoid(in_gate)
							in_act  = in_t * in_gsig
						else:
							in_act   = in_t

					with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
						inp_out  = tf.tensordot(gcn_in, w_out, axes=[2,0]) + tf.expand_dims(b_out, axis=0)

						def map_func3(i):
							adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
							return tf.sparse_tensor_dense_matmul(adj_mat, inp_out[i])
						out_t = tf.map_fn(map_func3, tf.range(batch_size), dtype=tf.float32)
						if self.p.dropout != 1.0: out_t    = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

						if w_gating:
							inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2,0]) + tf.expand_dims(b_gout, axis=0)
							def map_func4(i):
								adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i], [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
								return tf.sparse_tensor_dense_matmul(adj_mat, inp_gout[i])
								
							out_gate = tf.map_fn(map_func4, tf.range(batch_size), dtype=tf.float32)
							out_gsig = tf.sigmoid(out_gate)
							out_act  = out_t * out_gsig
						else:
							out_act = out_t

					with tf.name_scope('self_loop'):
						inp_loop  = tf.tensordot(gcn_in, w_loop,  axes=[2,0])
						if self.p.dropout != 1.0: inp_loop  = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

						if w_gating:
							inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2,0])
							loop_gsig = tf.sigmoid(inp_gloop)
							loop_act  = inp_loop * loop_gsig
						else:
							loop_act = inp_loop

					act_sum += in_act + out_act + loop_act

				gcn_out = tf.nn.relu(act_sum)
				out.append(gcn_out)

		return out

	# For handling TensorFlow issue: #
	def initialize_variables(self):
		def xavier(shape, isRelu = False):
			if isRelu: return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / (shape[0]))
			else:      return np.random.randn(*shape).astype(np.float32) * np.sqrt(1.0 / (shape[0]))

		inits = {}

		with tf.variable_scope('Embeddings') as scope:
			embed_init 	 = getEmbeddings(self.p.embed_loc, self.wrd_list, self.p.embed_dim)
			wrd_embeddings   = tf.get_variable('embeddings',      initializer=embed_init, trainable=True, regularizer=self.regularizer)
			pos1_embeddings  = tf.get_variable('pos1_embeddings', initializer=xavier([self.max_pos, self.p.pos_dim]), trainable=True,   regularizer=self.regularizer)
			pos2_embeddings  = tf.get_variable('pos2_embeddings', initializer=xavier([self.max_pos, self.p.pos_dim]), trainable=True,   regularizer=self.regularizer)

		if self.p.ProbY:
			with tf.variable_scope('AliasInfo') as scope:
				_alias_embeddings = tf.get_variable('alias_embeddings', initializer=xavier([self.num_class-1, self.p.alias_dim]), trainable=True, regularizer=self.regularizer)

		with tf.variable_scope('Bi-LSTM') as scope:
			fw_cell      = tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, name='FW_GRU')
			bk_cell      = tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, name='BW_GRU')

		# GCN Parmaeters
		name 	= 'GCN_DE'
		if self.p.lstm == 'add': in_dim = self.p.lstm_dim
		else: 			 in_dim = self.p.lstm_dim*2
		gcn_dim = self.p.de_gcn_dim

		if self.p.de_gcn:
			gcn_in_dim = in_dim

			for layer in range(self.p.de_layers):
				with tf.name_scope('%s-%d' % (name, layer)):
					if layer > 0: gcn_in_dim = gcn_dim

					for lbl in range(self.num_deLabel):

						with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
							w_in   = tf.get_variable('w_in',  	initializer=xavier([gcn_in_dim, gcn_dim], isRelu = True), 	regularizer=self.regularizer)
							b_in   = tf.get_variable('b_in',   	initializer=np.zeros([1, gcn_dim]).astype(np.float32),	regularizer=self.regularizer)
							w_out  = tf.get_variable('w_out', 	initializer=xavier([gcn_in_dim, gcn_dim], isRelu = True), 	regularizer=self.regularizer)
							b_out  = tf.get_variable('b_out',  	initializer=np.zeros([1, gcn_dim]).astype(np.float32),	regularizer=self.regularizer)
							w_loop = tf.get_variable('w_loop', 	initializer=xavier([gcn_in_dim, gcn_dim], isRelu = True), 	regularizer=self.regularizer)

							if self.p.wGate:
								w_gin  = tf.get_variable('w_gin',   initializer=xavier([gcn_in_dim, 1], isRelu = True),	regularizer=self.regularizer)
								b_gin  = tf.get_variable('b_gin',   initializer=np.zeros([1]).astype(np.float32),	regularizer=self.regularizer)
								w_gout = tf.get_variable('w_gout',  initializer=xavier([gcn_in_dim, 1], isRelu = True),	regularizer=self.regularizer)
								b_gout = tf.get_variable('b_gout',  initializer=np.zeros([1]).astype(np.float32),	regularizer=self.regularizer)
								w_gloop = tf.get_variable('w_gloop',initializer=xavier([gcn_in_dim, 1], isRelu = True),	regularizer=self.regularizer)



		if   self.p.de_gcn:  de_out_dim = self.p.de_gcn_dim + in_dim
		else: 		     de_out_dim = in_dim

		with tf.variable_scope('Word_attention') as scope:
			wrd_query    = tf.get_variable('wrd_query', initializer=xavier([de_out_dim, 1]))

		with tf.variable_scope('TypeInfo') as scope:
			self.type_dim    = self.p.type_dim
			_type_embeddings = tf.get_variable('type_embeddings', initializer=xavier([self.type_num, self.p.type_dim]), trainable=True, regularizer=self.regularizer)

		with tf.variable_scope('Sentence_attention') as scope:
			if self.p.ProbY: de_out_dim += self.p.alias_dim
			sent_atten_q = tf.get_variable('sent_atten_q', initializer=xavier([de_out_dim, 1]))

		with tf.variable_scope('FC1') as scope:
			if self.p.Type:  de_out_dim += 2*self.p.type_dim
			w_rel   = tf.get_variable('w_rel', initializer=xavier([de_out_dim, self.num_class]), 			regularizer=self.regularizer)
			b_rel   = tf.get_variable('b_rel', initializer=np.zeros([self.num_class]).astype(np.float32), 		regularizer=self.regularizer)


	def add_model(self):
		in_wrds, in_pos1, in_pos2 = self.input_x, self.input_pos1, self.input_pos2

		with tf.variable_scope('Embeddings', reuse=tf.AUTO_REUSE) as scope:
			wrd_pad  	= tf.zeros([1, self.p.embed_dim])
			_wrd_embeddings = tf.get_variable('embeddings')
			wrd_embeddings  = tf.concat([wrd_pad, _wrd_embeddings], axis=0)

			pos1_embeddings = tf.get_variable('pos1_embeddings')
			pos2_embeddings = tf.get_variable('pos2_embeddings')


		if self.p.ProbY:
			with tf.variable_scope('AliasInfo', reuse=tf.AUTO_REUSE) as scope:
				pad_alias_embed   = tf.zeros([1, self.p.alias_dim],     dtype=tf.float32, name='alias_pad')
				_alias_embeddings = tf.get_variable('alias_embeddings')
				alias_embeddings  = tf.concat([pad_alias_embed, _alias_embeddings], axis=0)

				alias_embed = tf.nn.embedding_lookup(alias_embeddings, self.input_proby_ind)
				alias_av    = tf.divide(tf.reduce_sum(alias_embed, axis=1), tf.expand_dims(self.input_proby_len, axis=1))

		wrd_embed  = tf.nn.embedding_lookup(wrd_embeddings,  in_wrds)
		pos1_embed = tf.nn.embedding_lookup(pos1_embeddings, in_pos1)
		pos2_embed = tf.nn.embedding_lookup(pos2_embeddings, in_pos2)
		embeds     = tf.concat([wrd_embed, pos1_embed, pos2_embed], axis=2)

		with tf.variable_scope('Bi-LSTM') as scope:
			fw_cell      = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, reuse=tf.AUTO_REUSE, name='FW_GRU'), output_keep_prob=self.rec_dropout)
			bk_cell      = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.p.lstm_dim, reuse=tf.AUTO_REUSE, name='BW_GRU'), output_keep_prob=self.rec_dropout)
			val, state   = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=self.x_len, dtype=tf.float32)

			lstm_out     = tf.concat((val[0], val[1]), axis=2)
			lstm_out_dim = self.p.lstm_dim*2

		if self.p.de_gcn:
			de_out = self.GCNLayer( gcn_in 		= lstm_out, 		in_dim 	    = lstm_out_dim, 		gcn_dim    = self.p.de_gcn_dim,
						batch_size 	= self.total_sents, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel,
						adj_ind 	= self.de_adj_ind, 	adj_data    = self.de_adj_data, 	w_gating   = self.p.wGate,
						num_layers 	= self.p.de_layers, 	name 	    = "GCN_DE")


			de_out 	   = de_out[-1]
			de_out 	   = tf.concat([lstm_out, de_out], axis=2)
			de_out_dim = self.p.de_gcn_dim + lstm_out_dim
		else:
			de_out 		= lstm_out
			de_out_dim	= lstm_out_dim


		with tf.variable_scope('Word_attention', reuse=tf.AUTO_REUSE) as scope:
			wrd_query    = tf.get_variable('wrd_query')
			sent_reps    = tf.reshape(
						tf.matmul(
							tf.reshape(
								tf.nn.softmax(
									tf.reshape(
										tf.matmul(
											tf.reshape(tf.tanh(de_out),[self.total_sents*self.seq_len, de_out_dim]),
											wrd_query),
										[self.total_sents, self.seq_len]
									)),
								[self.total_sents, 1, self.seq_len]),
							de_out),
						[self.total_sents, de_out_dim]
					)

			if self.p.ProbY:
				sent_reps  = tf.concat([sent_reps, alias_av], axis=1)
				de_out_dim += self.p.alias_dim

		if self.p.Type:
			with tf.variable_scope('TypeInfo', reuse=tf.AUTO_REUSE) as scope:
				pad_type_embed   = tf.zeros([1, self.type_dim],     dtype=tf.float32, name='type_pad')
				_type_embeddings = tf.get_variable('type_embeddings')
				type_embeddings  = tf.concat([pad_type_embed, _type_embeddings], axis=0)

				subtype = tf.nn.embedding_lookup(type_embeddings,  self.input_subtype)
				objtype = tf.nn.embedding_lookup(type_embeddings,  self.input_objtype)

				subtype_av = tf.divide(tf.reduce_sum(subtype, axis=1), tf.expand_dims(self.input_subtype_len, axis=1))
				objtype_av = tf.divide(tf.reduce_sum(objtype, axis=1), tf.expand_dims(self.input_objtype_len, axis=1))

				type_info = tf.concat([subtype_av, objtype_av], axis=1)

		with tf.variable_scope('Sentence_attention', reuse=tf.AUTO_REUSE) as scope:
			sent_atten_q = tf.get_variable('sent_atten_q')

			def getSentAtten(num):
				num_sents  	= num[1] - num[0]
				bag_sents   	= sent_reps[num[0]: num[1]]

				sent_atten_wts  = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(bag_sents), sent_atten_q), [num_sents]) )

				bag_rep_ 	= tf.reshape(
							tf.matmul(
								tf.reshape(sent_atten_wts, [1, num_sents]),
								bag_sents),
							[de_out_dim]
						  )
				return bag_rep_

			bag_rep = tf.map_fn(getSentAtten, self.sent_num, dtype=tf.float32)

		if self.p.Type:
			bag_rep    = tf.concat([bag_rep, type_info], axis=1)
			de_out_dim = de_out_dim + self.p.type_dim * 2

		with tf.variable_scope('FC1', reuse=tf.AUTO_REUSE) as scope:
			w_rel   = tf.get_variable('w_rel')
			b_rel   = tf.get_variable('b_rel')
			nn_out = tf.nn.xw_plus_b(bag_rep, w_rel, b_rel)

		with tf.name_scope('Accuracy') as scope:
			prob     = tf.nn.softmax(nn_out)
			y_pred   = tf.argmax(prob, 	   axis=1)
			y_actual = tf.argmax(self.input_y, axis=1)
			accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))

		''' Debugging command:
			res  = debug_nn([de_out], self.create_feed_dict( next(self.getBatches(self.data['train'])) ) ); pdb.set_trace()
		'''
		return nn_out, accuracy


	def add_loss(self, nn_out):
		with tf.name_scope('Loss_op'):
			loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return loss

	def add_optimizer(self, loss, isAdam=True):
		with tf.name_scope('Optimizer'):
			if isAdam: 	optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:		optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		return train_op

	def __init__(self, params):
		self.p  = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p)); pprint(vars(self.p))
		self.p.batch_size = self.p.batch_size

		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)

		self.load_data()
		self.initialize_variables()
		self.add_placeholders()

		nn_out, self.accuracy = self.add_model()

		self.loss      	= self.add_loss(nn_out)
		self.logits  	= tf.nn.softmax(nn_out)
		if self.p.opt == 'adam': self.train_op = self.add_optimizer(self.loss)
		else:			 self.train_op = self.add_optimizer(self.loss, isAdam=False)

		tf.summary.scalar('accmain', self.accuracy)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None

	def predict(self, sess, data, wLabels=True, shuffle=False):
		losses, accuracies, results, y_pred, y, logit_list, y_actual_hot, wrd_attens = [], [], [], [], [], [], [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):

			loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, dtype='test'))
			losses.    append(loss)
			accuracies.append(accuracy)

			pred_ind      = logits.argmax(axis=1)
			logit_list   += logits.tolist()
			y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
			y_pred       += pred_ind.tolist()
			y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
			bag_cnt      += len(batch['sent_num'])

			results.append(pred_ind)

			if step % 10 == 0:
				self.logger.info('Evaluating Test/Valid ({}/{}):\t{:.5}\t{:.5}\t{}'.format(bag_cnt, len(self.data['test']), np.mean(accuracies)*100, np.mean(losses), self.p.name))

		self.logger.info('Test/Valid Accuracy: {}'.format(accuracy))

		return np.mean(losses), results,  np.mean(accuracies)*100, y, y_pred, logit_list, y_actual_hot, wrd_attens

	def run_epoch(self, sess, data, epoch, shuffle=True):
		losses, accuracies = [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			summary_str, loss, accuracy, _ = sess.run([self.merged_summ, self.loss, self.accuracy, self.train_op], feed_dict=feed)

			losses.    append(loss)
			accuracies.append(accuracy)

			bag_cnt += len(batch['sent_num'])

			if step % 10 == 0:
				self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, len(self.data['train']), np.mean(accuracies)*100, np.mean(losses), self.p.name, self.best_val_area))
				self.summ_writer.add_summary(summary_str, epoch*len(self.data['train']) + bag_cnt)

		accuracy = np.mean(accuracies) * 100.0
		self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
		return np.mean(losses), accuracy

	def calc_prec_recall_f1(self, y_actual, y_pred, none_id):
		pos_pred, pos_gt, true_pos = 0.0, 0.0, 0.0

		for i in range(len(y_actual)):
			if y_actual[i] != none_id:
				pos_gt += 1.0

		for i in range(len(y_pred)):
			if y_pred[i] != none_id:
				pos_pred += 1.0					# classified as pos example (Is-A-Relation)
				if y_pred[i] == y_actual[i]:
					true_pos += 1.0

		precision 	= true_pos / (pos_pred + 1e-8)
		recall 		= true_pos / (pos_gt + 1e-8)
		f1 		= 2 * precision * recall / (precision + recall + 1e-8)

		return precision, recall, f1

	# Computes P@N for N = 100, 200, and 300
	def getPscore(self, data):
		val_loss, val_pred, val_acc, y, y_pred, logit_list, y_hot, wrd_attens = self.predict(sess, data)

		y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
		y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))

		allprob = np.reshape(np.array(y_scores), (-1))
		allans  = np.reshape(y_true, (-1))
		order   = np.argsort(-allprob)

		# P@100
		top100 = order[:100]
		correct_num_100 = 0.00
		for i in top100:
			if allans[i] == 1:
				correct_num_100 += 1.00

		p100 = correct_num_100 / 100.00

		# P@200
		top200 = order[:200]
		correct_num_200 = 0.00
		for i in top200:
			if allans[i] == 1:
				correct_num_200 += 1.00

		p200 = correct_num_200 / 200.00

		# P@300
		top300 = order[:300]
		correct_num_300 = 0.00
		for i in top300:
			if allans[i] == 1:
				correct_num_300 += 1.00

		p300 = correct_num_300 / 300.00

		return p100, p200, p300

	def fit(self, sess):
		self.summ_writer = tf.summary.FileWriter("tf_board/RE_NN/" + self.p.name, sess.graph)
		saver     = tf.train.Saver()
		save_dir  = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_path = os.path.join(save_dir, 'best_validation')
		self.val_areas = []

		if self.p.restore: saver.restore(sess, save_path)

		self.best_val_area  = 0.0
		self.best_train_acc = 0.0
		self.best_prf 	    = None

		for epoch in range(self.p.max_epochs):
			self.logger.info('Epoch: {}'.format(epoch))

			# train_loss, train_acc 				   	  	      = self.run_epoch(sess, self.data['train'], epoch)
			val_loss, val_pred, val_acc, y, y_pred, logit_list, y_hot, wrd_attens = self.predict  (sess, self.data['test'])

			val_prec, val_rec, val_f1 = self.calc_prec_recall_f1(y, y_pred, 0)	# 0: self.rel2id['NA']
			y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
			y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))

			area_pr  = average_precision_score(y_true, y_scores)
			self.val_areas.append(area_pr)

			self.logger.info('Main result: Prec:{} | Rec:{} | F1:{} | Area:{}'.format(val_prec, val_rec, val_f1, area_pr))

			if area_pr > self.best_val_area:
				self.best_val_area  = area_pr
				self.best_train_acc = train_acc
				self.best_prf 	    = {'prec': val_prec, 'rec': val_rec, 'f1': val_f1, 'area_pr': area_pr}
				pickle.dump({'logit_list': logit_list, 'y_hot': y_hot}, open("tf_board/RE_NN/{}/best_preds.pkl".format(self.p.name), 'wb'))
				pickle.dump(wrd_attens, open("tf_board/RE_NN/{}/word_attens.pkl".format(self.p.name), 'wb'))
				saver.save(sess=sess, save_path=save_path)

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_area))
			self.logger.info(self.best_prf)


		one_100, one_200, one_300 = self.getPscore(self.test_one)
		self.logger.info('TEST_ONE: P@100: {}, P@200: {}, P@300: {}'.format(one_100, one_200, one_300))
		one_avg = (one_100 + one_200 + one_300)/3

		two_100, two_200, two_300 = self.getPscore(self.test_two)
		self.logger.info('TEST_TWO: P@100: {}, P@200: {}, P@300: {}'.format(two_100, two_200, two_300))
		two_avg = (two_100 + two_200 + two_300)/3

		all_100, all_200, all_300 = self.getPscore(self.data['test'])
		self.logger.info('TEST_THREE: P@100: {}, P@200: {}, P@300: {}'.format(all_100, all_200, all_300))
		all_avg = (all_100 + all_200 + all_300)/3

		pprint ({
				'one_100':  one_100,
				'one_200':  one_200,
				'one_300':  one_300,
				'mean_one': avg_one,
				'two_100':  two_100,
				'two_200':  two_200,
				'two_300':  two_300,
				'mean_two': avg_two,
				'all_100':  all_100,
				'all_200':  all_200,
				'all_300':  all_300,
				'mean_all': avg_all,
		})


if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-data', 	 dest="dataset", 	required=True,							help='Dataset to use')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',							help='GPU to use')
	parser.add_argument('-pos_dim',  dest="pos_dim", 	default=16, 			type=int, 			help='Dimension of positional embeddings')
	parser.add_argument('-lstm',  	 dest="lstm", 		default='concat',	 					help='Bi-LSTM add/concat')
	parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=192,   	type=int, 					help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-num_class',dest="num_class", 	default=53,   	type=int, 					help='num classes for the dataset')

	parser.add_argument('-DE', 	 dest="de_gcn", 	action='store_false',   					help='Decide to include GCN in the model')
	parser.add_argument('-nGate', 	 dest="wGate", 		action='store_false',   					help='Decide to include gates in GCN')
	parser.add_argument('-Type', 	 dest="Type", 		action='store_false',						help='Decide to include Entity Type Side Information')
	parser.add_argument('-ProbY', 	 dest="ProbY", 		action='store_false', 						help='Decide to include Relation Alias Side Information')

	parser.add_argument('-type_dim', dest="type_dim", 	default=50,   			type=int, 			help='Type dimension')
	parser.add_argument('-alias_dim',dest="alias_dim", 	default=32,   			type=int, 			help='Alias dimension')
	parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=16,   			type=int, 			help='Hidden state dimension of GCN over dependency tree')
	parser.add_argument('-de_layer', dest="de_layers", 	default=1,   			type=int, 			help='Number of layers in GCN over dependency tree')
	parser.add_argument('-drop',	 dest="dropout", 	default=0.8,  			type=float,			help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.8,  			type=float,			help='Recurrent dropout for LSTM')

	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,			help='Learning rate')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 			help='L2 regularization')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=10,   			type=int, 			help='Max epochs')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=32,   			type=int, 			help='Batch size')
	parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 			help='Chunk size')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 						help='Restore from the previous best saved model')
	parser.add_argument('-opt',	 dest="opt", 		default='adam', 						help='Optimizer to use for training')

	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),				help='Name of the run')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,			help='Seed for randomization')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='/scratchd/home/shikhar/entity_linker/src/cesi/log/', 	help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='/scratchd/home/shikhar/entity_linker/src/config/', 	help='Config directory')
	#parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 						help='Log directory')
	#parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 						help='Config directory')
	parser.add_argument('-embed_loc',dest="embed_loc", 	default='./glove/glove.6B.50d_word2vec.txt', 			help='Log directory')
	parser.add_argument('-embed_dim',dest="embed_dim", 	default=50, type=int,						help='Dimension of embedding')
	args = parser.parse_args()

	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model  = RE_NN(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)
