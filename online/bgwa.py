from .base_model import *

class BGWA(Base):
	def add_placeholders(self):
		self.input_x  		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_data')
		self.input_y 		= tf.placeholder(tf.int32,   shape=[None, None],   name='input_labels')
		self.input_pos1 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos1')
		self.input_pos2 	= tf.placeholder(tf.int32,   shape=[None, None],   name='input_pos2')
		self.part_pos 		= tf.placeholder(tf.int32,   shape=[None, 2],      name='input_part_pos') 			# Subject and object position in the sentence.

		self.x_len		= tf.placeholder(tf.int32,   shape=[None],         name='input_len')
		self.sent_num 		= tf.placeholder(tf.int32,   shape=[None, 3], 	   name='sent_num')
		self.seq_len 		= tf.placeholder(tf.int32,   shape=(), 		   name='seq_len')
		self.total_bags 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_bags')
		self.total_sents 	= tf.placeholder(tf.int32,   shape=(), 		   name='total_sents')

		self.dropout 		= tf.placeholder_with_default(self.p.dropout, 	  shape=(), name='dropout')
		self.rec_dropout 	= tf.placeholder_with_default(self.p.rec_dropout, shape=(), name='rec_dropout')


	def pad_dynamic(self, X, pos1, pos2):
		seq_len, max_et = 0, 0
		x_len = np.zeros((len(X)), np.int32)

		for i, x in enumerate(X):
			seq_len  = max(seq_len, len(x))
			x_len[i] = len(x)

		x_pad,  _ 	= self.padData(X, seq_len)
		pos1_pad,  _ 	= self.padData(pos1, seq_len)
		pos2_pad,  _ 	= self.padData(pos2, seq_len)

		return x_pad, x_len, pos1_pad, pos2_pad, seq_len


	def create_feed_dict(self, batch, wLabels=True, dtype='train'):									# Where putting dropout for train?
		if not self.p.no_eval:
			X, Y, pos1, pos2, sent_num, part_pos = batch['X'], batch['Y'], batch['Pos1'], batch['Pos2'], batch['sent_num'], batch['PartPos']
		else:
			X, pos1, pos2, sent_num, part_pos = batch['X'], batch['Pos1'], batch['Pos2'], batch['sent_num'], batch['PartPos']

		total_sents = len(batch['X'])
		total_bags  = len(batch['Y'])
		x_pad, x_len, pos1_pad, pos2_pad, seq_len = self.pad_dynamic(X, pos1, pos2)

		if not self.p.no_eval: y_hot = self.getOneHot(Y, 	self.num_class)

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

		if not self.p.no_eval:
			if wLabels: feed_dict[self.input_y] 	= y_hot

		if dtype != 'train':
			feed_dict[self.dropout]     = 1.0
			feed_dict[self.rec_dropout] = 1.0
		else:
			feed_dict[self.dropout]     = self.p.dropout
			feed_dict[self.rec_dropout] = self.p.rec_dropout

		return feed_dict

	def add_model(self):
		in_wrds, in_pos1, in_pos2 = self.input_x, self.input_pos1, self.input_pos2

		with tf.variable_scope('Embeddings', reuse=tf.AUTO_REUSE) as scope:
			embed_init 	 = getGlove(self.wrd_list, self.p.embed_init)
			_wrd_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True, regularizer=self.regularizer)
			wrd_pad  	= tf.zeros([1, self.p.embed_dim])
			wrd_embeddings  = tf.concat([wrd_pad, _wrd_embeddings], axis=0)

			pos1_embeddings = tf.get_variable('pos1_embeddings', [self.max_pos, self.p.pos_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=True,   regularizer=self.regularizer)
			pos2_embeddings = tf.get_variable('pos2_embeddings', [self.max_pos, self.p.pos_dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=True,   regularizer=self.regularizer)

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

		with tf.variable_scope('Word_attention', reuse=tf.AUTO_REUSE) as scope:
			wrd_query    = tf.get_variable('wrd_query', [lstm_out_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
			sent_reps    = tf.reshape(
						tf.matmul(
							tf.reshape(
								tf.nn.softmax(
									tf.reshape(
										tf.matmul(
											tf.reshape(tf.tanh(lstm_out),[self.total_sents*self.seq_len, lstm_out_dim]),
											wrd_query),	
										[self.total_sents, self.seq_len]
									)),
								[self.total_sents, 1, self.seq_len]),
							lstm_out),
						[self.total_sents, lstm_out_dim]
					)

		with tf.variable_scope('Sentence_attention', reuse=tf.AUTO_REUSE) as scope:
			sent_atten_q = tf.get_variable('sent_atten_q', [lstm_out_dim, 1], initializer=tf.contrib.layers.xavier_initializer())

			def getSentAtten(num):
				num_sents  	= num[1] - num[0]
				bag_sents   	= sent_reps[num[0]: num[1]]

				sent_atten_wts  = tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(bag_sents), sent_atten_q), [num_sents]) )

				bag_rep_ 	= tf.reshape(
							tf.matmul(
								tf.reshape(sent_atten_wts, [1, num_sents]),
								bag_sents),
							[lstm_out_dim]
						  )
				return bag_rep_

			bag_rep = tf.map_fn(getSentAtten, self.sent_num, dtype=tf.float32)

		with tf.variable_scope('FC1', reuse=tf.AUTO_REUSE) as scope:
			w_rel   = tf.get_variable('w_rel', [lstm_out_dim, self.num_class], initializer=tf.contrib.layers.xavier_initializer(), 		regularizer=self.regularizer)
			b_rel   = tf.get_variable('b_rel', 				 initializer=np.zeros([self.num_class]).astype(np.float32), 	regularizer=self.regularizer)
			nn_out = tf.nn.xw_plus_b(bag_rep, w_rel, b_rel)

		with tf.name_scope('Accuracy') as scope:
			prob     = tf.nn.softmax(nn_out)
			y_pred   = tf.argmax(prob, 	   axis=1)
			y_actual = tf.argmax(self.input_y, axis=1)
			accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))



		''' Debugging command :
			res  = debug_nn([lstm_out], self.create_feed_dict( next(self.getBatches(self.data['train'])) ) ); pdb.set_trace()
		'''
		return nn_out, accuracy

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',								help='GPU to use')
	parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_50',	 						help='Embedding for initialization')
	parser.add_argument('-pos_dim',  dest="pos_dim", 	default=10, 			type=int, 				help='Dimension of positional embeddings')
	parser.add_argument('-lstm',  	 dest="lstm", 		default='concat',	 						help='Bi-LSTM add/concat')
	parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=128,   	type=int, 						help='Hidden state dimension of Bi-LSTM')

	parser.add_argument('-drop',	 dest="dropout", 	default=0.8,  			type=float,				help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.8,  			type=float,				help='Recurrent dropout for LSTM')

	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true',   							help='Calculate P@')
	parser.add_argument('-no_eval', dest="no_eval", 	action='store_true',   							help='Do not evaluate on the test set')
	parser.add_argument('-ntype',	 dest="Type", 		action='store_true', 							help='Restore from the previous best saved model')
	parser.add_argument('-nRelAlias',dest="RelAlias", 	action='store_true', 							help='Restore from the previous best saved model')
	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,				help='Learning rate')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 				help='L2 regularization')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=10,   			type=int, 				help='Max epochs')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=32,   			type=int, 				help='Batch size')
	parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 				help='Chunk size')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 							help='Restore from the previous best saved model')
	parser.add_argument('-opt',	 dest="opt", 		default='adam', 							help='Optimizer to use for training')
	parser.add_argument('-ngram', 	 dest="ngram", 		default=3, 			type=int, 				help='Window size')
	parser.add_argument('-n_filters',dest="num_filters", 	default=230, 			type=int, 				help='Filter size of BGWA')
	parser.add_argument('-max_pos',  dest="max_pos", 	default=60, 			type=int, 				help='Max length of pos')

	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),					help='Name of the run')
	parser.add_argument('-logdb', 	 dest="log_db", 	default='main_run',	 						help='MongoDB database for dumping results')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,				help='Seed for randomization')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='/scratchd/home/shikhar/entity_linker/src/cesi/log/', 		help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='/scratchd/home/shikhar/entity_linker/src/config/', 		help='Config directory')
	parser.add_argument('-rel2alias_file', 	default='../data/extended_m_inell_relations.json', 					help='File containing relation to alias mapping')
	parser.add_argument('-merge_rel',  default='/scratchd/home/shikhar/entity_linker/data/riedel_data/merge_decisions.json',	help='File containing remapping of relations')
	parser.add_argument('-index',  default='inell_bags_v3',	help='ES Index name')
	parser.add_argument('-redis',	 dest="redis", 	action='store_true', 							help='Restore from the previous best saved model')
	args = parser.parse_args()

	args.embed_dim = int(args.embed_init.split('_')[1])
	if not args.restore and not args.onlyTest: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model  = BGWA(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		if not args.onlyTest:
			model.fit(sess)
		else:
			saver     = tf.train.Saver()
			logger = model.logger

			save_dir  = os.path.join('checkpoints/', args.name)
			if not os.path.exists(save_dir):
				logger.info('Path {} doesnt exist.'.format(save_dir))
				sys.exit()
			save_path = os.path.join(save_dir, 'best_validation')

			saver.restore(sess, save_path) # Restore model

			if not args.no_eval:
				test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot, wrd_attens, e_pair = model.predict(sess, None, split = "test")
				test_prec, test_rec, test_f1 = model.calc_prec_recall_f1(y, y_pred, 0)
				y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
				y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
				area_pr  = average_precision_score(y_true, y_scores)
				logger.info('Main result (test): Prec:{} | Rec:{} | F1:{} | Area:{}'.format(test_prec, test_rec, test_f1, area_pr))
				pickle.dump({'logit_list': logit_list, 'y_hot': y_hot, 'e_pair':e_pair}, open("results/{}/onlyTest_precision_recall.pkl".format(args.name), 'wb'))	# Store predictions
			else:
				test_pred, y_pred, logit_list, wrd_attens, e_pair = model.predict(sess, None, split = "test", no_eval = True)
				pickle.dump({'logit_list': logit_list, 'e_pair':e_pair}, open("results/{}/onlyTest_precision_recall.pkl".format(args.name), 'wb'))	# Store predictions
