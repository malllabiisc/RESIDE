from .helper import *

sys.path.insert(0, '/scratchd/home/shikhar/services/queue/')
from queue_client import QueueClient

class Base(object):

	def padData(self, data, seq_len):
		temp = np.zeros((len(data), seq_len), np.int32)
		mask = np.zeros((len(data), seq_len), np.float32)

		for i, ele in enumerate(data):
			temp[i, :len(ele)] = ele[:seq_len]
			mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

		return temp, mask

	def getOneHot(self, data, num_class, isprob=False):
		temp = np.zeros((len(data), num_class), np.int32)
		for i, ele in enumerate(data):
			for rel in ele:
				if isprob:	temp[i, rel-1] = 1
				else:		temp[i, rel]   = 1
		return temp

	def getBatches(self, split= 'train', shuffle = True, no_eval = False, data = None):
		def posMap(pos):
			if   pos < -self.p.max_pos: 	return 0
			elif pos >  self.p.max_pos:  	return (self.p.max_pos + 1)*2
			else: 			  	return pos + (self.p.max_pos+1)

		def getId(wrd, wrd2id, def_val='UNK'):
			if wrd in wrd2id: return wrd2id[wrd]
			else: 		  return wrd2id[def_val]

		def redis_iter(client):
			keys = client.keys('REL||*')
			for k in keys:
				resp = client.hgetall(k)
				resp['sentences'] = eval(resp['sentences'])
				yield resp

		def get_prob(prob, type_):
			type_ind = self.type2id[type_]
			ohe = [0]*(len(self.type2id)+1)
			remaining_prob = float(1-prob)/(len(self.type2id)-1)
			ohe[type_ind] = prob-remaining_prob
			ohe = [x+remaining_prob for x in ohe]
			return ohe

		if data is not None:
			scanResp = data
		elif self.p.redis:
			redis_client = redis.StrictRedis('10.24.28.104', 6379, charset="utf-8", decode_responses=True)
			scanResp = redis_iter(redis_client)
		else:
			es = Elasticsearch(["10.24.28.112", ],
					sniff_on_start       	 = True,
					sniff_on_connection_fail = True,
					sniffer_timeout      	 = 6000,
					maxsize          	 = 50,
					timeout          	 = 1200,
					max_retries      	 = 10,
					http_auth		 = ('sai', 'mall123'),
					retry_on_timeout 	 = True)

			scanResp = scan(client= es,
					query={"query": {"match_phrase": {"doc.split": split}}},
					scroll= "23h", index=self.p.index, timeout="10m", size=10)

		old_num, num, bag_id = 0, 0, 0

		batch = ddict(list)
		self.total_bags_data = 0

		for k, ele in enumerate(scanResp):
			self.total_bags_data += 1
			
			if data is not None:
				ele = ele
			elif not self.p.redis:
				ele = ele['_source']['doc']
			sub = ele['sub']
			obj = ele['obj']
			if not no_eval:
				rel = ele['relation']
			else:
				rel = None
			subtype = self.type2id[ele['sub_type']]
			objtype = self.type2id[ele['obj_type']]
			
			if ele["sub_type"]=="outfit":
				try:
					batch["sub_prob"].append(get_prob(ele['prob'], ele['sub_type'])) #[ele["prob"]])
				except:
					batch["sub_prob"].append(get_prob(1, ele['sub_type']))
			else:		
				batch["sub_prob"].append(get_prob(1, ele['sub_type']))

			if ele["obj_type"]=="outfit":
				try:
					batch["obj_prob"].append(get_prob(ele['prob'], ele['obj_type']))
				except:
					batch["obj_prob"].append(get_prob(1, ele['obj_type']))
			else:		
				batch["obj_prob"].append(get_prob(1, ele['obj_type']))

			sub_words = re.split(" |-", sub)
			obj_words = re.split(" |-", obj)

			for sent_ in ele['sentences']:
				if sent_['corenlp']=='null':
					# print('CoreNLP missed !!!!')
					continue

				sent     = json.loads(sent_['corenlp'])['sentences'][0]
				tok_list = [tok['word'].lower() for tok in sent['tokens']]

				slen = len(tok_list)
				if sub_words[0] in tok_list and obj_words[0] in tok_list:
					sub_pos	= tok_list.index(sub_words[0])
					obj_pos	= tok_list.index(obj_words[0])
				else:
					sub_pos, obj_pos = int(slen/2-1), int(slen/2+1)

				if self.p.RelAlias:
					""" Extract Phrases from setences (Syntactic Context Extractor) """
					phrases = set()

					if sent['openie'] != None:
						for openie in sent['openie']:
							if openie['subject'].lower() == sub.replace('_', ' ') and openie['object'].lower() == obj.replace('_', ' '):
								phrases.add(openie['relation'])
					openie_phrases = phrases.copy()

					if abs(sub_pos - obj_pos) < 5:
						middle_phr = ' '.join(tok_list[min(sub_pos, obj_pos)+1: max(sub_pos, obj_pos)])
						phrases.add(middle_phr)
					else:   middle_phr = ''

					dep_links = []

					corenlp_sent = sent
					dep_edges = corenlp_sent['basicDependencies']
					for dep in dep_edges:
						if dep['governor'] == 0 or dep['dependent'] == 0: continue					# Ignore ROOT
						dep_links.append((dep['governor']-1, dep['dependent']-1, 0, 1))	# -1, because indexing starts from 0

					right_nbd_phrase, left_nbd_phrase, mid_phrase = set(), set(), set()

					for edge in dep_links:
						if edge[0] == sub_pos or edge[0] == obj_pos:
							if   edge[1] > min(sub_pos, obj_pos) and edge[1] < max(sub_pos, obj_pos): mid_phrase.add(tok_list[edge[1]])
							elif edge[1] < min(sub_pos, obj_pos): 					  left_nbd_phrase.add(tok_list[edge[1]])
							else: 									  right_nbd_phrase.add(tok_list[edge[1]])
						if edge[1] == sub_pos or edge[1] == obj_pos:
							if   edge[0] > min(sub_pos, obj_pos)  and edge[0] < max(sub_pos, obj_pos): mid_phrase.add(tok_list[edge[0]])
							elif edge[0] < min(sub_pos, obj_pos): 					   left_nbd_phrase.add(tok_list[edge[0]])
							else: 									   right_nbd_phrase.add(tok_list[edge[0]])

					left_nbd_phrase  = ' '.join(list(left_nbd_phrase  - {sub, obj}))
					right_nbd_phrase = ' '.join(list(right_nbd_phrase - {sub, obj}))
					mid_phrase 	 = ' '.join(list(mid_phrase))

					phrases.add(left_nbd_phrase)
					phrases.add(right_nbd_phrase)
					phrases.add(middle_phr)
					phrases.add(mid_phrase)
					phrases   = list(phrases - {''})

					""" Extracting Relation Alias Side Information (Refer paper RESIDE) """
					phr_embed = getPhr2vec(phrases, 'wiki_300')
					dist      = cdist(phr_embed, self.alias_embed, metric='cosine')
					rels      = set()

					for cphr in np.argmin(dist, 1):
						for i in range(dist.shape[0]):
							if dist[i, cphr] < 0.65: rels |= self.alias2rel[cphr]

					probY   = [self.rel2id[r] for r in rels if r in self.rel2id]
					batch['ProbY'].append(probY)

				batch['PartPos'].append([ sub_pos, obj_pos ])
				X, Pos1, Pos2, PartPos = [], [], [], []
				batch['X']	.append([getId(tok, self.voc2id) for tok in tok_list])
				batch['Pos1']	.append([posMap(i - sub_pos)     for i   in range(len(tok_list))])
				batch['Pos2']	.append([posMap(i - obj_pos)     for i   in range(len(tok_list))])
				num 	+= 1

			if old_num == num: continue

			if self.p.Type:
				batch['SubType'].append([subtype])
				batch['ObjType'].append([objtype])

			batch['sent_num'].append([old_num, num, bag_id])
			old_num = num
			if not no_eval:
				batch['Y'].append([self.rel2id[rel]])
			else:
				batch['Y'].append([None])
			batch['e_pair'].append((sub, obj, ))

			bag_id += 1

			if bag_id == self.p.batch_size:
				yield batch
				batch = ddict(list)
				old_num, num, bag_id = 0, 0, 0

		yield batch

	def load_data(self):
		c_dosa 	 = MongoClient('mongodb://10.24.28.104:27017/')
		glove_db = c_dosa['glove']['wiki_50']

		# Define vocabulary
		self.voc2id = {}
		for i, doc in enumerate(glove_db.find({}, {'_id':1})):
			self.voc2id[doc['_id']] = i
			if i > 100000: break

		self.voc2id['UNK'] = len(self.voc2id)
		self.max_pos 	   = self.p.max_pos
		self.num_deLabel   = 1

		rel2alias = json.loads(open(self.p.rel2alias_file).read())
		self.alias2rel = ddict(set)
		alias2id  = {}
		for rel, aliases in rel2alias.items():
			for alias in aliases:
				if alias in alias2id:
					self.alias2rel[alias2id[alias]].add(rel)
				else:
					alias2id[alias] = len(alias2id)
					self.alias2rel[alias2id[alias]].add(rel)

		temp = sorted(alias2id.items(), key=operator.itemgetter(1))
		temp.sort(key = lambda x:x[1])
		alias_list, _ = zip(*temp)
		self.alias_embed   = getPhr2vec(alias_list, 'wiki_300')

		self.rel2id = {'NA': 0}
		for i, rel in enumerate(sorted(rel2alias.keys())):
			self.rel2id[rel] = len(self.rel2id)

		# pdb.set_trace()

		self.type2id   = json.load(open(self.p.type2id_file))
		self.type_num  = len(self.type2id)
		self.num_class = len(self.rel2id)

		# Get Word List
		self.wrd_list 	   = list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list,_    = zip(*self.wrd_list)


	def add_placeholders(self):
		raise NotImplementedError

	def pad_dynamic(self, X, pos1, pos2):
		raise NotImplementedError

	def create_feed_dict(self, batch, wLabels=True, dtype='train'):									# Where putting dropout for train?
		raise NotImplementedError

	def add_model(self):
		raise NotImplementedError

	def add_loss(self, nn_out):
		with tf.name_scope('Loss_op'):
			loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		return loss

	def add_optimizer(self, loss, isAdam=True):
		with tf.name_scope('Optimizer'):
			if self.p.opt == 'adam': optimizer = tf.train.AdamOptimizer(self.p.lr)
			else:			 optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		return train_op

	def __init__(self, params):
		self.p  = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.log_db = MongoClient('mongodb://10.24.28.104:27017/')['ikt'][self.p.log_db]
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

	def predict(self, sess, data, wLabels=True, shuffle=False, split="valid", no_eval = False):
		losses, accuracies, results, y_pred, y, logit_list, y_actual_hot, e_pair, wrd_attens = [], [], [], [], [], [], [], [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(split = split, shuffle = shuffle, no_eval = no_eval)):
			if no_eval:
				logits = sess.run(self.logits, feed_dict = self.create_feed_dict(batch, dtype='test'))
			else:
				loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, dtype='test'))
				losses.    append(loss)
				accuracies.append(accuracy)
				y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
				y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()

			pred_ind      = logits.argmax(axis=1)
			e_pair	     += batch['e_pair']
			logit_list   += logits.tolist()
			y_pred       += pred_ind.tolist()
			bag_cnt      += len(batch['sent_num'])

			results.append(pred_ind)

			if step % 10 == 0:
				if no_eval:
					self.logger.info('Evaluating {} ({}/{}):\t{}'.format(split, bag_cnt, self.total_bags_data, self.p.name))
				else:
					self.logger.info('Evaluating {} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(split, bag_cnt, self.total_bags_data, np.mean(accuracies)*100, np.mean(losses), self.p.name))

		if not no_eval: self.logger.info('{} Accuracy: {}'.format(split, accuracy))

		if not no_eval:
			return np.mean(losses), results,  np.mean(accuracies)*100, y, y_pred, logit_list, y_actual_hot, wrd_attens, e_pair
		else:
			return results, y_pred, logit_list, wrd_attens, e_pair

	def run_epoch(self, sess, epoch, shuffle=True):
		losses, accuracies = [], []
		bag_cnt = 0

		for step, batch in enumerate(self.getBatches(shuffle = shuffle)):
			feed = self.create_feed_dict(batch)
			summary_str, loss, accuracy, _ = sess.run([self.merged_summ, self.loss, self.accuracy, self.train_op], feed_dict=feed)

			losses.    append(loss)
			accuracies.append(accuracy)

			bag_cnt += len(batch['sent_num'])

			if step % 10 == 0:
				self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, self.total_bags_data, np.mean(accuracies)*100, np.mean(losses), self.p.name, self.best_val_area))
				self.summ_writer.add_summary(summary_str, epoch*self.total_bags_data + bag_cnt)

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

	def fit(self, sess):
		self.summ_writer = tf.summary.FileWriter("tf_board/Base/" + self.p.name, sess.graph)
		saver     = tf.train.Saver()
		save_dir  = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_path = os.path.join(save_dir, 'best_validation')
		res_dir   = 'results/{}/'.format(self.p.name);     os.makedirs(res_dir)
		self.val_areas = []

		if self.p.restore:
			doc = self.log_db.find_one({'_id': self.p.name}, {'Best_val_area':1, 'Best_train_acc':1, 'results': 1})
			self.best_val_area 	= doc['Best_val_area']
			self.best_train_acc = doc['Best_train_acc']
			self.best_prf		= doc['results']
			saver.restore(sess, save_path)
		else:
			self.best_val_area  = 0.0
			self.best_train_acc = 0.0
			self.best_prf 	    = None

		for epoch in range(self.p.max_epochs):
			self.logger.info('Epoch: {}'.format(epoch))

			train_loss, train_acc 				   	  	      = self.run_epoch(sess, epoch)
			val_loss, val_pred, val_acc, y, y_pred, logit_list, y_hot, wrd_attens, e_pair = self.predict(sess, None, split = "valid")

			val_prec, val_rec, val_f1 = self.calc_prec_recall_f1(y, y_pred, 0)
			y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
			y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
			area_pr  = average_precision_score(y_true, y_scores)
			self.val_areas.append(area_pr)

			self.logger.info('Main result (val): Prec:{} | Rec:{} | F1:{} | Area:{}'.format(val_prec, val_rec, val_f1, area_pr))
			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_area))

			if area_pr > self.best_val_area:
				self.best_val_area  = area_pr
				self.best_train_acc = train_acc
				self.best_prf 	    = {'prec': val_prec, 'rec': val_rec, 'f1': val_f1, 'area_pr': area_pr}
				saver.save(sess=sess, save_path=save_path)
				
				test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot, wrd_attens, e_pair = self.predict(sess, None, split = "test")
				test_prec, test_rec, test_f1 = self.calc_prec_recall_f1(y, y_pred, 0)
				y_true   = np.array([e[1:] for e in y_hot]).  	 reshape((-1))
				y_scores = np.array([e[1:] for e in logit_list]).reshape((-1))
				area_pr  = average_precision_score(y_true, y_scores)

				self.logger.info('Main result (test): Prec:{} | Rec:{} | F1:{} | Area:{}'.format(val_prec, val_rec, val_f1, area_pr))
				pickle.dump({'logit_list': logit_list, 'y_hot': y_hot, 'e_pair':e_pair}, open("results/{}/precision_recall.pkl".format(self.p.name), 'wb'))	# Store predictions

			self.logger.info(self.best_prf)

			try:
				self.log_db.update({'_id': self.p.name}, {
					'$push': {
						"Train_loss": 	round(float(train_loss),3),
						"Train_acc": 	round(float(train_acc),	3),
						"Valid_loss": 	round(float(val_loss),	3),
						"Valid_area":	round(float(area_pr),	3)
					},
					'$set': {
						"Best_val_area":	round(float(self.best_val_area),  3),
						"Best_train_acc":	round(float(self.best_train_acc), 3),
						"results":		self.best_prf,
						"Params":		vars(self.p)
					}
				}, upsert=True)
			except Exception as e: continue
			

if __name__== "__main__":

	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-gpu', 	 dest="gpu", 		default='0',								help='GPU to use')
	parser.add_argument('-embed', 	 dest="embed_init", 	default='wiki_50',	 						help='Embedding for initialization')
	parser.add_argument('-pos_dim',  dest="pos_dim", 	default=10, 			type=int, 				help='Dimension of positional embeddings')
	parser.add_argument('-lstm',  	 dest="lstm", 		default='concat',	 						help='Bi-LSTM add/concat')
	parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=128,   	type=int, 						help='Hidden state dimension of Bi-LSTM')

	parser.add_argument('-type_dim', dest="type_dim", 	default=50,   			type=int, 			help='Type dimension')
	parser.add_argument('-alias_dim',dest="alias_dim", 	default=32,   			type=int, 			help='Alias dimension')
	parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=16,   			type=int, 			help='Hidden state dimension of GCN over dependency tree')
	parser.add_argument('-de_layer', dest="de_layers", 	default=1,   			type=int, 			help='Number of layers in GCN over dependency tree')
	parser.add_argument('-drop',	 dest="dropout", 	default=0.8,  			type=float,			help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.8,  			type=float,			help='Recurrent dropout for LSTM')

	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true',   							help='Calculate P@')
	parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,				help='Learning rate')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 				help='L2 regularization')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default=10,   			type=int, 				help='Max epochs')
	parser.add_argument('-batch', 	 dest="batch_size", 	default=32,   			type=int, 				help='Batch size')
	parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 				help='Chunk size')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 							help='Restore from the previous best saved model')
	parser.add_argument('-ntype',	 dest="Type", 	action='store_false', 							help='Restore from the previous best saved model')
	parser.add_argument('-nRelAlias',	 dest="RelAlias", 	action='store_false', 							help='Restore from the previous best saved model')
	parser.add_argument('-n_de_gcn',	 dest="de_gcn", 	action='store_false', 							help='Restore from the previous best saved model')
	parser.add_argument('-opt',	 dest="opt", 		default='adam', 							help='Optimizer to use for training')
	parser.add_argument('-ngram', 	 dest="ngram", 		default=3, 			type=int, 				help='Window size')
	parser.add_argument('-n_filters',dest="num_filters", 	default=230, 			type=int, 				help='Filter size of RESIDE')
	parser.add_argument('-max_pos',  dest="max_pos", 	default=60, 			type=int, 				help='Max length of pos')
	parser.add_argument('-nGate', 	 dest="wGate", 		action='store_false',   					help='Decide to include gates in GCN')

	parser.add_argument('-eps', 	 dest="eps", 		default=0.00000001,  		type=float, 			help='Value of epsilon')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),					help='Name of the run')
	parser.add_argument('-logdb', 	 dest="log_db", 	default='main_run',	 						help='MongoDB database for dumping results')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,				help='Seed for randomization')
	parser.add_argument('-logdir',	 dest="log_dir", 	default='/scratchd/home/shikhar/entity_linker/src/cesi/log/', 		help='Log directory')
	parser.add_argument('-config',	 dest="config_dir", 	default='/scratchd/home/shikhar/entity_linker/src/config/', 		help='Config directory')
	parser.add_argument('-rel2id_map',     	default='/scratche/home/shubham/ikgt/data/inell_relations.json', 						help='File containing relation to id mapping')
	parser.add_argument('-rel2alias_file', 	default='/scratche/home/shubham/ikgt/data/extended_m_inell_relations.json', 					help='File containing relation to alias mapping')
	parser.add_argument('-type2id_file',   	default='/scratche/home/shubham/ikgt/data/type2id.json', 							help='File containing type to id mapping')
	parser.add_argument('-merge_rel',  	default='/scratchd/home/shikhar/entity_linker/data/riedel_data/merge_decisions.json',	help='File containing remapping of relations')
	parser.add_argument('-index',  		default='inell_bags_v3',								help='ES Index name')
	args = parser.parse_args()

	args.embed_dim = int(args.embed_init.split('_')[1])
	if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)

	model  = RESIDE(args)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		model.fit(sess)