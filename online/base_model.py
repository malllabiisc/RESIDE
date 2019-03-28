import sys; sys.path.insert(0, './')
from helper import *
import tensorflow as tf
from scipy.spatial.distance import cdist

class Base(object):

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

	def read_data(self, bag):

		def posMap(pos):
			if   pos < -self.p.max_pos: 	return 0
			elif pos >  self.p.max_pos:  	return (self.p.max_pos + 1)*2
			else: 			  	return pos + (self.p.max_pos+1)

		def getId(wrd, wrd2id, def_val='UNK'):
			if wrd in wrd2id: return wrd2id[wrd]
			else: 		  return wrd2id[def_val]

		sub, obj = bag['sub'], bag['obj']
		sub_type = list({self.type2id[typ.split('/')[1]] for typ in self.ent2type[bag['sub_id']]})
		obj_type = list({self.type2id[typ.split('/')[1]] for typ in self.ent2type[bag['obj_id']]})

		batch, num = ddict(list), 0
			
		sub_words = re.split("_|- ", sub)
		obj_words = re.split("_|- ", obj)

		for sent_ in bag['sents']:
			if sent_['corenlp']=='null' or sent_['corenlp'] == None: continue

			sent     = sent_['corenlp']['sentences'][0]
			tok_list = [tok['word'].lower() for tok in sent['tokens']]

			slen = len(tok_list)
			if sub_words[0] in tok_list and obj_words[0] in tok_list:
				sub_pos	= tok_list.index(sub_words[0])
				obj_pos	= tok_list.index(obj_words[0])
			else:
				sub_pos, obj_pos = int(slen/2-1), int(slen/2+1)

			""" Extract Phrases from setences (Syntactic Context Extractor) """
			phrases = set()

			if 'openie' in sent and sent['openie'] != None:
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
				if dep['governor'] == 0 or dep['dependent'] == 0: continue	# Ignore ROOT
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
			phrases = list(phrases - {''})

			""" Extracting Relation Alias Side Information (Refer paper RESIDE) """
			phr_embed = getPhr2vec(self.voc_model, phrases, self.p.embed_dim)
			dist      = cdist(phr_embed, self.alias_embed, metric='cosine')
			rels      = set()

			for cphr in np.argmin(dist, 1):
				for i in range(dist.shape[0]):
					if dist[i, cphr] < 0.65: rels |= self.alias2rel[cphr]

			probY     = [self.rel2id[r] for r in rels if r in self.rel2id]
			batch['ProbY'].append(probY)

			batch['DepEdges'].append(dep_links)
			batch['PartPos']. append([ sub_pos, obj_pos ])
			batch['Pos1']. 	  append([posMap(i - sub_pos)     for i   in range(len(tok_list))])
			batch['Pos2']. 	  append([posMap(i - obj_pos)     for i   in range(len(tok_list))])
			batch['X']. 	  append([getId(tok, self.voc2id) for tok in tok_list])
			num += 1


		batch['Y'].append([None])
		batch['SubType'].append(sub_type)
		batch['ObjType'].append(obj_type)
		batch['sent_num'].append([0, num, 0])

		return dict(batch)

	def load_data(self):

		self.voc_model		= gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
		self.max_pos		= (self.p.max_pos+1)*2 + 1
		self.num_deLabel	= 1

		# Reading Relation Alias side information
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

		temp = sorted(alias2id.items(), key=lambda x: x[1])
		temp.sort(key		= lambda x: x[1])
		alias_list, _		= zip(*temp)
		self.alias_embed	= getPhr2vec(self.voc_model, alias_list, self.p.embed_dim)

		self.voc2id	   	= json.load(open('./data/{}_voc2id.json'.format(self.p.dataset)))
		self.rel2id		= json.load(open('./data/{}_rel2id.json'.format(self.p.dataset)))
		self.id2rel 		= {v: k for k, v in self.rel2id.items()}
		self.type2id		= json.load(open('./data/{}_type2id.json'.format(self.p.dataset)))
		self.type_num		= len(self.type2id)

		self.ent2type		= json.load(open(self.p.type2id_file))
		self.num_class		= len(self.rel2id)

		# Get Word List
		self.wrd_list 	   = list(self.voc2id.items())					# Get vocabulary
		self.wrd_list.sort(key=lambda x: x[1])						# Sort vocabulary based on ids
		self.wrd_list, _   = zip(*self.wrd_list)

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

