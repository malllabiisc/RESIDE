import numpy as np, os, sys, random,  argparse
import pickle, uuid, time, pdb, json, gensim, itertools
import logging, logging.config, pathlib

from collections import defaultdict as ddict
from nltk.tokenize import word_tokenize
from pprint import pprint
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score

# Set precision for numpy
np.set_printoptions(precision=4)

# Reads the embeddings in word2vec format
def getEmbeddings(model, wrd_list, embed_dims):
	embed_list = []

	for wrd in wrd_list:
		if wrd in model.vocab: 	embed_list.append(model.word_vec(wrd))
		else: 			embed_list.append(np.random.randn(embed_dims))

	return np.array(embed_list, dtype=np.float32)

def getPhr2vec(model, phr_list, embed_dims):	
	wrd_list = []
	embeds   = np.zeros((len(phr_list), embed_dims), np.float32)
	
	for i, phr in enumerate(phr_list):
		wrds = phr.split()
		vec  = np.zeros((embed_dims,), np.float32)
		
		for wrd in wrds:
			if wrd in model.vocab: 	vec += model.word_vec(wrd)
			else: 			vec += np.random.normal(size=embed_dims, loc=0, scale=0.05)
		vec = vec / len(wrds)
		embeds[i, :] = vec
		
	return embeds

def rel_encoder(model, phr_list, embed_dims):
	embed_list = []

	for phr in phr_list:
		if phr in model.vocab:
			embed_list.append(model.word_vec(phr))
		else:
			vec = np.zeros(embed_dims, np.float32)
			wrds = word_tokenize(phr)
			for wrd in wrds:
				if wrd in model.vocab: 	vec += model.word_vec(wrd)
				else:			vec += np.random.randn(embed_dims)
			embed_list.append(vec / len(wrds))

	return np.array(embed_list)


# Sets which gpus to use
def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

# Check whether file is present or not
def checkFile(filename):
	return pathlib.Path(filename).is_file()

# Creates the directory if doesn't exist
def make_dir(dir_path):
	if not os.path.exists(dir_path): 
		os.makedirs(dir_path)

# For debugging Tensorflow model
def debug_nn(res_list, feed_dict):
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

# Creates logger
def get_logger(name, log_dir, config_dir):
	make_dir(log_dir)
	config_dict = json.load(open(config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

# Splits inp_list into lists of size chunk_size
def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

# Paritions a given list into chunks of size n
def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

# Merges list of list into list
def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))