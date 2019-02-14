from pymongo import MongoClient
from random import randint
from pprint import pprint
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, average_precision_score
from collections import defaultdict as ddict

import numpy as np, sys, unicodedata, requests, os, random, ipdb as pdb, requests, json, itertools
import uuid, time, argparse, re, operator, pickle
import scipy.sparse as sp, networkx as nx
import logging, logging.config, itertools, pathlib

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError, scan

import redis

import tensorflow as tf

np.set_printoptions(precision=4)

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def getWord2vec(wrd_list):
	dim = 300
	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_word2vec.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds

def getGlove(wrd_list, embed_type, c_dosa=None):
	if c_dosa == None: c_dosa = MongoClient('mongodb://10.24.28.104:27017/')
	dim = int(embed_type.split('_')[1])
	db_glove = c_dosa['glove'][embed_type]

	embeds = np.zeros((len(wrd_list), dim), np.float32)
	embed_map = {}

	res = db_glove.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']

	count = 0
	for wrd in wrd_list:
		if wrd in embed_map: 	embeds[count, :] = np.float32(embed_map[wrd])
		else: 			embeds[count, :] = np.random.randn(dim)
		count += 1

	return embeds

def getPhr2vec(phr_list, embed_type, c_dosa=None):
	if c_dosa == None: c_dosa = MongoClient('mongodb://10.24.28.104:27017/')
	dim 	 = int(embed_type.split('_')[1])
	db_glove = c_dosa['glove'][embed_type]
	
	wrd_list = []

	embeds    = np.zeros((len(phr_list), dim), np.float32)
	embed_map = {}

	for phr in phr_list:
		wrd_list += phr.split()

	wrd_list = list(set(wrd_list))

	res = db_glove.find({"_id": {"$in": wrd_list}})
	for ele in res:
		embed_map[ele['_id']] = ele['vec']
	
	for i, phr in enumerate(phr_list):
		wrds = phr.split()
		vec  = np.zeros((dim,), np.float32)
		
		for wrd in wrds:
			if wrd in embed_map: 	vec += np.float32(embed_map[wrd])
			else: 			vec += np.random.normal(size=dim, loc=0, scale=0.05)

		vec = vec / len(wrds)
		embeds[i, :] = vec
	return embeds

def signal(message):
	requests.post( 'http://10.24.28.210:9999/jobComplete', data=message)

def len_key(tp):
	return len(tp[1])

def set_gpu(gpus):
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def shape(tensor):
	s = tensor.get_shape()
	return tuple([s[i].value for i in range(0, len(s))])


coreNLP_url = []

def callnlpServer(text):
        params = {
        	'properties': 	'{"annotators":"tokenize"}',
        	'outputFormat': 'json'
        }

        res = requests.post(	coreNLP_url[randint(0, len(coreNLP_url)-1)],
        			params=params, data=text, 
        			headers={'Content-type': 'text/plain'})

        if res.status_code == 200: 	return res.json()
        else: 				print("CoreNLP Error, status code:{}".format(res.status_codet))


def debug_nn(res_list, feed_dict):
	import tensorflow as tf
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	sess.run(tf.global_variables_initializer())
	summ_writer = tf.summary.FileWriter("tf_board/debug_nn", sess.graph)
	res = sess.run(res_list, feed_dict = feed_dict)
	return res

def stanford_tokenize(text):
	res = callnlpServer(text)
	toks = [ele['word'] for ele in res['tokens']]
	return toks


def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
 
	return False

def is_int(s):
	try:
		int(s)
		return True
	except ValueError:
		return False

def get_logger(name, log_dir, config_dir):
	# Disable logging for certain modules
	logging.getLogger('elasticsearch').setLevel(logging.CRITICAL)
	logging.getLogger('elasticsearch.helpers').setLevel(logging.CRITICAL)
	logging.getLogger('urllib3').setLevel(logging.CRITICAL)

	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)


	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def partition(lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def getChunks(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def mergeList(list_of_list):
	return list(itertools.chain.from_iterable(list_of_list))

# doc = 'Delhi is the capital of India. Mumbai is not the capital of India.'
# pprint(callnlpServer(doc))

from nltk import Tree

def get_phrases(parse_str):
	"""
	This function takes a Constituency parse string and returns the verb phrases 
	extracted from the tree
	Written by Rishabh Joshi @ 09/04/2018
	"""
	t = Tree.fromstring(parse_str)
	selected_phrases = []
	for i in t.subtrees(filter=lambda x: x.label() == 'VP'):
		phrase = i.leaves()
		rem = []
		for np in i.subtrees(filter=lambda x: x.label() == 'NP'):
			remove_phrase = np.leaves()
			rem += remove_phrase
		#print (set(phrase) - set(rem))
		selected_phrase = []
		for w in phrase:
			if w not in rem:
				selected_phrase.append(w)
			else:
				break  # break at first not found, other VP will find rest
		selected_phrases.append(selected_phrase)

	final = []
	i = 0
	# The list has subparts of paraphrases
	skipped = {}  # set of all skipped
	while i < len(selected_phrases):
		if i in skipped:
			i += 1
			continue
		j = i + 1
		while j < len(selected_phrases):
			if set(selected_phrases[i]) >= set(selected_phrases[j]):
				skipped[j] = 1
			j += 1
		if len(selected_phrases[i]) > 0:
			final.append(selected_phrases[i])
		i += 1
	return final
