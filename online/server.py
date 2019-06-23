import sys, os, pdb
sys.path.insert(0, './')

import tensorflow as tf 
from flask import Flask, render_template, request
from helper import *
from online_reside import RESIDE
from flask_cors import CORS, cross_origin

app		= Flask(__name__, static_folder='static-entice')
cors		= CORS(app, resources={r"/api/*": {"origins": "*"}})
bag_list	= []
model		= None


@app.route('/reside/')
def resideMain():
	bag	= random.choice(bag_list)
	batch	= model.read_data(bag)
	feed	= model.create_feed_dict(batch)
	logits	= sess.run(model.logits, feed_dict=feed)
	pred    = model.id2rel[np.argmax(logits)]

	return render_template('reside_main.html', actual = model.id2rel[bag['rel'][0]], pred=pred, bag=bag)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main Neural Network for Time Stamping Documents')

	parser.add_argument('-data', 	 	dest="dataset", 	default='riedel',				help='Dataset to use')
	parser.add_argument('-gpu', 	 	dest="gpu", 		default='0',					help='GPU to use')
	parser.add_argument('-nGate', 	 	dest="wGate", 		action='store_false',   			help='Include edgewise-gating in GCN')
	parser.add_argument('-lstm_dim', 	dest="lstm_dim", 	default=192,   		type=int, 		help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-port', 		dest="port", 		default=3535,   	type=int, 		help='Port of the server')
	parser.add_argument('-pos_dim',  	dest="pos_dim", 	default=16, 		type=int, 		help='Dimension of positional embeddings')
	parser.add_argument('-type_dim', 	dest="type_dim", 	default=50,   		type=int, 		help='Type dimension')
	parser.add_argument('-alias_dim',	dest="alias_dim", 	default=32,   		type=int, 		help='Alias dimension')
	parser.add_argument('-de_dim',   	dest="de_gcn_dim", 	default=16,   		type=int, 		help='Hidden state dimension of GCN over dependency tree')
	parser.add_argument('-max_pos',  	dest="max_pos", 	default=60,   		type=int, 		help='Maximum position difference to consider')
	parser.add_argument('-de_layer', 	dest="de_layers", 	default=1,   		type=int, 		help='Number of layers in GCN over dependency tree')
	parser.add_argument('-drop',	 	dest="dropout", 	default=0.8,  		type=float,		help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 	dest="rec_dropout", 	default=0.8,  		type=float,		help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 	dest="lr", 		default=0.001,  	type=float,		help='Learning rate')
	parser.add_argument('-l2', 	 	dest="l2", 		default=0.001,  	type=float, 		help='L2 regularization')
	parser.add_argument('-epoch', 	 	dest="max_epochs", 	default=2,   		type=int, 		help='Max epochs')
	parser.add_argument('-batch', 	 	dest="batch_size", 	default=32,   		type=int, 		help='Batch size')
	parser.add_argument('-chunk', 	 	dest="chunk_size", 	default=1000,   	type=int, 		help='Chunk size')
	parser.add_argument('-restore',	 	dest="restore", 	action='store_true', 				help='Restore from the previous best saved model')
	parser.add_argument('-only_eval',	dest="only_eval", 	action='store_true', 				help='Only Evaluate the pretrained model (skip training)')
	parser.add_argument('-opt',	 	dest="opt", 		default='sgd', 				help='Optimizer to use for training')
	parser.add_argument('-eps', 	 	dest="eps", 		default=0.00000001,  	type=float, 		help='Value of epsilon')
	parser.add_argument('-name', 	 	dest="name", 		default='pretrained_riedel',		help='Name of the run')
	parser.add_argument('-seed', 	 	dest="seed", 		default=1234, 		type=int,		help='Seed for randomization')
	parser.add_argument('-logdir',	 	dest="log_dir", 	default='./log/', 				help='Log directory')
	parser.add_argument('-config',	 	dest="config_dir", 	default='./config/', 				help='Config directory')
	parser.add_argument('-embed_loc',	dest="embed_loc", 	default='./glove/glove.6B.50d_word2vec.txt', 	help='Log directory')
	parser.add_argument('-embed_dim',	dest="embed_dim", 	default=50, type=int,				help='Dimension of embedding')
	parser.add_argument('-rel2alias_file', 	default='./side_info/relation_alias/riedel/relation_alias_from_wikidata_ppdb_extended.json', 	help='File containing relation alias information')
	parser.add_argument('-type2id_file',   	default='./side_info/entity_type/riedel/type_info.json', 					help='File containing entity type information')
	args = parser.parse_args()

	print('Loading RESIDE model.')
	set_gpu(args.gpu)

	for bag in open('./data/riedel_test_bags.json').readlines():
		bag	= json.loads(bag)
		if bag['rel'] == [0]: continue
		bag_list.append(bag)

	model  = RESIDE(args)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess   =  tf.Session(config=config)
	sess.run(tf.global_variables_initializer())

	# Restore previously saved model
	saver		= tf.train.Saver()
	save_dir	= os.path.join('checkpoints/', args.name)
	if not os.path.exists(save_dir):
		model.logger.info('Path {} doesnt exist.'.format(save_dir))
		sys.exit()
	save_path = os.path.join(save_dir, 'best_model')
	saver.restore(sess, save_path)

	print('Started server at port: {}'.format(args.port))
	app.static_folder = 'static'
	app.run(host="0.0.0.0", port = args.port, threaded = False)