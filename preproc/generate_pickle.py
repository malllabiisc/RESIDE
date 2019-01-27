import sys; 
sys.path.append('./')

from helper import *
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from orderedset import OrderedSet

parser = argparse.ArgumentParser(description='Main Preprocessing program')
parser.add_argument('-test', 	 dest="FULL", 		action='store_false')
parser.add_argument('-pos', 	 dest="MAX_POS", 	default=60,   	 	type=int, help='Max position to consider for positional embeddings')
parser.add_argument('-mvoc', 	 dest="MAX_VOCAB", 	default=150000,  	type=int, help='Maximum vocabulary to consider')
parser.add_argument('-maxw', 	 dest="MAX_WORDS", 	default=100, 	 	type=int)
parser.add_argument('-minw', 	 dest="MIN_WORDS", 	default=5, 	 	type=int)
parser.add_argument('-num', 	 dest="num_procs", 	default=40, 	 	type=int)
parser.add_argument('-thresh', 	 dest="thresh", 	default=0.65, 	 	type=float)
parser.add_argument('-nfinetype',dest='wFineType', 	action='store_false')
parser.add_argument('-metric',   default='cosine')
parser.add_argument('-data', 	 default='riedel')

# Change the below two arguments together
parser.add_argument('-embed',    dest="embed_loc", 	default='./glove/glove.6B.50d_word2vec.txt')
parser.add_argument('-embed_dim',default=50, 		type=int)

# Below arguments can be used for testing processing script (process a part of data instead of full)
parser.add_argument('-sample', 	 dest='FULL', 		action='store_false', 	 		help='To process the entire data or a sample of it')
parser.add_argument('-samp_size',dest='sample_size', 	default=200, 	 	type=int,	help='Sample size to use for testing processing script')
args = parser.parse_args()

print('Starting Data Pre-processing script...')
ent2type      = json.loads(open('./side_info/entity_type/{}/type_info.json'.format(args.data)).read())
rel2alias     = json.loads(open('./side_info/relation_alias/{}/relation_alias_from_wikidata_ppdb_extended.json'.format(args.data)).read())
rel2id        = json.loads(open('./preproc/{}_relation2id.json'.format(args.data)).read())
id2rel 	      = dict([(v, k) for k, v in rel2id.items()])
alias2rel     = ddict(set)
alias2id      = {}
embed_model   = gensim.models.KeyedVectors.load_word2vec_format(args.embed_loc, binary=False)

for rel, aliases in rel2alias.items():
	for alias in aliases:
		if alias in alias2id:
			alias2rel[alias2id[alias]].add(rel)
		else:
			alias2id[alias] = len(alias2id)
			alias2rel[alias2id[alias]].add(rel)

temp = sorted(alias2id.items(), key= lambda x: x[1])
temp.sort(key = lambda x:x[1])
alias_list, _ = zip(*temp)
alias_embed   = getPhr2vec(embed_model, alias_list, args.embed_dim)
id2alias      = dict([(v, k) for k, v in alias2id.items()])

data = {
	'train': [],
	'test':  []
}

def get_index(arr, ele):
	if ele in arr:  return arr.index(ele)
	else:       return -1

def read_file(file_path):
	temp = []

	with open(file_path) as f:
		for k, line in enumerate(f):
			bag   = json.loads(line.strip())

			wrds_list 	= []
			pos1_list 	= []
			pos2_list 	= []
			sub_pos_list   	= []
			obj_pos_list    = []
			dep_links_list 	= []
			phrase_list 	= []

			for sent in bag['sents']:

				if len(bag['sub']) > len(bag['obj']):
					sub_idx   	  = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['sub']]
					sub_start_off 	  = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0) for idx in sub_idx]
					if sub_start_off == []: sub_start_off = [m.start() for m in re.finditer(bag['sub'].replace('_', ' '), sent['rsent'].replace('_', ' '))]
					reserve_span      = [(start_off, start_off + len(bag['sub'])) for start_off in sub_start_off]

					obj_idx   	  = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['obj']]
					obj_start_off 	  = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0) for idx in obj_idx ]
					if obj_start_off == []: obj_start_off = [m.start() for m in re.finditer(bag['obj'].replace('_', ' '), sent['rsent'].replace('_', ' '))]
					obj_start_off 	  = [off for off in obj_start_off if all([off < spn[0] or off > spn[1] for spn in reserve_span])]
				else:
					obj_idx   	  = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['obj']]
					obj_start_off 	  = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0) for idx in obj_idx]
					if obj_start_off == []: obj_start_off = [m.start() for m in re.finditer(bag['obj'].replace('_', ' '), sent['rsent'].replace('_', ' '))]
					reserve_span 	  = [(start_off, start_off + len(bag['obj'])) for start_off in obj_start_off]

					sub_idx   	  = [i for i, e in enumerate(sent['rsent'].split()) if e == bag['sub']]
					sub_start_off 	  = [len(' '.join(sent['rsent'].split()[0: idx])) + (1 if idx != 0 else 0) for idx in sub_idx ]
					if sub_start_off == []: sub_start_off = [m.start() for m in re.finditer(bag['sub'].replace('_', ' '), sent['rsent'].replace('_', ' '))]
					sub_start_off 	  = [off for off in sub_start_off if all([off < spn[0] or off > spn[1] for spn in reserve_span])]

				sub_off = [(start_off, start_off + len(bag['sub']), 'sub') for start_off in sub_start_off]
				obj_off = [(start_off, start_off + len(bag['obj']), 'obj') for start_off in obj_start_off]

				if sub_off == [] or obj_off == [] or 'corenlp' not in sent: continue
				spans = [sub_off[0]] + [obj_off[0]]
				off_begin, off_end, _ = zip(*spans)
				
				tid_map, tid2wrd = ddict(dict), ddict(list)

				tok_idx 	 = 1
				sub_pos, obj_pos = None, None
				dep_links 	 = []

				for s_n, corenlp_sent in enumerate(sent['corenlp']['sentences']):				# Iterating over sentences

					i, tokens = 0, corenlp_sent['tokens']

					while i < len(tokens):
						if tokens[i]['characterOffsetBegin'] in off_begin:
							_, end_offset, identity = spans[off_begin.index(tokens[i]['characterOffsetBegin'])]

							if identity == 'sub': 	
								sub_pos    = tok_idx-1						# Indexing starts from 0 
								tok_list   = [tok['originalText'] for tok in tokens]
							else:			
								obj_pos = tok_idx-1
								tok_list   = [tok['originalText'] for tok in tokens]

							while i < len(tokens) and tokens[i]['characterOffsetEnd'] <= end_offset:
								tid_map[s_n][tokens[i]['index']] = tok_idx
								tid2wrd[tok_idx].append(tokens[i]['originalText'])
								i += 1

							tok_idx += 1
						else:
							tid_map[s_n][tokens[i]['index']] = tok_idx
							tid2wrd[tok_idx].append(tokens[i]['originalText'])

							i       += 1
							tok_idx += 1


				if sub_pos == None or obj_pos == None: 
					print('Skipped entry!!')
					print('{} | {} | {}'.format(bag['sub'], bag['obj'], sent['sent']))
					pdb.set_trace()
					continue

				wrds    = ['_'.join(e).lower() 	for e in tid2wrd.values()]
				pos1	= [i - sub_pos 		for i in range(tok_idx-1)]					# tok_id = (number of tokens + 1)
				pos2	= [i - obj_pos 		for i in range(tok_idx-1)]

				phrases = set()
				if sent['openie'] != None:
					for corenlp_sent in sent['openie']['sentences']:
						for openie in corenlp_sent['openie']:
							if openie['subject'].lower() == bag['sub'].replace('_', ' ') and openie['object'].lower() == bag['obj'].replace('_', ' '):
								phrases.add(openie['relation'])

				openie_phrases = phrases.copy()

				if abs(sub_pos - obj_pos) < 5:
					middle_phr = ' '.join(sent['rsent'].split()[min(sub_pos, obj_pos)+1: max(sub_pos, obj_pos)])
					phrases.add(middle_phr)
				else:   middle_phr = ''

				for s_n, corenlp_sent in enumerate(sent['corenlp']['sentences']):
					dep_edges = corenlp_sent['basicDependencies']
					for dep in dep_edges:
						if dep['governor'] == 0 or dep['dependent'] == 0: continue					# Ignore ROOT
						dep_links.append((tid_map[s_n][dep['governor']]-1, tid_map[s_n][dep['dependent']]-1, 0, 1))	# -1, because indexing starts from 0 


				right_nbd_phrase, left_nbd_phrase, mid_phrase = set(), set(), set()
				for edge in dep_links:
					if edge[0] == sub_pos or edge[0] == obj_pos:
						if edge[1] > min(sub_pos, obj_pos)  and edge[1] < max(sub_pos, obj_pos):
							mid_phrase.add(wrds[edge[1]])
						elif edge[1] < min(sub_pos, obj_pos):
							left_nbd_phrase.add(wrds[edge[1]])
						else:
							right_nbd_phrase.add(wrds[edge[1]])
							
					if edge[1] == sub_pos or edge[1] == obj_pos:
						if edge[0] > min(sub_pos, obj_pos)  and edge[0] < max(sub_pos, obj_pos):
							mid_phrase.add(wrds[edge[0]])
						elif edge[0] < min(sub_pos, obj_pos):
							left_nbd_phrase.add(wrds[edge[0]])
						else:
							right_nbd_phrase.add(wrds[edge[0]])

				left_nbd_phrase  = ' '.join(list(left_nbd_phrase  - {bag['sub'], bag['obj']}))
				right_nbd_phrase = ' '.join(list(right_nbd_phrase - {bag['sub'], bag['obj']}))
				mid_phrase 	 = ' '.join(list(mid_phrase))
				
				phrases.add(left_nbd_phrase)
				phrases.add(right_nbd_phrase)
				phrases.add(middle_phr)
				phrases.add(mid_phrase)

				wrds_list.append(wrds)
				pos1_list.append(pos1)
				pos2_list.append(pos2)
				sub_pos_list.append(sub_pos)
				obj_pos_list.append(obj_pos)
				dep_links_list.append(dep_links)
				phrase_list.append(list(phrases - {''}))

			temp.append({
				'sub':			bag['sub'],
				'obj':			bag['obj'],
				'rels':			bag['rel'],
				'phrase_list':		phrase_list,
				'sub_pos_list':		sub_pos_list,
				'obj_pos_list':		obj_pos_list,
				'wrds_list': 		wrds_list,
				'pos1_list': 		pos1_list,
				'pos2_list': 		pos2_list,
				'sub_type':		ent2type[bag['sub_id']],
				'obj_type':		ent2type[bag['obj_id']],
				'dep_links_list':	dep_links_list,
			})
				
			if k % 1000 == 0: print('Completed {}'.format(k))
			if not args.FULL and k > args.sample_size: break
	return temp

print('Reading train bags'); data['train'] = read_file( 'data/{}_train_bags.json'.format(args.data))
print('Reading test bags');  data['test']  = read_file( 'data/{}_test_bags.json'. format(args.data))

print('Bags processed: Train:{}, Test:{}'.format(len(data['train']), len(data['test'])))

"""*************************** REMOVE OUTLIERS **************************"""
del_cnt = 0
for dtype in ['train', 'test']:
	for i in range(len(data[dtype])-1, -1, -1):
		bag = data[dtype][i]
		
		for j in range(len(bag['wrds_list'])-1, -1, -1):
			data[dtype][i]['wrds_list'][j] 		= data[dtype][i]['wrds_list'][j][:args.MAX_WORDS]
			data[dtype][i]['pos1_list'][j] 		= data[dtype][i]['pos1_list'][j][:args.MAX_WORDS]
			data[dtype][i]['pos2_list'][j] 		= data[dtype][i]['pos2_list'][j][:args.MAX_WORDS]
			data[dtype][i]['dep_links_list'][j] 	= [e for e in data[dtype][i]['dep_links_list'][j] if e[0] < args.MAX_WORDS and e[1] < args.MAX_WORDS]
			if len(data[dtype][i]['dep_links_list'][j]) == 0: 
				del data[dtype][i]['dep_links_list'][j]			# Delete sentences with no dependency links

		if len(data[dtype][i]['wrds_list']) == 0 or len(data[dtype][i]['dep_links_list']) == 0:
			del data[dtype][i]
			del_cnt += 1
			continue

print('Bags deleted {}'.format(del_cnt))

"""*************************** GET PROBABLE RELATIONS **************************"""
def get_alias2rel(phr_list):
	phr_embed = getPhr2vec(embed_model, phr_list, args.embed_dim)
	dist      = cdist(phr_embed, alias_embed, metric=args.metric)
	rels = set()
	for i, cphr in enumerate(np.argmin(dist, 1)):
		if dist[i, cphr] < args.thresh: rels |= alias2rel[cphr]
	return [rel2id[r] for r in rels if r in rel2id]

def get_prob_rels(data):
	res_list = []
	for content in data:
		prob_rels = []
		for phr_list in content['phr_lists']:
			prob_rels.append(get_alias2rel(phr_list))

		content['prob_rels'] = prob_rels
		res_list.append(content)

	return res_list

train_mega_phr_list = []
for i, bag in enumerate(data['train']):
	train_mega_phr_list.append({
		'bag_index': i,
		'phr_lists': bag['phrase_list']
	})

chunks  = partition(train_mega_phr_list, args.num_procs)
results = mergeList(Parallel(n_jobs = args.num_procs)(delayed(get_prob_rels)(chunk) for chunk in chunks))
for res in results:
	data['train'][res['bag_index']]['prob_rels'] = res['prob_rels']
	if len(data['train'][res['bag_index']]['prob_rels']) != len(data['train'][res['bag_index']]['phrase_list']):
		pdb.set_trace()

test_mega_phr_list = []
for i, bag in enumerate(data['test']):
	test_mega_phr_list.append({
		'bag_index': i,
		'phr_lists': bag['phrase_list']
	})

chunks  = partition(test_mega_phr_list, args.num_procs)
results = mergeList(Parallel(n_jobs = args.num_procs)(delayed(get_prob_rels)(chunk) for chunk in chunks))
for res in results:
	data['test'][res['bag_index']]['prob_rels'] = res['prob_rels']
	if len(data['test'][res['bag_index']]['prob_rels']) != len(data['test'][res['bag_index']]['phrase_list']):
		pdb.set_trace()

"""*************************** FORM VOCABULARY **************************"""
voc_freq = ddict(int)
for bag in data['train']:
	for wrds in bag['wrds_list']:
		for wrd in wrds: voc_freq[wrd] += 1

freq 	 = list(voc_freq.items())
freq.sort(key = lambda x: x[1], reverse=True)
freq 	 = freq[:args.MAX_VOCAB]
vocab, _ = map(list, zip(*freq))

vocab.append('UNK')

"""*************************** WORD 2 ID MAPPING **************************"""
def getIdMap(vals, begin_idx=0):
	ele2id = {}
	for id, ele in enumerate(vals):
		ele2id[ele] = id + begin_idx
	return ele2id


voc2id     = getIdMap(vocab, 1)
id2voc 	   = dict([(v, k) for k,v in voc2id.items()])

type_vocab = OrderedSet(['NONE'] + list(set(mergeList(ent2type.values()))))
type2id    = getIdMap(type_vocab)

print('Chosen Vocabulary:\t{}'.format(len(vocab)))
print('Type Number:\t{}'.format(len(type2id)))

"""******************* CONVERTING DATA IN TENSOR FORM **********************"""

def getId(wrd, wrd2id, def_val='NONE'):
	if wrd in wrd2id: return wrd2id[wrd]
	else: 		  return wrd2id[def_val]

def posMap(pos):
	if   pos < -args.MAX_POS: return 0
	elif pos > args.MAX_POS:  return (args.MAX_POS + 1)*2
	else: 			  return pos + (args.MAX_POS+1)

def procData(data, split='train'):
	res_list = []

	for bag in data:
		res = {}												# Labels will be K - hot
		res['X'] 	 = [[getId(wrd, voc2id, 'UNK')  for wrd in wrds] for wrds in bag['wrds_list']]
		res['Pos1'] 	 = [[posMap(pos) 		for pos in pos1] for pos1 in bag['pos1_list']]
		res['Pos2'] 	 = [[posMap(pos) 		for pos in pos2] for pos2 in bag['pos2_list']]
		res['Y']    	 = bag['rels']
		res['SubType']   = [ getId(typ, type2id, 'NONE') for typ in bag['sub_type']]
		res['ObjType']   = [ getId(typ, type2id, 'NONE') for typ in bag['obj_type']]
		res['SubPos']    = bag['sub_pos_list']
		res['ObjPos']    = bag['obj_pos_list']
		res['ProbY']     = bag['prob_rels']
		res['DepEdges']  = bag['dep_links_list']

		if len(res['X']) != len(res['ProbY']): 
			print('Skipped One')
			continue

		res_list.append(res)

	return res_list

final_data = {
	'train':  	procData(data['train'], 'train'),
	'test':	  	procData(data['test'],  'test'),
	'voc2id': 	voc2id,
	'id2voc': 	id2voc,
	'type2id':	type2id,
	'max_pos':	(args.MAX_POS+1)*2 + 1
}

pickle.dump(final_data, open('{}_processed.pkl'.format(args.data), 'wb'))