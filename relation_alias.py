from helper import *
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description='Relation alias side information extractor')
parser.add_argument('-embed_dim',  default=50, 		type=int, 			help='Embedding dimension')
parser.add_argument('-sim_thresh', default=0.65, 	type=float, 			help='Threshold for relation similarity')
parser.add_argument('-metric',     default='cosine',					help='Similarity Metric')
parser.add_argument('-embed_loc',  default='./glove/glove.6B.50d_word2vec.txt', 	help='Word embedding location')
args = parser.parse_args()

rel2alias = json.load(open('./side_info/relation_alias/relation_alias_from_wikidata_ppdb_extended.json'))
model 	  = gensim.models.KeyedVectors.load_word2vec_format(args.embed_loc, binary=False)

alias2rel = ddict(set)
alias2id  = {}
for rel, aliases in rel2alias.items():
	for alias in aliases:
		if alias in alias2id:					# Assinging a unique id to each alias
			alias2rel[alias2id[alias]].add(rel)
		else:
			alias2id[alias] = len(alias2id)
			alias2rel[alias2id[alias]].add(rel)

temp = sorted(alias2id.items(), key=lambda x: x[1])
temp.sort(key = lambda x:x[1])
alias_list, _ = zip(*temp)
alias_embed   = getPhr2vec(model, alias_list, args.embed_dim)		# Encoding relation aliases from KG

def get_probable_rel(rel_phrs):
	phr_embed = getPhr2vec(model, rel_phrs, args.embed_dim)	# Encoding given relation phrases
	dist      = cdist(phr_embed, alias_embed, metric=args.metric)	# Computing similarity between given phrases and relation aliases from KG
	rels = set()
	for i, cphr in enumerate(np.argmin(dist, 1)):
		if dist[i, cphr] < 0.65: 				# Checking for similarity threshold
			rels |= alias2rel[cphr]				# Adding closest relation in the probable relation set
	return rels

rel_phrases = ['executive of', 'chief of']				# Relation phrases extracted using using OpenIE and dependency parse
probable_rels = get_probable_rel(rel_phrases)				# Probable relations predicted for given relation phrases
print('Given relation phrases: {}'.format(rel_phrases))
print('Probable relations: {}'.format(probable_rels))