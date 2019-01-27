import sys; 
sys.path.append('./')
from helper import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('-data', default='riedel')
args = parser.parse_args()

rel2id	 = json.loads(open('./preproc/{}_relation2id.json'.format(args.data)).read())

print('Constructing training bags...')
train_data = ddict(lambda: {'rels': ddict(list)})
with open('./data/{}_train.json'.format(args.data)) as f:
	for i, line in enumerate(f):
		data = json.loads(line.strip())

		_id = '{}_{}'.format(data['sub'], data['obj'])
		train_data[_id]['sub_id']	= data['sub_id']
		train_data[_id]['obj_id'] 	= data['obj_id']
		train_data[_id]['sub'] 		= data['sub']
		train_data[_id]['obj'] 		= data['obj']

		train_data[_id]['rels'][rel2id.get(data['rel'], rel2id['NA'])].append({
				'sent': 	data['sent'],
				'corenlp':	data['corenlp'],
				'rsent': 	data['rsent'],
				'openie':	data['openie'],
		})

		if i+1 % 1000 == 0: 
			print('Completed {}, {}'.format(i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
			break;

print('Constructing test bags...')
test_data = ddict(lambda : {'sents': [], 'rels': set()})
with open('./data/{}_test.json'.format(args.data)) as f:
	for i, line in enumerate(f):
		data = json.loads(line.strip())

		_id = '{}_{}'.format(data['sub'], data['obj'])
		test_data[_id]['sub_id']	= data['sub_id']
		test_data[_id]['obj_id'] 	= data['obj_id']
		test_data[_id]['sub'] 		= data['sub']
		test_data[_id]['obj'] 		= data['obj']
		test_data[_id]['rels'].add(rel2id.get(data['rel'], rel2id['NA']))

		test_data[_id]['sents'].append({
				'sent': 	data['sent'],
				'corenlp':	data['corenlp'],
				'rsent': 	data['rsent'],
				'openie':	data['openie'],
		})

		if i+1 % 1000 == 0: 
			print('Completed {}, {}'.format(i, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))
			break

count = 0
with open('./data/{}_train_bags.json'.format(args.data), 'w') as f:
	for _id, data in train_data.items():
		for rel, sents in data['rels'].items():

			entry = {}
			entry['sub']   	= data['sub']
			entry['obj']   	= data['obj']
			entry['sub_id'] = data['sub_id']
			entry['obj_id'] = data['obj_id']
			entry['sents']  = sents
			entry['rel']  	= [rel]

			f.write(json.dumps(entry)+'\n')
			count += 1
			if count % 10000 == 0: print('Writing Completed {}, {}'.format(count, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))

count = 0
with open('./data/{}_test_bags.json'.format(args.data), 'w') as f:
	for _id, data in test_data.items():

		entry = {}
		entry['sub']   	= data['sub']
		entry['obj']   	= data['obj']
		entry['sub_id'] = data['sub_id']
		entry['obj_id'] = data['obj_id']
		entry['sents']  = data['sents']
		entry['rel']  	= list(data['rels'])

		f.write(json.dumps(entry)+'\n')
		count += 1
		if count % 10000 == 0: print('Writing Completed {}, {}'.format(count, time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")))