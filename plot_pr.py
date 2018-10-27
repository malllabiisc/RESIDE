import numpy as np, argparse, pickle
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def loadData(path):
	preds 	   	= pickle.load(open(path, 'rb'))
	y_hot 	   	= np.array(preds['y_hot'])
	logit_list 	= np.array(preds['logit_list'])
	y_hot_new       = np.reshape(np.array([x[1:] for x in y_hot]),      (-1))
	logit_list_new  = np.reshape(np.array([x[1:] for x in logit_list]), (-1))
	return y_hot_new, logit_list_new

def plotPR():
	y_true, y_scores 	   = loadData('./results/{}/precision_recall.pkl'.format(args.name))
	precision,recall,threshold = precision_recall_curve(y_true,y_scores)
	area_under 	   	   = average_precision_score(y_true, y_scores)

	plt.plot(recall[:], precision[:], label='RESIDE', color ='red', lw=1, marker = 'o', markevery = 0.1, ms = 6)

	base_list = ['BGWA', 'PCNN+ATT', 'PCNN', 'MIMLRE', 'MultiR', 'Mintz']
	color     = ['purple', 'darkorange', 'green', 'xkcd:azure', 'orchid', 'cornflowerblue']
	marker	  = ['d', 's', '^', '*', 'v', 'x', 'h']
	plt.ylim([0.3, 1.0])
	plt.xlim([0.0, 0.45])

	for i, baseline in enumerate(base_list):
		precision = np.load(args.baselines + baseline + '/precision.npy')
		recall    = np.load(args.baselines + baseline + '/recall.npy')
		plt.plot(recall, precision, color = color[i], label = baseline, lw=1, marker = marker[i], markevery = 0.1, ms = 6)

	plt.xlabel('Recall',    fontsize = 14)
	plt.ylabel('Precision', fontsize = 14)
	plt.legend(loc="upper right", prop = {'size' : 12})
	plt.grid(True)
	plt.tight_layout()
	plt.show()

	plot_path = './results/{}/plot_pr.pdf'.format(args.name)
	plt.savefig(plot_path)
	print('Precision-Recall plot saved at: {}'.format(plot_path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-name', 	  default='pretrained_model')
	parser.add_argument('-baselines', default='./baselines_pr/')
	args = parser.parse_args()
	plotPR()