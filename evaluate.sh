# Variables
embed_path="./glove/glove.6B.50d_word2vec.txt"

# Check for GloVe Embeddings
if [ ! -f "$embed_path" ]
then
    	echo "Downloading GloVe Embeddings"
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip -d glove
	python -m gensim.scripts.glove2word2vec -i glove/glove.6B.50d.txt -o glove/glove.6B.50d_word2vec.txt
	rm glove.6B.zip
fi

# Evaluate the pre-trained model
python reside.py -data data/riedel_processed.pkl -name pretrained_model -restore -only_eval

# Plot PR-curve
python plot_pr.py -name pretrained_model