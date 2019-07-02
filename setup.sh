# Variables
embed_path="./glove/glove.6B.50d_word2vec.txt"

# Check for GloVe Embeddings
if [ ! -f "$embed_path" ]
then
    	echo "Downloading GloVe Embeddings"
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip -d glove
	rm glove.6B.zip
fi
