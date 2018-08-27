# Download Glove Embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove
python -m gensim.scripts.glove2word2vec -i glove/glove.6B.300d.txt -o glove/glove.6B.300d_word2vec.txt
python -m gensim.scripts.glove2word2vec -i glove/glove.6B.100d.txt -o glove/glove.6B.100d_word2vec.txt
python -m gensim.scripts.glove2word2vec -i glove/glove.6B.200d.txt -o glove/glove.6B.200d_word2vec.txt
python -m gensim.scripts.glove2word2vec -i glove/glove.6B.50d.txt -o glove/glove.6B.50d_word2vec.txt
rm glove.6B.zip

# Create log directory
#mkdir log
