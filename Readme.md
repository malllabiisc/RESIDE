## RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information

Source code for [EMNLP 2018](http://emnlp2018.org) paper: [RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information](http://malllabiisc.github.io/publications/papers/reside_emnlp18.pdf).

![](https://github.com/malllabiisc/RESIDE/blob/master/overview.png)Overview of *Overview of RESIDE (proposed method): RESIDE first encodes each sentence in the bag by concatenating embeddings (denoted by ⊕) from Bi-GRU and Syntactic GCN for each token, followed by word attention.*
*Then, sentence embedding is concatenated with relation alias information, which comes from the Side Information Acquisition Section, before computing attention over sentences. Finally, bag representation with entity type information is fed to a softmax classifier. Please refer to paper for more details.* 

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- ​

### Usage:

- After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

- `reside.py` contains TensorFlow (1.x) based implementation of the RESIDE (proposed method).

- To start training: 

  ```shell
  python reside.py -data data/riedel_processed.pkl -name test_run
  ```

  - `-name` is arbitrary name for the run.

### Citing:

```tex
@InProceedings{}
```
