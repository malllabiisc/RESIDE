## RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information

Source code for [EMNLP 2018](http://emnlp2018.org) paper: [RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information](http://malllabiisc.github.io/publications/papers/reside_emnlp18.pdf).

![](https://github.com/malllabiisc/RESIDE/blob/master/overview.png)*Overview of RESIDE (proposed method): RESIDE first encodes each sentence in the bag by concatenating embeddings (denoted by ⊕) from Bi-GRU and Syntactic GCN for each token, followed by word attention.*
*Then, sentence embedding is concatenated with relation alias information, which comes from the Side Information Acquisition Section, before computing attention over sentences. Finally, bag representation with entity type information is fed to a softmax classifier. Please refer to paper for more details.* 

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use [Riedel NYT](http://iesl.cs.umass.edu/riedel/ecml/) and [Google IISc Distant Supervision (GIDS)](https://arxiv.org/pdf/1804.06987.pdf) dataset​ for evaluation.
- The processed version of the datasets can be downloaded from [here](https://drive.google.com/file/d/1brGCxXm2ofbF_0HP4myfBSHphGg4v6BS/view?usp=sharing). 

### Evalute pre-trained model:

- `reside.py` contains TensorFlow (1.x) based implementation of RESIDE (proposed method).
- Download the pretrained model's parameters from [here](https://drive.google.com/open?id=16yuV5SoxHEdAURTw5wrqYKR1cStQrzTw). 
- Execute `evaluate.sh` for comparing pretrained RESIDE model against baselines (plots Precision-Recall curve). 

### Training from scratch:
- For training RESIDE execute:
  ```shell
  python reside.py -data data/riedel_processed.pkl -name new_run
  ```