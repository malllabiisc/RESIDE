## RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information

Source code for [EMNLP 2018](http://emnlp2018.org) paper: [RESIDE: Improving Distantly-Supervised Neural Relation Extraction using Side Information](http://malllabiisc.github.io/publications/papers/reside_emnlp18.pdf).

![](https://github.com/malllabiisc/RESIDE/blob/master/images/overview.png)*Overview of RESIDE (proposed method): RESIDE first encodes each sentence in the bag by concatenating embeddings (denoted by ⊕) from Bi-GRU and Syntactic GCN for each token, followed by word attention.*
*Then, sentence embedding is concatenated with relation alias information, which comes from the Side Information Acquisition Section, before computing attention over sentences. Finally, bag representation with entity type information is fed to a softmax classifier. Please refer to paper for more details.* 

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use [Riedel NYT](http://iesl.cs.umass.edu/riedel/ecml/) and [Google IISc Distant Supervision (GIDS)](https://arxiv.org/pdf/1804.06987.pdf) dataset​ for evaluation.

- The processed version of the datasets can be downloaded from [here](https://drive.google.com/file/d/1brGCxXm2ofbF_0HP4myfBSHphGg4v6BS/view?usp=sharing). The structure of the processed input data is as follows.

  ```java
  {
      "voc2id":   {"w1": 0, "w2": 1, ...},
      "type2id":  {"type1": 0, "type2": 1 ...},
      "max_pos": 123,
      "train": [
          {
              "X":        [[s1_w1, s1_w2, ...], [s2_w1, s2_w2, ...], ...],
              "Y":        [bag_label],
              "Pos1":     [[s1_p1_1, sent1_p1_2, ...], [s2_p1_1, s2_p1_2, ...], ...],
              "Pos2":     [[s1_p2_1, sent1_p2_2, ...], [s2_p2_1, s2_p2_2, ...], ...],
              "SubPos":   [s1_sub, s2_sub, ...],
              "ObjPos":   [s1_obj, s2_obj, ...],
              "SubType":  [s1_subType, s2_subType, ...],
              "ObjType":  [s1_objType, s2_objType, ...],
              "ProbY":    [rel_alias1, rel_alias2, ...]
              "DepEdges": [[s1_dep_edges], [s2_dep_edges] ...]
          },
          {}, ...
      ],
      "test":  { same as "train"},
      "valid": { same as "train"},
  }
  ```

  * `voc2id` is the mapping of words to their unique identifier
  * `type2id` is the maping of entity type to their unique identifier.
  * `max_pos` is the maximum position to consider for positional embeddings.
  * Each entry of `train`, `test` and `valid` is a bag of sentences, where
    * `X` denotes the sentences in bag as the list of list of word indices.
    * `Y` is the relation expressed by the sentences in the bag.
    * `Pos1` and `Pos2` are position of each word in sentences wrt to target entity 1 and entity 2.
    * `SubPos` and `ObjPos` contains the position of the target entity 1 and entity 2 in each sentence.
    * `SubType` and `ObjType` contains the target entity 1 and entity 2 type information obatined from KG.
    * `ProbY` is the relation alias side information (refer paper) for the bag.
    * `DepEdges` is the edgelist of dependency parse for each sentence (required for GCN).

### Evaluate pretrained model:

- `reside.py` contains TensorFlow (1.x) based implementation of RESIDE (proposed method).
- Download the pretrained model's parameters from [here](https://drive.google.com/open?id=16yuV5SoxHEdAURTw5wrqYKR1cStQrzTw). 
- Execute `evaluate.sh` for comparing pretrained RESIDE model against baselines (plots Precision-Recall curve). 

### Side Information:

* **Entity Type** information provided in `side_info/type_info.zip`. 

  * Entity type information can be used directly in the model.

* **Relation Alias Information** for provided in `side_info/relation_alias.zip`.

  * The preprocessing code for using relation alias information is provided in `rel_alias_side_info.py`. Following figure summarizes the method.

  ![](https://github.com/malllabiisc/RESIDE/blob/master/images/relation_alias.png)

### Training from scratch:
- For training RESIDE execute:
  ```shell
  python reside.py -data data/riedel_processed.pkl -name new_run
  ```



For any clarification, please contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).
