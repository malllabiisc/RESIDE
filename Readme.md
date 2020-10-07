<h1 align="center">
  RESIDE
</h1>

<h4 align="center">Improving Distantly-Supervised Neural Relation Extraction using Side Information</h4>

<p align="center">
  <a href="https://2018.emnlp.org/"><img src="http://img.shields.io/badge/EMNLP-2018-4b44ce.svg"></a>
  <a href="https://arxiv.org/abs/1812.04361"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://vimeo.com/305199302"><img src="http://img.shields.io/badge/Video-Vimeo-green.svg"></a>
  <a href="https://shikhar-vashishth.github.io/assets/pdf/reside_supp.pdf"><img src="http://img.shields.io/badge/Supplementary-PDF-B31B1B.svg"></a>
  <a href="https://shikhar-vashishth.github.io/assets/pdf/reside_poster.pdf"><img src="http://img.shields.io/badge/Poster-PDF-9cf.svg"></a>
  <a href="https://shikhar-vashishth.github.io/assets/pdf/slides_reside.pdf"><img src="http://img.shields.io/badge/Slides-PDF-orange.svg"></a>
  <a href="https://github.com/malllabiisc/RESIDE/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
</p>

<h2 align="center">
  Overview of RESIDE
  <img align="center"  src="./images/overview.png" alt="...">
</h2>

RESIDE first encodes each sentence in the bag by concatenating embeddings (denoted by ⊕) from Bi-GRU and Syntactic GCN for each token, followed by word attention. Then, sentence embedding is concatenated with relation alias information, which comes from the Side Information Acquisition Section, before computing attention over sentences. Finally, bag representation with entity type information is fed to a softmax classifier. Please refer to paper for more details.

Also includes implementation of [PCNN](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf), [PCNN+ATT](https://www.aclweb.org/anthology/P16-1200), [CNN](https://www.aclweb.org/anthology/C14-1220), CNN+ATT, and [BGWA](https://arxiv.org/pdf/1804.06987.pdf) models.

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use [Riedel NYT](http://iesl.cs.umass.edu/riedel/ecml/) and [Google IISc Distant Supervision (GIDS)](https://arxiv.org/pdf/1804.06987.pdf) dataset​ for evaluation.

- Datasets in json list format with side information can be downloaded from here: [RiedelNYT](https://drive.google.com/open?id=1D7bZPvrSAbIPaFSG7ZswYQcPA3tmouCw) and [GIDS](https://drive.google.com/open?id=1gTNAbv8My2QDmP-OHLFtJFlzPDoCG4aI).  

- The processed version of the datasets can be downloaded from [RiedelNYT](https://drive.google.com/file/d/1UD86c_6O_NSBn2DYirk6ygaHy_fTL-hN/view?usp=sharing) and [GIDS](https://drive.google.com/file/d/1UMS4EmWv5SWXfaSl_ZC4DcT3dk3JyHeq/view?usp=sharing). The structure of the processed input data is as follows.

  ```java
  {
      "voc2id":   {"w1": 0, "w2": 1, ...},
      "type2id":  {"type1": 0, "type2": 1 ...},
      "rel2id":   {"NA": 0, "/location/neighborhood/neighborhood_of": 1, ...}
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
              "ProbY":    [[s1_rel_alias1, s1_rel_alias2, ...], [s2_rel_alias1, ... ], ...]
              "DepEdges": [[s1_dep_edges], [s2_dep_edges] ...]
          },
          {}, ...
      ],
      "test":  { same as "train"},
      "valid": { same as "train"},
  }
  ```

  * `voc2id` is the mapping of word to its id
  * `type2id` is the maping of entity type to its id.
  * `rel2id` is the mapping of relation to its id. 
  * `max_pos` is the maximum position to consider for positional embeddings.
  * Each entry of `train`, `test` and `valid` is a bag of sentences, where
    * `X` denotes the sentences in bag as the list of list of word indices.
    * `Y` is the relation expressed by the sentences in the bag.
    * `Pos1` and `Pos2` are position of each word in sentences wrt to target entity 1 and entity 2.
    * `SubPos` and `ObjPos` contains the position of the target entity 1 and entity 2 in each sentence.
    * `SubType` and `ObjType` contains the target entity 1 and entity 2 type information obtained from KG.
    * `ProbY` is the relation alias side information (refer paper) for the bag.
    * `DepEdges` is the edgelist of dependency parse for each sentence (required for GCN).

### Evaluate pretrained model:

- `reside.py` contains TensorFlow (1.x) based implementation of **RESIDE** (proposed method).
- Download the pretrained model's parameters from [RiedelNYT](https://drive.google.com/file/d/1CUk10FTncaaZspAoh8YkHTML3RJHfW7e/view?usp=sharing) and [GIDS](https://drive.google.com/file/d/1X5pKkL6eOkGXw39baq0n9noBXa--5EhE/view?usp=sharing) (put downloaded folders in `checkpoint` directory). 
- Execute `evaluate.sh` for comparing pretrained **RESIDE** model against baselines (plots Precision-Recall curve). 

### Side Information:

- **Entity Type** information for both the datasets is provided in `side_info/type_info.zip`. 
  * Entity type information can be used directly in the model.
- **Relation Alias Information** for both the datasets is provided in `side_info/relation_alias.zip`.
  * The preprocessing code for using relation alias information: `rel_alias_side_info.py`. 
  * Following figure summarizes the method:
  ![](https://github.com/malllabiisc/RESIDE/blob/master/images/relation_alias.png)

### Training from scratch:
- Execute `setup.sh` for downloading GloVe embeddings.
- For training **RESIDE** run:
  ```shell
  python reside.py -data data/riedel_processed.pkl -name new_run
  ```

* The above model needs to be further trained with SGD optimizer for few epochs to match the performance reported in the paper. For that execute

  ```shell
  python reside.py -name new_run -restore -opt sgd -lr 0.001 -l2 0.0 -epoch 4
  ```

* Finally, run `python plot_pr.py -name new_run` to get the plot.

### Baselines:

* The repository also includes code for [PCNN](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf), [PCNN+ATT](https://www.aclweb.org/anthology/P16-1200), [CNN](https://www.aclweb.org/anthology/C14-1220), CNN+ATT, [BGWA](https://arxiv.org/pdf/1804.06987.pdf) models.

* For training **PCNN+ATT**:

  ```shell
  python pcnnatt.py -data data/riedel_processed.pkl -name new_run -attn # remove -attn for PCNN
  ```

  

* Similarly for training **CNN+ATT**:

  ```shell
  python cnnatt.py -data data/riedel_processed.pkl -name new_run # remove -attn for CNN
  ```

* For training **BGWA**:

  ```shell
  python bgwa.py -data data/riedel_processed.pkl -name new_run
  ```

### Preprocessing a new dataset:

* `preproc` directory contains code for getting a new dataset in the required format (`riedel_processed.pkl`) for `reside.py`.
* Get the data in the same format as followed in [riedel_raw](https://drive.google.com/file/d/1D7bZPvrSAbIPaFSG7ZswYQcPA3tmouCw/view?usp=sharing) or [gids_raw](https://drive.google.com/open?id=1gTNAbv8My2QDmP-OHLFtJFlzPDoCG4aI) for `Riedel NYT` dataset.
* Finally, run the script `preprocess.sh`.  `make_bags.py` is used for generating bags from sentence. `generate_pickle.py` is for converting the data in the required pickle format.

### Running pretrained model on new samples:

- The code for running pretrained model on a sample is included in `online` directory.

- A [flask](http://flask.pocoo.org/) based server is also provided. Use `python online/server.py` to start the server.

  - [riedel_test_bags.json](https://drive.google.com/open?id=1tIczJKU5NrZJvR-XHUEh7IrFrvbS_aHn) and other required [files](https://drive.google.com/open?id=17UNttRDo14O_Zgfr6y9tvY57fc0BGEjw) can be downloaded from the provided links.

  ![](./images/demo.png)

### Citation:
Please cite the following paper if you use this code in your work.

```bibtex
@inproceedings{reside2018,
  author = 	"Vashishth, Shikhar and 
  		Joshi, Rishabh and
		Prayaga, Sai Suman and
		Bhattacharyya, Chiranjib and
		Talukdar, Partha",
  title = 	"{RESIDE}: Improving Distantly-Supervised Neural Relation Extraction using Side Information",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  month = 	oct # "-" # nov,
  address = 	"Brussels, Belgium",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1257--1266",
  url = 	"http://aclweb.org/anthology/D18-1157"
}
```

For any clarification, comments, or suggestions please create an issue or contact [Shikhar](http://shikhar-vashishth.github.io).
