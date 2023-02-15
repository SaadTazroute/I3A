# Table Of Contents
-  [Introduction](#introduction)
-  [Requirements](#requirements)
-  [Datasets](#datasets)
-  [Training model](#training-model)
-  [Testing model](#testing-model)
-  [Testing model with SemEval scripts](#testing-model-with-semeval-scripts)
-  [Project structure](#project-structure)
-  [Pretrained RoBERTa](#pretrained-roberta)
-  [Credits](#credits)

## Introduction
Scope of this project is to use the [RoBERTa](https://arxiv.org/abs/1907.11692) model to determine interpretable semantic textual similarity (iSTS) between two sentences. Given two sentences of text, s1 and s2, the STS systems compute how similar s1 and s2 are, returning a similarity score. Although the score is useful for many tasks, it does not allow to know which parts of the sentences are equivalent in meaning (or very close in meaning) and which not. The aim of interpretable STS is to explore whether systems are able to explain WHY they think the two sentences are related / unrelated, adding an explanatory layer to the similarity score. The explanatory layer consists of an alignment of chunks across the two sentences, where alignments are annotated with a similarity score and a relation label. Task is inspired by [SemEval](https://alt.qcri.org/semeval2020/) competition.


## Requirements
```bash
pip install -r requirements.txt
```
## Datasets
All needed datasets are in `data/datasets` folder. Data has been downloaded from 2015 and 2016 SemEval competition [site](http://ixa2.si.ehu.eus/stswiki/index.php/Main_Page#Interpretable_STS) and processed. If needed datasets can be reproduced with following commands:
```bash
./download_semeval_data.sh
python tools/create_csv.py
```
# Training model 
To train model with default parameters run following command:
```bash
python tools/train_net.py
```
To change any parametr command can be run with additional arguments, for example to set `max_epochs` to 100 and `learning rate` to 0.0002 run following command:
```bash
python tools/train_net.py SOLVER.MAX_EPOCHS 100 SOLVER.BASE_LR 0.0002
```
All the available parameters are defined in `net_config/defaults.py` file

Parameters can also be loaded from file with `--config_file`:
```bash
python tools/train_net.py --config_file configs/roberta_config.yml
```

# Testing model 
To test model path to model weights has to be provided:
```bash
python .\tools\test_net.py TEST.WEIGHT  "output/29052021142858_model.pt"
```
Changing other parameters works just like in [training](#training-model).

**NOTE**: There are 4 test datasets:
- images
- headline
- answers-students
- all above combined (*default*)



At default 4 metrics are calculated:
- F1 score for similarity values
- F1 score for relation labels
- Pearson correlation for similarity values
- Pearson correlation for relation labels

For model trained with default parameters results were following:

Metric | Train set score | Test set score
---|---|---
`F1 score for similarity values` | 0.773 | 0.724
`F1 score for relation labels` | 0.933 | 0.742
`Pearson correlation for similarity values` | 0.887 | 0.811
`Pearson correlation for relation labels` | 0.889 | 0.725 

Trained model that achived these results is available to download under this [link](https://drive.google.com/file/d/1-2sRnEUoQsPidAC9jvc2ZJdRc4XNbRmc/view?usp=sharing).

# Testing model with SemEval scripts
There are 3 perl scripts created by competition organizers (dowloaded and saved in `tests` folder):
- evalF1_no_penalty.pl
- evalF1_penalty.pl
- wellformed.pl

To use them, output files need to be in specific format. To create files with gold standard file structure use following command:
```bash
python .\tools\create_wa.py TEST.WEIGHT  "output/29052021142858_model.pt" DATASETS.TEST_WA "data/datasets/STSint.testinput.answers-students.wa"
```
This will create the .wa file with predictions (keeping the gold standard .wa file structure) in `output` directory. To evalute this file perl interpreter has to be installed on the machine. 
```bash
perl .\evalF1_penalty.pl data/datasets/STSint.testinput.answers-students.wa output/STSint.testinput.answers-student_predicted.wa --debug=0
```

# Project structure

```
├──  net_config
│    └── defaults.py  - here's the default config file.
│
│
├──  configs  
│    └── roberta_config.yml  - here's the specific config file for specific model or dataset.
│ 
│
├──  data  
│    └── datasets       - here's the datasets folder that is responsible for all data handling.
│    └── build.py       - here's the file to make dataloader.
│    
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── modeling            - this folder contains roberta modeel.
│   └── roberta_ists.py
│
│
├── solver             - this folder contains optimizer.
│   └── build.py
│   
│ 
├──  tools            - here's the train/test model functionality.
│    └── train_net.py     - this file is responsible for the whole training pipeline.
|    └── test_net.py      - this file is responsible for the whole testing pipeline.
|    └── create_cvs.py    - this file creates .csv files.
|    └── create_wa.py     - this file creates .wa files.
|
└── tests            - here are SemEval files to test model performance
│    └── evalF1_no_penalty.pl
|    └── evalF1_penalty.pl
|    └── wellformed.pl
|
|
└── utils
│    └── logger.py

```

# Pretrained RoBERTa

Base RoBERTa model can be replaced with pretrained one before the proper training. Follow the instructions under this [link](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md). To fine-tune model on STS-B GLUE task modify command from 3):
```bash
TOTAL_NUM_UPDATES=3598  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=214      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=1
MAX_SENTENCES=16        # Batch size.
ROBERTA_PATH=/content/drive/MyDrive/NLP/fairseq/roberta.large/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train STS-B-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --regression-target --best-checkpoint-metric loss \
    --find-unused-parameters;
```

# Credits
Repo template - L1aoXingyu/Deep-Learning-Project-Template



