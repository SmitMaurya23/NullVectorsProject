
## Env Setting
```
git clone https://github.com/SmitMaurya23/NullVectorsProject.git
cd <FOLDER NAME YOU WANT TO RUN>
conda create -n [env name] pip
conda activate [env name]
pip install -r requirements.txt
```

## Dataset
Download [dataset of 2018 Amazon Review dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for the experiment. Should download metadata and reviews files and place them into data/amazon direcotory.

```
cd data/amazon
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz  # download review dataset
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz  # download metadata
gzip -d meta_Movies_and_TV.json.gz
```
  
## Pre-train CF-RecSys (SASRec)
```
cd pre_train/sasrec
python main.py --device=cuda --dataset Movies_and_TV
```

## A-LLMRec Train
- train stage1
```
cd ../../
python main.py --pretrain_stage1 --rec_pre_trained_data Movies_and_TV
```

- train stage2
```
python main.py --pretrain_stage2 --rec_pre_trained_data Movies_and_TV
```

To run with multi-gpu setting, assign devices with CUDA_VISIBLE_DEVICES command and add '--multi_gpu' argument.
- ex) CUDA_VISIBLE_DEVICES = 0,1 python main.py ... --multi_gpu
  


## Evaluation
Inference stage generates "recommendation_output.txt" file and write the recommendation result generated from the LLMs into the file. To evaluate the result, run the eval.py file.

```
python main.py --inference --rec_pre_trained_data Movies_and_TV
python eval.py
```
