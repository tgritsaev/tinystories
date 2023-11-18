# Tiny Stories
HSE Deep Learning course homework.

## Prerequisites

`torch, numpy, sentencepiece, wandb, tqdm`

## Download and create data

Download the TinySrories data
```shell
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
```
Create data in special format
```shell
python3 scripts/preprocess_data.py
```

## Train

Run `wandb login` in bash, if you have not used wandb before.

Then run
```shell
python3 train.py
```

## Project structure

* `scripts/` directory for preprocess pipelines.
* `loss.py` code for CrossEntropyLossWrapper.
* `model.py`: code for Transformer model and it's subparts. 
* `tinystories_dataset.py`: code for TinyStoriesDataset and it's funtions.
* `train.py`: code training and testing pipeline.

## Authors

* Timofei Gritsaev

## Assignment

Original task statement https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01.

You can see my wandb project and report https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev.
