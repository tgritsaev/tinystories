# Tiny Stories
HSE Deep Learning course homework, see the [statement](https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01).

## Installation

1. Run
```shell
pip3 install -r requirements.txt
```
2. Download the TinySrories data
```shell
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
```
3. Run to create the dataset
```shell
python3 preprocess_data.py -o output-dir -i input-dir -v vocab-size -l limit
```

## Train

1. Run `wandb login` in bash, if you have not used wandb before.
2. Run
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

## Wandb 

You can see my wandb project and report https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev.
