# Tiny Stories

HSE Deep Learning course homework.
During this homework I implemented language model (transformer), which creates simple stories for children.s
See the original [task statement](https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01).

## Code organization

```shell
├── README.md             <- Top-level README.
├── requirements.txt      <- project requirements.
├── preprocess_data.py    <- preprocessing code.
├── train.py              <- train code.
├── test.py               <- test code.
│
└── src                   <- main code directory.
    ├── loss.py                     <- transformer loss  
    ├── model.py                    <- transformer model  
    ├── tinystories_dataset.py      <- TinyStories dataset, collate_fn 
    └── utils.py                    <- utils: constants, Tokenizer and its' functions, WandbWriter
```

## Installation

1. Run from the root directory
```shell
pip3 install -r requirements.txt
```
2. Download TinySrories data
```shell
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
tar -xvzf TinyStories_all_data.tar.gz TinyStories_all_data
```
3. Run to create the dataset
```shell
python3 preprocess_data.py -o output-dir -i input-dir -v vocab-size -l limit
```

## Train

1. To setup wandb, if you have not used wandb before, run `wandb login` in bash.
2. To train run
```shell
python3 train.py
```

## Test

Run
```shell
python3 test.py
```
`test.py` contains two arguments:
* 

## Wandb 

You can see my wandb project and report https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev.

## Authors

* Timofei Gritsaev