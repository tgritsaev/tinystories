# tinystories
HSE Deep Learning course homework.

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

```shell
python3 train.py
```

## Authors

* Timofei Gritsaev

## Assignment

Original task statement https://github.com/puhsu/dl-hse/tree/main/week06-transformers/bhw01
