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
1. To install libraries run from the root directory
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
4. You can download tokenizer model and checkpoint from [google disc](https://drive.google.com/drive/folders/1wEMJGgeHT653O2UbM8LcjRau-W-BrUF4?usp=sharing) to reproduce only my final results.

## Example
The model generates simple stories for children:
Once upon a time, there was a little girl named Lily. She loved to play with her toys and eat yummy snacks. One day, her mommy took her to the park to play. While they were playing, Lily saw a big, scary dog. She was scared and started to cry. Her mommy hugged her and said, "Don't worry, we'll protect you." After a while, the dog stopped barking and went away. Lily was happy that she was safe and could play with her toys again. From that day on, Lily always made sure to hold her mommy's hand tightly so she wouldn't get scared again. Once upon a time, a little girl named Lily went to the park with her mommy. They saw a big building with lots of windows and windows. Lily thought it was very cool and wanted to go inside. Her mommy said, "Lily, we can't go inside. It's too dangerous." But Lily didn't listen and went inside anyway. Later that day, Lily's mommy gave her a bottle of poison to drink. Lily didn't like it and said, "Mommy, I don't like it." Her mommy said, "You need to take care of the plants. They will grow big and strong like you." Lily listened to her mommy and took care of the plants. She watered them every day and said, "Thank you, mommy. I love you."

## Train
1. To setup wandb, if you have not used wandb before, run `wandb login` in bash.
2. To train run
```shell
python3 train.py
```

## Test
```shell
python3 test.py
```
`test.py` contains two arguments:
* `"-c", "--config", default="configs/test.json", type=str, help="config file path (default: configs/test.json)"`
* `"-p", "--checkpoint-path", default="checkpoint.pth", type=str, help="checkpoint path (default: checkpoint.pth)"`
* `"-t", "--temperature", default=0.6, type=float, help="sampling temperature (default: 0.6)"`

## Wandb 
1. [Wandb project](https://wandb.ai/tgritsaev/tiny_stories_dl2/overview?workspace=user-tgritsaev).
2. [Wandb report](https://wandb.ai/tgritsaev/dl-2-tinystories/reports/bhw-dl-2-HSE-course-tinystories--Vmlldzo2MTUzNzk4).

## Authors
* Timofei Gritsaev

## Credits
Model implementation is took from [pytorch tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=language%20models).