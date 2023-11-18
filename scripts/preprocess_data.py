import os
import glob
import argparse
from tqdm import tqdm
import json

import sentencepiece as spm
import numpy as np


def main(args):
    assert args.vocab_size > 0, "Vocab size must be positive"

    print(f"limit: {args.limit}")
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))[: args.limit]
    for i, file in enumerate(tqdm(files)):
        with open(file) as fin:
            item_list = json.load(fin)
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file))[0]}.txt")
            with open(output_path, "w") as fout:
                fout.write("\n".join([item["story"] for item in item_list]))
    print(f"Files are in {output_path}")

    print("Train the vocab...")
    spm.SentencePieceTrainer.train(
        input=",".join(glob.glob(os.path.join(args.output_dir, "*.txt"))),
        model_prefix=str(args.vocab_size),
        vocab_size=args.vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=f"{args.vocab_size}.model")

    print("Preparing a dataset...")
    os.makedirs(args.output_dir, exist_ok=True)
    item_index = 0
    for file in tqdm(files):
        with open(file) as fin:
            item_list = json.load(fin)
            for item in item_list:
                tokenized_story = tokenizer.encode(item["story"])
                np.save(os.path.join(args.output_dir, f"{item_index}.npy"), tokenized_story)
                item_index += 1
    print(f"Dataset is saved in {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", type=str, default="data", help="output directory path")
    parser.add_argument("-i", "--input_dir", type=str, default="TinyStories_all_data", help="input directory path")
    parser.add_argument("-v", "--vocab_size", type=int, default=32768, help="vocabulary size")
    parser.add_argument("-l", "--limit", type=int, default=None, help="limit files from TinyStories_all_data number")
    args = parser.parse_args()
    main(args)
