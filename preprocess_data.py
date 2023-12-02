import os
import glob
import argparse
from tqdm import tqdm
import json

import sentencepiece as spm
import numpy as np

from src.utils import PAD_ID, UNK_ID, BOS_ID, EOS_ID


def main(args):
    assert args.vocab_size > 0, "Vocab size must be positive"

    print(f"limit: {args.limit}")
    os.makedirs(args.output_dir, exist_ok=True)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))[: args.limit]

    for file_name in tqdm(input_files):
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
        with open(file_name) as fin:
            data = json.load(fin)

        with open(output_path, "w") as fout:
            for text in tqdm(data, "json -> txt"):
                fout.write(text["story"] + "\n")
    print(f"Files are in {args.output_dir}.")

    print("Training a vocab...")
    spm.SentencePieceTrainer.train(
        input=",".join(glob.glob(os.path.join(args.output_dir, "*.txt"))),
        model_prefix=str(args.vocab_size),
        vocab_size=args.vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=f"{args.vocab_size}.model")

    print("Preparing the dataset...")
    os.makedirs(args.output_dir, exist_ok=True)
    for i, file in enumerate(tqdm(input_files)):
        with open(file) as fin:
            data = json.load(fin)
        output_file_name = "val" if (i + 1 == len(input_files) and args.limit > 1) else "train"
        tokenized = []
        idx = 0
        idxs = []
        for text in tqdm(data, "json -> npy"):
            tokenized.append(tokenizer.encode(text["story"]).astype(np.int16))
            idxs.append(np.array([idx, idx + len(tokenized[-1])]))
            idx += len(tokenized[-1]) + 1
        np.save(os.path.join(args.output_dir, f"{output_file_name}.npy"), np.concatenate(tokenized))
        np.save(os.path.join(args.output_dir, f"{output_file_name}_idxs.npy"), np.stack(idxs))

    print(f"Dataset is saved in {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, default="TinyStories_all_data", help="input directory path")
    parser.add_argument("-o", "--output-dir", type=str, default="data", help="output directory path")
    parser.add_argument("-v", "--vocab-size", type=int, default=4096, help="vocabulary size")
    parser.add_argument("-l", "--limit", type=int, default=None, help="limit input_files from the TinyStories_all_data")
    args = parser.parse_args()
    main(args)
