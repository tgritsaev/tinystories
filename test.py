import json
import argparse

import torch

from src.model import Transformer


def main(args):
    with open(args.config) as fin:
        config = json.load(fin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Transformer(**config["model"])
    model = model.to(device)
    model.eval()
    print("Write prefix, the model will continue.")
    with torch.no_grad():
        while prefix := input():
            print(f"prefix:\n{prefix}")
            print(f"generated:\n{model.inference(prefix, args.temperature)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/test.json", type=str, help="config file path (default: configs/test.json)")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="sampling temperature (default: 1.)")
    args = parser.parse_args()
    main(args)
