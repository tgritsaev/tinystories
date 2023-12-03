import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.tinystories_dataset import TinyStoriesDataset, collate_fn
from src.loss import CrossEntropyLossWrapper
from src.model import Transformer
from src.utils import WandbWriter


def move_batch_to_device(batch, device):
    for key in ["src"]:
        batch[key] = batch[key].to(device)


def train_epoch(model, dataloader, iterations, optimizer, lr_scheduler, loss_fn, wandb_writer, device):
    print("in train_epoch")
    model.train()
    loss_sum = 0.0
    data_iter = iter(dataloader)
    for _ in range(iterations):
        try:
            batch = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(dataloader)
            batch = next(data_iter)
        move_batch_to_device(batch, device)

        optimizer.zero_grad()
        # with torch.autocast(device_type=device.type, dtype=torch.float16):
        outputs = model(batch["src"][:, :-1])
        batch.update(outputs)
        loss = loss_fn(**batch)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        wandb_writer.log({"train loss": loss.item(), "learning rate": lr_scheduler.get_last_lr()[0]})

        loss_sum += loss.item()

    return loss_sum / iterations


def test(model, dataloader, loss_fn, device):
    print("in evaluation")
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(batch["src"][:, :-1])
                batch.update(outputs)
                loss = loss_fn(**batch)
            loss_sum += loss

    return loss_sum / len(dataloader)


def main(args):
    save_dir = "saved/" + datetime.now().strftime("%d-%m-%Y_%H-%M")
    os.makedirs(save_dir, exist_ok=True)

    with open(args.config) as fin:
        config = json.load(fin)

    wandb_writer = WandbWriter(config["wandb_project"], name=args.wandb_run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_dataset = TinyStoriesDataset("train", **config["data"]["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    val_dataset = TinyStoriesDataset("val", **config["data"]["val"])
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = Transformer(**config["model"])
    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable parameters: {}".format(params))

    epochs = config["train"]["epochs"]
    iterations = config["train"]["iterations"]
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config["optimizer"]["lr"],
        epochs * iterations,
        pct_start=0.2,
    )
    loss_fn = CrossEntropyLossWrapper()

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_epoch(model, train_dataloader, iterations, optimizer, lr_scheduler, loss_fn, wandb_writer, device)
        val_loss = test(model, val_dataloader, loss_fn, device)

        wandb_writer.log({"val loss": val_loss})

        print(f"----- epoch: {epoch} -----")
        print(f"train loss:\t{train_loss:.4f}\nval loss:\t{val_loss:.4f}\nlearning rate:\t{lr_scheduler.get_last_lr()[0]:.8f}")
        print(f"generated example:\n{model.inference()}")
        print(f"--------------------------")

        if epoch % config["train"]["save_period"] == 0:
            torch.save(model.state_dict(), f"{save_dir}/checkpoint-{epoch}.pth")
    wandb_writer.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/train.json", type=str, help="config file path (default: configs/train.json)")
    parser.add_argument("-w", "--wandb-run-name", default=None, type=str, help="wandb run name (default: None)")
    args = parser.parse_args()
    main(args)
