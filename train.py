import json
import argparse
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader

from tinystories_dataset import TinyStoriesDataset, collate_fn, move_batch_to_device
from loss import CrossEntropyLossWrapper
from model import Transformer


def train_epoch(model, dataloader, iterations, optimizer, scheduler, loss_fn, device):
    model.train()
    loss_sum = 0.0
    for i, batch in tqdm(enumerate(dataloader)):
        move_batch_to_device(device, **batch)

        optimizer.zero_grad()
        outputs = model(**batch)
        batch.update(outputs)

        loss = loss_fn(**batch)
        loss.backward()

        optimizer.step()
        scheduler.step()

        loss_sum += loss.item()

        if i + 1 >= iterations:
            break
    return loss_sum / iterations


def test(model, dataloader, loss_fn, device):
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            move_batch_to_device(device, **batch)
            outputs = model(**batch)
            batch.update(outputs)
            loss = loss_fn(**batch)
            loss_sum += loss

    return loss_sum / len(dataloader)


def main(args):
    with open("config.json") as fin:
        config = json.load(fin)

    wandb.init(**config["wandb"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    train_dataset = TinyStoriesDataset(**config["data"]["train"])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    test_dataset = TinyStoriesDataset(**config["data"]["test"])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    model = Transformer(**config["model"])
    model.to(device)

    epochs = config["train"]["epochs"]
    iterations_per_epochs = config["train"]["iterations_per_epoch"]
    optimizer = torch.optim.Adam(model.parameters(), **config["optimizer"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * iterations_per_epochs)
    loss_fn = CrossEntropyLossWrapper(train_dataset.tokenizer.pad_id())

    for epoch in tqdm(range(epochs)):
        train_loss_avg = train_epoch(model, train_dataloader, iterations_per_epochs, optimizer, scheduler, loss_fn, device)
        test_loss_avg = test(model, test_dataloader, loss_fn, device)

        print(f"train_loss {train_loss_avg} test_loss {test_loss_avg}")
        wandb.log({"train_loss": train_loss_avg, "test_loss": test_loss_avg})

    torch.save(model.state_dict(), "checkpoint.pth")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
