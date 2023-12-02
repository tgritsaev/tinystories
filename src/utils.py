import glob
import wandb
import sentencepiece as spm

import torch

# constants
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class TokenizerWrapper:
    def __init__(self):
        model_path = glob.glob("*.model")[0]
        # vocab_path = os.path.splitext(model_path)[0] + ".vocab"
        self.spm = spm.SentencePieceProcessor(model_file=model_path)


TOKENIZER = TokenizerWrapper()


def text2ids(texts):
    return TOKENIZER.spm.encode(texts)


def ids2text(ids):
    if torch.is_tensor(ids):
        return TOKENIZER.spm.decode(ids.tolist())
    return TOKENIZER.spm.decode(ids)


class WandbWriter:
    def __init__(self, project: str = None, name: str = None):
        print(f"wandb project: {project}")
        if project:
            self.skip = False
            wandb.init(project=project, name=name)
        else:
            self.skip = True

    def log(self, msg):
        if not self.skip:
            wandb.log(msg)

    def log_table(self, table):
        if not self.skip:
            wandb.log({"train": wandb.Table(data=table, columns=["pred", "target"])})

    def finish(self):
        if not self.skip:
            wandb.finish()
