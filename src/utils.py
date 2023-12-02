import wandb
import sentencepiece as spm

TOKENIZER_PATH = "4096.model"
TOKENIZER = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)


# constants
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def text2ids(texts):
    return TOKENIZER.encode(texts)


def ids2text(ids):
    return TOKENIZER.decode(ids)


class WandbWriter:
    def __init__(self, wandb_project: str = None):
        print(f"wandb_project: {wandb_project}")
        if wandb_project:
            self.skip = False
            wandb.init(wandb_project)
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
