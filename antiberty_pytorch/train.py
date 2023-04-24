import torch
import argparse
import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from antiberty_pytorch import AntiBERTy, OASDataset
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)

from pytorch_lightning.loggers import WandbLogger


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-a", "--accelerator", default="gpu")
    parser.add_argument("-b", "--batch-size", type=int, default=16)
    parser.add_argument("-p", "--mask-prob", type=float, default=0.15)
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision("high")  # Trade-off precision for speed.

    args = parse_argument()
    wandb_logger = WandbLogger(project="antiberty-pytorch", entity="dohlee")

    tokenizer = BertTokenizer.from_pretrained("tokenizer/ProteinTokenizer")
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mask_prob,
    )

    seqs = []
    for fp in os.listdir(args.input):
        with open(os.path.join(args.input, fp)) as f:
            seqs += f.read().splitlines()

    train_seqs, val_seqs = seqs[: int(len(seqs) * 0.99)], seqs[int(len(seqs) * 0.99) :]
    train_ds = OASDataset(train_seqs, tokenizer, max_len=512)
    val_ds = OASDataset(val_seqs, tokenizer, max_len=512)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=False,
    )

    model = AntiBERTy()
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=args.accelerator,
        devices=1,
        max_epochs=-1,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
