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
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-a', '--accelerator', default='gpu')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-p', '--mask-prob', type=float, default=0.15)
    return parser.parse_args()

def main():
    torch.set_float32_matmul_precision('high')  # Trade-off precision for speed.

    wandb_logger = WandbLogger(project='antiberty-pytorch', entity='dohlee')
    args = parse_argument()

    tokenizer = BertTokenizer.from_pretrained('tokenizer/ProteinTokenizer')
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mask_prob,
    )

    seqs = []
    for fp in os.listdir(args.input):
        with open(os.path.join(args.input, fp)) as f:
            seqs += f.read().splitlines()
    
    ds = OASDataset(seqs, tokenizer, max_len=512)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=4,
        shuffle=True,
    )

    model = AntiBERTy()
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        max_epochs=-1,
    )
    trainer.fit(model, loader)

if __name__ == '__main__':
    main()

