import torch.optim as optim

import pytorch_lightning as pl 
import transformers


class AntiBERTy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        config = transformers.BertConfig(
            vocab_size=25,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=512,
        )
        self.bert = transformers.BertForMaskedLM(config)

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids, labels=labels)
    
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch['input_ids'], batch['labels']
        out = self(input_ids=input_ids, labels=labels)

        self.log_dict({'loss': out.loss}, prog_bar=True, on_step=True, on_epoch=True)
        return out.loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-5)

if __name__ == '__main__':
    from transformers import DataCollatorForLanguageModeling
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    from data import OASDataset

    tokenizer = BertTokenizer.from_pretrained('tokenizer/ProteinTokenizer')
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.5,
    )
    data = ['ACGACGACGACGAGC', 'CGGCGAGCGAAG', 'CGACGACGACAGCGACGACGAGCAGCAG']
    
    ds = OASDataset(data, tokenizer, max_len=512)
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)

    model = AntiBERTy()

    for batch in loader:
        print(batch['input_ids'])
        print(batch['labels'])

        out = model(batch['input_ids'], labels=batch['labels'])
        print(out.loss)
        