import torch
import torch.nn as nn

from torch.utils.data import Dataset

class OASDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_len)
        return encoding['input_ids']
   
if __name__ == '__main__':
    from transformers import DataCollatorForLanguageModeling
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    tokenizer = BertTokenizer.from_pretrained('tokenizer/ProteinTokenizer')
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.5,
    )
    data = ['ACGACGACGACGAGC', 'CGGCGAGCGAAG', 'CGACGACGACAGCGACGACGAGCAGCAG']
    
    ds = OASDataset(data, tokenizer, max_len=512)
    loader = DataLoader(ds, batch_size=2, collate_fn=collator)

    for batch in loader:
        print(batch['input_ids'])
        print(batch['labels'])

        print(batch['input_ids'].shape)
        print(batch['labels'].shape)
        break