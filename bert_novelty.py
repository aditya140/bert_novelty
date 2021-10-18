from argparse import ArgumentParser

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as PLF
from torch.nn import functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

import json
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader


class NoveltyDataset(torch.utils.data.Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'source'])
        sent2 = str(self.data.loc[index, 'target'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids
        
        
        

class NoveltyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '/datastores/dlnd/dlnd.jsonl', batch_size: int = 32, max_len:int=128, bert_model:str='albert-base-v2'):
        super().__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.bert_model = bert_model
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        with open(self.data_dir,'r') as f:
            data = f.readlines()
        dataset = [json.loads(line) for line in data]
        source=[i["source"] for i in dataset]
        target=[i["target_text"] for i in dataset]
        labels=[1 if i["DLA"]=='Novel' else 0 for i in dataset]

        dataset = pd.DataFrame({"source":source,"target":target,"label":labels})
        df_train,df_test = train_test_split(dataset,test_size=0.1)
        df_train,df_val = train_test_split(df_train,test_size=0.1)

        self.df_train = df_train.reset_index(drop=True)
        self.df_val = df_val.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        self.train_data = NoveltyDataset(self.df_train,maxlen = self.max_len, with_labels=True, bert_model=self.bert_model)
        self.val_data = NoveltyDataset(self.df_val,maxlen = self.max_len,with_labels=True, bert_model=self.bert_model)
        self.test_data = NoveltyDataset(self.df_test,maxlen = self.max_len, with_labels=True, bert_model=self.bert_model)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    
    
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
class DocumentPairClassifier(nn.Module):
    def __init__(self, bert_model="albert-base-v2", freeze_bert=False):
        super(DocumentPairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        if bert_model == "albert-base-v2":  # 12M parameters
            hidden_size = 768
        elif bert_model == "albert-large-v2":  # 18M parameters
            hidden_size = 1024
        elif bert_model == "albert-xlarge-v2":  # 60M parameters
            hidden_size = 2048
        elif bert_model == "albert-xxlarge-v2":  # 235M parameters
            hidden_size = 4096
        elif bert_model == "bert-base-uncased": # 110M parameters
            hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        cont_reps, pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.
        logits = self.cls_layer(self.dropout(pooler_output))
        return logits

    
import torchmetrics

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs#.detach().cpu().numpy()


class LitClassifier(pl.LightningModule):
    def __init__(self, bert_model='albert-base-v2',freeze_bert=False,learning_rate=2e-5,batch_size = 32):
        super().__init__()
        self.save_hyperparameters()
        self.model = DocumentPairClassifier()
        self.test_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        (seq, attn_masks, token_type_ids, labels) = batch
        y_hat = self.model(seq, attn_masks, token_type_ids)
        loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(-1), labels.float())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (seq, attn_masks, token_type_ids, labels) = batch
        y_hat = self.model(seq, attn_masks, token_type_ids)
        loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(-1), labels.float())
        probs = get_probs_from_logits(y_hat.squeeze(-1)).squeeze(-1)
        preds =(probs>=0.5).int()
        self.valid_acc(preds, labels)
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=False)
        self.log('valid_loss', loss, on_step=True)

    def test_step(self, batch, batch_idx):
        (seq, attn_masks, token_type_ids, labels) = batch
        y_hat = self.model(seq, attn_masks, token_type_ids)
        loss = nn.BCEWithLogitsLoss()(y_hat.squeeze(-1), labels.float())
        probs = get_probs_from_logits(y_hat.squeeze(-1)).squeeze(-1)
        preds =(probs>=0.5).int()
        self.test_acc(preds, labels)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=False)
        self.log('test_loss', loss)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-2)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--bert_model', type=str, default='albert-base-v2')
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--freeze_bert', type=bool, default=False)
        return parser


dm = NoveltyDataModule()
lm = LitClassifier()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = pl.Trainer(gpus=0, max_epochs=2)
trainer.fit(lm, dm)
results = trainer.test(lm,dm)
print(results)




def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--bert_model', type=str, default="albert-base-v2")
    parser.add_argument('--max_len', type=int, default=512)

    # add trainer args (gpus=x, precision=...)
    parser = pl.Trainer.add_argparse_args(parser)

    # add model args (batch_size hidden_dim, etc...), anything defined in add_model_specific_args
    parser = LitClassifier.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_dir = os.path.join(args.data_dir,'dlnd.jsonl')
    datamodule = NoveltyDataModule(data_dir=data_dir,bert_model = args.bert_model, max_len = args.max_len, batch_size=args.batch_size)


    # ------------
    # model
    # ------------
    model = LitClassifier(
        bert_model=args.bert_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True)
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=datamodule.test_dataloader())
    print(result)


if __name__ == '__main__':
    cli_main()
