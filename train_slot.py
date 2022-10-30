import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import lr_scheduler
from tqdm import trange
from tqdm import tqdm

from dataset import SlotDataset
from utils import Vocab

import numpy as np
import random

from model import Slotmodel

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    datasets: Dict[str, SlotDataset] = {
      split: SlotDataset(split_data, vocab, tag2idx, args.max_len)
      for split, split_data in data.items()
    }

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True) for x in SPLITS}
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device: {}'.format(device))

    model = Slotmodel(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets["train"].num_classes, args.max_len, args.rnn)
    model.to(device)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=9) #ignore pad 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    model.train()
    best_acc = 0
    
    early_stop = args.early_stop
    early_stop_cnt = 0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_joint_correct, val_joint_correct = 0.0, 0.0
        train_token_correct, val_token_correct = 0.0, 0.0
        train_loss, val_loss = 0.0, 0.0
        train_joint_total, val_joint_total = 0, 0
        train_token_total, val_token_total = 0, 0

        for inputs, labels, text_len in dataloaders['train']:
            inputs = inputs.to(device, dtype=torch.long) 
            labels = labels.to(device, dtype=torch.long) 
            
            optimizer.zero_grad() 
            outputs = model(inputs)
            outputs = torch.transpose(outputs, 1, 2) #[batch_size, seq_len, num_class]
            
            loss = cal_loss(criterion, outputs, labels, model)
            # loss = criterion(outputs, labels)

            loss.backward() 
            optimizer.step()

            train_loss += loss.item()
            _, train_pred = torch.max(outputs, 1)
            

            for i in range(len(train_pred)):
              pred = train_pred[i][:text_len[i]]
              gt = labels[i][:text_len[i]]
              if pred.equal(gt):
                train_joint_correct += 1
              train_token_correct += sum(pred.eq(gt).view(-1)).item()
              train_token_total+=text_len[i]
              train_joint_total+=1

        print('\nTrain | Loss:{:.5f} Joint Acc: {:.3f}%  Token Acc: {:.3f}%'.format(train_loss/len(dataloaders['train']), \
         train_joint_correct/(train_joint_total)*100, train_token_correct/train_token_total*100))

        scheduler.step()
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        seq_pred = []
        seq_gt = []

        with torch.no_grad():
          for inputs, labels, text_len in dataloaders['eval']:
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            outputs = torch.transpose(outputs, 1, 2)
            
            loss = cal_loss(criterion, outputs, labels, model)
            # loss = criterion(outputs,labels)

            _, val_pred = torch.max(outputs, 1)
            val_loss += loss.item()

            for i in range(len(val_pred)):
              pred = val_pred[i][:text_len[i]]
              gt = labels[i][:text_len[i]]
              seq_pred.append(pred.tolist())
              seq_gt.append(gt.tolist())
              if pred.equal(gt):
                val_joint_correct += 1
              val_joint_total+=1
              val_token_correct += sum(pred.eq(gt).view(-1)).item()
              val_token_total+=text_len[i]


          print("Valid | Loss:{:.5f} Joint Acc: {:.3f}%  Token Acc: {:.3f}%".format(val_loss/len(dataloaders['eval']),\
          val_joint_correct/val_joint_total*100, val_token_correct/val_token_total*100))

          if val_joint_correct > best_acc:
            best_acc = val_joint_correct
            torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
            print('saving model with acc {:.3f}%'.format(val_joint_correct/(val_joint_total)*100))
            early_stop_cnt = 0
            seq_gt = totags(seq_gt, datasets['eval'])
            seq_pred = totags(seq_pred, datasets['eval'])
            print(classification_report(seq_gt, seq_pred, mode='strict', scheme=IOB2))
          else:
            early_stop_cnt += 1

          if early_stop_cnt > early_stop:
            break
        model.train()

def totags(seq, datasets):
  tags = []
  for i in seq:
    sentense = []
    for j in i:
      sentense.append(datasets.idx2label(j))
    tags.append(sentense)
  return tags


def cal_loss(criterion, pred, target, model):
  regularization_loss = 0
  for param in model.parameters():
      regularization_loss += torch.sum(param ** 2)
  return criterion(pred,target)+1e-6*regularization_loss



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--rnn", type=str, help="lstm, gru, cnn-lstm", default="gru")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--early_stop", type=int, default=10)
    args = parser.parse_args()
    return args

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    same_seeds(2) #固定參數 避免無法重現(slot比較有這個問題)
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)