import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import lr_scheduler
from tqdm import trange
from tqdm import tqdm

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from model import CNNLSTM

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last = True) for x in SPLITS}
    dataset_sizes = {x: len(datasets[x]) for x in SPLITS}

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)

    if args.rnn == 'lstm' or args.rnn == 'gru':
      model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets["train"].num_classes, args.rnn)
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
      model = CNNLSTM(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, datasets["train"].num_classes)
      optimizer = torch.optim.Adam(model.parameters(), lr=5*args.lr, weight_decay=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('using device: {}'.format(device))
    model = model.to(device)
    print(model)
    # TODO: init optimizer
    
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    model.train()
    best_acc = 0
    
    early_stop = args.early_stop
    early_stop_cnt = 0

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device, dtype=torch.long) 
            labels = labels.to(device, dtype=torch.long) 
            
            optimizer.zero_grad() 
            outputs = model(inputs)

            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        scheduler.step()
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}%'.format(train_loss/len(dataloaders['train']), train_acc/(dataset_sizes['train'])*100))
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
          for inputs, labels in dataloaders['eval']:
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()

          print("Valid | Loss:{:.5f} Acc: {:.3f}%".format(val_loss/len(dataloaders['eval']), val_acc/(dataset_sizes['eval'])*100))

          if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
            print('saving model with acc {:.3f}%'.format(val_acc/(dataset_sizes['eval'])*100))
            early_stop_cnt = 0
          else:
            early_stop_cnt += 1

          if early_stop_cnt > early_stop:
            break
        model.train()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--rnn", type=str, help="lstm, gru, cnn-lstm", default="lstm")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

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


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

