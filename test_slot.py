import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SlotDataset
from model import Slotmodel
from utils import Vocab
from tqdm import tqdm


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    test_path = args.test_file
    test_data = json.loads(test_path.read_text())
    test_dataset = SlotDataset(test_data,vocab, tag2idx, args.max_len)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 100, False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.ckpt_path, map_location=device)
    print('loadind model')
    print(model)
    model.eval()
    print('predict test data')

    predict_list = []

    with torch.no_grad():
      
        for inputs, text_len in test_dataloader:
          batch_pred_list = []
          inputs = inputs.to(device, dtype=torch.long)
          outputs = model(inputs)
          outputs = torch.transpose(outputs, 1, 2)

          _, predict = torch.max(outputs, 1) 

          
          for i in range(len(predict)):
            pred = predict[i][:text_len[i]]
            batch_pred_list.append(pred.tolist())
          predict_list.extend(batch_pred_list)

  
    predict_label = []
    for predict in predict_list:
      temp = []
      for token in predict:
        temp.append(test_dataset.idx2label(token))
      predict_label.append(temp)


    id_list = [test_id["id"] for test_id in test_data]

    with open(args.pred_file, "w", encoding="utf-8") as file:
        file.write("id,tags\n")
        if len(predict_label) == len(id_list):
          print("write predice result to csv")
          for i in range(len(predict_label)):
            tags = ' '.join(tag for tag in predict_label[i])
            file.write(f'{id_list[i]},{tags}\n')
    file.close()
    print('done')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
