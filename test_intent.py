import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from tqdm import tqdm

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    test_path = args.test_file
    test_data = json.loads(test_path.read_text())
    test_dataset = SeqClsDataset(test_data,vocab, intent2idx, args.max_len)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 100, False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.ckpt_path, map_location=device)
    print('loadind model')
    print(model)
    model.eval()
    print('predict test data')

    predict_list = []

    with torch.no_grad():
        for i, inputs in enumerate(test_dataloader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            _, predict = torch.max(outputs, 1) 
            predict_list.extend(predict.int().tolist())

    predict_label = [test_dataset.idx2label(predict) for predict in predict_list]
    id_list = [test_id["id"] for test_id in test_data]

    with open(args.pred_file, "w", encoding="utf-8") as file:
        file.write("id,intent\n")
        if len(predict_label) == len(id_list):
          for i in range(len(predict_label)):
            file.write(f'{id_list[i]},{predict_label[i]}\n')
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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
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
