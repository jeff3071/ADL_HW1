from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
import torch.nn.functional as F

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional : bool,
        num_class: int,
        rnn: str
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.rnn = rnn

        if rnn == 'lstm':
          self.net = nn.LSTM(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional , batch_first= True)
        else:
          self.net = nn.GRU(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional , batch_first= True)

        self.classifier = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_class)
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        if self.rnn =='lstm':
          out, (hn, cn)  = self.net(inputs)
        else:
          out, hn = self.net(inputs)

        x = self.classifier(torch.cat((hn[-1],hn[-2]),-1))
        return x



class CNNLSTM(torch.nn.Module):
  def __init__(
    self,
    embeddings: torch.tensor,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    num_class: int,
  )-> None:
    super(CNNLSTM, self).__init__()
    self.embed = Embedding.from_pretrained(embeddings, freeze=False)
    self.hidden_size = hidden_size
    self.conv = nn.Sequential(
      nn.Conv1d(embeddings.size(1), embeddings.size(1), 5, 1, 2),
      nn.LeakyReLU(),
    )
    
    self.net = nn.LSTM(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional , batch_first= True)

    self.classifier = nn.Sequential(
      nn.Linear(self.hidden_size*2, self.hidden_size),
      nn.LeakyReLU(),
      nn.Dropout(dropout),
      nn.Linear(self.hidden_size, num_class)
    )
    
  def forward(self, batch) -> Dict[str, torch.Tensor]:
    inputs = self.embed(batch) #[batch_size, max_len, 300]

    x = inputs.permute(0, 2, 1)
    x = self.conv(x)
    x = x.permute(0, 2, 1)
    out, (hn, cn) = self.net(x)

    x = torch.cat((hn[-1],hn[-2]),-1)
    x = self.classifier(x)
    return x


class Slotmodel(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional : bool,
        num_class: int,
        max_len: int,
        rnn: str
    ) -> None:
        super(Slotmodel, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.rnn = rnn

        if rnn == 'cnn-lstm':
          self.conv = nn.Sequential(
            nn.Conv1d(embeddings.size(1), embeddings.size(1), 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
          )

        if rnn == 'gru':
          self.net = nn.GRU(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional , batch_first= True)
        else:
          self.net = nn.LSTM(embeddings.size(1), hidden_size, num_layers = num_layers,bidirectional = bidirectional, batch_first= True)

        self.fc_layers = nn.Sequential(
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size*2, num_class)
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        
        inputs = self.embed(batch)
        if self.rnn == 'cnn-lstm':
          inputs = inputs.permute(0, 2, 1)
          inputs = self.conv(inputs)
          inputs = inputs.permute(0, 2, 1)

        if self.rnn == 'gru':
          out, hn = self.net(inputs)
        else:
          out, (hn, cn)  = self.net(inputs)  # out: [batch_size, seq_len, 2*hidden_size]
        
        # print(out.size())
        x = self.fc_layers(out)

        return x