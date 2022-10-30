# ADL HW1

## Enviroment

```
pip install -r requirements.in
```

## Training
```
#intent cls
python train_intent.py

#slot tag
python train_slot.py
```

### parameter
- rnn: Model type
- hidden_size: Model hidden size
- num_layers: Number of layers
- dropout: Dropout rate
- lr: Learning rate
- early_stop: Epoch of early stopping


## Download

```
#download model and cache
bash download.sh
```

## Predict

```
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Kaggle result

- intent
  - Public score: 0.93111
- slot
  - Public score: 0.80750