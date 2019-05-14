python mixhop_trainer.py --dataset_name=ind.cora --adj_pows=0:24:0,1:18:7,2:18:7 \
  --learn_rate=1 --lr_decrement_every=40 --early_stop_steps=200 --input_dropout=0.5 \
  --layer_dropout=0.9 --l2reg=5e-3 --retrain
