python mixhop_trainer.py --dataset_name=ind.citeseer --adj_pows=0:20:6,1:20:6,2:20:6 \
  --learn_rate=0.25 --lr_decrement_every=40 --early_stop_steps=200 \
  --input_dropout=0.5 --layer_dropout=0.9 --l2reg=5e-2 \
  --retrain
