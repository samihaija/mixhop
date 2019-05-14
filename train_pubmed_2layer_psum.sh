python mixhop_trainer.py --dataset_name=ind.pubmed --adj_pows=0:17:3,1:23:6,2:20:3 \
  --learn_rate=0.5 --lr_decrement_every=20 --early_stop_steps=200 \
  --input_dropout=0.5 --layer_dropout=0.9 --l2reg=5e-3 \
  --retrain
