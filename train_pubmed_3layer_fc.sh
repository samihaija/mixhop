python mixhop_trainer.py \
  --architecture=architectures/pubmed.json \
  --learn_rate=0.5 --lr_decrement_every=20 --early_stop_steps=200 \
  --input_dropout=0.7 --layer_dropout=0.7 --l2reg=5e-3 \
  --retrain
