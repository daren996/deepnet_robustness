import os

import deepmatcher as dm
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_path = os.path.join('.', 'model_state', 'hybrid_model_walmart_amazon.pth')

# Load sample data
train, validation, test = dm.data.process(
    path='sample_data/walmart_amazon_exp_data',
    train='train_processed.csv',
    validation='valid_processed.csv',
    test='test_processed.csv',
    embeddings_cache_path='/media/yibin/DataDisk/deepnet_sensitivity/.vector_cache'
)

# Train model
retrain = True
model = dm.MatchingModel(attr_summarizer='hybrid')

if 'model_state' not in os.listdir('./'):
    os.mkdir('model_state')
    retrain = True
elif os.path.exists(model_path):
    retrain = False
else:
    retrain = True

if retrain:
    print("start training...")
    model.run_train(
        train,
        validation,
        epochs=20,
        batch_size=16,
        best_save_path=model_path,
        pos_neg_ratio=3)
else:
    print("trained hybrid model detected, use the old state...")
    model.load_state(model_path)
