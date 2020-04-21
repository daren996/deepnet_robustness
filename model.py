# coding=utf-8
"""
Download dataset before running.

"""
import os

import deepmatcher as dm
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_path = os.path.join('.', 'model_state', 'hybrid_model.pth')

# Load sample data
train, validation, test = dm.data.process(
    path='sample_data/itunes-amazon',
    train='train.csv',
    validation='validation.csv',
    test='test.csv',
    embeddings_cache_path='./.vector_cache'
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

retrain = False

if retrain:
    print("start training...")
    model.run_train(
        train,
        validation,
        epochs=10,
        batch_size=16,
        best_save_path=model_path,
        pos_neg_ratio=3)
else:
    print("trained hybrid model detected, use the old state...")
    model.load_state(model_path)

# Evaluate the accuracy on test dataset
# _,predict_prob,prediction = model.run_eval(test)


# # Run predictions on the unlabeled dataset
candidiate = dm.data.process_unlabeled(
    path=os.path.join('.', 'sample_data', 'itunes-amazon', 'adversarial_examples.csv'),
    trained_model=model,
    ignore_columns=('ltable_id', 'rtable_id', 'label'))

predictions = model.run_prediction(test, output_attributes=True)