import os
import pandas as pd
import deepmatcher as dm
import random as rd

rd.seed(0)


DATA_PATH = '../sample_data/itunes-amazon/'
SUB_PATH = 'baseline_attack_data'
# ORIGINAL_DATASET = 'attack_testset3.csv'
# ADVERSARIAL_EXAMPLES = 'baseline_attack_data/walmart_amazon.csv'
EMBEDDING_CACHE_PATH = '../.vector_cache'
TRAINING_SET = 'train.csv'
VALIDATION_SET = 'validation.csv'
TEST_SET = 'test.csv'


##########################################################
# Load model

FILE_LIST = ['temp'+str(i)+'.csv' for i in range(0,20)]
fns = fps = tns = tps = 0

# FILE_LIST = ['to_split.csv']

for x in FILE_LIST:
    curr_eval = os.path.join(SUB_PATH, x)
    curr_f = pd.read_csv(os.path.join(DATA_PATH, curr_eval))
    if curr_f.empty:
        break

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    model_path = os.path.join('../', '.', 'model_state', 'hybrid_model.pth')


    # Load sample data
    train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                              validation=VALIDATION_SET, test=curr_eval,
                                              embeddings_cache_path=EMBEDDING_CACHE_PATH)

    model = dm.MatchingModel(attr_summarizer='hybrid')

    model.load_state(model_path)

    stats, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)
    tps = tps + stats.tps
    tns = tns + stats.tns
    fps = fps + stats.fps
    fns = fns + stats.fns
##########################################################

print("Total Tps = ", tps, "Total Tns = ", tns, "Total Fps = ", fps, "Total Fns = ", fns)
