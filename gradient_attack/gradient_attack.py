import os
import deepmatcher as dm
import pandas as pd


DATA_PATH = '../sample_data/itunes-amazon/gradient_attack_data'
#TRUE_POSITIVE_SET = 'true_positive.csv'
ADVERSARIAL_EXAMPLES = 'gradient_attack_data/gradient_adversarial.csv'
EMBEDDING_CACHE_PATH = '../.vector_cache'
TRAINING_SET = '../train.csv'
VALIDATION_SET = '../validation.csv'
TEST_SET = 'test_temp1.csv'


##########################################################
# Load model
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_path = os.path.join('../', '.', 'model_state', 'hybrid_model.pth')
model = dm.MatchingModel(attr_summarizer='hybrid')
model.load_state(model_path)

FILE_LIST = ['temp'+str(i)+'.csv' for i in range(0,108)]

NUM_TOKENS = 5

for x in FILE_LIST:
    dataset_path = os.path.join("../sample_data/itunes-amazon/gradient_attack_data", x)
    # Load sample data
    train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                              validation=VALIDATION_SET, test=x,
                                              embeddings_cache_path=EMBEDDING_CACHE_PATH)

    gradient_index_list = model.add_candidate_tokens(test, test, epochs=1, batch_size=1, dataset_path=dataset_path, num_tokens=NUM_TOKENS)
    print(len(gradient_index_list))
    train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                              validation=VALIDATION_SET, test=x,
                                              embeddings_cache_path=EMBEDDING_CACHE_PATH)

    model.run_attack_input2(test, test, epochs=1, batch_size=1, gradient_index_list=gradient_index_list, dataset_path=dataset_path)

##########################################################

##########################################################
# Pick true positive to attack
# fp_tn_fn_indices = ['a']
# test_f_path = os.path.join(DATA_PATH, TEST_SET)
# test_f = pd.read_csv(test_f_path)
# true_positive_path = os.path.join(DATA_PATH, TRUE_POSITIVE_SET)
#
# while 1 + 1 == 2:
#     fp_tn_fn_indices = []
#     _, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)
#
#     for index, row in test_f.iterrows():
#         if row['label'] == 0 or (row['label'] == 1 and prediction[index][1] < 0.5):
#             fp_tn_fn_indices.append(int(prediction[index][0]))
#
#     if len(fp_tn_fn_indices) == 0:
#         break
#
#     test_f = test_f[~test_f['id'].isin(fp_tn_fn_indices)]
#     test_f.to_csv(true_positive_path, index=False)
#
#     train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
#                                               validation=VALIDATION_SET, test=TRUE_POSITIVE_SET,
#                                               embeddings_cache_path=EMBEDDING_CACHE_PATH)
#     test_f = pd.read_csv(true_positive_path)
##########################################################

# input_tensor = model.get_input_tensor(test)




# print("\nStart to attack..")
#
# train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
#                                           validation=VALIDATION_SET, test=TRUE_POSITIVE_SET,
#                                           embeddings_cache_path=EMBEDDING_CACHE_PATH)
# _, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)
#
# true_positive_path = os.path.join(DATA_PATH, TRUE_POSITIVE_SET)
# attack_f = pd.read_csv(true_positive_path)
# adversarial_f_path = os.path.join(DATA_PATH, ADVERSARIAL_EXAMPLES)
# adversarial_f = pd.read_csv(true_positive_path)
#
# adversarial_f.to_csv(adversarial_f_path, index=False)
#
# print(test.embedding)