import os
import pandas as pd
import deepmatcher as dm
import random as rd
import re
import string
from datetime import datetime

rd.seed(0)

DATA_PATH = '../sample_data/itunes-amazon'
TRUE_POSITIVE_SET = 'true_positive.csv'
ADVERSARIAL_EXAMPLES = 'baseline_attack_data/baseline_adversarial1.csv'
EMBEDDING_CACHE_PATH = '/media/yibin/DataDisk/deepnet_sensitivity/.vector_cache'
TRAINING_SET = 'train.csv'
VALIDATION_SET = 'validation.csv'

TEST_SET = 'test.csv'


NOISE_CATEGORIES = ['Missing Value', 'Data Error', 'Data formatting', 'Abbreviation', 'Word permutation', 'Data truncation', 'Changing attribute', 'Misspelling']


def process_attack(attack_f, adversarial_f, save_path, noise_category, perturb_size):
    # Define which fields could inject the noise
    # For Missing Value / Changing attribute
    # fields_to_modified = list(attack_f)[2:]
    # For Abbreviation / Truncation / Misspelling
    fields_to_modified = list(attack_f)[2:6] + [list(attack_f)[7]] + list(attack_f)[10:14] + [list(attack_f)[15]]
    # For Data formatting
    # fields_to_modified = [list(attack_f)[9]] + [list(attack_f)[17]]
    # For Data Error
    # fields_to_modified = [list(attack_f)[6]] + [list(attack_f)[8]] + [list(attack_f)[14]] + [list(attack_f)[16]]
    # For Word permutation
    # fields_to_modified = list(attack_f)[3:6] + list(attack_f)[11:14]

    print(fields_to_modified)

    if noise_category == 'Changing attribute':
        the_field_to_modified = rd.sample(fields_to_modified, k=perturb_size)
        # adversarial_f = adversarial_f.drop(columns=the_field_to_modified)
    else:
        for index, row in attack_f.iterrows():
            the_field_to_modified = sample_fields(row, fields_to_modified, perturb_size)

            for x in the_field_to_modified:
                if noise_category == 'Missing Value':
                    new_cell = random_missing_value(row[x])
                elif noise_category == 'Abbreviation':
                    new_cell = random_abbreviation(row[x])
                elif noise_category == 'Data formatting':
                    new_cell = random_formatting(row[x])
                elif noise_category == 'Data Error':
                    new_cell = random_dataerror(row[x])
                elif noise_category == 'Word permutation':
                    new_cell = random_wordpermutation(row[x])
                elif noise_category == 'Data truncation':
                    new_cell = random_truncation(row[x])
                elif noise_category == 'Misspelling':
                    new_cell = random_misspelling(row[x])

                # print(new_cell)
                adversarial_f._set_value(index, x, new_cell)

    adversarial_f.to_csv(save_path, index=False)


def sample_fields(row, fields_to_modified, num):
    # Sample token > 1
    # check = True
    # while check:
    #     check = False
    #     the_field_to_modified = rd.sample(fields_to_modified, k=num)
    #
    #     for x in the_field_to_modified:
    #         if len(row[x].split(' ')) == 1:
    #             check = True

    # the_field_to_modified = rd.sample(fields_to_modified, k=num)
    the_field_to_modified = rd.choices(fields_to_modified, k=num)

    return the_field_to_modified


def random_missing_value(cell):
    missing_value = ['', ' ', 'NULL', 'UNKNOWN']
    return rd.choice(missing_value)


def random_abbreviation(cell):
    token_list = cell.split(' ')
    result_cell = ''

    pre_dot = False

    for i in token_list:
        if i.isdigit() or i.isupper() or len(i) == 1 or i[len(i) - 1] == '.':
            if pre_dot:
                result_cell = result_cell + ' '
            result_cell = result_cell + i + ' '
            pre_dot = False
        elif i[0].isupper():
            result_cell = result_cell + i[0] + '.'
            pre_dot = True

    return result_cell


def random_formatting(cell):
    format_choices = ['%m/%d/%y', '%d/%m/%y', '%b %d , %Y', '%B %d , %Y', '%d-%B-%Y', '%d-%b-%y']

    if len(cell.split('-')) > 1:
        objDate = datetime.strptime(cell, format_choices[5])
        return datetime.strftime(objDate, rd.choice(format_choices[0:5]))
    elif cell.split(' ')[0].isalpha():
        objDate = datetime.strptime(cell, '%B %d , %Y')
        return datetime.strftime(objDate, rd.choice(format_choices[0:3] + format_choices[4:]))
    else:
        return cell


def random_dataerror(cell):
    token_list = cell.split(' ')
    new_cell = None

    if token_list[0] == '$':
        new_cell = '$ ' + str(round(float(token_list[1]) * rd.uniform(0,10), 2))
    elif len(token_list) == 1:
        new_cell = str(int(cell.split(':')[0])+rd.randint(1,9)) + ':' + str(int(cell.split(':')[1])+rd.randint(1,30))

    return new_cell


def random_wordpermutation(cell):
    token_list = cell.split(' ')
    word_list = []
    symbol_list = []

    for i in token_list:
        if i not in string.punctuation:
            word_list.append(i)
        else:
            symbol_list.append(i)

    if len(word_list) == 2:
        word_list.reverse()
    else:
        rd.shuffle(word_list)
    # word_list.reverse()

    for i, x in enumerate(symbol_list):
        word_list.insert(2*i+1, x)

    new_cell = ' '.join(map(str, word_list))

    return new_cell


def random_truncation(cell):
    token_list = cell.split(' ')

    return ' '.join(map(str, token_list[0:int(len(token_list)/2)]))


def random_misspelling(cell):
    the_type = rd.choice(['insertion', 'deletion', 'substitution', 'permutation'])
    alphabet = set(string.punctuation).union([chr(i) for i in range(ord('A'), ord('z') + 1)]).union([i for i in range(0,10)]).union({' '})
    new_cell = None

    if the_type == 'insertion':
        insert_position = rd.randrange(0, len(cell)+1)
        insert_char = rd.sample(alphabet, k=1)[0]
        new_cell = cell[0:insert_position] + str(insert_char) + cell[insert_position:]
    elif the_type == 'deletion':
        delete_position = rd.randrange(0, len(cell))
        new_cell = cell[0:delete_position] + cell[delete_position+1:]
    elif the_type == 'substitution':
        substitute_position = rd.randrange(0, len(cell))
        substitute_char = rd.sample(alphabet, k=1)[0]
        new_cell = cell[0:substitute_position] + str(substitute_char) + cell[substitute_position+1:]
    elif the_type == 'permutation':
        permutation_position = rd.randrange(0, len(cell)-1)
        new_cell = cell[0:permutation_position] + cell[permutation_position+1] + cell[permutation_position] + cell[permutation_position+2:]

    return new_cell

##########################################################
# Load model
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_path = os.path.join('../', '.', 'model_state', 'hybrid_model.pth')

# Load sample data
train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                          validation=VALIDATION_SET, test=TEST_SET,
                                          embeddings_cache_path=EMBEDDING_CACHE_PATH)

model = dm.MatchingModel(attr_summarizer='hybrid')

if 'model_state' not in os.listdir('./'):
    # os.mkdir('model_state')
    retrain = True
elif os.path.exists(model_path):
    retrain = False
else:
    retrain = True

retrain = False

if retrain:
    print("start training...")
    model.run_train(train, validation, epochs=10, batch_size=16, best_save_path=model_path, pos_neg_ratio=3)
else:
    print("trained hybrid model detected, use the old state...")
    model.load_state(model_path)
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

print("\nStart to attack..")

train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                          validation=VALIDATION_SET, test=TRUE_POSITIVE_SET,
                                          embeddings_cache_path=EMBEDDING_CACHE_PATH)
_, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)

true_positive_path = os.path.join(DATA_PATH, TRUE_POSITIVE_SET)
attack_f = pd.read_csv(true_positive_path)
adversarial_f_path = os.path.join(DATA_PATH, ADVERSARIAL_EXAMPLES)
adversarial_f = pd.read_csv(true_positive_path)

process_attack(attack_f, adversarial_f, adversarial_f_path, NOISE_CATEGORIES[7], perturb_size=5)

# Results
train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                          validation=VALIDATION_SET, test=ADVERSARIAL_EXAMPLES,
                                          embeddings_cache_path=EMBEDDING_CACHE_PATH)
# adversarial_f = pd.read_csv(adversarial_f_path)
# # for index, row in adversarial_f.iterrows():
# #     print(row)
_, pred_prob, prediction = model.run_eval(test, return_predictions=True)
