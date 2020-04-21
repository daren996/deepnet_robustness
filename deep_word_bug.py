import os

import deepmatcher as dm
import pandas as pd
import random as rd
import string
import numpy as np
import copy

rd.seed(0)

# MAX_PERTURBATIONS = 5

PERTURBATION_PERCENTAGE = 0.2

DATA_PATH = 'sample_data/itunes-amazon'
EMBEDDING_CACHE_PATH = './.vector_cache'
TRAINING_SET = 'train.csv'
VALIDATION_SET = 'validation.csv'
TEST_SET = 'test_temp1.csv'
TRUE_POSITIVE_SET = 'true_positive.csv'
ADVERSARIAL_EXAMPLES = 'blackbox_attack_data/adversarial_examples.csv'


###########################################################
# functions
def process_attack(attack_f, attack_f_path, initial_pred_prob, adversarial_f, adversarial_f_path):
    fields_to_modified = list(attack_f)
    fields_to_modified.pop(0)
    fields_to_modified.pop(0)

    initial_pred_prob = np.asarray(initial_pred_prob)

    drop_index = 0
    orig_cell = []

    prob_diff_matrix = None

    while 1 + 1 == 2:
        continue_counter = 0
        for index, row in attack_f.iterrows():
            if len(orig_cell) == len(initial_pred_prob[0]):
                if orig_cell[index] is not None and orig_cell[index][0]:
                    orig_cell[index] = drop_per_row(fields_to_modified, row, index, attack_f, attack_f_path, drop_index)
                else:
                    continue_counter = continue_counter + 1
                    continue
            else:
                orig_cell.append(drop_per_row(fields_to_modified, row, index, attack_f, attack_f_path, drop_index))

        if continue_counter == len(initial_pred_prob[0]):
            break

        # Attack
        train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                                  validation=VALIDATION_SET, test=attack_f_path.split('/')[-1],
                                                  embeddings_cache_path=EMBEDDING_CACHE_PATH)
        _, pred_prob, prediction = model.run_eval(test, batch_size=128, return_predictions=True)

        # recover all rows
        for index, row in attack_f.iterrows():
            if orig_cell[index] is not None:
                recover_per_row(orig_cell[index], index, attack_f, attack_f_path)

        curr_diff = np.asarray(initial_pred_prob[0]) - np.asarray(pred_prob[0])
        curr_diff = np.reshape(curr_diff, (-1, 1))

        if prob_diff_matrix is None:
            prob_diff_matrix = copy.deepcopy(curr_diff)
        else:
            prob_diff_matrix = np.concatenate((prob_diff_matrix, curr_diff), axis=1)

        drop_index = drop_index + 1

    gen_pert_2(fields_to_modified, attack_f_path, prob_diff_matrix, adversarial_f, adversarial_f_path)


def recover_per_row(prev_tuple, row_index, attack_f, attack_f_path):
    attack_f._set_value(row_index, prev_tuple[0], prev_tuple[1])
    attack_f.to_csv(attack_f_path, index=False)


def drop_per_row(fields_to_modified, row, row_index, attack_f, attack_f_path, drop_index):
    the_row_index = 0

    for field in fields_to_modified:
        curr_cell = str(row[field])
        stored_cell = copy.deepcopy(curr_cell)
        for i in range(0, len(curr_cell)):
            if the_row_index == drop_index:
                new_cell = curr_cell[:i] + curr_cell[i + 1:]
                attack_f._set_value(row_index, field, new_cell)
                attack_f.to_csv(attack_f_path, index=False)

                return (field, stored_cell)
            else:
                the_row_index = the_row_index + 1

    return None


def gen_pert_2(fields_to_modified, attack_f_path, prob_diff_matrix, adversarial_f, adversarial_f_path):
    top_k_indices = np.argsort(prob_diff_matrix, axis=1)
    top_k_indices = np.flip(top_k_indices, axis=1)

    n_examples, _ = prob_diff_matrix.shape

    for index, row in adversarial_f.iterrows():
        num_chars = 0
        for field in fields_to_modified:
            num_chars = num_chars + len(str(row[field]))

        the_perturb_size = int(num_chars * PERTURBATION_PERCENTAGE)
        print(the_perturb_size)
        the_perturb_index = top_k_indices[index][0:the_perturb_size]
        print(the_perturb_index)
        for curr_pert_index in the_perturb_index:
            curr_index = 0
            finish_this = False

            for field in fields_to_modified:
                curr_cell = str(row[field])
                for i in range(0, len(curr_cell)):
                    if curr_pert_index == curr_index:
                        new_cell = perturb_it(curr_cell, i)
                        adversarial_f._set_value(index, field, new_cell)
                        adversarial_f.to_csv(adversarial_f_path, index=False)
                        finish_this = True
                        break
                    else:
                        curr_index = curr_index + 1

                if finish_this:
                    break


def gen_pert(fields_to_modified, attack_f_path, prob_diff_matrix, adversarial_f, adversarial_f_path):
    top_k_indices = np.argsort(prob_diff_matrix, axis=1)
    top_k_indices = np.flip(top_k_indices, axis=1)

    n_examples, _ = prob_diff_matrix.shape

    pert_index_in_top_k = [0 for i in range(0, n_examples)]
    last_pert_index = [-1 for i in range(0, n_examples)]
    pert_number = [1 for i in range(0, n_examples)]
    pert_fields = [None for i in range(0, n_examples)]

    for i in range(0, MAX_PERTURBATIONS):
        if sum(pert_index_in_top_k) == -1 * n_examples:
            break

        print("\nAttempt ", i)
        for index, row in adversarial_f.iterrows():
            finish_this_row = False

            if pert_index_in_top_k[index] == -1:
                continue
            else:
                pert_index = top_k_indices[index][pert_index_in_top_k[index]]
                print("Example index: ", index, "Current perturbing index: ", pert_index)

            curr_index = 0

            for field in fields_to_modified:
                curr_cell = str(row[field])
                for i in range(0, len(curr_cell)):
                    if pert_index == curr_index:
                        new_cell = perturb_it(curr_cell, i)
                        adversarial_f._set_value(index, field, new_cell)
                        adversarial_f.to_csv(adversarial_f_path, index=False)

                        if pert_fields[index] is None:
                            pert_fields[index] = [field]
                        else:
                            pert_fields[index].append(field)
                        finish_this_row = True
                        break
                    else:
                        curr_index = curr_index + 1
                if finish_this_row:
                    break

        train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                                  validation=VALIDATION_SET, test=ADVERSARIAL_EXAMPLES,
                                                  embeddings_cache_path=EMBEDDING_CACHE_PATH)
        _, pred_prob, prediction = model.run_eval(test, return_predictions=True)

        for j in range(0, n_examples):
            if prediction[j][1] < 0.5 and pert_index_in_top_k[j] != -1:
                last_pert_index[j] = pert_index_in_top_k[j]
                pert_index_in_top_k[j] = -1
                print("Perturb example", j, "success with flipping", pert_number[j], 'characters')
            elif pert_index_in_top_k[j] != -1:
                pert_index_in_top_k[j] = pert_index_in_top_k[j] + 1
            elif pert_index_in_top_k[j] == -1 and prediction[j][1] >= 0.5:
                pert_index_in_top_k[j] = last_pert_index[j] + 1

    for i in range(0, n_examples):
        if pert_index_in_top_k[i] != -1:
            pert_number[i] = pert_number[i] + pert_index_in_top_k[i] - 1
        elif pert_index_in_top_k[i] == -1:
            pert_number[i] = pert_number[i] + last_pert_index[i]

    # Write perturbation log
    log_f_path = attack_f_path[0:len(attack_f_path) - len(attack_f_path.split('/')[-1])] + 'perturbation_log.txt'
    with open(log_f_path, 'w') as log_f:
        log_f.write("Maximum number of perturbations allowed: " + str(MAX_PERTURBATIONS) + '\n')
        log_f.write(
            "[Adversarial example index], [Adversarial example id], [Attack result], [the number of perturbations], [perturbed fields]\n")
        total_success_num = 0
        total_pert_num = 0

        for i, x in enumerate(pert_number):
            attack_result = 'Fail'
            num_pert = str(MAX_PERTURBATIONS)
            if pert_index_in_top_k[i] == -1:
                attack_result = 'Success'
                total_success_num = total_success_num + 1
                num_pert = str(x)
            total_pert_num = total_pert_num + int(num_pert)
            log_f.write(
                str(i) + ' ' + str(adversarial_f.iloc[i, 0]) + ' ' + attack_result + ' ' + num_pert + ' ' + ' '.join(
                    [str(elem) for elem in pert_fields[i]]) + '\n')
        log_f.write("Total number of perturbation in this dataset: " + str(total_pert_num) + '\n')
        log_f.write("Attack success rate: " + str(total_success_num / n_examples * 100) + '%\n')


def perturb_it(the_cell, index):
    new_cell = copy.deepcopy(the_cell)
    print("Before perturbing: ", the_cell)

    ## Rule 1
    # if the_cell[index].isdigit():
    #     prob = rd.random()
    #     if the_cell[index] == '0' or prob < 0.5:
    #         new_cell = the_cell[0:index] + chr(ord(the_cell[index])+1) + the_cell[index+1:]
    #     elif the_cell[index] == '9' or prob >= 0.5:
    #         new_cell = the_cell[0:index] + chr(ord(the_cell[index])-1) + the_cell[index+1:]
    # elif the_cell[index] in string.punctuation:
    #     new_cell = the_cell[0:index] + rd.choice(string.punctuation) + the_cell[index+1:]
    # elif the_cell[index].isalpha():
    #     new_cell = the_cell[0:index] + rd.choice([chr(i) for i in range(ord('A'), ord('z') + 1)]) + the_cell[index+1:]
    # elif the_cell[index].isspace():
    #     new_cell = the_cell[:index] + the_cell[index + 1:]

    # Rule 2
    candidate_set = set(string.punctuation).union([chr(i) for i in range(ord('A'), ord('z') + 1)]).union(
        [i for i in range(0, 10)]).union({'drop'}).union({' '})
    replace_chr = rd.sample(candidate_set, k=1)
    if replace_chr[0] == 'drop':
        new_cell = the_cell[:index] + the_cell[index + 1:]
    else:
        new_cell = the_cell[:index] + str(replace_chr[0]) + the_cell[index + 1:]

    # # Rule 3
    # new_cell = the_cell[:index] + the_cell[index + 1:]

    print("After perturbing: ", new_cell)

    return new_cell


###########################################################


###########################################################
# main
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

model_path = os.path.join('.', 'model_state', 'hybrid_model.pth')

# Load sample data
train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                          validation=VALIDATION_SET, test=TEST_SET,
                                          embeddings_cache_path=EMBEDDING_CACHE_PATH)

# Load model
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
    model.run_train(train, validation, epochs=10, batch_size=16, best_save_path=model_path, pos_neg_ratio=3)
else:
    print("trained hybrid model detected, use the old state...")
    model.load_state(model_path)

# # Pick true positive to attack
# fp_tn_fn_indices = ['a']
test_f_path = os.path.join(DATA_PATH, TEST_SET)
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

_, initial_pred_prob, prediction = model.run_eval(test, batch_size=128, return_predictions=True)

# Start to attack

print("\nStart to attack..")
attack_f_path = test_f_path
adversarial_f_path = os.path.join(DATA_PATH, ADVERSARIAL_EXAMPLES)

adversarial_f = pd.read_csv(attack_f_path)

attack_f = pd.read_csv(attack_f_path)
attack_f = attack_f.astype(object)

process_attack(attack_f, attack_f_path, initial_pred_prob, adversarial_f, adversarial_f_path)

# Finally
train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
                                          validation=VALIDATION_SET, test=ADVERSARIAL_EXAMPLES,
                                          embeddings_cache_path=EMBEDDING_CACHE_PATH)
_, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)

print(prediction)