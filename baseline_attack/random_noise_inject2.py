import os
import pandas as pd
import deepmatcher as dm
import random as rd
import re
import string
from datetime import datetime


DATA_PATH = '../sample_data/itunes-amazon'
ORIGINAL_DATASET = 'test.csv'
ADVERSARIAL_EXAMPLES = 'baseline_attack_data/to_split.csv'
EMBEDDING_CACHE_PATH = '/media/yibin/DataDisk/deepnet_sensitivity/.vector_cache'
TRAINING_SET = 'train.csv'
VALIDATION_SET = 'validation.csv'
TEST_SET = 'to_split.csv'


NOISE_CATEGORIES = ['Missing Value', 'Data Error', 'Data formatting', 'Abbreviation', 'Word permutation', 'Data truncation', 'Changing attribute', 'Misspelling']


def process_attack(attack_f, adversarial_f, save_path, noise_category, perturb_size):
    print("Noise category: ", noise_category)

    # Define which fields could inject the noise
    # For Missing Value / Word permutation / Misspelling

    fields_to_modified = list(attack_f)[2:7] + list(attack_f)[7:]

    # For Data formatting / Changing attributes
    # fields_to_modified = list(attack_f)[6]
    print(fields_to_modified)

    print("Perturbation size: ", perturb_size)

    for index, row in attack_f.iterrows():
        if noise_category == 'Abbreviation':
            num_words_row = 0
            for field in list(attack_f)[2:]:
                num_words_row = num_words_row + len(str(row[field]).split(' '))

            the_perturb_size = int(num_words_row * perturb_size)

            num_words = 0
            abbreviation_index = []
            for field in fields_to_modified:
                for x in str(row[field]).split(' '):
                    if x.isalpha():
                        abbreviation_index.append(num_words)
                    num_words = num_words + 1

            word_index_to_modify = rd.sample(set(abbreviation_index), k=the_perturb_size)

            col_index = 0
            for field in fields_to_modified:
                word_list = str(row[field]).split(' ')
                for i in range(0, len(word_list)):
                    if col_index in word_index_to_modify:
                        word_list[i] = word_list[i][0] + '.'

                    col_index = col_index + 1
                new_cell = ' '.join(map(str, word_list))
                # print(new_cell)
                adversarial_f._set_value(index, field, new_cell)

        elif noise_category == 'Missing Value':
            the_field_to_modified = rd.sample(fields_to_modified, k=perturb_size)
            missing_value = ['', ' ', 'NULL', 'UNKNOWN']
            for x in the_field_to_modified:
                new_cell = rd.choice(missing_value)
                adversarial_f._set_value(index, x, new_cell)

        elif noise_category == 'Word permutation':
            num_words_row = 0
            candidate_index = []
            for field in list(attack_f)[2:]:
                field_length = len(str(row[field]).split(' '))
                # if field.startswith('left_'):
                #     candidate_index.extend([i+num_words_row for i in range(0,field_length-1)])
                candidate_index.extend([i + num_words_row for i in range(0, field_length - 1)])
                num_words_row = num_words_row + field_length

            the_perturb_size = int(num_words_row * perturb_size)
            # print(the_perturb_size)
            index_to_swap = rd.sample(set(candidate_index), k=the_perturb_size)

            col_index = 0
            for field in fields_to_modified:
                word_list = str(row[field]).split(' ')
                for i in range(0, len(word_list)):
                    if col_index in index_to_swap:
                        first_ele = word_list.pop(i)
                        second_ele = word_list.pop(i)
                        word_list.insert(i, first_ele)
                        word_list.insert(i, second_ele)
                    col_index = col_index + 1
                new_cell = ' '.join(map(str, word_list))
                adversarial_f._set_value(index, field, new_cell)

        elif noise_category == 'Data formatting':
            new_cell = '$ '+ str(row['left_price'])
            adversarial_f._set_value(index, 'left_price', new_cell)

        elif noise_category == 'Data Error':
            factor = rd.uniform(0,2)
            new_cell = round(factor * (row['left_price'] + 0.5), 2)
            adversarial_f._set_value(index, 'left_price', new_cell)

        elif noise_category == 'Data truncation':
            the_field_to_modified = rd.sample(fields_to_modified, k=perturb_size)
            # print(the_field_to_modified)
            for x in the_field_to_modified:
                curr_cell = str(row[x])
                new_cell = curr_cell[0:len(curr_cell)//2]
                adversarial_f._set_value(index, x, new_cell)

        elif noise_category == 'Changing attribute':
            new_cell = rd.randint(0,1000)
            adversarial_f._set_value(index, 'left_price', new_cell)

        elif noise_category == 'Misspelling':
            num_chars = 0
            left_num_chars = 0
            for field in list(attack_f)[2:]:
                num_chars = num_chars + len(str(row[field]))
                if field.startswith('left_'):
                    left_num_chars = left_num_chars + len(str(row[field]))


            # the_perturb_size = int(left_num_chars * perturb_size)
            the_perturb_size = int(num_chars * perturb_size)

            # print(num_chars)
            # candidate_index = [i for i in range(0, left_num_chars)]
            candidate_index = [i for i in range(0,num_chars)]

            the_chars_to_modify = rd.sample(set(candidate_index), k=the_perturb_size)
            the_chars_to_modify.sort()
            # print(the_chars_to_modify)
            row_char_index = 0
            char_index_to_modify = the_chars_to_modify.pop(0)

            for field in fields_to_modified:
                curr_cell = str(row[field])
                cell_char_index = 0
                # print(curr_cell)
                for c in curr_cell:
                    if row_char_index == char_index_to_modify:
                        alphabet = set(string.punctuation).union([chr(i) for i in range(ord('A'), ord('z') + 1)]).union([i for i in range(0,10)]).union({' '})
                        # the_type = rd.choice(['insertion', 'deletion', 'substitution', 'permutation'])
                        the_type = rd.choice(['substitution'])
                        new_cell = None
                        if the_type == 'insertion':
                            insert_char = rd.sample(alphabet, k=1)[0]
                            new_cell = curr_cell[0:cell_char_index] + str(insert_char) + curr_cell[cell_char_index:]
                        elif the_type == 'deletion':
                            if cell_char_index >= len(curr_cell):
                                temp_index = rd.randrange(0,len(curr_cell))
                                new_cell = curr_cell[0:temp_index] + curr_cell[temp_index+1:]
                            else:
                                new_cell = curr_cell[0:cell_char_index] + curr_cell[cell_char_index+1:]
                        elif the_type == 'substitution':
                            substitute_char = rd.sample(alphabet, k=1)[0]
                            if cell_char_index >= len(curr_cell):
                                temp_index = rd.randrange(0,len(curr_cell))
                                new_cell = curr_cell[0:temp_index] + str(substitute_char) + curr_cell[temp_index+1:]
                            else:
                                new_cell = curr_cell[0:cell_char_index] + str(substitute_char) + curr_cell[cell_char_index+1:]
                        elif the_type == 'permutation':
                            if cell_char_index >= len(curr_cell)-1:
                                if len(curr_cell)-1 != 0:
                                    temp_index = rd.randrange(0,len(curr_cell)-1)
                                    new_cell = curr_cell[0:temp_index] + curr_cell[temp_index + 1] + curr_cell[temp_index] + curr_cell[temp_index + 2:]
                            else:
                                new_cell = curr_cell[0:cell_char_index] + curr_cell[cell_char_index + 1] + curr_cell[cell_char_index] + curr_cell[cell_char_index + 2:]

                        curr_cell = new_cell
                        # print(new_cell)

                        adversarial_f._set_value(index, field, new_cell)

                        if len(the_chars_to_modify) == 0:
                            break
                        else:
                            char_index_to_modify = the_chars_to_modify.pop(0)

                    row_char_index = row_char_index + 1
                    cell_char_index = cell_char_index + 1

                # if len(the_chars_to_modify) == 0:
                #     break
            # print()

    print(len(adversarial_f))
    adversarial_f.to_csv(save_path, index=False)


##########################################################
# Load model
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
#
# model_path = os.path.join('../', '.', 'model_state', 'hybrid_model_walmart_amazon.pth')
#
# # Load sample data
# train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
#                                           validation=VALIDATION_SET, test=ORIGINAL_DATASET,
#                                           embeddings_cache_path=EMBEDDING_CACHE_PATH)
#
# model = dm.MatchingModel(attr_summarizer='hybrid')
#
# model.load_state(model_path)
#
# _, initial_pred_prob, prediction = model.run_eval(test, return_predictions=True)
##########################################################

print("\nStart to attack..")

perturb_size_list = [0.2]

for i in perturb_size_list:

    original_path = os.path.join(DATA_PATH, ORIGINAL_DATASET)
    original_f = pd.read_csv(original_path)
    adversarial_f_path = os.path.join(DATA_PATH, ADVERSARIAL_EXAMPLES)
    adversarial_f = pd.read_csv(original_path)
    adversarial_f = adversarial_f.astype(object)

    process_attack(original_f, adversarial_f, adversarial_f_path, NOISE_CATEGORIES[7], perturb_size=i)

    # train, validation, test = dm.data.process(path=DATA_PATH, train=TRAINING_SET,
    #                                           validation=VALIDATION_SET, test=ADVERSARIAL_EXAMPLES,
    #                                           embeddings_cache_path=EMBEDDING_CACHE_PATH)
    # _, pred_prob, prediction = model.run_eval(test, return_predictions=True)
    # print("Current perturbation size is: ", i)
    # print("=======================================\n")
