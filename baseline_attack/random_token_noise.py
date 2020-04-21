import os
import pandas as pd
import deepmatcher as dm
import random as rd
import re
import string


DATA_PATH = '../sample_data/itunes-amazon'
ORIGINAL_DATASET = 'test.csv'
ADVERSARIAL_EXAMPLES = 'baseline_attack_data/to_split.csv'


def process_attack(attack_f, adversarial_f, save_path, perturb_size):
    fields_to_modified = list(attack_f)[2:]
    # print(fields_to_modified)
    all_tokens = [i for i in range(0, 10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

    for index, row in attack_f.iterrows():
        num_token = 0
        word_list = attack_f.loc[index].tolist()[2:]
        word_list = ' '.join(map(str, word_list)).split(' ')
        perturb_index_list = rd.sample([i for i in range(0,len(word_list))], k=perturb_size)

        for x in perturb_index_list:
            new_word = rd.choices(all_tokens, k=len(word_list[x]))
            new_word = ''.join(map(str, new_word))

            count = 0
            for field in fields_to_modified:
                curr_cell = str(attack_f.loc[index][field]).split(' ')
                if count + len(curr_cell) > x:
                    print(' '.join(map(str, curr_cell)))
                    curr_cell[x - count] = new_word
                    new_cell = ' '.join(map(str, curr_cell))
                    print(new_cell)
                    attack_f._set_value(index, field, new_cell)
                    attack_f.to_csv(save_path, index=False)
                    break
                else:
                    count = count + len(curr_cell)



print("\nStart to attack..")

perturb_size_list = [5]

for i in perturb_size_list:

    original_path = os.path.join(DATA_PATH, ORIGINAL_DATASET)
    original_f = pd.read_csv(original_path)
    adversarial_f_path = os.path.join(DATA_PATH, ADVERSARIAL_EXAMPLES)
    adversarial_f = pd.read_csv(original_path)
    adversarial_f = adversarial_f.astype(object)

    process_attack(original_f, adversarial_f, adversarial_f_path, perturb_size=i)