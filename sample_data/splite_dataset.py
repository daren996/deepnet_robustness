import os
import pandas as pd

TO_SPLIT_FOLDER_PATH = './itunes-amazon/baseline_attack_data'
TO_SPLIT_F = 'to_split.csv'

FILE_LIST = ['temp'+str(i)+'.csv' for i in range(0,20)]

to_split_f = pd.read_csv(os.path.join(TO_SPLIT_FOLDER_PATH, TO_SPLIT_F))
column_fields = to_split_f.columns

new_file_size = (len(to_split_f)//len(FILE_LIST)) + 1
new_file_index = 0
# new_file_size = 1
print(new_file_size)

for x in FILE_LIST:
    the_curr_out_path = os.path.join(TO_SPLIT_FOLDER_PATH, x)
    the_curr_out_f = pd.DataFrame(columns=column_fields)
    the_curr_out_f = the_curr_out_f.append(to_split_f[new_file_index: new_file_index + new_file_size].copy(deep=True))
    new_file_index = new_file_index + new_file_size
    the_curr_out_f.to_csv(the_curr_out_path, index=False)
