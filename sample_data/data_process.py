import os
import pandas as pd

TABLE_A_PATH = './walmart_amazon_exp_data/tableA.csv'
TABLE_B_PATH = './walmart_amazon_exp_data/tableB.csv'
RELATION_PATH = './walmart_amazon_exp_data/valid.csv'
OUT_PATH = './walmart_amazon_exp_data/valid_processed.csv'


table_a_f = pd.read_csv(TABLE_A_PATH)
table_b_f = pd.read_csv(TABLE_B_PATH)
relation_f = pd.read_csv(RELATION_PATH)

# Set attribute fields
out_f_columns = table_a_f.columns.delete(0)
out_f_columns = out_f_columns.append(out_f_columns)
for i in range(0, out_f_columns.size):
    curr_index = str(out_f_columns[i])
    out_f_columns = out_f_columns.delete(i)
    if i < out_f_columns.size / 2:
        out_f_columns = out_f_columns.insert(i, 'left_'+curr_index)
    else:
        out_f_columns = out_f_columns.insert(i, 'right_' + curr_index)
out_f_columns = out_f_columns.insert(0, 'label')
out_f_columns = out_f_columns.insert(0, 'id')

out_f = pd.DataFrame(columns=out_f_columns)

# Load data
for index, row in relation_f.iterrows():
    left = list(table_a_f.loc[row['ltable_id']])
    left.pop(0)
    right = list(table_b_f.loc[row['rtable_id']])
    right.pop(0)
    out_f.loc[index] = [index] + [row['label']] + left + right

out_f.to_csv(OUT_PATH,index=False)
