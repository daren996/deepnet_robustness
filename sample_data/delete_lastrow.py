import os
import pandas as pd

PATH = './itunes-amazon/gradient_attack_data'
FILE_LIST = ['temp'+str(i)+'.csv' for i in range(0,108)]

NUM_DELETE = 5

for x in FILE_LIST:
    the_curr_out_path = os.path.join(PATH, x)
    df = pd.read_csv(the_curr_out_path)

    for i in range(0, NUM_DELETE):
       df = df.drop(df.index[-1])

    df.to_csv(the_curr_out_path, index=False)