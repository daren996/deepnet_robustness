# coding=utf-8
'''
Download the dataset before running, run ./script/download_data.sh when your first run.

Conduct model attack tests using foolbox
'''
import os

import deepmatcher as dm
import pandas as pd
import foolbox

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_path = os.path.join('.', 'sample_data', 'itunes-amazon')
model_path = os.path.join('.', 'model_state', 'hybrid_model.pth')


'''
Load the sample data

'''
def load_sample_data():
    return dm.data.process(
            path=data_path,
            train='train.csv',
            validation='validation.csv',
            test='test.csv')


'''
Train model

'''
def train_model(train, validation, test, attr_summarizer):
    retrain = None
    model = dm.MatchingModel(attr_summarizer=attr_summarizer)

    if 'model_state' not in os.listdir('./'):
        os.mkdir('model_state')
        retrain = True
    elif os.path.exists(model_path):
        retrain = False
    else: 
        retrain = True

    if retrain:
        print("start training...\n")
        model.run_train(
            train,
            validation,
            epochs=10,
            batch_size=16,
            best_save_path=model_path,
            pos_neg_ratio=3)
    else:
        print("trained hybrid model detected, use the old state...\n")
        model.load_state(model_path)

    return model

'''
FoolBox attacks.

'''
def attack_model(model):
    print('===================== \n')
    print('model before attack: \n')
    evaluate(model)
    print('===================== \n')

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=preprocessing)
    attack = foolbox.attacks.FGSM(fmodel)
    
    '''
    TODO:
    It seems difficult to apply FoolBox to our dataset, the attacks mainly focus on the images.
    We can try to impelement some NLP attack methods on our ER models, I didn't saw any works related to 
    RE models attacking, maybe we can start working on paper "Hotflip: White-Box Adversarial Examples for Text
     Classification", by calculating the gradient and change some words inside the data items. I don't know,
    the items in the dataset for ER problems should have some inner relationships, the attacks should not
    only changing the words from NLP level. Any insights? 
    '''




'''
Evaluate the accuracy on test dataset

'''
def evaluate(model):
    model.run_eval(test)


if __name__ == '__main__':
    train, validation, test = load_sample_data()
    model = train_model(train, validation, test, 'hybrid')
    attack_model(model)



