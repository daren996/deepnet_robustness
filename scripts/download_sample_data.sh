#!/bin/bash

FILE=./sample_data/itunes-amazon
if [ -d "$FILE" ]; then
    echo "$FILE exist, download stopped."
else 
    echo "$FILE does not exist, start downloading..."
    mkdir -p sample_data/itunes-amazon
    wget -qnc -P sample_data/itunes-amazon https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/sample_data/itunes-amazon/train.csv
    wget -qnc -P sample_data/itunes-amazon https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/sample_data/itunes-amazon/validation.csv
    wget -qnc -P sample_data/itunes-amazon https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/sample_data/itunes-amazon/test.csv
    wget -qnc -P sample_data/itunes-amazon https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/examples/sample_data/itunes-amazon/unlabeled.csv
echo 'successfully downloaded.'
fi

