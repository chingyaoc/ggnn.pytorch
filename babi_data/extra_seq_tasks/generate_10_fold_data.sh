#!/bin/bash

for fold in {1..10}; do
    echo
    echo Generating data for fold $fold
    echo

    mkdir -p fold_$fold/noisy_data
    mkdir -p fold_$fold/noisy_data/train
    mkdir -p fold_$fold/noisy_data/test

    for i in 4 5; do
        python generate_data.py $i 5 1000 4 > fold_$fold/noisy_data/train/${i}.txt
        python generate_data.py $i 5 1000 4 > fold_$fold/noisy_data/test/${i}.txt

        python preprocess.py $i fold_$fold/noisy_data fold_$fold/noisy_parsed
    done

    mkdir -p fold_$fold/noisy_rnn 
    mkdir -p fold_$fold/noisy_rnn/train
    mkdir -p fold_$fold/noisy_rnn/test

    for i in 4 5; do
        python rnn_preprocess.py fold_$fold/noisy_parsed/train/${i}_graphs.txt fold_$fold/noisy_rnn/train/${i}_tokens.txt --mode graph --nval 50
        python rnn_preprocess.py fold_$fold/noisy_rnn/train/${i}_tokens.txt fold_$fold/noisy_rnn/train/${i}_rnn.txt --mode rnn
        python rnn_preprocess.py fold_$fold/noisy_rnn/train/${i}_tokens.txt.val fold_$fold/noisy_rnn/train/${i}_rnn.txt.val --mode rnn --dict fold_$fold/noisy_rnn/train/${i}_rnn.txt.dict

        python rnn_preprocess.py fold_$fold/noisy_parsed/test/${i}_graphs.txt fold_$fold/noisy_rnn/test/${i}_tokens.txt --mode graph
        python rnn_preprocess.py fold_$fold/noisy_rnn/test/${i}_tokens.txt fold_$fold/noisy_rnn/test/${i}_rnn.txt --dict fold_$fold/noisy_rnn/train/${i}_rnn.txt.dict --mode rnn
    done
done
