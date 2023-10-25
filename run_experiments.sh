#!/bin/bash

fcst_steps=288
observe_steps=$((2 * fcst_steps))
window_width=$((fcst_steps + observe_steps))

python example.py -d nrel --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 0 --encoder rnn --pass_raw --shared_encoder --run_times 20
python example.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 0 --encoder dnn --pass_raw --shared_encoder --run_times 20

python example_benchmarks.py -d nrel --add_external_feature --observe_steps $observe_steps --window_width $window_width --gpu 0 --avg_terms_list 12 3 1 --run_times 20 --persistence naive 
python example_benchmarks.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --gpu 0 --avg_terms_list 12 3 1 --run_times 20 --persistence loop 




