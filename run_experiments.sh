#!/bin/bash


fcst_steps=288
observe_steps=$((2 * fcst_steps))
window_width=$((fcst_steps + observe_steps))
# latent_dims=(16)
# encoders=("dnn")
# python example.py -d solete --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder dnn --pass_raw --shared_encoder --run_times 20



# python post_opt.py
# python wind_opt.py







# python example.py -d solete --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder rnn --pass_raw --shared_encoder --run_times 20
# python example.py -d nrel --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder rnn --pass_raw --shared_encoder --run_times 20
# python example.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder rnn --pass_raw --shared_encoder --run_times 20
# python example.py -d australia --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder dnn --run_times 20 --pass_raw --shared_encoder
# python example.py -d mfred --observe_steps $observe_steps --window_width $window_width --latent_dim 16 --avg_terms_list 12 3 1 --gpu 3 --encoder dnn --run_times 20 --pass_raw --epochs 10

python example_benchmarks.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --gpu 1 --avg_terms_list 12 3 1 --run_times 20 --persistence loop 
# for i in ${latent_dims[*]}; do
#     for j in ${encoders[*]}; do
#         python example.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim $i --avg_terms_list 12 3 1 --gpu 3 --encoder $j --run_times 1
#         # python example.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim $i --avg_terms_list 12 3 1 --gpu 3 --encoder $j --shared_encoder --pass_raw
#         # python example.py -d mfred --add_external_feature --observe_steps $observe_steps --window_width $window_width --latent_dim $i --avg_terms_list 12 3 1 --gpu 3 --encoder $j --shared_encoder
#         # python example.py -d mfred --add_external_featur --observe_steps $observe_steps --window_width $window_width --latent_dim $i --avg_terms_list 12 3 1 --gpu 3 --encoder $j
#         # python example.py -d mfred --observe_steps $observe_steps --window_width $window_width --latent_dim $i --avg_terms_list 12 3 1 --gpu 3 --encoder $j --shared_encoder --pass_raw --weight_decay 0.0001
#     done
# done


