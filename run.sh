#!/bin/bash
nohup python -u main.py --batch_size_train 64 --batch_size_test 64 --l_epochs 5 --g_epochs 5 \
    --seed 2023 --pc_valid 0.05 --device 2 --lr 0.01 --momentum 0.9 --lr_min 1e-5 --lr_patience 6 \
    --lr_factor 2 --task_num 10 --clients_num 10 --selected_clients 2 > test3.log 2>&1 &


# python main.py --batch_size_train 64 --batch_size_test 64 --l_epochs 5 --g_epochs 20 \
#     --seed 2023 --pc_valid 0.05 --device 2 --lr 0.01 --momentum 0.9 --lr_min 1e-5 --lr_patience 6 \
#     --lr_factor 2 --task_num 10 --clients_num 10 --selected_clients 2 
