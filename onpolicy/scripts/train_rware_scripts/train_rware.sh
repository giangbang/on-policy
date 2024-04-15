#!/bin/sh
env="RWARE"
scenario="rware-tiny-2ag-v1" 
algo="rmappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_rware.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 \
    --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 3e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "_" --cuda
done