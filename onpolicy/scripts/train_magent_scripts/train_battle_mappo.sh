#!/bin/sh
env="MAgent2"
env_id="battle_v4" 
algo="mappo" #"mappo" "ippo"
exp="check"
seed_max=1

echo "env is ${env}, env id is ${env_id}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_magent.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --env_id ${env_id} --hidden_size 32 --seed ${seed} --use_proper_time_limits \
    --n_training_threads 1 --n_rollout_threads 16 --num_mini_batch 2 --episode_length 50 --num_env_steps 10000000 \
    --ppo_epoch 5 --use_ReLU --gain 0.01 --lr 1e-4 --critic_lr 3e-4 --wandb_name "xxx" --user_name "giangbang" \
    --share_policy --deterministic_eval --eval_episodes 10 --n_eval_rollout_threads 2 --entropy_coef 0.01 \
    --clip_param 0.1 --opti_eps 1e-8 --use_centralized_V --share_policy \
    --gamma 0.95
    #--use_eval
done
