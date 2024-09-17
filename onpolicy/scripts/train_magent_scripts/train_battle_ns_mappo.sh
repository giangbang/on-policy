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
    --env_id ${env_id} --num_agents ${num_agents} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 10 --num_mini_batch 1 --episode_length 80 --num_env_steps 100000 \
    --ppo_epoch 5 --use_ReLU --use_eval --gain 0.01 --lr 2.5e-4 --critic_lr 7e-4 --wandb_name "xxx" --user_name "giangbang" \
    --share_policy --deterministic_eval --eval_episodes 10 --n_eval_rollout_threads 2
done
