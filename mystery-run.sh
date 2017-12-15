#!/usr/bin/env bash

# Source bashrc
#source $HOME/.bashrc

# Activate the environment
#source activate duckmeagain

export dataset_now=fasion-mnist

# Run the script
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae2 --no-tsne --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae3 --no-tsne --divergence jensen --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae4 --no-tsne --divergence hellinger --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae5 --no-tsne --divergence wasserstein --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae6 --no-tsne --divergence wasserstein --anirudh --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae7 --no-tsne --divergence jensen --beta 0.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae8 --no-tsne --divergence jensen --beta 1.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae9 --no-tsne --divergence jensen --beta 2.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae10 --no-tsne --divergence jensen --beta 10 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae11 --no-tsne --divergence hellinger --beta 500 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae12 --no-tsne --divergence hellinger --beta 1000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae13 --no-tsne --divergence hellinger --beta 2000 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae14 --no-tsne --divergence hellinger --beta 3000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae15 --no-tsne --divergence hellinger --beta 4000 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae16 --no-tsne --divergence hellinger --beta 5000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae17 --no-tsne --divergence hellinger --beta 6000 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae18 --no-tsne --divergence hellinger --beta 7000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae19 --no-tsne --divergence hellinger --beta 8000 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae20 --no-tsne --divergence hellinger --beta 9000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae21 --no-tsne --divergence hellinger --beta 10000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae22 --no-tsne --divergence hellinger --beta 11000 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae23 --no-tsne --divergence wasserstein --beta 0.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae24 --no-tsne --divergence wasserstein --beta 1 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae25 --no-tsne --divergence wasserstein --beta 1.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae26 --no-tsne --divergence wasserstein --beta 2.5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae27 --no-tsne --divergence wasserstein --beta 5 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae28 --no-tsne --divergence wasserstein --beta 10 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae29 --no-tsne --divergence wasserstein --beta 20 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae30 --no-tsne --divergence wasserstein --beta 30 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae31 --no-tsne --divergence wasserstein --beta 40 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae32 --no-tsne --divergence wasserstein --beta 50 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae33 --no-tsne --divergence wasserstein --beta 60 --dataset $dataset_now
#sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae34 --no-tsne --divergence wasserstein --beta 70 --dataset $dataset_now
sbatch --gres=gpu --qos=high --mem=6000 --time=3:00:00 run.sh --output-folder mystery-vae35 --no-tsne --divergence wasserstein --beta 100 --dataset $dataset_now

$@
