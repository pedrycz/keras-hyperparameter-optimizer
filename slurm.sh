#!/bin/sh
#SBATCH -p plgrid-gpu
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1

module purge

module load plgrid/tools/python-intel
module load plgrid/apps/cuda

THEANO_FLAGS='device=gpu,floatX=float32,lib.cnmem=1.0' python -m optimizer_genetic.py
