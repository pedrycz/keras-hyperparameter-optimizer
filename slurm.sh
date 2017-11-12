#!/bin/sh
#SBATCH -p plgrid-gpu
#SBATCH -t 1:00:00
#SBATCH --gres=gpu

module purge

module load plgrid/tools/python/2.7.9
module load plgrid/apps/cuda

module unload plgrid/tools/openmpi

THEANO_FLAGS=device=cuda,floatX=float32 python -m optimizer_genetic.py

