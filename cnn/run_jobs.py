import concurrent.futures
import os
import subprocess
import sys
import uuid

import numpy as np

NUM_GPUS = 2
GPU_INDICES = [1, 2]
max_workers_per_gpu = 1 # num models per GPU

NUM_GENOTYPES=50
RESULTS_DIR = "/home/john/nas/darts/cnn/similarity_experiments"

assert len(GPU_INDICES) == NUM_GPUS



def launch_jobs(gpu_idx, genotype_idx):
    env = os.environ
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDICES[gpu_idx])

    command = "python train.py --auxiliary --cutout".split()
    command.extend(["--arch", f"DARTS_GENOTYPE_{genotype_idx}"])
    
    savedir = os.path.join(RESULTS_DIR, f"darts_genotype_{genotype_idx}")
    command.extend(["--save", savedir])

    subprocess.call(command, env=env)


with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GPUS * max_workers_per_gpu) as executor:
    for genotype_idx in range(0, 5):
        executor.submit(launch_jobs, genotype_idx % NUM_GPUS, genotype_idx)
