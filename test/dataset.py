#!/usr/bin/env python3
import os
import numpy as np

def write_txt(filename, values):
    np.savetxt(filename, np.array(values, dtype=np.float64), fmt="%.6f")

def make_dataset(name, N, K, clusters, stddev, seed):
    np.random.seed(seed)
    points = np.concatenate([
        np.random.normal(mu, stddev, N // K) for mu in clusters
    ])
    write_txt(f"point/{name}.txt", points)
    write_txt(f"centroid/{name}.txt", clusters)
    print(f"Generated {name:<7}: N={N:>7} K={K:<2} σ={stddev:<4} → points_{name}.txt")

make_dataset(name="small",  N=10_000,  K=4,  clusters=[0,10,20,30],    stddev=0.8, seed=42)
make_dataset(name="medium", N=100_000, K=8,  clusters=[i*10 for i in range(8)], stddev=1.0, seed=123)
make_dataset(name="large",  N=1_000_000, K=16, clusters=[i*10 for i in range(16)], stddev=1.5, seed=999)

print("\nAll datasets generated in ./data/")
for f in sorted(os.listdir("data")):
    print(" -", f)
