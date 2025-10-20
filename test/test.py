#!/usr/bin/env python3
"""
K-means Test Runner
Runs all permutations of point and centroid files
"""

import os
import subprocess
from pathlib import Path

# Configuration
TEST_DIR = Path("test")
POINT_DIR = TEST_DIR / "point"
CENTROID_DIR = TEST_DIR / "centroid"
RESULTS_DIR = TEST_DIR / "results"
BINARY = Path("build/bin/sequential")

MAX_ITER = 50
EPSILON = 0.000001

def main():
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all point files
    point_files = sorted(POINT_DIR.glob("*.txt"))
    if not point_files:
        print(f"No point files found in {POINT_DIR}")
        return
    
    # Get all centroid files
    centroid_files = sorted(CENTROID_DIR.glob("*.txt"))
    if not centroid_files:
        print(f"No centroid files found in {CENTROID_DIR}")
        return
    
    # Match files by index
    num_tests = min(len(point_files), len(centroid_files))
    print(f"Running {num_tests} tests...\n")
    
    for i in range(num_tests):
        point_file = point_files[i]
        centroid_file = centroid_files[i]
        
        point_name = point_file.stem
        centroid_name = centroid_file.stem
        test_name = f"{point_name}_{centroid_name}"
        
        # Create result directory
        result_dir = RESULTS_DIR / test_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare output files
        assign_file = result_dir / "assign.txt"
        output_centroid_file = result_dir / "centroids.txt"
        
        # Run the binary
        cmd = [
            str(BINARY),
            str(point_file),
            str(centroid_file),
            str(MAX_ITER),
            str(EPSILON),
            str(assign_file),
            str(output_centroid_file)
        ]
        
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
