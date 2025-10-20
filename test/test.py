#!/usr/bin/env python3
"""
K-means Test Runner
Runs all permutations of point and centroid files for both sequential and OpenMP versions
and repeats each test multiple times.
"""
import os
import subprocess
from pathlib import Path

# Configuration
TEST_DIR = Path("test")
POINT_DIR = TEST_DIR / "point"
CENTROID_DIR = TEST_DIR / "centroid"
RESULTS_DIR = TEST_DIR / "results"
BINARY_SEQ = Path("build/bin/sequential")
BINARY_OMP = Path("build/bin/openmp")
MAX_ITER = 50
EPSILON = 0.000001
NUM_THREADS = [1, 2, 4, 8, 16]  # Thread counts for OpenMP
NUM_RUNS = 20  # Number of times to repeat each test

def run_sequential_tests(point_files, centroid_files):
    print("=" * 60)
    print("RUNNING SEQUENTIAL TESTS")
    print("=" * 60)
    
    results_seq_dir = RESULTS_DIR / "sequential"
    results_seq_dir.mkdir(parents=True, exist_ok=True)
    
    num_tests = min(len(point_files), len(centroid_files))
    
    for i in range(num_tests):
        point_file = point_files[i]
        centroid_file = centroid_files[i]
        point_name = point_file.stem
        centroid_name = centroid_file.stem
        test_name = f"{point_name}_{centroid_name}"
        
        for run in range(1, NUM_RUNS + 1):
            print(f"\nTest {i+1}/{num_tests} Run {run}/{NUM_RUNS}: {test_name}")
            
            result_dir = results_seq_dir / f"{test_name}_run{run}"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            assign_file = result_dir / "assign.txt"
            output_centroid_file = result_dir / "centroids.txt"
            
            cmd = [
                str(BINARY_SEQ),
                str(point_file),
                str(centroid_file),
                str(MAX_ITER),
                str(EPSILON),
                str(assign_file),
                str(output_centroid_file)
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            output_file = result_dir / "output.txt"
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(result.stderr)
            
            if result.returncode != 0:
                print(f"  ERROR: Test failed with return code {result.returncode}")
            else:
                print(f"  ✓ Completed successfully")

def run_openmp_tests(point_files, centroid_files):
    print("\n" + "=" * 60)
    print("RUNNING OPENMP TESTS")
    print("=" * 60)
    
    results_omp_dir = RESULTS_DIR / "openmp"
    results_omp_dir.mkdir(parents=True, exist_ok=True)
    
    num_tests = min(len(point_files), len(centroid_files))
    
    for threads in NUM_THREADS:
        print(f"\n--- Testing with {threads} threads ---")
        for i in range(num_tests):
            point_file = point_files[i]
            centroid_file = centroid_files[i]
            point_name = point_file.stem
            centroid_name = centroid_file.stem
            test_name = f"{point_name}_{centroid_name}_threads{threads}"
            
            for run in range(1, NUM_RUNS + 1):
                print(f"\nTest {i+1}/{num_tests} Run {run}/{NUM_RUNS}: {test_name}")
                
                result_dir = results_omp_dir / f"{test_name}_run{run}"
                result_dir.mkdir(parents=True, exist_ok=True)
                
                assign_file = result_dir / "assign.txt"
                output_centroid_file = result_dir / "centroids.txt"
                
                cmd = [
                    str(BINARY_OMP),
                    str(point_file),
                    str(centroid_file),
                    str(MAX_ITER),
                    str(EPSILON),
                    str(assign_file),
                    str(output_centroid_file),
                    str(threads)
                ]
                
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                output_file = result_dir / "output.txt"
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n=== STDERR ===\n")
                        f.write(result.stderr)
                
                if result.returncode != 0:
                    print(f"  ERROR: Test failed with return code {result.returncode}")
                else:
                    print(f"  ✓ Completed successfully")

def main():
    if not BINARY_SEQ.exists():
        print(f"ERROR: Sequential binary not found at {BINARY_SEQ}")
        return
    if not BINARY_OMP.exists():
        print(f"ERROR: OpenMP binary not found at {BINARY_OMP}")
        return
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    point_files = sorted(POINT_DIR.glob("*.txt"))
    centroid_files = sorted(CENTROID_DIR.glob("*.txt"))
    
    if not point_files or not centroid_files:
        print("No point or centroid files found.")
        return
    
    print(f"Found {min(len(point_files), len(centroid_files))} test pairs")
    
    run_sequential_tests(point_files, centroid_files)
    run_openmp_tests(point_files, centroid_files)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print(f"\nResults saved in:")
    print(f"  Sequential: {RESULTS_DIR / 'sequential'}")
    print(f"  OpenMP: {RESULTS_DIR / 'openmp'}")

if __name__ == "__main__":
    main()
