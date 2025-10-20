#!/usr/bin/env python3
"""
K-means Performance Analysis
Analyzes results with statistical consolidation and generates performance graphs
"""
import re
import json
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configuration
RESULTS_DIR = Path("test/results")
MEASUREMENTS_DIR = Path("test/measurements")

def parse_output_file(output_file):
    """Parse the output.txt file to extract metrics"""
    try:
        with open(output_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return None
    
    data = {}
    
    # Extract N, K, max_iter, eps, threads
    match = re.search(r'N=(\d+)\s+K=(\d+)\s+max_iter=(\d+)\s+eps=([\d.e-]+)\s+threads=(\d+)', content)
    if match:
        data['N'] = int(match.group(1))
        data['K'] = int(match.group(2))
        data['max_iter'] = int(match.group(3))
        data['eps'] = float(match.group(4))
        data['threads'] = int(match.group(5))
    else:
        print(f"Warning: Could not parse configuration from {output_file}")
        print(f"Content preview: {content[:200]}")
        return None
    
    # Extract iterations, SSE, time
    match = re.search(r'Iterações:\s*(\d+)\s*\|\s*SSE final:\s*([\d.]+)\s*\|\s*Tempo:\s*([\d.]+)\s*ms', content)
    if match:
        data['iterations'] = int(match.group(1))
        data['sse'] = float(match.group(2))
        data['time_ms'] = float(match.group(3))
    else:
        print(f"Warning: Could not parse metrics from {output_file}")
        print(f"Content preview: {content[:200]}")
        return None
    
    return data

def collect_results():
    """Collect all results from sequential and openmp folders"""
    results = {
        'sequential': {},
        'openmp': {}
    }
    
    # Collect sequential results
    seq_dir = RESULTS_DIR / "sequential"
    if seq_dir.exists():
        for test_dir in seq_dir.iterdir():
            if test_dir.is_dir():
                output_file = test_dir / "output.txt"
                if output_file.exists():
                    data = parse_output_file(output_file)
                    if data:
                        test_name = test_dir.name
                        results['sequential'][test_name] = data
                    else:
                        print(f"Skipping {test_dir.name} - invalid output format")
    
    # Collect OpenMP results
    omp_dir = RESULTS_DIR / "openmp"
    if omp_dir.exists():
        for test_dir in omp_dir.iterdir():
            if test_dir.is_dir():
                output_file = test_dir / "output.txt"
                if output_file.exists():
                    data = parse_output_file(output_file)
                    if data:
                        # Extract base test name and threads
                        test_name = test_dir.name
                        match = re.search(r'(.+)_threads(\d+)', test_name)
                        if match:
                            base_name = match.group(1)
                            threads = int(match.group(2))
                            
                            if base_name not in results['openmp']:
                                results['openmp'][base_name] = {}
                            results['openmp'][base_name][threads] = data
                        else:
                            print(f"Warning: Could not extract thread count from {test_name}")
                    else:
                        print(f"Skipping {test_dir.name} - invalid output format")
    
    return results

def consolidate_sequential_runs(results):
    """Consolidate multiple runs of the same test using median and statistics"""
    consolidated = {}
    
    # Group runs by base test name (remove _run1, _run2, etc.)
    test_groups = {}
    for test_name, data in results['sequential'].items():
        # Extract base name (e.g., "0_0" from "0_0_run1")
        match = re.match(r'(.+?)_run\d+$', test_name)
        if match:
            base_name = match.group(1)
        else:
            base_name = test_name
        
        if base_name not in test_groups:
            test_groups[base_name] = []
        test_groups[base_name].append(data)
    
    # Calculate statistics for each group
    for base_name, runs in test_groups.items():
        if len(runs) == 1:
            consolidated[base_name] = runs[0]
            consolidated[base_name]['stats'] = {
                'runs': 1,
                'time_median': runs[0]['time_ms'],
                'time_mean': runs[0]['time_ms'],
                'time_std': 0.0,
                'time_min': runs[0]['time_ms'],
                'time_max': runs[0]['time_ms']
            }
        else:
            times = [run['time_ms'] for run in runs]
            sses = [run['sse'] for run in runs]
            
            # Use median time as representative
            median_time = statistics.median(times)
            
            # Find the run closest to median time
            median_idx = min(range(len(times)), key=lambda i: abs(times[i] - median_time))
            
            consolidated[base_name] = runs[median_idx].copy()
            consolidated[base_name]['stats'] = {
                'runs': len(runs),
                'time_median': median_time,
                'time_mean': statistics.mean(times),
                'time_std': statistics.stdev(times) if len(times) > 1 else 0.0,
                'time_min': min(times),
                'time_max': max(times),
                'sse_mean': statistics.mean(sses),
                'sse_std': statistics.stdev(sses) if len(sses) > 1 else 0.0
            }
            
            # Update time_ms to use median
            consolidated[base_name]['time_ms'] = median_time
    
    return consolidated

def calculate_speedup(sequential_results, openmp_results):
    """Calculate speedup for each test"""
    speedup_data = {}
    
    for test_name in openmp_results:
        if test_name not in sequential_results:
            continue
        
        seq_time = sequential_results[test_name]['time_ms']
        speedup_data[test_name] = {
            'sequential_time': seq_time,
            'sequential_stats': sequential_results[test_name].get('stats', {}),
            'threads': [],
            'speedup': [],
            'efficiency': [],
            'time_ms': []
        }
        
        for threads in sorted(openmp_results[test_name].keys()):
            omp_time = openmp_results[test_name][threads]['time_ms']
            speedup = seq_time / omp_time if omp_time > 0 else 0
            efficiency = (speedup / threads) * 100 if threads > 0 else 0
            
            speedup_data[test_name]['threads'].append(threads)
            speedup_data[test_name]['speedup'].append(speedup)
            speedup_data[test_name]['efficiency'].append(efficiency)
            speedup_data[test_name]['time_ms'].append(omp_time)
    
    return speedup_data

def validate_sse(sequential_results, openmp_results):
    """Validate that SSE values are consistent across runs"""
    validation = {}
    
    for test_name in openmp_results:
        if test_name not in sequential_results:
            continue
        
        seq_sse = sequential_results[test_name]['sse']
        validation[test_name] = {
            'sequential_sse': seq_sse,
            'openmp_sse': {},
            'valid': True
        }
        
        for threads in sorted(openmp_results[test_name].keys()):
            omp_sse = openmp_results[test_name][threads]['sse']
            validation[test_name]['openmp_sse'][threads] = omp_sse
            
            # Check if SSE is approximately equal (allowing small floating point differences)
            if abs(seq_sse - omp_sse) > 0.01:  # More lenient threshold
                validation[test_name]['valid'] = False
    
    return validation

def plot_speedup(speedup_data, output_dir):
    """Generate speedup plots"""
    for test_name, data in speedup_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        threads = data['threads']
        speedup = data['speedup']
        efficiency = data['efficiency']
        
        # Speedup plot
        ax1.plot(threads, speedup, 'b-o', linewidth=2, markersize=8, label='Actual Speedup')
        ax1.plot(threads, threads, 'r--', linewidth=1, label='Ideal (Linear) Speedup')
        ax1.set_xlabel('Number of Threads', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title(f'Speedup - {test_name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(threads)
        
        # Add speedup values on points
        for t, s in zip(threads, speedup):
            ax1.annotate(f'{s:.2f}x', xy=(t, s), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)
        
        # Efficiency plot
        ax2.plot(threads, efficiency, 'g-s', linewidth=2, markersize=8)
        ax2.axhline(y=100, color='r', linestyle='--', linewidth=1, label='100% Efficiency')
        ax2.set_xlabel('Number of Threads', fontsize=12)
        ax2.set_ylabel('Efficiency (%)', fontsize=12)
        ax2.set_title(f'Parallel Efficiency - {test_name}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(threads)
        ax2.set_ylim(0, max(110, max(efficiency) + 10))
        
        # Add efficiency values on points
        for t, e in zip(threads, efficiency):
            ax2.annotate(f'{e:.1f}%', xy=(t, e), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'speedup_{test_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: speedup_{test_name}.png")

def plot_execution_time(speedup_data, sequential_results, output_dir):
    """Generate execution time comparison plots"""
    for test_name, data in speedup_data.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        threads = [1] + data['threads']
        times = [data['sequential_time']] + data['time_ms']
        
        colors = ['red'] + ['blue'] * len(data['threads'])
        bars = ax.bar(range(len(threads)), times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, time) in enumerate(zip(bars, times)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.3f}ms',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Threads', fontsize=12)
        ax.set_ylabel('Execution Time (ms)', fontsize=12)
        ax.set_title(f'Execution Time Comparison - {test_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(threads)))
        ax.set_xticklabels([f'{t}\n(Sequential)' if t == 1 else str(t) for t in threads])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics if available
        if test_name in sequential_results and 'stats' in sequential_results[test_name]:
            stats = sequential_results[test_name]['stats']
            if stats['runs'] > 1:
                info_text = f"Sequential: {stats['runs']} runs, median={stats['time_median']:.3f}ms, "
                info_text += f"std={stats['time_std']:.3f}ms"
                ax.text(0.5, 0.98, info_text, transform=ax.transAxes,
                       ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / f'time_{test_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Generated: time_{test_name}.png")

def plot_overall_comparison(speedup_data, output_dir):
    """Generate overall comparison plot across all test cases"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get all test names and sort them
    test_names = sorted(speedup_data.keys())
    colors = plt.cm.tab10(range(len(test_names)))
    
    # Plot 1: Speedup comparison
    for idx, test_name in enumerate(test_names):
        data = speedup_data[test_name]
        ax1.plot(data['threads'], data['speedup'], 'o-', linewidth=2, 
                markersize=8, label=test_name, color=colors[idx])
    
    # Add ideal speedup line
    max_threads = max(max(data['threads']) for data in speedup_data.values())
    ideal_threads = list(range(1, max_threads + 1))
    ax1.plot(ideal_threads, ideal_threads, 'k--', linewidth=1.5, alpha=0.5, label='Ideal Speedup')
    
    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Speedup Comparison Across Test Cases', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Efficiency comparison
    for idx, test_name in enumerate(test_names):
        data = speedup_data[test_name]
        ax2.plot(data['threads'], data['efficiency'], 's-', linewidth=2,
                markersize=8, label=test_name, color=colors[idx])
    
    ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='100% Efficiency')
    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Parallel Efficiency Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Generated: overall_comparison.png")

def generate_summary_report(sequential_results, openmp_results, speedup_data, validation, output_dir):
    """Generate a summary report"""
    report_file = output_dir / "summary_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("K-MEANS PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        f.write("TESTS ANALYZED\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of test cases: {len(speedup_data)}\n")
        all_threads = sorted(set([t for data in speedup_data.values() for t in data['threads']]))
        f.write(f"Thread configurations tested: {all_threads}\n\n")
        
        # Sequential run statistics
        f.write("SEQUENTIAL RUN CONSOLIDATION\n")
        f.write("-" * 80 + "\n")
        for test_name in sorted(sequential_results.keys()):
            stats = sequential_results[test_name].get('stats', {})
            runs = stats.get('runs', 1)
            f.write(f"\n{test_name}:\n")
            f.write(f"  Runs: {runs}\n")
            if runs > 1:
                f.write(f"  Time - Median: {stats['time_median']:.6f}ms, "
                       f"Mean: {stats['time_mean']:.6f}ms, Std: {stats['time_std']:.6f}ms\n")
                f.write(f"  Time - Min: {stats['time_min']:.6f}ms, Max: {stats['time_max']:.6f}ms\n")
                f.write(f"  SSE - Mean: {stats.get('sse_mean', 0):.10f}, "
                       f"Std: {stats.get('sse_std', 0):.2e}\n")
        
        f.write("\n\n")
        
        # Detailed results for each test
        for test_name in sorted(speedup_data.keys()):
            f.write("=" * 80 + "\n")
            f.write(f"TEST: {test_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Test configuration
            if test_name in sequential_results:
                seq_data = sequential_results[test_name]
                f.write(f"Configuration: N={seq_data.get('N', 'N/A')}, K={seq_data.get('K', 'N/A')}, "
                       f"max_iter={seq_data.get('max_iter', 'N/A')}, eps={seq_data.get('eps', 'N/A')}\n")
                f.write(f"Iterations to converge: {seq_data.get('iterations', 'N/A')}\n")
                f.write(f"Final SSE: {seq_data.get('sse', 0):.10f}\n\n")
            else:
                f.write("Warning: No sequential data found for this test\n\n")
            
            # SSE Validation
            if test_name in validation:
                f.write("SSE VALIDATION:\n")
                if validation[test_name]['valid']:
                    f.write("  ✓ PASSED - All SSE values match (within tolerance)\n")
                else:
                    f.write("  ✗ FAILED - SSE values differ significantly!\n")
                f.write(f"  Sequential SSE: {validation[test_name]['sequential_sse']:.10f}\n")
                for threads, sse in sorted(validation[test_name]['openmp_sse'].items()):
                    diff = abs(sse - validation[test_name]['sequential_sse'])
                    f.write(f"  OpenMP ({threads:2d} threads) SSE: {sse:.10f} (diff: {diff:.2e})\n")
                f.write("\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"{'Threads':<10} {'Time (ms)':<15} {'Speedup':<12} {'Efficiency (%)':<15}\n")
            f.write("-" * 60 + "\n")
            
            seq_time = speedup_data[test_name]['sequential_time']
            f.write(f"{'1 (Seq)':<10} {seq_time:<15.6f} {'1.00':<12} {'100.00':<15}\n")
            
            for i, threads in enumerate(speedup_data[test_name]['threads']):
                time_ms = speedup_data[test_name]['time_ms'][i]
                speedup = speedup_data[test_name]['speedup'][i]
                efficiency = speedup_data[test_name]['efficiency'][i]
                f.write(f"{threads:<10} {time_ms:<15.6f} {speedup:<12.2f} {efficiency:<15.2f}\n")
            
            f.write("\n")
            
            # Best speedup
            if speedup_data[test_name]['speedup']:
                max_speedup_idx = speedup_data[test_name]['speedup'].index(max(speedup_data[test_name]['speedup']))
                best_threads = speedup_data[test_name]['threads'][max_speedup_idx]
                best_speedup = speedup_data[test_name]['speedup'][max_speedup_idx]
                best_efficiency = speedup_data[test_name]['efficiency'][max_speedup_idx]
                f.write(f"Best speedup: {best_speedup:.2f}x with {best_threads} threads "
                       f"(efficiency: {best_efficiency:.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"  Generated: summary_report.txt")

def save_json_data(sequential_results, openmp_results, speedup_data, validation, output_dir):
    """Save all data in JSON format for further analysis"""
    data = {
        'results': {
            'sequential': sequential_results,
            'openmp': openmp_results
        },
        'speedup': speedup_data,
        'validation': validation
    }
    
    json_file = output_dir / "analysis_data.json"
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Generated: analysis_data.json")

def main():
    # Create measurements directory
    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("K-MEANS PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print("\n[1/7] Collecting results...")
    raw_results = collect_results()
    
    if not raw_results['sequential'] or not raw_results['openmp']:
        print("ERROR: No results found. Please run tests first.")
        return
    
    print(f"  Found {len(raw_results['sequential'])} sequential test runs")
    print(f"  Found {len(raw_results['openmp'])} OpenMP test groups")
    
    print("\n[2/7] Consolidating sequential runs (calculating median and statistics)...")
    sequential_results = consolidate_sequential_runs(raw_results)
    print(f"  Consolidated into {len(sequential_results)} unique test cases")
    for test_name, data in sequential_results.items():
        runs = data.get('stats', {}).get('runs', 1)
        if runs > 1:
            print(f"    {test_name}: {runs} runs consolidated")
    
    print("\n[3/7] Calculating speedup metrics...")
    speedup_data = calculate_speedup(sequential_results, raw_results['openmp'])
    print(f"  Calculated speedup for {len(speedup_data)} test cases")
    
    print("\n[4/7] Validating SSE values...")
    validation = validate_sse(sequential_results, raw_results['openmp'])
    all_valid = all(v['valid'] for v in validation.values())
    if all_valid:
        print("  ✓ All tests passed SSE validation")
    else:
        failed = [name for name, v in validation.items() if not v['valid']]
        print(f"  ✗ {len(failed)} test(s) failed SSE validation: {', '.join(failed)}")
    
    print("\n[5/7] Generating individual test plots...")
    plot_speedup(speedup_data, MEASUREMENTS_DIR)
    plot_execution_time(speedup_data, sequential_results, MEASUREMENTS_DIR)
    
    print("\n[6/7] Generating overall comparison plot...")
    plot_overall_comparison(speedup_data, MEASUREMENTS_DIR)
    
    print("\n[7/7] Generating reports...")
    generate_summary_report(sequential_results, raw_results['openmp'], 
                          speedup_data, validation, MEASUREMENTS_DIR)
    save_json_data(sequential_results, raw_results['openmp'], 
                  speedup_data, validation, MEASUREMENTS_DIR)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {MEASUREMENTS_DIR}/")
    print("\nGenerated files:")
    print("  - speedup_*.png: Speedup and efficiency plots for each test")
    print("  - time_*.png: Execution time comparison plots")
    print("  - overall_comparison.png: Comparison across all test cases")
    print("  - summary_report.txt: Detailed text report with statistics")
    print("  - analysis_data.json: Raw data in JSON format")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Test':<20} {'Best Speedup':<15} {'Threads':<10} {'Efficiency':<15}")
    print("-" * 80)
    for test_name in sorted(speedup_data.keys()):
        data = speedup_data[test_name]
        if data['speedup']:
            max_idx = data['speedup'].index(max(data['speedup']))
            best_speedup = data['speedup'][max_idx]
            best_threads = data['threads'][max_idx]
            best_eff = data['efficiency'][max_idx]
            print(f"{test_name:<20} {best_speedup:>8.2f}x      {best_threads:<10} {best_eff:>8.2f}%")

if __name__ == "__main__":
    main()