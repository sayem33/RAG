"""
Quick Statistics Analyzer for APUOPE-RE Test Results

Analyzes rag_results.json and generates comprehensive statistics.

Usage:
    python quick_stats.py [--output stats_report.txt]
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def load_results(filename: str = "rag_results.json") -> List[Dict]:
    """Load test results from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        print("Run test_runner.py first to generate results.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {filename}: {e}")
        sys.exit(1)


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate comprehensive statistics from results."""
    
    stats = {
        'total_tests': len(results),
        'succeeded': 0,
        'failed': 0,
        'by_task_type': defaultdict(lambda: {
            'count': 0,
            'succeeded': 0,
            'failed': 0,
            'latencies': [],
            'errors': [],
            'llm_scores': [],
            'automated_metrics': []
        }),
        'by_material': defaultdict(lambda: {
            'count': 0,
            'succeeded': 0,
            'failed': 0,
            'latencies': []
        }),
        'total_latency': 0.0,
        'latencies': [],
        'errors': []
    }
    
    for result in results:
        task_type = result.get('task_type', 'unknown')
        material_id = result.get('material_id', 'unknown')
        error = result.get('error')
        latency = result.get('latency_seconds', 0.0)
        
        # Overall stats
        stats['latencies'].append(latency)
        stats['total_latency'] += latency
        
        if error:
            stats['failed'] += 1
            stats['errors'].append({
                'test_id': result.get('test_id'),
                'task_type': task_type,
                'error': error
            })
        else:
            stats['succeeded'] += 1
        
        # By task type
        task_stats = stats['by_task_type'][task_type]
        task_stats['count'] += 1
        task_stats['latencies'].append(latency)
        if error:
            task_stats['failed'] += 1
            task_stats['errors'].append({
                'test_id': result.get('test_id'),
                'error': error
            })
        else:
            task_stats['succeeded'] += 1
            
            # Collect evaluation scores
            llm_eval = result.get('llm_evaluation', {})
            if llm_eval and 'scores' in llm_eval:
                overall_score = llm_eval['scores'].get('overall')
                if overall_score is not None:
                    task_stats['llm_scores'].append(overall_score)
            
            # Collect automated metrics
            auto_metrics = result.get('automated_metrics', {})
            if auto_metrics:
                task_stats['automated_metrics'].append(auto_metrics)
        
        # By material
        material_stats = stats['by_material'][material_id]
        material_stats['count'] += 1
        material_stats['latencies'].append(latency)
        if error:
            material_stats['failed'] += 1
        else:
            material_stats['succeeded'] += 1
    
    return stats


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate percentiles for a list of values."""
    if not values:
        return {'min': 0, 'p50': 0, 'p95': 0, 'p99': 0, 'max': 0}
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    return {
        'min': sorted_vals[0],
        'p50': sorted_vals[n // 2],
        'p95': sorted_vals[int(n * 0.95)] if n > 0 else sorted_vals[-1],
        'p99': sorted_vals[int(n * 0.99)] if n > 0 else sorted_vals[-1],
        'max': sorted_vals[-1]
    }


def print_report(stats: Dict, output_file: str = None):
    """Print comprehensive statistics report."""
    
    output_lines = []
    
    def print_line(line: str = ""):
        """Helper to print to both console and file."""
        print(line)
        output_lines.append(line)
    
    print_line("=" * 80)
    print_line("APUOPE-RE TEST RESULTS - STATISTICS REPORT")
    print_line("=" * 80)
    print_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_line()
    
    # Overall Summary
    print_line("📊 OVERALL SUMMARY")
    print_line("─" * 80)
    total = stats['total_tests']
    succeeded = stats['succeeded']
    failed = stats['failed']
    success_rate = (succeeded / total * 100) if total > 0 else 0
    
    print_line(f"Total Tests:          {total}")
    print_line(f"✅ Succeeded:         {succeeded} ({success_rate:.1f}%)")
    print_line(f"❌ Failed:            {failed} ({(failed/total*100) if total > 0 else 0:.1f}%)")
    print_line()
    
    # Latency Statistics
    print_line("⏱️  LATENCY STATISTICS")
    print_line("─" * 80)
    percentiles = calculate_percentiles(stats['latencies'])
    avg_latency = stats['total_latency'] / total if total > 0 else 0
    
    print_line(f"Total Time:           {stats['total_latency']:.2f} seconds")
    print_line(f"Average:              {avg_latency:.2f} seconds/test")
    print_line(f"Min:                  {percentiles['min']:.2f}s")
    print_line(f"Median (p50):         {percentiles['p50']:.2f}s")
    print_line(f"95th percentile:      {percentiles['p95']:.2f}s")
    print_line(f"99th percentile:      {percentiles['p99']:.2f}s")
    print_line(f"Max:                  {percentiles['max']:.2f}s")
    print_line()
    
    # By Task Type
    print_line("📋 STATISTICS BY TASK TYPE")
    print_line("─" * 80)
    
    for task_type in sorted(stats['by_task_type'].keys()):
        task_stats = stats['by_task_type'][task_type]
        count = task_stats['count']
        succeeded = task_stats['succeeded']
        failed = task_stats['failed']
        success_rate = (succeeded / count * 100) if count > 0 else 0
        
        avg_latency = sum(task_stats['latencies']) / count if count > 0 else 0
        task_percentiles = calculate_percentiles(task_stats['latencies'])
        
        print_line(f"\n{task_type.upper()}")
        print_line(f"  Total Tests:        {count}")
        print_line(f"  Success Rate:       {succeeded}/{count} ({success_rate:.1f}%)")
        print_line(f"  Failed:             {failed}")
        print_line(f"  Avg Latency:        {avg_latency:.2f}s")
        print_line(f"  Latency Range:      {task_percentiles['min']:.2f}s - {task_percentiles['max']:.2f}s")
        print_line(f"  Median Latency:     {task_percentiles['p50']:.2f}s")
        
        # LLM evaluation scores
        if task_stats['llm_scores']:
            avg_score = sum(task_stats['llm_scores']) / len(task_stats['llm_scores'])
            min_score = min(task_stats['llm_scores'])
            max_score = max(task_stats['llm_scores'])
            print_line(f"  LLM Score (avg):    {avg_score:.2f}/10")
            print_line(f"  LLM Score Range:    {min_score:.1f} - {max_score:.1f}")
        
        # Automated metrics summary
        if task_stats['automated_metrics']:
            # Average word F1
            f1_scores = [m.get('word_f1', 0) for m in task_stats['automated_metrics']]
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            print_line(f"  Word F1 (avg):      {avg_f1:.3f}")
    
    print_line()
    
    # By Material/Slide
    print_line("📄 STATISTICS BY MATERIAL/SLIDE")
    print_line("─" * 80)
    
    for material_id in sorted(stats['by_material'].keys()):
        material_stats = stats['by_material'][material_id]
        count = material_stats['count']
        succeeded = material_stats['succeeded']
        failed = material_stats['failed']
        success_rate = (succeeded / count * 100) if count > 0 else 0
        
        avg_latency = sum(material_stats['latencies']) / count if count > 0 else 0
        
        print_line(f"\n{material_id}")
        print_line(f"  Tests:              {count}")
        print_line(f"  Success:            {succeeded}/{count} ({success_rate:.1f}%)")
        print_line(f"  Failed:             {failed}")
        print_line(f"  Avg Latency:        {avg_latency:.2f}s")
    
    print_line()
    
    # Error Analysis
    if stats['errors']:
        print_line("❌ ERROR ANALYSIS")
        print_line("─" * 80)
        print_line(f"Total Errors: {len(stats['errors'])}\n")
        
        # Group errors by type
        error_types = defaultdict(list)
        for error_info in stats['errors']:
            error_msg = error_info['error']
            # Extract error type (first line or first 50 chars)
            error_type = error_msg.split('\n')[0][:80] if error_msg else "Unknown"
            error_types[error_type].append(error_info)
        
        print_line("Error Types:")
        for error_type, occurrences in sorted(error_types.items(), key=lambda x: -len(x[1])):
            print_line(f"\n  [{len(occurrences)}x] {error_type}")
            print_line(f"  Affected Tests: {', '.join([e['test_id'] for e in occurrences[:3]])}")
            if len(occurrences) > 3:
                print_line(f"  ... and {len(occurrences) - 3} more")
        
        print_line()
    
    # Task Type Distribution
    print_line("📊 TASK TYPE DISTRIBUTION")
    print_line("─" * 80)
    task_counts = [(task, stats['by_task_type'][task]['count']) 
                   for task in stats['by_task_type'].keys()]
    task_counts.sort(key=lambda x: -x[1])
    
    for task_type, count in task_counts:
        percentage = (count / total * 100) if total > 0 else 0
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        print_line(f"{task_type:20s} [{count:3d}] {bar} {percentage:.1f}%")
    
    print_line()
    print_line("=" * 80)
    
    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            print(f"\n📄 Report saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save report to file: {e}")


def analyze_output_quality(results: List[Dict]) -> Dict:
    """Analyze the quality of generated outputs."""
    
    quality_stats = {
        'empty_outputs': 0,
        'short_outputs': 0,  # Less than 50 chars
        'long_outputs': 0,   # More than 1000 chars
        'avg_output_length': 0,
        'by_task_type': defaultdict(lambda: {
            'avg_length': 0,
            'lengths': []
        })
    }
    
    total_length = 0
    valid_outputs = 0
    
    for result in results:
        output = result.get('generated_output')
        task_type = result.get('task_type', 'unknown')
        
        if output is None or result.get('error'):
            quality_stats['empty_outputs'] += 1
            continue
        
        # Convert to string for length calculation
        output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        length = len(output_str)
        
        total_length += length
        valid_outputs += 1
        
        if length < 50:
            quality_stats['short_outputs'] += 1
        elif length > 1000:
            quality_stats['long_outputs'] += 1
        
        quality_stats['by_task_type'][task_type]['lengths'].append(length)
    
    quality_stats['avg_output_length'] = total_length / valid_outputs if valid_outputs > 0 else 0
    
    # Calculate avg length per task type
    for task_type, task_data in quality_stats['by_task_type'].items():
        lengths = task_data['lengths']
        task_data['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
    
    return quality_stats


def print_quality_report(results: List[Dict]):
    """Print output quality analysis."""
    quality_stats = analyze_output_quality(results)
    
    print("\n" + "=" * 80)
    print("📝 OUTPUT QUALITY ANALYSIS")
    print("─" * 80)
    print(f"Empty/Error Outputs:  {quality_stats['empty_outputs']}")
    print(f"Short Outputs (<50):  {quality_stats['short_outputs']}")
    print(f"Long Outputs (>1000): {quality_stats['long_outputs']}")
    print(f"Avg Output Length:    {quality_stats['avg_output_length']:.0f} characters")
    
    print("\nAverage Output Length by Task Type:")
    for task_type in sorted(quality_stats['by_task_type'].keys()):
        avg_length = quality_stats['by_task_type'][task_type]['avg_length']
        print(f"  {task_type:20s}: {avg_length:.0f} chars")
    
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze APUOPE-RE test results')
    parser.add_argument('--input', type=str, default='rag_results.json',
                        help='Input results file (default: rag_results.json)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save report to file (optional)')
    parser.add_argument('--quality', action='store_true',
                        help='Include output quality analysis')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.input)
    
    if not results:
        print("No results found in the file.")
        return
    
    # Calculate and print statistics
    stats = calculate_statistics(results)
    print_report(stats, args.output)
    
    # Print quality analysis if requested
    if args.quality:
        print_quality_report(results)


if __name__ == "__main__":
    main()

