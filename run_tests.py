"""
Convenience wrapper to validate setup and run tests.

This script:
1. Validates the setup
2. Asks for confirmation
3. Runs the test runner
4. Displays statistics

Usage:
    python run_tests.py [--limit N]
"""

import sys
import subprocess
import argparse


def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate and run APUOPE-RE tests')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tests to run')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip validation step')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("APUOPE-RE Test Suite Runner")
    print("=" * 70)
    print()
    
    # Step 1: Validate setup (unless skipped)
    if not args.skip_validation:
        print("Step 1: Validating setup...")
        print("-" * 70)
        
        if not run_command("python validate_setup.py"):
            print("\n❌ Validation failed. Please fix the errors above.")
            return 1
        
        print()
        print("=" * 70)
        print()
    
    # Step 2: Confirm execution
    if args.limit:
        message = f"Ready to run {args.limit} tests. Continue? (y/n): "
    else:
        message = "Ready to run ALL 100 tests. This may take 15-30 minutes. Continue? (y/n): "
    
    try:
        response = input(message).lower().strip()
        if response not in ['y', 'yes']:
            print("Cancelled.")
            return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    
    print()
    print("=" * 70)
    print()
    
    # Step 3: Run tests
    print("Step 2: Running tests...")
    print("-" * 70)
    print()
    
    if args.limit:
        cmd = f"python test_runner.py --limit {args.limit}"
    else:
        cmd = "python test_runner.py"
    
    if not run_command(cmd):
        print("\n⚠️  Test runner encountered errors. Check output above.")
        print("Partial results may be available in existing_app_results.json")
        return 1
    
    print()
    print("=" * 70)
    print()
    
    # Step 4: Display statistics
    print("Step 3: Analyzing results...")
    print("-" * 70)
    print()
    
    if not run_command("python quick_stats.py"):
        print("\n⚠️  Could not generate statistics. Check if results file exists.")
        return 1
    
    print()
    print("=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
    print()
    print("Results saved to: existing_app_results.json")
    print()
    print("Next steps:")
    print("  - Review the statistics above")
    print("  - Check existing_app_results.json for detailed results")
    print("  - Run 'python quick_stats.py --quality' for quality analysis")
    print("  - Run 'python quick_stats.py --output report.txt' to save report")
    print()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)

