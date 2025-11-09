"""
Setup Validation Script for Test Runner

Validates that all dependencies and functions are working before running the full test suite.

Usage:
    python validate_setup.py
"""

import sys
import os

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 70)
    print("VALIDATING SETUP")
    print("=" * 70)
    print("\n1. Checking imports...")
    
    errors = []
    
    # Standard library
    try:
        import json
        import time
        from datetime import datetime
        print("   ✅ Standard library modules: OK")
    except Exception as e:
        errors.append(f"Standard library: {e}")
        print(f"   ❌ Standard library modules: FAILED - {e}")
    
    # OpenAI
    try:
        import openai
        print("   ✅ OpenAI library: OK")
    except Exception as e:
        errors.append(f"OpenAI: {e}")
        print(f"   ❌ OpenAI library: FAILED - {e}")
        print("      Install with: pip install openai==0.28.0")
    
    # Application modules
    try:
        from components.conceptual_examples import generate_content
        print("   ✅ components.conceptual_examples: OK")
    except Exception as e:
        errors.append(f"conceptual_examples: {e}")
        print(f"   ❌ components.conceptual_examples: FAILED - {e}")
        print("      Make sure you're running from the project root directory")
    
    try:
        from quiz_handler import generate_quiz
        print("   ✅ quiz_handler: OK")
    except Exception as e:
        errors.append(f"quiz_handler: {e}")
        print(f"   ❌ quiz_handler: FAILED - {e}")
    
    return len(errors) == 0, errors


def check_api_key():
    """Check if OpenAI API key is set."""
    print("\n2. Checking API key...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("   ❌ OPENAI_API_KEY not set")
        print("      Option 1 (Windows PowerShell): $env:OPENAI_API_KEY='sk-your-key-here'")
        print("      Option 2 (Windows CMD): set OPENAI_API_KEY=sk-your-key-here")
        print("      Option 3 (Linux/Mac): export OPENAI_API_KEY='sk-your-key-here'")
        print("      Option 4: Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if api_key == "API KEY":
        print("   ⚠️  OPENAI_API_KEY is set to placeholder value")
        print("      Set a real key with: export OPENAI_API_KEY='sk-...'")
        return False
    
    if not api_key.startswith("sk-"):
        print("   ⚠️  OPENAI_API_KEY doesn't look like a valid key (should start with 'sk-')")
        return False
    
    print(f"   ✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    return True


def check_test_dataset():
    """Check if test dataset file exists."""
    print("\n3. Checking test dataset...")
    
    if not os.path.exists("test_dataset_re_90.json"):
        print("   ❌ test_dataset_re_90.json not found")
        return False
    
    try:
        import json
        with open("test_dataset_re_90.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        num_slides = len(data)
        total_tests = sum(len(slide['test_cases']) for slide in data)
        
        print(f"   ✅ Dataset loaded: {num_slides} slides, {total_tests} test cases")
        
        # Count by task type
        task_types = {}
        for slide in data:
            for test_case in slide['test_cases']:
                task_type = test_case['task_type']
                task_types[task_type] = task_types.get(task_type, 0) + 1
        
        print("   📊 Task type distribution:")
        for task_type, count in sorted(task_types.items()):
            print(f"      - {task_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False


def test_functions():
    """Test if key functions work with sample data."""
    print("\n4. Testing key functions...")
    
    try:
        from components.conceptual_examples import generate_content
        from quiz_handler import generate_quiz
        
        # Test data
        test_content = "Requirements Engineering is about understanding and documenting what a system should do."
        
        # Test 1: generate_content (for summarization/QA)
        print("   Testing generate_content()...")
        try:
            result = generate_content("Summarize this in one sentence: " + test_content)
            if result and len(result) > 10:
                print(f"   ✅ generate_content works (returned {len(result)} chars)")
            else:
                print("   ⚠️  generate_content returned suspicious result")
                print(f"      Result: {result}")
        except Exception as e:
            print(f"   ❌ generate_content failed: {e}")
            return False
        
        # Test 2: generate_quiz
        print("   Testing generate_quiz()...")
        try:
            questions, answers = generate_quiz(test_content, "easy")
            if questions and answers:
                print(f"   ✅ generate_quiz works (returned {len(questions)} questions)")
            else:
                print("   ⚠️  generate_quiz returned empty results")
        except Exception as e:
            print(f"   ❌ generate_quiz failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Function test failed: {e}")
        return False


def check_write_permissions():
    """Check if we can write output files."""
    print("\n5. Checking write permissions...")
    
    try:
        test_file = "test_write_permission.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("   ✅ Write permissions: OK")
        return True
    except Exception as e:
        print(f"   ❌ Cannot write files: {e}")
        return False


def main():
    """Run all validation checks."""
    
    all_passed = True
    
    # Run checks
    imports_ok, import_errors = check_imports()
    all_passed = all_passed and imports_ok
    
    api_key_ok = check_api_key()
    all_passed = all_passed and api_key_ok
    
    dataset_ok = check_test_dataset()
    all_passed = all_passed and dataset_ok
    
    write_ok = check_write_permissions()
    all_passed = all_passed and write_ok
    
    # Only test functions if everything else passed
    if all_passed:
        functions_ok = test_functions()
        all_passed = all_passed and functions_ok
    else:
        print("\n4. Testing key functions... SKIPPED (fix errors above first)")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("=" * 70)
        print("\nYou're ready to run the test runner:")
        print("  1. Test with a small sample:")
        print("     python test_runner.py --limit 5")
        print("\n  2. Run full test suite:")
        print("     python test_runner.py")
        print("\n  3. Analyze results:")
        print("     python quick_stats.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print("\nPlease fix the errors above before running the test runner.")
        if import_errors:
            print("\nImport errors:")
            for error in import_errors:
                print(f"  - {error}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

