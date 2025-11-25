"""
Test Runner for APUOPE-RE Application
Evaluates the application using a 90-test benchmark dataset.

Usage:
    python test_runner.py [--limit N] [--start-from N]
    
Options:
    --limit N: Run only first N tests (for testing)
    --start-from N: Start from test N (for resuming)
"""

import json
import time
import os
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple
import openai
from dotenv import load_dotenv

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load environment variables from .env file
load_dotenv()

# Import application functions
from components.conceptual_examples import generate_content
from quiz_handler import generate_quiz

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not set!")
    print("   Please set it in your .env file or environment variable")
    sys.exit(1)
openai.api_key = api_key

# Configuration
TEST_DATASET_FILE = "test_dataset_re_90.json"
OUTPUT_FILE = "rag_results.json"
TEMP_OUTPUT_FILE = "rag_results_temp.json"

# Statistics tracking
stats = {
    "total_tests": 0,
    "succeeded": 0,
    "failed": 0,
    "by_task_type": {},
    "total_latency": 0.0,
    "start_time": None,
    "end_time": None
}


def load_test_dataset() -> List[Dict]:
    """Load the test dataset from JSON file."""
    try:
        with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {TEST_DATASET_FILE} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {TEST_DATASET_FILE}: {e}")
        sys.exit(1)


def load_existing_results() -> List[Dict]:
    """Load existing results if the output file exists."""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_results_incremental(results: List[Dict]):
    """Save results incrementally to both temp and final file."""
    # Save to temp file first
    with open(TEMP_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Copy to final file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# REMOVED: Grading function no longer needed (grading tests removed from dataset)
# def generate_grading_response(...): ...


def run_summarization_test(slide_content: str, instruction: str) -> str:
    """Run a summarization test."""
    prompt = instruction
    return generate_content(prompt, "test_slide", slide_content)


def run_quiz_generation_test(slide_content: str, constraints: Dict) -> Any:
    """Run a quiz generation test."""
    difficulty = constraints.get('difficulty', 'medium')
    num_questions = constraints.get('num_questions', 3)
    
    # Note: The original generate_quiz doesn't take num_questions
    # It generates based on difficulty level
    questions, answers = generate_quiz(slide_content, difficulty, "test_slide")
    
    # Format output as structured data
    return {
        "questions": questions,
        "answers": answers,
        "requested_num": num_questions,
        "generated_num": len(questions)
    }


def run_qa_test(slide_content: str, instruction: str) -> str:
    """Run a Q&A test (conceptual or application)."""
    prompt = instruction
    return generate_content(prompt, "test_slide", slide_content)


# REMOVED: Grading test handler no longer needed (grading tests removed from dataset)
# def run_grading_test(...): ...


def evaluate_with_llm_judge(generated_output: Any, reference_answer: str, task_type: str, instruction: str) -> Dict:
    """
    Use GPT-4 as a judge to evaluate the generated output.
    Returns scores and reasoning.
    """
    # Convert output to string if needed
    if isinstance(generated_output, dict):
        generated_str = json.dumps(generated_output, indent=2)
    else:
        generated_str = str(generated_output)
    
    prompt = f"""You are an expert evaluator for educational AI systems. Evaluate the following output.

Task Type: {task_type}
Instruction: {instruction}

Reference Answer:
{reference_answer}

Generated Output:
{generated_str}

Evaluate on these criteria (score each 0-10):
1. Correctness: Factual accuracy and alignment with reference
2. Completeness: Covers all required points
3. Clarity: Clear, well-structured, readable
4. Relevance: Stays on topic, addresses the question

Provide your evaluation in this EXACT format:
CORRECTNESS: [score]
COMPLETENESS: [score]
CLARITY: [score]
RELEVANCE: [score]
OVERALL: [average of above 4]
REASONING: [2-3 sentences explaining the scores]"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert educational content evaluator. Provide objective, consistent scores."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3  # Lower temperature for consistent evaluation
        )
        
        evaluation_text = response["choices"][0]["message"]["content"]
        
        # Parse scores
        scores = {}
        for criterion in ["CORRECTNESS", "COMPLETENESS", "CLARITY", "RELEVANCE", "OVERALL"]:
            match = re.search(f"{criterion}:\\s*(\\d+(?:\\.\\d+)?)", evaluation_text)
            if match:
                scores[criterion.lower()] = float(match.group(1))
        
        # Extract reasoning
        reasoning_match = re.search(r"REASONING:\\s*(.+?)(?:\n\n|$)", evaluation_text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else evaluation_text
        
        return {
            "scores": scores,
            "reasoning": reasoning,
            "raw_evaluation": evaluation_text
        }
        
    except Exception as e:
        return {
            "scores": {},
            "reasoning": f"Evaluation failed: {str(e)}",
            "raw_evaluation": None
        }


def calculate_automated_metrics(generated_output: Any, reference_answer: str, task_type: str) -> Dict:
    """
    Calculate automated metrics without external libraries.
    Uses simple but effective metrics.
    """
    # Convert to strings
    if isinstance(generated_output, dict):
        generated_str = json.dumps(generated_output)
    else:
        generated_str = str(generated_output).lower()
    
    reference_str = str(reference_answer).lower()
    
    metrics = {}
    
    # 1. Length ratio
    gen_len = len(generated_str.split())
    ref_len = len(reference_str.split())
    metrics['length_ratio'] = gen_len / ref_len if ref_len > 0 else 0
    
    # 2. Word overlap (simple precision/recall)
    gen_words = set(re.findall(r'\w+', generated_str))
    ref_words = set(re.findall(r'\w+', reference_str))
    
    overlap = gen_words.intersection(ref_words)
    metrics['word_precision'] = len(overlap) / len(gen_words) if gen_words else 0
    metrics['word_recall'] = len(overlap) / len(ref_words) if ref_words else 0
    metrics['word_f1'] = (2 * metrics['word_precision'] * metrics['word_recall'] / 
                          (metrics['word_precision'] + metrics['word_recall'])) if (metrics['word_precision'] + metrics['word_recall']) > 0 else 0
    
    # 3. Character-level similarity (simple Jaccard)
    gen_chars = set(generated_str)
    ref_chars = set(reference_str)
    char_overlap = gen_chars.intersection(ref_chars)
    char_union = gen_chars.union(ref_chars)
    metrics['char_jaccard'] = len(char_overlap) / len(char_union) if char_union else 0
    
    # 4. Task-specific metrics
    if task_type == 'quiz_generation':
        # Count questions generated
        if isinstance(generated_output, dict) and 'questions' in generated_output:
            metrics['num_questions'] = len(generated_output['questions'])
        else:
            metrics['num_questions'] = generated_str.count('?')
    
    return metrics


def execute_test(slide_content: str, test_case: Dict) -> Tuple[Any, float, str]:
    """
    Execute a single test case.
    
    Supported task types: summarization, quiz_generation, qa_conceptual, qa_application
    
    Returns:
        Tuple of (output, latency_seconds, error_message)
    """
    task_type = test_case['task_type']
    start_time = time.time()
    error = None
    output = None
    
    try:
        if task_type == 'summarization':
            output = run_summarization_test(slide_content, test_case['instruction'])
            
        elif task_type == 'quiz_generation':
            output = run_quiz_generation_test(slide_content, test_case.get('constraints', {}))
            
        elif task_type in ['qa_conceptual', 'qa_application']:
            output = run_qa_test(slide_content, test_case['instruction'])
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    except Exception as e:
        error = str(e)
        output = None
    
    latency = time.time() - start_time
    
    return output, latency, error


def format_reference_answer(reference_answer: Any) -> str:
    """Format reference answer for consistent storage."""
    if isinstance(reference_answer, dict) or isinstance(reference_answer, list):
        return json.dumps(reference_answer, ensure_ascii=False)
    return str(reference_answer)


def run_all_tests(limit: int = None, start_from: int = 0):
    """
    Run all tests from the dataset.
    
    Args:
        limit: Maximum number of tests to run (None for all)
        start_from: Test index to start from (for resuming)
    """
    global stats
    
    print("=" * 70)
    print("APUOPE-RE Test Runner")
    print("=" * 70)
    print(f"Loading test dataset from: {TEST_DATASET_FILE}")
    
    dataset = load_test_dataset()
    
    # Load existing results if resuming
    all_results = load_existing_results()
    existing_test_ids = {r['test_id'] for r in all_results}
    
    print(f"Found {len(dataset)} slides in dataset")
    
    # Flatten test cases
    all_test_cases = []
    for slide in dataset:
        for test_case in slide['test_cases']:
            all_test_cases.append({
                'slide': slide,
                'test_case': test_case
            })
    
    total_available = len(all_test_cases)
    print(f"Total test cases: {total_available}")
    
    if start_from > 0:
        print(f"Starting from test #{start_from}")
        all_test_cases = all_test_cases[start_from:]
    
    if limit:
        all_test_cases = all_test_cases[:limit]
        print(f"Running limited set: {len(all_test_cases)} tests")
    
    print(f"Existing results loaded: {len(all_results)}")
    print("=" * 70)
    print()
    
    stats['start_time'] = datetime.now().isoformat()
    stats['total_tests'] = len(all_test_cases)
    
    for idx, item in enumerate(all_test_cases):
        slide = item['slide']
        test_case = item['test_case']
        
        test_id = test_case['test_id']
        task_type = test_case['task_type']
        material_id = slide['material_id']
        
        # Skip if already processed
        if test_id in existing_test_ids:
            print(f"⏭️  [{idx+1}/{len(all_test_cases)}] Skipping {test_id} (already processed)")
            continue
        
        print(f"🔄 [{idx+1}/{len(all_test_cases)}] Processing {material_id} - {task_type}...")
        print(f"   Test ID: {test_id}")
        
        # Execute the test
        output, latency, error = execute_test(slide['content'], test_case)
        
        # Track statistics
        if task_type not in stats['by_task_type']:
            stats['by_task_type'][task_type] = {'count': 0, 'succeeded': 0, 'failed': 0, 'total_latency': 0.0}
        
        stats['by_task_type'][task_type]['count'] += 1
        stats['total_latency'] += latency
        stats['by_task_type'][task_type]['total_latency'] += latency
        
        if error:
            stats['failed'] += 1
            stats['by_task_type'][task_type]['failed'] += 1
            print(f"   ❌ FAILED: {error[:100]}...")
        else:
            stats['succeeded'] += 1
            stats['by_task_type'][task_type]['succeeded'] += 1
            print(f"   ✅ SUCCESS (latency: {latency:.2f}s)")
        
        # Evaluate output if no error
        llm_evaluation = None
        automated_metrics = None
        
        if not error and output:
            reference = test_case.get('reference_answer', '')
            
            # LLM-as-Judge evaluation
            print(f"   📊 Evaluating with LLM-as-Judge...")
            llm_evaluation = evaluate_with_llm_judge(
                output, 
                format_reference_answer(reference),
                task_type,
                test_case.get('instruction', '')
            )
            
            # Automated metrics
            automated_metrics = calculate_automated_metrics(
                output,
                format_reference_answer(reference),
                task_type
            )
            
            # Show quick score
            if llm_evaluation.get('scores', {}).get('overall'):
                print(f"   ⭐ LLM Score: {llm_evaluation['scores']['overall']:.1f}/10")
        
        # Create result entry
        result = {
            "test_id": test_id,
            "task_type": task_type,
            "material_id": material_id,
            "instruction": test_case.get('instruction', ''),
            "generated_output": output,
            "reference_answer": format_reference_answer(test_case.get('reference_answer', '')),
            "error": error,
            "latency_seconds": round(latency, 3),
            "timestamp": datetime.now().isoformat(),
            "llm_evaluation": llm_evaluation,
            "automated_metrics": automated_metrics
        }
        
        # Add task-specific metadata
        if task_type == 'quiz_generation' and 'constraints' in test_case:
            result['constraints'] = test_case['constraints']
        
        # Append to results
        all_results.append(result)
        
        # Save incrementally every test
        save_results_incremental(all_results)
        
        # Rate limiting: small delay between tests to avoid API rate limits
        if idx < len(all_test_cases) - 1:  # Don't delay after last test
            time.sleep(1.0)  # 1 second delay
        
        print()
    
    stats['end_time'] = datetime.now().isoformat()
    
    print("=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)
    print_summary_statistics()
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 70)


def print_summary_statistics():
    """Print summary statistics of test run."""
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"{'─' * 70}")
    print(f"Total Tests Run:     {stats['total_tests']}")
    print(f"✅ Succeeded:        {stats['succeeded']} ({stats['succeeded']/max(stats['total_tests'],1)*100:.1f}%)")
    print(f"❌ Failed:           {stats['failed']} ({stats['failed']/max(stats['total_tests'],1)*100:.1f}%)")
    print(f"⏱️  Total Time:       {stats['total_latency']:.2f} seconds")
    if stats['total_tests'] > 0:
        print(f"⏱️  Average Latency:  {stats['total_latency']/stats['total_tests']:.2f} seconds/test")
    
    print(f"\n📋 BY TASK TYPE:")
    print(f"{'─' * 70}")
    for task_type, task_stats in stats['by_task_type'].items():
        count = task_stats['count']
        succeeded = task_stats['succeeded']
        failed = task_stats['failed']
        avg_latency = task_stats['total_latency'] / count if count > 0 else 0
        
        print(f"\n{task_type.upper()}:")
        print(f"  Tests:    {count}")
        print(f"  Success:  {succeeded}/{count} ({succeeded/max(count,1)*100:.1f}%)")
        print(f"  Failed:   {failed}/{count}")
        print(f"  Avg Time: {avg_latency:.2f}s")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run APUOPE-RE benchmark tests')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tests to run (for testing)')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from test number N (for resuming)')
    
    args = parser.parse_args()
    
    try:
        run_all_tests(limit=args.limit, start_from=args.start_from)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        print(f"Partial results saved to: {OUTPUT_FILE}")
        print_summary_statistics()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

