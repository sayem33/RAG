import json
import os
from dotenv import load_dotenv
from rag_engine import rag_generate_quiz

# Load environment variables from .env file
load_dotenv()

def generate_quiz(pdf_content, difficulty, pdf_path=None):
    """
    Generate quiz questions and answers using RAG approach.

    Args:
        pdf_content (str): The text content extracted from the lecture PDF.
        difficulty (str): Selected difficulty level: 'easy', 'medium', or 'hard'.
        pdf_path (str): Path to the PDF file for RAG.

    Returns:
        tuple: A list of question dictionaries and a dictionary of correct answers.
    """
    try:
        if pdf_path:
            raw_data = rag_generate_quiz(pdf_path, pdf_content, difficulty)
        else:
            raw_data = rag_generate_quiz("temp", pdf_content, difficulty)

        # Extract JSON block
        json_start = raw_data.find("[")
        json_end = raw_data.rfind("]")
        if json_start == -1 or json_end == -1 or json_start > json_end:
            raise ValueError("No valid JSON block found in the response.")
        quiz_data = raw_data[json_start:json_end + 1]

        # Parse the JSON block
        quiz_questions = json.loads(quiz_data)
        correct_answers = {idx: q["answer"] for idx, q in enumerate(quiz_questions)}
        return quiz_questions, correct_answers

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return [], {}
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return [], {}

def evaluate_quiz(submitted_answers, correct_answers):
    """
    Evaluate submitted quiz answers.

    Args:
        submitted_answers (dict): Answers submitted by the student.
        correct_answers (dict): The correct answers for the quiz.

    Returns:
        tuple: Score, total questions, and detailed feedback.
    """
    score = 0
    total = len(correct_answers)
    feedback = {}

    for idx, correct in correct_answers.items():
        submitted = submitted_answers.get(idx, None)
        if isinstance(correct, list):
            # For MCQ (multiple answers), compare sets
            if set(submitted) == set(correct):
                score += 1
                feedback[idx] = "Correct"
            else:
                feedback[idx] = f"Incorrect. Correct answers: {', '.join(correct)}"
        else:
            # For MCQ (single answer) and True/False
            if submitted == correct:
                score += 1
                feedback[idx] = "Correct"
            else:
                feedback[idx] = f"Incorrect. Correct answer: {correct}"

    return score, total, feedback
