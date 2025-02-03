import json
import ollama
import re
import ast
from typing import Dict, List, Optional, Any

model_name = 'llama3.2'

# Enhanced problem categories with more specific biological contexts
PROBLEM_CATEGORIES = [
    {
        "category": "Sequence Analysis",
        "subcategories": [
            {
                "name": "DNA/RNA Processing",
                "problems": [
                    "Find overlapping genes in a DNA sequence",
                    "Identify alternative splice sites in RNA",
                    "Detect CpG islands in promoter regions",
                    "Find palindromic sequences (restriction sites)",
                    "Calculate GC content in sliding windows"
                ]
            },
            {
                "name": "Protein Analysis",
                "problems": [
                    "Predict protein secondary structure",
                    "Find conserved protein domains",
                    "Calculate hydrophobicity profiles",
                    "Identify signal peptides",
                    "Detect transmembrane regions"
                ]
            }
        ]
    },
    {
        "category": "Genomic Algorithms",
        "subcategories": [
            {
                "name": "Assembly & Alignment",
                "problems": [
                    "Implement De Bruijn graph for genome assembly",
                    "Create a multiple sequence alignment",
                    "Find syntenic blocks between genomes",
                    "Detect genomic inversions",
                    "Identify repetitive elements"
                ]
            },
            {
                "name": "Variant Analysis",
                "problems": [
                    "Call SNPs from sequence data",
                    "Detect copy number variations",
                    "Find structural variants",
                    "Identify microsatellites",
                    "Calculate linkage disequilibrium"
                ]
            }
        ]
    },
    {
        "category": "Network Biology",
        "subcategories": [
            {
                "name": "Pathway Analysis",
                "problems": [
                    "Find feedback loops in signaling pathways",
                    "Identify network motifs in gene regulation",
                    "Calculate pathway enrichment scores",
                    "Detect protein complexes in PPI networks",
                    "Find metabolic bottlenecks"
                ]
            },
            {
                "name": "Disease Networks",
                "problems": [
                    "Identify disease modules in networks",
                    "Find drug targets in pathways",
                    "Calculate disease similarity networks",
                    "Detect cancer driver mutations",
                    "Analyze patient similarity networks"
                ]
            }
        ]
    }
]


def clean_llm_response(response_str: str) -> Dict[str, Any]:
    """
    Attempt to extract JSON from the LLM response string.
    1) Remove markdown fences (triple backticks).
    2) Find the first '{' and last '}'.
    3) Try json.loads on that substring.
    4) If that fails, try ast.literal_eval -> json.dumps -> json.loads
       This fallback helps with Python tuples like (0,5) that break standard JSON.
    5) Print debug info if parsing fails.
    """
    # Remove any Markdown fences
    cleaned = re.sub(r"```(\w+)?", "", response_str)
    cleaned = re.sub(r"```", "", cleaned)

    # Find the first '{' and the last '}'
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx == -1 or end_idx == -1:
        print("Raw model response (no JSON found):")
        print(response_str)
        raise ValueError("No valid JSON object found in LLM response.")

    json_part = cleaned[start_idx:end_idx + 1].strip()

    # First attempt standard JSON parse
    try:
        return json.loads(json_part)

    # If that fails, fallback to a python parse with ast.literal_eval
    except json.JSONDecodeError:
        try:
            python_obj = ast.literal_eval(json_part)
            # Now convert the python object to a valid JSON string
            as_json_str = json.dumps(python_obj)
            return json.loads(as_json_str)
        except Exception as e:
            print("\nCould not parse JSON or Python literal. Raw substring was:")
            print(json_part)
            raise ValueError(f"Parsing error: {e}")


def generate_llm_prompt(problem_info: Dict, difficulty: int) -> str:
    """Generate an enhanced prompt for the LLM with more biological context."""
    return f"""Create a bioinformatics coding problem following these specifications:

Topic: {problem_info['name']}
Difficulty Level: {difficulty}/5
Category: {problem_info['category']}

Requirements:
1. Problem should be based on real biological data and processes
2. Include practical applications in research or medicine
3. Use standard Python data structures and algorithms
4. Include edge cases specific to biological data
5. function_signature should be a one line function prototype.
6. Output should be an integer or string value.
7. Example should show an example invocation and expected output.
8. Three test cases minimum.

Return a JSON object with:
{{
    "title": "Clear, specific title",
    "problem_description": "Detailed problem description",
    "function_signature": "def function_name(parameters):",
    "test_cases": [
        {{
            "input": "Sample input",
            "expected": "Expected output",
            "invocation": "function_name(input)"
        }}
    ]
}}

Format as clean JSON with arrays for lists (no Python tuples). Focus on biological accuracy and practical relevance."""


def parse_function_name(func_sig: str) -> str:
    """
    Given a function signature string like:
      'def transcribe_dna(seq):\n    pass'
    return the function name: 'transcribe_dna'
    """
    match = re.search(r'^\s*def\s+([A-Za-z_]\w*)\s*\(', func_sig)
    if match:
        return match.group(1)
    raise ValueError("Could not parse function name from signature.")

def generate_problem(category: Dict, index: int) -> Optional[Dict]:
    """Generate a more sophisticated problem using LLM."""
    try:
        # Select subcategory and problem
        subcategory = category["subcategories"][index % len(category["subcategories"])]
        problem = {
            "name": subcategory["problems"][index % len(subcategory["problems"])],
            "category": category["category"]
        }

        # Calculate difficulty (progressive)
        difficulty = min(1 + (index // 5), 5)

        # Generate prompt
        prompt = generate_llm_prompt(problem, difficulty)

        # If Ollama supports a system prompt:
        system_msg = (
            "You are a JSON output machine. "
            "Return ONLY valid JSON arrays/lists (no Python tuples). "
            "No extra text or markdown fences."
        )

        # Lower temperature for more structured output
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            system=system_msg,  # or system_prompt=system_msg if needed
            options={"temperature":0.2}
        )

        # Attempt to parse or fallback with ast.literal_eval
        problem_data = clean_llm_response(response['response'])

        # Enhanced problem format
        enhanced_problem = {
            "id": index + 1,
            "title": problem_data['title'],
            "description": problem_data['problem_description'],
            "starterCode": problem_data['function_signature'] + "\n\n    ",
            "difficulty": difficulty,
            "category": category["category"],
            "subcategory": subcategory["name"],
            "testCases": []
        }

        function_name = problem_data['function_signature'].split(' ')[1].split('(')[0]
        first = True
        # Convert test cases with function-calling code
        for test in problem_data.get('test_cases', []):
            if first:
                enhanced_problem['example'] = f"{function_name}({test['input']}) == {test['expected']}"
                first = False
            test_case = {
                "input": test['input'],
                "expected": test['expected'],
                "code": f"print({test['invocation']})"
            }
            enhanced_problem['testCases'].append(test_case)

        return enhanced_problem

    except Exception as e:
        print(f"Error generating problem: {e}")
        return None


def generate_problems(num_problems: int = 100) -> List[Dict]:
    """Generate multiple problems across categories."""
    problems = []
    retries = 0
    max_retries = num_problems * 2

    while len(problems) < num_problems and retries < max_retries:
        try:
            category_index = len(problems) % len(PROBLEM_CATEGORIES)
            category = PROBLEM_CATEGORIES[category_index]

            problem = generate_problem(category, len(problems))
            if problem:
                problems.append(problem)
                print(f"Generated problem {len(problems)}/{num_problems}")

                # Save progress periodically
                if len(problems) % 10 == 0:
                    save_problems(problems)
                    print(f"Saved progress: {len(problems)} problems")

            retries += 1

        except Exception as e:
            print(f"Error in problem generation: {e}")
            retries += 1
            continue

    return problems


def save_problems(problems: List[Dict], output_file: str = 'biocode_problems.js'):
    """Save problems as a JavaScript module."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("const questions = ")
        json.dump(problems, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(f"Generating problems using {model_name}...")

    try:
        # Generate fewer problems for demo
        problems = generate_problems(num_problems=100)
        save_problems(problems)
        print(f"\nSuccessfully generated {len(problems)} problems")

    except KeyboardInterrupt:
        print("\nSaving progress before exit...")
        if 'problems' in locals() and problems:
            save_problems(problems)
            print(f"Saved {len(problems)} problems")

    except Exception as e:
        print(f"Fatal error: {e}")
