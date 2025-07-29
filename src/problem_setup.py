#!/usr/bin/env python3
"""
Problem environment generator for LiveCodeBench agent testing.

This script creates isolated environments for individual problems with:
- Problem description file
- Solution stub with proper signatures
- Standalone test runner (no lcb_runner dependencies)
- All test cases

Usage:
    python problem_setup.py --problem-id leetcode_2933
    python problem_setup.py --random --difficulty easy
    python problem_setup.py --interactive
"""

import argparse
import json
import os
import sys
import random
import re
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from problems import (
    load_code_generation_dataset,
    Platform,
    Difficulty
)


def sanitize_filename(name):
    """Convert problem title to safe directory name."""
    # Remove or replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = name.lower()
    # Limit length
    if len(name) > 50:
        name = name[:50]
    return name


def extract_function_signature(problem):
    """Extract function signature from problem metadata or content."""
    func_name = problem.metadata.get('func_name')
    if not func_name:
        return None, []
    
    # Try to extract parameter info from problem content
    params = []
    
    # Look for common patterns in problem descriptions
    content = problem.question_content.lower()
    
    # Common parameter patterns
    if 'array' in content or 'list' in content:
        if 'integer' in content or 'int' in content:
            params.append('nums: List[int]')
        elif 'string' in content:
            params.append('strs: List[str]')
        else:
            params.append('arr: List')
    
    if 'target' in content:
        params.append('target: int')
    
    if 'matrix' in content:
        params.append('matrix: List[List[int]]')
    
    if 'string' in content and 'array' not in content:
        params.append('s: str')
    
    # If we couldn't infer parameters, use generic ones
    if not params:
        params = ['*args']
    
    return func_name, params


def generate_solution_stub(problem):
    """Generate solution stub based on problem type."""
    func_name, params = extract_function_signature(problem)
    
    if func_name:
        # Function-based problem
        param_str = ', '.join(params)
        
        stub = f'''"""
Solution for: {problem.question_title}
Problem ID: {problem.question_id}
Platform: {problem.platform.value}
Difficulty: {problem.difficulty.value}

{problem.question_content[:200]}...
"""

from typing import List, Optional, Dict, Set, Tuple


def {func_name}({param_str}):
    """
    TODO: Implement your solution here.
    
    Args:
        {chr(10).join(f"        {param.split(':')[0]}: {param.split(':')[1] if ':' in param else 'Input parameter'}" for param in params)}
    
    Returns:
        The result according to problem requirements.
    """
    # TODO: Replace this with your solution
    pass


# For LeetCode-style problems, also provide a Solution class
class Solution:
    def {func_name}(self, {param_str}):
        """LeetCode-style solution wrapper."""
        return {func_name}({', '.join(p.split(':')[0] for p in params)})


if __name__ == "__main__":
    # Test your solution locally
    print("Testing solution locally...")
    
    # TODO: Add your test cases here
    # Example:
    # result = {func_name}(test_input)
    # print(f"Result: {{result}}")
    
    # Run the comprehensive tests
    print("\\nRunning comprehensive tests...")
    os.system("python test.py")
'''
    else:
        # Standard input/output problem
        stub = f'''"""
Solution for: {problem.question_title}
Problem ID: {problem.question_id}
Platform: {problem.platform.value}
Difficulty: {problem.difficulty.value}

{problem.question_content[:200]}...
"""

import sys
from typing import List, Optional, Dict, Set, Tuple


def solve():
    """
    TODO: Implement your solution here.
    
    This is a standard input/output problem.
    Read input from stdin and write output to stdout.
    """
    
    # Common input reading patterns:
    
    # Single integer
    # n = int(input())
    
    # Multiple integers on one line
    # a, b, c = map(int, input().split())
    
    # List of integers
    # numbers = list(map(int, input().split()))
    
    # Multiple lines
    # lines = []
    # for _ in range(n):
    #     lines.append(input().strip())
    
    # Read all input at once
    # data = sys.stdin.read().strip().split('\\n')
    
    # TODO: Replace this with your solution
    result = "TODO"
    
    # Output the result
    print(result)


if __name__ == "__main__":
    solve()
'''
    
    return stub


def generate_test_script(problem):
    """Generate standalone test script with NO lcb_runner dependencies."""
    func_name = problem.metadata.get('func_name')
    is_function = func_name is not None
    
    if is_function:
        # Function-based test script
        test_script = f'''#!/usr/bin/env python3
"""
Standalone test runner for: {problem.question_title}
Problem ID: {problem.question_id}

This script tests your solution against all test cases.
Stops at first failure (fail-fast).
Run with: python test.py
"""

import json
import sys
import signal
import time


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Test timed out")


def load_test_cases():
    """Load test cases from JSON file."""
    with open('test_cases.json', 'r') as f:
        return json.load(f)


def run_function_based_test(test_cases, timeout=6):
    """Run tests for function-based problems."""
    # Import the solution
    try:
        from solution import {func_name}
    except ImportError:
        try:
            from solution import Solution
            solution_instance = Solution()
            {func_name} = getattr(solution_instance, '{func_name}')
        except (ImportError, AttributeError):
            print("❌ Could not import solution function '{func_name}'")
            print("Make sure your solution.py defines the function or Solution class.")
            return False
    
    passed = 0
    total = len(test_cases)
    
    print(f"Running {{total}} test cases for function: {func_name}")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases):
        test_num = i + 1
        print(f"Test {{test_num}}/{{total}}: ", end="")
        
        try:
            # Parse input and expected output
            input_lines = test_case['input'].strip().split('\\n')
            inputs = [json.loads(line) for line in input_lines]
            expected = json.loads(test_case['output'])
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Call the function
                start_time = time.time()
                result = {func_name}(*inputs)
                execution_time = time.time() - start_time
                signal.alarm(0)
                
                # Compare results
                if result == expected:
                    print(f"✅ PASS ({{execution_time:.3f}}s)")
                    passed += 1
                else:
                    print(f"❌ WRONG ANSWER")
                    print(f"   Input: {{inputs}}")
                    print(f"   Expected: {{expected}}")
                    print(f"   Got: {{result}}")
                    print("=" * 60)
                    passed_percent = passed / test_num * 100
                    print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
                    print("\\n❌ Test failed - stopping execution.")
                    return False
                
            except TimeoutException:
                signal.alarm(0)
                print(f"⏰ TIMEOUT (>{{timeout}}s)")
                print("=" * 60)
                passed_percent = passed / test_num * 100
                print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
                print("\\n❌ Test timed out - stopping execution.")
                return False
            except Exception as e:
                signal.alarm(0)
                print(f"💥 ERROR: {{e}}")
                print("=" * 60)
                passed_percent = passed / test_num * 100
                print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
                print("\\n❌ Test error - stopping execution.")
                return False
                
        except Exception as e:
            print(f"💥 TEST SETUP ERROR: {{e}}")
            print("=" * 60)
            passed_percent = passed / test_num * 100
            print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
            print("\\n❌ Test setup error - stopping execution.")
            return False
        
        finally:
            signal.alarm(0)
    
    print("=" * 60)
    print(f"Results: {{passed}}/{{total}} tests passed ({{passed/total*100:.1f}}%)")
    return passed == total


def main():
    """Main test runner."""
    print("LiveCodeBench Problem Test Runner")
    print()
    
    # Load test cases
    try:
        test_cases = load_test_cases()
    except Exception as e:
        print(f"❌ Failed to load test cases: {{e}}")
        return False
    
    # Run function-based tests
    success = run_function_based_test(test_cases)
    
    if success:
        print("\\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print("\\n❌ Some tests failed. Keep working on your solution!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    else:
        # Standard I/O test script
        test_script = f'''#!/usr/bin/env python3
"""
Standalone test runner for: {problem.question_title}
Problem ID: {problem.question_id}

This script tests your solution against all test cases.
Stops at first failure (fail-fast).
Run with: python test.py
"""

import json
import sys
import subprocess
import time


def load_test_cases():
    """Load test cases from JSON file."""
    with open('test_cases.json', 'r') as f:
        return json.load(f)


def run_stdio_test(test_cases, timeout=6):
    """Run tests for standard input/output problems."""
    passed = 0
    total = len(test_cases)
    
    print(f"Running {{total}} test cases for stdin/stdout problem")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases):
        test_num = i + 1
        print(f"Test {{test_num}}/{{total}}: ", end="")
        
        try:
            input_data = test_case['input']
            expected_output = test_case['output'].strip()
            
            # Run the solution as a subprocess
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, 'solution.py'],
                input=input_data,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"💥 RUNTIME ERROR")
                print(f"   stderr: {{result.stderr.strip()}}")
                print("=" * 60)
                passed_percent = passed / test_num * 100
                print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
                print("\\n❌ Runtime error - stopping execution.")
                return False
            
            actual_output = result.stdout.strip()
            
            if actual_output == expected_output:
                print(f"✅ PASS ({{execution_time:.3f}}s)")
                passed += 1
            else:
                print(f"❌ WRONG ANSWER")
                print(f"   Input: {{repr(input_data)}}")
                print(f"   Expected: {{repr(expected_output)}}")
                print(f"   Got: {{repr(actual_output)}}")
                print("=" * 60)
                passed_percent = passed / test_num * 100
                print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
                print("\\n❌ Test failed - stopping execution.")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT (>{{timeout}}s)")
            print("=" * 60)
            passed_percent = passed / test_num * 100
            print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
            print("\\n❌ Test timed out - stopping execution.")
            return False
        except Exception as e:
            print(f"💥 TEST ERROR: {{e}}")
            print("=" * 60)
            passed_percent = passed / test_num * 100
            print(f"Results: {{passed}}/{{test_num}} tests passed ({{passed_percent:.1f}}%)")
            print("\\n❌ Test error - stopping execution.")
            return False
    
    print("=" * 60)
    print(f"Results: {{passed}}/{{total}} tests passed ({{passed/total*100:.1f}}%)")
    return passed == total


def main():
    """Main test runner."""
    print("LiveCodeBench Problem Test Runner")
    print()
    
    # Load test cases
    try:
        test_cases = load_test_cases()
    except Exception as e:
        print(f"❌ Failed to load test cases: {{e}}")
        return False
    
    # Run stdio tests
    success = run_stdio_test(test_cases)
    
    if success:
        print("\\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print("\\n❌ Some tests failed. Keep working on your solution!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    return test_script


def generate_problem_description(problem):
    """Generate markdown problem description."""
    description = f'''# {problem.question_title}

**Problem ID:** `{problem.question_id}`  
**Platform:** {problem.platform.value}  
**Difficulty:** {problem.difficulty.value}  
**Contest Date:** {problem.contest_date.strftime('%Y-%m-%d')}  

## Problem Description

{problem.question_content}

## Additional Information

- **Total Test Cases:** {len(problem.public_test_cases + problem.private_test_cases)}
- **Public Test Cases:** {len(problem.public_test_cases)}
- **Private Test Cases:** {len(problem.private_test_cases)}

{"**Function Name:** `" + problem.metadata.get('func_name', '') + "`" if problem.metadata.get('func_name') else "**Type:** Standard Input/Output"}

## Starter Code

{"```python" if problem.starter_code else "No starter code provided."}
{problem.starter_code if problem.starter_code else ""}
{"```" if problem.starter_code else ""}

## Sample Test Cases

### Public Test Cases
'''
    
    for i, test in enumerate(problem.public_test_cases[:3]):  # Show first 3
        description += f'''
**Test Case {i+1}:**
```
Input: {test.input}
Output: {test.output}
```
'''
    
    if len(problem.public_test_cases) > 3:
        description += f"\n*... and {len(problem.public_test_cases) - 3} more public test cases*\n"
    
    description += f'''
### Private Test Cases
*This problem has {len(problem.private_test_cases)} private test cases that will be used for evaluation.*

## How to Solve

1. Edit `solution.py` to implement your solution
2. Run `python test.py` to test against all test cases (stops at first failure)
3. Iterate until all tests pass

## Files in this Directory

- `solution.py` - Your solution code (edit this!)
- `test.py` - Standalone test runner (no external dependencies)
- `test_cases.json` - All test cases in JSON format
- `problem.md` - This problem description
- `README.md` - Quick start guide
'''
    
    return description


def generate_readme(problem):
    """Generate problem-specific README."""
    func_name = problem.metadata.get('func_name')
    
    readme = f'''# {problem.question_title}

Quick start guide for solving this problem.

## Problem Summary
- **ID:** {problem.question_id}
- **Platform:** {problem.platform.value}
- **Difficulty:** {problem.difficulty.value}
- **Type:** {"Function-based" if func_name else "Standard Input/Output"}

## Quick Start

1. **Read the problem:** Open `problem.md` for full details
2. **Edit solution:** Modify `solution.py` with your implementation
3. **Test:** Run `python test.py` to check your solution (stops at first failure)
4. **Repeat:** Keep iterating until all tests pass

## Solution Template

Your `solution.py` already contains the proper template for this problem type.

{"### Function-based Problem" if func_name else "### Standard Input/Output Problem"}
{"You need to implement the `" + func_name + "()` function." if func_name else "You need to read from stdin and write to stdout."}

## Testing

```bash
# Run all tests (stops at first failure)
python test.py

# Test specific logic locally
python solution.py
```

## Expected Output

When your solution is correct, you should see:
```
Running {len(problem.public_test_cases + problem.private_test_cases)} test cases...
Test 1/X: ✅ PASS
Test 2/X: ✅ PASS
...
🎉 ALL TESTS PASSED! 🎉
```

## Common Issues

{"**Function signature:** Make sure your function name and parameters match exactly." if func_name else "**Input/Output format:** Make sure you're reading and writing in the correct format."}

**Timeout:** If tests are timing out, optimize your algorithm.

**Wrong answer:** Check edge cases and ensure you understand the problem correctly.

Good luck! 🚀
'''
    
    return readme


def create_problem_environment(problem, base_dir="problems"):
    """Create isolated environment for a problem."""
    # Create directory name
    problem_dir_name = f"{problem.question_id}_{sanitize_filename(problem.question_title)}"
    problem_dir = Path(base_dir) / problem_dir_name
    
    # Create directory
    problem_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating problem environment: {problem_dir}")
    
    # Generate all test cases data
    test_cases = []
    for test in problem.public_test_cases + problem.private_test_cases:
        test_cases.append({
            "input": test.input,
            "output": test.output,
            "testtype": test.testtype.value
        })
    
    # Write files
    files_created = []
    
    # 1. Solution stub
    solution_file = problem_dir / "solution.py"
    solution_file.write_text(generate_solution_stub(problem))
    files_created.append(solution_file)
    
    # 2. Test script
    test_file = problem_dir / "test.py"
    test_file.write_text(generate_test_script(problem))
    test_file.chmod(0o755)  # Make executable
    files_created.append(test_file)
    
    # 3. Test cases JSON
    test_cases_file = problem_dir / "test_cases.json"
    test_cases_file.write_text(json.dumps(test_cases, indent=2))
    files_created.append(test_cases_file)
    
    # 4. Problem description
    problem_file = problem_dir / "problem.md"
    problem_file.write_text(generate_problem_description(problem))
    files_created.append(problem_file)
    
    # 5. README
    readme_file = problem_dir / "README.md"
    readme_file.write_text(generate_readme(problem))
    files_created.append(readme_file)
    
    return problem_dir, files_created


def setup_problem_by_id(problem_id: str, output_dir: str = "problems", 
                        release_version: str = "release_v1", verbose: bool = False):
    """
    Create a problem environment for a specific problem ID.
    
    Args:
        problem_id: The specific problem ID to set up
        output_dir: Base directory for problem environments  
        release_version: Dataset version to use
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (problem_dir_path, created_files_list) on success, or None on failure
    """
    try:
        # Load dataset
        if verbose:
            print(f"Loading dataset (version: {release_version})...")
        
        problems = load_code_generation_dataset(release_version=release_version)
        
        # Find the specific problem
        matching = [p for p in problems if p.question_id == problem_id]
        if not matching:
            if verbose:
                print(f"Problem '{problem_id}' not found")
            return None
            
        selected_problem = matching[0]
        
        # Create environment
        problem_dir, files = create_problem_environment(selected_problem, output_dir)
        
        if verbose:
            print(f"✅ Problem environment created successfully!")
            print(f"📁 Directory: {problem_dir}")
            print(f"📋 Problem: {selected_problem.question_title}")
            print(f"🏷️  ID: {selected_problem.question_id}")
            print(f"📊 Difficulty: {selected_problem.difficulty.value}")
            print(f"🎯 Platform: {selected_problem.platform.value}")
        
        return problem_dir, files
        
    except Exception as e:
        if verbose:
            print(f"Error creating environment: {e}")
        return None


def setup_random_problem(output_dir: str = "problems", difficulty: str = None,
                        platform: str = None, release_version: str = "release_v1", 
                        verbose: bool = False):
    """
    Create a problem environment for a randomly selected problem.
    
    Args:
        output_dir: Base directory for problem environments
        difficulty: Filter by difficulty ("easy", "medium", "hard")
        platform: Filter by platform ("leetcode", "codeforces", "atcoder")
        release_version: Dataset version to use
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of (problem_dir_path, created_files_list) on success, or None on failure
    """
    try:
        # Load dataset
        if verbose:
            print(f"Loading dataset (version: {release_version})...")
        
        problems = load_code_generation_dataset(
            release_version=release_version,
            difficulty=difficulty
        )
        
        if platform:
            platform_enum = Platform(platform)
            problems = [p for p in problems if p.platform == platform_enum]
        
        if not problems:
            if verbose:
                print("No problems available with the specified filters")
            return None
            
        selected_problem = random.choice(problems)
        if verbose:
            print(f"Randomly selected: {selected_problem.question_id} - {selected_problem.question_title}")
        
        # Create environment
        problem_dir, files = create_problem_environment(selected_problem, output_dir)
        
        if verbose:
            print(f"✅ Problem environment created successfully!")
            print(f"📁 Directory: {problem_dir}")
            print(f"📋 Problem: {selected_problem.question_title}")
        
        return problem_dir, files
        
    except Exception as e:
        if verbose:
            print(f"Error creating environment: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate isolated problem environments")
    
    # Problem selection
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument("--problem-id", help="Specific problem ID")
    selection_group.add_argument("--random", action="store_true", help="Random problem")
    selection_group.add_argument("--interactive", action="store_true", help="Interactive selection")
    
    # Filters (for random selection)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                       help="Filter by difficulty")
    parser.add_argument("--platform", choices=["leetcode", "codeforces", "atcoder"],
                       help="Filter by platform")
    parser.add_argument("--release-version", default="release_v1",
                       help="Dataset version")
    
    # Output options
    parser.add_argument("--output-dir", default="problems",
                       help="Base directory for problem environments")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing environment")
    
    args = parser.parse_args()
    
    # Use the new functions
    if args.problem_id:
        result = setup_problem_by_id(
            problem_id=args.problem_id,
            output_dir=args.output_dir,
            release_version=args.release_version,
            verbose=True
        )
    elif args.random:
        result = setup_random_problem(
            output_dir=args.output_dir,
            difficulty=args.difficulty,
            platform=args.platform,
            release_version=args.release_version,
            verbose=True
        )
    elif args.interactive:
        # Load dataset for interactive selection
        print(f"Loading dataset (version: {args.release_version})...")
        try:
            problems = load_code_generation_dataset(
                release_version=args.release_version,
                difficulty=args.difficulty
            )
            
            if args.platform:
                platform_enum = Platform(args.platform)
                problems = [p for p in problems if p.platform == platform_enum]
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return 1
            
        if not problems:
            print("No problems available with the specified filters")
            return 1
            
        print(f"\nFound {len(problems)} problems. Select one:")
        for i, problem in enumerate(problems[:20]):  # Show first 20
            print(f"{i+1:2d}. [{problem.question_id}] {problem.question_title} ({problem.difficulty.value})")
        
        if len(problems) > 20:
            print(f"... and {len(problems) - 20} more")
            
        try:
            choice = int(input("\nEnter problem number: ")) - 1
            if 0 <= choice < len(problems):
                selected_problem = problems[choice]
                problem_dir, files = create_problem_environment(selected_problem, args.output_dir)
                result = (problem_dir, files)
            else:
                print("Invalid selection")
                return 1
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or cancelled")
            return 1
    
    if result is None:
        return 1
    
    problem_dir, files = result
    print(f"\n📄 Files created:")
    for file in files:
        print(f"   {file.name}")
    
    print(f"\n🚀 Next steps:")
    print(f"   cd {problem_dir}")
    print(f"   # Read the problem: cat problem.md")
    print(f"   # Edit your solution: nano solution.py") 
    print(f"   # Test your solution: python test.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())