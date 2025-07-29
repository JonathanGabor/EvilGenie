#!/usr/bin/env python3
"""
Main CLI for automated agent testing.

This script provides a command-line interface for running automated tests
of coding agents on LiveCodeBench problems.
"""

import argparse
import json
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from automated_tester import AutomatedTester
from problems import load_code_generation_dataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_problems_from_config(config: Dict[str, Any]) -> List[str]:
    """Get list of problems to test based on configuration."""
    # If specific problems are listed, use those
    if config.get('specific_problems'):
        return config['specific_problems']
    
    # Otherwise use problem filters
    filters = config.get('problem_filters', {})
    
    try:
        # Load problems using the actual LiveCodeBench dataset loader
        problems = load_code_generation_dataset(
            release_version="release_v1",
            start_date=filters.get('date_range', {}).get('start'),
            end_date=filters.get('date_range', {}).get('end'),
            difficulty=None  # We'll filter difficulty below
        )
        
        # Filter by difficulty
        if 'difficulties' in filters:
            difficulty_values = filters['difficulties']
            problems = [p for p in problems if p.difficulty.value in difficulty_values]
        
        # Filter by platform
        if 'platforms' in filters:
            platform_values = filters['platforms']
            problems = [p for p in problems if p.platform.value in platform_values]
        
        # Limit number of problems
        max_problems = filters.get('max_problems', len(problems))
        problems = problems[:max_problems]
        
        return [p.question_id for p in problems]
        
    except Exception as e:
        print(f"Error loading problems from dataset: {e}")
        # Return some fallback problems
        return ['leetcode_1', 'leetcode_2']


def main():
    parser = argparse.ArgumentParser(description='Automated agent testing for LiveCodeBench')
    
    # Configuration
    parser.add_argument('--config', '-c', 
                       default='test_config.yaml',
                       help='Configuration file path')
    
    # Agent selection
    parser.add_argument('--agent', '-a',
                       help='Specific agent to test (overrides config)')
    parser.add_argument('--agents',
                       nargs='+',
                       help='Multiple agents to test (overrides config)')
    
    # Model selection
    parser.add_argument('--model',
                       help='Model to use (e.g., gpt-4, gpt-4o-mini, o3-mini, claude-3-5-sonnet)')
    parser.add_argument('--reasoning-effort',
                       choices=['low', 'medium', 'high'],
                       help='Reasoning effort for O3 models (low/medium/high)')
    
    # Problem selection
    parser.add_argument('--problem', '-p',
                       help='Specific problem ID to test')
    parser.add_argument('--problems',
                       nargs='+',
                       help='Multiple problem IDs to test')
    
    parser.add_argument('--difficulty', '-d',
                       choices=['easy', 'medium', 'hard'],
                       help='Filter problems by difficulty')
    
    parser.add_argument('--platform',
                       choices=['leetcode', 'atcoder', 'codeforces'],
                       help='Filter problems by platform')
    
    parser.add_argument('--max-problems', '-m',
                       type=int,
                       help='Maximum number of problems to test')
    
    # Test settings
    parser.add_argument('--timeout', '-t',
                       type=int,
                       help='Timeout per test in seconds')
    
    parser.add_argument('--no-cleanup',
                       action='store_true',
                       help='Keep workspace directories after testing')
    
    parser.add_argument('--scenario', '-s',
                       help='Use predefined test scenario from config')
    
    # Output settings
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Reduce output verbosity')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Increase output verbosity')
    
    args = parser.parse_args()
    
    # Create timestamp-based run directory for better organization
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created run directory: {run_dir}")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration file...")
        
        # Create default config if it doesn't exist
        default_config = {
            'timeout': 300,
            'cleanup': True,
            'agents': [{'name': 'claude', 'flags': []}],
            'problem_filters': {
                'difficulties': ['easy'],
                'max_problems': 3
            }
        }
        
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        config = default_config
        print(f"Created {args.config} with default settings")
    
    # Set up organized workspace directories
    config['workspace_base_dir'] = str(run_dir / "workspaces")
    config['results_dir'] = str(run_dir / "results")
    
    # Copy configuration to run directory for reproducibility
    config_copy_path = run_dir / "config.yaml"
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"📋 Configuration saved to: {config_copy_path}")
    
    # Apply command line overrides
    if args.scenario:
        if args.scenario in config.get('scenarios', {}):
            scenario_config = config['scenarios'][args.scenario]
            config.update(scenario_config)
            print(f"Using scenario: {args.scenario}")
        else:
            print(f"Scenario '{args.scenario}' not found in configuration")
            sys.exit(1)
    
    if args.timeout:
        config['timeout'] = args.timeout
    
    if args.no_cleanup:
        config['cleanup'] = False
    
    if args.verbose:
        config['log_level'] = 'DEBUG'
    elif args.quiet:
        config['log_level'] = 'WARNING'
    
    if args.output:
        config['results_file'] = args.output
    
    # Apply model overrides only to compatible agent configurations
    if args.model or args.reasoning_effort:
        # Define which agents support model selection
        model_compatible_agents = {'openai'}  # Add more as needed
        
        for agent_config in config.get('agents', []):
            agent_name = agent_config.get('name', '').lower()
            
            if agent_name not in model_compatible_agents:
                continue  # Skip agents that don't support model selection
                
            if args.model:
                # Update model in llm_config or model_config depending on agent type
                if 'llm_config' in agent_config:
                    agent_config['llm_config']['model'] = args.model
                elif 'model_config' in agent_config:
                    agent_config['model_config']['model'] = args.model
                else:
                    # Create llm_config if it doesn't exist
                    agent_config['llm_config'] = {'model': args.model}
            
            if args.reasoning_effort and 'llm_config' in agent_config:
                agent_config['llm_config']['reasoning_effort'] = args.reasoning_effort
    else:
        # Default results file in run directory
        config['results_file'] = str(run_dir / "test_results.json")
    
    # Determine agents to test
    agents = config.get('agents', [])
    if args.agent:
        agents = [{'name': args.agent, 'flags': []}]
    elif args.agents:
        agents = [{'name': name, 'flags': []} for name in args.agents]
    
    # Apply model overrides to selected agents (especially important for dynamically created ones)
    if args.model or args.reasoning_effort:
        model_compatible_agents = {'openai'}
        
        # Check if any selected agents are incompatible with model selection
        incompatible_agents = []
        for agent_config in agents:
            agent_name = agent_config.get('name', '').lower()
            if agent_name not in model_compatible_agents:
                incompatible_agents.append(agent_name)
        
        if incompatible_agents:
            print(f"⚠️  Warning: --model flag not supported by agents: {', '.join(incompatible_agents)}")
            print(f"   Model selection only works with: {', '.join(sorted(model_compatible_agents))}")
            if len(incompatible_agents) == len(agents):
                print("   No compatible agents selected - model flag will be ignored")
        
        for agent_config in agents:
            agent_name = agent_config.get('name', '').lower()
            if agent_name not in model_compatible_agents:
                continue  # Skip incompatible agents
                
            if args.model:
                if 'llm_config' not in agent_config:
                    agent_config['llm_config'] = {}
                agent_config['llm_config']['model'] = args.model
            
            if args.reasoning_effort:
                if 'llm_config' not in agent_config:
                    agent_config['llm_config'] = {}
                agent_config['llm_config']['reasoning_effort'] = args.reasoning_effort
    
    if not agents:
        print("No agents specified")
        sys.exit(1)
    
    # Determine problems to test
    problems = []
    if args.problem:
        problems = [args.problem]
    elif args.problems:
        problems = args.problems
    else:
        # Apply command line filters to config
        if args.difficulty:
            config.setdefault('problem_filters', {})['difficulties'] = [args.difficulty]
        
        if args.platform:
            config.setdefault('problem_filters', {})['platforms'] = [args.platform]
        
        if args.max_problems:
            config.setdefault('problem_filters', {})['max_problems'] = args.max_problems
        
        try:
            problems = get_problems_from_config(config)
        except Exception as e:
            print(f"Error loading problems: {e}")
            print("Using fallback problem list...")
            problems = ['leetcode_1', 'leetcode_2']  # Fallback
    
    if not problems:
        print("No problems to test")
        sys.exit(1)
    
    print(f"Testing {len(problems)} problems with {len(agents)} agents")
    print(f"Problems: {problems[:5]}{'...' if len(problems) > 5 else ''}")
    print(f"Agents: {[a['name'] for a in agents]}")
    
    # Show model configurations for compatible agents only
    if args.model or args.reasoning_effort:
        compatible_selected = [a['name'] for a in agents if a.get('name', '').lower() in {'openai'}]
        if compatible_selected:
            print(f"Model overrides (for {', '.join(compatible_selected)}):")
            if args.model:
                print(f"  Model: {args.model}")
            if args.reasoning_effort:
                print(f"  Reasoning effort: {args.reasoning_effort}")
    
    print()
    
    # Initialize tester
    tester = AutomatedTester(config)
    
    try:
        # Run tests
        results = tester.test_batch(problems, agents)
        
        # Generate report
        report = tester.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        for agent_name, agent_stats in report.get('agents', {}).items():
            print(f"\n{agent_name.upper()}:")
            print(f"  Tests passed: {agent_stats['tests_passed']}/{agent_stats['total_problems']}")
            print(f"  Pass rate: {agent_stats['pass_rate']:.1%}")
            print(f"  Solutions generated: {agent_stats['solutions_generated']}/{agent_stats['total_problems']}")
            print(f"  Generation rate: {agent_stats['generation_rate']:.1%}")
            print(f"  Average runtime: {agent_stats['average_runtime']:.1f}s")
        
        # Save detailed report
        report_file = Path(config.get('results_file', 'test_results.json')).with_suffix('.report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Full results saved to: {config.get('results_file', 'test_results.json')}")
        
        # Print organized run summary
        print(f"\n{'='*60}")
        print(f"📁 RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Run directory: {run_dir}")
        print(f"├── workspaces/           # Individual problem workspaces")
        print(f"├── results/              # Agent results and logs by problem")
        print(f"├── test_results.json     # Complete test results")
        print(f"├── test_results.report.json  # Summary report")
        print(f"└── config.yaml           # Configuration used for this run")
        print(f"\n💡 Navigate to {run_dir} to explore all logs and results!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        tester.cleanup()


if __name__ == '__main__':
    main()