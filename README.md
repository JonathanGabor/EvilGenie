# Agent Testing Framework

An independent framework for automated testing of coding agents on programming problems from competitive programming platforms like LeetCode, Codeforces, and AtCoder.

## Features

- **Multi-Agent Support**: Test various coding agents including Claude Code, OpenAI models (GPT-4, O3, O4), Gemini, and more
- **Problem Sources**: Access to thousands of problems from LiveCodeBench dataset via HuggingFace
- **Isolated Testing**: Each problem runs in its own workspace with standalone test runners
- **Comprehensive Monitoring**: Track solution progress, test results, and agent performance
- **Reward Hacking Detection**: Built-in detection for agents attempting to modify test files
- **Flexible Configuration**: YAML-based configuration with command-line overrides
- **Detailed Reporting**: Generate comprehensive reports with pass rates, timing, and logs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/livecodebench/agent-testing.git
cd agent-testing

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

1. **Configure your agents** in `test_config.yaml`:

```yaml
agents:
  - name: claude
    flags: []
  
  - name: openai
    llm_config:
      model: "gpt-4o-mini"
      temperature: 0.1

problem_filters:
  difficulties: [easy, medium]
  max_problems: 5
```

2. **Run tests**:

```bash
# Test all configured agents
python run_agent_tests.py

# Test specific agent with specific model
python run_agent_tests.py --agent openai --model o3-mini

# Test specific problems
python run_agent_tests.py --problem abc123_a --problem leetcode_456
```

3. **View results** in the generated `runs/` directory with detailed logs and reports.

## Supported Agents

| Agent | Description | Models |
|-------|-------------|---------|
| `claude` | Claude Code CLI | claude-3-5-sonnet, claude-3-opus |  
| `openai` | OpenAI Function Calling | gpt-4, gpt-4o, o3-mini, o4-mini |
| `gemini` | Gemini CLI | gemini-pro, gemini-1.5 |
| `smolagents` | HuggingFace SmolAgents | various HF models |

## Architecture

```
agent-testing/
├── src/
│   ├── agent_runners/    # Agent implementations
│   ├── testing/         # Core testing logic  
│   ├── workspace/       # Workspace management
│   ├── detection/       # Reward hacking detection
│   └── problems/        # Problem loading & data models
├── examples/            # Example configurations
├── docs/               # Documentation
└── tests/              # Unit tests
```

## Key Components

### Problem Loading
- Direct access to LiveCodeBench dataset via HuggingFace `datasets`
- No dependency on LiveCodeBench repository 
- Support for filtering by difficulty, platform, date range
- Extensible to other problem sources

### Workspace Management
- Isolated environments for each problem/agent combination
- Standalone test runners with no external dependencies
- Automatic cleanup with result preservation
- File monitoring and change detection

### Agent Runners
- Abstract base class for easy extension
- Built-in support for major coding agents
- Configurable timeouts, retries, and parameters
- Subprocess management with proper isolation

### Results & Reporting
- Detailed test results with timing and logs
- Pass rate analysis across agents and problems
- Organized output in timestamped run directories
- JSON and HTML report generation

## Configuration

The framework uses YAML configuration files with the following structure:

```yaml
# General settings
timeout: 300                    # Max time per test (seconds)
cleanup: true                   # Clean up workspaces after testing
log_level: INFO                 # Logging verbosity

# Agent configurations
agents:
  - name: openai
    llm_config:
      model: "gpt-4o-mini"
      temperature: 0.1
      reasoning_effort: "medium"  # For O3/O4 models

# Problem selection
problem_filters:
  difficulties: [easy, medium, hard]
  platforms: [leetcode, atcoder, codeforces]
  max_problems: 10
  date_range:
    start: "2024-01-01"
    end: "2024-12-31"

# Or specify exact problems
specific_problems:
  - leetcode_2933
  - abc123_a
  - cf789_A

# Reward hacking detection
reward_hacking_detection:
  enabled: true
  fail_fast: true
  monitored_files: [test.py, test_cases.json]
```

## Command Line Interface

```bash
# Basic usage
python run_agent_tests.py [OPTIONS]

# Agent selection
--agent AGENT               # Test specific agent
--agents AGENT1 AGENT2      # Test multiple agents

# Model configuration (for compatible agents)
--model MODEL               # Override model (gpt-4, o3-mini, etc.)
--reasoning-effort EFFORT   # For O3/O4 models (low/medium/high)

# Problem selection  
--problem PROBLEM_ID        # Test specific problem
--problems ID1 ID2          # Test multiple problems
--difficulty DIFFICULTY     # Filter by difficulty
--platform PLATFORM        # Filter by platform
--max-problems N            # Limit number of problems

# Test configuration
--timeout SECONDS           # Override timeout
--no-cleanup               # Keep workspace directories
--config CONFIG_FILE        # Use different config file

# Output options
--output FILE              # Save results to file
--quiet                    # Reduce output
--verbose                  # Increase output
```

## Examples

### Test OpenAI O3 model on easy problems:
```bash
python run_agent_tests.py --agent openai --model o3-mini --difficulty easy --max-problems 5
```

### Compare multiple agents on specific problems:
```bash
python run_agent_tests.py --agents claude openai gemini --problems abc123_a leetcode_456 cf789_A
```

### Use custom configuration:
```bash
python run_agent_tests.py --config my_config.yaml --no-cleanup --verbose
```

## Development

### Adding New Agents

1. Create a new agent runner class extending `AgentRunner`
2. Implement the required methods (`run_agent`, `cleanup`, etc.)
3. Register the agent in `create_agent_runner()` factory function
4. Add configuration schema and documentation

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run integration tests (requires API keys)
pytest -m integration tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## Migration from LiveCodeBench

If you're migrating from the original LiveCodeBench agent testing:

1. **Update imports**: Change `from lcb_runner.benchmarks.code_generation import *` to `from problems import *`
2. **Update paths**: The new structure uses `src/` subdirectories
3. **Update configuration**: Agent configurations may need minor adjustments
4. **Update dependencies**: Install the new `agent-testing` package instead of full LiveCodeBench

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/livecodebench/agent-testing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/livecodebench/agent-testing/discussions)
- **Documentation**: [Full Documentation](https://livecodebench.github.io/agent-testing)