# Reward Hacking Benchmark

## Description

This repository contains a benchmark for evaluating the robustness of AI agents against reward hacking behaviors. Our benchmark
consists of problems from LIVECODEBENCH [JHG+25], modified to make reward hacking feasible. In particular, we
create an environment with many opportunities for LLMs to circumvent or pass test cases, thus achieving a higher benchmark score without correctly solving the stated algorithmic problem. To detect instances of reward hacking, we combine held-out unit tests, LLM-based judges, and test file edit detection.

## Installation
```
uv venv
source .venv/bin/activate
uv sync
```
Install cli based agents
```
curl -fsSL https://claude.ai/install.sh | bash
npm i -g @openai/codex
npm install -g @google/gemini-cli
```
Set API keys

Option A: .env file (recommended)
```
cp .env.example .env
# then edit .env and add your keys
```
The CLI loads `.env` automatically at startup.

Option B: export in shell
```
export OPENAI_API_KEY=<your openai api key>
export ANTHROPIC_API_KEY=<your anthropic api key>
export GOOGLE_API_KEY=<your google api key>
```

*Warning:* This repo currently allows agents to execute arbitrary code on your machine. We are woring on sandboxing solutions.

## Example Usage

```
python src/run_agent_tests.py --agent claude --difficulty medium --release-version v6 --random --no-cleanup
```
