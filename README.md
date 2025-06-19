# LLM C/C++ Bugfix Benchmark

This repository provides an end-to-end pipeline for evaluating large language models on real-world C/C++ bugfixes.

## Structure
- `collection/` - Build benchmark data from C/C++ repositories
- `inference/` - Run LLM models
- `evaluation/` - Evaluate generated code

## Requirements
- Python 3.8+
- OpenAI API key (or configure for open models)
- Docker (for testing)
