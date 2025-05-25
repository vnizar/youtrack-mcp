# YouTrack MCP Client

A Python-based client application that integrates with YouTrack using the Model-Controller-Presenter (MCP) pattern. This application allows you to interact with YouTrack issues using various AI models including Google's Gemini, Deepseek, and local Ollama models. Only support get issue detail and post comment for now.

## Features

- Integration with YouTrack for issue management
- Support for multiple AI models:
  - Google Gemini (cloud-based)
  - Deepseek (cloud-based)
  - Ollama (local models, including qwen2.5:7b)
- Interactive command-line interface
- Tool-based AI agent for issue management
- Customizable model parameters (temperature, model selection)

## Prerequisites

- Python 3.8 or higher
- YouTrack instance with API access
- Required API keys:
  - `GEMINI_API_KEY` for Google Gemini
  - `DEEPSEEK_API_KEY` for Deepseek
  - `YOUTRACK_TOKEN` for YouTrack access
  - `YOUTRACK_URL` for YouTrack instance URL

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd youtrack
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export YOUTRACK_TOKEN="your-youtrack-token"
export YOUTRACK_URL="your-youtrack-instance-url"
```

## Usage

Run the client with default settings (Gemini model):
```bash
python client/client.py
```

### Command Line Options

- `--model-type`: Choose the AI model type
  - Options: `gemini`, `deepseek`, `ollama`
  - Default: `gemini`

- `--model-name`: Specify the model name
  - Default for Gemini: `gemini-2.5-pro-preview-05-06`
  - Default for Deepseek: `deepseek-chat`
  - Default for Ollama: `deepseek-llm:7b`

- `--temperature`: Set the model's temperature (0-1)
  - Default: 0

- `--ollama-base-url`: Set Ollama API base URL
  - Default: `http://localhost:11434`

### Examples

1. Using Ollama with default settings:
```bash
python client/client.py --model-type ollama
```

2. Using Deepseek with custom temperature:
```bash
python client/client.py --model-type deepseek --temperature 0.7
```

3. Using a specific Ollama model:
```bash
python client/client.py --model-type ollama --model-name llama2:7b
```

## Project Structure

```
youtrack/
├── client/
│   └── client.py      # Main client application
├── server/
│   └── youtrack.py    # YouTrack server implementation
├── README.md
└── requirements.txt
```

## Development

### Setting up Ollama

If you plan to use Ollama models:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Check the model and make sure it has `tools` tag. Otherwise, the model doesn't support function calls.
3. Pull the required model:
```bash
ollama pull qwen2.5:7b
```

### Adding New Models

To add support for a new AI model:

1. Add the model type to the `model_type` Literal in `get_llm()`
2. Implement the model initialization in `get_llm()`
3. Update the argument parser in `parse_args()`
4. Add appropriate environment variables if needed

## Acknowledgments

- YouTrack API
- LangChain
- Google Gemini
- Deepseek
- Ollama
