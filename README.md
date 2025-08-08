# Product Image Analysis & Variation Generator

This project provides a comprehensive workflow for analyzing product images and generating creative variations using AI-powered analysis and image generation.

## Features

- **Image Analysis**: Analyzes product images using GPT-4 Vision
- **Web Scraping**: Extracts relevant information from provided URLs
- **Variation Generation**: Creates creative variations of product images using FLUX.1
- **Concept Validation**: Evaluates generated images using SPLICE concept analysis
- **Web UI**: Easy-to-use Gradio interface for demo and testing

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Set the following environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export HF_TOKEN="your_huggingface_token"
   ```

## Usage

### Web UI (Recommended for Demo)

Run the Gradio interface:
```bash
python main.py
```

This will start a web server at `http://localhost:7860` with the following features:

- **Image Upload**: Drag and drop or click to upload product images
- **URL Input**: Enter multiple URLs (one per line) for web scraping
- **Generate Button**: Start the analysis and variation generation process
- **Results Display**: View analysis, variations, and validation results in JSON format

### Command Line Interface

For programmatic use, you can also use the original command-line interface:
```bash
python workflow.py --image_path path/to/image.jpg --urls "https://example.com/page1" "https://example.com/page2"
```

## Workflow Steps

1. **Image Analysis**: The system analyzes the uploaded product image using GPT-4 Vision
2. **Text Corpus**: Scrapes content from provided URLs to build a text corpus
3. **Variation Generation**: Creates creative variations based on the analysis
4. **Image Generation**: Uses FLUX.1 model to generate new product images
5. **Validation**: Evaluates generated images using SPLICE concept analysis

## Output

The system generates:
- **Analysis Results**: JSON containing color palette, scene description, emotional triggers, etc.
- **Variations Results**: JSON containing generated variation details and prompts
- **Generated Images**: Saved to `output_images/` directory
- **Validation Results**: Concept analysis scores for generated images

## File Structure

```
├── main.py              # Gradio UI entry point
├── workflow.py          # Main workflow logic
├── models.py            # Data models and schemas
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── system_prompts/      # System prompts for AI agents
├── output_images/       # Generated image outputs
└── README.md           # This file
```

## Requirements

- Python 3.8+
- OpenAI API key
- Hugging Face token
- Sufficient disk space for generated images

## Troubleshooting

- Ensure all environment variables are set correctly
- Check that the required API keys have sufficient credits
- Verify that the output_images directory is writable
- For large images, processing may take several minutes 