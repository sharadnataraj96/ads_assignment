import os
import json
import base64
import tempfile
from typing import List, Dict, Any
import gradio as gr
from PIL import Image
import io

# Import the workflow components
from workflow import build_graph, State
from utils import scrape_pages
from langchain_openai import ChatOpenAI

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1", temperature=0.8, api_key=os.getenv("OPENAI_API_KEY"))

def process_workflow(image: Any, urls_text: str):
    """
    Process the workflow with the uploaded image and URLs
    """
    try:
        # Convert image to base64
        if image is None:
            return "Error: Please upload an image", "{}", "{}", "{}", []
        
        # Convert PIL image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            image.save(tmp_file.name, format="PNG")
            input_image_path = tmp_file.name
        
        # Parse URLs from text input
        urls_to_scrape = [url.strip().strip('"').strip("'") for url in urls_text.split('\n') if url.strip()]
        if not urls_to_scrape:
            return "Error: Please provide at least one URL", "{}", "{}", "{}", []
        
        # Scrape pages to get text corpus
        text_corpus, image_paths = scrape_pages(urls_to_scrape)
        
        # Create initial state
        initial_state = State(
            input_image_path=input_image_path,
            input_image_base64=img_str,
            urls_to_scrape=urls_to_scrape,
            text_corpus=text_corpus,
            image_paths=image_paths
        )
        
        # Build and run the graph
        graph = build_graph(llm)
        result = graph.invoke(initial_state)
        
        # Extract results
        analysis_agent_response = result.get("analysis_agent_response", {})
        variations_agent_response = result.get("variations_agent_response", {})
        validation_dict = result.get("validation_dict", {})
        generated_images_paths = result.get("generated_images_paths", [])
        
        # Format outputs
        analysis_json = json.dumps(analysis_agent_response, indent=2) if analysis_agent_response else "{}"
        variations_json = json.dumps(variations_agent_response, indent=2) if variations_agent_response else "{}"
        validation_text = json.dumps(validation_dict, indent=2) if validation_dict else "{}"
        
        # Prepare generated images for display
        generated_images = []
        for img_path in generated_images_paths:
            if os.path.exists(img_path):
                generated_images.append(img_path)
        
        # Clean up temporary file
        try:
            os.unlink(input_image_path)
        except:
            pass
        
        return "Success! Workflow completed.", analysis_json, variations_json, validation_text, generated_images
        
    except Exception as e:
        return f"Error: {str(e)}", "{}", "{}", "{}", []

def create_ui():
    """
    Create the Gradio UI
    """
    with gr.Blocks(title="Product Image Analysis & Variation Generator") as demo:
        gr.Markdown("# 🎨 Product Image Analysis & Variation Generator")
        gr.Markdown("Upload a product image and provide URLs to analyze and generate variations.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    label="Upload Product Image",
                    type="pil",
                    height=400
                )
                
                # URL input
                urls_input = gr.Textbox(
                    label="Enter URLs (one per line)",
                    placeholder="https://example.com/page1\nhttps://example.com/page2",
                    lines=5
                )
                
                # Generate button
                generate_btn = gr.Button("🚀 Generate Analysis & Variations", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Status output
                status_output = gr.Textbox(
                    label="Status",
                    lines=2
                )
        
        with gr.Row():
            with gr.Column():
                # Analysis results
                analysis_output = gr.JSON(
                    label="Analysis Results"
                )
            
            with gr.Column():
                # Variations results
                variations_output = gr.JSON(
                    label="Variations Results"
                )
        
        with gr.Row():
            # Validation results
            validation_output = gr.Textbox(
                label="Validation Results",
                lines=10
            )
        
        with gr.Row():
            # Generated images gallery
            generated_images_output = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=3,
                rows=2,
                height="auto"
            )
        
        # Connect the button to the processing function
        generate_btn.click(
            fn=process_workflow,
            inputs=[image_input, urls_input],
            outputs=[status_output, analysis_output, variations_output, validation_output, generated_images_output]
        )
        
        # Add some helpful information
        gr.Markdown("""
        ### How to use:
        1. **Upload an image** of your product using the image upload area
        2. **Enter URLs** of relevant web pages (one per line) that contain information about your product or brand
        3. **Click Generate** to start the analysis and variation generation process
        4. **View results** in the JSON and text outputs below
        
        ### What happens:
        - The system analyzes your product image and scraped text content
        - Generates creative variations of your product image
        - Evaluates the generated images using concept analysis
        - Provides detailed analysis and validation results
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()
    demo.launch(
        server_port=7860,
        share=True,
        show_error=True
    ) 