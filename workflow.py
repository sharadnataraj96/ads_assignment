import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
import json
from typing import TypedDict, Optional,Dict,List,Any
import base64
import argparse

from PIL import Image

from huggingface_hub import InferenceClient

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,BaseMessage,AIMessage,HumanMessage
from langchain.tools import tool

from pydantic import BaseModel


from utils import scrape_pages

from models import State,AnalysisSchema,VariationsSchema


llm = ChatOpenAI(model="gpt-4.1",temperature =0.8,api_key = os.getenv("OPENAI_API_KEY"))


def analyze_product_img_node(state: State) -> Dict[str,Any]:

    """
    Node for the analysis agent
    Inputs:
    - state: State
    Outputs:
    - state: State
    """
    # Load system prompt
    system_prompt = open("system_prompts/brand_summary.txt", "r").read()

    # Get inputs
    text_corpus = state.get("text_corpus")
    product_shot_base64 = state.get("input_image_base64")

    # Compose messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text_corpus),
        HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{product_shot_base64}"}
            }
        ])
    ]

    # Bind structured LLM
    structured_llm = llm.bind_tools(
        tools=[],
        response_format=AnalysisSchema,
        strict=True,
    )

    # Invoke and parse response
    response = structured_llm.invoke(messages)
    parsed = response.additional_kwargs["parsed"]
    parsed_response_dict = parsed.dict()
    parsed_response_str = parsed.json()

    # Optionally keep the message trace (if you use it later)
    updated_messages = messages + [AIMessage(content=response.content)]

    # Return updated state
    return {
        "analysis_agent_response": parsed_response_dict,
        "analysis_agent_response_str": parsed_response_str,
        "analysis_agent_messages": updated_messages
    }




def init_var_messages(state:State):
    """
    Initialize the messages for the variations agent
    Inputs:
    - state: State
    Outputs:
    - variations_llm_messages: list[BaseMessage]
    """
    variation_system_prompt = open("system_prompts/variations.txt", "r").read()
    product_shot_base64 = state.get("input_image_base64")
    analysis = state.get("analysis_agent_response_str")
    text_corpus = state.get("text_corpus")

    variations_llm_messages = [
        SystemMessage(content=variation_system_prompt),
        HumanMessage(content="Here is the Text Corpus" + text_corpus),
        HumanMessage(content="Here is the analysis" + analysis),
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{product_shot_base64}"}}
        ])
    ]

    return variations_llm_messages
    
    
    

def generate_variations_node(state: State) -> Dict[str,Any]:
    """
    Node for the variations agent
    Inputs:
    - state: State
    Outputs:
    - state: State
    """
    # Init messages if not already present
    messages = state.get("variations_agent_messages")
    if not messages:
        messages = init_var_messages(state)

    # Bind tool-using LLM
    var_structured_llm = llm.bind_tools(
        tools=[],
        response_format=VariationsSchema,
        strict=True,
    )

    # Invoke and parse
    response = var_structured_llm.invoke(messages)
    parsed = response.additional_kwargs["parsed"]
    parsed_response_dict = parsed.dict()
    parsed_response_str = parsed.json()

    # Append new message
    updated_messages = messages + [AIMessage(content=response.content)]

    # Return updated state
    return {
        "variations_agent_response": parsed_response_dict,
        "variations_agent_response_str": parsed_response_str,
        "variations_agent_messages": updated_messages
    }

def generate_images_node(state:State):

    """
    Node for generating images
    Inputs:
    - state: State
    Outputs:
    - state: State
    """
    output_dir = "output_images"
    os.makedirs(output_dir,exist_ok=True)

    variations = state.get("variations_agent_response")

    input_image_path = state.get("input_image_path")
    hf_token = os.getenv("HF_TOKEN")

    with open(input_image_path, "rb") as image_file:
        input_image = image_file.read()

    client = InferenceClient(
            provider="replicate",
            api_key=hf_token,
        )

    for variation in variations["variations"]:
        variation_prompt = " ".join(variation["changes"])

        image = client.image_to_image(
            input_image,
            prompt=variation_prompt,
            model="black-forest-labs/FLUX.1-Kontext-dev",
        )

        image.save(os.path.join(output_dir,f"{variation['title']}.png"))
        
    return {}
        
        


def build_graph(llm:ChatOpenAI):
    """
    Build the graph for the workflow
    Inputs:
    - llm: ChatOpenAI
    Outputs:
    - graph: StateGraph
    """
    builder = StateGraph(State)

    builder.add_node("analyze_product_img",analyze_product_img_node)
    builder.add_node("generate_variations",generate_variations_node)
    builder.add_node("generate_images",generate_images_node)
    builder.set_entry_point("analyze_product_img")
    builder.add_edge("analyze_product_img","generate_variations")
    builder.add_edge("generate_variations","generate_images")
    builder.set_finish_point("generate_images")

    return builder.compile()


    

def main():
    """
    Main function for running the workflow
    Inputs:
    - None
    Outputs:
    - None
    """
    parser = argparse.ArgumentParser(description="Run the workflow with image path and URLs.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the product image')
    parser.add_argument('--urls', type=str, nargs='+', required=True, help='List of URLs')
    args = parser.parse_args()

    base64_image = base64.b64encode(open(args.image_path, "rb").read()).decode("utf-8")
    urls_to_scrape = args.urls


    text_corpus,image_paths = scrape_pages(urls_to_scrape)

    # Example: create initial state
    initial_state = State(
        input_image_path=args.image_path,
        input_image_base64=base64_image,
        urls_to_scrape=urls_to_scrape,
        text_corpus=text_corpus,
        image_paths=image_paths
    )

    # Build the graph with a ChatOpenAI instance (assume llm is defined elsewhere)
    graph = build_graph(llm)
    result = graph.invoke(initial_state)
    analysis_agent_response = result["analysis_agent_response"]
    variations_agent_response = result["variations_agent_response"]

    with open("analysis_agent_response.json","w") as f:
        json.dump(analysis_agent_response,f,indent=4)
    with open("variations_agent_response.json","w") as f:
        json.dump(variations_agent_response,f,indent=4)

if __name__ == "__main__":
    main()



