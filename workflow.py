import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
import json
from typing import TypedDict, Optional,Dict,List,Any
import base64
import argparse
from collections import defaultdict


import clip
import torch

from PIL import Image

from huggingface_hub import InferenceClient

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,BaseMessage,AIMessage,HumanMessage
from langchain.tools import tool

from pydantic import BaseModel

import splice


from utils import scrape_pages,get_variation_by_title

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

    print("ENTERING ANALYSIS NODE")
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

    print("ENTERING INIT VAR MESSAGES")
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
    print("ENTERING GENERATE VARIATIONS NODE")
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
    print("ENTERING GENERATE IMAGES NODE")
    output_dir = "output_images"
    os.makedirs(output_dir,exist_ok=True)
    generated_images_paths:List[str] = []

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
            prompt=variation_prompt+"remove all text"+"do not change the product",
            model="black-forest-labs/FLUX.1-Kontext-dev",
        )

        image_path = os.path.join(output_dir,f"{variation['title'].strip().replace(' ','_')}.png")
        generated_images_paths.append(image_path)
        image.save(image_path)

    print("Images generated")
        
    return {
        "generated_images_paths":generated_images_paths
    }

def get_splice_inputs(vocab:List[str]):
    """
    Get the splice inputs for the vocabulary
    Inputs:
    - vocab: List[str]
    - image_path: str
    Outputs:
    - splice_inputs: List[torch.Tensor]
    """

    model, _ = clip.load("ViT-B/32", device="cpu")
    
    
    concepts = []
    for line in vocab:
        with torch.no_grad():
            tokens = clip.tokenize(line).to("cpu")
            text_features = model.encode_text(tokens).to(torch.float32)
            mask = torch.isnan(text_features)
            text_features[mask] = 0
            text_features /= text_features.norm(dim=-1, keepdim=True)
            concepts.append(text_features)
    
    concepts_tensor = torch.stack(concepts).squeeze(1)
    concepts_norm = torch.nn.functional.normalize(concepts_tensor, dim=1)
    concepts_norm = torch.nn.functional.normalize(concepts_norm-torch.mean(concepts_norm, dim=0), dim=1)

    return concepts_norm






def evaluate_images_node(state:State):
    """
    Node for evaluating images
    Inputs:
    - state: State
    Outputs:
    - state: State
    """

    print("ENTERING EVALUATE IMAGES NODE")

    variations = state.get("variations_agent_response")["variations"]
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    for variation in variations:
        title = variation["title"].strip().replace(" ","_")
        vocab = variation["feature_unigrams"] + variation["feature_bigrams"]
        
        concepts_norm = get_splice_inputs(vocab)
        image_mean = torch.zeros_like(concepts_norm[0])

        splicemodel = splice.SPLICE(image_mean, concepts_norm, clip=model, device="cpu")

        preprocess = splice.get_preprocess("clip:ViT-B/32")
        image = Image.open(f"output_images/{title}.png")
        image_tensor = preprocess(image).unsqueeze(0).to("cpu")

        weights, l0, cosine = splice.decompose_image(image_tensor,splicemodel = splicemodel, device="cpu")

        validation_dict = defaultdict(list)

        with open("image_validation.txt","a") as f:
            f.write(f"{title}\n")
            for weight,concept in zip(weights.squeeze(0).cpu().numpy(),vocab):
                validation_dict[title].append((str(concept),float(weight)))
                f.write(f"{concept}: {weight}\n")
            f.write("\n")

    with open("image_validation.json","a") as f:
        json.dump(validation_dict,f,indent=4)
        



    return {
        "validation_dict":validation_dict
    }

            


        






        


            
    
    


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
    builder.add_node("evaluate_images",evaluate_images_node)
    builder.set_entry_point("analyze_product_img")
    builder.add_edge("analyze_product_img","generate_variations")
    builder.add_edge("generate_variations","generate_images")
    builder.add_edge("generate_images","evaluate_images")
    builder.set_finish_point("evaluate_images")

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



