from typing import TypedDict, Optional,Dict,List,Any

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage,BaseMessage,AIMessage,HumanMessage
from langchain.tools import tool

from pydantic import BaseModel


class State(TypedDict):

    """
    State for the workflow
    analysis_agent_messages : list[BaseMessage]
    variations_agent_messages : list[BaseMessage]
    text_corpus : str
    analysis_agent_response : Dict[str,Any]
    variations_agent_response : Dict[str,Any]
    analysis_agent_response_str : str
    variations_agent_response_str : str
    input_image_base64 : str
    input_image_path : str
    urls_to_scrape : list[str]
    image_paths : list[str]
    """
    analysis_agent_messages : list[BaseMessage]
    variations_agent_messages : list[BaseMessage]

    text_corpus : str

    analysis_agent_response : Dict[str,Any]
    variations_agent_response : Dict[str,Any]
    analysis_agent_response_str : str
    variations_agent_response_str : str
    generated_images_paths : list[str]

    input_image_base64 : str
    input_image_path : str
    urls_to_scrape : list[str]
    image_paths : list[str]
    

class AnalysisSchema(BaseModel):
    """
    Schema for the analysis agent
    color_pallette : list[str]
    scene_description : list[str]
    visual_b_words : list[str]
    emotional_triggers : list[str]
    usage_insight : list[str]
    keywords_and_phrases : list[str]
    summary : str
    """
    color_pallette : list[str]
    scene_description : list[str]
    visual_b_words : list[str]
    emotional_triggers : list[str]
    usage_insight : list[str]
    keywords_and_phrases : list[str]
    summary : str

class VariationFormat(BaseModel):
    """
    Schema for the variations agent
    title : str
    audience : list[str]
    changes : list[str]
    """
    title : str
    audience : list[str]
    changes : list[str]
    feature_unigrams : list[str]
    feature_bigrams : list[str]

class VariationsSchema(BaseModel):
    """
    Schema for the variations agent
    variations : list[VariationFormat]
    """

    variations : list[VariationFormat]

