from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os

api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

def get_weather(location: str) :
    resp = "it's sunny"
    return resp


class OutputSchema(BaseModel):
    """Schema for response."""

    answer: str
    justification: str


llm = ChatOpenAI(model="gpt-4.1",openai_api_key = api_key)

structured_llm = llm.bind_tools(
    [get_weather],
    response_format=OutputSchema,
    strict=True,
)

# Response contains tool calls:
tool_call_response = structured_llm.invoke("What is the weather in SF?")

# structured_response.additional_kwargs["parsed"] contains parsed output
structured_response = structured_llm.invoke(
    "What weighs more, a pound of feathers or a pound of gold?"
)


print(f"structured_response.additional_kwargs['parsed'] : {structured_response.additional_kwargs['parsed']}\n\n")
print(f"structured_response : {structured_response.content}\n\n")
print(f"tool_call_response : {tool_call_response.content}\n\n")
