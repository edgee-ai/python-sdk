# Edgee Gateway SDK

Lightweight Python SDK for Edgee AI Gateway with built-in tool execution support.

## Installation

```bash
pip install edgee
```

## Usage

```python
import os
from edgee import Edgee

edgee = Edgee(os.environ.get("EDGEE_API_KEY"))
```

### Simple Input

```python
response = edgee.send(
    model="gpt-4o",
    input="What is the capital of France?",
)

print(response.text)
```

### Full Input with Messages

```python
response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    },
)
```

## Tools

The SDK supports two modes for working with tools:

### Simple Mode: Automatic Tool Execution

Use the `Tool` class with Pydantic models for automatic tool execution. The SDK will handle the entire agentic loop for you:

```python
from pydantic import BaseModel
from edgee import Edgee, Tool

edgee = Edgee(os.environ.get("EDGEE_API_KEY"))

# Define tool parameters with Pydantic
class WeatherParams(BaseModel):
    location: str

# Define the tool handler
def get_weather(params: WeatherParams) -> dict:
    # Your implementation here
    return {"temperature": 22, "condition": "sunny", "location": params.location}

# Create the tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    schema=WeatherParams,
    handler=get_weather,
)

# The SDK automatically:
# 1. Sends the request with tools
# 2. Executes tools when the model requests them
# 3. Sends results back to the model
# 4. Returns the final response
response = edgee.send(
    model="gpt-4o",
    input="What's the weather in Paris?",
    tools=[weather_tool],
)

print(response.text)
# "The weather in Paris is sunny with a temperature of 22°C."
```

#### Multiple Tools

```python
from typing import Literal
from pydantic import BaseModel

class CalculatorParams(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float
    b: float

def calculate(params: CalculatorParams) -> dict:
    ops = {
        "add": params.a + params.b,
        "subtract": params.a - params.b,
        "multiply": params.a * params.b,
        "divide": params.a / params.b if params.b != 0 else "Error: division by zero",
    }
    return {"result": ops[params.operation]}

calculator_tool = Tool(
    name="calculate",
    description="Perform arithmetic operations",
    schema=CalculatorParams,
    handler=calculate,
)

response = edgee.send(
    model="gpt-4o",
    input="What's 25 * 4, and what's the weather in London?",
    tools=[weather_tool, calculator_tool],
)
```

#### Configuration Options

```python
response = edgee.send(
    model="gpt-4o",
    input="Complex query requiring multiple tool calls",
    tools=[tool1, tool2],
    max_tool_iterations=15,  # Default: 10
)
```

### Advanced Mode: Manual Tool Handling

For full control over tool execution, use the advanced mode with raw tool definitions:

```python
response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                },
            },
        ],
        "tool_choice": "auto",
    },
)

# Handle tool calls manually
if response.tool_calls:
    print(response.tool_calls)
    # Execute tools and send results back...
```

## Streaming

```python
for chunk in edgee.stream(model="gpt-4o", input="Tell me a story"):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

#### Alternative: Using send(stream=True)

```python
for chunk in edgee.send(model="gpt-4o", input="Tell me a story", stream=True):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

## Response

```python
@dataclass
class SendResponse:
    choices: list[Choice]
    usage: Usage | None

    # Convenience properties
    text: str | None         # choices[0].message["content"]
    message: dict | None     # choices[0].message
    finish_reason: str | None  # choices[0].finish_reason
    tool_calls: list | None  # choices[0].message["tool_calls"]

@dataclass
class Choice:
    index: int
    message: dict  # {"role": str, "content": str | None, "tool_calls": list | None}
    finish_reason: str | None

@dataclass
class InputTokenDetails:
    cached_tokens: int

@dataclass
class OutputTokenDetails:
    reasoning_tokens: int

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_tokens_details: InputTokenDetails
    output_tokens_details: OutputTokenDetails
```

### Streaming Response

```python
@dataclass
class StreamChunk:
    choices: list[StreamChoice]

    # Convenience properties
    text: str | None         # choices[0].delta.content
    role: str | None         # choices[0].delta.role
    finish_reason: str | None  # choices[0].finish_reason

@dataclass
class StreamChoice:
    index: int
    delta: StreamDelta
    finish_reason: str | None

@dataclass
class StreamDelta:
    role: str | None
    content: str | None
    tool_calls: list[dict] | None
```

## API Reference

### `Tool` Class

```python
from pydantic import BaseModel
from edgee import Tool

class MyParams(BaseModel):
    param1: str
    param2: int

tool = Tool(
    name="my_tool",           # Unique tool name
    description="...",        # Tool description for the model
    schema=MyParams,          # Pydantic model for parameters
    handler=my_function,      # Function to execute
)

# Methods
tool.to_dict()               # Convert to OpenAI tool format
tool.execute(args)           # Validate and execute handler
```

### `create_tool` Helper

```python
from edgee import create_tool

tool = create_tool(
    name="my_tool",
    schema=MyParams,
    handler=my_function,
    description="...",
)
```
