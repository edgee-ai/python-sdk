"""Example usage of Edgee Gateway SDK"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgee import Edgee

edgee = Edgee(os.environ.get("EDGEE_API_KEY", "test-key"))

# Test 1: Simple string input
print("Test 1: Simple string input")
response1 = edgee.send(
    model="gpt-4o",
    input="What is the capital of France?",
)
print(f"Content: {response1.choices[0].message['content']}")
print(f"Usage: {response1.usage}")
print()

# Test 2: Full input object with messages
print("Test 2: Full input object with messages")
response2 = edgee.send(
    model="gpt-4o",
    input={
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ],
    },
)
print(f"Content: {response2.choices[0].message['content']}")
print()

# Test 3: With tools
print("Test 3: With tools")
response3 = edgee.send(
    model="gpt-4o",
    input={
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ],
        "tool_choice": "auto",
    },
)
print(f"Content: {response3.choices[0].message.get('content')}")
print(f"Tool calls: {response3.choices[0].message.get('tool_calls')}")
