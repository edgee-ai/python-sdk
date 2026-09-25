"""Example: Token compression with Edgee Gateway SDK

This example demonstrates how to:
1. Turn tool-result trimming on for a single request
2. Access compression metrics from the response

Tool-result trimming shortens the output of tool calls (here a long `ls -la`
listing) before it reaches the model. The per-request toggles
(`tool_result_trimming`, `tool_surface_reduction`, `output_brevity`) override
the API key settings for this request only; leave one out to keep the key's
setting.
"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgee import Edgee

# Initialize the client
edgee = Edgee(os.environ.get("EDGEE_API_KEY"))

# A long directory listing, the kind of tool output coding agents send back.
LS_OUTPUT = "total 800\n" + "\n".join(
    f"-rw-r--r--  1 user  staff  {1000 + i} Jan  1 12:00 src/components/module_{i:03}.tsx"
    for i in range(200)
)

print("=" * 70)
print("Edgee Token Compression Example")
print("=" * 70)
print()

print("Example: Large tool result with tool-result trimming turned on")
print("-" * 70)
print(f"Tool output length: {len(LS_OUTPUT)} characters")
print()

response = edgee.send(
    model="anthropic/claude-haiku-4-5",
    input={
        "messages": [
            {"role": "user", "content": "How many files are in src/components?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": '{"command":"ls -la src/components"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": LS_OUTPUT},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "description": "Run a shell command and return its output.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "tool_result_trimming": True,
    },
)

print(f"Response: {response.text}")
print()

# Display usage information
if response.usage:
    print("Token Usage:")
    print(f"  Prompt tokens:     {response.usage.prompt_tokens}")
    print(f"  Completion tokens: {response.usage.completion_tokens}")
    print(f"  Total tokens:      {response.usage.total_tokens}")
    print()

# Display compression information
if response.compression:
    print("Compression Metrics:")
    print(f"  Saved tokens:  {response.compression.saved_tokens}")
    print(f"  Reduction:     {response.compression.reduction}%")
    print(f"  Cost savings:  ${response.compression.cost_savings / 1_000_000:.3f}")
    print(f"  Time:          {response.compression.time_ms} ms")
    if response.compression.reduction > 0:
        original_tokens = response.compression.saved_tokens * 100 // response.compression.reduction
        tokens_after = original_tokens - response.compression.saved_tokens
        print()
        print("  💡 Without compression, this request would have used")
        print(f"     {original_tokens} input tokens.")
        print(f"     With compression, only {tokens_after} tokens were processed!")
else:
    print("No compression data available in response.")
    print("Note: Compression data is only returned when trimming actually shortened")
    print("      a tool result.")

print()
print("=" * 70)
