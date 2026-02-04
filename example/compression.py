"""Example: Token compression with Edgee Gateway SDK

This example demonstrates how to:
1. Enable compression for a request
2. Set a custom compression rate
3. Access compression metrics from the response
"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgee import Edgee

# Initialize the client
edgee = Edgee(os.environ.get("EDGEE_API_KEY"))

print("=" * 70)
print("Edgee Token Compression Example")
print("=" * 70)
print()

# Example 1: Request with compression enabled
print("Example 1: Request with compression enabled")
print("-" * 70)
response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
        "enable_compression": True,
        "compression_rate": 0.5,
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
    print(f"  Input tokens:  {response.compression.input_tokens}")
    print(f"  Saved tokens:  {response.compression.saved_tokens}")
    print(f"  Compression rate: {response.compression.rate:.2%}")
    print(f"  Token savings: {response.compression.saved_tokens} tokens saved!")
else:
    print("No compression data available in response.")
    print("Note: Compression data is only returned when compression is enabled")
    print("      and supported by your API key configuration.")

print()
print("=" * 70)
