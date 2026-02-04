"""Example: Token compression with Edgee Gateway SDK

This example demonstrates how to:
1. Enable compression for a request with a large input context
2. Set a custom compression rate
3. Access compression metrics from the response

IMPORTANT: Only USER messages are compressed. System messages are not compressed.
This example includes a large context in the user message to demonstrate meaningful
compression savings.
"""

import os
import sys

# Add parent directory to path for local testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgee import Edgee

# Initialize the client
edgee = Edgee(os.environ.get("EDGEE_API_KEY"))

# Large context document to demonstrate input compression
LARGE_CONTEXT = """
The History and Impact of Artificial Intelligence

Artificial intelligence (AI) has evolved from a theoretical concept to a
transformative technology that influences nearly every aspect of modern life.
The field began in earnest in the 1950s when pioneers like Alan Turing and
John McCarthy laid the groundwork for machine intelligence.

Early developments focused on symbolic reasoning and expert systems. These
rule-based approaches dominated the field through the 1970s and 1980s, with
systems like MYCIN demonstrating practical applications in medical diagnosis.
However, these early systems were limited by their inability to learn from data
and adapt to new situations.

The resurgence of neural networks in the 1980s and 1990s, particularly with
backpropagation algorithms, opened new possibilities. Yet it wasn't until the
2010s, with the advent of deep learning and the availability of massive datasets
and computational power, that AI truly began to revolutionize industries.

Modern AI applications span numerous domains:
- Natural language processing enables machines to understand and generate human language
- Computer vision allows machines to interpret visual information from the world
- Robotics combines AI with mechanical systems for autonomous operation
- Healthcare uses AI for diagnosis, drug discovery, and personalized treatment
- Finance leverages AI for fraud detection, algorithmic trading, and risk assessment
- Transportation is being transformed by autonomous vehicles and traffic optimization

The development of large language models like GPT, BERT, and others has
particularly accelerated progress in natural language understanding and generation.
These models, trained on vast amounts of text data, can perform a wide range of
language tasks with remarkable proficiency.

Despite remarkable progress, significant challenges remain. Issues of bias,
interpretability, safety, and ethical considerations continue to be areas of
active research and debate. The AI community is working to ensure that these
powerful technologies are developed and deployed responsibly, with consideration
for their societal impact.

Looking forward, AI is expected to continue advancing rapidly, with potential
breakthroughs in areas like artificial general intelligence, quantum machine
learning, and brain-computer interfaces. The integration of AI into daily life
will likely deepen, raising important questions about human-AI collaboration,
workforce transformation, and the future of human cognition itself.
"""

print("=" * 70)
print("Edgee Token Compression Example")
print("=" * 70)
print()

# Example: Request with compression enabled and large input
print("Example: Large user message with compression enabled")
print("-" * 70)
print(f"Input context length: {len(LARGE_CONTEXT)} characters")
print()

# NOTE: Only USER messages are compressed
# Put the large context in the user message to demonstrate compression
user_message = f"""Here is some context about AI:

{LARGE_CONTEXT}

Based on this context, summarize the key milestones in AI development in 3 bullet points."""

response = edgee.send(
    model="gpt-4o",
    input={
        "messages": [
            {"role": "user", "content": user_message},
        ],
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
    savings_pct = (
        (response.compression.saved_tokens / response.compression.input_tokens * 100)
        if response.compression.input_tokens > 0
        else 0
    )
    print(f"  Savings: {savings_pct:.1f}% of input tokens saved!")
    print()
    print("  💡 Without compression, this request would have used")
    print(f"     {response.compression.input_tokens} input tokens.")
    print(
        f"     With compression, only {response.compression.input_tokens - response.compression.saved_tokens} tokens were processed!"
    )
else:
    print("No compression data available in response.")
    print("Note: Compression data is only returned when compression is enabled")
    print("      and supported by your API key configuration.")

print()
print("=" * 70)
