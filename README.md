Open Responses Python Client (Unofficial)
A robust, production-ready Python client for the Open Responses specification.

This client is designed to work seamlessly with Local LLMs (via Ollama, vLLM, etc.) and includes built-in workarounds for common edge cases found in smaller models (like Llama 3.2 3B) and local inference servers.

 Why use this client?
While complying with the official Open Responses standard, this client adds an extra layer of robustness:

 Mojibake Fixer: Automatically forces UTF-8 encoding for streaming responses, fixing garbled text (亂碼) often seen with Ollama/requests.

Tolerant Tool Calling: Includes a "Monkey Patch" for models (like Llama 3.2) that sometimes return empty tool names. If you only have one tool, it auto-routes the call.

Fuzzy Parameter Matching: Handles cases where small models hallucinate parameter names (e.g., guessing city instead of location).

Strongly Typed Events: Uses a Factory Pattern to convert raw SSE JSON events into Python objects (ContentPartDelta, ResponseDone, etc.) with full IDE autocomplete support.

Installation
Clone this repository:

Bash

git clone https://github.com/your-username/openresponses-python-client.git
cd openresponses-python-client
Install dependencies:

Bash

pip install requests
Quick Start
1. Streaming Chat (The Typewriter Effect)
Python

from openresponses_client import OpenResponsesClient, ContentPartDelta, ResponseDone

client = OpenResponsesClient(
    base_url="http://localhost:11434",
    api_key="ollama",
    timeout_s=300
)

# Enable streaming
stream = client.create(
    model="ministral-3:3b-instruct-2512-q4_K_M", # Or llama3.2
    input="Explain Quantum Computing in 3 sentences.",
    stream=True
)

print("Response: ", end="")
for event in stream:
    if isinstance(event, ContentPartDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ResponseDone):
        print(f"\n[Usage] {event.usage}")
2. Tool Calling (Function Calling)
This client includes a ToolRunner that handles the tool-use loop automatically.

Python

from openresponses_client import OpenResponsesClient, ToolRunner

# 1. Define your Python function
def get_weather(args):
    # The client handles parameter parsing automatically
    loc = args.get("location") or args.get("city") or "Unknown"
    return {"temp": 25, "condition": "Sunny", "location": loc}

# 2. Define Schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}]

# 3. Run the Agent
client = OpenResponsesClient(base_url="http://localhost:11434", api_key="ollama")
runner = ToolRunner(client, tool_impls={"get_weather": get_weather})

final_resp, history = runner.run(
    model="llama3.2",
    user_text="What's the weather in Taipei?",
    tools_schema=tools,
    tool_choice="auto" # Forces tool use if needed
)

print(client.extract_text(final_resp))
🛠️ Advanced Features
Event Factory & Typing
Instead of dealing with raw dictionaries, you work with dataclasses:

Python

# No more dict["delta"]["text"] lookups!
if isinstance(event, ContentPartDelta):
    print(event.text)
elif isinstance(event, ResponseCreated):
    print(event.response_id)
Robustness for Local LLMs
Many local setups (e.g., Ollama + Proxy) have subtle bugs. This client fixes them on the fly:

Missing Name Bug: If Llama 3.2 returns {"name": ""} for a tool call, the client auto-detects it and routes to the available tool.

Stream Encoding: Explicitly sets r.encoding = "utf-8" to prevent ISO-8859-1 fallbacks.

Project Structure
openresponses_client.py: The core library. Contains the Client, Event Models, and ToolRunner.

test.py: Example script for streaming chat.

test_tools.py: Example script for tool calling agent.

Compatibility
Tested with:

Server: Ollama (v0.5.x)

Models:

Ministral 3B

Llama 3.2 (1B/3B)
