# Open Responses Python Client (Unofficial)

A robust, production-ready Python client for the Open Responses specification.

This client is designed to work seamlessly with **Local LLMs** (via Ollama, vLLM, etc.) and includes built-in workarounds for common edge cases found in smaller models (like Llama 3.2 3B) and local inference servers.

**New:** Now includes an **MCP Adapter** to use Model Context Protocol servers as tools!

---

## Why use this client?

While complying with the official Open Responses standard, this client adds an extra layer of robustness:

### Mojibake Fixer
Automatically forces UTF-8 encoding for streaming responses, fixing garbled text (亂碼) often seen with Ollama/requests.

### Tolerant Tool Calling
Includes a "Monkey Patch" for models (like Llama 3.2) that sometimes return empty tool names.
If you only have one tool, it auto-routes the call.

### Fuzzy Parameter Matching
Handles cases where small models hallucinate parameter names
(e.g., guessing `city` instead of `location`).

### MCP Support (Model Context Protocol)
Bridges the standard `tools/list` and `tools/call` from MCP servers into OpenAI/Ollama-compatible schemas. Supports both **HTTP** (SSE/JSON-RPC) and **STDIO** transports.

---

## Installation

Clone this repository:

```bash
git clone [https://github.com/your-username/openresponses-python-client.git](https://github.com/your-username/openresponses-python-client.git)
cd openresponses-python-client
```

Install dependencies:

```bash
pip install requests python-dotenv
```

---

## Quick Start

### 1. Streaming Chat (The Typewriter Effect)

```python
from openresponses_client import OpenResponsesClient, ContentPartDelta, ResponseDone

client = OpenResponsesClient(
    base_url="http://localhost:11434",
    api_key="ollama",
    timeout_s=300
)

# Enable streaming
stream = client.create(
    model="ministral-3:3b-instruct-2512-q4_K_M",  # Or llama3.2
    input="Explain Quantum Computing in 3 sentences.",
    stream=True
)

print("Response: ", end="")
for event in stream:
    if isinstance(event, ContentPartDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ResponseDone):
        print(f"\n[Usage] {event.usage}")
```

### 2. Tool Calling (Standard)

The client includes a `ToolRunner` that handles the tool-use loop automatically.

```python
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
            "properties": {
                "location": {"type": "string"}
            },
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
    tool_choice="auto"  # Forces tool use if needed
)

print(client.extract_text(final_resp))
```

---

## Ollama x MCP Adapter

Use standard [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers as tools for your local LLM. The adapter handles the protocol handshake, JSON-RPC, and schema conversion.

### Usage (STDIO Mode)

Ideal for running local python scripts or binaries as MCP servers.

**Environment Setup (`.env`):**
```bash
MCP_TRANSPORT=stdio
# Command to launch your MCP server (must be a JSON list of strings)
MCP_STDIO_CMD=["python", "my_weather_server.py"]
```

**Python Code:**
```python
from openresponses_client import OpenResponsesClient, ToolRunner
from ollama_mcp_adapter import MCPStdioClient, MCPStdioConfig, OllamaMCPAdapter
import json, os

# 1. Configure Adapter
cmd = json.loads(os.getenv("MCP_STDIO_CMD"))
adapter = OllamaMCPAdapter(
    stdio=MCPStdioClient(MCPStdioConfig(cmd=cmd))
)

# 2. Build Tools from MCP
mcp_tools = adapter.build_tools_schema()
tool_impls = adapter.build_tool_impls()

# 3. Run Agent
client = OpenResponsesClient(base_url="http://localhost:11434", api_key="ollama")
runner = ToolRunner(client=client, tool_impls=tool_impls)

resp, _ = runner.run(
    model="llama3.2",
    user_text="What is the weather in Taipei?",
    tools_schema=mcp_tools
)
```

### Usage (HTTP Mode)

Ideal for connecting to running MCP servers (e.g., n8n, remote servers).

**Environment Setup:**
```bash
MCP_TRANSPORT=http
MCP_ENDPOINT_URL=http://localhost:8080/mcp
```

### Key MCP Features

1.  **Windows UTF-8 Support:**
    The `MCPStdioClient` automatically forces `PYTHONUTF8=1` and `encoding="utf-8"` to prevent `UnicodeDecodeError` (cp950 issue) when using Chinese characters on Windows.

2.  **Strict Protocol Compliance:**
    * Handles the mandatory `initialize` -> `notifications/initialized` handshake automatically.
    * Strictly handles JSON-RPC `params` (omits them when empty) to satisfy strict servers like FastMCP.

---

## Project Structure

| File | Description |
|------|------------|
| `openresponses_client.py` | Core library: Client, Event Models, ToolRunner |
| `ollama_mcp_adapter.py` | **NEW**: Bridges MCP (HTTP/STDIO) to OpenResponses tools |
| `test.py` | Example script for streaming chat |
| `test_tools.py` | Example script for standard tool calling |
| `test_tools_mcp.py` | Example script for MCP tool calling |

---

## Compatibility

### Tested with

**Server**
- Ollama (v0.5.x)

**Models**
- Ministral 3B
- Llama 3.2 (1B / 3B)

**MCP Servers**
- Python `mcp` SDK (FastMCP)
- n8n MCP (HTTP)