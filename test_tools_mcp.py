import json
import os

from openresponses_client import OpenResponsesClient, ToolRunner
from ollama_mcp_adapter import MCPHttpClient, MCPHttpConfig, MCPStdioClient, MCPStdioConfig, OllamaMCPAdapter

"""Demo: Ollama x MCP Adapter

This is a drop-in replacement for your original test_tools.py, except:
- tools_schema is auto-generated from MCP `tools/list`
- tool implementations forward to MCP `tools/call`

Usage
-----
# A) MCP via HTTP JSON-RPC
export MCP_TRANSPORT=http
export MCP_ENDPOINT_URL=http://localhost:8080/mcp

# Optional: whitelist tools
export MCP_ALLOWED_TOOLS=get_weather,search_docs

# B) MCP via STDIO
export MCP_TRANSPORT=stdio
export MCP_STDIO_CMD='["python","-m","my_mcp_server"]'

Then:
python test_tools_mcp.py
"""


def _parse_allowed_tools(env_val: str) -> list[str]:
    items = [x.strip() for x in (env_val or "").split(",")]
    return [x for x in items if x]


def build_adapter_from_env() -> OllamaMCPAdapter:
    transport = (os.getenv("MCP_TRANSPORT") or "http").strip().lower()
    allowed = _parse_allowed_tools(os.getenv("MCP_ALLOWED_TOOLS", ""))

    if transport == "http":
        endpoint = os.getenv("MCP_ENDPOINT_URL", "http://localhost:8080/mcp")
        # If your MCP server needs auth, add headers here.
        headers = {}
        # Example:
        # token = os.getenv("MCP_BEARER_TOKEN")
        # if token:
        #     headers["Authorization"] = f"Bearer {token}"

        http = MCPHttpClient(MCPHttpConfig(endpoint_url=endpoint, headers=headers or None, timeout_s=60))
        return OllamaMCPAdapter(http=http, allowed_tools=allowed or None)

    if transport == "stdio":
        raw = os.getenv("MCP_STDIO_CMD")
        if not raw:
            raise SystemExit("MCP_STDIO_CMD is required when MCP_TRANSPORT=stdio (expects JSON array string)")
        cmd = json.loads(raw)
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise SystemExit("MCP_STDIO_CMD must be a JSON array of strings, e.g. ['python','-m','server']")

        stdio = MCPStdioClient(MCPStdioConfig(cmd=cmd, timeout_s=60))
        stdio.start()
        return OllamaMCPAdapter(stdio=stdio, allowed_tools=allowed or None)

    raise SystemExit("MCP_TRANSPORT must be 'http' or 'stdio'")


# --- 1) Open Responses client (Ollama) ---
client = OpenResponsesClient(
    base_url="http://localhost:11434",
    api_key="ollama",
    timeout_s=300,
)


# --- 2) MCP adapter: auto-build tools_schema + tool_impls ---
adapter = build_adapter_from_env()

# Pull tools from MCP
mcp_tools_schema = adapter.build_tools_schema()

tool_impls = adapter.build_tool_impls()

runner = ToolRunner(
    client=client,
    tool_impls=tool_impls,
)


# --- 3) Execution ---
target_model = os.getenv("OLLAMA_MODEL", "llama3.2")

tool_choice_policy = "auto"

# Keep your instruction style (and make tool usage explicit)
final_instructions = """
你是台灣專業助理。
- 使用者提問若需要外部資料，請主動呼叫工具。
- 工具回傳後，用台灣繁體中文把結果轉成自然語言，不要解釋 JSON 欄位名稱。
- 如果同時查到多個地點或多筆資料，請做比較並給出建議。
"""

user_text = os.getenv(
    "USER_TEXT",
    "請幫我查台北跟高雄現在天氣，然後告訴我哪裡比較適合出遊？",
)

final_resp, history = runner.run(
    model=target_model,
    user_text=user_text,
    tools_schema=mcp_tools_schema,
    instructions=final_instructions,
    tool_choice=tool_choice_policy,
    max_rounds=6,
)


# --- 4) Debugging Results ---
print("\n" + "=" * 30)
print(f"[Info] MCP tools exposed to model: {[t['function']['name'] for t in mcp_tools_schema]}")

if len(history) > 1:
    print(f"[Success] Tool loop triggered. Total rounds: {len(history)}")
    print("--- Final Response ---")
    print(client.extract_text(final_resp))
else:
    print("[Fail] The model did not call any tool.")
    print(json.dumps(final_resp.raw, indent=2, ensure_ascii=False))
