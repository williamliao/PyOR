import json
import os

from openresponses_client import OpenResponsesClient, ToolRunner
from ollama_mcp_adapter import MCPHttpClient, MCPHttpConfig, MCPStdioClient, MCPStdioConfig, OllamaMCPAdapter
from dotenv import load_dotenv

# 初始化
load_dotenv()

"""Demo: Ollama x MCP Adapter

This is a drop-in replacement for your original test_tools.py, except:
- tools_schema is auto-generated from MCP `tools/list`
- tool implementations forward to MCP `tools/call`
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
        token = os.getenv("MCP_BEARER_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

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
        return OllamaMCPAdapter(stdio=stdio, allowed_tools=allowed or None)

    raise SystemExit("MCP_TRANSPORT must be 'http' or 'stdio'")

def extract_all_text(resp) -> str:
    texts = []

    # 1) 如果是對像 (Object) 且有 output_text
    ot = getattr(resp, "output_text", None)
    if isinstance(ot, str) and ot.strip():
        texts.append(ot)

    # 2) 如果是字典 (Dict) 且符合 OpenAI 格式 (choices -> message -> content)
    # 這是你缺少的關鍵邏輯！
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

    # 3) 檢查 raw 屬性 (針對 OpenResponsesClient 的包裝物件)
    raw = getattr(resp, "raw", None)
    if isinstance(raw, dict):
        choices = raw.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        
        # Fallback: 找常見欄位
        for k in ("output_text", "text", "content"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v)

    return "\n".join(texts).strip()

# --- 1) Open Responses client (vllm) ---
client = OpenResponsesClient(
    base_url=os.getenv("VLLM_BASE_URL"),
    api_key=os.getenv("VLLM_API_KEY"),
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
target_model = os.getenv("VLLM_MODEL"),

tool_choice_policy = "auto"

final_instructions = """
You are a professional weather assistant in Taiwan.
1. When users inquire, you **must** use the `get_current_weather` tool to retrieve data.
2. **Do not** interpret JSON format or field names;
3. Compare the weather in the two locations based on the data and provide clothing or travel suggestions.
"""

user_text = os.getenv(
    "USER_TEXT",
    "Please check the current weather in Taipei and Kaohsiung and tell me which place is better for a trip?"
)

final_resp, history = runner.run(
    model=target_model,
    user_text=user_text,
    tools_schema=mcp_tools_schema,
    instructions=final_instructions,
    tool_choice=tool_choice_policy,
    max_rounds=6,
    mode="chat"
)

# --- 4) Debugging Results ---
print("\n" + "=" * 30)
print(f"[Info] MCP tools exposed to model: {[t['function']['name'] for t in mcp_tools_schema]}")

print("\n--- Debug: output item types ---")
try:
    out = getattr(final_resp, "output", None)
    tc = final_resp.get("choices",[{}])[0].get("message",{}).get("tool_calls")
except Exception as e:
    print("debug failed:", e)

final_text = extract_all_text(final_resp)

if len(history) > 1:
    print(f"[Success] Tool loop triggered. Total rounds: {len(history)}")
    print("--- Final Response ---")
    print(final_text if final_text else "[Empty final text - but tool loop succeeded]")
else:
    print("[Fail] The model did not call any tool.")
    if isinstance(final_resp, dict):
        print(json.dumps(final_resp, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(final_resp.raw, indent=2, ensure_ascii=False))
