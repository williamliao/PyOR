"""Ollama x MCP Adapter

This module bridges MCP tools into an Open Responses / OpenAI-compatible tool calling loop.

What it does
- Discovers tools from an MCP server via JSON-RPC 2.0 `tools/list`.
- Converts MCP tool definitions into OpenAI/Open-Responses `{"type":"function", "function": ...}` schema.
- When the model requests a tool call, forwards it to MCP `tools/call` and returns a JSON-serializable output.

Transports
- HTTP JSON-RPC (Streamable HTTP / SSE endpoints still accept POSTed JSON-RPC)
- STDIO JSON-RPC (launch an MCP server subprocess and speak JSON-RPC over stdin/stdout)

MCP spec reference for tool messages: `tools/list` and `tools/call`.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Iterable, Union, Tuple

import requests

Json = Dict[str, Any]


# -----------------------------
# MCP JSON-RPC: common helpers
# -----------------------------

class MCPError(RuntimeError):
    pass

def _is_event_stream(content_type: str | None) -> bool:
    if not content_type:
        return False
    return "text/event-stream" in content_type.lower()

def _parse_sse_first_jsonrpc(r: requests.Response) -> dict:
    """
    Parse SSE stream and return the first JSON object we can decode from 'data:' lines.
    Many MCP StreamableHTTP servers return JSON-RPC responses as SSE events.
    """
    # iter_lines will stream; decode_unicode makes it str
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        # SSE format: "data: {...json...}"
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                # Some servers may send multiple 'data:' fragments; ignore until a valid JSON appears
                continue
    raise MCPError("SSE stream ended without a valid JSON-RPC payload.")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_jsonable(x: Any) -> Any:
    """Best-effort conversion to something json.dumps can handle."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _coerce_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_coerce_jsonable(v) for v in x]
    # fallback
    return str(x)


def _mcp_result_to_plain(result: Json) -> Json:
    """Convert MCP ToolResult into a plain JSON object for the model.

    MCP ToolResult shape (spec):
      { "content": [ {"type":"text","text":"..."}, ... ], "isError": false, ... }

    We return:
      {"text": "..."} if we can find text content
      else: {"content": [...], "isError": bool, ...} (raw-ish)
    """
    if not isinstance(result, dict):
        return {"raw": _coerce_jsonable(result)}

    content = result.get("content")
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text")
                if isinstance(t, str) and t:
                    texts.append(t)
        if texts:
            return {
                "text": "\n".join(texts),
                "isError": bool(result.get("isError", False)),
            }

    # fallback: keep fields that are likely useful
    out: Json = {
        "isError": bool(result.get("isError", False)),
    }
    for k in ("content", "structuredContent", "error", "metadata"):
        if k in result:
            out[k] = _coerce_jsonable(result.get(k))
    # keep everything else (but jsonable)
    for k, v in result.items():
        if k not in out:
            out[k] = _coerce_jsonable(v)
    return out


# -----------------------------
# HTTP transport client
# -----------------------------

@dataclass
class MCPHttpConfig:
    endpoint_url: str
    headers: Optional[Dict[str, str]] = None
    timeout_s: int = 60
    # MCP protocol version to send in MCP-Protocol-Version header.
    # Most servers accept 2025-03-26; newer specs also list 2025-06-18 / 2025-11-25.
    protocol_version: str = "2025-03-26"


class MCPHttpClient:
    """Minimal MCP client over HTTP JSON-RPC.
    
    Updated: Added threading lock and strict param handling.
    """

    def __init__(self, cfg: MCPHttpConfig, session: Optional[requests.Session] = None):
        self.cfg = cfg
        self.session = session or requests.Session()
        self._id = 0
        self._session_id: Optional[str] = None
        self._initialized: bool = False
        self._negotiated_protocol_version: Optional[str] = None
        
        # FIX 1: Add the missing lock
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _post(self, payload: Json, *, allow_stream: bool = False) -> Tuple[Json, Dict[str, str]]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self._negotiated_protocol_version or self.cfg.protocol_version,
        }
        if self.cfg.headers:
            headers.update(self.cfg.headers)

        if self._session_id and payload.get("method") != "initialize":
            headers["Mcp-Session-Id"] = self._session_id

        try:
            r = requests.post(
                self.cfg.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.cfg.timeout_s,
                stream=True,
            )
        except requests.RequestException as e:
            raise MCPError(f"HTTP request failed: {e}") from e

        r.encoding = "utf-8"
        resp_headers = {k: v for k, v in r.headers.items()}

        sid = resp_headers.get("Mcp-Session-Id") or resp_headers.get("mcp-session-id")
        if sid:
            self._session_id = sid

        if r.status_code < 200 or r.status_code >= 300:
            body_head = ""
            try:
                body_head = (r.text or "")[:400]
            except Exception:
                body_head = "<unable to read body>"
            raise MCPError(
                f"MCP HTTP {r.status_code} (content-type={r.headers.get('content-type')}): {body_head}"
            )

        ct = r.headers.get("content-type", "")

        if _is_event_stream(ct):
            data = _parse_sse_first_jsonrpc(r)
            return data, resp_headers

        try:
            return r.json(), resp_headers
        except Exception as e:
            body_head = ""
            try:
                body_head = (r.text or "")[:400]
            except Exception:
                body_head = "<unable to read body>"
            raise MCPError(
                f"Invalid JSON-RPC response (content-type={ct}): {e}; body_head={body_head}"
            ) from e

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        init_id = self._next_id()
        init_payload = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": self.cfg.protocol_version,
                "clientInfo": {"name": "ollama-mcp-adapter", "version": "0.1"},
                "capabilities": {"tools": {}},
            },
        }
        data, _hdrs = self._post(init_payload)
        if "error" in data:
            raise MCPError(str(data["error"]))
        result = data.get("result", {})

        if isinstance(result, dict) and isinstance(result.get("protocolVersion"), str):
            self._negotiated_protocol_version = result["protocolVersion"]

        notif_payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            # Strict mode: no params for notifications if empty
        }
        try:
            self._post(notif_payload)
        except Exception:
            pass

        self._initialized = True

    def call(self, method: str, params: Optional[Json] = None) -> Json:
        if method != "initialize":
            self._ensure_initialized()
        
        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        
        # FIX 2: Only add 'params' if it exists (Same fix as Stdio)
        if params:
            payload["params"] = params
            
        data, _hdrs = self._post(payload)
        if "error" in data:
            raise MCPError(str(data["error"]))
        return data.get("result", {})

    def tools_list(self) -> List[Json]:
        tools: List[Json] = []
        cursor: Optional[str] = None
        while True:
            params = {} if cursor is None else {"cursor": cursor}
            # call() will strip empty params automatically
            result = self.call("tools/list", params)
            page = result.get("tools", [])
            if isinstance(page, list):
                tools.extend([t for t in page if isinstance(t, dict)])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def tools_call(self, name: str, arguments: Json) -> Json:
        return self.call("tools/call", {"name": name, "arguments": arguments or {}})


# -----------------------------
# STDIO transport client
# -----------------------------

@dataclass
class MCPStdioConfig:
    cmd: List[str]
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_s: int = 60


class MCPStdioClient:
    """Minimal MCP client over stdio JSON-RPC.
    
    Updated with:
    1. UTF-8 forcing for Windows (cp950 fix)
    2. Auto-initialization (MCP handshake)
    3. Correct parameter handling
    """

    def __init__(self, cfg: MCPStdioConfig):
        self.cfg = cfg
        self._id = 0
        self._initialized = False  # Track initialization state

        # FIX 1: Ensure the child process uses UTF-8 for I/O
        env = os.environ.copy()
        if cfg.env:
            env.update(cfg.env)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        self._proc = subprocess.Popen(
            cfg.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cfg.cwd,
            env=env,
            text=True,
            encoding="utf-8",  # Force Python to read UTF-8
            errors="replace",
            bufsize=1,
        )
        if not self._proc.stdin or not self._proc.stdout:
            raise MCPError("Failed to start MCP stdio process")

        self._lock = threading.Lock()

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _ensure_initialized(self):
        """Perform the MCP handshake: initialize -> notifications/initialized"""
        if self._initialized:
            return

        # 1. Send 'initialize' request
        init_payload = {
            "protocolVersion": "2024-11-05", 
            "capabilities": {
                "roots": {"listChanged": False},
                "sampling": {},
            },
            "clientInfo": {"name": "ollama-mcp-adapter", "version": "0.1"}
        }
        self.call("initialize", init_payload)

        # 2. Send 'notifications/initialized'
        notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        line = json.dumps(notif, ensure_ascii=False)
        with self._lock:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        
        self._initialized = True

    def call(self, method: str, params: Optional[Json] = None) -> Json:
        # Auto-initialize if not done yet (except for initialize itself)
        if method != "initialize" and not self._initialized:
            self._ensure_initialized()

        req_id = self._next_id()
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        
        # FIX 2: Only include 'params' if it contains actual data.
        if params:
            payload["params"] = params

        line = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            assert self._proc.stdin is not None
            assert self._proc.stdout is not None

            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

            deadline = time.time() + self.cfg.timeout_s
            while time.time() < deadline:
                resp_line = self._proc.stdout.readline()
                if not resp_line:
                    if self._proc.poll() is not None:
                        raise MCPError("MCP stdio process exited")
                    continue
                resp_line = resp_line.strip()
                if not resp_line:
                    continue
                try:
                    data = json.loads(resp_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and data.get("id") == req_id:
                    if "error" in data:
                        raise MCPError(str(data["error"]))
                    return data.get("result", {})

            raise MCPError(f"MCP stdio call timeout: {method}")

    def tools_list(self) -> List[Json]:
        tools: List[Json] = []
        cursor: Optional[str] = None
        while True:
            # We send empty dict here, but 'call' will strip it out because of FIX 2
            params = {} if cursor is None else {"cursor": cursor}
            result = self.call("tools/list", params)
            page = result.get("tools", [])
            if isinstance(page, list):
                tools.extend([t for t in page if isinstance(t, dict)])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def tools_call(self, name: str, arguments: Json) -> Json:
        return self.call("tools/call", {"name": name, "arguments": arguments or {}})


# -----------------------------
# Adapter: MCP tools -> Open Responses tools + tool_impls
# -----------------------------

class OllamaMCPAdapter:
    """Builds OpenAI/Open-Responses tool schema + tool implementations from an MCP server."""

    def __init__(
        self,
        *,
        http: Optional[MCPHttpClient] = None,
        stdio: Optional[MCPStdioClient] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        if (http is None) == (stdio is None):
            raise ValueError("Provide exactly one of: http= or stdio=")
        self.http = http
        self.stdio = stdio
        self.allowed_tools = set(allowed_tools or [])

    def _client(self) -> Union[MCPHttpClient, MCPStdioClient]:
        return self.http or self.stdio  # type: ignore

    def list_tools(self) -> List[Json]:
        tools = self._client().tools_list()
        if self.allowed_tools:
            tools = [t for t in tools if t.get("name") in self.allowed_tools]
        return tools

    @staticmethod
    def mcp_tool_to_openresponses_schema(t: Json) -> Json:
        name = t.get("name", "")
        if not isinstance(name, str) or not name:
            raise MCPError(f"Invalid MCP tool name: {name!r}")

        description = t.get("description") or t.get("title") or ""
        if not isinstance(description, str):
            description = str(description)

        input_schema = t.get("inputSchema")
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}

        # OpenAI/Open-Responses expects `parameters` as JSON Schema object
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        }

    def build_tools_schema(self) -> List[Json]:
        return [self.mcp_tool_to_openresponses_schema(t) for t in self.list_tools()]

    def build_tool_impls(self) -> Dict[str, Callable[[Json], Any]]:
        impls: Dict[str, Callable[[Json], Any]] = {}
        for t in self.list_tools():
            name = t.get("name")
            if not isinstance(name, str) or not name:
                continue

            def _make_impl(tool_name: str) -> Callable[[Json], Any]:
                def _impl(args: Json) -> Any:
                    result = self._client().tools_call(tool_name, args or {})
                    # If MCP returns isError, we still return the object; ToolRunner will mark completed.
                    # You can choose to raise MCPError here if you want status=failed.
                    return _mcp_result_to_plain(result)

                return _impl

            impls[name] = _make_impl(name)

        return impls
