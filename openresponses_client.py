"""
Open Responses Python Client (Unofficial Reference Implementation)
Supports:
- /v1/responses endpoint
- Server-Sent Events (SSE) Streaming
- Tool Calling (Function Calling) with auto-correction
- Robust error handling for Local LLMs (Ollama, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Callable, Tuple, Union

import requests


Json = Dict[str, Any]


# -----------------------------
# Data models (Typed Events & Objects)
# -----------------------------

@dataclass
class ResponseObject:
    """A tolerant wrapper for the /v1/responses response JSON (Non-streaming)."""
    raw: Json

    @property
    def id(self) -> Optional[str]:
        return self.raw.get("id")

    @property
    def status(self) -> Optional[str]:
        return self.raw.get("status")

    @property
    def output(self) -> List[Json]:
        out = self.raw.get("output")
        return out if isinstance(out, list) else []

    @property
    def usage(self) -> Json:
        return self.raw.get("usage", {})


@dataclass
class ToolCall:
    """Represents a tool/function call requested by the model."""
    name: str
    arguments: Json
    call_id: Optional[str] = None
    raw_item: Optional[Json] = None


# --- Streaming Event Classes ---

@dataclass
class StreamEvent:
    """Base class for all streaming events."""
    raw: Json
    
    @property
    def type(self) -> str:
        return self.raw.get("type", "unknown")


@dataclass
class ResponseCreated(StreamEvent):
    """Event: response.created"""
    @property
    def response_id(self) -> Optional[str]:
        # Structure: { "response": { "id": "..." } }
        return self.raw.get("response", {}).get("id")


@dataclass
class OutputItemAdded(StreamEvent):
    """Event: response.output_item.added"""
    @property
    def output_index(self) -> int:
        return self.raw.get("output_index", 0)
    
    @property
    def item(self) -> Json:
        return self.raw.get("item", {})


@dataclass
class ContentPartDelta(StreamEvent):
    """Event: response.content_part.delta (Text chunks)"""
    @property
    def delta(self) -> Union[str, Json]:
        return self.raw.get("delta", "")

    @property
    def text(self) -> str:
        """
        Helper: Automatically extracts text content from delta.
        Handles both string deltas and dict deltas (OpenAI/Anthropic style differences).
        """
        d = self.delta
        if isinstance(d, str):
            return d
        if isinstance(d, dict):
            return d.get("text", "") or d.get("value", "")
        return ""


@dataclass
class ResponseDone(StreamEvent):
    """Event: response.done (Final usage stats)"""
    @property
    def usage(self) -> Json:
        # Structure: { "usage": { "input_tokens": 10, ... } }
        return self.raw.get("usage", {})


# --- Event Factory ---

def event_factory(data: Json) -> StreamEvent:
    """
    Factory function to convert raw JSON events into Typed Event Objects.
    Uses loose matching ("in") to handle server variations.
    """
    evt_type = data.get("type", "")
    
    # [Robustness] Loose matching for 'delta' to catch variations like 'response.text.delta'
    if "delta" in evt_type:
        return ContentPartDelta(raw=data)
    
    elif evt_type == "response.created":
        return ResponseCreated(raw=data)
        
    elif "output_item.added" in evt_type:
        return OutputItemAdded(raw=data)
        
    elif evt_type == "response.done":
        return ResponseDone(raw=data)
    
    # Fallback for unknown events
    return StreamEvent(raw=data)


# -----------------------------
# Client
# -----------------------------

class OpenResponsesClient:
    """
    Client for Open Responses compatible servers.
    Handles authentication, request formatting, streaming, and tool extraction.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: int = 60,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def create(
        self,
        model: str,
        input: Union[str, List[Json], List[Dict[str, Any]]],
        instructions: Optional[str] = None,
        tools: Optional[List[Json]] = None,
        tool_choice: Optional[Union[str, Json]] = None,
        max_tool_calls: Optional[int] = None,
        stream: bool = False,
        previous_response_id: Optional[str] = None,
        **kwargs,
    ) -> Union[ResponseObject, Generator[StreamEvent, None, None]]:
        """
        Send a request to /v1/responses.
        
        Args:
            model: Model ID (e.g., 'llama3.2', 'gpt-4').
            input: User input string or list of content parts/outputs.
            instructions: System prompt.
            tools: List of tool definitions.
            tool_choice: "auto", "required", or specific tool.
            stream: Whether to stream the response.
            previous_response_id: For continuing a conversation.
        """
        url = f"{self.base_url}/v1/responses"

        payload = {
            "model": model,
            "input": input,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if max_tool_calls:
            payload["max_tool_calls"] = max_tool_calls
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        
        # Add extra args
        payload.update(kwargs)

        if stream:
            payload["stream"] = True
            r = self.session.post(
                url,
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=self.timeout_s,
            )
            # [Robustness] Force UTF-8 encoding to prevent mojibake
            r.encoding = "utf-8"
            
            r.raise_for_status()
            return self._iter_sse_events(r.iter_lines(decode_unicode=True))
        else:
            r = self.session.post(
                url,
                headers=self._headers(),
                json=payload,
                stream=False,
                timeout=self.timeout_s,
            )
            r.encoding = "utf-8"
            r.raise_for_status()
            return ResponseObject(r.json())

    @staticmethod
    def _iter_sse_events(lines: Iterable[str]) -> Generator[StreamEvent, None, None]:
        """
        Parses SSE lines and yields typed StreamEvent objects using the factory.
        """
        for line in lines:
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                    if isinstance(evt, dict):
                        # Use Factory to create typed objects
                        yield event_factory(evt)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def extract_text(resp: ResponseObject) -> str:
        """
        Helper to extract text from a non-streaming response.
        """
        text_parts = []
        for item in resp.output:
            if item.get("type") == "message":
                content = item.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
        return "".join(text_parts)

    @staticmethod
    def extract_text_from_event(event: StreamEvent) -> str:
        """
        Helper to extract text from a streaming event (Backward compatibility).
        """
        if isinstance(event, ContentPartDelta):
            return event.text
        return ""

    @staticmethod
    def find_tool_calls(resp: ResponseObject) -> List[ToolCall]:
        """
        Find tool calls in the response output.
        [Robustness] Includes fixes for missing 'name' fields from some Local LLMs.
        """
        calls: List[ToolCall] = []
        for item in resp.output:
            if not isinstance(item, dict):
                continue
            t = item.get("type")

            # Check for top-level tool_call/function_call
            if t in ("function_call", "tool_call"):
                name = item.get("name")
                raw_args = item.get("arguments", "{}")
                call_id = item.get("id") or item.get("call_id") or item.get("tool_call_id")
                args = _safe_json_args(raw_args)
                
                # [Robustness] Handle None name -> empty string
                if name is None:
                    name = ""
                
                if isinstance(name, str):
                    calls.append(ToolCall(name=name, arguments=args, call_id=call_id, raw_item=item))
                continue

            # Check for nested function objects
            fn = item.get("function")
            if isinstance(fn, dict):
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                call_id = item.get("id") or item.get("call_id") or item.get("tool_call_id")
                args = _safe_json_args(raw_args)
                
                if name is None:
                    name = ""
                    
                if isinstance(name, str):
                    calls.append(ToolCall(name=name, arguments=args, call_id=call_id, raw_item=item))

        return calls


def _safe_json_args(args_obj: Any) -> Dict[str, Any]:
    """Helper to parse tool arguments that might be strings or dicts."""
    if isinstance(args_obj, dict):
        return args_obj
    if isinstance(args_obj, str):
        try:
            return json.loads(args_obj)
        except json.JSONDecodeError:
            return {}
    return {}


# -----------------------------
# Tool Runner (Agent Loop)
# -----------------------------

class ToolRunner:
    """
    A simple loop to handle the turn-based tool calling flow.
    """
    def __init__(self, client: OpenResponsesClient, tool_impls: Dict[str, Callable[[Json], Any]]):
        self.client = client
        self.tool_impls = tool_impls

    def run(
        self,
        *,
        model: str,
        user_text: str,
        tools_schema: List[Json],
        instructions: Optional[str] = None,
        max_rounds: int = 8,
        max_tool_calls: int = 8,
        tool_choice: Optional[Union[str, Json]] = "auto",
    ) -> Tuple[ResponseObject, List[ResponseObject]]:
        """
        Executes the tool calling loop.
        
        Returns:
            (final_response, list_of_all_responses)
        """
        rounds: List[ResponseObject] = []

        # Round 1: Initial user request
        resp = self.client.create(
            model=model,
            input=user_text,
            instructions=instructions,
            tools=tools_schema,
            tool_choice=tool_choice,
            max_tool_calls=max_tool_calls,
        )
        assert isinstance(resp, ResponseObject)
        rounds.append(resp)

        for _ in range(max_rounds):
            calls = self.client.find_tool_calls(resp)
            
            # If no tools called, we are done
            if not calls:
                return resp, rounds

            outputs: List[Json] = []
            for call in calls:
                target_name = call.name
                
                # [Robustness] If server sent empty name & we have only 1 tool, assume it's that one.
                if not target_name and len(self.tool_impls) == 1:
                    target_name = list(self.tool_impls.keys())[0]
                    print(f"[ToolRunner] ⚠️ Warning: Received unnamed tool call. Auto-routing to '{target_name}'")

                fn = self.tool_impls.get(target_name)
                
                if fn is None:
                    tool_out = {"error": f"Tool '{target_name}' not implemented."}
                    status = "failed"
                else:
                    try:
                        # Execute the Python function
                        tool_out = fn(call.arguments)
                        status = "completed"
                    except Exception as e:
                        tool_out = {"error": str(e)}
                        status = "failed"

                # Build the output payload for the next turn
                outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "status": status,
                    "output": json.dumps(tool_out, ensure_ascii=False),
                })

            # Next round: Send tool outputs back to model
            resp = self.client.create(
                model=model,
                previous_response_id=resp.id,
                input=outputs,
            )
            assert isinstance(resp, ResponseObject)
            rounds.append(resp)

        return resp, rounds