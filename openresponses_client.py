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
from typing import Any, Dict, Generator, Iterable, List, Optional, Callable, Tuple, Union, cast

import requests


Json = Dict[str, Any]

# In this repo we support two server styles:
# - Open Responses /v1/responses -> ResponseObject
# - OpenAI-compatible /v1/chat/completions -> plain dict JSON
ResponseLike = Union["ResponseObject", Json]


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

    def create_chat(
        self,
        *,
        model: str,
        messages: List[Json],
        tools: Optional[List[Json]] = None,
        tool_choice: Optional[Union[str, Json]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Json:
        """Send a request to /v1/chat/completions (OpenAI-compatible).

        Notes:
        - Ollama's tool calling support is currently much more reliable on chat/completions
          than on /v1/responses for many local models.
        - We return raw dict JSON (not ResponseObject).
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload: Json = {
            "model": model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        payload.update(kwargs)

        # Streaming for chat/completions isn't wired into this client.
        if stream:
            raise NotImplementedError("Streaming chat/completions is not implemented in this client.")

        r = self.session.post(
            url,
            headers=self._headers(),
            json=payload,
            stream=False,
            timeout=self.timeout_s,
        )
        r.encoding = "utf-8"
        r.raise_for_status()
        data = r.json()
        return cast(Json, data)

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
    def extract_text(resp: ResponseLike) -> str:
        """Extract text for either /v1/responses (ResponseObject) or chat/completions (dict)."""
        # chat/completions
        if isinstance(resp, dict):
            try:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        return c if isinstance(c, str) else ""
            except Exception:
                return ""
            return ""

        # /v1/responses
        text_parts: List[str] = []
        for item in resp.output:
            if item.get("type") == "message":
                content = item.get("content", [])
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                            t = part.get("text")
                            if isinstance(t, str):
                                text_parts.append(t)
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
    def find_tool_calls(resp: ResponseLike) -> List[ToolCall]:
        """Find tool calls for both /v1/responses and chat/completions.

        - /v1/responses: scan resp.output items
        - chat/completions: scan resp["choices"][0]["message"]["tool_calls"]

        [Robustness]
        - name may be missing/None on some local servers
        - arguments may be JSON string or dict
        """

        def _coerce_name(x: Any) -> str:
            if x is None:
                return ""
            return x if isinstance(x, str) else str(x)

        def _coerce_args(x: Any) -> Json:
            if isinstance(x, dict):
                return cast(Json, x)
            if x is None:
                return {}
            return cast(Json, _safe_json_args(x if isinstance(x, str) else str(x)))

        calls: List[ToolCall] = []

        # chat/completions
        if isinstance(resp, dict):
            try:
                choices = resp.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict):
                        tc_list = msg.get("tool_calls")
                        if isinstance(tc_list, list):
                            for tc in tc_list:
                                if not isinstance(tc, dict):
                                    continue
                                fn = tc.get("function")
                                if not isinstance(fn, dict):
                                    continue
                                name = _coerce_name(fn.get("name"))
                                args = _coerce_args(fn.get("arguments"))
                                call_id = tc.get("id") or tc.get("tool_call_id") or tc.get("call_id")
                                calls.append(ToolCall(name=name, arguments=args, call_id=call_id, raw_item=tc))
                return calls
            except Exception:
                # fall through to /v1/responses parsing
                pass

        # /v1/responses
        if not isinstance(resp, ResponseObject):
            return calls

        for item in resp.output:
            if not isinstance(item, dict):
                continue
            t = item.get("type")

            # Top-level tool_call/function_call
            if t in ("function_call", "tool_call"):
                name = _coerce_name(item.get("name"))
                args = _coerce_args(item.get("arguments"))
                call_id = item.get("id") or item.get("call_id") or item.get("tool_call_id")
                calls.append(ToolCall(name=name, arguments=args, call_id=call_id, raw_item=item))
                continue

            # Nested function object
            fn = item.get("function")
            if isinstance(fn, dict):
                name = _coerce_name(fn.get("name"))
                args = _coerce_args(fn.get("arguments"))
                call_id = item.get("id") or item.get("call_id") or item.get("tool_call_id")
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
        mode: str = "responses",
    ) -> Tuple[ResponseLike, List[ResponseLike]]:
        """
        Executes the tool calling loop.
        
        Returns:
            (final_response, list_of_all_responses)
        """
        rounds: List[ResponseLike] = []

        # -------------------------
        # Mode A) /v1/responses loop
        # -------------------------
        if mode == "responses":
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
                            tool_out = fn(call.arguments)
                            status = "completed"
                        except Exception as e:
                            tool_out = {"error": str(e)}
                            status = "failed"

                    outputs.append({
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "status": status,
                        "output": json.dumps(tool_out, ensure_ascii=False),
                    })

                resp = self.client.create(
                    model=model,
                    previous_response_id=resp.id,
                    input=outputs,
                )
                assert isinstance(resp, ResponseObject)
                rounds.append(resp)

            return resp, rounds

        # ------------------------------
        # Mode B) chat/completions loop
        # ------------------------------
        if mode == "chat":
            messages: List[Json] = []
            if instructions:
                messages.append({"role": "system", "content": instructions})
            messages.append({"role": "user", "content": user_text})

            resp = self.client.create_chat(
                model=model,
                messages=messages,
                tools=tools_schema,
                tool_choice=tool_choice,
            )
            rounds.append(resp)

            for _ in range(max_rounds):
                calls = self.client.find_tool_calls(resp)
                if not calls:
                    return resp, rounds

                # Append assistant message (which contains tool_calls) back into the chat history
                try:
                    choices = resp.get("choices") if isinstance(resp, dict) else None
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                        msg = choices[0].get("message")
                        if isinstance(msg, dict):
                            messages.append(cast(Json, msg))
                except Exception:
                    pass

                for call in calls:
                    target_name = call.name
                    if not target_name and len(self.tool_impls) == 1:
                        target_name = list(self.tool_impls.keys())[0]
                        print(f"[ToolRunner] ⚠️ Warning: Received unnamed tool call. Auto-routing to '{target_name}'")

                    fn = self.tool_impls.get(target_name)
                    if fn is None:
                        tool_out = {"error": f"Tool '{target_name}' not implemented."}
                    else:
                        try:
                            tool_out = fn(call.arguments)
                        except Exception as e:
                            tool_out = {"error": str(e)}

                    # role=tool message (OpenAI-compatible)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.call_id or "",
                        "content": json.dumps(tool_out, ensure_ascii=False),
                    })

                resp = self.client.create_chat(
                    model=model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice=tool_choice,
                )
                rounds.append(resp)

            return resp, rounds

        raise ValueError("mode must be 'responses' or 'chat'")