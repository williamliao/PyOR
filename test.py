import os
from openresponses_client import OpenResponsesClient, ContentPartDelta, ResponseCreated, ResponseDone

client = OpenResponsesClient(
    base_url="http://localhost:11434",
    api_key="ollama",
    timeout_s=300,
)

print("Sending a request (Strongly Typed Mode)...")

try:
    stream_iterator = client.create(
        model="ministral-3:8b-instruct-2512-q4_K_M",
        input="Please briefly introduce yourself.",
        stream=True
    )

    for event in stream_iterator:
        # 1. 處理 Response ID
        if isinstance(event, ResponseCreated):
            print(f"[System] Response Created ID: {event.response_id}")
            print("--- Start ---")

        # 2. Processing text streams
        elif isinstance(event, ContentPartDelta):
            print(event.text, end="", flush=True)

        # 3. Token statistics at the end of processing
        elif isinstance(event, ResponseDone):
            usage = event.usage
            in_tokens = usage.get("input_tokens", 0)
            out_tokens = usage.get("output_tokens", 0)
            print(f"\n\n[Stats] Input: {in_tokens} | Output: {out_tokens} | Total: {in_tokens + out_tokens}")

    print("\n--- End ---")

except Exception as e:
    print(f"\n[Error] An error occurred: {e}")