import json
import sys
from openresponses_client import OpenResponsesClient, ToolRunner

def get_current_weather(args: dict):
    print(f"\n[Debug] 🔧 Tool received raw args: {args}")
    
    target_value = None
    possible_keys = ["location", "city", "place", "area"]
    
    for key, val in args.items():
        if key.lower() in possible_keys:
            target_value = val
            break

    if target_value is None:
        target_value = str(args)

    location_str = str(target_value)
    
    if "Taipei" in location_str:
        return {"temperature": 22, "condition": "cloud", "description": "It's a bit chilly, bring a jacket"}
    elif "Kaohsiung" in location_str:
        return {"temperature": 29, "condition": "sun", "description": "It's very hot, be sure to use sunscreen."}
    else:
        return {"temperature": 20, "condition": "unknown", "raw_input": location_str}

# --- 2. Schema ---
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a specific location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string", 
                        "description": "City name (e.g. Taipei, Kaohsiung)"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# --- 3. Client Setup ---
client = OpenResponsesClient(
    base_url="http://localhost:11434",
    api_key="ollama",
    timeout_s=300,
)

runner = ToolRunner(
    client=client,
    tool_impls={"get_current_weather": get_current_weather}
)

# --- 4. Execution ---
target_model = "llama3.2" 
tool_choice_policy = "auto" 

# [Fix] 強化指令：禁止解釋 JSON，強制扮演氣象員
final_instructions = """
You are a professional weather assistant in Taiwan.
1. When users inquire, you **must** use the `get_current_weather` tool to retrieve data.
2. After receiving the data, **please answer in Traditional Chinese**.
3. **Do not** interpret JSON format or field names; please translate the data directly into natural spoken language.
4. Compare the weather in the two locations based on the data and provide clothing or travel suggestions.
"""

final_resp, history = runner.run(
    model=target_model,
    user_text="Please check the current weather in Taipei and Kaohsiung and tell me which place is better for a trip?",
    tools_schema=tools_schema,
    instructions=final_instructions,
    tool_choice=tool_choice_policy,
    max_rounds=5
)

# --- 5. Debugging Results ---
print("\n" + "="*30)
if len(history) > 1:
    print(f"[Success] Tool successfully triggered! A total of {len(history)} rounds of dialogue were conducted.")
    print("--- Final Response ---")
    print(client.extract_text(final_resp))
else:
    print("[Fail] The model has no trigger tool.")
    print(json.dumps(final_resp.raw, indent=2, ensure_ascii=False))


