import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Client (International Endpoint)
# We use the standard OpenAI client but point it to Alibaba Cloud.
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"), 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# ---------------------------------------------------------
# PART A: Define the "Real" Function (The Implementation)
# ---------------------------------------------------------
def get_current_weather(location, unit="celsius"):
    """
    Get the current weather in a given location.
    In a real app, you would call an external API (like OpenWeatherMap) here.
    """
    print(f" > [System Log] Tool triggered for: {location}")
    
    # Mock Data for demonstration
    loc = location.lower()
    if "singapore" in loc:
        return json.dumps({"location": "Singapore", "temperature": "30", "condition": "Thunderstorms"})
    elif "cairo" in loc:
        return json.dumps({"location": "Cairo", "temperature": "25", "condition": "Sunny"})
    elif "london" in loc:
        return json.dumps({"location": "London", "temperature": "10", "condition": "Foggy"})
    elif "tokyo" in loc:
        return json.dumps({"location": "Tokyo", "temperature": "18", "condition": "Clear"})
    else:
        # Fallback for unknown cities
        return json.dumps({"location": location, "temperature": "22", "condition": "Unknown"})

# ---------------------------------------------------------
# PART B: Define the Schema (The Menu for Qwen)
# ---------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Map string names to actual Python functions
available_functions = {
    "get_current_weather": get_current_weather,
}

# ---------------------------------------------------------
# PART C: The Agent Loop
# ---------------------------------------------------------
def run_agent():
    print("--- Qwen Agent Started (Type 'quit' to exit) ---")
    print("Try asking: 'What's the weather in Singapore?' or 'Is it raining in Cairo?'\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        # 1. Initialize conversation with the user's message
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools if needed."},
            {"role": "user", "content": user_input}
        ]

        # 2. First API Call: Ask Qwen "What do you want to do?"
        try:
            response = client.chat.completions.create(
                model="qwen-plus", # Strong reasoning capability
                messages=messages,
                tools=tools,
                tool_choice="auto", # Let Qwen decide if it needs the tool
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            # 3. Check if Qwen wants to use a tool
            if tool_calls:
                # print(f" > [Thinking] Qwen wants to call {len(tool_calls)} function(s)...")
                
                # Append Qwen's specific request to history so it remembers
                messages.append(response_message) 

                # 4. Execute the tool(s)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Run the Python code
                    function_response = function_to_call(
                        location=function_args.get("location"),
                        unit=function_args.get("unit"),
                    )
                    
                    # 5. Send result BACK to Qwen
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )

                # 6. Second API Call: Get the final natural language answer
                second_response = client.chat.completions.create(
                    model="qwen-plus",
                    messages=messages,
                )
                print(f"Qwen: {second_response.choices[0].message.content}\n")
            
            else:
                # If no tool was needed (e.g., "Hi, how are you?"), just print the answer
                print(f"Qwen: {response_message.content}\n")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_agent()
