import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

def chat_with_qwen():
    # 2. Initialize the Client
    # This is the "Magic Trick". We use the OpenAI client, but we point
    # the 'base_url' to Alibaba Cloud's International Endpoint.
    client = OpenAI(
        # Get key from .env file
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        
        # CRITICAL: This URL connects to the Singapore/Intl gateway.
        # If you are in China, you might use 'dashscope.aliyuncs.com' instead.
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    print("Connecting to Qwen...")

    try:
        # 3. Send the Prompt
        # We are using 'qwen-plus', a great balance of speed and intelligence.
        completion = client.chat.completions.create(
            model="qwen-plus",  
            messages=[
                {'role': 'system', 'content': 'You are a helpful expert on Cloud Computing.'},
                {'role': 'user', 'content': 'Explain the concept of "Serverless" to a 10-year-old.'}
            ]
        )
        
        # 4. Print the Response
        print("\n--- Qwen says: ---")
        print(completion.choices[0].message.content)
        print("------------------")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Check if your API Key matches the region in your base_url.")

if __name__ == '__main__':
    chat_with_qwen()
