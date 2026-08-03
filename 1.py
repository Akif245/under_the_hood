# import ollama

# # 1. Define a "Tool" for your Agent
# # This is just a normal Python function. You can make tools to read files, search the web, etc.
# def apply_discount(price: float, discount_percent: float) -> str:
#     """Calculates the final price after applying a discount percentage."""
#     final_price = price - (price * (discount_percent / 100))
#     return f"The final price is ${final_price:.2f}"

# # 2. Set up the Agent's brain and give it a task
# # The "system" prompt is where you define your custom model's personality and rules.
# messages = [
#     {
#         "role": "system", 
#         "content": "You are a helpful retail assistant agent. If the user asks about prices, use the apply_discount tool to do the math."
#     },
#     {
#         "role": "user", 
#         "content": "I want to buy a laptop that costs $1200, but I have a 15% off coupon. How much will it cost?"
#     }
# ]

# print("Thinking...")

# # 3. Ask your local model to process the request
# response = ollama.chat(
#     model='qwen3:1.7b', # Using the model you already downloaded!
#     messages=messages,
#     tools=[apply_discount] # We hand the model our Python tool
# )

# # 4. See what the Agent decided to do
# # 4. See what the Agent decided to do
# if response['message'].get('tool_calls'):
#     print("\nSUCCESS! Your Agent decided to use a tool...")
    
#     # Add the AI's tool request to our conversation history
#     messages.append(response['message'])
    
#     for tool in response['message']['tool_calls']:
#         if tool['function']['name'] == 'apply_discount':
#             # 5. Extract the numbers the AI found
#             args = tool['function']['arguments']
#             price = args['price']
#             discount = args['discount_percent']
            
#             # 6. Actually run our Python function!
#             tool_result = apply_discount(price, discount)
#             print(f"-> Python calculated secretly: {tool_result}")
            
#             # 7. Give the result back to the AI as a "tool" message
#             messages.append({
#                 "role": "tool",
#                 "content": tool_result,
#                 "name": "apply_discount"
#             })
            
#     # 8. Let the AI give the final answer to the user
#     print("\nSending math result back to the Agent...\n")
#     final_response = ollama.chat(model='qwen3:1.7b', messages=messages)
    
#     print("🤖 Agent Final Answer:")
#     print(final_response['message']['content'])
    
# else:
#     print("\nAgent answered normally:")
#     print(response['message']['content'])










import ollama

# ---------------------------------------------------------
# 1. Define ALL your Tools (Python functions)
# ---------------------------------------------------------
def apply_discount(price: float, discount_percent: float) -> str:
    """Calculates the final price after applying a discount percentage."""
    final_price = price - (price * (discount_percent / 100))
    return f"The final discounted price is ${final_price:.2f}"

def calculate_shipping(destination: str, weight_lbs: float) -> str:
    """Calculates the shipping cost based on destination and weight."""
    # A fake shipping calculator for our example
    base_rate = 5.00
    if destination.lower() == "hawaii" or destination.lower() == "alaska":
        base_rate = 15.00
    
    total_shipping = base_rate + (weight_lbs * 2.50)
    return f"Shipping to {destination} for a {weight_lbs}lb package is ${total_shipping:.2f}"

# We put our functions in a dictionary so we can easily look them up by name later
available_functions = {
    'apply_discount': apply_discount,
    'calculate_shipping': calculate_shipping
}

# ---------------------------------------------------------
# 2. Set up the Agent's brain and give it a task
# ---------------------------------------------------------
messages = [
    {
        "role": "system", 
        "content": "You are a helpful retail assistant. Use the available tools to answer the user's questions about prices and shipping."
    },
    {
        "role": "user", 
        # Notice we are asking a question that requires TWO separate math operations
        "content": "I want to buy a laptop that costs $1200 with a 15% off coupon. It weighs 5 lbs and needs to be shipped to Hawaii. What is the discount, and what is the shipping cost?"
    }
]

print("Thinking...\n")

# ---------------------------------------------------------
# 3. Ask your local model to process the request
# ---------------------------------------------------------
response = ollama.chat(
    model='qwen3:1.7b', 
    messages=messages,
    # We pass BOTH functions in a list here
    tools=[apply_discount, calculate_shipping] 
)

# ---------------------------------------------------------
# 4. The Agent Loop (Handling multiple tools)
# ---------------------------------------------------------+---------------------------
if response['message'].get('tool_calls'):
    print("SUCCESS! Your Agent decided to use a tool...")
    
    # 1. Save the AI's tool requests to the chat history
    messages.append(response['message'])
    
    # 2. Loop through EVERY tool the AI asked to use
    for tool in response['message']['tool_calls']:
        tool_name = tool['function']['name']
        tool_args = tool['function']['arguments']
        
        # 3. Find the requested function in our dictionary
        if function_to_call := available_functions.get(tool_name):
            print(f"-> AI is running {tool_name} with arguments: {tool_args}")
            
            # 4. Execute the function (** unpacks the dictionary args)
            tool_result = function_to_call(**tool_args)
            print(f"   Result: {tool_result}")
            
            # 5. Append the result of THIS specific tool to the chat history
            messages.append({
                "role": "tool",
                "content": tool_result,
                "name": tool_name
            })
        else:
            print(f"Error: Model tried to use a tool that doesn't exist: {tool_name}")
            
    # 6. Once all tools are run, ask the AI to formulate a final answer
    print("\nSending all tool results back to the Agent...\n")
    final_response = ollama.chat(model='qwen3:1.7b', messages=messages)
    
    print(" Agent Final Answer:")
    print(final_response['message']['content'])
    
else:
    print("\nAgent answered normally (No tools needed):")
    print(response['message']['content'])