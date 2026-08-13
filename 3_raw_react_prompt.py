# # CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
# import re
# import inspect
# from dotenv import load_dotenv

# load_dotenv()

# import ollama
# from langsmith import traceable

# MAX_ITERATIONS = 10
# MODEL = "qwen3:1.7b"


# # --- Tools (LangChain @tool decorator) ---


# @traceable(run_type="tool")
# def get_product_price(product: str) -> float:
#     """Look up the price of a product in the catalog."""
#     print(f"    >> Executing get_product_price(product='{product}')")
#     prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
#     return prices.get(product, 0)


# @traceable(run_type="tool")
# def apply_discount(price: float, discount_tier: str) -> float:
#     """Apply a discount tier to a price and return the final price.
#     Available tiers: bronze, silver, gold."""
#     print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
#     price = float(price)
#     discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
#     discount = discount_percentages.get(discount_tier, 0)
#     return round(price * (1 - discount / 100), 2)

# tools = {
#     "get_product_price": get_product_price,
#     "apply_discount": apply_discount,
# }

# # CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# # We derive descriptions from the functions themselves using inspect.

# def get_tool_descriptions(tools_dict):
#     descriptions = []
#     for tool_name, tool_function in tools_dict.items():
#         # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
#         original_function = getattr(tool_function, "__wrapped__", tool_function)
#         signature = inspect.signature(original_function)
#         docstring = inspect.getdoc(tool_function) or ""
#         descriptions.append(f"{tool_name}{signature} - {docstring}")
#     return "\n".join(descriptions)

# tool_descriptions = get_tool_descriptions(tools)
# tool_names = ", ".join(tools.keys())

# react_prompt = f"""
# STRICT RULES — you must follow these exactly:
# 1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
# 2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
# 3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
# 4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

# Answer the following questions as best you can. You have access to the following tools:

# {tool_descriptions}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action, as comma separated values
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {{question}}
# Thought:"""




# # CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# # all agency comes from the prompt above and our regex parsing below.

# @traceable(name="Ollama Chat", run_type="llm")
# def ollama_chat_traced(model, messages, options):
#     return ollama.chat(model=model, messages=messages, options=options)





# # --- Agent Loop ---


# @traceable(name="Ollama Agent Loop")
# def run_agent(question: str):
#     print(f"Question: {question}")
#     print("=" * 60)


#     # CHANGE 5: One prompt string replaces the system/user message split.
#     prompt = react_prompt.format(question=question)
#     scratchpad = "" 

#     for iteration in range(1, MAX_ITERATIONS + 1):
#         print(f"\n--- Iteration {iteration} ---")
#         full_prompt = prompt + scratchpad

#         # Stop token prevents the LLM from generating its own Observation —
#         # we inject the real tool result instead.
#         response = ollama_chat_traced(
#             model=MODEL,
#             messages=[{"role": "user", "content": full_prompt}],
#             options={"stop": ["\nObservation"], "temperature": 0},
#         )
#         output = response.message.content
#         print(f"LLM Output:\n{output}")

#         print(f"  [Parsing] Looking for Final Answer in LLM output...")
#         final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
#         if final_answer_match:
#             final_answer = final_answer_match.group(1).strip()
#             print(f"  [Parsed] Final Answer: {final_answer}")
#             print("\n" + "=" * 60)
#             print(f"Final Answer: {final_answer}")
#             return final_answer



#               # CHANGE 6: Parse tool calls from raw text with regex — robust against multiline blocks
#         print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

#         # Unified regex matching both lines across the string block cleanly
#         tool_match = re.search(
#             r"Action:\s*([^\n]+)\n+Action Input:\s*([^\n]+)", 
#             output, 
#             re.IGNORECASE
#         )

#         if not tool_match:
#             # Fallback debug: check if it outputted the fields with slight formatting variances
#             print("  [Parsing] ERROR: Could not parse Action/Action Input block sequentially.")
            
#             # Simple healing attempt: did it just provide Action without input?
#             action_fallback = re.search(r"Action:\s*([^\n]+)", output)
#             if action_fallback:
#                 tool_name = action_fallback.group(1).strip()
#                 print(f"  [Parser Self-Healing] Found action '{tool_name}' but missing input line.")
#             break

#         tool_name = tool_match.group(1).strip()
#         tool_input_raw = tool_match.group(2).strip()

#         print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")


#         # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
#         scratchpad += f"{output}\nObservation: {observation}\nThought:"


#     print("ERROR: Max iterations reached without a final answer")
#     return None


# if __name__ == "__main__":
#     print("Hello LangChain Agent (.bind_tools)!")
#     print()
#     result = run_agent("What is the price of a laptop after applying a gold discount?")



import re
import inspect
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
# Note: Ensure "qwen3:1.7b" is pulled and accessible locally in Ollama
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format. Ensure Action and Action Input are on SEPARATE lines, or provide args right after the comma:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the name of the tool (e.g., get_product_price)
Action Input: the tool arguments only (e.g., laptop)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# --- Agent Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = "" 

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # --- Resilient Dual-Strategy Parser ---
        print(f"  [Parsing] Looking for Action and Action Input...")
        tool_name = None
        tool_input_raw = None

        # Strategy 1: Check standard two-line format
        two_line_match = re.search(r"Action:\s*([^\n]+)\n+Action Input:\s*([^\n]+)", output, re.IGNORECASE)
        
        if two_line_match:
            tool_name = two_line_match.group(1).strip()
            tool_input_raw = two_line_match.group(2).strip()
        else:
            # Strategy 2: Single-line fallback: "Action: tool_name, product=laptop"
            single_line_match = re.search(r"Action:\s*([^\n]+)", output, re.IGNORECASE)
            if single_line_match:
                full_line = single_line_match.group(1).strip()
                if "," in full_line:
                    parts = full_line.split(",", 1)
                    tool_name = parts[0].strip()
                    tool_input_raw = parts[1].strip()
                else:
                    tool_name = full_line
                    tool_input_raw = ""

        if not tool_name:
            print("  [Parsing] ERROR: Could not parse any valid Action from LLM output")
            break

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Clean argument parsing ("product=laptop" -> "laptop")
        args = []
        if tool_input_raw:
            raw_args = [x.strip() for x in tool_input_raw.split(",")]
            args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args if x]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            try:
                # Execute tool unpacking args safely
                observation = str(tools[tool_name](*args))
            except Exception as e:
                observation = f"Error executing tool: {str(e)}"

        print(f"  [Tool Result] {observation}")

        # Update historical scratchpad context string 
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Starting Resilient ReAct Agent...")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")