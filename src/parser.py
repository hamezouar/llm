from .models import FunctionDefinition, Prompt
import json
from pydantic import ValidationError


def read_data(path: str) -> dict[str, FunctionDefinition]:
    try:
        with open(path, "r") as file:
            data = json.load(file)
            
    except FileNotFoundError:
        print("ERROR: File not found")
        return {}
    except json.JSONDecodeError:
        print("ERROR: Invalid JSON format")
        return {}

    my_objects: dict[str, FunctionDefinition] = {}
    try:
        for obj in data:
            obj_name = obj["name"]
            new_object = FunctionDefinition(**obj)
            my_objects[obj_name] = new_object
    except (ValidationError, KeyError) as e:
        if isinstance(e, KeyError):
            print(f"ERROR: Validation failed {e}")
        else:
            for error in e.errors():
                print(f"- {error['msg']}")
        return {}

    return my_objects

def read_prompt(path : str) -> list[Prompt]:
    try:
        with open(path, "r") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Please check your prompt syntax")
    except( FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return []
    my_prompts: list[Prompt] = []
    try:
        for p in data:
            new_object = Prompt(**p)
            my_prompts.append(new_object)
    except ValidationError as e:
        print("ERROR: Validation failed")
        for error in e.errors():
            print(f"- {error['msg']}")
        return []
    return my_prompts

def get_list_functions():
    function_name = read_data("data/input/functions_definition.json")
    list_functions = []
    for i in function_name:
        list_functions.append(i)
    return list_functions

def user_prompts(prompt_index):
    prompts = read_prompt("data/input/function_calling_tests.json")
    return prompts[prompt_index].prompt


def prompt_builded(prompt):
    list_functions = read_data("data/input/functions_definition.json")
    my_prompt = (
    "You are a function-calling assistant. Match the User Request to the best Function Name.\n"
    "Return ONLY the function name. No explanation.\n\n"
    
    "Available Functions:\n"
    f"1. {list_functions['fn_add_numbers'].name}: {list_functions['fn_add_numbers'].description}\n"
    f"2. {list_functions['fn_greet'].name}: {list_functions['fn_greet'].description}\n"
    f"3. {list_functions['fn_reverse_string'].name}: {list_functions['fn_reverse_string'].description}\n"
    f"4. {list_functions['fn_get_square_root'].name}: {list_functions['fn_get_square_root'].description}\n"
    f"5. {list_functions['fn_substitute_string_with_regex'].name}: {list_functions['fn_substitute_string_with_regex'].description}\n"

    "Examples:\n"
    "Input: Hello, can you greet me?\n"
    "Output: fn_greet\n\n"
    
    "Input: What is the capital of France?\n"
    "Output: non_seported\n\n"
    
    "Input: Calculate the sum of 10 and 20\n"
    "Output: fn_add_numbers\n\n"
    
    "Input: Find the square root of 16\n"
    "Output: fn_get_square_root\n\n"
    
    "Input: Reverse 'hello'\n"
    "Output: fn_reverse_string\n"

    f"User Request: {prompt}\n"
    "Output: "
)
    return my_prompt

def build_prompt(all_functions, function_name, prompt):
    return (
        "<|im_start|>system\n"
        "You are a smart parameter extraction engine.\n"
        "You are given a function name and a user request.\n"
        "Infer the parameters from the request.\n\n"

        f"Function: {function_name}\n"
        f"parameters: {all_functions[function_name].parameters}\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON.\n"
        "- Format:\n"
        '{ "name": "<function>", "parameters": { ... } }\n'
        "- Do NOT add explanation.\n"
        "- Infer parameter names logically.\n"
        "- Use meaningful parameter names based on the function name.\n\n"
        "- All numeric values MUST be returned as floats, even if they are whole numbers (e.g., 2 -> 2.0). Never return integers."

        "Examples:\n"
        "Function: fn_add_numbers\n"
        "User: what is the sum of 2 and 4\n"
        'Assistant: { "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 4.0} }\n\n'

        "Function: fn_greet\n"
        "User: Greet shrek\n"
        'Assistant: { "name": "fn_greet", "parameters": {"name": shrek} }\n\n'

        "Function: fn_reverse_string\n"
        "User: reverse hello\n"
        'Assistant: {"prompt": "Reverse the string hello", "name": "fn_reverse_string", "parameters": {"s": "hello"} }\n\n'

        "<|im_end|>\n"

        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"

        "<|im_start|>assistant\n"
    )

def string_prompt_optimized(all_functions, function_name, prompt):
    return (
        "<|im_start|>system\n"
        "You are a specialized parameter extraction engine.\n"
        "Your ONLY task is to extract exact string values from the user request based on the function definition.\n\n"
        
        f"TARGET_FUNCTION: {function_name}\n"
        f"PARAMETERS_SCHEMA: {all_functions[function_name].parameters}\n\n"
        
        "STRICT_RULES:\n"
        "1. Extract the string value EXACTLY as it appears in the user request.\n"
        "2. Do NOT repeat the prompt, do NOT repeat the parameter name inside the value.\n"
        "3. For string parameters, provide only the core value (e.g., if the user says 'Reverse hello', the value is 'hello').\n"
        "4. Output format must be STRICT JSON.\n"
        "5. Numbers must always be floats (e.g., 5.0).\n\n"

        "EXTRACTION_EXAMPLES:\n"
        "User: Greet john\n"
        'Assistant: { "name": "fn_greet", "parameters": { "name": "john" } }\n\n'
        
        "User: Reverse the string 'world'\n"
        'Assistant: { "name": "fn_reverse_string", "parameters": { "s": "world" } }\n\n'
        
        "User: Replace 'apple' with 'orange' in 'I like apple'\n"
        'Assistant: { "name": "fn_substitute", "parameters": { "source": "I like apple", "regex": "apple", "replacement": "orange" } }\n\n'

        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Function: {function_name}\n"
        f"User Request: {prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
