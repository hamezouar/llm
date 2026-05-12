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
        index = 0
        for obj in data:
            obj_name = index
            new_object = FunctionDefinition(**obj)
            my_objects[obj_name] = new_object
            index += 1
    except (ValidationError, KeyError) as e:
        if isinstance(e, KeyError):
            print(f"ERROR: Validation failed {e}")
        else:
            for error in e.errors():
                print(f"- {error['msg']}")
        return {}

    return my_objects

def create_function_string(all_functions, function_count):
    string = ""
    i = 0
    while i < function_count:
        string += f"{i}. {all_functions[i].name}: {all_functions[i].description}\n"
        i += 1
    return string

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

def get_list_functions(all_functions, len_function):
    list_functions = []
    index = 0
    while index < len_function:
        list_functions.append(all_functions[index].name)
        index += 1
    return list_functions


def index_of_function(all_functions, function_name):
    index = 0
    for name in all_functions:
        if name == function_name:
            return index
        index += 1
    return index
def user_prompts(prompt_index, path):
    prompts = read_prompt(path)
    return prompts[prompt_index].prompt


# i used this prompt to count prompt count and functions count
def prompt_counted(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Please check your prompt syntax")
    except( FileNotFoundError, ValueError) as e:
        return len(data)
    
    return len(data)


# i used this prompt to find function name
def prompt_builded(prompt, path, join_functions):
    list_functions = read_data(path)
    my_prompt = (
    "You are a function-calling assistant. Match the User Request to the best Function Name.\n"
    "Return ONLY the function name. No explanation.\n\n"
    
    "Available Functions:\n"
    f"{join_functions}"

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


# i used this prompt to build json format
def build_prompt(all_functions, index, prompt):
    return (
        "<|im_start|>system\n"
        "You are a smart parameter extraction engine.\n"
        "You are given a function name and a user request.\n"
        "Infer the parameters from the request.\n\n"

        f"Function: {all_functions[index].name}\n"
        f"parameters: {all_functions[index].parameters}\n\n"
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
