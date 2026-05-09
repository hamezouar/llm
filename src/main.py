from llm_sdk import Small_LLM_Model
import numpy as np
import json
from src.decoder import BuildJson, FunctionCaller
from .parser import read_data, get_list_functions, prompt_builded, user_prompts


llm = Small_LLM_Model()


functions_list = get_list_functions()
all_functions = read_data("data/input/functions_definition.json")
user_prompt = user_prompts(4)
prompt_builded = prompt_builded(user_prompt)


defined_function = FunctionCaller(llm, functions_list)


function_name = defined_function.functionfcaller(prompt_builded)
json_format = BuildJson(llm, user_prompt, function_name, all_functions, prompt_builded)

text = json_format.get_json_format()

