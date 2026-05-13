from llm_sdk import Small_LLM_Model
import numpy as np
import json
from src.decoder import BuildJson, FunctionCaller
from .parser import create_function_string, index_of_function, read_data, get_list_functions, prompt_builded, user_prompts, build_prompt,  prompt_counted

input_file = 'data/input/function_calling_tests.json'
functions_definition = 'data/input/functions_definition.json'
# output = 'data/output/function_calling_results.json'

print("Enter custom paths for --functions, --input, and --output, or press Enter to use the default files.\n\n")
func_path = input('--functions_definition: ')
prompt_input = input('--input: ')
json_output =  input('--output: ')
if func_path != "":
    functions_definition = func_path
if prompt_input != "":
    input_file = prompt_input
# if json_output != "":
#     output = json_output


llm = Small_LLM_Model()

prompt_count = prompt_counted(input_file)
function_count = prompt_counted(functions_definition)
all_functions = read_data(functions_definition) 
join_functions = create_function_string(all_functions, function_count)
i = 0
while True:
    functions_list = get_list_functions(all_functions, function_count)  
    user_prompt = user_prompts(i, input_file)
    prompt_build = prompt_builded(user_prompt, functions_definition, join_functions)
    defined_function = FunctionCaller(llm, functions_list)
    function_name = defined_function.functionfcaller(prompt_build)
    index_function = index_of_function(functions_list, function_name)
    my_prompt = build_prompt(all_functions, index_function, user_prompt)
    json_format = BuildJson(llm, user_prompt, function_name, all_functions, my_prompt, index_function)
    text = json_format.get_json_format()
    if i == prompt_count - 1:
        break
    i += 1
    print(f"\n\n\n{text}")


