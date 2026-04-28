from llm_sdk import Small_LLM_Model
import numpy as np
import json
from .parser import read_data, read_prompt, get_list_functions, get_prompt, build_prompt

class FunctionCaller:
    def __init__(self, llm, list_functions):
        self.llm = llm
        self.list_functions = list_functions
    
    def __get_tokens(self):
        tokens = {}
        for f in self.list_functions:
            id = self.llm.encode(f).squeeze().tolist()
            tokens[f] = id
        return tokens
    
    def __allowed_tokens(self, generated, functions_tokens):

        allowed = []

        if not generated:
            for log in functions_tokens:
                allowed.append(functions_tokens[log][0])

            return allowed
        else:
            index = len(generated)
            for log in functions_tokens:
                if len(functions_tokens[log]) > index:
                    id = functions_tokens[log][index]
                    if functions_tokens[log][:len(generated)] == generated:
                        allowed.append(id)
            return allowed

    def __finished(self,generate, functions_tokens):
        decoded = self.llm.decode(generate).strip()
        return decoded in functions_tokens
    
    def functionfcaller(self, prompt):
        input_ids = self.llm.encode(prompt).squeeze().tolist()
        fun_logits = self.__get_tokens()
        generate = []

        while True:
            logits = self.llm.get_logits_from_input_ids(input_ids)

            allowed = self.__allowed_tokens(generate, fun_logits)

            if not allowed:
                break
            
            for i in range(len(logits)):
                if i not in set(allowed):
                    logits[i] = float("-inf")
            next_token = np.argmax(logits)
            generate.append(next_token)
            input_ids.append(next_token)

            if self.__finished(generate, fun_logits):
                break
        return self.llm.decode(generate)

def get_state(llm,state, prompt, function_name):
    my_state = []
    current_state = ""
    if state == "START":
        current_state = llm.encode("{ ").squeeze().tolist()
    elif state == "COTES":
        current_state = llm.encode('"').squeeze().tolist()
    elif state == "PROMPT":
        current_state = llm.encode('prompt').squeeze().tolist()
    elif state == 'CLOSE_PROMPT':
        current_state = llm.encode('": ').squeeze().tolist()
    elif state == "PROMPT_VALUE":
        current_state = llm.encode('"' + prompt + '", ').squeeze().tolist()
    elif state == "NAME":
        current_state = llm.encode('"name": ').squeeze().tolist()
    elif state == "FUNCTION_NAME":
        current_state = llm.encode('"' + function_name + '", ').squeeze().tolist()
    elif state == "PARAM":
        current_state = llm.encode('"parameters":').squeeze().tolist()
    elif state == "OPEN_PARAM":
        current_state = llm.encode(' { ').squeeze().tolist()
    elif state == "NUMBERS":
        current_state = llm.encode('0123456789').squeeze().tolist()
    elif state == "ALL_JSON":
            current_state = llm.encode(prompt).squeeze().tolist()
    elif state == "N_VALUES":
        current_state = llm.encode("0123456789,").squeeze().tolist()
    elif state == "N_LAST_VALUES":
        current_state = llm.encode("0123456789}").squeeze().tolist()
    elif state == "FORBIDDEN_S":
        current_state = llm.encode("'\n\t\\").squeeze().tolist()
    if not isinstance(current_state, list):
        my_state.append(current_state)
        return my_state
    return current_state

def get_param(function_name):
    function = read_data("data/input/functions_definition.json")
    param = []
    for k in function[function_name].parameters.keys():
        c = f' "{k}": '
        param.append(c)
    return param


llm = Small_LLM_Model()
f = read_data("data/input/functions_definition.json")
p = get_prompt()
f = get_list_functions()
my_class  = FunctionCaller(llm,f)
function_name = my_class.functionfcaller(p)
funct = read_data("data/input/functions_definition.json")
state_list = ["START","COTES", "PROMPT", "CLOSE_PROMPT", "PROMPT_VALUE", "NAME", "FUNCTION_NAME","PARAM", "OPEN_PARAM", "END"]
j = build_prompt(function_name)
input_ids = llm.encode(j).squeeze().tolist()
state = "START"
text = ""
for i in range(400):
    if state != "OPEN_PARAM":
        state = state_list[i]
        current_state =  get_state(llm, state,  "What is the sum of 2995675567 and 376576564?", function_name)
        for v in range(len(current_state)):
            logits = llm.get_logits_from_input_ids(input_ids)
            for x in range(len(logits)):
                if x != current_state[v]:
                    logits[x] = float("-inf")
            id = np.argmax(logits)
            c = llm.decode(id)
            print(c, end="", flush=True)
            input_ids.append(id)
            d = f"{c}"
            text += d


    else:
        param = get_param(function_name)
        len_param = len(param)
        final_param = 0
        for p in param:
            current_id = llm.encode(p).squeeze().tolist()
            for v in range(len(current_id)):
                logits = llm.get_logits_from_input_ids(current_id)
                for x in range(len(logits)):
                    if x != current_id[v]:
                        logits[x] = float("-inf")
                id = np.argmax(logits)
                c = llm.decode(id)
                print(c, end="", flush=True)
                input_ids.append(id)
                d = f"{c}"
                text += d
            while True:
                current_state =  get_state(llm, "ALL_JSON", text, function_name)
                logits = llm.get_logits_from_input_ids(current_state)
                par_name = p.replace('"', "").replace(':', "").replace(' ', '')
                if funct[function_name].parameters[par_name].type == "number":
                    state_list = get_state(llm, "N_VALUES", text, function_name)
                    pos_char = ','
                    if final_param == len_param - 1:
                        state_list = get_state(llm, "N_LAST_VALUES", text, function_name)
                        pos_char = '}'
                    for values in range(len(logits)):
                        if values not in state_list:
                            logits[values] = float('-inf')
                    id = np.argmax(logits)
                    c = llm.decode(id)
                    input_ids.append(id)
                    d = f"{c}"
                    text += d
                    if pos_char in d :
                        decimal_point = llm.decode([13, 15])
                        input_ids.append(13)
                        input_ids.append(15)
                        print( decimal_point, end="", flush=True)
                        print(c, end="", flush=True)
                        break
                    print(c, end="", flush=True)
                else:
                    break
            final_param += 1
        if final_param == len_param:
            c = llm.decode(92)
            print(c, end="", flush=True)
        break
    












