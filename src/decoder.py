import numpy as np
from .parser import build_prompt

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


class BuildJson:
    def __init__(self, llm, prompt, function_caller, all_functions, prompt_builded, index_function):
        self.llm = llm
        self.prompt = prompt
        self.function_caller = function_caller
        self.all_functions = all_functions
        self.prompt_builded = prompt_builded
        self.index_function = index_function
    
    def __get_state(self,state, prompt):

        my_state = []
        current_state = ""
        if state == "START":
            current_state = self.llm.encode("{ ").squeeze().tolist()
        elif state == "COTES":
            current_state = self.llm.encode('"').squeeze().tolist()
        elif state == "PROMPT":
            current_state = self.llm.encode('prompt').squeeze().tolist()
        elif state == 'CLOSE_PROMPT':
            current_state = self.llm.encode('": ').squeeze().tolist()
        elif state == "PROMPT_VALUE":
            current_state = self.llm.encode('"' + prompt + '", ').squeeze().tolist()
        elif state == "NAME":
            current_state = self.llm.encode('"name": ').squeeze().tolist()
        elif state == "FUNCTION_NAME":
            current_state = self.llm.encode('"' + self.function_caller + '", ').squeeze().tolist()
        elif state == "PARAM":
            current_state = self.llm.encode('"parameters":').squeeze().tolist()
        elif state == "OPEN_PARAM":
            current_state = self.llm.encode(' { ').squeeze().tolist()
        elif state == "ALL_JSON":
                current_state = self.llm.encode(prompt).squeeze().tolist()
        elif state == "N_VALUES":
            current_state = self.llm.encode("0123456789,").squeeze().tolist()
        elif state == "N_LAST_VALUES":
            current_state = self.llm.encode("0123456789}").squeeze().tolist()
        elif state == "S_LAST_VALUES":
            current_state = self.llm.encode(prompt).squeeze().tolist()
        if not isinstance(current_state, list):
            my_state.append(current_state)
            return my_state
        return current_state

    
    def __function_parameters(self):

        param = []
        for k in self.all_functions[self.index_function].parameters.keys():
            c = f'"{k}": '
            param.append(c)
        return param
    
    def get_json_format(self):
        json_map = ["START","COTES", "PROMPT", "CLOSE_PROMPT", "PROMPT_VALUE", "NAME", "FUNCTION_NAME","PARAM", "OPEN_PARAM"]
        input_ids = self.llm.encode(self.prompt_builded).squeeze().tolist()
        state = "START"
        text = ""
        stored = ""
        for i in range(1000):
            if state != "OPEN_PARAM":
                state = json_map[i]
                current_state =  self.__get_state(state, self.prompt)
                for v in range(len(current_state)):
                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    for x in range(len(logits)):
                        if x != current_state[v]:
                            logits[x] = float("-inf")
                    id = np.argmax(logits)
                    c = self.llm.decode(id)
                    input_ids.append(id)
                    text += f"{c}"
                    print(f"{c}", end="", flush=True)


            else:
                param = self.__function_parameters()
                parameters_count = len(param)
                final_param = 0
                # len_param
                for p in param:
                    parametr_id = self.llm.encode(p).squeeze().tolist()
                    for v in range(len(parametr_id)):
                        logits = self.llm.get_logits_from_input_ids(parametr_id)
                        for x in range(len(logits)):
                            if x != parametr_id[v]:
                                logits[x] = float("-inf")
                        id = np.argmax(logits)
                        c = self.llm.decode(id)
                        input_ids.append(id)
                        text += f"{c}"
                        print(f"{c}", end="", flush=True)

                    add_quote = True
                    while True:
                        current_state =  self.__get_state("ALL_JSON", text)
                        logits = self.llm.get_logits_from_input_ids(current_state)
                        par_name = p.replace('"', "").replace(':', "").replace(' ', '')
                        if self.all_functions[self.index_function].parameters[par_name].type == "number":
                            if final_param == parameters_count - 1:
                                state_list = self.__get_state("N_LAST_VALUES", text)
                                pos_char = '}'
                            else:
                                state_list = self.__get_state("N_VALUES", text)
                                pos_char = ','

                            for values in range(len(logits)):
                                if values not in state_list:
                                    logits[values] = float('-inf')
                            id = np.argmax(logits)
                            c = self.llm.decode(id)
                            input_ids.append(id)
                            d = f"{c}"
                            if pos_char in d :
                                float_numbers = self.llm.encode('.0').squeeze().tolist()
                                decimal_point = self.llm.decode(float_numbers)
                                input_ids.append(float_numbers)
                                text += f"{decimal_point}"
                                text += f"{c}"
                                print(f"{c}", end="", flush=True)
                                break
                            text += f"{c}"
                            print(f"{c}", end="", flush=True)




                        else:
                            if add_quote:
                                ids = self.llm.encode('"').squeeze().tolist()
                                c = self.llm.decode(ids)
                                print(f"{c}", end="", flush=True)
                                text += f"{c}"
                                add_quote = False
                                input_ids.append(ids)

                            logits = self.llm.get_logits_from_input_ids(input_ids)
                            x = self.llm.encode('"\n').squeeze().tolist()
                            forbidden_ids = []
                            forbidden_ids.append(x)

                            for for_id in forbidden_ids:
                                logits[for_id] = float("-inf")


                            id = np.argmax(logits)
                            c = self.llm.decode(id)
                            d = f"{c}"
                            print(f"{c}", end="", flush=True)
                            text += d
                            input_ids.append(id)
                            if '"' in d :
                                if final_param == parameters_count - 1 and d[-1] != '}':
                                    ids = self.llm.encode('}').squeeze().tolist()
                                    c = self.llm.decode(ids)
                                    d = f"{c}"
                                    print(f"{c}", end="", flush=True)
                                    text += d
                                    input_ids.append(ids)
                                else:
                                    if d[-1] != ',':
                                        ids = self.llm.encode(',').squeeze().tolist()
                                        c = self.llm.decode(ids)
                                        print(f"{c}", end="", flush=True)
                                        d = f"{c}"
                                        text += d
                                        input_ids.append(ids)
                                break




                    final_param += 1
                if final_param == parameters_count:
                    ids = self.llm.encode(' } ').squeeze().tolist()
                    close_parametrs = self.llm.encode('}').squeeze().tolist()
                    log = self.llm.get_logits_from_input_ids(ids)
                    for values in range(len(log)):
                        if values != close_parametrs:
                            log[values] = float("-inf")
                    c = self.llm.decode(np.argmax(log))
                    print(f"{c}", end="", flush=True)
                    text += f"{c}"

                break
        return text
    



