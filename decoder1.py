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
    def __init__(self, llm, prompt, function_caller, all_functions, prompt_builded):
        self.llm = llm
        self.prompt = prompt
        self.function_caller = function_caller
        self.all_functions = all_functions
        self.prompt_builded = prompt_builded
    
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
        elif state == "S_VALUES":
            current_state = self.llm.encode(f'{prompt}",').squeeze().tolist()
        elif state == "S_LAST_VALUES":
            current_state = self.llm.encode(f'{prompt}').squeeze().tolist()
            x = self.llm.encode('"').squeeze().tolist()
            current_state.append(x)
            x = self.llm.encode('"}').squeeze().tolist()
            current_state.append(x)
        if not isinstance(current_state, list):
            my_state.append(current_state)
            return my_state
        return current_state
    
    def __function_parameters(self):

        param = []
        for k in self.all_functions[self.function_caller].parameters.keys():
            c = f' "{k}": '
            param.append(c)
        return param
    
    def get_json_format(self):
        json_map = ["START","COTES", "PROMPT", "CLOSE_PROMPT", "PROMPT_VALUE", "NAME", "FUNCTION_NAME","PARAM", "OPEN_PARAM"]
        prompt_builded = build_prompt(self.all_functions, self.function_caller, self.prompt)
        input_ids = self.llm.encode(self.prompt_builded).squeeze().tolist()
        state = "START"
        text = ""
        for i in range(400):
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
                    print(c, end="", flush=True)
                    input_ids.append(id)
                    text += f"{c}"


            else:
                param = self.__function_parameters()
                parameters_count = len(param)
                final_param = 0
                # len_param
                for p in param:
                    parametr_id = self.llm.encode(p).squeeze().tolist()
                    # current_id
                    for v in range(len(parametr_id)):
                        logits = self.llm.get_logits_from_input_ids(parametr_id)
                        for x in range(len(logits)):
                            if x != parametr_id[v]:
                                logits[x] = float("-inf")
                        id = np.argmax(logits)
                        c = self.llm.decode(id)
                        print(c, end="", flush=True)
                        input_ids.append(id)
                        text += f"{c}"


                    while True:
                        current_state =  self.__get_state("ALL_JSON", text)
                        logits = self.llm.get_logits_from_input_ids(current_state)
                        par_name = p.replace('"', "").replace(':', "").replace(' ', '')
                        if self.all_functions[self.function_caller].parameters[par_name].type == "number":
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
                            text += f"{c}"
                            if pos_char in d :
                                float_numbers = self.llm.encode('.0').squeeze().tolist()
                                decimal_point = self.llm.decode(float_numbers)
                                input_ids.append(float_numbers)
                                print( decimal_point, end="", flush=True)
                                print(c, end="", flush=True)
                                break
                            print(c, end="", flush=True)
                        else:
                            if final_param == parameters_count - 1:
                                state_list = self.__get_state("S_LAST_VALUES", self.prompt.replace("'", '"').replace(" ", ""))
                                pos_char = '}'
                            else:
                                state_list = self.__get_state("S_VALUES", self.prompt)
                                pos_char = ','
                            logits += self.llm.get_logits_from_input_ids(state_list)
                            for values in range(len(logits)):
                                if values not in state_list :
                                    logits[values] = float("-inf")
                            id = np.argmax(logits)
                            state_list.append(id)
                            c = self.llm.decode(id)
                            d = f"{c}"
                            text += d
                            print(d, end="", flush=True)
                            if pos_char in d:
                                break
                            

                    final_param += 1
                if final_param == parameters_count:
                    """add } in parametrs """
                    ids = self.llm.encode(' } ').squeeze().tolist()
                    close_parametrs = self.llm.encode('}').squeeze().tolist()
                    log = self.llm.get_logits_from_input_ids(ids)
                    for values in range(len(log)):
                        if values != close_parametrs:
                            log[values] = float("-inf")
                    c = self.llm.decode(np.argmax(log))
                    text += f"{d}"
                    print(c, end="", flush=True)

                break
        return text



