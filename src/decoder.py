import json
from typing import Dict, List
from llm_sdk import Small_LLM_Model
import numpy as np
from .models import FunctionDefinition


class FunctionCaller:

    def __init__(self, llm: Small_LLM_Model, list_functions: List[str]):
        self.llm = llm
        self.list_functions = list_functions

    def __get_tokens(self) -> Dict[str, List[int]]:
        tokens = {}
        for f in self.list_functions:
            id = self.llm.encode(f).squeeze().tolist()
            tokens[f] = id
        return tokens

    def __allowed_tokens(self, generated: List,
                         functions_tokens: Dict[str, List[int]]) -> List:

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

    def __finished(self, generate: List,
                   functions_tokens: dict[str, list[int]]) -> bool:
        decoded = self.llm.decode(generate).strip()
        return decoded in functions_tokens

    def functionfcaller(self, prompt: str) -> str:
        input_ids = self.llm.encode(prompt).squeeze().tolist()
        fun_logits = self.__get_tokens()
        generate: List = []

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
        my_function: str = self.llm.decode(generate)
        return my_function


class BuildJson:

    def __init__(self, llm: Small_LLM_Model, prompt: str, function_caller: str,
                 all_functions: Dict[int, FunctionDefinition],
                 prompt_builded: str,
                 index_function: int, prompt_count: int, counter: int):
        self.llm = llm
        self.prompt = prompt
        self.function_caller = function_caller
        self.all_functions = all_functions
        self.prompt_builded = prompt_builded
        self.index_function = index_function
        self.prompt_count = prompt_count
        self.counter = counter

    def __get_state(self, state: str, prompt: str) -> List[int]:

        my_state: List = []
        if state == "START":
            current_state = self.llm.encode("{ ").squeeze().tolist()
        elif state == "COTES":
            current_state = self.llm.encode('"').squeeze().tolist()
        elif state == "PROMPT":
            current_state = self.llm.encode('prompt').squeeze().tolist()
        elif state == 'CLOSE_PROMPT':
            current_state = self.llm.encode('": ').squeeze().tolist()
        elif state == "PROMPT_VALUE":
            current_state = self.llm.encode('"' + prompt +
                                            '", ').squeeze().tolist()
        elif state == "NAME":
            current_state = self.llm.encode('"name": ').squeeze().tolist()
        elif state == "FUNCTION_NAME":
            current_state = self.llm.encode('"' + self.function_caller +
                                            '", ').squeeze().tolist()
        elif state == "PARAM":
            current_state = self.llm.encode('"parameters":').squeeze().tolist()
        elif state == "OPEN_PARAM":
            current_state = self.llm.encode(' { ').squeeze().tolist()
        elif state == "ALL_JSON":
            current_state = self.llm.encode(prompt).squeeze().tolist()
        elif state == "N_VALUES":
            current_state = self.llm.encode(".0123456789").squeeze().tolist()
        elif state == "N_LAST_VALUES":
            current_state = self.llm.encode(".0123456789").squeeze().tolist()
        elif state == "S_LAST_VALUES":
            current_state = self.llm.encode(prompt).squeeze().tolist()
        if not isinstance(current_state, list):
            my_state.append(current_state)
            return my_state
        return current_state

    def __function_parameters(self) -> List:

        param = []
        for k in self.all_functions[self.index_function].parameters.keys():
            c = f'"{k}": '
            param.append(c)
        return param

    json_text = ""

    def get_json_format(self) -> str:
        json_map = [
            "START", "COTES", "PROMPT", "CLOSE_PROMPT", "PROMPT_VALUE", "NAME",
            "FUNCTION_NAME", "PARAM", "OPEN_PARAM"
        ]
        input_ids = self.llm.encode(self.prompt_builded).squeeze().tolist()
        state = "START"
        text = ""
        for i in range(1000):
            if state != "OPEN_PARAM":
                state = json_map[i]
                current_state = self.__get_state(
                    state, self.prompt.replace('"', '\\"'))
                for v in range(len(current_state)):
                    logits = self.llm.get_logits_from_input_ids(input_ids)
                    for x in range(len(logits)):
                        if x != current_state[v]:
                            logits[x] = float("-inf")
                    id = np.argmax(logits)
                    c = self.llm.decode([int(id)])
                    input_ids.append(id)
                    text += f"{c}"

            else:
                param = self.__function_parameters()
                parameters_count = len(param)
                final_param = 0
                # len_param
                for p in param:
                    parametr_id = self.llm.encode(p).squeeze().tolist()
                    for v in range(len(parametr_id)):
                        logits = self.llm.get_logits_from_input_ids(
                            parametr_id)
                        for x in range(len(logits)):
                            if x != parametr_id[v]:
                                logits[x] = float("-inf")
                        id = np.argmax(logits)
                        c = self.llm.decode([int(id)])
                        input_ids.append(id)
                        text += f"{c}"

                    add_quote = True
                    param_text = ""
                    while True:
                        current_state = self.__get_state("ALL_JSON", text)
                        logits = self.llm.get_logits_from_input_ids(
                            current_state)
                        par_name = p.replace('"',
                                             "").replace(':',
                                                         "").replace(' ', '')
                        if self.all_functions[self.index_function].parameters[
                                par_name].type == "number":
                            if final_param == parameters_count - 1:
                                state_list = self.__get_state(
                                    "N_LAST_VALUES", text)
                                pos_char = '} '
                            else:
                                state_list = self.__get_state("N_VALUES", text)
                                pos_char = ', '

                            for values in range(len(logits)):
                                if values not in state_list:
                                    logits[values] = float('-inf')
                            id = np.argmax(logits)
                            c = self.llm.decode([int(id)])
                            input_ids.append(id)
                            d = f"{c}"
                            param_text += d
                            if pos_char in d or ".0" in param_text:
                                if '.0' not in param_text:
                                    float_numbers = self.llm.encode(
                                        '.0').squeeze().tolist()
                                    decimal_point = self.llm.decode(
                                        float_numbers)
                                    input_ids.append(float_numbers)
                                    text += f"{decimal_point}"
                                text += f"{c}"
                                if pos_char not in param_text[-1]:
                                    close_param = self.llm.encode(
                                        pos_char).squeeze().tolist()
                                    print_param = self.llm.decode(close_param)
                                    text += f"{print_param}"
                                break

                            text += f"{c}"

                        else:
                            if add_quote:
                                ids = self.llm.encode('"').squeeze().tolist()
                                c = self.llm.decode(ids)
                                text += f"{c}"
                                add_quote = False
                                input_ids.append(ids)

                            logits = self.llm.get_logits_from_input_ids(
                                input_ids)
                            forbidden_ids = []
                            forbidden_ids = (
                                self.llm.encode('"\n').squeeze().tolist()
                                )

                            for for_id in forbidden_ids:
                                logits[for_id] = float("-inf")

                            id = np.argmax(logits)
                            c = self.llm.decode([int(id)])
                            d = f"{c}"
                            text += d
                            input_ids.append(id)
                            if '"' in d:
                                if final_param == parameters_count - 1:
                                    if d[-1] != '}':
                                        ids = self.llm.encode(
                                            '} ').squeeze().tolist()
                                        c = self.llm.decode(ids)
                                        d = f"{c}"
                                        text += d
                                        input_ids.append(ids)
                                else:
                                    if d[-1] != ',':
                                        ids = self.llm.encode(
                                            ', ').squeeze().tolist()
                                        c = self.llm.decode(ids)
                                        d = f"{c}"
                                        text += d
                                        input_ids.append(ids)
                                break

                    final_param += 1
                if final_param == parameters_count:
                    text += "}"
                    self.json_text += text
                    try:
                        obg = json.loads(text)
                        if self.counter == 0:
                            print('[')
                        print(json.dumps(obg, indent=2), end="", flush=True)

                    except json.JSONDecodeError:
                        print(
                            f"We have error in json formatplease"
                            f"edit your prompt \n  {self.prompt} and try again"
                        )
                        exit(1)
                    if self.counter != self.prompt_count - 1:
                        print(",")
                    else:
                        print('\n]')
                    text = ""

                break
        return self.json_text
