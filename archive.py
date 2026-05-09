
# def get_state(llm,state, prompt, function_name):
#     my_state = []
#     current_state = ""
#     if state == "START":
#         current_state = llm.encode("{ ").squeeze().tolist()
#     elif state == "COTES":
#         current_state = llm.encode('"').squeeze().tolist()
#     elif state == "PROMPT":
#         current_state = llm.encode('prompt').squeeze().tolist()
#     elif state == 'CLOSE_PROMPT':
#         current_state = llm.encode('": ').squeeze().tolist()
#     elif state == "PROMPT_VALUE":
#         current_state = llm.encode('"' + prompt + '", ').squeeze().tolist()
#     elif state == "NAME":
#         current_state = llm.encode('"name": ').squeeze().tolist()
#     elif state == "FUNCTION_NAME":
#         current_state = llm.encode('"' + function_name + '", ').squeeze().tolist()
#     elif state == "PARAM":
#         current_state = llm.encode('"parameters":').squeeze().tolist()
#     elif state == "OPEN_PARAM":
#         current_state = llm.encode(' { ').squeeze().tolist()
#     elif state == "ALL_JSON":
#             current_state = llm.encode(prompt).squeeze().tolist()
#     elif state == "N_VALUES":
#         current_state = llm.encode("0123456789,").squeeze().tolist()
#     elif state == "N_LAST_VALUES":
#         current_state = llm.encode("0123456789}").squeeze().tolist()
#     elif state == "S_VALUES":
#         current_state = llm.encode(f'{prompt} , "').squeeze().tolist()
#     elif state == "S_LAST_VALUES":
#         current_state = llm.encode(f'{prompt} }} "').squeeze().tolist()
#     if not isinstance(current_state, list):
#         my_state.append(current_state)
#         return my_state
#     return current_state

# def get_param(function_name):
#     function = read_data("data/input/functions_definition.json")
#     param = []
#     for k in function[function_name].parameters.keys():
#         c = f' "{k}": '
#         param.append(c)
#     return param


# llm = Small_LLM_Model()
# f = read_data("data/input/functions_definition.json")

# prompts = read_prompt("data/input/function_calling_tests.json")
# my_prompt = prompts[2].prompt

# p = get_prompt()
# f = get_list_functions()
# my_class  = FunctionCaller(llm,f)
# function_name = my_class.functionfcaller(p)
# funct = read_data("data/input/functions_definition.json")
# state_list = ["START","COTES", "PROMPT", "CLOSE_PROMPT", "PROMPT_VALUE", "NAME", "FUNCTION_NAME","PARAM", "OPEN_PARAM"]
# j = build_prompt(function_name)
# input_ids = llm.encode(j).squeeze().tolist()
# state = "START"
# text = ""
# for i in range(400):
#     if state != "OPEN_PARAM":
#         state = state_list[i]
#         current_state =  get_state(llm, state,  my_prompt, function_name)
#         for v in range(len(current_state)):
#             logits = llm.get_logits_from_input_ids(input_ids)
#             for x in range(len(logits)):
#                 if x != current_state[v]:
#                     logits[x] = float("-inf")
#             id = np.argmax(logits)
#             c = llm.decode(id)
#             print(c, end="", flush=True)
#             input_ids.append(id)
#             d = f"{c}"
#             text += d


#     else:
#         param = get_param(function_name)
#         len_param = len(param)
#         final_param = 0
#         for p in param:
#             current_id = llm.encode(p).squeeze().tolist()
#             for v in range(len(current_id)):
#                 logits = llm.get_logits_from_input_ids(current_id)
#                 for x in range(len(logits)):
#                     if x != current_id[v]:
#                         logits[x] = float("-inf")
#                 id = np.argmax(logits)
#                 c = llm.decode(id)
#                 print(c, end="", flush=True)
#                 input_ids.append(id)
#                 d = f"{c}"
#                 text += d
#             while True:
#                 current_state =  get_state(llm, "ALL_JSON", text, function_name)
#                 logits = llm.get_logits_from_input_ids(current_state)
#                 par_name = p.replace('"', "").replace(':', "").replace(' ', '')
#                 if funct[function_name].parameters[par_name].type == "number":
#                     state_list = get_state(llm, "N_VALUES", text, function_name)
#                     pos_char = ','
#                     if final_param == len_param - 1:
#                         state_list = get_state(llm, "N_LAST_VALUES", text, function_name)
#                         pos_char = '}'
#                     for values in range(len(logits)):
#                         if values not in state_list:
#                             logits[values] = float('-inf')
#                     id = np.argmax(logits)
#                     c = llm.decode(id)
#                     input_ids.append(id)
#                     d = f"{c}"
#                     text += d
#                     if pos_char in d :
#                         decimal_point = llm.decode([13, 15])
#                         input_ids.append(13)
#                         input_ids.append(15)
#                         print( decimal_point, end="", flush=True)
#                         print(c, end="", flush=True)
#                         break
#                     print(c, end="", flush=True)
#                 else:
#                     pos_char = ','
#                     state_list = get_state(llm, "S_VALUES", my_prompt.replace("'", '"'), function_name)
#                     if final_param == len_param - 1:
#                         state_list = get_state(llm, "S_LAST_VALUES", my_prompt.replace("'", '"'), function_name)
#                         pos_char = '}'
#                     for values in range(len(logits)):
#                         if values not in state_list:
#                             logits[values] = float('-inf')
#                     id = np.argmax(logits)
#                     c = llm.decode(id)
#                     input_ids.append(id)
#                     d = f"{c}"
#                     text += d
#                     print(c, end="", flush=True)
#                     if pos_char in d :
#                         break

#             final_param += 1
#         if final_param == len_param:
#             c = llm.decode(92)
#             print(c, end="", flush=True)
#         break