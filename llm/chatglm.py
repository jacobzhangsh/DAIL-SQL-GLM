import json.decoder

from utils.enums import LLM
import time
from transformers import AutoTokenizer, AutoModel

# define model/tokenizer
tokenizer = None
model = None

def init_chatglm():
    global tokenizer = AutoTokenizer.from_pretrained(LLM.GLM3_7B, trust_remote_code=True)
    global model = AutoModel.from_pretrained(LLM.GLM3_7B, trust_remote_code=True,max_length=2048).half().cuda()

def ask_completion(batch):
    response, history = model.chat(tokenizer, batch, history=[])
    response_clean = [_["text"] for _ in response["choices"]]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def ask_chat(messages: list, n):
    response, history = model.chat(tokenizer, messages, history=[])
    response_clean = [choice["message"]["content"] for choice in response["choices"]]
    if n == 1:
        response_clean = response_clean[0]
    return dict(
        response=response_clean,
        **response["usage"]
    )


def ask_llm(task_type: str, batch: list, temperature: float, n:int):
    n_repeat = 0
    while True:
        try:
            if task_type in LLM.TASK_COMPLETIONS:
                # TODO: self-consistency in this mode
                assert n == 1
                response = ask_completion(batch)
            elif task_type in LLM.TASK_CHAT:
                # batch size must be 1
                assert len(batch) == 1, "batch must be 1 in this mode"
                messages = [{"role": "user", "content": batch[0]}]
                response = ask_chat(messages, n)
                response['response'] = [response['response']]
            break
        except json.decoder.JSONDecodeError:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for JSONDecodeError", end="\n")
            time.sleep(1)
            continue
        except Exception as e:
            n_repeat += 1
            print(f"Repeat for the {n_repeat} times for exception: {e}", end="\n")
            time.sleep(1)
            continue

    return response
