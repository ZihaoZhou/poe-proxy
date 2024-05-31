import os
import secrets
import fastapi_poe as fp

from sanic import Sanic
from sanic.response import json
from json import dumps, loads

from sanic_cors import CORS
from base64 import urlsafe_b64encode 
import time
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def poe_exception_handler(e):
    if isinstance(e, fp.client.BotError):
        try:
            error_str = repr(e)
            error_str = error_str[error_str.index("(") + 1: error_str.rindex(")")]
            e = loads(error_str[1:-1])
            return json({"poe_error": e}, status=500)
        except Exception:
            pass
    return json({"error": str(e)}, status=500)


class API:
    def __init__(self):
        self.models = [
            "GPT-4",
            "GPT-4-32k",
            "Claude-3-Opus",
            "Claude-3-Sonnet",
            "GPT-3.5-Turbo",
            "ChatGPT-16k",
            "Mistral-Large",
            "Gemini-Pro"
        ]

    def engines_list(self):
        for model in self.models:
            yield {
                "id": model,
                "object": "engine",
                "owner": "openai",
                "ready": True,
            }


api = API()

app = Sanic("Poe")
CORS(app)

model_adapter = {
    'gpt-3.5-turbo': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-0301': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-0613': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-16k': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-16k-0613': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-1106': 'GPT-3.5-Turbo',
    'gpt-3.5-turbo-0125': 'GPT-3.5-Turbo',
    'gpt-4': 'GPT-4',
    'gpt-4-0314': 'GPT-4',
    'gpt-4-0613': 'GPT-4',
    'gpt-4-32k': 'GPT-4',
    'gpt-4-32k-0314': 'GPT-4',
    'gpt-4-32k-0613': 'GPT-4',
    'gpt-4-1106-preview': 'GPT-4',
    'gpt-4-0125-preview': 'GPT-4'
}


@app.route("/v1/engines")
async def v1_engines_list(request):
    res = {"object": "list", "data": []}
    for result in api.engines_list():
        res["data"].append(result)
    return json(res)


def random_id(prefix, nbytes=18):
    token = secrets.token_bytes(nbytes)
    return prefix + "-" + urlsafe_b64encode(token).decode("utf8")


@app.route("/v1/chat/completions", methods=["POST"])
async def v1_engines_completions(request):
    kws = request.json
    messages = kws.pop("messages", None)
    
    model = kws.pop("model", "GPT-4")
    if model in model_adapter:
        model = model_adapter[model]
    temperature = kws.pop("temperature", 1)
    stream = kws.pop("stream", False)
    
    api_key = request.headers.get("Authorization", "").split("Bearer ")[-1]
    if any([api_key == key for key in config["extra_api_keys"]]):
        api_key = config["api_key"]

    if not api_key or not messages:
        return json({"error": "API key and prompt are required"}, status=400)

    protocol_messages = []
    for message in messages:
        if "role" not in message:
            return json({"error": "role is required in message"}, status=400)
        if "content" not in message:
            return json({"error": "content is required in message"}, status=400)
        role = message["role"].replace("assistant", "bot")
        if "GPT" not in model or "Gemini" not in model or "gpt" not in model:
            if role == "system":
                continue
        protocol_messages.append(
            fp.ProtocolMessage(
                role=role,
                content=message["content"]
            )
        )

    if stream:
        i = 0
        flag = True
        
        try:
            async for partial in fp.get_bot_response(
                messages=protocol_messages, 
                bot_name=model, 
                api_key=api_key,
                temperature=temperature
            ):
                chunk = {
                    "id": f"chatcmpl-{secrets.token_hex(16)}",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "created": int(time.time()),
                    "choices": [
                        {
                            "delta": {
                                "content": partial.text,
                                "function_call": None,
                                "role": "assistant",
                                "tool_calls": None
                            },
                            "finish_reason": None,
                            "index": i,
                            "logprobs": None
                        }
                    ],
                    "system_fingerprint": f"fp_{secrets.token_hex(8)}"
                }
                i += 1
                if flag:
                    response = await request.respond()
                    flag = False
                await response.send("data: " + dumps(chunk) + "\n\n")
            await response.send("data: [DONE]\n\n")
        except Exception as e:
            return poe_exception_handler(e)
    else:
        # Handle static API call
        partial_responses = []
        try:
            async for partial in fp.get_bot_response(
                messages=protocol_messages,
                bot_name=model,
                api_key=api_key,
                temperature=temperature
            ):
                partial_responses.append(partial.text)
            
            bot_response = "".join(partial_responses)
        except Exception as e:
            return poe_exception_handler(e)
        
        response = {
            "id": f"chatcmpl-{secrets.token_hex(16)}",
            "object": "chat.completion",
            "model": model,
            "created": int(time.time()),
            "choices": [
                {
                    "message": {
                        "content": bot_response,
                        "role": "assistant"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content) for m in protocol_messages),
                "completion_tokens": len(bot_response),
                "total_tokens": sum(len(m.content) for m in protocol_messages) + len(bot_response)
            }
        }
        return json(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", config["port"]))
    if config["ssl"]:
        app.run("::", port, ssl=config["ssl_dir"])
    else:
        app.run(host="0.0.0.0", port=port)
