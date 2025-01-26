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
from fastapi_poe.client import PROTOCOL_VERSION, stream_request_base
from fastapi_poe.types import QueryRequest, ToolResultDefinition, ToolCallDefinition
from aggregate import aggregate_chunk

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


def adapt_model(model_name):
    if model_name.startswith("gpt-3.5-turbo"):
        return "GPT-3.5-Turbo"
    elif model_name.startswith("gpt-4o-mini"):
        return "GPT-4o-mini"
    elif model_name.startswith("gpt-4o"):
        return "GPT-4o"
    elif model_name.startswith("gpt-4"):
        return "GPT-4"
    else:
        return model_name


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
    model = adapt_model(model)
    stop_sequences = kws.pop("stop", [])
    logit_bias = kws.pop("logit_bias", {})
    temperature = kws.pop("temperature", 0.7)
    language_code = kws.pop("language", "en")
    stream = kws.pop("stream", False)
    tools_dict_list = kws.pop("tools", None)
    if tools_dict_list is None:
        tools = None
    else:
        tools = [fp.ToolDefinition(**tools_dict) for tools_dict in tools_dict_list]
    if type(stop_sequences) is str:
        stop_sequences = [stop_sequences]
    
    api_key = request.headers.get("Authorization", "").split("Bearer ")[-1]
    if any([api_key == key for key in config["extra_api_keys"]]):
        api_key = config["api_key"]

    if not api_key or not messages:
        return json({"error": "API key and prompt are required"}, status=400)

    protocol_messages = []
    tool_calls = None
    tool_results = None
    for message in messages:
        if "role" not in message:
            return json({"error": "role is required in message"}, status=400)
        if "content" not in message:
            return json({"error": "content is required in message"}, status=400)
        if "tool_calls" in message:  # tool_results
            tool_calls = []
            for tool_call in message["tool_calls"]:
                tool_calls.append(ToolCallDefinition(
                    id=tool_call["id"],
                    function=ToolCallDefinition.FunctionDefinition(
                        name=tool_call["function"]["name"],
                        arguments=tool_call["function"]["arguments"]
                    ),
                    type=tool_call["type"]
                ))
        elif "tool_call_id" in message:
            if tool_results is None:
                tool_results = []
            tool_results.append(ToolResultDefinition(**message))
        else:
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
    query = QueryRequest(
        query=protocol_messages,
        user_id="",
        conversation_id="",
        message_id="",
        version=PROTOCOL_VERSION,
        type="query",
        temperature=temperature,
        skip_system_prompt=False,
        language_code=language_code,
        stop_sequences=stop_sequences,
        logit_bias=logit_bias,
    )
    
    if stream:
        flag = True
        try:
            async for partial in stream_request_base(
                request=query,
                bot_name=model,
                api_key=api_key,
                tools=tools,
                tool_calls=tool_calls,
                tool_results=tool_results
            ):
                if partial.data is None:
                    # Handle text completion chunk
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
                                "index": 0,
                                "logprobs": None
                            }
                        ],
                        "system_fingerprint": f"fp_{secrets.token_hex(8)}"
                    }
                else:
                    chunk = partial.data
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
        tool_flag = False
        try:
            async for partial in stream_request_base(
                request=query,
                bot_name=model,
                api_key=api_key,
                tools=tools,
                tool_calls=tool_calls,
                tool_results=tool_results
            ):
                if partial.data is None:
                    partial_responses.append(partial.text)
                else:
                    tool_flag = True
                    partial_responses.append(loads(dumps(partial.data, default=lambda obj: obj.__dict__)))
            
        except Exception as e:
            return poe_exception_handler(e)
        
        if tool_flag:
            return json(aggregate_chunk(partial_responses))
        else:
            bot_response = "".join(partial_responses)
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
    if "USE_HTTP" in os.environ and os.environ["USE_HTTP"].lower() == "true":
        config["ssl"] = False
    if config["ssl"]:
        app.run("::", port, ssl=config["ssl_dir"])
    else:
        app.run(host="0.0.0.0", port=port, auto_reload=True, debug=True)
