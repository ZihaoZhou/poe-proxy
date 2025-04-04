import os
import secrets
import fastapi_poe as fp
import logging
import asyncio
import boto3
import json
import time
import yaml
from sanic import Sanic
from sanic.response import json as sanic_json
from json import dumps, loads
from sanic_cors import CORS
from base64 import urlsafe_b64encode
from fastapi_poe.client import PROTOCOL_VERSION, stream_request_base
from fastapi_poe.types import QueryRequest, ToolResultDefinition, ToolCallDefinition
from aggregate import aggregate_chunk
import requests
import base64
from loguru import logger as ll
import filetype

# =====================================================================
# CONFIGURATION AND LOGGING
# =====================================================================


def setup_logging():
    """Configure and return logger"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("poe_proxy")


def infer_format(raw_bytes):
    """Infer the image format from raw bytes using the filetype library."""
    kind = filetype.guess(raw_bytes)
    
    if kind is None:
        return None
    
    # Return the extension without the dot
    return kind.extension


def chunk_format(text_delta):
    return {
        "id": f"chatcmpl-{secrets.token_hex(16)}",
        "object": "chat.completion.chunk",
        "model": "bedrock",
        "created": int(time.time()),
        "choices": [
            {
                "delta": {
                    "content": text_delta,
                    "function_call": None,
                    "role": "assistant",
                    "tool_calls": None,
                },
                "finish_reason": None,
                "index": 0,
                "logprobs": None,
            }
        ],
        "system_fingerprint": f"fp_{secrets.token_hex(8)}",
    }


def load_config():
    """Load configuration from YAML file"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# Initialize logging and configuration
logger = setup_logging()
config = load_config()

# =====================================================================
# AWS BEDROCK CLIENT SETUP
# =====================================================================


class BedrockClient:
    """AWS Bedrock client manager"""

    # AWS model ID mapping
    MODEL_MAPPING = {
        "Claude-3.7-Sonnet-Reasoning": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude-3.7-Sonnet":           "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude-3.5-Sonnet":           "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "Claude-3.5-Haiku":            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "Claude-3-Sonnet":             "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "Claude-3-Haiku":              "us.anthropic.claude-3-haiku-20240307-v1:0",
        "DeepSeek-R1":                 "us.deepseek.r1-v1:0",
    }

    def __init__(self, config):
        self.client = None
        self.available = False

        # ---------------------------------
        # 1) AWS credentials check
        # ---------------------------------
        has_aws_credentials = (
            config.get("aws_access_key_id") is not None
            and config.get("aws_secret_access_key") is not None
        )

        # ---------------------------------
        # 2) Initialize client if possible
        # ---------------------------------
        if has_aws_credentials:
            logging.info("AWS credentials found. Initializing bedrock-runtime client.")
            try:
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=os.environ.get("AWS_REGION", "us-east-1"),
                    aws_access_key_id=config["aws_access_key_id"],
                    aws_secret_access_key=config["aws_secret_access_key"],
                )
                self.available = True
                logging.info("AWS Bedrock client (bedrock-runtime) initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
        else:
            logging.warning("AWS credentials not found. The Bedrock client is unavailable.")

        # ---------------------------------
        # 3) Add extended mapping variants
        # ---------------------------------
        extended_mapping = {}
        # For each existing model in MODEL_MAPPING, add:
        #   - A '-200k' variant
        #   - A lowercase version
        #   - A lowercase '-200k' variant
        for model_name, bedrock_id in self.MODEL_MAPPING.items():
            # '-200k' version
            extended_mapping[model_name + "-200k"] = bedrock_id
            # lowercase
            lowercase_name = model_name.lower()
            extended_mapping[lowercase_name] = bedrock_id
            extended_mapping[lowercase_name + "-200k"] = bedrock_id

        # Merge them back into our dict
        self.MODEL_MAPPING.update(extended_mapping)

    def is_model_supported(self, model_name: str) -> bool:
        """Check if a given model name is mapped and available."""
        return self.available and (model_name in self.MODEL_MAPPING)

    def get_model_id(self, model_name: str) -> str:
        """Return the underlying Bedrock model ID for the user-supplied model name."""
        return self.MODEL_MAPPING.get(model_name)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================


def random_id(prefix, nbytes=18):
    """Generate a random ID with the given prefix"""
    token = secrets.token_bytes(nbytes)
    return prefix + "-" + urlsafe_b64encode(token).decode("utf8")


async def try_api_keys(request_func, api_keys, request=None, is_streaming=False):
    """
    Try multiple API keys in sequence until one succeeds or all fail.
    For streaming responses, we need special handling to avoid multiple respond() calls.
    """
    if not isinstance(api_keys, list):
        api_keys = [api_keys]

    # For streaming requests, we need to create the response once before trying keys
    response = None
    if is_streaming and request is not None:
        response = await request.respond(
            content_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    last_exception = None
    for i, key in enumerate(api_keys):
        try:
            logger.info(f"Attempting with API key {i + 1}/{len(api_keys)}")
            if is_streaming and request is not None:
                # Pass the already created response to the handler
                return await request_func(key, response)
            else:
                return await request_func(key)
        except Exception as e:
            logger.warning(f"API key {i + 1}/{len(api_keys)} failed: {str(e)}")
            last_exception = e

    # If we get here, all keys have failed
    logger.error(f"All API keys failed. Last error: {str(last_exception)}")
    
    # For streaming, we need to send an error back over the stream
    if is_streaming and response is not None:
        error_chunk = {
            "id": f"error-{secrets.token_hex(16)}",
            "object": "chat.completion.chunk",
            "model": "error",
            "created": int(time.time()),
            "choices": [
                {
                    "delta": {
                        "content": f"Error: {str(last_exception)}",
                        "role": "assistant",
                    },
                    "finish_reason": "error",
                    "index": 0,
                }
            ],
        }
        await response.send("data: " + dumps(error_chunk) + "\n\n")
        await response.send("data: [DONE]\n\n")
        return True  # Indicate that we've handled the response
    else:
        raise last_exception


def poe_exception_handler(e):
    """Handle exceptions from Poe API calls"""
    if isinstance(e, fp.client.BotError):
        try:
            error_str = repr(e)
            error_str = error_str[error_str.index("(") + 1:error_str.rindex(")")]
            e = loads(error_str[1:-1])
            return sanic_json({"poe_error": e}, status=500)
        except Exception:
            pass
    return sanic_json({"error": str(e)}, status=500)


# =====================================================================
# API MODELS
# =====================================================================


class API:
    """API models handler"""

    def __init__(self):
        self.models = [
            "Claude-3.7-Sonnet",
            "Claude-3.5-Sonnet",
            "Claude-3.5-Haiku",
            "Claude-3-Sonnet",
            "Claude-3-Haiku",
            "DeepSeek-R1",
        ]

    def engines_list(self):
        """List available models/engines"""
        for model in self.models:
            yield {
                "id": model,
                "object": "engine",
                "owner": "openai",
                "ready": True,
            }


# =====================================================================
# AWS BEDROCK REQUEST HANDLERS
# =====================================================================


class BedrockHandler:
    """
    Handle AWS Bedrock requests using the `converse` and `converse_stream` APIs.

    This version supports documents and images via raw bytes, as well as streaming
    and static (non-streaming) requests. 
    """

    def __init__(self, bedrock_client):
        self.bedrock = bedrock_client

    def format_messages(self, messages):
        """
        Convert OpenAI-style messages into the format required by Bedrock's
        `converse` API. For documents/images, pass the raw bytes, not base64.

        Expected input example for a PDF/file:
        {
        "role": "user", 
        "content": [
            {"type": "text", "text": "What is in this PDF?"},
            {"type": "file", "file": {
            "name": "example.pdf", 
            "type": "application/pdf", 
            "content": "data:application/pdf;base64,JVBERi0xLjcN..."
            }}
        ]
        }
        
        For a file from a URL:
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this PDF?"},
            {"type": "file", "file": {
            "name": "example.pdf", 
            "type": "application/pdf", 
            "content": "http://example.com/myfile.pdf"
            }}
        ]
        }

        For an image from a URL:
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {
            "url": "http://example.com/myimage.png",
            "detail": "auto"
            }}
        ]
        }

        For an image already in base64:
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBORw0K...",
            "detail": "auto"
            }}
        ]
        }
        """
        formatted_messages = []
        for message in messages:
            role = message.get("role", "")
            # Doesn't support "system" role in this context
            if role == "system":
                continue
            
            # We will build an array of content blocks in the style Anthropic expects
            # Each block is typically {"text": "..."} or {"document": {...}} or {"image": {...}}
            content_blocks = []

            # Process the message content which must be an array in the new format
            if "content" in message:
                raw_content = message["content"]
                # If content is a string, treat it as plain text (for simple messages)
                if isinstance(raw_content, str) and raw_content.strip():
                    content_blocks.append({"text": raw_content})
                # If content is a list, process each item based on its type
                elif isinstance(raw_content, list):
                    for c in raw_content:
                        if isinstance(c, dict):
                            if "type" in c and c["type"] == "text" and "text" in c:
                                content_blocks.append({"text": c["text"]})
                            elif "type" in c and c["type"] == "file" and "file" in c:
                                file_obj = c["file"]
                                file_name = file_obj.get("name", "document")
                                file_data = file_obj.get("content", "")
                                
                                # Handle file from URL
                                if file_data.startswith("http://") or file_data.startswith("https://"):
                                    try:
                                        resp = requests.get(file_data, timeout=30)
                                        resp.raise_for_status()
                                        raw_bytes = resp.content
                                    except Exception as e:
                                        ll.error(f"Failed to fetch file from URL {file_data}: {str(e)}")
                                        raise ValueError(f"Failed to fetch file: {str(e)}")
                                else:
                                    # Handle base64 content
                                    raw_bytes = self._extract_raw_bytes(file_data)
                                
                                doc_format = infer_format(raw_bytes)
                                content_blocks.append({
                                    "document": {
                                        "name": self._sanitize_name(file_name), 
                                        "format": doc_format,
                                        "source": {
                                            "bytes": raw_bytes
                                        }
                                    }
                                })
                            elif "type" in c and c["type"] == "image_url" and "image_url" in c:
                                image_obj = c["image_url"]
                                if "url" in image_obj:
                                    url = image_obj["url"]
                                    # Check if it's a data URL or a regular URL
                                    if url.startswith("data:"):
                                        # Extract content type from data URL (e.g., "image/png")
                                        content_type = url.split(";")[0].split(":")[1] if ";" in url else "image/png"
                                        raw_bytes = self._extract_raw_bytes(url)
                                        content_blocks.append({
                                            "image": {
                                                "format": infer_format(raw_bytes),
                                                "source": {
                                                    "bytes": raw_bytes
                                                }
                                            }
                                        })
                                    else:
                                        # It's a regular URL, fetch it
                                        try:
                                            resp = requests.get(url, timeout=10)
                                            resp.raise_for_status()
                                            raw_bytes = resp.content
                                            # Try to determine content type from response headers or URL
                                            content_type = resp.headers.get("Content-Type", "image/png")
                                            content_blocks.append({
                                                "image": {
                                                    "format": self._map_mime_to_extension(content_type, is_image=True),
                                                    "source": {
                                                        "bytes": raw_bytes
                                                    }
                                                }
                                            })
                                        except Exception:
                                            raise ValueError(f"Invalid or private image URL: {url}")
                        elif isinstance(c, str):
                            content_blocks.append({"text": c})

            # If we collected any content blocks, add them
            if content_blocks:
                formatted_messages.append({
                    "role": role,
                    "content": content_blocks
                })

        return formatted_messages

    def prepare_request_body(self, formatted_messages, temperature, top_p, stop_sequences, reasoning=False):
        """
        Build the request body for the `converse` or `converse_stream` call.
        The temperature and other parameters should be in the inferenceConfig object.
        
        For example:
        {
            "modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [...],
            "inferenceConfig": {
                "temperature": 0.7,
                "stopSequences": [...]
            }
        }
        """
        # Create the inferenceConfig object
        inference_config = {
            "temperature": temperature,
            "topP": top_p
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            inference_config["stopSequences"] = stop_sequences

        # Build the complete request body
        request_body = {
            "messages": formatted_messages
        }
        
        if reasoning:
            request_body["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 4000
                }
            }
        else:
            request_body["inferenceConfig"] = inference_config

        return request_body
    
    async def handle_stream(self, request, model_id, request_body, response=None):
        """
        Handle streaming response from AWS Bedrock using `converse_stream`.

        `request_body` should contain the `messages` list (and temperature, etc.).
        We run the synchronous call in a thread executor, then parse each streaming chunk
        and convert it to OpenAI-like chunk_format.
        """
        # Only create a response if one wasn't provided (first attempt)
        if response is None:
            response = await request.respond(
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Run the synchronous `converse_stream` in a separate thread
        loop = asyncio.get_event_loop()
        aws_response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.client.converse_stream(
                modelId=model_id,
                **request_body
            ),
        )

        # The 'stream' key should contain the streaming events
        aws_stream = aws_response.get("stream", [])
        start_reasoning = False
        for event in aws_stream:
            if "messageStart" in event:
                continue

            elif "contentBlockDelta" in event:
                # This is the actual partial text chunk
                delta = event["contentBlockDelta"].get("delta", {})
                
                # Handle reasoning part
                if "reasoningContent" in delta:
                    text = delta["reasoningContent"].get("text", "")
                    if not start_reasoning:
                        text = "> " + text
                        start_reasoning = True
                    else:
                        text = text.replace("\n", "\n> ")
                else:
                    text = delta.get("text", "")
                    if start_reasoning:
                        start_reasoning = False
                        text = "\n\n" + text
                if text:
                    # Wrap in OpenAI-like chunk format
                    chunk_data = chunk_format(text)
                    await response.send("data: " + json.dumps(chunk_data) + "\n\n")

            elif "contentBlockStop" in event:
                # End of a chunk block
                continue

            elif "messageStop" in event:
                await response.send("data: [DONE]\n\n")
                break

            elif "metadata" in event:
                pass

            # Yield control back to the event loop
            await asyncio.sleep(0)

        return True
    
    async def handle_static(self, model_id, request_body):
        """
        Handle non-streaming (synchronous) response from AWS Bedrock using `converse`.
        We'll call `converse` in a thread executor, parse the JSON result, and wrap it
        in an OpenAI-like completion structure.
        """
        loop = asyncio.get_event_loop()
        aws_response = await loop.run_in_executor(
            None,
            lambda: self.bedrock.client.converse(
                modelId=model_id,
                **request_body
            ),
        )
        # The response for a non-streaming call is typically a single message from Claude
        # For example:
        # {
        #   "messages":[
        #       {"content":[{"text":"Hello, world!"}], "role":"assistant"}
        #   ],
        #   "metadata": {... usage stats...}
        # }
        messages = aws_response.get("messages", [])
        assistant_message = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                # It's an array of content blocks: e.g. [{"text":"some text"}, {"text":"..."}, ...]
                content_list = msg.get("content", [])
                for c in content_list:
                    # c might be {"text": "..."} or {"document": {...}} or {"image": {...}}
                    if "text" in c:
                        assistant_message += c["text"]
                # We only gather from the first assistant block or combine them all
                break

        # Wrap it in an OpenAI-style JSON
        return {
            "id": f"chatcmpl-{secrets.token_hex(16)}",
            "object": "chat.completion",
            "model": model_id,
            "created": int(time.time()),
            "choices": [
                {
                    "message": {"content": assistant_message, "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": -1,  # Not provided by Bedrock
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }

    # ----------------------------------------------------------------
    # HELPER FUNCTIONS
    # ----------------------------------------------------------------

    def _extract_raw_bytes(self, data_uri):
        """
        Given a data URI like 'data:application/pdf;base64,JVBERi0xLjcN...' or
        'data:image/png;base64,iVBOR...', return the raw bytes.
        If `data_uri` is empty or unrecognized, return b''.
        """
        if not data_uri:
            return b""
        # Typical structure: data:<mime>;base64,<base64_data>
        if data_uri.startswith("data:") and ";base64," in data_uri:
            base64_part = data_uri.split(";base64,", 1)[1]
            return base64.b64decode(base64_part)
        # In case it's plain base64, no prefix:
        try:
            return base64.b64decode(data_uri)
        except Exception:
            # Not valid base64, return empty
            return b""

    def _sanitize_name(self, name_str):
        """
        Because document names can only contain certain characters for Bedrock,
        remove/replace disallowed characters. (Just an example.)
        """
        import re
        return re.sub(r"[^0-9A-Za-z \-\(\)\[\]]", "_", name_str)


# =====================================================================
# POE REQUEST HANDLERS
# =====================================================================


class PoeHandler:
    """Handle Poe API requests"""

    def __init__(self, config):
        self.config = config

    def format_protocol_messages(self, messages, model):
        """Format messages for Poe API"""
        protocol_messages = []
        tool_calls = None
        tool_results = None

        for message in messages:
            if "role" not in message:
                raise ValueError("role is required in message")
            if "content" not in message:
                raise ValueError("content is required in message")

            if "tool_calls" in message:  # tool_results
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    tool_calls.append(
                        ToolCallDefinition(
                            id=tool_call["id"],
                            function=ToolCallDefinition.FunctionDefinition(
                                name=tool_call["function"]["name"],
                                arguments=tool_call["function"]["arguments"],
                            ),
                            type=tool_call["type"],
                        )
                    )
            elif "tool_call_id" in message:
                if tool_results is None:
                    tool_results = []
                tool_results.append(ToolResultDefinition(**message))
            else:
                assert "content" in message
                content = ""
                if isinstance(message["content"], str):
                    content = message["content"]
                elif isinstance(message["content"], list):
                    for c in message["content"]:
                        assert c['type'] in ["text"], "This model only supports text input. Please switch to Anthropic models or Deepseek R1."
                        if content != "":
                            content += "\n---\n"
                        content += c["text"]
                role = message["role"].replace("assistant", "bot")
                if "GPT" not in model or "Gemini" not in model or "gpt" not in model:
                    if role == "system":
                        continue
                protocol_messages.append(
                    fp.ProtocolMessage(role=role, content=content)
                )

        return protocol_messages, tool_calls, tool_results

    def create_query_request(
        self, protocol_messages, temperature, stop_sequences, logit_bias, language_code
    ):
        """Create query request for Poe API"""
        return QueryRequest(
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
    
    async def handle_stream(
        self, request, query, model, api_key, tools, tool_calls, tool_results, response=None
    ):
        """
        Handle streaming response from Poe API
        Now accepts a pre-created response object to avoid creating it multiple times
        """
        # Only create a response if one wasn't provided (first attempt)
        if response is None:
            response = await request.respond(
                content_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        async for partial in stream_request_base(
            request=query,
            bot_name=model,
            api_key=api_key,
            tools=tools,
            tool_calls=tool_calls,
            tool_results=tool_results,
        ):
            if partial.data is None:
                # Handle text completion chunk
                chunk = chunk_format(partial.text)
            else:
                chunk = partial.data

            await response.send("data: " + dumps(chunk) + "\n\n")

        await response.send("data: [DONE]\n\n")
        return True

    async def handle_static(
        self, query, model, api_key, tools, tool_calls, tool_results
    ):
        """Handle static (non-streaming) response from Poe API"""
        partial_responses = []
        tool_flag = False

        async for partial in stream_request_base(
            request=query,
            bot_name=model,
            api_key=api_key,
            tools=tools,
            tool_calls=tool_calls,
            tool_results=tool_results,
        ):
            if partial.data is None:
                partial_responses.append(partial.text)
            else:
                tool_flag = True
                partial_responses.append(
                    loads(dumps(partial.data, default=lambda obj: obj.__dict__))
                )

        if tool_flag:
            return aggregate_chunk(partial_responses)
        else:
            bot_response = "".join(partial_responses)
            return {
                "id": f"chatcmpl-{secrets.token_hex(16)}",
                "object": "chat.completion",
                "model": model,
                "created": int(time.time()),
                "choices": [
                    {
                        "message": {"content": bot_response, "role": "assistant"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "usage": {
                    "prompt_tokens": sum(len(m.content) for m in query.query),
                    "completion_tokens": len(bot_response),
                    "total_tokens": sum(len(m.content) for m in query.query)
                    + len(bot_response),
                },
            }


# =====================================================================
# SANIC APP AND ROUTES
# =====================================================================

# Initialize at module level
app = Sanic("Poe")
CORS(app)


def configure_app(app, config):
    """Configure Sanic app with routes"""

    @app.route("/v1/engines")
    async def v1_engines_list(request):
        """List available engines/models"""
        res = {"object": "list", "data": []}
        for result in api.engines_list():
            res["data"].append(result)
        return sanic_json(res)
    
    @app.route("/v1/chat/completions", methods=["POST"])
    async def v1_engines_completions(request):
        """Handle chat completions endpoint"""
        try:
            # Extract request parameters
            kws = request.json
            messages = kws.pop("messages", None)
            model = kws.pop("model", "GPT-4")
            top_p = kws.pop("top_p", 1.0)
            stop_sequences = kws.pop("stop", [])
            logit_bias = kws.pop("logit_bias", {})
            temperature = kws.pop("temperature", 0.7)
            language_code = kws.pop("language", "en")
            stream = kws.pop("stream", False)
            tools_dict_list = kws.pop("tools", None)

            # Process tool definitions
            if tools_dict_list is None:
                tools = None
            else:
                tools = [
                    fp.ToolDefinition(**tools_dict) for tools_dict in tools_dict_list
                ]

            # Process stop sequences
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            # Validate API key
            api_key = request.headers.get("Authorization", "").split("Bearer ")[-1]
            if any([api_key == key for key in config["extra_api_keys"]]):
                api_key = config["api_key"]

            if not api_key or not messages:
                return sanic_json(
                    {"error": "API key and prompt are required"}, status=400
                )

            # Check if we should use AWS Bedrock
            use_aws = bedrock_client.is_model_supported(model)

            if use_aws:
                logger.info(f"Using AWS Bedrock backend for model: {model}")

                reasoning = False
                # Map the model name to AWS Bedrock model ID
                if "Reasoning" in model:
                    reasoning = True
                model_id = bedrock_client.get_model_id(model)
                if not model_id:
                    raise ValueError(f"Unsupported model for AWS Bedrock: {model}")

                # Format messages for AWS Bedrock
                formatted_messages = bedrock_handler.format_messages(messages)

                # Prepare request body
                request_body = bedrock_handler.prepare_request_body(
                    formatted_messages, temperature, top_p, stop_sequences, reasoning
                )

                if stream:
                    # Use the updated stream handler with the fixed try_api_keys function
                    # For AWS Bedrock we don't need to retry with multiple keys,
                    # but we're keeping the same pattern for consistency
                    async def make_stream_request(current_api_key, response=None):
                        return await bedrock_handler.handle_stream(
                            request, model_id, request_body, response
                        )
                    
                    # Note: we're not really using API keys for AWS, but using this function 
                    # for error handling consistency
                    await try_api_keys(
                        make_stream_request, 
                        ["aws_key"],  # Dummy key for AWS calls
                        request=request, 
                        is_streaming=True
                    )
                    return
                else:
                    # Handle static response
                    result = await bedrock_handler.handle_static(model_id, request_body)
                    return sanic_json(result)

            # Continue with Poe backend if not using AWS
            logger.info(f"Using Poe backend for model: {model}")

            try:
                # Format messages for Poe API
                protocol_messages, tool_calls, tool_results = (
                    poe_handler.format_protocol_messages(messages, model)
                )

                # Create query request
                query = poe_handler.create_query_request(
                    protocol_messages,
                    temperature,
                    stop_sequences,
                    logit_bias,
                    language_code,
                )

                if stream:
                    # Define the streaming request function using the updated approach
                    async def make_stream_request(current_api_key, response=None):
                        return await poe_handler.handle_stream(
                            request,
                            query,
                            model,
                            current_api_key,
                            tools,
                            tool_calls,
                            tool_results,
                            response
                        )

                    await try_api_keys(
                        make_stream_request, 
                        config["api_key"],
                        request=request,
                        is_streaming=True
                    )
                    return
                else:
                    # Define the static request function to use with multiple API keys
                    async def make_static_request(current_api_key):
                        return await poe_handler.handle_static(
                            query,
                            model,
                            current_api_key,
                            tools,
                            tool_calls,
                            tool_results,
                        )

                    result = await try_api_keys(make_static_request, config["api_key"])
                    return sanic_json(result)

            except Exception as e:
                return poe_exception_handler(e)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return sanic_json({"error": str(e)}, status=500)
        
    return app


# Initialize API and AWS Bedrock client at module level
api = API()
bedrock_client = BedrockClient(config)
bedrock_handler = BedrockHandler(bedrock_client)
poe_handler = PoeHandler(config)

# Configure the app with routes
configure_app(app, config)

# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    # Reset AWS_ENDPOINT_URL if it's in environment variables
    if "AWS_ENDPOINT_URL" in os.environ:
        os.environ["AWS_ENDPOINT_URL"] = ""

    # Get port from environment variable or config
    port = int(os.environ.get("PORT", config["port"]))

    # Check if SSL should be used
    use_ssl = config["ssl"]
    if "USE_HTTP" in os.environ and os.environ["USE_HTTP"].lower() == "true":
        use_ssl = False

    # Run the app
    if use_ssl:
        app.run("::", port, ssl=config["ssl_dir"])
    else:
        app.run(host="0.0.0.0", port=port, auto_reload=True, debug=True)
