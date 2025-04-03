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
        "Claude-3.7-Sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "Claude-3.5-Sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "Claude-3.5-Haiku": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "Claude-3-Sonnet": "us.anthropic.claude-3-sonnet-20240229-v1:0",
        "Claude-3-Haiku": "us.anthropic.claude-3-haiku-20240307-v1:0",
    }

    def __init__(self, config):
        """Initialize AWS Bedrock client if credentials are available"""
        self.client = None
        self.available = False

        # Check if AWS credentials are available
        has_aws_credentials = (
            config.get("aws_access_key_id") is not None
            and config.get("aws_secret_access_key") is not None
        )

        if has_aws_credentials:
            logger.info("AWS credentials found. AWS Bedrock backend is available.")
            # Initialize AWS Bedrock client
            try:
                self.client = boto3.client(
                    service_name="bedrock-runtime",
                    region_name=os.environ.get("AWS_REGION", "us-east-2"),
                    aws_access_key_id=config["aws_access_key_id"],
                    aws_secret_access_key=config["aws_secret_access_key"],
                )
                self.available = True
                logger.info("AWS Bedrock client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize AWS Bedrock client: {str(e)}")
        else:
            logger.info("AWS credentials not found. Using Poe backend only.")

        # Create extended model mapping with "-200k" variants
        extended_mapping = {}
        for model, model_id in self.MODEL_MAPPING.items():
            extended_mapping[model + "-200k"] = model_id
            # uncapitalized version
            extended_mapping[model.lower()] = model_id
            extended_mapping[model.lower() + "-200k"] = model_id

        # Merge the dictionaries to include both regular and 200k variants
        self.MODEL_MAPPING.update(extended_mapping)

    def is_model_supported(self, model):
        """Check if model is supported by AWS Bedrock"""
        return self.available and model in self.MODEL_MAPPING

    def get_model_id(self, model):
        """Get AWS Bedrock model ID for the given model name"""
        return self.MODEL_MAPPING.get(model)


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
    """Handle AWS Bedrock requests"""

    def __init__(self, bedrock_client):
        self.bedrock = bedrock_client.client

    def format_messages(self, messages):
        """Format messages for AWS Bedrock"""
        formatted_messages = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Skip empty content
            if not content:
                continue

            # Map roles to AWS Bedrock format
            if role == "system":
                continue  # Skip system messages
            elif role == "user":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})

        return formatted_messages

    def prepare_request_body(self, formatted_messages, temperature, stop_sequences, reasoning=False):
        """Prepare request body for AWS Bedrock"""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": formatted_messages,
            "temperature": 1.0 if reasoning else temperature,
            "max_tokens": 8192,
            **({"thinking": {
                "type": "enabled",
                "budget_tokens": 4000
            }} if reasoning else {}),
        }

        # Add stop sequences if provided
        if stop_sequences:
            request_body["stop_sequences"] = stop_sequences

        return request_body
    
    async def handle_stream(self, request, model_id, request_body, response=None):
        """
        Handle streaming response from AWS Bedrock
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

        # Run the synchronous AWS call in a separate thread
        loop = asyncio.get_event_loop()
        aws_response = await loop.run_in_executor(
            None,  # Use default executor
            lambda: self.bedrock.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="*/*",
                body=json.dumps(request_body),
            ),
        )

        aws_stream = aws_response.get("body")
        if aws_stream:
            # Process the stream chunks
            for event in aws_stream:
                chunk = event.get("chunk")
                if chunk:
                    data_str = chunk["bytes"].decode("utf-8")
                    partial_json = json.loads(data_str)

                    chunk_type = partial_json.get("type")
                    print(f"Chunk type: {chunk_type}")

                    if chunk_type == "content_block_start" or chunk_type == "content_block_stop":
                        if "thinking" in request_body:
                            chunk_data = chunk_format('\n***\n')
                        else:
                            chunk_data = ""
                        await response.send("data: " + dumps(chunk_data) + "\n\n")
                        # Yield control back to the event loop
                        await asyncio.sleep(0)
                    elif chunk_type == "content_block_delta":
                        if partial_json["delta"].get("type", "") == "thinking_delta":
                            text_delta = partial_json["delta"].get("thinking", "")
                        else:
                            text_delta = partial_json["delta"].get("text", "")
                        if text_delta:
                            chunk_data = chunk_format(text_delta)
                            await response.send("data: " + dumps(chunk_data) + "\n\n")
                            # Yield control back to the event loop
                            await asyncio.sleep(0)

                    elif (
                        chunk_type == "message_stop"
                    ):
                        await response.send("data: [DONE]\n\n")
                        break
                    else:
                        continue
        
        return True
    
    async def handle_static(self, model_id, request_body):
        """Handle static (non-streaming) response from AWS Bedrock"""
        # Run the synchronous AWS call in a separate thread
        loop = asyncio.get_event_loop()
        aws_response = await loop.run_in_executor(
            None,  # Use default executor
            lambda: self.bedrock.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="*/*",
                body=json.dumps(request_body),
            ),
        )
        
        # Parse the response body
        response_body = json.loads(aws_response["body"].read())
        
        # Extract the response text
        assistant_message = ""
        if "content" in response_body:
            assistant_message = response_body["content"][0]["text"]
        
        # Format in OpenAI-compatible response format
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
                "prompt_tokens": -1,  # AWS doesn't provide token counts
                "completion_tokens": -1,
                "total_tokens": -1,
            },
        }


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
                        assert c['type'] in ["text"]
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
                    formatted_messages, temperature, stop_sequences, reasoning
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
