# Poe Proxy

A proxy server that converts Poe and AWS model APIs to OpenAI protocol, allowing you to use these models with tools and clients built for the OpenAI API.

## Features

- Converts Poe API to OpenAI API format
- Supports AWS Bedrock for Claude-series models
- Supports function calling
- Multiple API key management
- Production-ready with proper error handling

## Installation

### Requirements

- Python 3.10+
- Poetry for dependency management

### Setup

1. Clone the repository
2. Install dependencies with Poetry:
```bash
poetry install
```
3. Copy the example config and customize it:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your API keys and settings
```

## Usage

Run the server locally:

```bash
poetry run python poe-server.py
```

## Docker Deployment (Recommended)

The repository includes a Dockerfile for easy deployment. For production use, we recommend using Docker Compose.

Example `docker-compose.yml`:

```yaml
version: '3.8'

services:
  poe-server:
    image: zihaokevinzhou/poe-proxy:latest
    container_name: poe-server
    ports:
      - "8000:8080"
    environment:
      - PORT=8080
      - USE_HTTP=true
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    restart: unless-stopped

networks:
  default:
    name: poe-network
    driver: bridge
```

Deploy with:

```bash
docker-compose up -d
```

Access the OpenAI-compatible API at `http://localhost:8000/v1/`.

## Configuration

```
port: 8080
ssl: true
ssl_dir: /etc/certificate
aws_secret_access_key: <your_aws_secret_access_key> 
aws_access_key_id: <your_aws_access_key_id>
# If aws keys are set, the server will use AWS Bedrock for Claude models
api_key:
  - <Poe API Key at https://poe.com/api_key>
  - <You can add more keys such that if the first key is rate limited, the second key will be used>
extra_api_keys:
  - <your_extra_api_key>
  - <These keys can be used for authentication without leaking your Poe API key>
```

## License

MIT License