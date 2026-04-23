import copy
import json
import os
import logging
import uuid
import httpx
import asyncio
import requests  # <-- ДОБАВЛЕНО для Bing Search
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
)

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response,
    format_non_streaming_response,
    convert_to_pf_format,
    format_pf_non_streaming_response,
)

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    
    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e
    
    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"


# Frontend Settings via Environment Variables
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}


# Enable Microsoft Defender for Cloud Integration
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"


azure_openai_tools = []
azure_openai_available_tools = []


# ============================================================================
# BING SEARCH INTEGRATION
# ============================================================================

def search_bing_web(query: str) -> str:
    """Выполняет поиск в Bing и возвращает результаты в JSON формате"""
    bing_key = os.getenv("BING_SEARCH_KEY")
    bing_url = os.getenv("BING_SEARCH_URL", "https://api.bing.microsoft.com/v7.0/search")
    
    if not bing_key:
        logging.warning("BING_SEARCH_KEY not configured")
        return json.dumps({"error": "Bing search not configured"})
    
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {
        "q": query,
        "count": 5,
        "textDecorations": False,
        "textFormat": "Raw",
        "mkt": "ru-RU"  # Регион Россия для русскоязычных результатов
    }
    
    try:
        response = requests.get(bing_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        
        output = []
        for result in search_results.get("webPages", {}).get("value", []):
            output.append({
                "title": result.get("name", ""),
                "snippet": result.get("snippet", ""),
                "url": result.get("url", "")
            })
        
        # Если есть новости, добавляем их
        for result in search_results.get("news", {}).get("value", []):
            output.append({
                "title": result.get("name", ""),
                "snippet": result.get("description", ""),
                "url": result.get("url", ""),
                "source": result.get("provider", [{}])[0].get("name", "")
            })
        
        return json.dumps(output, ensure_ascii=False)
        
    except requests.exceptions.RequestException as e:
        logging.exception(f"Bing search request failed: {e}")
        return json.dumps({"error": f"Search request failed: {str(e)}"})
    except Exception as e:
        logging.exception(f"Bing search failed: {e}")
        return json.dumps({"error": str(e)})


# Описание инструмента поиска для Azure OpenAI
bing_tool = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for real-time information. Use this function for questions about current events, weather, news, sports scores, stock prices, or any information that changes over time or after May 2025.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information. For weather, use format 'weather [city name]'. For news, use '[topic] news'."
                }
            },
            "required": ["query"]
        }
    }
}


# ============================================================================
# AZURE OPENAI CLIENT INITIALIZATION
# ============================================================================

async def init_openai_client():
    azure_openai_client = None
    
    try:
        # API version check
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint and
            not app_settings.azure_openai.resource
        ):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Authentication
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        # Default Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )

        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        raise e


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


def prepare_model_args(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        messages = [
            {
                "role": "system",
                "content": app_settings.azure_openai.system_message
            }
        ]

    for message in request_messages:
        if message:
            match message["role"]:
                case "user":
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                case "assistant" | "function" | "tool":
                    messages_helper = {"role": message["role"]}
                    if "name" in message:
                        messages_helper["name"] = message["name"]
                    if "function_call" in message:
                        messages_helper["function_call"] = message["function_call"]
                    messages_helper["content"] = message["content"]
                    if "context" in message:
                        context_obj = json.loads(message["context"])
                        messages_helper["context"] = context_obj
                    messages.append(messages_helper)

    user_security_context = None
    if MS_DEFENDER_ENABLED:
        authenticated_user_details = get_authenticated_user_details(request_headers)
        application_name = app_settings.ui.title
        user_security_context = get_msdefender_user_json(
            authenticated_user_details, request_headers, application_name
        )

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model
    }

    if len(messages) > 0 and messages[-1]["role"] == "user":
        if app_settings.datasource:
            model_args["extra_body"] = {
                "data_sources": [
                    app_settings.datasource.construct_payload_configuration(
                        request=request
                    )
                ]
            }

    if model_args.get("extra_body") is None:
        model_args["extra_body"] = {}
    if user_security_context:
        model_args["extra_body"]["user_security_context"] = user_security_context.to_dict()

    logging.debug(f"REQUEST BODY: {json.dumps(model_args, indent=4, default=str)}")
    return model_args


# ============================================================================
# MAIN CHAT REQUEST WITH BING SEARCH INTEGRATION
# ============================================================================

async def send_chat_request(request_body, request_headers):
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)
    
    request_body['messages'] = filtered_messages
    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()
        
        # Первый вызов с инструментом поиска
        response = await azure_openai_client.chat.completions.create(
            model=model_args["model"],
            messages=model_args["messages"],
            tools=[bing_tool],
            tool_choice="auto",
            temperature=model_args.get("temperature", 0.7),
            max_tokens=model_args.get("max_tokens", 1000),
            top_p=model_args.get("top_p", 1),
        )
        
        response_message = response.choices[0].message
        
        # Проверяем, нужно ли вызывать функцию поиска
        if response_message.tool_calls:
            # Добавляем ответ ассистента в историю
            model_args["messages"].append(response_message)
            
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_web":
                    # Выполняем поиск в Bing
                    args = json.loads(tool_call.function.arguments)
                    search_results = search_bing_web(args.get("query", ""))
                    
                    # Добавляем результат поиска в историю
                    model_args["messages"].append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_results
                    })
            
            # Второй вызов с результатами поиска
            final_response = await azure_openai_client.chat.completions.create(
                model=model_args["model"],
                messages=model_args["messages"],
                temperature=model_args.get("temperature", 0.7),
                max_tokens=model_args.get("max_tokens", 1000),
                top_p=model_args.get("top_p", 1),
            )
            
            return final_response, str(uuid.uuid4())
        
        return response, str(uuid.uuid4())
        
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e


async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})
    
    async def generate(apim_request_id, history_metadata):
        if hasattr(response, 'choices') and len(response.choices) > 0:
            yield format_stream_response(response, history_metadata, apim_request_id)
        else:
            chunk = {
                "choices": [{
                    "delta": {"content": str(response)},
                    "finish_reason": "stop"
                }]
            }
            yield format_stream_response(chunk, history_metadata, apim_request_id)

    return generate(apim_request_id=apim_request_id, history_metadata=history_metadata)


async def complete_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})
    return format_non_streaming_response(response, history_metadata, apim_request_id)


async def conversation_internal(request_body, request_headers):
    try:
        if app_settings.azure_openai.stream and not app_settings.base_settings.use_promptflow:
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()
    return await conversation_internal(request_json, request.headers)


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# CONVERSATION HISTORY API
# ============================================================================

@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No user message found")

        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        if not conversation_id:
            raise Exception("No conversation_id found")

        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        return jsonify({"success": True}), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    if not current_app.cosmos_conversation_client:
        return jsonify({"error": "CosmosDB is not configured"}), 500

    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    if not current_app.cosmos_conversation_client:
        return jsonify({"error": "CosmosDB is not configured"}), 500

    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return jsonify({"error": "Conversation not found"}), 404

    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)
    title = request_json.get("title", None)

    if not conversation_id or not title:
        return jsonify({"error": "conversation_id and title are required"}), 400

    if not current_app.cosmos_conversation_client:
        return jsonify({"error": "CosmosDB is not configured"}), 500

    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return jsonify({"error": "Conversation not found"}), 404

    conversation["title"] = title
    updated_conversation = await current_app.cosmos_conversation_client.upsert_conversation(
        conversation
    )

    return jsonify(updated_conversation), 200


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    if not current_app.cosmos_conversation_client:
        return jsonify({"error": "CosmosDB is not configured"}), 500

    await current_app.cosmos_conversation_client.delete_messages(
        conversation_id, user_id
    )
    await current_app.cosmos_conversation_client.delete_conversation(
        user_id, conversation_id
    )

    return jsonify({"success": True}), 200


async def generate_title(conversation_messages) -> str:
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=messages,
            temperature=1,
            max_tokens=64
        )
        title = response.choices[0].message.content.strip()
        title = title.replace('"', '').replace("'", "").replace(".", "").replace("?", "")
        return title[:50]
    except Exception as e:
        logging.exception("Exception while generating title", e)
        for msg in conversation_messages:
            if msg["role"] == "user":
                return msg["content"][:50]
        return "New Chat"


app = create_app()


if __name__ == "__main__":
    app.run()
