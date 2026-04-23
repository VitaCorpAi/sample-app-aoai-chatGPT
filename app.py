import os
import logging
from quart import Blueprint, Quart, jsonify, request

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

PROJECT_ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
AGENT_NAME = os.environ.get("AZURE_AI_AGENT_NAME")  # "Agent649"

def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    return app

async def call_agent(user_message: str) -> str:
    """Вызывает агента Agent649 с Bing Search через SDK"""
    try:
        # Подключаемся к проекту через Project Endpoint
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=PROJECT_ENDPOINT
        )
        
        # Получаем OpenAI клиент, привязанный к агенту
        # Это стандартный метод из документации Microsoft Foundry [citation:4]
        openai_client = project_client.get_openai_client(agent_name=AGENT_NAME)
        
        # Отправляем запрос через Responses API
        response = openai_client.responses.create(
            input=user_message,
        )
        
        return response.output_text
        
    except Exception as e:
        logging.exception(f"Error calling agent: {e}")
        return f"Ошибка: {str(e)}"

@bp.route("/conversation", methods=["POST"])
async def conversation():
    data = await request.get_json()
    messages = data.get("messages", [])
    user_message = messages[-1].get("content", "") if messages else ""
    
    response_text = await call_agent(user_message)
    
    return jsonify({
        "choices": [{
            "message": {"role": "assistant", "content": response_text}
        }]
    })

app = create_app()
