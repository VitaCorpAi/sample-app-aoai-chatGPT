import os
import json
import logging
import uuid
from quart import Blueprint, Quart, jsonify, make_response, request, send_from_directory, render_template

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

# === НАСТРОЙКИ ===
PROJECT_ENDPOINT = os.environ.get("AZURE_AI_PROJECT_ENDPOINT", "https://openai-chatbot-vita.services.ai.azure.com/api/projects/proj-vita")
AGENT_NAME = os.environ.get("AZURE_AI_AGENT_NAME", "test-web-search")
AGENT_VERSION = os.environ.get("AZURE_AI_AGENT_VERSION", "1")

def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    return app

@bp.route("/")
async def index():
    return await render_template("index.html")

@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")

@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)

async def call_agent(user_message: str, conversation_history: list = None) -> str:
    """
    Вызывает агента test-web-search через Azure AI Projects SDK.
    Агент автоматически использует Bing Search для актуальных вопросов.
    """
    try:
        # Подключаемся к проекту
        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=DefaultAzureCredential(),
        )
        
        # Получаем OpenAI клиент, привязанный к агенту
        # Это ключевой момент — именно так SDK подключается к агенту
        openai_client = project_client.get_openai_client()
        
        # Формируем историю сообщений для контекста
        input_messages = []
        if conversation_history:
            # Ограничиваем историю последними 10 сообщениями для контекста
            for msg in conversation_history[-10:]:
                input_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # Добавляем текущее сообщение пользователя
        input_messages.append({"role": "user", "content": user_message})
        
        # Отправляем запрос агенту через Responses API
        response = openai_client.responses.create(
            input=input_messages,
            extra_body={
                "agent_reference": {
                    "name": AGENT_NAME,
                    "version": AGENT_VERSION,
                    "type": "agent_reference"
                }
            }
        )
        
        return response.output_text
        
    except Exception as e:
        logging.exception(f"Error calling agent: {e}")
        return f"Извините, произошла ошибка при обработке запроса: {str(e)}"

@bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    
    data = await request.get_json()
    messages = data.get("messages", [])
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    # Извлекаем последнее сообщение пользователя
    user_message = messages[-1].get("content", "")
    
    # Извлекаем историю для контекста (без последнего сообщения, его добавим отдельно)
    conversation_history = messages[:-1] if len(messages) > 1 else None
    
    # Вызываем агента
    response_text = await call_agent(user_message, conversation_history)
    
    # Формируем ответ в формате, ожидаемом фронтендом
    response_data = {
        "id": str(uuid.uuid4()),
        "model": AGENT_NAME,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": None
    }
    
    return jsonify(response_data)

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50505, debug=True)
