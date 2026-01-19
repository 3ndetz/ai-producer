# AI Producer - MVP

Агентная система для итеративной генерации изображений с использованием LLM и VLM. Система автоматически улучшает качество изображений через цикл: Планировщик -> Генератор -> Критик.

## Быстрый запуск

### 1. Установка зависимостей

```bash
# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройка переменных окружения

```bash
cp .env.example .env
```

Заполните `.env` файл:

```env
# API ключи
GOOGLE_API_KEY=ваш_ключ_google_ai_studio
MISTRAL_API_KEY=ваш_ключ_mistral  # или OPENROUTER_API_KEY
```

Получить ключи:
- Google AI Studio: https://aistudio.google.com/
- OpenRouter: https://openrouter.ai/
- Mistral: https://mistral.ai/

### 3. Запуск Docker (для БД и MinIO)

```bash
docker-compose up -d
```

### 4. Запуск программы

#### Вариант 1: Напрямую

```bash
# Запустите backend (FastAPI)
uvicorn src.backend.server:app --host 0.0.0.0 --port 8001

# В новом терминале запустите frontend (Chainlit)
chainlit run src/web/app.py --host 0.0.0.0 --port 8008
```

Откройте браузер: http://localhost:8008

#### Вариант 2: Через main.py

```bash
# Запуск web-интерфейса 
python main.py --mode web

# Запуск backend
python main.py --mode backend
```


## Доступы и порты

- **Chainlit (веб-интерфейс)**: http://localhost:8008
- **FastAPI Backend**: http://localhost:8001
- **PostgreSQL**: localhost:5432
  - База: `chainlit_db`
  - Пользователь: `chainlit_user`
  - Пароль: `chainlit_pass`
- **MinIO (S3 storage)**: http://localhost:9001 (веб-консоль), API на 9000
  - Пользователь: `minioadmin`
  - Пароль: `supersecretpassword`
  - Бакет: `attachments`
- **pgAdmin**: http://localhost:8080
  - Email: `admin@admin.com`
  - Пароль: `admin`

## Backend API (FastAPI)

FastAPI сервер предоставляет REST API для управления генерацией изображений:

- `POST /workflow/start` - Запуск новой сессии генерации
- `POST /workflow/{thread_id}/continue` - Продолжение итерации с фидбеком
- `POST /workflow/{thread_id}/stop` - Остановка сессии
- `GET /workflow/{thread_id}/state` - Текущее состояние генерации
- `GET /workflow/{thread_id}/events` - Server-Sent Events для стриминга статусов
- `GET /workflow/{thread_id}/history` - История событий сессии
- `GET /workflow/threads` - Список всех thread_id
- `WS /workflow/{thread_id}/ws` - WebSocket для событий
- `GET /health` - Проверка здоровья сервиса
