import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import chainlit as cl

from src.web.backend_client import BackendClient
from src.web.event_manager import EventManager
from src.web.status_handler import StatusHandler
from src.web.ui_callbacks import UICallbacks

backend_client = BackendClient()
event_manager = EventManager()
status_handler = StatusHandler(event_manager)
ui_callbacks = UICallbacks(backend_client, event_manager, status_handler)

cl.password_auth_callback(ui_callbacks.auth_callback)
cl.action_callback('continue_generation')(ui_callbacks.continue_generation)
cl.action_callback('stop_generation')(ui_callbacks.stop_generation)
cl.on_chat_start(ui_callbacks.start)
cl.on_message(ui_callbacks.main)

if __name__ == '__main__':
    pass
