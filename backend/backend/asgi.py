import os

# MUST be set before any Django imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

# MUST be called before importing any Django models or consumers
from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()

# Only NOW is the app registry ready — safe to import consumers/models
from channels.routing import ProtocolTypeRouter, URLRouter
from django.urls import path
from trainer.consumers import TrainConsumer

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": URLRouter([
        path("ws/train/<str:model_id>/", TrainConsumer.as_asgi()),
    ]),
})