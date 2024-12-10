from .engine import ChatInstill
from .string_function import GeneralStringFunction


def get_instill_engine(model_string: str = "gpt-3.5-turbo", **kwargs):
    """Helper function to create an Instill engine."""
    return ChatInstill(model_string=model_string, **kwargs)

__all__ = ['get_instill_engine', 'GeneralStringFunction']
