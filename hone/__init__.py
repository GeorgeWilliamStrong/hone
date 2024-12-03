from .engine import ChatInstill


def get_instill_engine(model_string: str = "gpt-4o", **kwargs):
    """Helper function to create an Instill engine."""
    return ChatInstill(model_string=model_string, **kwargs)
