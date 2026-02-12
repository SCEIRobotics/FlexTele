__all__ = ["PicoServer", "PicoClient"]

def __getattr__(name):
    if name == "PicoServer":
        from .pico_server import PicoServer
        return PicoServer
    if name == "PicoClient":
        from .pico_client import PicoClient
        return PicoClient
    raise AttributeError(f"module 'pico' has no attribute {name}")