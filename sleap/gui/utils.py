"""Generic module containing utilities used for the GUI."""

import zmq
from typing import Optional


def is_port_free(port: int, zmq_context: Optional[zmq.Context] = None) -> bool:
    """Checks if a port is free."""
    ctx = zmq.Context.instance() if zmq_context is None else zmq_context
    socket = ctx.socket(zmq.REP)
    address = f"tcp://127.0.0.1:{port}"
    try:
        socket.bind(address)
        socket.unbind(address)
        return True
    except zmq.error.ZMQError:
        return False
    finally:
        socket.close()


def select_zmq_port(zmq_context: Optional[zmq.Context] = None) -> int:
    """Select a port that is free to connect within the given context."""
    ctx = zmq.Context.instance() if zmq_context is None else zmq_context
    socket = ctx.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    socket.close()
    return port
