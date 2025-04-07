"""Generic module containing utilities used for the GUI."""

import zmq
import time
from typing import Optional


def is_port_free(port: int, zmq_context: Optional[zmq.Context] = None) -> bool:
    """Checks if a port is free."""
    ctx = zmq.Context.instance() if zmq_context is None else zmq_context
    socket = ctx.socket(zmq.REP)
    address = f"tcp://127.0.0.1:{port}"
    try:
        socket.bind(address)
        socket.unbind(address)
        time.sleep(0.1)
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


def find_free_port(port: int, zmq_context: zmq.Context):
    """Find free port to bind to.

    Args:
        port: The port to start searching from.
        zmq_context: The ZMQ context to use.

    Returns:
        The free port.
    """
    attempts = 0
    max_attempts = 10
    while not is_port_free(port=port, zmq_context=zmq_context):
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not find free port to display training progress after "
                f"{max_attempts} attempts. Please check your network settings "
                "or use the CLI `sleap-train` command."
            )
        port = select_zmq_port(zmq_context=zmq_context)
        attempts += 1

    return port
