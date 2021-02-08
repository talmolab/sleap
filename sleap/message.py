"""
Module with classes for sending and receiving messages between processes.

These use ZMQ pub/sub sockets.

Most of the time you'll want the PairedSender and PairedReceiver.
These support a "handshake" to confirm connection. Without an initial
handshake there's a good chance early messages will be dropped.

Each message is either dictionary or dictionary + numpy ndarray.
"""
import attr
import jsonpickle
import numpy as np
import time
import zmq

from typing import Any, Callable, List, Optional, Text


@attr.s(auto_attribs=True)
class BaseMessageParticipant:
    """Base class for simple Sender and Receiver."""

    address: Text = "tcp://127.0.0.1:9001"
    context: Optional[zmq.Context] = None
    _socket: Optional[zmq.Socket] = None

    def __attrs_post_init__(self):
        if self.context is None:
            self._owns_context = True
            self.context = zmq.Context()
        else:
            self._owns_context = False

    def __del__(self):
        if self._owns_context and self.context is not None:
            self.context.term()


@attr.s(auto_attribs=True)
class Receiver(BaseMessageParticipant):
    """Receives messages from corresponding Sender."""

    _message_queue: List[Any] = attr.ib(factory=list)

    def setup(self):
        self._socket = self.context.socket(zmq.SUB)
        self._socket.subscribe("")
        self._socket.bind(self.address)

    def __del__(self):
        if self._socket is not None:
            self._socket.unbind(self._socket.LAST_ENDPOINT)
            self._socket.close()
            self._socket = None

    def push_back_message(self, message):
        """Act like we didn't receive this message yet."""
        self._message_queue.append(message)

    def _recv(self, flags=0, copy=True, track=False):
        json_message = self._socket.recv_json(flags=flags)

        if "dtype" in json_message and "shape" in json_message:
            msg = self._socket.recv(flags=flags, copy=copy, track=track)
            buf = memoryview(msg)
            A = np.frombuffer(buf, dtype=json_message["dtype"]).reshape(
                json_message["shape"]
            )
            json_message["ndarray"] = A

        return json_message

    def check_message(self, timeout: int = 10, fresh: bool = False) -> Any:
        """Attempt to receive a single message."""
        if self._message_queue and not fresh:
            return self._message_queue.pop(0)

        if self._socket is None:
            self.setup()

        if self._socket and self._socket.poll(timeout, zmq.POLLIN):
            return self._recv()
        else:
            return None

    def check_messages(self, timeout: int = 10, times_to_check: int = 10) -> List[dict]:
        """
        Attempt to receive multiple messages.

        This method allows us to keep up with the messages by getting
        multiple messages that have been sent since the last check.
        It keeps checking until limit is reached *or* we check without
        getting any messages back.
        """
        messages = []

        # keep looping until we don't receive a message or have checked enough times
        while True:
            this_message = self.check_message(timeout)

            # if we didn't get a message, we're done checking
            if this_message is None:
                return messages

            # we got a message so add it to list
            messages.append(this_message)

            # if we've checked enough times, we're done checking
            if times_to_check <= 0:
                return messages

            # count down the number of times to check for messages
            times_to_check -= 1


@attr.s(auto_attribs=True)
class Sender(BaseMessageParticipant):
    """Publishes messages to corresponding Receiver."""

    def setup(self):
        self._socket = self.context.socket(zmq.PUB)
        self._socket.connect(self.address)

    def __del__(self):
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.close()
        super().__del__()

    def send_dict(self, data: dict):
        """Sends dictionary."""
        if self._socket is None:
            self.setup()
        self._socket.send_json(data)

    def send_array(
        self, header_data: dict, A: np.ndarray, flags=0, copy=True, track=False
    ):
        """Sends dictionary + numpy ndarray."""
        if self._socket is None:
            self.setup()

        header_data["dtype"] = str(A.dtype)
        header_data["shape"] = A.shape

        self._socket.send_json(header_data, flags | zmq.SNDMORE)
        return self._socket.send(A, flags, copy=copy, track=track)


@attr.s(auto_attribs=True)
class PairedMessageParticipant:
    sender_address: Text
    receiver_address: Text
    context: Optional[zmq.Context] = None

    @classmethod
    def from_tcp_ports(cls, send_port, rec_port):
        sender_address = f"tcp://127.0.0.1:{send_port}"
        receiver_address = f"tcp://127.0.0.1:{rec_port}"

        return cls(sender_address=sender_address, receiver_address=receiver_address)

    def setup(self):
        self._sender = Sender(address=self.sender_address, context=self.context)
        self._receiver = Receiver(address=self.receiver_address, context=self.context)
        self._sender.setup()
        self._receiver.setup()

    def close(self):
        if hasattr(self, "_sender"):
            del self._sender
        if hasattr(self, "_receiver"):
            del self._receiver


@attr.s(auto_attribs=True)
class PairedSender(PairedMessageParticipant):
    connected: bool = False

    @classmethod
    def from_defaults(cls):
        return cls.from_tcp_ports(9001, 9002)

    def send_handshake(self, timeout_sec=30):
        """Send handshake until we get reply."""
        wait_till = time.time() + timeout_sec
        while time.time() < wait_till:
            self._sender.send_dict(dict(type="handshake request"))
            reply = self._receiver.check_message()
            if self._is_handshake_reply(reply):
                return True
            else:
                # currently we drop replies until handshake is acknowledged
                pass
            time.sleep(0.1)
        return False

    def _is_handshake_reply(self, message: Any) -> bool:
        if message:
            return message.get("type", "") == "handshake reply"
        return False

    def send_dict(self, *args, **kwargs):
        self._sender.send_dict(*args, **kwargs)

    def send_array(self, *args, **kwargs):
        self._sender.send_array(*args, **kwargs)


@attr.s(auto_attribs=True)
class PairedReceiver(PairedMessageParticipant):
    connected: bool = False

    @classmethod
    def from_defaults(cls):
        return cls.from_tcp_ports(9002, 9001)

    def receive_handshake(self, timeout_sec=30):
        """Waits to receive and acknowledge handshake message."""
        wait_till = time.time() + timeout_sec
        while time.time() < wait_till and not self.connected:
            message = self._receiver.check_message(fresh=True)

            if message is None:
                continue
            if self._is_handshake(message):
                self._respond_to_handshake()
                return True
            else:
                self._receiver.push_back_message(message)
                return True
        return False

    def _respond_to_handshake(self):
        self._sender.send_dict(dict(type="handshake reply"))
        self.connected = True

    def _is_handshake(self, message: Any):
        if message:
            return message.get("type", "") == "handshake request"
        return False

    def check_messages(self, ack_handshakes: bool = True, *args, **kwargs):
        """
        Checks for messages.

        Args:
            ack_handshakes: If True, then any handshake messages are
                acknowledged and aren't included in return results

        Results:
            List of messages, possibly excluding any handshake requests.
        """
        messages = self._receiver.check_messages(*args, **kwargs)

        if ack_handshakes:
            non_handshakes = [m for m in messages if not self._is_handshake(m)]
            if len(non_handshakes) < len(messages):
                self._respond_to_handshake()
                messages = non_handshakes

        return messages
