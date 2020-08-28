from sleap.message import PairedSender, PairedReceiver
import time


def run_send():
    from time import sleep

    sender = PairedSender.from_defaults()

    sender.setup()
    success = sender.send_handshake()

    # Make sure handshake was successful
    assert success

    # Send 10 messages
    for i in range(10):
        sender.send_dict(dict(message_id=i))

    sender.close()


def run_receive():
    receiver = PairedReceiver.from_defaults()
    receiver.setup()

    success = receiver.receive_handshake()

    # Make sure handshake was succesful
    assert success

    messages = []

    # Keep checking messages for up to 5 seconds (or until we got last)
    until = time.time() + 5
    while time.time() < until:
        messages.extend(receiver.check_messages(timeout=30, times_to_check=20))
        if messages and messages[-1]["message_id"] == 9:
            break

    # Make sure we got all the messages
    assert len(messages) == 10
    assert messages[-1]["message_id"] == 9

    receiver.close()


def test_send_receive_pair():
    from multiprocessing import Process

    # run "sender" in a separate process
    Process(target=run_send).start()

    # receive messages in the main process
    run_receive()
