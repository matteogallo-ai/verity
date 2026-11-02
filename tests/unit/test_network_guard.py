"""The fast lane cannot reach the network (see tests/conftest.py)."""

from __future__ import annotations

import socket

import pytest
import pytest_socket


def test_fast_lane_test_cannot_open_an_inet_socket() -> None:
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


@pytest.mark.slow
def test_slow_lane_test_may_open_an_inet_socket() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()
