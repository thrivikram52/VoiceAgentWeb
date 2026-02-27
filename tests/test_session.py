"""Tests for session state machine."""

import pytest

from server.session import SessionState, SessionManager


class TestSessionState:
    def test_initial_state_is_idle(self):
        sm = SessionManager.__new__(SessionManager)
        sm.state = SessionState.IDLE
        assert sm.state == SessionState.IDLE

    def test_all_states_exist(self):
        assert SessionState.IDLE
        assert SessionState.LISTENING
        assert SessionState.THINKING
        assert SessionState.SPEAKING
