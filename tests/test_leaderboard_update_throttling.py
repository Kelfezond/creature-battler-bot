import asyncio
import time
import pytest
import creature_battler_bot as cbb

class DummyMessage:
    def __init__(self):
        self.count = 0
    async def edit(self, *, content):
        self.count += 1

class DummyPool:
    async def fetch(self, *args, **kwargs):
        return []
    async def execute(self, *args, **kwargs):
        return None

@pytest.mark.asyncio
async def test_leaderboard_update_respects_cooldown(monkeypatch):
    dummy_msg = DummyMessage()

    async def dummy_db_pool():
        return DummyPool()

    async def dummy_get_leaderboard_channel_id():
        return 1

    async def dummy_get_or_create_leaderboard_message(channel_id):
        return dummy_msg

    async def dummy_get_or_create_pvp_message(channel_id):
        return None

    monkeypatch.setattr(cbb, 'db_pool', dummy_db_pool)
    monkeypatch.setattr(cbb, '_get_leaderboard_channel_id', dummy_get_leaderboard_channel_id)
    monkeypatch.setattr(cbb, '_get_or_create_leaderboard_message', dummy_get_or_create_leaderboard_message)
    monkeypatch.setattr(cbb, '_get_or_create_pvp_leaderboard_message', dummy_get_or_create_pvp_message)
    monkeypatch.setattr(cbb, '_format_leaderboard_lines', lambda rows: '')
    monkeypatch.setattr(cbb, '_format_pvp_leaderboard_lines', lambda rows: '')

    # speed up cooldown for test
    cbb.LEADERBOARD_UPDATE_COOLDOWN = 0.1
    cbb._last_leaderboard_update = 0

    start = time.monotonic()
    await cbb.update_leaderboard_now('first')
    mid = time.monotonic()
    await cbb.update_leaderboard_now('second')
    end = time.monotonic()

    assert dummy_msg.count == 2
    assert end - mid >= 0.1
