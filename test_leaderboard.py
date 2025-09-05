import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("DISCORD_TOKEN", "x")
os.environ.setdefault("DATABASE_URL", "postgres://localhost")
import creature_battler_bot as cbb

class FakePool:
    def __init__(self):
        self.ovr = None
    async def execute(self, query, *params):
        # Capture OVR from _ensure_record insert
        if "INSERT INTO creature_records" in query:
            self.ovr = params[3]
    async def fetch(self, query):
        if "FROM pvp_records" in query:
            return []
        return [
            {
                "name": "Testmon",
                "wins": 0,
                "losses": 0,
                "is_dead": False,
                "trainer_name": "Tester",
                "max_glyph_tier": 0,
                "ovr": self.ovr,
            }
        ]

class LeaderboardTest(unittest.IsolatedAsyncioTestCase):
    async def test_leaderboard_updates_ovr(self):
        pool = FakePool()
        message = SimpleNamespace(edit=AsyncMock())
        with patch("creature_battler_bot.db_pool", AsyncMock(return_value=pool)), \
             patch("creature_battler_bot._get_leaderboard_channel_id", AsyncMock(return_value=123)), \
             patch("creature_battler_bot._get_or_create_leaderboard_message", AsyncMock(return_value=message)), \
             patch("creature_battler_bot._get_or_create_pvp_leaderboard_message", AsyncMock(return_value=None)):
            await cbb._ensure_record(1, 1, "Testmon", ovr=50)
            await cbb.update_leaderboard_now(reason="test")
        message.edit.assert_called_once()
        content = message.edit.call_args.kwargs["content"]
        self.assertIn("50", content)

if __name__ == "__main__":
    unittest.main()
