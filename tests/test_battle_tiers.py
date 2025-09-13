import asyncio
import json
from types import SimpleNamespace

from test_exhaustion_eliminator import cbb, DummyResponse


def _setup(monkeypatch):
    class FakePool:
        async def fetchrow(self, query, *args):
            if "FROM creatures" in query:
                owner_id, creature_id = args
                return {
                    "id": creature_id,
                    "name": "Alpha",
                    "stats": json.dumps({"HP": 10}),
                    "current_hp": 50,
                }
            return None

    async def fake_db_pool():
        return FakePool()

    async def fake_ensure_registered(inter):
        return True

    async def fake_max_tier(creature_id: int) -> int:
        return 3

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    monkeypatch.setattr(cbb, "_max_unlocked_tier", fake_max_tier)
    cbb.active_battles.clear()


def test_cannot_battle_lower_tier(monkeypatch):
    _setup(monkeypatch)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._battle_impl(interaction, 1, 2))
    assert (
        interaction.response.message
        == "Alpha must battle at Tier 3; lower tiers are no longer available."
    )
