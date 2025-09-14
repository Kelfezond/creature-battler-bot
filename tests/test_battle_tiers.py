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

    async def fake_get_wins(creature_id: int, tier: int) -> int:
        return 0

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    monkeypatch.setattr(cbb, "_max_unlocked_tier", fake_max_tier)
    monkeypatch.setattr(cbb, "_get_wins_for_tier", fake_get_wins)
    cbb.active_battles.clear()


def test_cannot_battle_higher_tier(monkeypatch):
    _setup(monkeypatch)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._battle_impl(interaction, 1, 4))
    assert (
        interaction.response.message
        == "Tier 4 is locked for **Alpha**. Current unlock: **1..3**. You need **5 wins at Tier 3** to unlock Tier 4 (current: 0/5)."
    )


def test_leaderboard_records_only_highest_tier_wins(monkeypatch):
    recorded = []

    async def fake_record_result(owner_id, name, won):
        recorded.append((owner_id, name, won))

    class FakePool:
        async def execute(self, *args, **kwargs):
            pass
        def acquire(self):
            return self
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass

    async def fake_db_pool():
        return FakePool()

    async def fake_ensure_record(*args, **kwargs):
        pass

    async def fake_record_win_and_maybe_unlock(*args, **kwargs):
        return 1, False

    def fake_update_leaderboard_now(*args, **kwargs):
        pass

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "_ensure_record", fake_ensure_record)
    monkeypatch.setattr(cbb, "_record_result", fake_record_result)
    monkeypatch.setattr(cbb, "_record_win_and_maybe_unlock", fake_record_win_and_maybe_unlock)
    monkeypatch.setattr(cbb, "update_leaderboard_now", fake_update_leaderboard_now)
    monkeypatch.setattr(cbb.asyncio, "create_task", lambda *args, **kwargs: None)
    monkeypatch.setattr(cbb.random, "choice", lambda seq: seq[0])

    st = cbb.BattleState(
        user_id=1,
        creature_id=1,
        tier=2,
        user_creature={"name": "Alpha", "stats": {"HP": 1}},
        user_current_hp=5,
        user_max_hp=5,
        opp_creature={"name": "Oppy", "stats": {"HP": 1}},
        opp_current_hp=0,
        opp_max_hp=5,
        logs=[],
        highest_eligible_tier=3,
    )
    asyncio.run(cbb.finalize_battle(None, st))
    assert recorded == []

    st_high = cbb.BattleState(
        user_id=1,
        creature_id=1,
        tier=3,
        user_creature={"name": "Alpha", "stats": {"HP": 1}},
        user_current_hp=5,
        user_max_hp=5,
        opp_creature={"name": "Oppy", "stats": {"HP": 1}},
        opp_current_hp=0,
        opp_max_hp=5,
        logs=[],
        highest_eligible_tier=3,
    )
    recorded.clear()
    asyncio.run(cbb.finalize_battle(None, st_high))
    assert recorded == [(1, "Alpha", True)]
