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
                    "stats": json.dumps({"HP": 2, "A": 1, "Ac": 1, "Ag": 1, "Sp": 1, "Df": 1}),
                    "current_hp": 10,
                }
            if "FROM trainers" in query:
                return {"cash": 0}
            return None
        async def fetch(self, query, *args):
            if "augment_name" in query:
                return []
            return []
        async def execute(self, query, *args):
            pass
        def acquire(self):
            return self
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass

    async def fake_db_pool():
        return FakePool()

    async def fake_ensure_registered(inter):
        return True

    async def fake_max_tier(creature_id: int) -> int:
        return 1

    async def fake_can_start(creature_id: int):
        return True, 1

    async def fake_generate_name(rarity):
        return "Oppy"

    def fake_allocate_stats(rarity, extra):
        return {"HP": 1, "A": 0, "Ac": 0, "Ag": 0, "Sp": 0, "Df": 0}

    def fake_stat_block(name, cur, maxhp, stats):
        return f"{name} {cur}/{maxhp}"

    async def fake_finalize(inter, st):
        return {}

    def fake_format_summary(st, summary, trainer_name):
        return "summary"

    async def fake_resolve_name(uid):
        return "Trainer"

    def fake_simulate_round(st):
        st.opp_current_hp = 0

    def fake_rarity_for_tier(tier):
        return "Common"

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    monkeypatch.setattr(cbb, "_max_unlocked_tier", fake_max_tier)
    monkeypatch.setattr(cbb, "_can_start_battle_and_increment", fake_can_start)
    monkeypatch.setattr(cbb, "generate_creature_name", fake_generate_name)
    monkeypatch.setattr(cbb, "allocate_stats", fake_allocate_stats)
    monkeypatch.setattr(cbb, "stat_block", fake_stat_block)
    monkeypatch.setattr(cbb, "finalize_battle", fake_finalize)
    monkeypatch.setattr(cbb, "format_public_battle_summary", fake_format_summary)
    monkeypatch.setattr(cbb, "_resolve_trainer_name_from_db", fake_resolve_name)
    monkeypatch.setattr(cbb, "simulate_round", fake_simulate_round)
    monkeypatch.setattr(cbb, "rarity_for_tier", fake_rarity_for_tier)
    cbb.active_battles.clear()
    cbb.current_battler_id = None


def test_battle_uses_followup_after_initial_response(monkeypatch):
    _setup(monkeypatch)
    messages = []

    class DummyFollowup:
        async def send(self, message, ephemeral=False):
            messages.append((message, ephemeral))

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1, name="Tester"),
        response=DummyResponse(),
        followup=DummyFollowup(),
    )

    # Simulate prior response (dropdown selection step)
    asyncio.run(interaction.response.send_message("choose", ephemeral=True))

    # Should not raise even though response is already done
    asyncio.run(cbb._battle_impl(interaction, 1, 1))

    assert any("Battle Start" in msg[0] for msg in messages)
