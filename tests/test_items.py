import asyncio
import json
from types import SimpleNamespace
import pytest

# Reuse environment stubs and module import from the existing test module
from tests.test_exhaustion_eliminator import (
    cbb,
    DummyResponse,
    EXHAUSTION_ELIMINATOR,
)
from creature_battler_bot import (
    SMALL_HEALING_INJECTOR,
    LARGE_HEALING_INJECTOR,
    FULL_HEALING_INJECTOR,
)


class FakeConn:
    def __init__(self, state):
        self.state = state

    async def fetchrow(self, query, *args):
        if "FROM creatures" in query:
            owner_id, name = args
            for c in self.state["creatures"]:
                if c["owner_id"] == owner_id and c["name"].lower() == str(name).lower():
                    data = {"id": c["id"], "name": c["name"]}
                    if "stats" in query:
                        data["stats"] = c["stats"]
                        data["current_hp"] = c["current_hp"]
                    return data
            return None
        if "FROM battle_caps" in query:
            creature_id, day = args
            for bc in self.state["battle_caps"]:
                if bc["creature_id"] == creature_id and bc["day"] == day:
                    return {"count": bc["count"]}
            return None

    async def fetchval(self, query, *args):
        if "trainer_items" in query:
            user_id, item_name = args
            return self.state["trainer_items"].get((user_id, item_name), 0)
        if "now() AT TIME ZONE" in query:
            return self.state["day"]
        return None

    async def execute(self, query, *args):
        self.state.setdefault("executed", []).append(query)
        if "UPDATE creatures SET current_hp" in query:
            new_hp, creature_id = args
            for c in self.state["creatures"]:
                if c["id"] == creature_id:
                    c["current_hp"] = new_hp
                    break
        elif "DELETE FROM battle_caps" in query:
            creature_id, day = args
            self.state["battle_caps"] = [
                bc for bc in self.state["battle_caps"]
                if not (bc["creature_id"] == creature_id and bc["day"] == day)
            ]
        elif "UPDATE battle_caps SET count" in query:
            creature_id, day, new_count = args
            for bc in self.state["battle_caps"]:
                if bc["creature_id"] == creature_id and bc["day"] == day:
                    bc["count"] = new_count
                    break
        elif "UPDATE trainer_items SET quantity=quantity-1" in query:
            user_id, item_name = args
            self.state["trainer_items"][(user_id, item_name)] -= 1

    def transaction(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


class FakePool:
    def __init__(self, state):
        self.conn = FakeConn(state)

    def acquire(self):
        return self.conn


def _setup(monkeypatch, state):
    async def fake_db_pool():
        return FakePool(state)

    async def fake_ensure_registered(inter):
        return {"cash": 0}

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)


def test_small_healing_injector(monkeypatch):
    state = {
        "creatures": [
            {
                "id": 1,
                "owner_id": 1,
                "name": "Alpha",
                "stats": json.dumps({"HP": 10}),
                "current_hp": 20,
            }
        ],
        "trainer_items": {(1, SMALL_HEALING_INJECTOR): 1},
    }
    _setup(monkeypatch, state)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_small_healing_injector(interaction, "Alpha"))
    assert state["creatures"][0]["current_hp"] == 32
    assert state["trainer_items"][(1, SMALL_HEALING_INJECTOR)] == 0
    assert interaction.response.message == "Healed **Alpha** for 12 HP."


def test_large_healing_injector(monkeypatch):
    state = {
        "creatures": [
            {
                "id": 1,
                "owner_id": 1,
                "name": "Alpha",
                "stats": json.dumps({"HP": 10}),
                "current_hp": 10,
            }
        ],
        "trainer_items": {(1, LARGE_HEALING_INJECTOR): 1},
    }
    _setup(monkeypatch, state)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_large_healing_injector(interaction, "Alpha"))
    assert state["creatures"][0]["current_hp"] == 35
    assert state["trainer_items"][(1, LARGE_HEALING_INJECTOR)] == 0
    assert interaction.response.message == "Healed **Alpha** for 25 HP."


def test_full_healing_injector(monkeypatch):
    state = {
        "creatures": [
            {
                "id": 1,
                "owner_id": 1,
                "name": "Alpha",
                "stats": json.dumps({"HP": 10}),
                "current_hp": 30,
            }
        ],
        "trainer_items": {(1, FULL_HEALING_INJECTOR): 1},
    }
    _setup(monkeypatch, state)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_full_healing_injector(interaction, "Alpha"))
    assert state["creatures"][0]["current_hp"] == 50
    assert state["trainer_items"][(1, FULL_HEALING_INJECTOR)] == 0
    assert interaction.response.message == "Healed **Alpha** for 20 HP."


def test_exhaustion_eliminator_does_not_modify_hp(monkeypatch):
    state = {
        "creatures": [
            {"id": 1, "owner_id": 1, "name": "Alpha", "current_hp": 10},
            {"id": 2, "owner_id": 1, "name": "Beta", "current_hp": 20},
        ],
        "battle_caps": [{"creature_id": 1, "day": "2024-01-01", "count": 2}],
        "trainer_items": {(1, EXHAUSTION_ELIMINATOR): 1},
        "day": "2024-01-01",
    }
    _setup(monkeypatch, state)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_exhaustion_eliminator(interaction, "Alpha"))
    assert state["battle_caps"][0]["count"] == 1
    assert state["trainer_items"][(1, EXHAUSTION_ELIMINATOR)] == 0
    assert state["creatures"][0]["current_hp"] == 10
    assert state["creatures"][1]["current_hp"] == 20
    assert not any("current_hp" in q for q in state.get("executed", []))
    assert (
        interaction.response.message
        == "Restored one daily battle for **Alpha**."
    )
