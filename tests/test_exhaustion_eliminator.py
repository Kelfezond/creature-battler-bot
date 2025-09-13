import os
import sys
import types
import pytest
from types import SimpleNamespace
import asyncio

# Provide minimal stubs for external dependencies required at import time
os.environ.setdefault("DISCORD_TOKEN", "test")
OSENV_DATABASE_URL = os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("OPENAI_API_KEY", "test")

discord = types.ModuleType("discord")
class Intents:
    def __init__(self):
        self.message_content = False
    @staticmethod
    def default():
        return Intents()
discord.Intents = Intents
class DummyInteraction:
    pass
discord.Interaction = DummyInteraction

class ButtonStyle:
    blurple = gray = green = success = danger = primary = secondary = None
discord.ButtonStyle = ButtonStyle

ui_module = types.ModuleType("ui")
class Modal:
    def __init_subclass__(cls, **kwargs):
        pass
class View:
    pass
class Button:
    pass
class Select:
    pass
class TextInput:
    def __init__(self, *a, **kw):
        pass
def button(*a, **kw):
    def decorator(func):
        return func
    return decorator
def select(*a, **kw):
    def decorator(func):
        return func
    return decorator
ui_module.Modal = Modal
ui_module.View = View
ui_module.Button = Button
ui_module.Select = Select
ui_module.TextInput = TextInput
ui_module.button = button
ui_module.select = select
discord.ui = ui_module
ext = types.ModuleType("ext")
commands = types.ModuleType("commands")
class Bot:
    def __init__(self, *args, **kwargs):
        self.tree = types.SimpleNamespace(
            copy_global_to=lambda *a, **kw: None,
            clear_commands=lambda *a, **kw: None,
            sync=lambda *a, **kw: None,
            command=lambda *a, **kw: (lambda f: f),
            context_menu=lambda *a, **kw: (lambda f: f),
        )
    def event(self, func):
        return func
    def add_view(self, *a, **kw):
        pass
    def add_cog(self, *a, **kw):
        pass
commands.Bot = Bot
class Cog:
    pass
commands.Cog = Cog
tasks = types.ModuleType("tasks")
def loop(*args, **kwargs):
    def decorator(func):
        return func
    return decorator
tasks.loop = loop
ext.commands = commands
ext.tasks = tasks
discord.ext = ext
sys.modules['discord'] = discord
sys.modules['discord.ext'] = ext
sys.modules['discord.ext.commands'] = commands
sys.modules['discord.ext.tasks'] = tasks

asyncpg = types.ModuleType("asyncpg")
async def create_pool(*args, **kwargs):
    class DummyPool:
        pass
    return DummyPool()
asyncpg.create_pool = create_pool
sys.modules['asyncpg'] = asyncpg

openai_mod = types.ModuleType("openai")
class OpenAI:
    def __init__(self, api_key=None):
        pass
openai_mod.OpenAI = OpenAI
sys.modules['openai'] = openai_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import creature_battler_bot as cbb
EXHAUSTION_ELIMINATOR = cbb.EXHAUSTION_ELIMINATOR

class DummyResponse:
    def __init__(self):
        self.message = None
        self.ephemeral = None
        self._done = False
    async def send_message(self, message, ephemeral=False):
        self.message = message
        self.ephemeral = ephemeral
        self._done = True
    async def defer(self, **kwargs):
        if self._done:
            raise AssertionError("defer called after response already done")
        self._done = True
    def is_done(self):
        return self._done

class FakeConn:
    def __init__(self, state):
        self.state = state
    async def fetchrow(self, query, *args):
        if "FROM creatures" in query:
            owner_id, name = args
            for c in self.state["creatures"]:
                if c["owner_id"] == owner_id and c["name"].lower() == str(name).lower():
                    return {"id": c["id"], "name": c["name"]}
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
        if "DELETE FROM battle_caps" in query:
            creature_id, day = args
            self.state["battle_caps"] = [bc for bc in self.state["battle_caps"] if not (bc["creature_id"]==creature_id and bc["day"]==day)]
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

def test_exhaustion_eliminator_restores_battle_without_healing(monkeypatch):
    state = {
        "creatures": [
            {"id": 1, "owner_id": 1, "name": "Alpha", "current_hp": 0},
            {"id": 2, "owner_id": 1, "name": "Beta", "current_hp": 5},
        ],
        "battle_caps": [{"creature_id": 1, "day": "2024-01-01", "count": 2}],
        "trainer_items": {(1, EXHAUSTION_ELIMINATOR): 1},
        "day": "2024-01-01",
    }
    async def fake_db_pool():
        return FakePool(state)
    async def fake_ensure_registered(inter):
        return {"cash": 0}
    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_exhaustion_eliminator(interaction, "Alpha"))
    assert state["battle_caps"][0]["count"] == 1
    assert state["trainer_items"][(1, EXHAUSTION_ELIMINATOR)] == 0
    assert state["creatures"][0]["current_hp"] == 0
    assert state["creatures"][1]["current_hp"] == 5
    assert not any("current_hp" in q for q in state.get("executed", []))
    assert interaction.response.message == "Restored one daily battle for **Alpha**."


def test_exhaustion_eliminator_multiple_uses(monkeypatch):
    state = {
        "creatures": [
            {"id": 1, "owner_id": 1, "name": "Alpha"},
        ],
        "battle_caps": [{"creature_id": 1, "day": "2024-01-01", "count": 2}],
        "trainer_items": {(1, EXHAUSTION_ELIMINATOR): 2},
        "day": "2024-01-01",
    }
    async def fake_db_pool():
        return FakePool(state)
    async def fake_ensure_registered(inter):
        return {"cash": 0}
    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_exhaustion_eliminator(interaction, "Alpha"))
    assert state["battle_caps"][0]["count"] == 1
    assert state["trainer_items"][(1, EXHAUSTION_ELIMINATOR)] == 1
    interaction.response = DummyResponse()
    asyncio.run(cbb._use_exhaustion_eliminator(interaction, "Alpha"))
    assert state["battle_caps"] == []
    assert state["trainer_items"][(1, EXHAUSTION_ELIMINATOR)] == 0


def test_exhaustion_eliminator_deletes_stale_rows(monkeypatch):
    state = {
        "creatures": [
            {"id": 1, "owner_id": 1, "name": "Alpha"},
        ],
        "battle_caps": [{"creature_id": 1, "day": "2024-01-01", "count": 0}],
        "trainer_items": {(1, EXHAUSTION_ELIMINATOR): 1},
        "day": "2024-01-01",
    }
    async def fake_db_pool():
        return FakePool(state)
    async def fake_ensure_registered(inter):
        return {"cash": 0}
    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)
    monkeypatch.setattr(cbb, "ensure_registered", fake_ensure_registered)
    interaction = SimpleNamespace(user=SimpleNamespace(id=1), response=DummyResponse())
    asyncio.run(cbb._use_exhaustion_eliminator(interaction, "Alpha"))
    assert state["battle_caps"] == []
    assert state["trainer_items"][(1, EXHAUSTION_ELIMINATOR)] == 1
