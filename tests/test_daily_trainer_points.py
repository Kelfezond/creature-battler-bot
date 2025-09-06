import asyncio
from datetime import date, timedelta
import os, sys

# Reuse environment stubs and module import from the existing test module
sys.path.append(os.path.dirname(__file__))
from test_exhaustion_eliminator import cbb


class FakeConn:
    def __init__(self, state):
        self.state = state

    async def fetchval(self, query, *args):
        if "now() AT TIME ZONE" in query:
            return self.state["today"]
        if "SELECT last_tp_grant" in query:
            trainers = sorted(self.state["trainers"], key=lambda t: t["user_id"])
            return trainers[0]["last_tp_grant"]
        return None

    async def execute(self, query, *args):
        today = args[0]
        for trainer in self.state["trainers"]:
            base = 5 + min(max(trainer["facility_level"] - 1, 0), 5)
            last = trainer.get("last_tp_grant")
            if last is None:
                last = today - timedelta(days=1)
            days = max(0, (today - last).days)
            trainer["trainer_points"] += base * days
            trainer["last_tp_grant"] = today

    def acquire(self):
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


def test_new_trainer_receives_points_next_day(monkeypatch):
    today = date(2024, 1, 2)
    state = {
        "today": today,
        "trainers": [
            {
                "user_id": 1,
                "trainer_points": 5,
                "facility_level": 1,
                "last_tp_grant": None,
            }
        ],
    }

    async def fake_db_pool():
        return FakePool(state)

    monkeypatch.setattr(cbb, "db_pool", fake_db_pool)

    asyncio.run(cbb.distribute_points())

    trainer = state["trainers"][0]
    assert trainer["trainer_points"] == 10
    assert trainer["last_tp_grant"] == today
