from __future__ import annotations

async def _max_glyph_for_trainer(user_id: int) -> int:
    """Return the highest glyph tier unlocked by any of the user's creatures."""
    pool = await db_pool()
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT COALESCE(MAX(cp.tier), 0)
            FROM creature_progress cp
            JOIN creatures c ON c.id = cp.creature_id
            WHERE c.owner_id = $1 AND cp.glyph_unlocked = TRUE
            """, user_id
        )
        try:
            return int(val or 0)
        except Exception:
            return 0

import asyncio, json, logging, math, os, random, time, re

# Global battle lock
active_battle_user_id = None
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import discord
from discord.ext import commands, tasks
from openai import OpenAI
from datetime import datetime, timedelta, timezone, time as dtime
from zoneinfo import ZoneInfo

# ─── Basic config & logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN     = os.getenv("DISCORD_TOKEN")
DB_URL    = os.getenv("DATABASE_URL")
GUILD_ID  = os.getenv("GUILD_ID") or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5-mini")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1400"))
OPENAI_DEBUG = os.getenv("OPENAI_DEBUG", "0") == "1"

logger.info("Using TEXT_MODEL=%s, IMAGE_MODEL=%s", TEXT_MODEL, IMAGE_MODEL)

# Optional: channel where the live leaderboard is posted/updated.
LEADERBOARD_CHANNEL_ID_ENV = os.getenv("LEADERBOARD_CHANNEL_ID")
# Optional: channel where the live creature shop is posted/updated.
SHOP_CHANNEL_ID_ENV = os.getenv("SHOP_CHANNEL_ID")
# Optional: channel where interactive controls are posted.
CONTROLS_CHANNEL_ID_ENV = os.getenv("CONTROLS_CHANNEL_ID")

# Admin allow-list for privileged commands (e.g., /cashadd, /setleaderboardchannel)
def _parse_admin_ids(raw: Optional[str]) -> set[int]:
    ids: set[int] = set()
    if not raw:
        return ids
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            logger.warning("Ignoring non-integer ADMIN_USER_IDS entry: %r", part)
    return ids

ADMIN_USER_IDS: set[int] = _parse_admin_ids(os.getenv("ADMIN_USER_IDS"))

for env_name, env_val in {
    "DISCORD_TOKEN": TOKEN,
    "DATABASE_URL": DB_URL,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}.items():
    if not env_val:
        raise RuntimeError(f"Missing environment variable: {env_name}")

if ADMIN_USER_IDS:
    logger.info("Admin allow-list loaded: %s", ", ".join(map(str, ADMIN_USER_IDS)))
else:
    logger.warning("ADMIN_USER_IDS is not set; privileged commands will be denied for all users.")

# ─── Discord client ──────────────────────────────────────────
intents = discord.Intents.default()
# Keep message content if you already had it enabled for your app
intents.message_content = True
# IMPORTANT: do NOT enable members intent; we resolve trainer names via REST/cache
bot = commands.Bot(command_prefix="/", intents=intents)

# ─── Database helpers ────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trainers (
  user_id BIGINT PRIMARY KEY,
  joined_at TIMESTAMPTZ DEFAULT now(),
  cash BIGINT DEFAULT 0,
  trainer_points BIGINT DEFAULT 0,
  facility_level INT DEFAULT 1
);

ALTER TABLE trainers
  ADD COLUMN IF NOT EXISTS last_tp_grant DATE;

CREATE TABLE IF NOT EXISTS creatures (
  id SERIAL PRIMARY KEY,
  owner_id BIGINT NOT NULL,
  name TEXT,
  rarity TEXT,
  descriptors TEXT[],
  stats JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE creatures
  ADD COLUMN IF NOT EXISTS current_hp BIGINT;


ALTER TABLE creatures
  ADD COLUMN IF NOT EXISTS personality JSONB;

-- store display names for leaderboards
ALTER TABLE trainers
  ADD COLUMN IF NOT EXISTS display_name TEXT;

-- Per-creature/day battle cap (resets at midnight Europe/London)
CREATE TABLE IF NOT EXISTS battle_caps (
  creature_id INT NOT NULL REFERENCES creatures(id) ON DELETE CASCADE,
  day DATE NOT NULL,
  count INT NOT NULL DEFAULT 0,
  PRIMARY KEY (creature_id, day)
);

-- Per-creature per-tier progress (wins & glyph unlock)
CREATE TABLE IF NOT EXISTS creature_progress (
  creature_id INT NOT NULL REFERENCES creatures(id) ON DELETE CASCADE,
  tier INT NOT NULL,
  wins INT NOT NULL DEFAULT 0,
  glyph_unlocked BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (creature_id, tier)
);

-- Lifetime win/loss records that persist even after a creature dies or is sold
CREATE TABLE IF NOT EXISTS creature_records (
  creature_id INT,
  owner_id BIGINT NOT NULL,
  name TEXT NOT NULL,
  wins INT NOT NULL DEFAULT 0,
  losses INT NOT NULL DEFAULT 0,
  is_dead BOOLEAN NOT NULL DEFAULT FALSE,
  died_at TIMESTAMPTZ,
  PRIMARY KEY (owner_id, name)
);

CREATE INDEX IF NOT EXISTS cr_rank_idx ON creature_records (wins DESC, losses ASC, name ASC);

-- Persisted highest glyph tier (even if creature dies)
ALTER TABLE creature_records
  ADD COLUMN IF NOT EXISTS highest_glyph_tier INT NOT NULL DEFAULT 0;

-- Store overall rating (sum of base stats)
ALTER TABLE creature_records
  ADD COLUMN IF NOT EXISTS ovr INT NOT NULL DEFAULT 0;

-- Aggregate trainer PvP records
CREATE TABLE IF NOT EXISTS pvp_records (
  user_id BIGINT PRIMARY KEY,
  wins INT NOT NULL DEFAULT 0,
  losses INT NOT NULL DEFAULT 0,
  last_battle_at TIMESTAMPTZ
);
ALTER TABLE pvp_records
  ADD COLUMN IF NOT EXISTS last_battle_at TIMESTAMPTZ;

-- Store the message we keep editing for the live leaderboard
CREATE TABLE IF NOT EXISTS leaderboard_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT,
  pvp_message_id BIGINT
);
ALTER TABLE leaderboard_messages
  ADD COLUMN IF NOT EXISTS pvp_message_id BIGINT;

-- Store creature shop listings
CREATE TABLE IF NOT EXISTS creature_shop (
  creature_id INT PRIMARY KEY REFERENCES creatures(id) ON DELETE CASCADE,
  price BIGINT NOT NULL,
  listed_at TIMESTAMPTZ DEFAULT now()
);

-- Store the message we keep editing for the live creature shop
CREATE TABLE IF NOT EXISTS shop_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT
);

-- Store the message we keep posting for the controls
CREATE TABLE IF NOT EXISTS controls_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT
);

-- Encyclopedia target channel
CREATE TABLE IF NOT EXISTS encyclopedia_channel (
  channel_id BIGINT PRIMARY KEY
);
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game constants ──────────────────────────────────────────
MAX_CREATURES = 5
DAILY_BATTLE_CAP = 2  # <— used for display of remaining battles

SELL_PRICES: Dict[str, int] = {
    "Common": 1_000,
    "Uncommon": 2_000,
    "Rare": 10_000,
    "Epic": 20_000,
    "Legendary": 50_000,
}

def spawn_rarity() -> str:
    r = random.random() * 100.0
    if r < 75.0: return "Common"
    elif r < 88.0: return "Uncommon"
    elif r < 95.0: return "Rare"
    elif r < 99.5: return "Epic"
    else: return "Legendary"

TIER_RARITY_WEIGHTS: Dict[int, Tuple[List[str], List[int]]] = {
    1: (["Common"], [100]),
    2: (["Common"], [100]),
    3: (["Common", "Uncommon"], [75, 25]),
    4: (["Common", "Uncommon"], [75, 25]),
    5: (["Common", "Uncommon", "Rare"], [50, 33, 16]),
    6: (["Common", "Uncommon", "Rare"], [50, 33, 16]),
    7: (["Common", "Uncommon", "Rare", "Epic"], [40, 30, 20, 10]),
    8: (["Common", "Uncommon", "Rare", "Epic"], [40, 30, 20, 10]),
    9: (["Common", "Uncommon", "Rare", "Epic", "Legendary"], [33, 26, 20, 13, 6]),
}

MAX_FACILITY_LEVEL = 6
FACILITY_LEVELS: Dict[int, Dict[str, Any]] = {
    1: {"name": "Basic Training Yard", "bonus": 0, "cost": None, "desc": "A patch of land with scattered targets, sand pits, and makeshift climbing posts. Rough, simple, and functional."},
    2: {"name": "Reinforced Combat Pit", "bonus": 1, "cost": 18_000, "desc": "Expanded grounds with adjustable barriers, weighted obstacles, sky hoops, and water trenches. Built to be tough and versatile."},
    3: {"name": "Kinetic Optimization Center", "bonus": 2, "cost": 55_000, "desc": "Modular platforms with reactive surfaces, telescoping tracks, and pressure pads. The environment adapts to suit each training style."},
    4: {"name": "Neuro-Combat Simulator", "bonus": 3, "cost": 130_000, "desc": "Holographic arenas simulate dynamic opponents and shifting terrain. Training is personalized and reactive in real time."},
    5: {"name": "BioSync Reactor Chamber", "bonus": 4, "cost": 275_000, "desc": "A synchronized chamber tuned to physical and mental rhythms. Terrain and resistance fields shift unpredictably to enhance reflex development."},
    6: {"name": "SynapseForge Hyperlab", "bonus": 5, "cost": 500_000, "desc": "A high-tech fusion of neural feedback, virtual training microcosms, and time-compressed simulations. Mastery is forged at the speed of thought."},
}
def facility_bonus(level: int) -> int:
    level = max(1, min(MAX_FACILITY_LEVEL, level))
    return FACILITY_LEVELS[level]["bonus"]
def daily_trainer_points_for(level: int) -> int:
    return 5 + facility_bonus(level)

POINT_POOLS = {
    "Common": (25, 50), "Uncommon": (50, 100), "Rare": (100, 200),
    "Epic": (200, 400), "Legendary": (400, 800),
}
TIER_EXTRAS = {
    1: (0, 10),  2: (10, 30), 3: (30, 60), 4: (60, 100),
    5: (100, 140), 6: (140, 180), 7: (180, 220),
    8: (220, 260), 9: (260, 300)
}
PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]
# ─── Personalities (stat-focused training bonus) ─────────────
# Each personality boosts training for its listed stat(s) by 2× output (same TP cost).
# We store: {"name": str, "stats": [..]}. Selection uses the given weights (sum ~100%).
PERSONALITY_TYPES = [
    # Singles (7.58% each)
    {"name": "Goliath", "stats": ["HP"], "weight": 7.58},
    {"name": "Stoic", "stats": ["AR"], "weight": 7.58},
    {"name": "Aggressive", "stats": ["PATK"], "weight": 7.58},
    {"name": "Pensive", "stats": ["SATK"], "weight": 7.58},
    {"name": "Hyperactive", "stats": ["SPD"], "weight": 7.58},

    # Duos (3.79% each)
    {"name": "Immovable Goliath", "stats": ["HP","AR"], "weight": 3.79},
    {"name": "Titan Mauler", "stats": ["HP","PATK"], "weight": 3.79},
    {"name": "Contemplative Colossus", "stats": ["HP","SATK"], "weight": 3.79},
    {"name": "Charging Colossus", "stats": ["HP","SPD"], "weight": 3.79},
    {"name": "Stoic Bruiser", "stats": ["AR","PATK"], "weight": 3.79},
    {"name": "Stoic Sage", "stats": ["AR","SATK"], "weight": 3.79},
    {"name": "Disciplined Sprinter", "stats": ["AR","SPD"], "weight": 3.79},
    {"name": "Calculating Predator", "stats": ["PATK","SATK"], "weight": 3.79},
    {"name": "Hot-headed Charger", "stats": ["PATK","SPD"], "weight": 3.79},
    {"name": "Quick-Witted Livewire", "stats": ["SATK","SPD"], "weight": 3.79},

    # Trios (1.90% each)
    {"name": "Iron Colossus", "stats": ["HP","AR","PATK"], "weight": 1.90},
    {"name": "Sage Bastion", "stats": ["HP","AR","SATK"], "weight": 1.90},
    {"name": "Restless Sentinel", "stats": ["HP","AR","SPD"], "weight": 1.90},
    {"name": "War Architect", "stats": ["HP","PATK","SATK"], "weight": 1.90},
    {"name": "Stampeding Titan", "stats": ["HP","PATK","SPD"], "weight": 1.90},
    {"name": "Swift Savant", "stats": ["HP","SATK","SPD"], "weight": 1.90},
    {"name": "Stoic War-Planner", "stats": ["AR","PATK","SATK"], "weight": 1.90},
    {"name": "Regimented Charger", "stats": ["AR","PATK","SPD"], "weight": 1.90},
    {"name": "Composed Whirlwind", "stats": ["AR","SATK","SPD"], "weight": 1.90},
    {"name": "Lightning Daredevil", "stats": ["PATK","SATK","SPD"], "weight": 1.90},

    # Quads (0.95% each) — note 'AR' not 'AP'
    {"name": "Ascetic Overachiever", "stats": ["AR","PATK","SATK","SPD"], "weight": 0.95},
    {"name": "Freewheeling Titan", "stats": ["HP","PATK","SATK","SPD"], "weight": 0.95},
    {"name": "Gentle Mastermind", "stats": ["HP","AR","SATK","SPD"], "weight": 0.95},
    {"name": "Pure Bruiser", "stats": ["HP","AR","PATK","SPD"], "weight": 0.95},
    {"name": "Deliberate Dominator", "stats": ["HP","AR","PATK","SATK"], "weight": 0.95},

    # Prodigy (0.45%)
    {"name": "Prodigy", "stats": ["HP","AR","PATK","SATK","SPD"], "weight": 0.45},
]

def choose_personality() -> dict:
    weights = [p.get("weight", 1.0) for p in PERSONALITY_TYPES]
    choice = random.choices(PERSONALITY_TYPES, weights=weights, k=1)[0]
    # store only needed fields
    return {"name": choice["name"], "stats": list(choice["stats"])}
    

ACTIONS = ["Attack", "Aggressive", "Special", "Defend"]
ACTION_WEIGHTS = [38, 22, 22, 18]

TIER_PAYOUTS: Dict[int, Tuple[int, int]] = {
    1: (1000, 500),
    2: (7130, 3560),
    3: (13250, 6630),
    4: (19380, 9690),
    5: (25500, 12750),
    6: (31630, 15810),
    7: (37750, 18880),
    8: (43880, 21940),
    9: (50000, 25000),
}

# ─── Battle state ────────────────────────────────────────────
@dataclass
class BattleState:
    user_id: int
    creature_id: int
    tier: int
    user_creature: Dict[str, Any]
    user_current_hp: int
    user_max_hp: int
    opp_creature: Dict[str, Any]
    opp_current_hp: int
    opp_max_hp: int
    logs: List[str]
    is_pvp: bool = False
    opp_user_id: Optional[int] = None
    opp_creature_id: Optional[int] = None
    wager: int = 0
    opp_trainer_name: Optional[str] = None
    next_log_idx: int = 0
    rounds: int = 0

active_battles: Dict[int, BattleState] = {}

# ─── Global battle lock (one at a time) ───────────
battle_lock = asyncio.Lock()
current_battler_id: Optional[int] = None

# ─── Scheduled rewards & regen ───────────────────────────────
@tasks.loop(hours=1)
async def distribute_cash():
    if distribute_cash.current_loop == 0:
        logger.info("Skipping first hourly cash distribution after restart")
        return
    await (await db_pool()).execute("UPDATE trainers SET cash = cash + 60")
    logger.info("Distributed 60 cash to all trainers")

@tasks.loop(time=dtime(hour=0, tzinfo=ZoneInfo("Europe/London")), reconnect=True)
async def distribute_points():
    pool = await db_pool()
    async with pool.acquire() as conn:
        today = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
        last = await conn.fetchval("SELECT last_tp_grant FROM trainers ORDER BY user_id LIMIT 1")
        if last == today:
            logger.info("Trainer points already granted for %s; skipping", today)
            return
        await conn.execute("""
            WITH today AS (
              SELECT $1::date AS d
            )
            UPDATE trainers t
            SET trainer_points = t.trainer_points
              + ((5 + LEAST(GREATEST(t.facility_level - 1, 0), 5))
                 * GREATEST(0, (SELECT d FROM today) - COALESCE(t.last_tp_grant, (SELECT d FROM today)))),
                last_tp_grant = (SELECT d FROM today)
        """, today)
    logger.info("Distributed daily trainer points (catch-up safe)")
async def _catch_up_trainer_points_now():
    """Grant any missed daily trainer points since last_tp_grant without double-granting today."""
    await (await db_pool()).execute("""
        WITH today AS (
          SELECT (now() AT TIME ZONE 'Europe/London')::date AS d
        )
        UPDATE trainers t
        SET trainer_points = t.trainer_points
          + ((5 + LEAST(GREATEST(t.facility_level - 1, 0), 5))
             * GREATEST(0, (SELECT d FROM today) - COALESCE(t.last_tp_grant, (SELECT d FROM today)))),
            last_tp_grant = (SELECT d FROM today)
    """)

@tasks.loop(hours=12)
async def regenerate_hp():
    await (await db_pool()).execute("""
        UPDATE creatures
        SET current_hp = LEAST(
            COALESCE(current_hp, (stats->>'HP')::int * 5)
            + CEIL((stats->>'HP')::numeric * 1.0),
            (stats->>'HP')::int * 5
        )
    """)
    logger.info("Regenerated 20%% HP for all creatures")

# ─── Utility functions ───────────────────────────────────────
def roll_d100() -> int: return random.randint(1, 100)
def rarity_for_tier(tier: int) -> str:
    rarities, weights = TIER_RARITY_WEIGHTS[tier]
    return random.choices(rarities, weights=weights, k=1)[0]
def allocate_stats(rarity: str, extra: int = 0) -> Dict[str, int]:
    pool = random.randint(*POINT_POOLS[rarity]) + extra
    stats = {k: 1 for k in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)
    for _ in range(pool):
        stats[random.choice(PRIMARY_STATS)] += 1
    return stats
def stat_block(name: str, cur_hp: int, max_hp: int, s: Dict[str, int]) -> str:
    return (f"{name} – HP:{cur_hp}/{max_hp} "
            f"AR:{s['AR']} PATK:{s['PATK']} SATK:{s['SATK']} SPD:{s['SPD']}")
def choose_action() -> str:
    return random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]

async def ensure_registered(inter: discord.Interaction) -> Optional[asyncpg.Record]:
    row = await (await db_pool()).fetchrow(
        "SELECT cash, trainer_points, facility_level FROM trainers WHERE user_id=$1", inter.user.id
    )
    if not row:
        await inter.response.send_message("Use /register first.", ephemeral=True)
        return None
    # Opportunistically update stored display name to avoid REST lookups later
    try:
        current_name = (
            getattr(inter.user, 'global_name', None)
            or getattr(inter.user, 'display_name', None)
            or inter.user.name
        )
        await (await db_pool()).execute(
            "UPDATE trainers SET display_name=$1 WHERE user_id=$2 AND COALESCE(display_name,'') <> $1",
            current_name, inter.user.id
        )
    except Exception:
        pass
    return row

async def get_creature_count(user_id: int) -> int:
    return await (await db_pool()).fetchval(
        "SELECT COUNT(*) FROM creatures WHERE owner_id=$1", user_id
    )

async def enforce_creature_cap(inter: discord.Interaction) -> bool:
    count = await get_creature_count(inter.user.id)
    if count >= MAX_CREATURES:
        await inter.response.send_message(
            f"You already own the maximum of {MAX_CREATURES} creatures. "
            "Sell one before spawning a new egg.",
            ephemeral=True
        )
        return False
    return True

async def generate_creature_meta(rarity: str) -> Dict[str, Any]:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    used_names = [r["name"].lower() for r in rows]
    used_words = {w.lower() for r in rows for w in (r["descriptors"] or [])}

    # Random 50 sample for prompt compactness
    used_names_list = list(used_names)
    used_words_list = list(used_words)
    from random import sample as _rsample
    avoid_names = _rsample(used_names_list, min(50, len(used_names_list)))
    avoid_words = _rsample(used_words_list, min(50, len(used_words_list)))
    prompt = f"""
Invent a creature of rarity **{rarity}**. Return ONLY JSON:
{{"name":"1-3 words","descriptors":["w1","w2","w3"]}}
Avoid names: {', '.join(sorted(avoid_names)) if avoid_names else 'None'}
Avoid words: {', '.join(sorted(avoid_words)) if avoid_words else 'None'}
"""
    import asyncio as _asyncio, json as _json, re as _re
    for _ in range(3):
        try:
            resp = await _with_timeout(
                _to_thread(lambda: client.responses.create(
                    model=TEXT_MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )),
                timeout=20.0
            )
            text = (getattr(resp, 'output_text', '') or '').strip()
            if not text:
                continue
            try:
                data = _json.loads(text)
            except Exception:
                m = _re.search(r"\{[\s\S]*\}", text)
                if not m:
                    raise
                data = _json.loads(m.group(0))
            if "name" in data and len(data.get("descriptors", [])) == 3:
                data["name"] = str(data["name"]).title()
                return data
        except _asyncio.TimeoutError:
            logger.warning("generate_creature_meta timed out; retrying…")
        except Exception as e:
            logger.error("OpenAI error: %s", e)
    return {"name": f"Wild{random.randint(1000,9999)}", "descriptors": []}

# ─── Name-only generator for battles ─────────────────────────

async def generate_creature_name(rarity: str) -> str:
    """Ask the model for a name only (short) to reduce tokens. Non-blocking with timeout."""
    # Pull some existing names to avoid dupes
    pool = await db_pool()
    rows = await pool.fetch("SELECT name FROM creatures")
    used = sorted({r["name"].lower() for r in rows})[:50]
    prompt = (
        f"Invent a creature of rarity **{rarity}**. Return ONLY JSON\n"
        f"{{\"name\":\"1-3 words\"}}\n"
        f"Avoid names: {', '.join(used) if used else 'None'}\n"
    )
    import random as _random, re as _re, json as _json
    for _ in range(3):
        try:
            resp = await _with_timeout(
                _to_thread(lambda: client.responses.create(
                    model=TEXT_MODEL,
                    input=prompt,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                )),
                timeout=15.0
            )
            try:
                text = getattr(resp, "output_text", None) or ""
            except Exception:
                text = str(resp) if resp is not None else ""
            m = _re.search(r"\{[\s\S]*\}", text or "")
            data = _json.loads(m.group(0)) if m else {}
            if "name" in data:
                return str(data["name"]).title()
        except asyncio.TimeoutError:
            logger.warning("generate_creature_name timed out; retrying…")
        except Exception as e:
            logger.error("OpenAI name-only error: %s", e)
    return f"Wild{_random.randint(1000,9999)}"

# Robust extractor for Images API responses (dict or SDK object)
def _extract_image_url(img_resp):
    # Newer SDK returns ImagesResponse with .data -> objects with .url
    try:
        data = getattr(img_resp, "data", None)
        if data and len(data) > 0:
            url = getattr(data[0], "url", None)
            if url:
                return url
    except Exception:
        pass
    # Older / dict-like
    try:
        return img_resp["data"][0]["url"]
    except Exception:
        return None

def _extract_image_bytes(img_resp):
    # Try SDK object path
    try:
        data = getattr(img_resp, "data", None)
        if data and len(data) > 0:
            b64v = getattr(data[0], "b64_json", None)
            if b64v:
                import base64 as _b64
                return _b64.b64decode(b64v)
    except Exception:
        pass
    # Try dict-like
    try:
        b64v = img_resp["data"][0].get("b64_json")
        if b64v:
            import base64 as _b64
            return _b64.b64decode(b64v)
    except Exception:
        pass
    return None
# ─── OpenAI helpers (Responses API) ─────────────────────────

def _safe_dump_response(resp) -> str:
    try:
        if hasattr(resp, "model_dump"):
            d = resp.model_dump(exclude_none=True)
        elif hasattr(resp, "dict"):
            d = resp.dict()
        else:
            return str(resp)[:1200]
        import json as _json
        return _json.dumps(d, ensure_ascii=False)[:1200]
    except Exception:
        try:
            return str(resp)[:1200]
        except Exception:
            return "<unprintable response>"

# ─── Async helpers to isolate blocking calls + enforce timeouts ──────────────
async def _to_thread(fn):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)

async def _with_timeout(coro, timeout: float = 15.0):
    return await asyncio.wait_for(coro, timeout=timeout)

async def _gpt_json_object(prompt: str, *, temperature: float = 1.0, max_output_tokens: int = 512) -> dict | None:
    """
    Calls the Responses API and requests a strict JSON object back.
    Returns a dict or None on failure.
    """
    loop = asyncio.get_running_loop()
    try:
        resp = await _with_timeout(
            loop.run_in_executor(
                None,
                lambda: client.responses.create(
                    model=TEXT_MODEL,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                    response_format={"type": "json_object"},
                ),
            ),
            timeout=20.0,
        )
        text = getattr(resp, "output_text", None) or ""
        text = text.strip()
        if not text:
            if OPENAI_DEBUG:
                logger.error("Responses debug (empty): %s", _safe_dump_response(resp))
            return None
        return json.loads(text)
    except asyncio.TimeoutError:
        logger.error("OpenAI JSON call timed out.")
        return None
    except Exception as e:
        logger.error("OpenAI JSON call failed: %s", e)
        return None

async def send_chunks(inter: discord.Interaction, content: str, ephemeral: bool = False):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    if not inter.response.is_done():
        await inter.response.send_message(chunks[0], ephemeral=ephemeral)
    else:
        await inter.followup.send(chunks[0], ephemeral=ephemeral)
    for chunk in chunks[1:]:
        await inter.followup.send(chunk, ephemeral=ephemeral)

# ─── Command listing helper ─────────────────────────────────
def _build_command_list(bot: commands.Bot) -> str:
    """
    Returns a formatted string listing all slash commands and their parameters.
    Automatically paginated later by send_chunks().
    """
    try:
        cmds = list(bot.tree.get_commands())
    except Exception:
        cmds = []
    # Sort by name for stable output
    cmds.sort(key=lambda c: getattr(c, "name", "").lower())
    lines = ["**Commands**"]
    for c in cmds:
        name = getattr(c, "name", None) or "<unknown>"
        desc = getattr(c, "description", "") or ""
        # Collect parameter names (with ? for optional) if available
        params = []
        try:
            for p in getattr(c, "parameters", []):
                # discord.app_commands.Parameter has attributes: name, required
                pname = getattr(p, "name", None) or getattr(p, "display_name", None) or "arg"
                preq = getattr(p, "required", True)
                params.append(f"<{pname}>" if preq else f"[{pname}]")
        except Exception:
            pass
        sig = (" " + " ".join(params)) if params else ""
        # Keep each command as a single concise bullet line
        line = f"/{name}{sig} — {desc}".strip()
        # Discord hard cap ~2000 chars per message; send_chunks handles chunking,
        # but keep individual lines under ~180 chars to avoid split mid-line.
        if len(line) > 180:
            line = line[:177] + "…"
        lines.append(line)
    return "\n".join(lines) or "No commands registered."

# ─── Battle cap helper ───────────────────────────────────────
async def _can_start_battle_and_increment(creature_id: int) -> Tuple[bool, int]:
    pool = await db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row = await conn.fetchrow(
                "SELECT count FROM battle_caps WHERE creature_id=$1 AND day=$2 FOR UPDATE",
                creature_id, day
            )
            if not row:
                await conn.execute(
                    "INSERT INTO battle_caps(creature_id, day, count) VALUES($1,$2,1)",
                    creature_id, day
                )
                return True, 1
            current = row["count"]
            if current >= DAILY_BATTLE_CAP:
                return False, current
            new_count = current + 1
            await conn.execute(
                "UPDATE battle_caps SET count=$3 WHERE creature_id=$1 AND day=$2",
                creature_id, day, new_count
            )
            return True, new_count

# NEW: bulk fetch remaining battles for /creatures
async def _battles_left_map(creature_ids: List[int]) -> Dict[int, int]:
    """
    Returns {creature_id: remaining} for the given ids for today's Europe/London day.
    Creatures with no row yet are assumed to have full cap remaining.
    """
    result: Dict[int, int] = {cid: DAILY_BATTLE_CAP for cid in creature_ids}
    if not creature_ids:
        return result
    pool = await db_pool()
    async with pool.acquire() as conn:
        day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
        rows = await conn.fetch(
            "SELECT creature_id, count FROM battle_caps WHERE day=$1 AND creature_id = ANY($2::int[])",
            day, creature_ids
        )
    for r in rows:
        left = max(0, DAILY_BATTLE_CAP - int(r["count"]))
        result[int(r["creature_id"])] = left
    return result

# ─── Progress / Glyphs helpers ───────────────────────────────
async def _get_progress(conn: asyncpg.Connection, creature_id: int, tier: int) -> Optional[asyncpg.Record]:
    return await conn.fetchrow(
        "SELECT wins, glyph_unlocked FROM creature_progress WHERE creature_id=$1 AND tier=$2",
        creature_id, tier
    )

async def _get_wins_for_tier(creature_id: int, tier: int) -> int:
    pool = await db_pool()
    async with pool.acquire() as conn:
        row = await _get_progress(conn, creature_id, tier)
        return (row["wins"] if row else 0)

async def _max_unlocked_tier(creature_id: int) -> int:
    pool = await db_pool()
    async with pool.acquire() as conn:
        unlocked = 1
        for t in range(1, 9):
            row = await _get_progress(conn, creature_id, t)
            wins_t = (row["wins"] if row else 0)
            if wins_t >= 5:
                unlocked = t + 1
            else:
                break
        return unlocked

async def _record_win_and_maybe_unlock(creature_id: int, tier: int) -> Tuple[int, bool]:
    pool = await db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await _get_progress(conn, creature_id, tier)
            if not row:
                wins = 1
                glyph = (wins >= 5)
                await conn.execute(
                    "INSERT INTO creature_progress(creature_id,tier,wins,glyph_unlocked) VALUES($1,$2,$3,$4)",
                    creature_id, tier, wins, glyph
                )
                return wins, glyph
            wins = row["wins"] + 1
            glyph = row["glyph_unlocked"]
            if not glyph and wins >= 5:
                await conn.execute(
                    "UPDATE creature_progress SET wins=$3, glyph_unlocked=true WHERE creature_id=$1 AND tier=$2",
                    creature_id, tier, wins
                )
                return wins, True
            else:
                await conn.execute(
                    "UPDATE creature_progress SET wins=$3 WHERE creature_id=$1 AND tier=$2",
                    creature_id, tier, wins
                )
                return wins, False

def simulate_round(st: BattleState):
    st.rounds += 1
    st.logs.append(f"Round {st.rounds}")
    sudden_death_mult = 1.1 ** (st.rounds // 10)
    if st.rounds % 10 == 0:
        st.logs.append(f"⚡ Sudden Death intensifies! Global damage ×{sudden_death_mult:.2f}")
    while True:
        user_act, opp_act = choose_action(), choose_action()
        if not (user_act == "Defend" and opp_act == "Defend"):
            break
    st.logs.append(
        f"{st.user_creature['name']} chooses **{user_act}** | "
        f"{st.opp_creature['name']} chooses **{opp_act}**"
    )
    st.logs.append(
        f"{st.user_creature['name']} HP {st.user_current_hp}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {st.opp_current_hp}/{st.opp_max_hp}"
    )
    uc, oc = st.user_creature, st.opp_creature
    order = [("user", uc, oc, user_act, opp_act), ("opp", oc, uc, opp_act, user_act)]
    if uc["stats"]["SPD"] < oc["stats"]["SPD"] or (
        uc["stats"]["SPD"] == oc["stats"]["SPD"] and random.choice([0, 1])
    ):
        order.reverse()
    for side, atk, dfn, act, dfn_act in order:
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0: break
        if act == "Defend":
            st.logs.append(f"{atk['name']} is defending.")
            continue
        swings = 2 if atk["stats"]["SPD"] >= 1.5 * dfn["stats"]["SPD"] else 1
        for _ in range(swings):
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0: break
            S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
            AR_val = 0 if act == "Special" else dfn["stats"]["AR"] // 2
            rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
            s = sum(rolls)
            dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
            if act == "Aggressive": dmg = math.ceil(dmg * 1.25)
            if dfn_act == "Defend": dmg = max(1, math.ceil(dmg * 0.5))
            if sudden_death_mult > 1.0: dmg = max(1, math.ceil(dmg * sudden_death_mult))
            if side == "user": st.opp_current_hp -= dmg
            else: st.user_current_hp -= dmg
            act_word = {"Attack":"hits","Aggressive":"aggressively hits","Special":"unleashes a special attack on"}[act]
            note = " (defended)" if dfn_act == "Defend" else ""
            st.logs.append(f"{atk['name']} {act_word} {dfn['name']} for {dmg} dmg"
                           f"{' (rolls '+str(rolls)+')' if act!='Special' else ''}{note}")
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                st.logs.append(f"{dfn['name']} is down!")
                break
    st.logs.append(
        f"{st.user_creature['name']} HP {max(st.user_current_hp,0)}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {max(st.opp_current_hp,0)}/{st.opp_max_hp}"
    )
    st.logs.append("")

# ─── Leaderboard helpers ─────────────────────────────────────
NAME_W = 22
TRAINER_W = 16
_owner_name_cache: Dict[int, str] = {}

async def _resolve_trainer_name_from_db(user_id: int) -> Optional[str]:
    """Get trainer's stored display name from DB (preferred) or None.
    This avoids hitting the Discord REST API and rate limits.
    """
    try:
        return await (await db_pool()).fetchval(
            "SELECT COALESCE(display_name, user_id::text) FROM trainers WHERE user_id=$1",
            user_id,
        )
    except Exception:
        return None

async def _resolve_trainer_name(owner_id: int) -> str:
    """Resolve trainer display name without privileged member intent."""
    if owner_id in _owner_name_cache:
        return _owner_name_cache[owner_id]
    name: Optional[str] = None
    u = bot.get_user(owner_id)
    if not u:
        try:
            u = await bot.fetch_user(owner_id)  # REST; no privileged intent needed
        except Exception:
            u = None
    if u:
        name = getattr(u, "global_name", None) or u.name
    if not name:
        name = str(owner_id)
    if len(name) > TRAINER_W:
        name = name[:TRAINER_W]
    _owner_name_cache[owner_id] = name
    return name

async def _backfill_creature_records():
    pool = await db_pool()
    await pool.execute("""
        INSERT INTO creature_records (creature_id, owner_id, name, ovr)
        SELECT c.id, c.owner_id, c.name,
               COALESCE((c.stats->>'HP')::int,0) +
               COALESCE((c.stats->>'AR')::int,0) +
               COALESCE((c.stats->>'PATK')::int,0) +
               COALESCE((c.stats->>'SATK')::int,0) +
               COALESCE((c.stats->>'SPD')::int,0)
        FROM creatures c
        LEFT JOIN creature_records r
          ON r.owner_id = c.owner_id AND LOWER(r.name) = LOWER(c.name)
        WHERE r.owner_id IS NULL
    """)
    await pool.execute("""
        UPDATE creature_records r
        SET ovr =
            COALESCE((c.stats->>'HP')::int,0) +
            COALESCE((c.stats->>'AR')::int,0) +
            COALESCE((c.stats->>'PATK')::int,0) +
            COALESCE((c.stats->>'SATK')::int,0) +
            COALESCE((c.stats->>'SPD')::int,0)
        FROM creatures c
        WHERE r.creature_id = c.id
    """)
    logger.info("Backfilled creature_records for existing creatures (if any missing).")

async def _ensure_record(owner_id: int, creature_id: int, name: str, ovr: Optional[int] = None):
    if ovr is None:
        row = await (await db_pool()).fetchrow("SELECT stats FROM creatures WHERE id=$1", creature_id)
        if row:
            try:
                stats = json.loads(row["stats"])
                ovr = int(sum(stats.values()))
            except Exception:
                ovr = 0
        else:
            ovr = 0
    await (await db_pool()).execute("""
        INSERT INTO creature_records (creature_id, owner_id, name, ovr)
        VALUES ($1,$2,$3,$4)
        ON CONFLICT (owner_id, name) DO NOTHING
    """, creature_id, owner_id, name, ovr)

async def _record_result(owner_id: int, name: str, won: bool):
    if won:
        await (await db_pool()).execute(
            "UPDATE creature_records SET wins = wins + 1 WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
            owner_id, name
        )
    else:
        await (await db_pool()).execute(
            "UPDATE creature_records SET losses = losses + 1 WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
            owner_id, name
        )

async def _record_pvp_result(user_id: int, won: bool):
    """Record a PvP win/loss for a trainer."""
    await (await db_pool()).execute(
        """
        INSERT INTO pvp_records(user_id, wins, losses, last_battle_at)
        VALUES ($1, $2, $3, now())
        ON CONFLICT (user_id) DO UPDATE
        SET wins = pvp_records.wins + $2,
            losses = pvp_records.losses + $3,
            last_battle_at = now()
        """,
        user_id,
        1 if won else 0,
        0 if won else 1,
    )

async def _record_death(owner_id: int, name: str):
    await (await db_pool()).execute(
        "UPDATE creature_records SET is_dead=true, died_at=now() WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
        owner_id, name
    )

async def _get_leaderboard_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM leaderboard_messages LIMIT 1")
    if chan:
        return int(chan)
    if LEADERBOARD_CHANNEL_ID_ENV:
        try:
            return int(LEADERBOARD_CHANNEL_ID_ENV)
        except Exception:
            logger.error("LEADERBOARD_CHANNEL_ID env was set but not an integer.")
    return None

async def _get_or_create_leaderboard_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch leaderboard channel %s: %s", channel_id, e)
        return None

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT message_id FROM leaderboard_messages WHERE channel_id=$1", channel_id
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
        except Exception:
            message = None

    if message is None:
        try:
            message = await channel.send("Initializing leaderboard…")
            await pool.execute("""
                INSERT INTO leaderboard_messages(channel_id, message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET message_id=EXCLUDED.message_id
            """, channel_id, message.id)
        except Exception as e:
            logger.error("Failed to create leaderboard message: %s", e)
            return None

    return message

async def _get_or_create_pvp_leaderboard_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch leaderboard channel %s: %s", channel_id, e)
        return None

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT pvp_message_id FROM leaderboard_messages WHERE channel_id=$1",
        channel_id,
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
        except Exception:
            message = None

    if message is None:
        try:
            message = await channel.send("Initializing PvP leaderboard…")
            await pool.execute(
                """
                INSERT INTO leaderboard_messages(channel_id, pvp_message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET pvp_message_id=EXCLUDED.pvp_message_id
                """,
                channel_id,
                message.id,
            )
        except Exception as e:
            logger.error("Failed to create PvP leaderboard message: %s", e)
            return None

    return message

# UPDATED: include glyph and OVR columns

def _format_leaderboard_lines(
    rows: List[Tuple[str, int, int, bool, str, Optional[int], Optional[int]]]
) -> str:
    """
    Input rows: (name, wins, losses, is_dead, trainer_name, max_glyph_tier, ovr)
    Use a diff code block; prefix '-' for dead to render red.
    """
    lines: List[str] = []
    header = (
        f"{'#':>3}. {'Name':<{NAME_W}} {'Trainer':<{TRAINER_W}} {'W':>4} {'L':>4} {'Glyph':>5} {'OVR':>5} Status"
    )
    lines.append("  " + header)
    for idx, (name, wins, losses, dead, trainer_name, glyph_tier, ovr) in enumerate(rows, start=1):
        name = (name or "")[:NAME_W]
        trainer = (trainer_name or "")[:TRAINER_W]
        glyph_display = "-" if not glyph_tier or glyph_tier <= 0 else str(glyph_tier)
        ovr_display = "-" if not ovr or ovr <= 0 else str(int(((ovr + 5) // 10) * 10))
        base_line = (
            f"{idx:>3}. {name:<{NAME_W}} {trainer:<{TRAINER_W}} {wins:>4} {losses:>4} {glyph_display:>5} {ovr_display:>5} "
            f"{'💀 DEAD' if dead else ''}"
        )
        lines.append(("- " if dead else "  ") + base_line)
    return "```diff\n" + "\n".join(lines) + "\n```"

def _format_pvp_leaderboard_lines(rows: List[Tuple[str, int, int]]) -> str:
    lines: List[str] = []
    header = f"{'#':>3}. {'Trainer':<{TRAINER_W}} {'W':>4} {'L':>4}"
    lines.append("  " + header)
    for idx, (trainer, wins, losses) in enumerate(rows, start=1):
        trainer = (trainer or "")[:TRAINER_W]
        lines.append(f"  {idx:>3}. {trainer:<{TRAINER_W}} {wins:>4} {losses:>4}")
    return "```\n" + "\n".join(lines) + "\n```"
# ─── Encyclopedia helpers ────────────────────────────────────
async def _get_encyclopedia_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM encyclopedia_channel LIMIT 1")
    if chan:
        try:
            return int(chan)
        except Exception:
            return None
    return None

async def _ensure_encyclopedia_channel(channel_id: int) -> Optional[discord.TextChannel]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            return channel
        return None
    except Exception as e:
        logger.error("Failed to fetch encyclopedia channel %s: %s", channel_id, e)
        return None

def _format_stats_block(stats: Dict[str, int]) -> str:
    # Show HP as both points and max HP (×5)
    max_hp = stats.get("HP", 0) * 5
    return (
        f"HP: {stats.get('HP', 0)} (Max {max_hp}) | "
        f"AR: {stats.get('AR', 0)} | "
        f"PATK: {stats.get('PATK', 0)} | "
        f"SATK: {stats.get('SATK', 0)} | "
        f"SPD: {stats.get('SPD', 0)}"
    )

async def _gpt_generate_bio_and_image(cre_name: str, rarity: str, traits: list[str], stats: Dict[str, int]) -> tuple[str, Optional[str], Optional[bytes]]:
    """
    Returns (bio_text, image_url or None)
    Uses OpenAI for both text (ChatCompletion) and image (Images.create).
    """
    # 1) Bio text
    try:
        sys_prompt = (
            "You are writing a concise creature entry for an arena-battler encyclopedia. "
            "These creatures are NOT from natural Warcraft lore—they are laboratory-bred specifically for fighting arenas. "
            "Tone: punchy, evocative, 3–6 sentences max. "
            "Include a hint of distinctive abilities implied by the stats/traits. Do not mention that you are an AI."
        )
        user_prompt = (
            f"Name: {cre_name}\n"
            f"Rarity: {rarity}\n"
            f"Traits/Descriptors: {', '.join(traits) if traits else 'None'}\n"
            f"Stats (HP, AR, PATK, SATK, SPD): {stats}\n\n"
            "Write the bio now."
        )

        # Keep compatible with your existing OpenAI usage pattern
        loop = asyncio.get_running_loop()
        resp = await _with_timeout(
            loop.run_in_executor(
                None,
                lambda: client.responses.create(
                    model=TEXT_MODEL,
                    input=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ),
            ),
            timeout=25.0,
        )
        bio_text = getattr(resp, 'output_text', '') or ''
    except Exception as e:
        logger.error("OpenAI bio error: %s", e)
        bio_text = "A lab-forged arena combatant. (Bio generation failed.)"

    

# 2) Image (Warcraft Cinematic CGI style)
    image_url = None
    try:
        traits_str = ", ".join(traits) if traits else ""
        img_prompt = (
            f"{cre_name}, {rarity} rarity, {traits_str}. "
            "Ultra-detailed digital image in the style of Warcraft Cinematic CGI, dramatic lighting, "
            "fantasy composition, moody backdrop, crisp focus, volumetric light, high contrast."
        )
        loop = asyncio.get_running_loop()
        image_url = None
        image_bytes = None

        # Single high-res attempt with NO timeout. If it fails for any reason, fall back to 512x512 (also no timeout).
        try:
            img_resp = await loop.run_in_executor(
                None,
                lambda: client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=img_prompt,
                    n=1,
                    size="1024x1024",
                ),
            )
        except Exception as _e1:
            logger.warning("Primary image generation failed (%s). Falling back to 512x512.", type(_e1).__name__)
            img_resp = await loop.run_in_executor(
                None,
                lambda: client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=img_prompt,
                    n=1,
                    size="512x512",
                ),
            )

        # Prefer URL; bytes fallback handled if present
        image_url = _extract_image_url(img_resp)
        if not image_url:
            image_bytes = _extract_image_bytes(img_resp)
        if not (image_url or image_bytes):
            raise RuntimeError("Image generation returned no data")
    except Exception as e:
        logger.error("OpenAI image error (%s): %s", type(e).__name__, e)
        image_url = None
        image_bytes = None
    return bio_text, image_url, image_bytes

async def update_leaderboard_now(reason: str = "manual/trigger") -> None:
    channel_id = await _get_leaderboard_channel_id()
    if not channel_id:
        return
    message = await _get_or_create_leaderboard_message(channel_id)
    pvp_message = await _get_or_create_pvp_leaderboard_message(channel_id)
    if message is None and pvp_message is None:
        return

    pool = await db_pool()
    # UPDATED: pull highest obtained glyph tier for each creature via subquery
    # Refresh lifetime highest glyphs from creature_progress so leaderboard never regresses to '-'
    try:
        await pool.execute(
            """
            UPDATE creature_records r
            SET highest_glyph_tier = GREATEST(
                COALESCE(r.highest_glyph_tier, 0),
                COALESCE(g.max_tier, 0)
            )
            FROM (
                SELECT cp.creature_id, MAX(cp.tier) AS max_tier
                FROM creature_progress cp
                WHERE cp.glyph_unlocked = TRUE
                GROUP BY cp.creature_id
            ) g
            WHERE r.creature_id = g.creature_id
            """
        )
    except Exception as e:
        logger.warning("Failed to refresh highest_glyph_tier before leaderboard: %s", e)

    rows = await pool.fetch(
    """
        SELECT
            r.name,
            r.wins,
            r.losses,
            r.is_dead,
            COALESCE(t.display_name, r.owner_id::text) AS trainer_name,
            COALESCE(r.highest_glyph_tier, 0) AS max_glyph_tier,
            COALESCE(r.ovr, 0) AS ovr
        FROM creature_records r
        LEFT JOIN trainers t ON t.user_id = r.owner_id
        ORDER BY max_glyph_tier DESC, r.wins DESC, r.losses ASC, r.name ASC
        LIMIT 20
    """
)
    formatted: List[Tuple[str, int, int, bool, str, int, int]] = [
        (
            r["name"],
            r["wins"],
            r["losses"],
            r["is_dead"],
            r["trainer_name"],
            r["max_glyph_tier"],
            r["ovr"],
        )
        for r in rows
    ]
    updated_ts = int(time.time())
    title = (
        f"**Creature Leaderboard — Top 20 (Wins / Losses / Highest Glyph / OVR)**\n"
        f"Updated: <t:{updated_ts}:R>\n"
    )
    content = title + _format_leaderboard_lines(formatted)
    if message is not None:
        try:
            await message.edit(content=content)
            logger.info("Leaderboard updated (%s).", reason)
        except Exception as e:
            logger.error("Failed to edit leaderboard message: %s", e)

    # PvP leaderboard
    pvp_rows = await pool.fetch(
        """
        SELECT
            COALESCE(t.display_name, pr.user_id::text) AS trainer_name,
            pr.wins,
            pr.losses
        FROM pvp_records pr
        LEFT JOIN trainers t ON t.user_id = pr.user_id
        ORDER BY pr.wins DESC, pr.losses ASC, trainer_name ASC
        LIMIT 20
        """
    )
    pvp_formatted: List[Tuple[str, int, int]] = [
        (r["trainer_name"], r["wins"], r["losses"]) for r in pvp_rows
    ]
    pvp_title = (
        f"**Trainer PvP Leaderboard — Top 20 (Wins / Losses)**\n"
        f"Updated: <t:{updated_ts}:R>\n"
    )
    pvp_content = pvp_title + _format_pvp_leaderboard_lines(pvp_formatted)
    if pvp_message is not None:
        try:
            await pvp_message.edit(content=pvp_content)
        except Exception as e:
            logger.error("Failed to edit PvP leaderboard message: %s", e)

@tasks.loop(minutes=5)
async def update_leaderboard_periodic():
    await update_leaderboard_now(reason="periodic")


async def _get_shop_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM shop_messages LIMIT 1")
    if chan:
        return int(chan)
    if SHOP_CHANNEL_ID_ENV:
        try:
            return int(SHOP_CHANNEL_ID_ENV)
        except Exception:
            logger.error("SHOP_CHANNEL_ID env was set but not an integer.")
    return None


async def _get_or_create_shop_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch shop channel %s: %s", channel_id, e)
        return None

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT message_id FROM shop_messages WHERE channel_id=$1", channel_id
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
        except Exception:
            message = None

    if message is None:
        try:
            message = await channel.send("Initializing shop…")
            await pool.execute(
                """
                INSERT INTO shop_messages(channel_id, message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET message_id=EXCLUDED.message_id
                """,
                channel_id,
                message.id,
            )
        except Exception as e:
            logger.error("Failed to create shop message: %s", e)
            return None

    return message


async def _get_controls_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM controls_messages LIMIT 1")
    if chan:
        return int(chan)
    if CONTROLS_CHANNEL_ID_ENV:
        try:
            return int(CONTROLS_CHANNEL_ID_ENV)
        except Exception:
            logger.error("CONTROLS_CHANNEL_ID env was set but not an integer.")
    return None


async def _get_or_create_controls_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch controls channel %s: %s", channel_id, e)
        return None

    global controls_view
    if controls_view is None:
        controls_view = ControlsView()

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT message_id FROM controls_messages WHERE channel_id=$1",
        channel_id,
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
            bot.add_view(controls_view, message_id=message.id)
        except Exception:
            message = None

    if message is None:
        try:
            embed = discord.Embed(
                title="Controls",
                description="Use the buttons below to manage your trainer and creatures.",
            )
            message = await channel.send(embed=embed, view=controls_view)
            # Explicitly register the persistent view so the buttons continue
            # working even after the bot restarts. Without this, the message
            # would remain but future interaction callbacks would fail with
            # "This interaction failed" because the view was not re-bound.
            bot.add_view(controls_view, message_id=message.id)
            await pool.execute(
                """
                INSERT INTO controls_messages(channel_id, message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET message_id=EXCLUDED.message_id
                """,
                channel_id,
                message.id,
            )
        except Exception as e:
            logger.error("Failed to create controls message: %s", e)
            return None

    return message


async def update_shop_now(reason: str = "manual") -> None:
    channel_id = await _get_shop_channel_id()
    if not channel_id:
        return
    message = await _get_or_create_shop_message(channel_id)
    if message is None:
        return
    pool = await db_pool()
    rows = await pool.fetch(
        """
        SELECT c.name, c.stats, c.rarity, c.personality, ct.price,
               COALESCE(t.display_name, c.owner_id::text) AS trainer_name
        FROM creature_shop ct
        JOIN creatures c ON c.id = ct.creature_id
        LEFT JOIN trainers t ON t.user_id = c.owner_id
        ORDER BY ct.listed_at
        """
    )
    updated_ts = int(time.time())
    embed = discord.Embed(
        title="Creature Shop",
        description=f"Updated: <t:{updated_ts}:R>",
    )
    if not rows:
        embed.add_field(
            name="No listings",
            value="_No creatures currently for sale._",
            inline=False,
        )
    else:
        for r in rows:
            stats = json.loads(r["stats"]) if r["stats"] else {}
            ovr = int(sum(stats.values()))
            personality = _parse_personality(r.get("personality"))
            p_name = personality.get("name") if personality else "-"
            embed.add_field(
                name=r["name"],
                value=(
                    f"Trainer: {r['trainer_name']} — {r['rarity']} "
                    f"— OVR {ovr} — Personality: {p_name} — Price: {r['price']}"
                ),
                inline=False,
            )
    try:
        await message.edit(content=None, embed=embed, view=ShopView())
        logger.info("Shop updated (%s).", reason)
    except Exception as e:
        logger.error("Failed to edit shop message: %s", e)


@tasks.loop(minutes=5)
async def update_shop_periodic():
    await update_shop_now(reason="periodic")

# ─── Battle finalize (records + leaderboard) ─────────────────
async def finalize_battle(inter: discord.Interaction, st: BattleState):
    player_won = st.opp_current_hp <= 0 and st.user_current_hp > 0
    pool = await db_pool()

    if st.is_pvp:
        await _ensure_record(
            st.user_id,
            st.creature_id,
            st.user_creature["name"],
            int(sum(st.user_creature.get("stats", {}).values())),
        )
        await _ensure_record(
            st.opp_user_id,
            st.opp_creature_id,
            st.opp_creature["name"],
            int(sum(st.opp_creature.get("stats", {}).values())),
        )
        await _record_result(st.user_id, st.user_creature["name"], player_won)
        await _record_result(st.opp_user_id, st.opp_creature["name"], not player_won)
        await _record_pvp_result(st.user_id, player_won)
        await _record_pvp_result(st.opp_user_id, not player_won)

        if player_won:
            await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", st.wager, st.user_id)
            await pool.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", st.wager, st.opp_user_id)
        else:
            await pool.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", st.wager, st.user_id)
            await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", st.wager, st.opp_user_id)

        st.logs.append(
            f"You {'won' if player_won else 'lost'} the PvP battle: {'+' if player_won else '-'}{st.wager} cash."
        )

        gained_stat = None
        try:
            winner_cre = st.user_creature if player_won else st.opp_creature
            winner_id = st.creature_id if player_won else st.opp_creature_id
            winner_cur_hp = st.user_current_hp if player_won else st.opp_current_hp
            gained_stat = random.choice(PRIMARY_STATS)
            new_stats = dict(winner_cre["stats"])
            new_stats[gained_stat] = int(new_stats.get(gained_stat, 0)) + 1
            new_max_hp = int(new_stats["HP"]) * 5
            new_cur_hp = winner_cur_hp
            if gained_stat == "HP":
                new_cur_hp = min(winner_cur_hp + 5, new_max_hp)
            await pool.execute(
                "UPDATE creatures SET stats=$1, current_hp=$2 WHERE id=$3",
                json.dumps(new_stats), new_cur_hp, winner_id,
            )
            winner_cre["stats"] = new_stats
            if player_won:
                st.user_max_hp = new_max_hp
                st.user_current_hp = new_cur_hp
            else:
                st.opp_max_hp = new_max_hp
                st.opp_current_hp = new_cur_hp
            if gained_stat == "HP":
                st.logs.append(
                    f"✨ **{winner_cre['name']}** gained **+1 HP** from the victory (Max HP is now {new_max_hp}, current {new_cur_hp}/{new_max_hp})."
                )
            else:
                st.logs.append(
                    f"✨ **{winner_cre['name']}** gained **+1 {gained_stat}** from the victory."
                )
        except Exception as e:
            logger.error("Failed to apply PvP victory stat gain: %s", e)

        loser_cre = st.opp_creature if player_won else st.user_creature
        loser_id = st.opp_creature_id if player_won else st.creature_id
        loser_user = st.opp_user_id if player_won else st.user_id
        death_roll = random.random()
        pct = int(death_roll * 100)
        loser_died = False
        if death_roll < 0.5:
            loser_died = True
            await _record_death(loser_user, loser_cre["name"])
            await pool.execute("DELETE FROM creatures WHERE id=$1", loser_id)
            st.logs.append(f"💀 Death roll {pct} (<50): **{loser_cre['name']}** died (kept on leaderboard).")
        else:
            st.logs.append(f"🛡️ Death roll {pct} (≥50): **{loser_cre['name']}** survived the defeat.")

        asyncio.create_task(update_leaderboard_now(reason="battle_finalize"))
        return {
            "player_won": player_won,
            "payout": st.wager if player_won else -st.wager,
            "gained_stat": gained_stat,
            "loser_died": loser_died,
        }

    wins = None
    unlocked_now = False
    gained_stat = None
    died = False
    win_cash, loss_cash = TIER_PAYOUTS[st.tier]
    payout = win_cash if player_won else loss_cash

    await _ensure_record(
        st.user_id,
        st.creature_id,
        st.user_creature["name"],
        int(sum(st.user_creature.get("stats", {}).values())),
    )
    await _record_result(st.user_id, st.user_creature["name"], player_won)

    await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", payout, st.user_id)
    st.logs.append(f"You {'won' if player_won else 'lost'} the Tier {st.tier} battle: +{payout} cash awarded.")

    if player_won:
        wins, unlocked_now = await _record_win_and_maybe_unlock(st.creature_id, st.tier)
        st.logs.append(f"Progress: Tier {st.tier} wins = {wins}/5.")
        if unlocked_now:
            try:
                await pool.execute(
                    """
                    UPDATE creature_records
                    SET highest_glyph_tier = GREATEST(COALESCE(highest_glyph_tier,0), $3)
                    WHERE owner_id=$1 AND LOWER(name)=LOWER($2)
                    """,
                    st.user_id, st.user_creature['name'], st.tier
                )
            except Exception as _e:
                logger.warning("Failed to bump highest_glyph_tier: %s", _e)
            st.logs.append(
                f"🏅 **Tier {st.tier} Glyph unlocked!**" +
                (f" {st.user_creature['name']} may now battle **Tier {st.tier + 1}**." if st.tier < 9 else "")
            )

        try:
            gained_stat = random.choice(PRIMARY_STATS)
            new_stats = dict(st.user_creature["stats"])
            new_stats[gained_stat] = int(new_stats.get(gained_stat, 0)) + 1
            new_max_hp = int(new_stats["HP"]) * 5
            new_cur_hp = st.user_current_hp
            if gained_stat == "HP":
                new_cur_hp = min(st.user_current_hp + 5, new_max_hp)
            await pool.execute(
                "UPDATE creatures SET stats=$1, current_hp=$2 WHERE id=$3",
                json.dumps(new_stats), new_cur_hp, st.creature_id
            )
            st.user_creature["stats"] = new_stats
            st.user_max_hp = new_max_hp
            st.user_current_hp = new_cur_hp
            if gained_stat == "HP":
                st.logs.append(
                    f"✨ **{st.user_creature['name']}** gained **+1 HP** from the victory "
                    f"(Max HP is now {new_max_hp}, current {new_cur_hp}/{new_max_hp})."
                )
            else:
                st.logs.append(
                    f"✨ **{st.user_creature['name']}** gained **+1 {gained_stat}** from the victory."
                )
        except Exception as e:
            logger.error("Failed to apply victory stat gain: %s", e)

    else:
        death_roll = random.random()
        pct = int(death_roll * 100)
        if death_roll < 0.5:
            await _record_death(st.user_id, st.user_creature["name"])
            died = True
            await pool.execute("DELETE FROM creatures WHERE id=$1", st.creature_id)
            st.logs.append(f"💀 Death roll {pct} (<50): **{st.user_creature['name']}** died (kept on leaderboard).")
        else:
            st.logs.append(f"🛡️ Death roll {pct} (≥50): **{st.user_creature['name']}** survived the defeat.")

    asyncio.create_task(update_leaderboard_now(reason="battle_finalize"))

    return {"player_won": player_won, "payout": payout, "wins": wins, "unlocked_now": unlocked_now, "gained_stat": gained_stat, "died": died}

# ─── Public battle summary helper ───────────────────────────
def format_public_battle_summary(st: BattleState, summary: dict, trainer_name: str) -> str:
    if st.is_pvp:
        opp_name = st.opp_trainer_name or "Opponent"
        winner_trainer = trainer_name if summary.get("player_won") else opp_name
        winner_cre = st.user_creature["name"] if summary.get("player_won") else st.opp_creature["name"]
        loser_cre = st.opp_creature["name"] if summary.get("player_won") else st.user_creature["name"]
        lines = [
            f"**PvP Battle Result** — {trainer_name}'s **{st.user_creature['name']}** vs {opp_name}'s **{st.opp_creature['name']}**",
            f"🏅 Winner: {winner_trainer}'s **{winner_cre}**",
            f"💰 {winner_trainer} wins {abs(summary.get('payout',0))} cash!",
        ]
        gained = summary.get("gained_stat")
        if gained:
            lines.append(f"✨ {winner_cre} gained +1 {gained}.")
        if summary.get("loser_died"):
            lines.append(f"💀 {loser_cre} died from the defeat.")
        return "\n".join(lines)
    else:
        winner = st.user_creature["name"] if summary.get("player_won") else st.opp_creature["name"]
        lines = [
            f"**Battle Result** — {trainer_name}'s **{st.user_creature['name']}** vs **{st.opp_creature['name']}** (Tier {st.tier})",
            f"🏅 Winner: **{winner}**",
            f"💰 Payout: **+{summary.get('payout', 0)}** cash to {trainer_name}",
        ]
        wins = summary.get("wins")
        if wins is not None:
            lines.append(f"📈 Progress: Tier {st.tier} wins = {wins}/5")
        if summary.get("unlocked_now"):
            lines.append(f"🔓 **Tier {st.tier+1} unlocked!**")
        gained = summary.get("gained_stat")
        if gained:
            lines.append(f"✨ Victory bonus: **+1 {gained}** to {st.user_creature['name']}")
        if summary.get("died"):
            lines.append(f"💀 {st.user_creature['name']} died and was removed from your stable (kept on leaderboard).")
        return "\n".join(lines)


async def _backfill_personalities():
    """
    Assign a random personality to any existing creature lacking one.
    Uses the same odds as spawn.
    """
    pool = await db_pool()
    rows = await pool.fetch("SELECT id FROM creatures WHERE personality IS NULL")
    if not rows:
        logger.info("No personality backfill needed.")
        return
    for r in rows:
        pid = int(r["id"])
        p = choose_personality()
        try:
            await pool.execute("UPDATE creatures SET personality=$1 WHERE id=$2", json.dumps(p), pid)
        except Exception as e:
            logger.warning("Failed to assign personality to creature %s: %s", pid, e)
    logger.info("Backfilled personalities for %d creatures.", len(rows))


# ─── Personality helpers ─────────────────────────────────────
def _parse_personality(raw):
    """
    Accepts a DB value that may be a dict, JSON string, or None.
    Returns a dict with keys: name (str), stats (list[str]) — or {}.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    # Some drivers return JSON/JSONB columns as text
    try:
        obj = json.loads(raw) if isinstance(raw, str) else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

# ─── Bot events ──────────────────────────────────────────────
@bot.event
async def setup_hook():

    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    # Catch up missed daily trainer points immediately on startup
    try:
        await _catch_up_trainer_points_now()
    except Exception as e:
        logger.warning("Trainer-point catch-up failed on startup: %s", e)

    await _backfill_creature_records()

    # Backfill personalities for existing creatures
    try:
        await _backfill_personalities()
    except Exception as e:
        logger.warning("Personality backfill failed on startup: %s", e)

    if GUILD_ID:
        guild_obj = discord.Object(id=int(GUILD_ID))
        # Copy all defined commands to the guild and sync there
        bot.tree.copy_global_to(guild=guild_obj)
        await bot.tree.sync(guild=guild_obj)
        # IMPORTANT: Clear global commands and sync empty to remove duplicates in suggestions
        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        logger.info("Synced to guild %s and cleared global commands.", GUILD_ID)
    else:
        await bot.tree.sync()
        logger.info("Synced globally")

    for loop in (distribute_cash, distribute_points, regenerate_hp, update_leaderboard_periodic, update_shop_periodic):
        if not loop.is_running():
            loop.start()

    chan_id = await _get_leaderboard_channel_id()
    if chan_id:
        await _get_or_create_leaderboard_message(chan_id)
        await _get_or_create_pvp_leaderboard_message(chan_id)
        await update_leaderboard_now(reason="startup")
    else:
        logger.info("No leaderboard channel configured yet. Use /setleaderboardchannel in the desired channel or set LEADERBOARD_CHANNEL_ID.")

    shop_chan = await _get_shop_channel_id()
    if shop_chan:
        await _get_or_create_shop_message(shop_chan)
        await update_shop_now(reason="startup")
    else:
        logger.info("No shop channel configured yet. Use /setshopchannel in the desired channel or set SHOP_CHANNEL_ID.")

    controls_chan = await _get_controls_channel_id()
    if controls_chan:
        await _get_or_create_controls_message(controls_chan)
    else:
        logger.info("No controls channel configured yet. Use /setcontrolchannel in the desired channel or set CONTROLS_CHANNEL_ID.")

@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)
    try:
        # No global sync here; setup_hook already handled guild/global registration and cleanup.
        synced = []
        logger.info("Ready. (%d commands)", len(synced))
    except Exception as e:
        logger.exception("Error during on_ready: %s", e)

# ─── Admin helper resolvers ─────────────────────────────────
async def _extract_user_id_from_mention_or_id(s: str) -> Optional[int]:
    """
    Accepts formats: <@123>, <@!123>, raw numeric id '123'.
    Returns int user_id or None.
    """
    try:
        s = s.strip()
        m = re.match(r"<@!?(\d+)>", s)
        if m:
            return int(m.group(1))
        if s.isdigit():
            return int(s)
        return None
    except Exception:
        return None

async def _find_single_trainer_id_by_display_name(name_like: str) -> Optional[int]:
    """
    Try to resolve a single trainer by display_name (case-insensitive).
    Prefers exact (ILIKE) match; if multiple, returns None (ambiguous).
    """
    pool = await db_pool()
    # First try exact case-insensitive match
    rows = await pool.fetch(
        "SELECT user_id FROM trainers WHERE display_name ILIKE $1",
        name_like
    )
    if not rows:
        # Try wildcard contains search
        rows = await pool.fetch(
            "SELECT user_id FROM trainers WHERE display_name ILIKE $1",
            f"%{name_like}%"
        )
    if not rows:
        return None
    if len(rows) == 1:
        return int(rows[0]["user_id"])
    # Ambiguous
    return None

# ─── Slash commands ─────────────────────────────────────────

@bot.tree.command(description="(Admin) Set this channel as the live leaderboard channel")
async def setleaderboardchannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute(
        """
        INSERT INTO leaderboard_messages(channel_id, message_id, pvp_message_id)
        VALUES ($1, NULL, NULL)
        ON CONFLICT (channel_id) DO NOTHING
        """,
        inter.channel.id,
    )

    await inter.response.send_message(f"Leaderboard channel set to {inter.channel.mention}. Initializing…", ephemeral=True)
    await _get_or_create_leaderboard_message(inter.channel.id)
    await _get_or_create_pvp_leaderboard_message(inter.channel.id)
    await update_leaderboard_now(reason="admin_set_channel")

@bot.tree.command(description="(Admin) Set this channel as the creature shop channel")
async def setshopchannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute("DELETE FROM shop_messages")
    await pool.execute(
        "INSERT INTO shop_messages(channel_id, message_id) VALUES($1, NULL)",
        inter.channel.id,
    )
    await inter.response.send_message(f"Shop channel set to {inter.channel.mention}. Initializing…", ephemeral=True)
    await _get_or_create_shop_message(inter.channel.id)
    await update_shop_now(reason="admin_set_channel")
@bot.tree.command(description="(Admin) Set this channel as the controls channel")
async def setcontrolchannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute("DELETE FROM controls_messages")
    await pool.execute(
        "INSERT INTO controls_messages(channel_id, message_id) VALUES($1, NULL)",
        inter.channel.id,
    )
    await inter.response.send_message(f"Controls channel set to {inter.channel.mention}. Initializing…", ephemeral=True)
    await _get_or_create_controls_message(inter.channel.id)
@bot.tree.command(description="(Admin) Set this channel as the Encyclopedia channel")
async def setencyclopediachannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute("""
        INSERT INTO encyclopedia_channel(channel_id)
        VALUES ($1)
        ON CONFLICT (channel_id) DO UPDATE SET channel_id = EXCLUDED.channel_id
    """, inter.channel.id)

    await inter.response.send_message(f"Encyclopedia channel set to {inter.channel.mention}.", ephemeral=True)

@bot.tree.command(description="Register as a trainer")
async def register(inter: discord.Interaction):
    
    pool = await db_pool()
    if await pool.fetchval("SELECT 1 FROM trainers WHERE user_id=$1", inter.user.id):
        return await inter.response.send_message("Already registered!", ephemeral=True)

    await pool.execute(
        "INSERT INTO trainers(user_id, cash, trainer_points, facility_level, display_name) VALUES($1,$2,$3,$4,$5)",
        inter.user.id, 20000, 5, 1,
        (getattr(inter.user, 'global_name', None) or getattr(inter.user, 'display_name', None) or inter.user.name)
    )

    # Try to grant the 'Testers' role on registration
    try:
        guild = inter.guild
        # Fallback to configured guild if invoked outside a guild context
        if guild is None and GUILD_ID:
            try:
                gid = int(GUILD_ID)
                guild = (bot.get_guild(gid) or await bot.fetch_guild(gid))
            except Exception:
                guild = None

        if guild is not None:
            # Resolve the role by name (case-insensitive)
            role = None
            for r in getattr(guild, "roles", []):
                if str(r.name).lower() == "testers":
                    role = r
                    break

            if role is not None:
                # Resolve the member object robustly without requiring privileged member intent
                member = inter.user if isinstance(inter.user, discord.Member) else None
                if member is None:
                    try:
                        member = guild.get_member(inter.user.id) or await guild.fetch_member(inter.user.id)
                    except Exception:
                        member = None

                if member is not None:
                    try:
                        await member.add_roles(role, reason="Auto-grant 'Testers' on /register")
                    except discord.Forbidden:
                        logger.warning("Missing permissions to assign 'Testers' role.")
                    except Exception as e:
                        logger.warning("Failed to add 'Testers' role: %s", e)
                else:
                    logger.warning("Could not resolve member %s to add 'Testers' role.", inter.user.id)
            else:
                logger.warning("Role 'Testers' not found in guild %s.", getattr(guild, 'id', 'unknown'))
        else:
            logger.warning("No guild context available to add 'Testers' role for user %s.", inter.user.id)
    except Exception as e:
        logger.warning("Unexpected error while assigning 'Testers' role: %s", e)

    await inter.response.send_message(
        "Profile created! You received 20 000 cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a new creature egg (10 000 cash)")
async def spawn(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    can_spawn = await enforce_creature_cap(inter)
    if not can_spawn:
        return
    if row["cash"] < 10_000:
        return await inter.response.send_message("Not enough cash.", ephemeral=True)

    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", inter.user.id
    )
    await inter.response.defer(thinking=True)

    rarity = spawn_rarity()
    meta = await generate_creature_meta(rarity)
    stats = allocate_stats(rarity)
    ovr = int(sum(stats.values()))
    max_hp = stats["HP"] * 5

    personality = choose_personality()

    rec = await (await db_pool()).fetchrow(
        "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats,current_hp,personality)"
        " VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING id",
        inter.user.id, meta["name"], rarity, meta["descriptors"], json.dumps(stats), max_hp, json.dumps(personality)
    )
    await _ensure_record(inter.user.id, rec["id"], meta["name"], ovr)

    embed = discord.Embed(
        title=f"{meta['name']} ({rarity})",
        description="Descriptors: " + ", ".join(meta["descriptors"])
    )
    for s, v in stats.items():
        embed.add_field(name=s, value=str(v*5 if s == "HP" else v))
    try:
        _p = personality
        pstats = '+'.join(_p.get('stats', []))
        embed.add_field(name="Personality", value=f"{_p.get('name','?')} ({pstats})")
    except Exception:
        pass
    embed.set_footer(text="Legendary spawn chance: 0.5%")
    await inter.followup.send(embed=embed)
    asyncio.create_task(update_leaderboard_now(reason="spawn"))
@bot.tree.command(description="List your creatures")
async def creatures(inter: discord.Interaction):
    if not await ensure_registered(inter):
        return
    await inter.response.defer(ephemeral=True)
    rows = await (await db_pool()).fetch(
        "SELECT id,name,rarity,descriptors,stats,current_hp,personality FROM creatures "
        "WHERE owner_id=$1 ORDER BY id", inter.user.id
    )
    if not rows:
        return await inter.response.send_message("You own no creatures.", ephemeral=True)

    ids = [int(r["id"]) for r in rows]

    glyph_rows = await (await db_pool()).fetch(
        """
        SELECT creature_id,
               COALESCE(MAX(CASE WHEN glyph_unlocked THEN tier ELSE 0 END), 0) AS max_glyph
        FROM creature_progress
        WHERE creature_id = ANY($1::int[])
        GROUP BY creature_id
        """,
        ids
    )
    glyph_map = {int(r["creature_id"]): int(r["max_glyph"] or 0) for r in glyph_rows}

    left_map = await _battles_left_map(ids)

    first = True
    for r in rows:
        st = json.loads(r["stats"])
        desc = ", ".join(r["descriptors"] or []) if (r["descriptors"] or []) else "None"
        personality = _parse_personality(r.get("personality"))
        max_hp = int(st.get("HP", 0)) * 5
        left = int(left_map.get(int(r["id"]), DAILY_BATTLE_CAP))
        g = int(glyph_map.get(int(r["id"]), 0))
        glyph_disp = "-" if g <= 0 else str(g)
        overall = int(st.get("HP", 0) + st.get("AR", 0) + st.get("PATK", 0) + st.get("SATK", 0) + st.get("SPD", 0))

        lines = [
            f"**{r['name']}** ({r['rarity']})",
            f"{desc}",
            f"HP: {r['current_hp']}/{max_hp}",
            f"AR: {st.get('AR', 0)}  PATK: {st.get('PATK', 0)}  SATK: {st.get('SATK', 0)}  SPD: {st.get('SPD', 0)}",
            f"Overall: {overall}  |  Glyph: {glyph_disp}",
            f"Personality: { (personality.get('name') + ' (' + ','.join(personality.get('stats', [])) + ')') if personality else '-' }",
            f"Battles left today: **{left}/{DAILY_BATTLE_CAP}**",
        ]
        msg = "\n".join(lines)

        await inter.followup.send(msg, ephemeral=True)

@bot.tree.command(description="See your creature's lifetime win/loss record")
async def record(inter: discord.Interaction, creature_name: str):
    if not await ensure_registered(inter):
        return
    row = await (await db_pool()).fetchrow("""
        SELECT name, wins, losses, is_dead, died_at
        FROM creature_records
        WHERE owner_id=$1 AND name ILIKE $2
    """, inter.user.id, creature_name)
    if not row:
        return await inter.response.send_message("No record found for that creature name.", ephemeral=True)
    total = row["wins"] + row["losses"]
    wr = (row["wins"] / total * 100.0) if total > 0 else 0.0
    status = "💀 DEAD" if row["is_dead"] else "ALIVE"
    died_line = f"\nDied: {row['died_at']:%Y-%m-%d %H:%M %Z}" if row["is_dead"] and row["died_at"] else ""
    msg = (f"**{row['name']} – Lifetime Record**\n"
           f"Wins: **{row['wins']}** | Losses: **{row['losses']}** | Winrate: **{wr:.1f}%**\n"
           f"Status: **{status}**{died_line}")
    await inter.response.send_message(msg, ephemeral=True)

@bot.tree.command(description="Quickly sell one of your creatures for cash (price depends on rarity)")
async def quicksell(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    c_row = await (await db_pool()).fetchrow(
        "SELECT id, name, rarity FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)
    st = active_battles.get(inter.user.id)
    if st and st.creature_id == c_row["id"]:
        return await inter.response.send_message(
            f"**{c_row['name']}** is currently in a battle. Finish or cancel the battle before selling.",
            ephemeral=True
        )
    rarity = c_row["rarity"]
    price = SELL_PRICES.get(rarity, 0)
    await _ensure_record(inter.user.id, c_row["id"], c_row["name"])
    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM creatures WHERE id=$1", c_row["id"])
            await conn.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", price, inter.user.id)
    await inter.response.send_message(
        f"Sold **{c_row['name']}** ({rarity}) for **{price}** cash.", ephemeral=True
    )
    asyncio.create_task(update_leaderboard_now(reason="quicksell"))
    asyncio.create_task(update_shop_now(reason="quicksell"))


@bot.tree.command(description="List one of your creatures for sale in the shop")
async def sell(inter: discord.Interaction, creature_name: str, price: int):
    if price <= 0:
        return await inter.response.send_message("Price must be positive.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    c_row = await pool.fetchrow(
        "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id,
        creature_name,
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)
    st = active_battles.get(inter.user.id)
    if st and st.creature_id == c_row["id"]:
        return await inter.response.send_message(
            f"**{c_row['name']}** is currently in a battle. Finish or cancel the battle before selling.",
            ephemeral=True,
        )
    listed = await pool.fetchval(
        "SELECT 1 FROM creature_shop WHERE creature_id=$1",
        c_row["id"],
    )
    if listed:
        return await inter.response.send_message(
            f"{c_row['name']} is already listed in the shop.", ephemeral=True
        )
    await pool.execute(
        "INSERT INTO creature_shop(creature_id, price) VALUES($1,$2)",
        c_row["id"],
        price,
    )
    await inter.response.send_message(
        f"Listed **{c_row['name']}** for **{price}** cash in the shop.", ephemeral=True
    )
    asyncio.create_task(update_shop_now(reason="sell"))


@bot.tree.command(description="Withdraw one of your creatures from the shop")
async def withdraw(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    c_row = await pool.fetchrow(
        "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id,
        creature_name,
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)
    st = active_battles.get(inter.user.id)
    if st and st.creature_id == c_row["id"]:
        return await inter.response.send_message(
            f"**{c_row['name']}** is currently in a battle. Finish or cancel the battle before withdrawing.",
            ephemeral=True,
        )
    deleted = await pool.fetchrow(
        "DELETE FROM creature_shop WHERE creature_id=$1 RETURNING price",
        c_row["id"],
    )
    if not deleted:
        return await inter.response.send_message(
            f"{c_row['name']} is not listed in the shop.", ephemeral=True
        )
    await inter.response.send_message(
        f"Withdrew **{c_row['name']}** from the shop (was listed for {deleted['price']} cash).",
        ephemeral=True,
    )
    asyncio.create_task(update_shop_now(reason="withdraw"))


@bot.tree.command(description="Buy a creature from the shop")
async def buy(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    c_row = await pool.fetchrow(
        """
        SELECT c.id, c.name, c.owner_id, c.stats, ct.price,
               COALESCE(t.display_name, c.owner_id::text) AS trainer_name
        FROM creature_shop ct
        JOIN creatures c ON c.id = ct.creature_id
        LEFT JOIN trainers t ON t.user_id = c.owner_id
        WHERE LOWER(c.name) = LOWER($1)
        """,
        creature_name,
    )
    if not c_row:
        return await inter.response.send_message("That creature is not for sale.", ephemeral=True)
    if c_row["owner_id"] == inter.user.id:
        return await inter.response.send_message("You cannot buy your own creature.", ephemeral=True)
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM creatures WHERE owner_id=$1",
        inter.user.id,
    )
    if int(count or 0) >= MAX_CREATURES:
        return await inter.response.send_message(
            f"You already have the maximum of {MAX_CREATURES} creatures.", ephemeral=True
        )
    price = int(c_row["price"])
    if row["cash"] < price:
        return await inter.response.send_message("You don't have enough cash.", ephemeral=True)

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM creature_shop WHERE creature_id=$1",
                c_row["id"],
            )
            await conn.execute(
                "UPDATE creatures SET owner_id=$1 WHERE id=$2",
                inter.user.id,
                c_row["id"],
            )
            await conn.execute(
                "UPDATE trainers SET cash = cash - $1 WHERE user_id=$2",
                price,
                inter.user.id,
            )
            await conn.execute(
                "UPDATE trainers SET cash = cash + $1 WHERE user_id=$2",
                price,
                c_row["owner_id"],
            )

    await _ensure_record(inter.user.id, c_row["id"], c_row["name"])
    await inter.response.send_message(
        f"You bought **{c_row['name']}** from {c_row['trainer_name']} for {price} cash.",
        ephemeral=True,
    )
    asyncio.create_task(update_shop_now(reason="buy"))
    asyncio.create_task(update_leaderboard_now(reason="buy"))

class SellModal(discord.ui.Modal, title="Sell Creature"):
    creature_name = discord.ui.TextInput(label="Creature Name")
    price = discord.ui.TextInput(label="Price")

    async def on_submit(self, interaction: discord.Interaction):
        try:
            price_val = int(self.price.value)
        except ValueError:
            return await interaction.response.send_message("Price must be an integer.", ephemeral=True)
        await sell.callback(interaction, self.creature_name.value, price_val)


class WithdrawModal(discord.ui.Modal, title="Withdraw Listing"):
    creature_name = discord.ui.TextInput(label="Creature Name")

    async def on_submit(self, interaction: discord.Interaction):
        await withdraw.callback(interaction, self.creature_name.value)


class BuyModal(discord.ui.Modal, title="Buy Creature"):
    creature_name = discord.ui.TextInput(label="Creature Name")

    async def on_submit(self, interaction: discord.Interaction):
        await buy.callback(interaction, self.creature_name.value)


class ShopView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Sell", style=discord.ButtonStyle.blurple)
    async def sell_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(SellModal())

    @discord.ui.button(label="Withdraw", style=discord.ButtonStyle.gray)
    async def withdraw_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(WithdrawModal())

    @discord.ui.button(label="Buy", style=discord.ButtonStyle.green)
    async def buy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(BuyModal())

async def _rename_creature(inter: discord.Interaction, creature_name: str, new_name: str):
    """Core logic for renaming a creature."""
    row = await ensure_registered(inter)
    if not row:
        return
    new_name = new_name.strip().title()
    if not new_name:
        return await inter.response.send_message("New name cannot be empty.", ephemeral=True)
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id, creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        st = active_battles.get(inter.user.id)
        if st and st.creature_id == c_row["id"]:
            return await inter.response.send_message(
                f"**{c_row['name']}** is currently in a battle. Finish or cancel the battle before renaming.",
                ephemeral=True,
            )
        exists = await conn.fetchrow(
            "SELECT 1 FROM creatures WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
            inter.user.id, new_name,
        )
        if exists:
            return await inter.response.send_message(
                "You already have a creature with that name.", ephemeral=True
            )
        rec_exist = await conn.fetchrow(
            "SELECT creature_id FROM creature_records WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
            inter.user.id, new_name,
        )
        if rec_exist and int(rec_exist["creature_id"] or 0) != c_row["id"]:
            return await inter.response.send_message(
                "You already have a record with that name; choose another.", ephemeral=True
            )
        async with conn.transaction():
            await conn.execute("UPDATE creatures SET name=$1 WHERE id=$2", new_name, c_row["id"])
            await conn.execute(
                "UPDATE creature_records SET name=$1 WHERE owner_id=$2 AND creature_id=$3",
                new_name, inter.user.id, c_row["id"],
            )
    await _ensure_record(inter.user.id, c_row["id"], new_name)
    await inter.response.send_message(
        f"Renamed **{c_row['name']}** to **{new_name}**.", ephemeral=True
    )
    asyncio.create_task(update_leaderboard_now(reason="rename"))
    asyncio.create_task(update_shop_now(reason="rename"))


@bot.tree.command(description="Rename one of your creatures")
async def rename(inter: discord.Interaction):
    await inter.response.send_modal(RenameModal())

@bot.tree.command(description="Show glyphs and tier progress for a creature")
async def glyphs(inter: discord.Interaction, creature_name: str):
    if not await ensure_registered(inter):
        return
    c_row = await (await db_pool()).fetchrow(
        "SELECT id,name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)
    pool = await db_pool()
    progress: Dict[int, Tuple[int, bool]] = {}
    async with pool.acquire() as conn:
        for t in range(1, 10):
            row = await _get_progress(conn, c_row["id"], t)
            wins = row["wins"] if row else 0
            glyph = (row["glyph_unlocked"] if row else False)
            progress[t] = (wins, glyph)
    max_tier = await _max_unlocked_tier(c_row["id"])
    lines = [f"**{c_row['name']} – Glyphs & Progress**"]
    for t in range(1, 10):
        wins, glyph = progress[t]
        lines.append(f"• Tier {t}: Wins {wins}/5 | Glyph: {'✅' if glyph else '❌'}")
    lines += ["", f"**Unlocked Tiers:** 1..{max_tier}"]
    if max_tier < 9:
        need_prev = max_tier
        wins_prev, _ = progress[need_prev]
        lines.append(
            f"Win **5 battles at Tier {need_prev}** to unlock **Tier {need_prev+1}** "
            f"(current: {wins_prev}/5)."
        )
    await inter.response.send_message("\n".join(lines), ephemeral=True)

@bot.tree.command(description="(Admin) Reimburse daily battle usage for a creature by name")
async def dailylimit(inter: discord.Interaction, creature_name: str, number: int):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if number <= 0:
        return await inter.response.send_message("Number must be positive.", ephemeral=True)

    pool = await db_pool()
    rows = await pool.fetch(
        "SELECT id, name, owner_id FROM creatures WHERE name ILIKE $1",
        creature_name
    )
    if not rows:
        rows = await pool.fetch(
            "SELECT id, name, owner_id FROM creatures WHERE name ILIKE $1",
            f"%{creature_name}%"
        )
    if not rows:
        return await inter.response.send_message("No creature found by that name.", ephemeral=True)

    exact = [r for r in rows if str(r["name"]).lower() == creature_name.lower()]
    if len(rows) > 1 and len(exact) == 1:
        rows = exact
    if len(rows) != 1:
        preview_names = []
        for r in rows[:8]:
            owner_name = await _resolve_trainer_name_from_db(int(r["owner_id"])) or str(r["owner_id"])
            preview_names.append(f"{r['name']} (owner {owner_name})")
        return await inter.response.send_message(
            "Multiple creatures match that name. Be more specific.\nMatches: " + ", ".join(preview_names),
            ephemeral=True
        )

    creature_id = int(rows[0]["id"])
    cname = str(rows[0]["name"])
    owner_id = int(rows[0]["owner_id"])
    owner_name = await _resolve_trainer_name_from_db(owner_id) or str(owner_id)

    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row = await conn.fetchrow(
                "SELECT count FROM battle_caps WHERE creature_id=$1 AND day=$2 FOR UPDATE",
                creature_id, day
            )
            if not row:
                return await inter.response.send_message(
                    f"No battle usage recorded today for **{cname}** (owner {owner_name}); nothing to reimburse.",
                    ephemeral=True
                )
            current = int(row["count"])
            new_count = max(0, current - number)
            if new_count == 0:
                await conn.execute(
                    "DELETE FROM battle_caps WHERE creature_id=$1 AND day=$2",
                    creature_id, day
                )
            else:
                await conn.execute(
                    "UPDATE battle_caps SET count=$3 WHERE creature_id=$1 AND day=$2",
                    creature_id, day, new_count
                )

    await inter.response.send_message(
        f"Reimbursed **{number}** daily usage for **{cname}** (owner {owner_name}).",
        ephemeral=True
    )

@bot.tree.command(description="Battle one of your creatures vs. a tiered opponent")
async def battle(inter: discord.Interaction, creature_name: str, tier: int):
    if tier not in TIER_EXTRAS:
        return await inter.response.send_message("Invalid tier (1-9).", ephemeral=True)
    if inter.user.id in active_battles:
        return await inter.response.send_message("You already have an active battle in progress.", ephemeral=True)
    if not await ensure_registered(inter):
        return

    # Global battle gate: only one battle may run at a time
    global current_battler_id
    if battle_lock.locked() and (current_battler_id is not None) and (current_battler_id != inter.user.id):
        await inter.response.send_message(
            f"⚔ A battle is already in progress by <@{current_battler_id}>. Please try again shortly.",
            ephemeral=True
        )
        return
    c_row = await (await db_pool()).fetchrow(
        "SELECT id,name,stats,current_hp FROM creatures "
        "WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    listed = await (await db_pool()).fetchval(
        "SELECT 1 FROM creature_shop WHERE creature_id=$1",
        c_row["id"],
    )
    if listed:
        return await inter.response.send_message(
            f"{c_row['name']} is listed in the shop and cannot battle.",
            ephemeral=True,
        )

    stats = json.loads(c_row["stats"])
    max_hp = stats["HP"] * 5
    if c_row["current_hp"] <= 0:
        return await inter.response.send_message(f"{c_row['name']} has fainted and needs healing.", ephemeral=True)

    allowed_tier = await _max_unlocked_tier(c_row["id"])
    if tier > allowed_tier:
        need_prev = tier - 1
        wins_prev = await _get_wins_for_tier(c_row["id"], need_prev)
        return await inter.response.send_message(
            f"Tier {tier} is locked for **{c_row['name']}**. Current unlock: **1..{allowed_tier}**. "
            f"You need **5 wins at Tier {need_prev}** to unlock Tier {tier} (current: {wins_prev}/5).",
            ephemeral=True
        )

    allowed, count = await _can_start_battle_and_increment(c_row["id"])
    if not allowed:
        return await inter.response.send_message(
            f"Daily battle cap reached for **{c_row['name']}**: {DAILY_BATTLE_CAP}/{DAILY_BATTLE_CAP} used. "
            "Try again after midnight Europe/London.", ephemeral=True
        )

    await battle_lock.acquire()
    current_battler_id = inter.user.id
    try:
        await inter.response.defer(thinking=True)

        user_cre = {"name": c_row["name"], "stats": stats}
        rarity = rarity_for_tier(tier)
        name_only = await generate_creature_name(rarity)
        extra = random.randint(*TIER_EXTRAS[tier])
        opp_stats = allocate_stats(rarity, extra)
        opp_cre = {"name": name_only, "stats": opp_stats}
        st = BattleState(
            inter.user.id, c_row["id"], tier,
            user_cre, c_row["current_hp"], max_hp,
            opp_cre, opp_stats["HP"] * 5, opp_stats["HP"] * 5,
            logs=[]
        )
        active_battles[inter.user.id] = st
        st.logs += [
            f"Battle start! Tier {tier} (+{extra} pts) — Daily battle use for {user_cre['name']}: {count}/{DAILY_BATTLE_CAP}",
            f"{user_cre['name']} vs {opp_cre['name']}",
            f"Opponent rarity (tier table) → {rarity}",
            "",
            "Your creature:",
            stat_block(user_cre["name"], st.user_current_hp, st.user_max_hp, stats),
            "Opponent:",
            stat_block(opp_cre["name"], st.opp_max_hp, st.opp_max_hp, opp_stats),
            "",
            "Rules: Action weights A/Ag/Sp/Df = 38/22/22/18, Aggressive +25% dmg, Special ignores AR, "
            "AR softened (halved), extra swing at 1.5× SPD, +10% global damage every 10 rounds.",
            f"Daily cap: Each creature can start at most {DAILY_BATTLE_CAP} battles per Europe/London day.",
        ]
        max_tier = await _max_unlocked_tier(c_row["id"])
        st.logs.append(f"Tier gate: {user_cre['name']} can currently queue Tier 1..{max_tier}.")

    
        # Public start (concise)
        start_public = (
            f"**Battle Start** — {user_cre['name']} vs {opp_cre['name']} (Tier {tier})\n"
            f"Opponent rarity: **{rarity}**"
        )
        await inter.followup.send(start_public, ephemeral=False)

        # Drip-feed rounds privately
        st.next_log_idx = len(st.logs)
        while st.user_current_hp > 0 and st.opp_current_hp > 0:
            simulate_round(st)
            new_logs = st.logs[st.next_log_idx:]
            st.next_log_idx = len(st.logs)
            if new_logs:
                await send_chunks(inter, "\n".join(new_logs), ephemeral=True)
                await asyncio.sleep(0.2)

        await (await db_pool()).execute(
            "UPDATE creatures SET current_hp=$1 WHERE id=$2",
            max(st.user_current_hp, 0), st.creature_id
        )

        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            winner = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
            st.logs.append(f"Winner: {winner}")
            summary = await finalize_battle(inter, st)
            active_battles.pop(inter.user.id, None)
            trainer_name = await _resolve_trainer_name_from_db(inter.user.id) or (getattr(inter.user, 'display_name', None) or inter.user.name)
            pending = st.logs[st.next_log_idx:]
            st.next_log_idx = len(st.logs)
            if pending:
                await send_chunks(inter, "\n".join(pending), ephemeral=True)
                await asyncio.sleep(0.35)
            await inter.followup.send(format_public_battle_summary(st, summary, trainer_name), ephemeral=False)
    finally:
        try:
            current_battler_id = None
            if battle_lock.locked():
                battle_lock.release()
        except Exception:
            pass

@bot.tree.command(description="Challenge another trainer to a PvP battle")
async def pvp(inter: discord.Interaction, opponent: discord.User, wager: int):
    if opponent.id == inter.user.id:
        return await inter.response.send_message("You cannot challenge yourself.", ephemeral=True)
    if wager <= 0:
        return await inter.response.send_message("Wager must be positive.", ephemeral=True)
    if inter.user.id in active_battles or opponent.id in active_battles:
        return await inter.response.send_message("One of the participants is already in a battle.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    opp_tr = await pool.fetchrow("SELECT cash FROM trainers WHERE user_id=$1", opponent.id)
    if not opp_tr:
        return await inter.response.send_message("Opponent is not registered.", ephemeral=True)
    if row["cash"] < wager:
        return await inter.response.send_message("You don't have enough cash for this wager.", ephemeral=True)
    if opp_tr["cash"] < wager:
        return await inter.response.send_message(
            f"Opponent only has {opp_tr['cash']} cash; reduce the wager below that.", ephemeral=True
        )
    now = datetime.now(timezone.utc)
    cd_rows = await pool.fetch(
        "SELECT user_id, last_battle_at FROM pvp_records WHERE user_id = ANY($1)",
        [[inter.user.id, opponent.id]],
    )
    cd = {r["user_id"]: r["last_battle_at"] for r in cd_rows}
    for uid in (inter.user.id, opponent.id):
        last = cd.get(uid)
        if last and now - last < timedelta(hours=12):
            who = "You" if uid == inter.user.id else "Opponent"
            return await inter.response.send_message(
                f"{who} must wait before engaging in another PvP battle.", ephemeral=True
            )
    my_creatures = await pool.fetch(
        "SELECT id,name,stats FROM creatures WHERE owner_id=$1 AND NOT EXISTS (SELECT 1 FROM creature_shop cs WHERE cs.creature_id = creatures.id)",
        inter.user.id,
    )
    if not my_creatures:
        return await inter.response.send_message("You have no creatures.", ephemeral=True)
    my_options = [discord.SelectOption(label=r["name"], value=str(r["id"])) for r in my_creatures]

    class ChallengerSelect(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)
            self.selected = None

        @discord.ui.select(placeholder="Select your creature", options=my_options)
        async def select_callback(self, interaction: discord.Interaction, select: discord.ui.Select):
            if interaction.user.id != inter.user.id:
                return await interaction.response.send_message("This menu isn't for you.", ephemeral=True)
            self.selected = next(r for r in my_creatures if str(r["id"]) == select.values[0])
            await interaction.response.defer()
            self.stop()

    my_view = ChallengerSelect()
    await inter.response.send_message("Choose your creature:", view=my_view, ephemeral=True)
    await my_view.wait()
    if not my_view.selected:
        return
    c_row = my_view.selected
    stats = json.loads(c_row["stats"])
    max_hp = stats["HP"] * 5
    challenger_ovr = sum(stats.values())
    opp_creatures = await pool.fetch(
        "SELECT id,name,stats FROM creatures WHERE owner_id=$1 AND NOT EXISTS (SELECT 1 FROM creature_shop cs WHERE cs.creature_id = creatures.id)",
        opponent.id,
    )
    opp_creatures = [
        r
        for r in opp_creatures
        if abs(sum(json.loads(r["stats"]).values()) - challenger_ovr) <= 50
    ]
    if not opp_creatures:
        return await inter.followup.send(
            "Opponent has no creature within 50 OVR of your creature.", ephemeral=True
        )
    options = [discord.SelectOption(label=r["name"], value=str(r["id"])) for r in opp_creatures]

    class OpponentSelectView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)
            self.selected = None

        @discord.ui.select(placeholder="Select opponent's creature", options=options)
        async def select_callback(self, interaction: discord.Interaction, select: discord.ui.Select):
            if interaction.user.id != inter.user.id:
                return await interaction.response.send_message("This menu isn't for you.", ephemeral=True)
            self.selected = next(r for r in opp_creatures if str(r["id"]) == select.values[0])
            await interaction.response.send_message(
                f"Challenged {opponent.mention}'s {self.selected['name']}!", ephemeral=True
            )
            self.stop()

    select_view = OpponentSelectView()
    await inter.followup.send(
        "Choose an opponent creature to challenge:", view=select_view, ephemeral=True
    )
    await select_view.wait()
    if not select_view.selected:
        return
    opp_cre = select_view.selected
    opp_stats = json.loads(opp_cre["stats"])
    opp_max_hp = opp_stats["HP"] * 5
    opp_name = await _resolve_trainer_name_from_db(opponent.id) or (
        getattr(opponent, 'display_name', None) or opponent.name
    )

    class PvPChallengeView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=3600)

        async def on_timeout(self):
            try:
                if self.message:
                    await self.message.delete()
            except Exception:
                pass

        @discord.ui.button(label="Accept", style=discord.ButtonStyle.success)
        async def accept(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != opponent.id:
                return await interaction.response.send_message("This challenge isn't for you.", ephemeral=True)
            if inter.user.id in active_battles or opponent.id in active_battles:
                return await interaction.response.send_message("One of the participants is already in a battle.", ephemeral=True)
            global current_battler_id
            if battle_lock.locked() and (current_battler_id not in (inter.user.id, opponent.id)):
                return await interaction.response.send_message(
                    "⚔ A battle is already in progress. Please try again later.", ephemeral=True
                )
            await battle_lock.acquire()
            current_battler_id = inter.user.id
            try:
                await interaction.response.defer(thinking=True)
                await pool.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", max_hp, c_row["id"])
                await pool.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", opp_max_hp, opp_cre["id"])
                st = BattleState(
                    inter.user.id, c_row["id"], 0,
                    {"name": c_row["name"], "stats": stats}, max_hp, max_hp,
                    {"name": opp_cre["name"], "stats": opp_stats}, opp_max_hp, opp_max_hp,
                    logs=[], is_pvp=True, opp_user_id=opponent.id, opp_creature_id=opp_cre["id"], wager=wager, opp_trainer_name=opp_name
                )
                active_battles[inter.user.id] = st
                active_battles[opponent.id] = st
                st.logs += [
                    f"PvP battle start! Wager {wager} cash.",
                    f"{st.user_creature['name']} vs {st.opp_creature['name']}",
                    "",
                    "Challenger:",
                    stat_block(st.user_creature['name'], st.user_current_hp, st.user_max_hp, st.user_creature['stats']),
                    "Opponent:",
                    stat_block(st.opp_creature['name'], st.opp_current_hp, st.opp_max_hp, st.opp_creature['stats']),
                ]
                st.next_log_idx = len(st.logs)
                await interaction.followup.send(
                    f"**PvP Battle Start** — {st.user_creature['name']} vs {st.opp_creature['name']} (Wager {wager})",
                    ephemeral=False
                )
                while st.user_current_hp > 0 and st.opp_current_hp > 0:
                    simulate_round(st)
                    new_logs = st.logs[st.next_log_idx:]
                    st.next_log_idx = len(st.logs)
                    if new_logs:
                        await send_chunks(interaction, "\n".join(new_logs), ephemeral=False)
                        await asyncio.sleep(0.2)
                await pool.execute(
                    "UPDATE creatures SET current_hp=$1 WHERE id=$2",
                    max(st.user_current_hp, 0), st.creature_id
                )
                await pool.execute(
                    "UPDATE creatures SET current_hp=$1 WHERE id=$2",
                    max(st.opp_current_hp, 0), st.opp_creature_id
                )
                winner = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
                st.logs.append(f"Winner: {winner}")
                summary = await finalize_battle(interaction, st)
                active_battles.pop(inter.user.id, None)
                active_battles.pop(opponent.id, None)
                trainer_name = await _resolve_trainer_name_from_db(inter.user.id) or (
                    getattr(inter.user, 'display_name', None) or inter.user.name
                )
                pending = st.logs[st.next_log_idx:]
                st.next_log_idx = len(st.logs)
                if pending:
                    await send_chunks(interaction, "\n".join(pending), ephemeral=False)
                    await asyncio.sleep(0.35)
                await interaction.followup.send(
                    format_public_battle_summary(st, summary, trainer_name), ephemeral=False
                )
            finally:
                try:
                    current_battler_id = None
                    if battle_lock.locked():
                        battle_lock.release()
                except Exception:
                    pass

        @discord.ui.button(label="Decline", style=discord.ButtonStyle.danger)
        async def decline(self, interaction: discord.Interaction, button: discord.ui.Button):
            if interaction.user.id != opponent.id:
                return await interaction.response.send_message("This challenge isn't for you.", ephemeral=True)
            await interaction.response.send_message("Challenge declined.", ephemeral=True)

    view = PvPChallengeView()
    challenge_msg = await inter.followup.send(
        f"{opponent.mention}, {inter.user.mention} challenges you to a PvP battle wagering {wager} cash with your {opp_cre['name']}! Their {c_row['name']} has an OVR of {int(challenger_ovr)}.",
        view=view
    )
    view.message = challenge_msg

@bot.tree.command(description="Check your cash")
async def cash(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        await inter.response.send_message(f"You have {row['cash']} cash.", ephemeral=True)

@bot.tree.command(description="(Admin) Add cash to a player by name/mention/id, or 'all'")
async def cashadd(inter: discord.Interaction, amount: int, target: str = "me"):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if amount <= 0:
        return await inter.response.send_message("Positive amounts only.", ephemeral=True)

    tgt = (target or "").strip()
    if tgt.lower() in ("me", "self"):
        user_id = inter.user.id
    elif tgt.lower() == "all":
        updated = await (await db_pool()).execute("UPDATE trainers SET cash = cash + $1", amount)
        try:
            count = int(str(updated).split()[-1])
        except Exception:
            count = 0
        return await inter.response.send_message(f"Added {amount} cash to **all** trainers ({count} rows).", ephemeral=True)
    else:
        user_id = await _extract_user_id_from_mention_or_id(tgt)
        if user_id is None:
            user_id = await _find_single_trainer_id_by_display_name(tgt)
        if user_id is None:
            return await inter.response.send_message(
                "Couldn't uniquely resolve that trainer. Try a mention, raw ID, or an exact display name.", ephemeral=True
            )

    pool = await db_pool()
    exists = await pool.fetchval("SELECT 1 FROM trainers WHERE user_id=$1", user_id)
    if not exists:
        return await inter.response.send_message("That user isn't registered.", ephemeral=True)

    await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", amount, user_id)
    name = await _resolve_trainer_name_from_db(user_id) or str(user_id)
    await inter.response.send_message(f"Added **{amount}** cash to **{name}**.", ephemeral=True)

@bot.tree.command(description="(Admin) Add trainer points to a player by name/mention/id, or 'all'")
async def trainerpointsadd(inter: discord.Interaction, amount: int, target: str = "me"):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if amount <= 0:
        return await inter.response.send_message("Positive amounts only.", ephemeral=True)

    tgt = (target or "").strip()
    if tgt.lower() in ("me", "self"):
        user_id = inter.user.id
    elif tgt.lower() == "all":
        updated = await (await db_pool()).execute("UPDATE trainers SET trainer_points = trainer_points + $1", amount)
        try:
            count = int(str(updated).split()[-1])
        except Exception:
            count = 0
        return await inter.response.send_message(f"Added {amount} trainer points to **all** trainers ({count} rows).", ephemeral=True)
    else:
        user_id = await _extract_user_id_from_mention_or_id(tgt)
        if user_id is None:
            user_id = await _find_single_trainer_id_by_display_name(tgt)
        if user_id is None:
            return await inter.response.send_message(
                "Couldn't uniquely resolve that trainer. Try a mention, raw ID, or an exact display name.", ephemeral=True
            )

    pool = await db_pool()
    exists = await pool.fetchval("SELECT 1 FROM trainers WHERE user_id=$1", user_id)
    if not exists:
        return await inter.response.send_message("That user isn't registered.", ephemeral=True)

    await pool.execute("UPDATE trainers SET trainer_points = trainer_points + $1 WHERE user_id=$2", amount, user_id)
    name = await _resolve_trainer_name_from_db(user_id) or str(user_id)
    await inter.response.send_message(f"Added **{amount}** trainer points to **{name}**.", ephemeral=True)

@bot.tree.command(description="Check your trainer points")
async def trainerpoints(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        level = row["facility_level"]
        bonus = facility_bonus(level)
        daily = 5 + bonus
        await inter.response.send_message(
            f"You have {row['trainer_points']} points. "
            f"Facility Level {level} ({FACILITY_LEVELS[level]['name']}) gives +{bonus} extra per day "
            f"(total {daily}/day).",
            ephemeral=True
        )

async def _train_creature(inter: discord.Interaction, creature_name: str, stat: str, increase: int):
    """Core logic for training a creature."""
    stat = stat.upper()
    if stat not in PRIMARY_STATS:
        return await inter.response.send_message(
            f"Stat must be one of {', '.join(PRIMARY_STATS)}.", ephemeral=True
        )
    if increase <= 0:
        return await inter.response.send_message("Increase must be positive.", ephemeral=True)

    row = await ensure_registered(inter)
    if not row or row["trainer_points"] < increase:
        return await inter.response.send_message("Not enough trainer points.", ephemeral=True)

    c = await (await db_pool()).fetchrow(
        "SELECT id,name,stats,current_hp,personality FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    listed = await (await db_pool()).fetchval(
        "SELECT 1 FROM creature_shop WHERE creature_id=$1",
        c["id"],
    )
    if listed:
        return await inter.response.send_message(
            f"{c['name']} is listed in the shop and cannot be trained.",
            ephemeral=True,
        )

    stats = json.loads(c["stats"])
    personality = _parse_personality(c.get("personality"))
    pstats = set(personality.get("stats", []))
    mult = 2 if stat in pstats else 1
    effective = increase * mult
    stats[stat] += effective
    new_max_hp = stats["HP"] * 5
    new_cur_hp = c["current_hp"]
    if stat == "HP":
        new_cur_hp += effective * 5
        new_cur_hp = min(new_cur_hp, new_max_hp)

    await (await db_pool()).execute(
        "UPDATE creatures SET stats=$1,current_hp=$2 WHERE id=$3",
        json.dumps(stats), new_cur_hp, c["id"]
    )
    await (await db_pool()).execute(
        "UPDATE trainers SET trainer_points = trainer_points - $1 WHERE user_id=$2",
        increase, inter.user.id
    )
    display_inc = effective * 5 if stat == "HP" else effective
    await inter.response.send_message(
        f"{c['id']} – {creature_name.title()} trained: +{display_inc} {stat}{' (x2 personality bonus)' if mult == 2 else ''}.",
        ephemeral=True
    )


@bot.tree.command(description="Train a creature stat")
async def train(inter: discord.Interaction):
    await inter.response.send_modal(TrainModal())

@bot.tree.command(description="Show and confirm upgrading your training facility")
async def upgrade(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    level = row["facility_level"]
    current = FACILITY_LEVELS[level]
    msg = [
        f"**Your Training Facility**",
        f"Level {level}: **{current['name']}**",
        f"Bonus trainer points/day: +{current['bonus']} (total daily = {daily_trainer_points_for(level)})",
        f"Description: {current['desc']}",
        ""
    ]
    if level >= MAX_FACILITY_LEVEL:
        msg.append("You're already at the **maximum level**. No further upgrades available.")
        return await inter.response.send_message("\n".join(msg), ephemeral=True)

    next_level = level + 1
    nxt = FACILITY_LEVELS[next_level]
    msg += [
        f"**Next Upgrade → Level {next_level}: {nxt['name']}**",
        f"Cost: {nxt['cost']} cash",
        f"New bonus: +{nxt['bonus']} (daily total = {daily_trainer_points_for(next_level)})",
        f"Description: {nxt['desc']}",
        "",
        "Confirm upgrade?"
    ]
    await inter.response.send_message(
        "\n".join(msg),
        ephemeral=True,
        view=UpgradeConfirmView(inter.user.id),
    )

@bot.tree.command(description="Confirm upgrading your training facility (costs cash)")
async def upgradeyes(inter: discord.Interaction):
    await _do_upgrade(inter)


async def _do_upgrade(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    level = row["facility_level"]
    if level >= MAX_FACILITY_LEVEL:
        return await inter.response.send_message(
            "You're already at the maximum facility level.", ephemeral=True
        )

    next_level = level + 1
    cost = FACILITY_LEVELS[next_level]["cost"]

    if row["cash"] < cost:
        return await inter.response.send_message(
            f"Not enough cash. You need {cost} but only have {row['cash']}.",
            ephemeral=True,
        )

    pool = await db_pool()
    await pool.execute(
        "UPDATE trainers SET cash = cash - $1, facility_level = facility_level + 1 WHERE user_id=$2",
        cost,
        inter.user.id,
    )
    new_bonus = FACILITY_LEVELS[next_level]["bonus"]
    await inter.response.send_message(
        f"✅ Upgraded to **Level {next_level} – {FACILITY_LEVELS[next_level]['name']}**!\n",
        f"Your facility now grants **+{new_bonus} trainer points/day** "
        f"(total {daily_trainer_points_for(next_level)}/day).",
        ephemeral=True,
    )


class UpgradeConfirmView(discord.ui.View):
    def __init__(self, user_id: int):
        super().__init__(timeout=None)
        self.user_id = user_id

    @discord.ui.button(label="Yes", style=discord.ButtonStyle.success)
    async def yes(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This isn't your confirmation.", ephemeral=True)
        await _do_upgrade(interaction)
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass

    @discord.ui.button(label="No", style=discord.ButtonStyle.danger)
    async def no(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This isn't your confirmation.", ephemeral=True)
        await interaction.response.send_message("Upgrade canceled.", ephemeral=True)
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass

@bot.tree.command(description="Add one of your creatures to the Encyclopedia")
async def enc(inter: discord.Interaction, creature_name: str):
    # Ensure player exists
    if not await ensure_registered(inter):
        return

    # Find creature by owner/name (case-insensitive)
    c_row = await (await db_pool()).fetchrow(
        "SELECT id, name, rarity, descriptors, stats, current_hp FROM creatures "
        "WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    # Gate: Common/Uncommon cannot be added to the Encyclopedia
    # unless they have >10 total wins OR have unlocked Glyph 3.
    rarity_val = str(c_row["rarity"])
    if rarity_val in ("Common", "Uncommon"):
        pool = await db_pool()
        async with pool.acquire() as conn:
            total_wins = await conn.fetchval(
                "SELECT COALESCE(SUM(wins),0) FROM creature_progress WHERE creature_id=$1",
                c_row["id"]
            )
            glyph3_unlocked = await conn.fetchval(
                "SELECT glyph_unlocked FROM creature_progress WHERE creature_id=$1 AND tier=3",
                c_row["id"]
            )
        if (total_wins or 0) <= 10 and not bool(glyph3_unlocked):
            return await inter.response.send_message(
                f"Encyclopedia entry denied for **{c_row['name']}** ({rarity_val}). "
                "Common/Uncommon creatures must have **>10 total wins** or **Glyph 3 unlocked**.\n"
                f"Current: Wins **{(total_wins or 0)}**, Glyph 3 {'✅' if glyph3_unlocked else '❌'}.",
                ephemeral=True
            )

    # Encyclopedia channel
    enc_chan_id = await _get_encyclopedia_channel_id()
    if not enc_chan_id:
        return await inter.response.send_message(
            "No Encyclopedia channel configured. Ask an admin to run `/setencyclopediachannel` in the target channel.",
            ephemeral=True
        )
    channel = await _ensure_encyclopedia_channel(enc_chan_id)
    if not channel:
        return await inter.response.send_message("Failed to resolve Encyclopedia channel.", ephemeral=True)

    await inter.response.defer(thinking=True, ephemeral=True)

    # Prepare data
    name = c_row["name"]
    rarity = c_row["rarity"]
    traits = c_row["descriptors"] or []
    stats = json.loads(c_row["stats"])
    stats_block = _format_stats_block(stats)

    # GPT bio + image
    bio, image_url, image_bytes = await _gpt_generate_bio_and_image(name, rarity, traits, stats)
    # Build embed
    embed = discord.Embed(
        title=f"{name} — {rarity}",
        description=bio
    )
    embed.add_field(name="Traits", value=", ".join(traits) if traits else "None", inline=False)
    embed.add_field(name="Stats", value=stats_block, inline=False)
    if image_url:
        embed.set_image(url=image_url)
        file_to_send = None
    elif image_bytes:
        embed.set_image(url="attachment://creature.png")
        import io as _io
        file_to_send = discord.File(_io.BytesIO(image_bytes), filename="creature.png")
    else:
        file_to_send = None

    try:
        if file_to_send:
            msg = await channel.send(embed=embed, file=file_to_send)
        else:
            msg = await channel.send(embed=embed)
    except Exception as e:
        logger.error("Failed to post encyclopedia entry: %s", e)
        return await inter.followup.send("Failed to post to the Encyclopedia channel.", ephemeral=True)

    await inter.followup.send(f"Added **{name}** to the Encyclopedia: {msg.jump_url}", ephemeral=True)


# ─── Interactive controls (modals & buttons) ──────────────────

class RenameModal(discord.ui.Modal, title="Rename Creature"):
    creature_name: discord.ui.TextInput = discord.ui.TextInput(label="Current Name")
    new_name: discord.ui.TextInput = discord.ui.TextInput(label="New Name")

    async def on_submit(self, interaction: discord.Interaction):
        await _rename_creature(interaction, self.creature_name.value, self.new_name.value)


class TrainModal(discord.ui.Modal, title="Train Creature"):
    creature_name: discord.ui.TextInput = discord.ui.TextInput(label="Creature Name")
    stat: discord.ui.TextInput = discord.ui.TextInput(label="Stat")
    amount: discord.ui.TextInput = discord.ui.TextInput(label="Increase Amount")

    async def on_submit(self, interaction: discord.Interaction):
        try:
            inc = int(self.amount.value)
        except ValueError:
            return await interaction.response.send_message("Amount must be an integer.", ephemeral=True)
        await _train_creature(interaction, self.creature_name.value, self.stat.value, inc)


class ControlsView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(
        label="Creatures",
        style=discord.ButtonStyle.secondary,
        custom_id="controls_creatures",
    )
    async def btn_creatures(self, interaction: discord.Interaction, button: discord.ui.Button):
        await creatures.callback(interaction)

    @discord.ui.button(
        label="Trainer Points",
        style=discord.ButtonStyle.secondary,
        custom_id="controls_tp",
    )
    async def btn_tp(self, interaction: discord.Interaction, button: discord.ui.Button):
        await trainerpoints.callback(interaction)

    @discord.ui.button(
        label="Cash",
        style=discord.ButtonStyle.secondary,
        custom_id="controls_cash",
    )
    async def btn_cash(self, interaction: discord.Interaction, button: discord.ui.Button):
        await cash.callback(interaction)

    @discord.ui.button(
        label="Upgrade",
        style=discord.ButtonStyle.primary,
        custom_id="controls_upgrade",
    )
    async def btn_upgrade(self, interaction: discord.Interaction, button: discord.ui.Button):
        await upgrade.callback(interaction)

    @discord.ui.button(
        label="Train",
        style=discord.ButtonStyle.success,
        custom_id="controls_train",
    )
    async def btn_train(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(TrainModal())

    @discord.ui.button(
        label="Rename",
        style=discord.ButtonStyle.success,
        custom_id="controls_rename",
    )
    async def btn_rename(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(RenameModal())

controls_view: Optional[ControlsView] = None

@bot.tree.command(description="Show all commands and what they do")
async def info(inter: discord.Interaction):
    # Overview kept brief; the main goal is the dynamic command list.
    caps_line = "• Passive income: 60 cash/hour | Creature cap: " + str(MAX_CREATURES) + " | Daily cap: " + str(DAILY_BATTLE_CAP) + "/creature/day (Europe/London)."
    header = (
        "**Game Overview**\n"
        "Collect, train, and battle creatures. Progress through tiers to unlock glyphs.\n"
        + caps_line + "\n"
    )
    try:
        cmd_text = _build_command_list(bot)
    except Exception:
        cmd_text = "Failed to build command list."
    content = header + "\n" + cmd_text
    await send_chunks(inter, content, ephemeral=True)

if __name__ == "__main__":
    bot.run(TOKEN)
