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

import asyncio, json, logging, math, os, random, time, re, secrets, string


def fmt_cash(amount: int) -> str:
    """Return a cash value formatted with apostrophes as thousands separators."""
    return f"{int(amount):,}".replace(",", "'")


def _generate_friend_code() -> str:
    """Return a random 8-character alphanumeric friend code."""
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(8))

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
# Optional: channel where the item store is posted/updated.
ITEM_STORE_CHANNEL_ID_ENV = os.getenv("ITEM_STORE_CHANNEL_ID")
# Optional: channel where the augment store is posted/updated.
AUGMENT_STORE_CHANNEL_ID_ENV = os.getenv("AUGMENT_STORE_CHANNEL_ID")
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

ALTER TABLE creatures
  ADD COLUMN IF NOT EXISTS last_hp_regen TIMESTAMPTZ;

ALTER TABLE creatures
  ADD COLUMN IF NOT EXISTS original_name TEXT;

UPDATE creatures
  SET original_name = name
  WHERE original_name IS NULL;

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

-- Per-trainer/day spawn cap (resets at midnight Europe/London)
CREATE TABLE IF NOT EXISTS spawn_caps (
  user_id BIGINT NOT NULL REFERENCES trainers(user_id) ON DELETE CASCADE,
  day DATE NOT NULL,
  count INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, day)
);

-- Cache of pre-generated spawn names/descriptors
CREATE TABLE IF NOT EXISTS spawn_name_cache (
  id SERIAL PRIMARY KEY,
  rarity TEXT NOT NULL,
  name TEXT NOT NULL,
  descriptors TEXT[] NOT NULL
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

-- Per-creature PvP cooldowns
CREATE TABLE IF NOT EXISTS pvp_cooldowns (
  creature_id INT PRIMARY KEY REFERENCES creatures(id) ON DELETE CASCADE,
  last_battle_at TIMESTAMPTZ
);
ALTER TABLE pvp_cooldowns
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

-- Store the message we keep editing for the item store
CREATE TABLE IF NOT EXISTS item_store_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT
);

-- Store the message we keep editing for the augment store
CREATE TABLE IF NOT EXISTS augment_store_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT
);

-- Per-creature installed augments
CREATE TABLE IF NOT EXISTS creature_augments (
  creature_id INT NOT NULL REFERENCES creatures(id) ON DELETE CASCADE,
  augment_name TEXT NOT NULL,
  grade TEXT NOT NULL,
  augment_type TEXT NOT NULL,
  PRIMARY KEY (creature_id, augment_name)
);

-- Per-trainer inventory of items
CREATE TABLE IF NOT EXISTS trainer_items (
  user_id BIGINT NOT NULL,
  item_name TEXT NOT NULL,
  quantity INT NOT NULL DEFAULT 0,
  PRIMARY KEY (user_id, item_name)
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

-- Friend recruitment codes
CREATE TABLE IF NOT EXISTS friend_codes (
  user_id BIGINT PRIMARY KEY REFERENCES trainers(user_id) ON DELETE CASCADE,
  code TEXT NOT NULL UNIQUE
);

-- Friend recruitment redemptions
CREATE TABLE IF NOT EXISTS friend_recruits (
  new_user_id BIGINT PRIMARY KEY REFERENCES trainers(user_id) ON DELETE CASCADE,
  code TEXT NOT NULL REFERENCES friend_codes(code)
);
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game constants ──────────────────────────────────────────
MAX_CREATURES = 5
DAILY_BATTLE_CAP = 2  # <— used for display of remaining battles
DAILY_SPAWN_CAP = 2
PVP_COOLDOWN_HOURS = 12

RARITY_FLOORS: Dict[str, int] = {
    "Common": 2_000,
    "Uncommon": 3_000,
    "Rare": 5_000,
    "Epic": 12_000,
    "Legendary": 25_000,
}

RARITY_MULTS: Dict[str, float] = {
    "Common": 1.00,
    "Uncommon": 1.10,
    "Rare": 1.30,
    "Epic": 1.60,
    "Legendary": 2.20,
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a value between minimum and maximum."""
    return max(minimum, min(maximum, value))


def calc_quicksell_price(
    ovr: int,
    rarity: str,
    current_hp: int,
    max_hp: int,
    wins: int,
    losses: int,
    glyph_tier: int,
) -> int:
    """Calculate the quicksell price for a creature."""
    rarity_mult = RARITY_MULTS.get(rarity, 1.0)
    floor_price = RARITY_FLOORS.get(rarity, 0)

    health_mult = 0.30 + 0.70 * (current_hp / max_hp) if max_hp > 0 else 0.30
    performance = 1 + 0.12 * (wins - losses) / (wins + losses + 8)
    performance = clamp(performance, 0.90, 1.12)

    glyph_mult = 1 + 0.02 * glyph_tier
    if glyph_tier >= 3:
        glyph_mult += 0.03
    if glyph_tier >= 6:
        glyph_mult += 0.05

    price = (60 * (ovr ** 1.05)) * rarity_mult * health_mult * performance * glyph_mult
    price = clamp(price, floor_price, 500_000)
    return int(round(price / 100.0)) * 100


async def _get_quicksell_info(owner_id: int, creature_name: str):
    """Fetch creature info and compute its quicksell price."""
    row = await (await db_pool()).fetchrow(
        """
        SELECT c.id, c.name, c.rarity, c.stats, c.current_hp,
               COALESCE(r.wins, 0) AS wins,
               COALESCE(r.losses, 0) AS losses,
               COALESCE(r.highest_glyph_tier, 0) AS glyph
        FROM creatures c
        LEFT JOIN creature_records r ON r.creature_id = c.id
        WHERE c.owner_id=$1 AND c.name ILIKE $2
        """,
        owner_id,
        creature_name,
    )
    if not row:
        return None
    try:
        stats = json.loads(row["stats"])
    except Exception:
        stats = {}
    ovr = int(sum(stats.values()))
    max_hp = stats.get("HP", 0) * 5
    price = calc_quicksell_price(
        ovr,
        row["rarity"],
        row["current_hp"],
        max_hp,
        row["wins"],
        row["losses"],
        row["glyph"],
    )
    return row, price, ovr

SMALL_HEALING_INJECTOR = "Small Healing Injector"
SMALL_HEALING_INJECTOR_PRICE = 26_000
SMALL_HEALING_INJECTOR_DESC = "Heals a creature for 25% of its max HP."
LARGE_HEALING_INJECTOR = "Large Healing Injector"
LARGE_HEALING_INJECTOR_PRICE = 34_000
LARGE_HEALING_INJECTOR_DESC = "Heals a creature for 50% of its max HP."
FULL_HEALING_INJECTOR = "Full Healing Injector"
FULL_HEALING_INJECTOR_PRICE = 45_000
FULL_HEALING_INJECTOR_DESC = "Fully restores a creature's HP."
STAT_TRAINER = "Stat Trainer"
STAT_TRAINER_PRICE = 52_500
STAT_TRAINER_DESC = "Train one stat by +1 (HP +5)."
PREMIUM_STAT_TRAINER = "Premium Stat Trainer"
PREMIUM_STAT_TRAINER_PRICE = 235_000
PREMIUM_STAT_TRAINER_DESC = "Train all stats by +1."
EXHAUSTION_ELIMINATOR = "Exhaustion Eliminator"
EXHAUSTION_ELIMINATOR_PRICE = 60_000
EXHAUSTION_ELIMINATOR_DESC = "Restores one daily battle use."
GENETIC_RESHUFFLER = "Genetic Reshuffler"
GENETIC_RESHUFFLER_PRICE = 35_000
GENETIC_RESHUFFLER_DESC = "Randomizes a creature's personality."

SUBDERMAL_BALLISTICS_GEL = "Subdermal Ballistics Gel"
SUBDERMAL_BALLISTICS_GEL_PRICE = 90_000
SUBDERMAL_BALLISTICS_GEL_DESC = "Incoming regular attack damage is reduced by 10%"
SUBDERMAL_BALLISTICS_GEL_GRADE = "C"
SUBDERMAL_BALLISTICS_GEL_TYPE = "Passive"

ABLATIVE_FOAM_PODS = "Ablative Foam-Pods"
ABLATIVE_FOAM_PODS_PRICE = 1_600_000
ABLATIVE_FOAM_PODS_DESC = (
    "Once per battle, when the creature first drops to \u226440% max HP, the next 2 incoming hits it takes are reduced to \u00d70.35 final damage."
)
ABLATIVE_FOAM_PODS_GRADE = "B"
ABLATIVE_FOAM_PODS_TYPE = "Passive"

FLUX_CAPACITOR = "Flux Capacitor"
FLUX_CAPACITOR_PRICE = 10_000_000
FLUX_CAPACITOR_DESC = (
    "Every 5 rounds starting from round 5 the creature makes 1 additional regular attack"
)
FLUX_CAPACITOR_GRADE = "A"
FLUX_CAPACITOR_TYPE = "Passive"

IMPROVED_CLOTTING_MATRIX = "Improved Clotting Matrix"
IMPROVED_CLOTTING_MATRIX_PRICE = 150_000
IMPROVED_CLOTTING_MATRIX_DESC = "Heals the creature for 10% of their Max HP"
IMPROVED_CLOTTING_MATRIX_GRADE = "C"
IMPROVED_CLOTTING_MATRIX_TYPE = "Active"

AEGIS_COUNTER = "Aegis Counter"
AEGIS_COUNTER_PRICE = 2_000_000
AEGIS_COUNTER_DESC = (
    "The next time the creature is hit reflects 50% of the damage back to the attacker. Can stack if activated multiple times before taking damage."
)
AEGIS_COUNTER_GRADE = "B"
AEGIS_COUNTER_TYPE = "Active"

PRISM_COIL = "Prism Coil"
PRISM_COIL_PRICE = 12_000_000
PRISM_COIL_DESC = "Fires the Prism Coil"
PRISM_COIL_GRADE = "A"
PRISM_COIL_TYPE = "Active"

MARKING_SPINES = "Marking Spines"
MARKING_SPINES_PRICE = 90_000
MARKING_SPINES_DESC = "Aggressive action weight +5pp; Attack weight -5pp"
MARKING_SPINES_GRADE = "C"
MARKING_SPINES_TYPE = "Passive"

QUANTUM_BLINK = "Quantum Blink"
QUANTUM_BLINK_PRICE = 2_000_000
QUANTUM_BLINK_DESC = "On Activate: Defend this turn and make an Aggressive attack"
QUANTUM_BLINK_GRADE = "B"
QUANTUM_BLINK_TYPE = "Active"

SHOCK_FANGS = "Shock Fangs"
SHOCK_FANGS_PRICE = 12_000_000
SHOCK_FANGS_DESC = "Shock strike with partial armor penetration; chance to stun"
SHOCK_FANGS_GRADE = "A"
SHOCK_FANGS_TYPE = "Active"

AUGMENTS = {
    SUBDERMAL_BALLISTICS_GEL.lower(): {
        "name": SUBDERMAL_BALLISTICS_GEL,
        "price": SUBDERMAL_BALLISTICS_GEL_PRICE,
        "desc": SUBDERMAL_BALLISTICS_GEL_DESC,
        "grade": SUBDERMAL_BALLISTICS_GEL_GRADE,
        "type": SUBDERMAL_BALLISTICS_GEL_TYPE,
    },
    ABLATIVE_FOAM_PODS.lower(): {
        "name": ABLATIVE_FOAM_PODS,
        "price": ABLATIVE_FOAM_PODS_PRICE,
        "desc": ABLATIVE_FOAM_PODS_DESC,
        "grade": ABLATIVE_FOAM_PODS_GRADE,
        "type": ABLATIVE_FOAM_PODS_TYPE,
    },
    FLUX_CAPACITOR.lower(): {
        "name": FLUX_CAPACITOR,
        "price": FLUX_CAPACITOR_PRICE,
        "desc": FLUX_CAPACITOR_DESC,
        "grade": FLUX_CAPACITOR_GRADE,
        "type": FLUX_CAPACITOR_TYPE,
    },
    IMPROVED_CLOTTING_MATRIX.lower(): {
        "name": IMPROVED_CLOTTING_MATRIX,
        "price": IMPROVED_CLOTTING_MATRIX_PRICE,
        "desc": IMPROVED_CLOTTING_MATRIX_DESC,
        "grade": IMPROVED_CLOTTING_MATRIX_GRADE,
        "type": IMPROVED_CLOTTING_MATRIX_TYPE,
    },
    AEGIS_COUNTER.lower(): {
        "name": AEGIS_COUNTER,
        "price": AEGIS_COUNTER_PRICE,
        "desc": AEGIS_COUNTER_DESC,
        "grade": AEGIS_COUNTER_GRADE,
        "type": AEGIS_COUNTER_TYPE,
    },
    PRISM_COIL.lower(): {
        "name": PRISM_COIL,
        "price": PRISM_COIL_PRICE,
        "desc": PRISM_COIL_DESC,
        "grade": PRISM_COIL_GRADE,
        "type": PRISM_COIL_TYPE,
    },
    MARKING_SPINES.lower(): {
        "name": MARKING_SPINES,
        "price": MARKING_SPINES_PRICE,
        "desc": MARKING_SPINES_DESC,
        "grade": MARKING_SPINES_GRADE,
        "type": MARKING_SPINES_TYPE,
    },
    QUANTUM_BLINK.lower(): {
        "name": QUANTUM_BLINK,
        "price": QUANTUM_BLINK_PRICE,
        "desc": QUANTUM_BLINK_DESC,
        "grade": QUANTUM_BLINK_GRADE,
        "type": QUANTUM_BLINK_TYPE,
    },
    SHOCK_FANGS.lower(): {
        "name": SHOCK_FANGS,
        "price": SHOCK_FANGS_PRICE,
        "desc": SHOCK_FANGS_DESC,
        "grade": SHOCK_FANGS_GRADE,
        "type": SHOCK_FANGS_TYPE,
    },
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
    

ACTIONS = ["Attack", "Activate", "Aggressive", "Special", "Defend"]
ACTION_WEIGHTS = [50, 15, 15, 10, 10]

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
    user_afp_charges: int = 0
    opp_afp_charges: int = 0
    user_afp_triggered: bool = False
    opp_afp_triggered: bool = False
    user_aegis_charges: int = 0
    opp_aegis_charges: int = 0
    user_temp_defend: bool = False
    opp_temp_defend: bool = False
    user_stunned: int = 0
    opp_stunned: int = 0
    user_stun_immunity: int = 0
    opp_stun_immunity: int = 0

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

@tasks.loop(hours=1)
async def regenerate_hp():
    now = datetime.now(timezone.utc)
    # Pre-calc 1 hour ago to avoid relying on SQL interval arithmetic on placeholders
    now_minus_1h = now - timedelta(hours=1)
    await (await db_pool()).execute(
        """
        WITH regen AS (
            SELECT id,
                   FLOOR(EXTRACT(EPOCH FROM ($1::timestamptz - COALESCE(last_hp_regen, $2::timestamptz))) / 3600) AS cycles
            FROM creatures
        )
        UPDATE creatures c
        SET current_hp = LEAST(
                COALESCE(c.current_hp, (c.stats->>'HP')::int * 5)
                + CEIL((c.stats->>'HP')::numeric * 5 * 0.03) * r.cycles,
                (c.stats->>'HP')::int * 5
            ),
            last_hp_regen = CASE
                WHEN LEAST(
                        COALESCE(c.current_hp, (c.stats->>'HP')::int * 5)
                        + CEIL((c.stats->>'HP')::numeric * 5 * 0.03) * r.cycles,
                        (c.stats->>'HP')::int * 5
                     ) > COALESCE(c.current_hp, 0)
                THEN $1
                ELSE c.last_hp_regen
            END
        FROM regen r
        WHERE c.id = r.id AND r.cycles > 0 AND c.current_hp < (c.stats->>'HP')::int * 5
        """,
        now,
        now_minus_1h,
    )
    logger.info("Regenerated HP for creatures needing it")

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
def choose_action(augments: List[dict]) -> str:
    weights = ACTION_WEIGHTS.copy()
    if any(a.get("name") == MARKING_SPINES for a in augments):
        weights[2] += 5
        giveback = 5
        take = min(giveback, weights[0] - 35)
        weights[0] -= take
        giveback -= take
        if giveback > 0:
            take = min(giveback, weights[3])
            weights[3] -= take
            giveback -= take
        if giveback > 0:
            take = min(giveback, weights[4])
            weights[4] -= take
    return random.choices(ACTIONS, weights=weights, k=1)[0]


def _deal_damage(
    st: BattleState,
    side: str,
    atk: dict,
    dfn: dict,
    dmg: int,
    dfn_act: str,
    has_afp: bool,
    dfn_temp_defend: bool,
    sudden_death_mult: float,
) -> int:
    """Apply damage with defense, pods, sudden death and Aegis reflection."""
    mult = 1.0
    if dfn_act == "Defend" or dfn_temp_defend:
        mult = min(mult, 0.5)
    if has_afp:
        if side == "user":
            if st.opp_afp_charges > 0:
                mult = min(mult, 0.35)
                st.opp_afp_charges -= 1
        else:
            if st.user_afp_charges > 0:
                mult = min(mult, 0.35)
                st.user_afp_charges -= 1
    dmg = max(1, math.ceil(dmg * mult))
    if sudden_death_mult > 1.0:
        dmg = max(1, math.ceil(dmg * sudden_death_mult))
    if side == "user":
        st.opp_current_hp -= dmg
        new_hp = st.opp_current_hp
        max_hp = st.opp_max_hp
        if has_afp and not st.opp_afp_triggered and new_hp > 0 and new_hp <= math.floor(0.4 * max_hp):
            st.opp_afp_charges = 2
            st.opp_afp_triggered = True
            st.logs.append(f"{dfn['name']}'s {ABLATIVE_FOAM_PODS} deploy!")
    else:
        st.user_current_hp -= dmg
        new_hp = st.user_current_hp
        max_hp = st.user_max_hp
        if has_afp and not st.user_afp_triggered and new_hp > 0 and new_hp <= math.floor(0.4 * max_hp):
            st.user_afp_charges = 2
            st.user_afp_triggered = True
            st.logs.append(f"{dfn['name']}'s {ABLATIVE_FOAM_PODS} deploy!")
    if side == "user" and st.opp_aegis_charges > 0:
        reflect = max(1, math.ceil(dmg * 0.5 * st.opp_aegis_charges))
        st.user_current_hp -= reflect
        st.logs.append(
            f"{dfn['name']}'s {AEGIS_COUNTER} reflects {reflect} dmg back to {atk['name']}!"
        )
        st.opp_aegis_charges = 0
        st.user_aegis_charges = 0
    elif side == "opp" and st.user_aegis_charges > 0:
        reflect = max(1, math.ceil(dmg * 0.5 * st.user_aegis_charges))
        st.opp_current_hp -= reflect
        st.logs.append(
            f"{dfn['name']}'s {AEGIS_COUNTER} reflects {reflect} dmg back to {atk['name']}!"
        )
        st.user_aegis_charges = 0
        st.opp_aegis_charges = 0
    return dmg


def _standard_attack(
    st: BattleState,
    side: str,
    atk: dict,
    dfn: dict,
    act: str,
    dfn_act: str,
    sudden_death_mult: float,
) -> None:
    has_afp = any(a.get("name") == ABLATIVE_FOAM_PODS for a in dfn.get("augments", []))
    has_sbg = any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", []))
    dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
    swings = 2 if atk["stats"]["SPD"] >= 1.5 * dfn["stats"]["SPD"] else 1
    for _ in range(swings):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
        AR_val = 0 if act == "Special" else dfn["stats"]["AR"] // 2
        rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
        s = sum(rolls)
        dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
        if act == "Aggressive":
            dmg = math.ceil(dmg * 1.25)
        if act in ("Attack", "Aggressive") and has_sbg:
            dmg = max(1, math.ceil(dmg * 0.9))
        note = " (defended)" if dfn_act == "Defend" or dfn_temp_defend else ""
        dmg = _deal_damage(st, side, atk, dfn, dmg, dfn_act, has_afp, dfn_temp_defend, sudden_death_mult)
        act_word = {
            "Attack": "hits",
            "Aggressive": "aggressively hits",
            "Special": "unleashes a special attack on",
        }[act]
        st.logs.append(
            f"{atk['name']} {act_word} {dfn['name']} for {dmg} dmg"
            + (f" (rolls {rolls})" if act != "Special" else "")
            + note
        )
        if st.user_current_hp <= 0:
            st.logs.append(f"{st.user_creature['name']} is down!")
        if st.opp_current_hp <= 0:
            st.logs.append(f"{st.opp_creature['name']} is down!")
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break


def _blink_attack(
    st: BattleState,
    side: str,
    atk: dict,
    dfn: dict,
    dfn_act: str,
    sudden_death_mult: float,
) -> None:
    has_afp = any(a.get("name") == ABLATIVE_FOAM_PODS for a in dfn.get("augments", []))
    has_sbg = any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", []))
    dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
    swings = 2 if atk["stats"]["SPD"] >= 1.5 * dfn["stats"]["SPD"] else 1
    for _ in range(swings):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
        AR_val = dfn["stats"]["AR"] // 2
        rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
        s = sum(rolls)
        dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
        dmg = math.ceil(dmg * 1.25)
        if has_sbg:
            dmg = max(1, math.ceil(dmg * 0.9))
        note = " (defended)" if dfn_act == "Defend" or dfn_temp_defend else ""
        dmg = _deal_damage(st, side, atk, dfn, dmg, dfn_act, has_afp, dfn_temp_defend, sudden_death_mult)
        st.logs.append(
            f"{atk['name']} blinks and strikes {dfn['name']} for {dmg} dmg (rolls {rolls}){note}"
        )
        if st.user_current_hp <= 0:
            st.logs.append(f"{st.user_creature['name']} is down!")
        if st.opp_current_hp <= 0:
            st.logs.append(f"{st.opp_creature['name']} is down!")
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break


def _prism_coil_attack(
    st: BattleState,
    side: str,
    atk: dict,
    dfn: dict,
    dfn_act: str,
    sudden_death_mult: float,
    aggressive: bool = False,
) -> None:
    has_afp = any(a.get("name") == ABLATIVE_FOAM_PODS for a in dfn.get("augments", []))
    has_sbg = any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", []))
    dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
    S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
    rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10) + 2)]
    s = sum(rolls)
    dmg = max(1, math.ceil((s * s) / (s if s > 0 else 1)))
    dmg = math.ceil(dmg * 1.20)
    if aggressive:
        dmg = math.ceil(dmg * 1.25)
        if has_sbg:
            dmg = max(1, math.ceil(dmg * 0.9))
    note = " (defended)" if dfn_act == "Defend" or dfn_temp_defend else ""
    dmg = _deal_damage(st, side, atk, dfn, dmg, dfn_act, has_afp, dfn_temp_defend, sudden_death_mult)
    st.logs.append(
        f"{atk['name']} fires {PRISM_COIL} at {dfn['name']} for {dmg} dmg (rolls {rolls}){note}"
    )
    if st.user_current_hp <= 0:
        st.logs.append(f"{st.user_creature['name']} is down!")
    if st.opp_current_hp <= 0:
        st.logs.append(f"{st.opp_creature['name']} is down!")


def _shock_fangs_attack(
    st: BattleState,
    side: str,
    atk: dict,
    dfn: dict,
    dfn_act: str,
    sudden_death_mult: float,
) -> None:
    has_afp = any(a.get("name") == ABLATIVE_FOAM_PODS for a in dfn.get("augments", []))
    has_sbg = any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", []))
    dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
    swings = 2 if atk["stats"]["SPD"] >= 1.5 * dfn["stats"]["SPD"] else 1
    for _ in range(swings):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
        AR_val = math.floor(dfn["stats"]["AR"] * 0.35)
        rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
        s = sum(rolls)
        dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
        dmg = math.ceil(dmg * 1.10)
        if has_sbg:
            dmg = max(1, math.ceil(dmg * 0.9))
        true_dmg = math.ceil((atk["stats"]["PATK"] + atk["stats"]["SATK"]) / 8)
        dmg += true_dmg
        note = " (defended)" if dfn_act == "Defend" or dfn_temp_defend else ""
        dmg = _deal_damage(st, side, atk, dfn, dmg, dfn_act, has_afp, dfn_temp_defend, sudden_death_mult)
        st.logs.append(
            f"{atk['name']} strikes with {SHOCK_FANGS} at {dfn['name']} for {dmg} dmg (rolls {rolls}){note}"
        )
        if st.user_current_hp <= 0:
            st.logs.append(f"{st.user_creature['name']} is down!")
        if st.opp_current_hp <= 0:
            st.logs.append(f"{st.opp_creature['name']} is down!")
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        if ((side == "user" and st.opp_current_hp > 0 and st.opp_stunned == 0 and st.opp_stun_immunity == 0) or
            (side == "opp" and st.user_current_hp > 0 and st.user_stunned == 0 and st.user_stun_immunity == 0)):
            p_stun = max(20, min(80, 40 + 0.8 * (atk["stats"]["SPD"] - dfn["stats"]["SPD"])))
            if random.randint(1, 100) <= p_stun:
                if side == "user":
                    st.opp_stunned = 1
                else:
                    st.user_stunned = 1
                st.logs.append(f"{dfn['name']} is stunned and will miss the next turn!")


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

async def _generate_creature_meta_batch(rarity: str, count: int = 10) -> List[Dict[str, Any]] | None:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    used_names = {r["name"].lower() for r in rows}
    used_words = {w.lower() for r in rows for w in (r["descriptors"] or [])}

    lb_rows = await pool.fetch(
        """
        SELECT COALESCE(c.original_name, r.name) AS name
        FROM creature_records r
        LEFT JOIN creatures c ON c.id = r.creature_id
        ORDER BY r.wins DESC, r.losses ASC, r.name ASC
        LIMIT 20
        """
    )
    leaderboard_names = [row["name"].lower() for row in lb_rows if row["name"]]
    used_names |= set(leaderboard_names)

    from random import sample as _rsample
    remaining = list(used_names - set(leaderboard_names))
    sampled = _rsample(remaining, min(50, len(remaining)))
    avoid_names = sorted(set(leaderboard_names + sampled))

    used_words_list = list(used_words)
    avoid_words = _rsample(used_words_list, min(50, len(used_words_list)))

    prompt = f"""
Invent {count} creatures of rarity **{rarity}**. Return ONLY JSON list:
[
{{"name":"1-3 words","descriptors":["w1","w2","w3"]}},
... (total {count} entries)
]
Avoid names: {', '.join(avoid_names) if avoid_names else 'None'}
Avoid words: {', '.join(sorted(avoid_words)) if avoid_words else 'None'}
"""
    import asyncio as _asyncio, json as _json, re as _re
    for _ in range(3):
        try:
            resp = await _with_timeout(
                _to_thread(lambda: client.responses.create(
                    model=TEXT_MODEL,
                    input=prompt,
                    # Generate multiple names/descriptors in a single request,
                    # so scale the token allowance with the batch size.
                    max_output_tokens=MAX_OUTPUT_TOKENS * count,
                )),
                timeout=60.0
            )
            text = (getattr(resp, 'output_text', '') or '').strip()
            if not text:
                continue
            try:
                data = _json.loads(text)
            except Exception:
                m = _re.search(r"\[[\s\S]*\]", text)
                if not m:
                    raise
                data = _json.loads(m.group(0))
            if isinstance(data, list) and data:
                result = []
                for entry in data[:count]:
                    if "name" in entry and len(entry.get("descriptors", [])) == 3:
                        entry["name"] = str(entry["name"]).title()
                        result.append(entry)
                if len(result) >= count:
                    return result
        except _asyncio.TimeoutError:
            logger.warning("generate_creature_meta timed out; retrying…")
        except Exception as e:
            logger.error("OpenAI error: %s", e)
    return None


async def get_spawn_meta(rarity: str) -> Dict[str, Any] | None:
    pool = await db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, descriptors FROM spawn_name_cache WHERE rarity=$1 ORDER BY id LIMIT 1",
            rarity,
        )
        if not row:
            metas = await _generate_creature_meta_batch(rarity)
            if not metas:
                return None
            await conn.executemany(
                "INSERT INTO spawn_name_cache(rarity, name, descriptors) VALUES ($1,$2,$3)",
                [(rarity, m["name"], m["descriptors"]) for m in metas],
            )
            row = await conn.fetchrow(
                "SELECT id, name, descriptors FROM spawn_name_cache WHERE rarity=$1 ORDER BY id LIMIT 1",
                rarity,
            )
            if not row:
                return None
        await conn.execute("DELETE FROM spawn_name_cache WHERE id=$1", row["id"])
    return {"name": row["name"], "descriptors": row["descriptors"]}


# ─── Name-only generator for battles ──────────────────────────────────────

FIRST_NAMES = [
    "Ashen", "Blazing", "Burning", "Crimson", "Dread", "Ember", "Fiery", "Flame",
    "Frost", "Frozen", "Glacial", "Icebound", "Polar", "Snowy", "Winter", "Storm",
    "Thunder", "Lightning", "Tempest", "Gale", "Windborne", "Skyward", "Clouded",
    "Misty", "Foggy", "Shadow", "Dark", "Duskwind", "Midnight", "Moonlit", "Lunar",
    "Starry", "Cosmic", "Solar", "Sunscorch", "Radiant", "Dawnbreak", "Twilight",
    "Gloom", "Nightfall", "Bonewhite", "Skulking", "Grim", "Deathly", "Carrion",
    "Corpse", "Bloodied", "Crimsoned", "Marrow", "Blackened", "Venomous",
    "Poisonous", "Toxic", "Corrupted", "Pestilent", "Plagued", "Mire", "Swampy",
    "Marshy", "Bogborn", "Bramble", "Thorned", "Rooted", "Viney", "Leafy", "Verdant",
    "Barked", "Stony", "Iron", "Steel", "Rusted", "Copper", "Bronze", "Silver",
    "Golden", "Platinum", "Auric", "Gemmed", "Crystal", "Glassy", "Obsidian", "Onyx",
    "Quartz", "Jade", "Sapphire", "Ruby", "Emerald", "Diamond", "Adamant", "Mithril",
    "Bronzeblooded", "Cobalt", "Leaden", "Mercury", "Acidic", "Fiendish", "Infernal",
    "Charring", "Smoldering", "Scorched", "Smoking", "Molten", "Magma", "Igneous",
    "Volcanic", "Seething", "Blistering", "Wild", "Savage", "Feral", "Untamed",
    "Primeval", "Ancient", "Timeless", "Eternal", "Ageless", "Mystic", "Arcane",
    "Enchanted", "Cursed", "Hexed", "Runed", "Glyphic", "Warded", "Blessed", "Holy",
    "Sacred", "Hallowed", "Celestial", "Stellar", "Astral", "Ethereal", "Spiritual",
    "Phantom", "Ghostly", "Wraithlike", "Spectral", "Haunted", "Soulbound",
    "Necrotic", "Graveborn", "Deathborn", "Undying", "Rotting", "Decayed",
    "Withered", "Hollow", "Shattered", "Broken", "Ruined", "Forsaken", "Doomed",
    "Fallen", "Lost", "Wandering", "Drifting", "Soaring", "Swift", "Rapid", "Fleet",
    "Nimble", "Quick", "Leaping", "Bounding", "Rushing", "Charging", "Relentless",
    "Tireless", "Vicious", "Brutal", "Fierce", "Merciless", "Ruthless",
    "Savagehearted", "Bloodthirsty", "Frenzied", "Maddened", "Rabid", "Foaming",
    "Snarling", "Growling", "Biting", "Slashing", "Piercing", "Crushing", "Rending",
    "Tearing", "Shredding", "Hunting", "Stalking", "Lurking", "Watching", "Silent",
    "Quiet", "Cold", "Chilling", "Icy", "Bitter", "Harsh", "Bleak",
]

SECOND_NAMES = [
    "Wolf", "Lion", "Tiger", "Bear", "Boar", "Bull", "Stag", "Elk", "Ram", "Goat",
    "Horse", "Stallion", "Charger", "Zebra", "Camel", "Mammoth", "Elephant", "Ox",
    "Bison", "Buffalo", "Crocodile", "Alligator", "Snapper", "Tortoise", "Turtle",
    "Lizard", "Gecko", "Monitor", "Basilisk", "Drake", "Wyrmling", "Wyrm", "Dragon",
    "Wyvern", "Serpent", "Viper", "Cobra", "Python", "Anaconda", "Scuttler",
    "Creeper", "Crawler", "Scorpion", "Spider", "Widow", "Tarantula", "Webspinner",
    "Silkfang", "Hornet", "Wasp", "Bee", "Hivebeast", "Drone", "Swarmkin", "Mantis",
    "Locust", "Cricket", "Beetle", "Scarab", "Firefly", "Moth", "Butterfly", "Raven",
    "Crow", "Vulture", "Hawk", "Falcon", "Eagle", "Owl", "Harrier", "Buzzard",
    "Kite", "Gull", "Albatross", "Roc", "Phoenix", "Thunderbird", "Stormbird",
    "Skyhunter", "Cloudstalker", "Bat", "Vampbat", "Duskfang", "Rat", "Mouse",
    "Vermin", "Mole", "Shrew", "Weasel", "Ferret", "Badger", "Otter", "Beaver",
    "Stoat", "Lynx", "Cougar", "Panther", "Jaguar", "Leopard", "Cheetah", "Puma",
    "Catbeast", "Sabrecat", "Saberfang", "Hyena", "Jackal", "Hound", "Direwolf",
    "Warg", "Prowler", "Stalker", "Ravager", "Mauler", "Devourer", "Horror",
    "Abomination", "Monstrosity", "Colossus", "Titanbeast", "Behemoth",
    "Leviathan", "Terror", "Nightmare", "Shadebeast", "Wraithfang", "Spectral",
    "Phantom", "Ghostfang", "Spiritbeast", "Soulfang", "Etherfang", "Arcaneclaw",
    "Runehound", "Hexbeast", "Cursefang", "Spellmaw", "Warfang", "Doomfang",
    "Chaosfang", "Frenzybeast", "Ragehound", "Bloodfang", "Fleshfang", "Bonefang",
    "Skullfang", "Fangmaw", "Clawmaw", "Toothmaw", "Talonmaw", "Hookmaw",
    "Spearmaw", "Needlefang", "Spineback", "Spiketail", "Barbedbeast", "Thornhide",
    "Bramblepaw", "Rootfang", "Mossfang", "Vineclaw", "Barkhide", "Stonehide",
    "Ironhide", "Rusthide", "Steelhide", "Bronzehide", "Silverhide", "Goldhide",
    "Gemhide", "Crystalhide", "Glasshide", "Obsidianhide", "Onyxhide", "Quartzhide",
    "Diamondhide", "Jadefang", "Sapphirefang", "Rubyfang", "Emeraldfang",
    "Topazfang", "Seraphbeast", "Celestial", "Astral", "Cosmicfang", "Starfang",
    "Moonfang", "Sunfang", "Flamefang", "Frostfang", "Stormfang", "Thunderfang",
    "Tempestfang", "Dreadfang", "Nightfang", "Shadowfang", "Venomfang", "Poisonfang",
    "Toxicfang", "Pestilence", "Plaguefang",
]

async def generate_creature_name(rarity: str) -> str:
    """Generate a creature name from predefined lists instead of GPT."""
    pool = await db_pool()
    rows = await pool.fetch("SELECT name FROM creatures")
    used = {r["name"].lower() for r in rows}
    for _ in range(10):
        name = f"{random.choice(FIRST_NAMES)} {random.choice(SECOND_NAMES)}"
        if name.lower() not in used:
            return name
    return name

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

async def _decrement_battle_count(creature_id: int) -> None:
    pool = await db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row = await conn.fetchrow(
                "SELECT count FROM battle_caps WHERE creature_id=$1 AND day=$2 FOR UPDATE",
                creature_id, day,
            )
            if not row:
                return
            current = int(row["count"]) - 1
            if current <= 0:
                await conn.execute(
                    "DELETE FROM battle_caps WHERE creature_id=$1 AND day=$2",
                    creature_id, day,
                )
            else:
                await conn.execute(
                    "UPDATE battle_caps SET count=$3 WHERE creature_id=$1 AND day=$2",
                    creature_id, day, current,
                )

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

# ─── Spawn cap helper ────────────────────────────────────────
async def _can_spawn_and_increment(user_id: int) -> Tuple[bool, int]:
    pool = await db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row = await conn.fetchrow(
                "SELECT count FROM spawn_caps WHERE user_id=$1 AND day=$2 FOR UPDATE",
                user_id, day,
            )
            if not row:
                await conn.execute(
                    "INSERT INTO spawn_caps(user_id, day, count) VALUES($1,$2,1)",
                    user_id, day,
                )
                return True, 1
            current = row["count"]
            if current >= DAILY_SPAWN_CAP:
                return False, current
            new_count = current + 1
            await conn.execute(
                "UPDATE spawn_caps SET count=$3 WHERE user_id=$1 AND day=$2",
                user_id, day, new_count,
            )
            return True, new_count

async def _decrement_spawn_count(user_id: int) -> None:
    pool = await db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row = await conn.fetchrow(
                "SELECT count FROM spawn_caps WHERE user_id=$1 AND day=$2 FOR UPDATE",
                user_id, day,
            )
            if not row:
                return
            current = int(row["count"]) - 1
            if current <= 0:
                await conn.execute(
                    "DELETE FROM spawn_caps WHERE user_id=$1 AND day=$2",
                    user_id, day,
                )
            else:
                await conn.execute(
                    "UPDATE spawn_caps SET count=$3 WHERE user_id=$1 AND day=$2",
                    user_id, day, current,
                )

# ─── PvP cooldown helpers ────────────────────────────────────
async def _pvp_ready_map(creature_ids: List[int]) -> Dict[int, bool]:
    """Return {creature_id: bool} indicating PvP availability per creature."""
    result: Dict[int, bool] = {cid: True for cid in creature_ids}
    if not creature_ids:
        return result
    pool = await db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT creature_id, last_battle_at FROM pvp_cooldowns WHERE creature_id = ANY($1::int[])",
            creature_ids,
        )
    now = datetime.now(timezone.utc)
    for r in rows:
        last = r["last_battle_at"]
        if last is not None and now - last < timedelta(hours=PVP_COOLDOWN_HOURS):
            result[int(r["creature_id"])] = False
    return result

async def _record_pvp_battle(creature_id: int):
    """Record the timestamp of a PvP battle for a creature."""
    await (await db_pool()).execute(
        """
        INSERT INTO pvp_cooldowns(creature_id, last_battle_at)
        VALUES ($1, now())
        ON CONFLICT (creature_id) DO UPDATE SET last_battle_at = EXCLUDED.last_battle_at
        """,
        creature_id,
    )

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
    st.user_temp_defend = False
    st.opp_temp_defend = False
    st.logs.append(f"Round {st.rounds}")
    sudden_death_mult = 1.1 ** (st.rounds // 10)
    if st.rounds % 10 == 0:
        st.logs.append(f"⚡ Sudden Death intensifies! Global damage ×{sudden_death_mult:.2f}")
    uc, oc = st.user_creature, st.opp_creature
    while True:
        if st.user_stunned > 0:
            user_act = "Stunned"
        else:
            user_act = choose_action(uc.get("augments", []))
        if st.opp_stunned > 0:
            opp_act = "Stunned"
        else:
            opp_act = choose_action(oc.get("augments", []))
        if not (user_act == "Defend" and opp_act == "Defend"):
            break
    if user_act == "Stunned":
        st.user_stunned -= 1
        if st.user_stunned == 0:
            st.user_stun_immunity = 1
    elif st.user_stun_immunity > 0:
        st.user_stun_immunity -= 1
    if opp_act == "Stunned":
        st.opp_stunned -= 1
        if st.opp_stunned == 0:
            st.opp_stun_immunity = 1
    elif st.opp_stun_immunity > 0:
        st.opp_stun_immunity -= 1
    st.logs.append(
        f"{st.user_creature['name']} chooses **{user_act}** | "
        f"{st.opp_creature['name']} chooses **{opp_act}**"
    )
    st.logs.append(
        f"{st.user_creature['name']} HP {st.user_current_hp}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {st.opp_current_hp}/{st.opp_max_hp}"
    )
    order = [("user", uc, oc, user_act, opp_act), ("opp", oc, uc, opp_act, user_act)]
    if uc["stats"]["SPD"] < oc["stats"]["SPD"] or (
        uc["stats"]["SPD"] == oc["stats"]["SPD"] and random.choice([0, 1])
    ):
        order.reverse()
    for side, atk, dfn, act, dfn_act in order:
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        has_flux = any(a.get("name") == FLUX_CAPACITOR for a in atk.get("augments", []))
        has_afp = any(a.get("name") == ABLATIVE_FOAM_PODS for a in dfn.get("augments", []))
        if act == "Stunned":
            st.logs.append(f"{atk['name']} is stunned and cannot act.")
            continue
        attacks: List[Tuple[str, bool]] = []
        if act == "Activate":
            active_augs = [
                a for a in atk.get("augments", []) if a.get("type", "").lower() == "active"
            ]
            if active_augs:
                st.logs.append(
                    f"{atk['name']} activates {', '.join(a['name'] for a in active_augs)}!"
                )
                blink_attack = False
                for aug in active_augs:
                    name = aug["name"]
                    if name == IMPROVED_CLOTTING_MATRIX:
                        max_hp = st.user_max_hp if side == "user" else st.opp_max_hp
                        heal = max(1, math.ceil(0.10 * max_hp))
                        if side == "user":
                            st.user_current_hp = min(st.user_current_hp + heal, st.user_max_hp)
                        else:
                            st.opp_current_hp = min(st.opp_current_hp + heal, st.opp_max_hp)
                        st.logs.append(f"{atk['name']} heals {heal} HP.")
                    elif name == AEGIS_COUNTER:
                        if side == "user":
                            st.user_aegis_charges += 1
                            st.logs.append(
                                f"{atk['name']}'s {AEGIS_COUNTER} is primed ({st.user_aegis_charges})."
                            )
                        else:
                            st.opp_aegis_charges += 1
                            st.logs.append(
                                f"{atk['name']}'s {AEGIS_COUNTER} is primed ({st.opp_aegis_charges})."
                            )
                    elif name == QUANTUM_BLINK:
                        if side == "user":
                            st.user_temp_defend = True
                        else:
                            st.opp_temp_defend = True
                        blink_attack = True
                    elif name == SHOCK_FANGS:
                        attacks.append(("shock_fangs", False))
                    elif name == PRISM_COIL:
                        attacks.append(("prism_coil", False))
                # handle interactions
                prism_idx = next((i for i, t in enumerate(attacks) if t[0] == "prism_coil"), None)
                if prism_idx is not None and blink_attack:
                    attacks[prism_idx] = ("prism_coil", True)
                    blink_attack = False
                if blink_attack:
                    attacks.append(("blink", False))
                if not attacks:
                    attacks.append(("Attack", False))
            else:
                st.logs.append(f"{atk['name']} has no activate augment; attacks aggressively.")
                attacks.append(("Aggressive", False))
        elif act == "Defend":
            st.logs.append(f"{atk['name']} is defending.")
            if has_flux and st.rounds % 5 == 0 and st.user_current_hp > 0 and st.opp_current_hp > 0:
                extra_attacks = st.rounds // 5
                for _ in range(extra_attacks):
                    S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
                    AR_val = dfn["stats"]["AR"] // 2
                    rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
                    s = sum(rolls)
                    dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
                    if any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", [])):
                        dmg = max(1, math.ceil(dmg * 0.9))
                    dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
                    dmg = _deal_damage(
                        st,
                        side,
                        atk,
                        dfn,
                        dmg,
                        dfn_act,
                        has_afp,
                        dfn_temp_defend,
                        sudden_death_mult,
                    )
                    st.logs.append(
                        f"{atk['name']} makes an extra attack on {dfn['name']} for {dmg} dmg (rolls {rolls})"
                        + (" (defended)" if dfn_act == "Defend" or dfn_temp_defend else "")
                    )
                    if st.user_current_hp <= 0:
                        st.logs.append(f"{st.user_creature['name']} is down!")
                    if st.opp_current_hp <= 0:
                        st.logs.append(f"{st.opp_creature['name']} is down!")
                    if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                        break
            continue
        else:
            attacks.append((act, False))

        for atype, flag in attacks:
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                break
            if atype == "shock_fangs":
                _shock_fangs_attack(st, side, atk, dfn, dfn_act, sudden_death_mult)
            elif atype == "prism_coil":
                _prism_coil_attack(st, side, atk, dfn, dfn_act, sudden_death_mult, aggressive=flag)
            elif atype == "blink":
                _blink_attack(st, side, atk, dfn, dfn_act, sudden_death_mult)
            else:
                _standard_attack(st, side, atk, dfn, atype, dfn_act, sudden_death_mult)
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                break

        if has_flux and st.rounds % 5 == 0 and st.user_current_hp > 0 and st.opp_current_hp > 0:
            extra_attacks = st.rounds // 5
            for _ in range(extra_attacks):
                S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
                AR_val = dfn["stats"]["AR"] // 2
                rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
                s = sum(rolls)
                dmg = max(1, math.ceil((s * s) / (s + AR_val) if (s + AR_val) > 0 else s))
                if any(a.get("name") == SUBDERMAL_BALLISTICS_GEL for a in dfn.get("augments", [])):
                    dmg = max(1, math.ceil(dmg * 0.9))
                dfn_temp_defend = st.opp_temp_defend if side == "user" else st.user_temp_defend
                dmg = _deal_damage(
                    st,
                    side,
                    atk,
                    dfn,
                    dmg,
                    dfn_act,
                    has_afp,
                    dfn_temp_defend,
                    sudden_death_mult,
                )
                st.logs.append(
                    f"{atk['name']} makes an extra attack on {dfn['name']} for {dmg} dmg (rolls {rolls})"
                    + (" (defended)" if dfn_act == "Defend" or dfn_temp_defend else "")
                )
                if st.user_current_hp <= 0:
                    st.logs.append(f"{st.user_creature['name']} is down!")
                if st.opp_current_hp <= 0:
                    st.logs.append(f"{st.opp_creature['name']} is down!")
                if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
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
        ON CONFLICT (owner_id, name) DO UPDATE
        SET ovr = EXCLUDED.ovr,
            creature_id = EXCLUDED.creature_id
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


async def _get_item_store_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM item_store_messages LIMIT 1")
    if chan:
        return int(chan)
    if ITEM_STORE_CHANNEL_ID_ENV:
        try:
            return int(ITEM_STORE_CHANNEL_ID_ENV)
        except Exception:
            logger.error("ITEM_STORE_CHANNEL_ID env was set but not an integer.")
    return None


async def _get_or_create_item_store_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch item store channel %s: %s", channel_id, e)
        return None

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT message_id FROM item_store_messages WHERE channel_id=$1", channel_id
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
        except Exception:
            message = None

    if message is None:
        try:
            message = await channel.send("Initializing item store…")
            await pool.execute(
                """
                INSERT INTO item_store_messages(channel_id, message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET message_id=EXCLUDED.message_id
                """,
                channel_id,
                message.id,
            )
        except Exception as e:
            logger.error("Failed to create item store message: %s", e)
            return None

    return message


async def _get_augment_store_channel_id() -> Optional[int]:
    pool = await db_pool()
    chan = await pool.fetchval("SELECT channel_id FROM augment_store_messages LIMIT 1")
    if chan:
        return int(chan)
    if AUGMENT_STORE_CHANNEL_ID_ENV:
        try:
            return int(AUGMENT_STORE_CHANNEL_ID_ENV)
        except Exception:
            logger.error("AUGMENT_STORE_CHANNEL_ID env was set but not an integer.")
    return None


async def _get_or_create_augment_store_message(channel_id: int) -> Optional[discord.Message]:
    try:
        channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    except Exception as e:
        logger.error("Failed to fetch augment store channel %s: %s", channel_id, e)
        return None

    pool = await db_pool()
    msg_id = await pool.fetchval(
        "SELECT message_id FROM augment_store_messages WHERE channel_id=$1", channel_id
    )

    message: Optional[discord.Message] = None
    if msg_id:
        try:
            message = await channel.fetch_message(int(msg_id))
        except Exception:
            message = None

    if message is None:
        try:
            message = await channel.send("Initializing augment store…")
            await pool.execute(
                """
                INSERT INTO augment_store_messages(channel_id, message_id)
                VALUES ($1,$2)
                ON CONFLICT (channel_id) DO UPDATE SET message_id=EXCLUDED.message_id
                """,
                channel_id,
                message.id,
            )
        except Exception as e:
            logger.error("Failed to create augment store message: %s", e)
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
                    f"— OVR {ovr} — Personality: {p_name} — Price: {fmt_cash(r['price'])}"
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


async def update_item_store_now(reason: str = "manual") -> None:
    channel_id = await _get_item_store_channel_id()
    if not channel_id:
        return
    message = await _get_or_create_item_store_message(channel_id)
    if message is None:
        return
    updated_ts = int(time.time())
    embed = discord.Embed(
        title="Item Store",
        description=f"Updated: <t:{updated_ts}:R>",
    )
    embed.add_field(
        name=SMALL_HEALING_INJECTOR,
        value=f"Price: {fmt_cash(SMALL_HEALING_INJECTOR_PRICE)}\n{SMALL_HEALING_INJECTOR_DESC}",
        inline=False,
    )
    embed.add_field(
        name=LARGE_HEALING_INJECTOR,
        value=f"Price: {fmt_cash(LARGE_HEALING_INJECTOR_PRICE)}\n{LARGE_HEALING_INJECTOR_DESC}",
        inline=False,
    )
    embed.add_field(
        name=FULL_HEALING_INJECTOR,
        value=f"Price: {fmt_cash(FULL_HEALING_INJECTOR_PRICE)}\n{FULL_HEALING_INJECTOR_DESC}",
        inline=False,
    )
    embed.add_field(
        name=STAT_TRAINER,
        value=f"Price: {fmt_cash(STAT_TRAINER_PRICE)}\n{STAT_TRAINER_DESC}",
        inline=False,
    )
    embed.add_field(
        name=PREMIUM_STAT_TRAINER,
        value=f"Price: {fmt_cash(PREMIUM_STAT_TRAINER_PRICE)}\n{PREMIUM_STAT_TRAINER_DESC}",
        inline=False,
    )
    embed.add_field(
        name=GENETIC_RESHUFFLER,
        value=f"Price: {fmt_cash(GENETIC_RESHUFFLER_PRICE)}\n{GENETIC_RESHUFFLER_DESC}",
        inline=False,
    )
    embed.add_field(
        name=EXHAUSTION_ELIMINATOR,
        value=f"Price: {fmt_cash(EXHAUSTION_ELIMINATOR_PRICE)}\n{EXHAUSTION_ELIMINATOR_DESC}",
        inline=False,
    )
    try:
        await message.edit(content=None, embed=embed, view=ItemStoreView())
        logger.info("Item store updated (%s).", reason)
    except Exception as e:
        logger.error("Failed to edit item store message: %s", e)


async def update_augment_store_now(reason: str = "manual") -> None:
    channel_id = await _get_augment_store_channel_id()
    if not channel_id:
        return
    message = await _get_or_create_augment_store_message(channel_id)
    if message is None:
        return
    updated_ts = int(time.time())
    embed = discord.Embed(
        title="Augment Store",
        description=f"Updated: <t:{updated_ts}:R>",
    )
    for grade in ["A", "B", "C"]:
        grade_augs = [a for a in AUGMENTS.values() if a["grade"].upper() == grade]
        if not grade_augs:
            continue
        grade_augs.sort(key=lambda a: a["price"])
        lines = []
        for data in grade_augs:
            lines.append(
                f"**{data['name']}** ({data['type']})\n{data['desc']}\nCost: {fmt_cash(data['price'])}"
            )
        embed.add_field(name=f"Grade {grade}", value="\n\n".join(lines), inline=False)
    try:
        await message.edit(content=None, embed=embed, view=AugmentStoreView())
        logger.info("Augment store updated (%s).", reason)
    except Exception as e:
        logger.error("Failed to edit augment store message: %s", e)

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
        await _record_pvp_battle(st.creature_id)
        await _record_pvp_battle(st.opp_creature_id)

        if player_won:
            await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", st.wager, st.user_id)
            await pool.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", st.wager, st.opp_user_id)
        else:
            await pool.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", st.wager, st.user_id)
            await pool.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", st.wager, st.opp_user_id)

        st.logs.append(
            f"You {'won' if player_won else 'lost'} the PvP battle: {'+' if player_won else '-'}{fmt_cash(st.wager)} cash."
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
            new_ovr = int(sum(new_stats.values()))
            winner_owner_id = st.user_id if player_won else st.opp_user_id
            await _ensure_record(winner_owner_id, winner_id, winner_cre["name"], new_ovr)
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
            asyncio.create_task(update_shop_now(reason="death"))
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
    st.logs.append(
        f"You {'won' if player_won else 'lost'} the Tier {st.tier} battle: +{fmt_cash(payout)} cash awarded."
    )

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
            new_ovr = int(sum(new_stats.values()))
            await _ensure_record(st.user_id, st.creature_id, st.user_creature["name"], new_ovr)
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
            asyncio.create_task(update_shop_now(reason="death"))
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
            f"💰 {winner_trainer} wins {fmt_cash(abs(summary.get('payout',0)))} cash!",
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
            f"💰 Payout: **+{fmt_cash(summary.get('payout', 0))}** cash to {trainer_name}",
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

    item_chan = await _get_item_store_channel_id()
    if item_chan:
        await _get_or_create_item_store_message(item_chan)
        await update_item_store_now(reason="startup")
    else:
        logger.info("No item store channel configured yet. Use /setitemstore in the desired channel or set ITEM_STORE_CHANNEL_ID.")

    augment_chan = await _get_augment_store_channel_id()
    if augment_chan:
        await _get_or_create_augment_store_message(augment_chan)
        await update_augment_store_now(reason="startup")
    else:
        logger.info("No augment store channel configured yet. Use /setaugments in the desired channel or set AUGMENT_STORE_CHANNEL_ID.")

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

@bot.tree.command(description="(Admin) Set this channel as the item store channel")
async def setitemstore(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute("DELETE FROM item_store_messages")
    await pool.execute(
        "INSERT INTO item_store_messages(channel_id, message_id) VALUES($1, NULL)",
        inter.channel.id,
    )
    await inter.response.send_message(
        f"Item store channel set to {inter.channel.mention}. Initializing…",
        ephemeral=True,
    )
    await _get_or_create_item_store_message(inter.channel.id)
    await update_item_store_now(reason="admin_set_channel")

@bot.tree.command(description="(Admin) Set this channel as the augment store channel")
async def setaugments(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    pool = await db_pool()
    await pool.execute("DELETE FROM augment_store_messages")
    await pool.execute(
        "INSERT INTO augment_store_messages(channel_id, message_id) VALUES($1, NULL)",
        inter.channel.id,
    )
    await inter.response.send_message(
        f"Augment store channel set to {inter.channel.mention}. Initializing…",
        ephemeral=True,
    )
    await _get_or_create_augment_store_message(inter.channel.id)
    await update_augment_store_now(reason="admin_set_channel")
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
        f"Profile created! You received {fmt_cash(20000)} cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a new creature egg (10'000 cash, max 2/day)")
async def spawn(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    can_spawn = await enforce_creature_cap(inter)
    if not can_spawn:
        return
    if row["cash"] < 10_000:
        return await inter.response.send_message("Not enough cash.", ephemeral=True)
    ok, count = await _can_spawn_and_increment(inter.user.id)
    if not ok:
        return await inter.response.send_message(
            f"Daily spawn limit reached: {count}/{DAILY_SPAWN_CAP} used.",
            ephemeral=True,
        )

    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", inter.user.id
    )
    # Defer the response early so the interaction doesn't time out while
    # generating the creature. The final spawn result will be sent publicly.
    await inter.response.defer(thinking=True)

    rarity = spawn_rarity()
    meta = await get_spawn_meta(rarity)
    if not meta:
        await (await db_pool()).execute(
            "UPDATE trainers SET cash = cash + 10000 WHERE user_id=$1", inter.user.id
        )
        await _decrement_spawn_count(inter.user.id)
        await inter.followup.send(
            f"Creature generation timed out. {fmt_cash(10000)} cash has been reimbursed.",
            ephemeral=True,
        )
        return
    stats = allocate_stats(rarity)
    ovr = int(sum(stats.values()))
    max_hp = stats["HP"] * 5

    personality = choose_personality()

    rec = await (await db_pool()).fetchrow(
        "INSERT INTO creatures(owner_id,name,original_name,rarity,descriptors,stats,current_hp,personality,last_hp_regen)"
        " VALUES($1,$2,$3,$4,$5,$6,$7,$8, now()) RETURNING id",
        inter.user.id, meta["name"], meta["name"], rarity, meta["descriptors"], json.dumps(stats), max_hp, json.dumps(personality)
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
    # Send the final creature details publicly so everyone can see the spawn.
    await inter.followup.send(embed=embed)
    asyncio.create_task(update_leaderboard_now(reason="spawn"))

@bot.tree.command(description="Admin: Force spawn ignoring daily limit")
async def spawnforce(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized.", ephemeral=True)
    await _decrement_spawn_count(inter.user.id)
    await spawn(inter)
@bot.tree.command(description="List your creatures")
async def creatures(inter: discord.Interaction):
    if not await ensure_registered(inter):
        return
    await inter.response.defer(ephemeral=True)
    rows = await (await db_pool()).fetch(
        """
        SELECT c.id, c.name, c.rarity, c.descriptors, c.stats, c.current_hp,
               c.personality, (cs.creature_id IS NULL) AS not_listed
        FROM creatures c
        LEFT JOIN creature_shop cs ON cs.creature_id = c.id
        WHERE c.owner_id = $1
        ORDER BY c.id
        """,
        inter.user.id,
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
    pvp_ready_map = await _pvp_ready_map(ids)
    aug_rows = await (await db_pool()).fetch(
        """
        SELECT creature_id,
               ARRAY_AGG(augment_name ORDER BY augment_name) AS aug_names
        FROM creature_augments
        WHERE creature_id = ANY($1::int[])
        GROUP BY creature_id
        """,
        ids,
    )
    augment_map = {
        int(r["creature_id"]): (list(r["aug_names"]) if r["aug_names"] else [])
        for r in aug_rows
    }

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

        # For PvP battles, creatures are fully healed at the start, so current HP
        # shouldn't restrict eligibility. Only enforce the PvP cooldown map.
        pvp_ready = pvp_ready_map.get(int(r["id"]), True)
        pvp_icon = "✅" if pvp_ready else "❌"

        augments = augment_map.get(int(r["id"]), [])
        aug_str = ", ".join(augments) if augments else "None"

        lines = [
            f"**{r['name']}** ({r['rarity']})",
            f"{desc}",
            f"HP: {r['current_hp']}/{max_hp}",
            f"AR: {st.get('AR', 0)}  PATK: {st.get('PATK', 0)}  SATK: {st.get('SATK', 0)}  SPD: {st.get('SPD', 0)}",
            f"Overall: {overall}  |  Glyph: {glyph_disp}",
            f"Personality: { (personality.get('name') + ' (' + ','.join(personality.get('stats', [])) + ')') if personality else '-' }",
            f"Augments ({len(augments)}/3): {aug_str}",
            f"Battles left today: **{left}/{DAILY_BATTLE_CAP}**",
            f"PvP: {pvp_icon}",
        ]
        msg = "\n".join(lines)

        await inter.followup.send(msg, ephemeral=True, view=CreatureView(r["name"]))

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
    info = await _get_quicksell_info(inter.user.id, creature_name)
    if not info:
        return await inter.response.send_message("Creature not found.", ephemeral=True)
    c_row, price, ovr = info
    st = active_battles.get(inter.user.id)
    if st and st.creature_id == c_row["id"]:
        return await inter.response.send_message(
            f"**{c_row['name']}** is currently in a battle. Finish or cancel the battle before selling.",
            ephemeral=True
        )
    await _ensure_record(inter.user.id, c_row["id"], c_row["name"], ovr)
    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM creatures WHERE id=$1", c_row["id"])
            await conn.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", price, inter.user.id)
    await inter.response.send_message(
        f"Sold **{c_row['name']}** ({c_row['rarity']}) for **{fmt_cash(price)}** cash.",
        ephemeral=True,
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
        f"Listed **{c_row['name']}** for **{fmt_cash(price)}** cash in the shop.",
        ephemeral=True,
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
        f"Withdrew **{c_row['name']}** from the shop (was listed for {fmt_cash(deleted['price'])} cash).",
        ephemeral=True,
    )
    asyncio.create_task(update_shop_now(reason="withdraw"))


async def _notify_seller_sale(
    seller_id: int, creature_name: str, buyer_name: str, price: int
) -> None:
    """DM the seller when their creature is purchased."""
    u = bot.get_user(seller_id)
    if not u:
        try:
            u = await bot.fetch_user(seller_id)
        except Exception:
            u = None
    if not u:
        return
    try:
        await u.send(
            f"Your creature **{creature_name}** was bought by {buyer_name} for {fmt_cash(price)} cash."
        )
    except Exception as e:
        logger.warning("Failed to DM seller %s about sale: %s", seller_id, e)


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
            # Transfer leaderboard record ownership to the buyer
            await conn.execute(
                "DELETE FROM creature_records WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
                inter.user.id,
                c_row["name"],
            )
            await conn.execute(
                "UPDATE creature_records SET owner_id=$1 WHERE creature_id=$2",
                inter.user.id,
                c_row["id"],
            )

    await _ensure_record(inter.user.id, c_row["id"], c_row["name"])
    await inter.response.send_message(
        f"You bought **{c_row['name']}** from {c_row['trainer_name']} for {fmt_cash(price)} cash.",
        ephemeral=True,
    )
    buyer_name = getattr(inter.user, "global_name", None) or inter.user.name
    asyncio.create_task(
        _notify_seller_sale(c_row["owner_id"], c_row["name"], buyer_name, price)
    )
    asyncio.create_task(update_shop_now(reason="buy"))
    asyncio.create_task(update_leaderboard_now(reason="buy"))


async def _buy_augment(inter: discord.Interaction, augment_name: str, quantity: int):
    if quantity <= 0:
        return await inter.response.send_message("Quantity must be positive.", ephemeral=True)
    aug_key = augment_name.strip().lower()
    data = AUGMENTS.get(aug_key)
    if not data:
        return await inter.response.send_message("Unknown augment.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    cost = data["price"] * quantity
    pool = await db_pool()
    async with pool.acquire() as conn:
        cash = await conn.fetchval("SELECT cash FROM trainers WHERE user_id=$1", inter.user.id)
        if cash is None or cash < cost:
            return await inter.response.send_message("You don't have enough cash.", ephemeral=True)
        await conn.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", cost, inter.user.id)
        await conn.execute(
            """
            INSERT INTO trainer_items(user_id, item_name, quantity)
            VALUES ($1,$2,$3)
            ON CONFLICT (user_id, item_name)
            DO UPDATE SET quantity = trainer_items.quantity + EXCLUDED.quantity
            """,
            inter.user.id,
            data["name"],
            quantity,
        )
    await inter.response.send_message(
        f"Purchased {quantity} {data['name']}(s).", ephemeral=True
    )
    asyncio.create_task(update_augment_store_now(reason="buy"))


async def _buy_item(inter: discord.Interaction, item_name: str, quantity: int):
    if quantity <= 0:
        return await inter.response.send_message("Quantity must be positive.", ephemeral=True)
    item_key = item_name.strip().lower()
    if item_key == SMALL_HEALING_INJECTOR.lower():
        item_const = SMALL_HEALING_INJECTOR
        price = SMALL_HEALING_INJECTOR_PRICE
    elif item_key == LARGE_HEALING_INJECTOR.lower():
        item_const = LARGE_HEALING_INJECTOR
        price = LARGE_HEALING_INJECTOR_PRICE
    elif item_key == FULL_HEALING_INJECTOR.lower():
        item_const = FULL_HEALING_INJECTOR
        price = FULL_HEALING_INJECTOR_PRICE
    elif item_key == STAT_TRAINER.lower():
        item_const = STAT_TRAINER
        price = STAT_TRAINER_PRICE
    elif item_key == PREMIUM_STAT_TRAINER.lower():
        item_const = PREMIUM_STAT_TRAINER
        price = PREMIUM_STAT_TRAINER_PRICE
    elif item_key == GENETIC_RESHUFFLER.lower():
        item_const = GENETIC_RESHUFFLER
        price = GENETIC_RESHUFFLER_PRICE
    elif item_key == EXHAUSTION_ELIMINATOR.lower():
        item_const = EXHAUSTION_ELIMINATOR
        price = EXHAUSTION_ELIMINATOR_PRICE
    else:
        return await inter.response.send_message("Unknown item.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    cost = price * quantity
    pool = await db_pool()
    async with pool.acquire() as conn:
        cash = await conn.fetchval("SELECT cash FROM trainers WHERE user_id=$1", inter.user.id)
        if cash is None or cash < cost:
            return await inter.response.send_message("You don't have enough cash.", ephemeral=True)
        await conn.execute("UPDATE trainers SET cash = cash - $1 WHERE user_id=$2", cost, inter.user.id)
        await conn.execute(
            """
            INSERT INTO trainer_items(user_id, item_name, quantity)
            VALUES ($1,$2,$3)
            ON CONFLICT (user_id, item_name)
            DO UPDATE SET quantity = trainer_items.quantity + EXCLUDED.quantity
            """,
            inter.user.id,
            item_const,
            quantity,
        )
    await inter.response.send_message(
        f"Purchased {quantity} {item_const}(s).", ephemeral=True
    )
    asyncio.create_task(update_item_store_now(reason="buy"))

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


class BuyAugmentModal(discord.ui.Modal, title="Buy Augment"):
    augment_name = discord.ui.TextInput(label="Augment Name")
    quantity = discord.ui.TextInput(label="Quantity")

    async def on_submit(self, interaction: discord.Interaction):
        try:
            qty = int(self.quantity.value)
        except ValueError:
            return await interaction.response.send_message("Quantity must be an integer.", ephemeral=True)
        await _buy_augment(interaction, self.augment_name.value, qty)


class AugmentStoreView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Buy", style=discord.ButtonStyle.green)
    async def buy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(BuyAugmentModal())


class BuyItemModal(discord.ui.Modal, title="Buy Item"):
    item_name = discord.ui.TextInput(label="Item Name")
    quantity = discord.ui.TextInput(label="Quantity")

    async def on_submit(self, interaction: discord.Interaction):
        try:
            qty = int(self.quantity.value)
        except ValueError:
            return await interaction.response.send_message("Quantity must be an integer.", ephemeral=True)
        await _buy_item(interaction, self.item_name.value, qty)


class ItemStoreView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Buy", style=discord.ButtonStyle.green)
    async def buy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(BuyItemModal())

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

    await battle_lock.acquire()
    current_battler_id = inter.user.id
    incremented = False
    try:
        allowed, count = await _can_start_battle_and_increment(c_row["id"])
        incremented = allowed
        if not allowed:
            await inter.response.send_message(
                f"Daily battle cap reached for **{c_row['name']}**: {DAILY_BATTLE_CAP}/{DAILY_BATTLE_CAP} used. "
                "Try again after midnight Europe/London.", ephemeral=True
            )
            return

        await inter.response.defer(thinking=True)

        aug_rows = await (await db_pool()).fetch(
            "SELECT augment_name, augment_type FROM creature_augments WHERE creature_id=$1",
            c_row["id"],
        )
        user_cre = {
            "name": c_row["name"],
            "stats": stats,
            "augments": [
                {"name": r["augment_name"], "type": r["augment_type"]}
                for r in aug_rows
            ],
        }
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
            "Rules: Action weights A/Ac/Ag/Sp/Df = 50/15/15/10/10, Aggressive +25% dmg, Special ignores AR, "
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
    except Exception:
        if incremented:
            await _decrement_battle_count(c_row["id"])
        raise
    finally:
        active_battles.pop(inter.user.id, None)
        try:
            current_battler_id = None
            if battle_lock.locked():
                battle_lock.release()
        except Exception:
            pass

@bot.tree.command(description="Challenge another trainer to a PvP battle")
async def pvp(inter: discord.Interaction):
    if inter.user.id in active_battles:
        return await inter.response.send_message("You're already in a battle.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    my_creatures = await pool.fetch(
        "SELECT id,name,stats FROM creatures WHERE owner_id=$1",
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

    ready_map = await _pvp_ready_map([int(c_row["id"])])
    if not ready_map.get(int(c_row["id"]), True):
        return await inter.followup.send("That creature must wait before engaging in PvP again.", ephemeral=True)

    opp_rows = await pool.fetch(
        """
        SELECT c.id, c.name, c.owner_id, c.stats, c.current_hp, t.display_name
        FROM creatures c
        JOIN trainers t ON t.user_id = c.owner_id
        WHERE c.owner_id <> $1
        """,
        inter.user.id,
    )
    pvp_map = await _pvp_ready_map([int(r["id"]) for r in opp_rows])
    opp_rows = [
        r
        for r in opp_rows
        if pvp_map.get(int(r["id"]), True)
        and abs(sum(json.loads(r["stats"]).values()) - challenger_ovr) <= 50
    ]
    if not opp_rows:
        return await inter.followup.send("No eligible opponent creatures found.", ephemeral=True)

    options = [
        discord.SelectOption(
            label=f"{r['display_name']}'s {r['name']}",
            value=f"{r['owner_id']}:{r['id']}"
        )
        for r in opp_rows
    ]

    async def handle_challenge(modal_inter: discord.Interaction, opp_row: asyncpg.Record, wager: int):
        opponent_id = int(opp_row["owner_id"])
        if inter.user.id in active_battles or opponent_id in active_battles:
            return await modal_inter.response.send_message("One of the participants is already in a battle.", ephemeral=True)
        opp_tr = await pool.fetchrow("SELECT cash FROM trainers WHERE user_id=$1", opponent_id)
        if not opp_tr:
            return await modal_inter.response.send_message("Opponent is not registered.", ephemeral=True)
        if row["cash"] < wager:
            return await modal_inter.response.send_message("You don't have enough cash for this wager.", ephemeral=True)
        if opp_tr["cash"] < wager:
            return await modal_inter.response.send_message(
                f"Opponent only has {fmt_cash(opp_tr['cash'])} cash; reduce the wager.",
                ephemeral=True,
            )
        ready_map = await _pvp_ready_map([int(c_row["id"]), int(opp_row["id"])])
        if not all(ready_map.values()):
            return await modal_inter.response.send_message(
                "One of the creatures must wait before another PvP battle.",
                ephemeral=True,
            )
        opponent_user = bot.get_user(opponent_id) or await bot.fetch_user(opponent_id)
        opp_stats = json.loads(opp_row["stats"])
        opp_max_hp = opp_stats["HP"] * 5
        opp_name = await _resolve_trainer_name_from_db(opponent_id) or (
            getattr(opponent_user, 'display_name', None) or opponent_user.name
        )

        class PvPChallengeView(discord.ui.View):
            def __init__(self):
                super().__init__(timeout=3600)
                self.message: Optional[discord.Message] = None

            async def on_timeout(self):
                try:
                    if self.message:
                        await self.message.delete()
                except Exception:
                    pass

            @discord.ui.button(label="Accept", style=discord.ButtonStyle.success)
            async def accept(self, interaction: discord.Interaction, button: discord.ui.Button):
                if interaction.user.id != opponent_id:
                    return await interaction.response.send_message("This challenge isn't for you.", ephemeral=True)
                if inter.user.id in active_battles or opponent_id in active_battles:
                    return await interaction.response.send_message("One of the participants is already in a battle.", ephemeral=True)
                ready_map = await _pvp_ready_map([int(c_row["id"]), int(opp_row["id"])])
                if not all(ready_map.values()):
                    return await interaction.response.send_message(
                        "One of the creatures must wait before another PvP battle.",
                        ephemeral=True,
                    )
                global current_battler_id
                if battle_lock.locked() and (current_battler_id not in (inter.user.id, opponent_id)):
                    return await interaction.response.send_message(
                        "⚔ A battle is already in progress. Please try again later.", ephemeral=True
                    )
                await battle_lock.acquire()
                current_battler_id = inter.user.id
                try:
                    await interaction.response.defer(thinking=True)
                    await pool.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", max_hp, c_row["id"])
                    await pool.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", opp_max_hp, opp_row["id"])
                    user_aug = await pool.fetch(
                        "SELECT augment_name, augment_type FROM creature_augments WHERE creature_id=$1",
                        c_row["id"],
                    )
                    opp_aug = await pool.fetch(
                        "SELECT augment_name, augment_type FROM creature_augments WHERE creature_id=$1",
                        opp_row["id"],
                    )
                    st = BattleState(
                        inter.user.id, c_row["id"], 0,
                        {
                            "name": c_row["name"],
                            "stats": stats,
                            "augments": [
                                {"name": r["augment_name"], "type": r["augment_type"]}
                                for r in user_aug
                            ],
                        },
                        max_hp,
                        max_hp,
                        {
                            "name": opp_row["name"],
                            "stats": opp_stats,
                            "augments": [
                                {"name": r["augment_name"], "type": r["augment_type"]}
                                for r in opp_aug
                            ],
                        },
                        opp_max_hp,
                        opp_max_hp,
                        logs=[],
                        is_pvp=True,
                        opp_user_id=opponent_id,
                        opp_creature_id=opp_row["id"],
                        wager=wager,
                        opp_trainer_name=opp_name,
                    )
                    active_battles[inter.user.id] = st
                    active_battles[opponent_id] = st
                    st.logs += [
                        f"PvP battle start! Wager {fmt_cash(wager)} cash.",
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
                    active_battles.pop(opponent_id, None)
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
                        if current_battler_id == inter.user.id:
                            current_battler_id = None
                        if battle_lock.locked():
                            battle_lock.release()
                    except Exception:
                        pass

        view = PvPChallengeView()
        challenge_msg = await inter.channel.send(
            f"{inter.user.mention} challenges {opponent_user.mention}'s {opp_row['name']}! (Wager {wager})",
            view=view
        )
        view.message = challenge_msg
        try:
            await opponent_user.send(
                f"{inter.user.display_name if hasattr(inter.user,'display_name') else inter.user.name} challenged your {opp_row['name']} in {inter.channel.mention}."
            )
        except Exception:
            pass
        await modal_inter.response.send_message("Challenge sent!", ephemeral=True)

    class OpponentSelectView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=60)
            self.selected: Optional[asyncpg.Record] = None

        @discord.ui.select(placeholder="Select opponent's creature", options=options)
        async def select_callback(self, interaction: discord.Interaction, select: discord.ui.Select):
            if interaction.user.id != inter.user.id:
                return await interaction.response.send_message("This menu isn't for you.", ephemeral=True)
            self.selected = next(
                r for r in opp_rows if f"{r['owner_id']}:{r['id']}" == select.values[0]
            )
            modal = WagerModal(self.selected)
            await interaction.response.send_modal(modal)
            self.stop()

    class WagerModal(discord.ui.Modal, title="Enter Wager"):
        amount: discord.ui.TextInput = discord.ui.TextInput(label="Wager", placeholder="Cash amount")

        def __init__(self, opp_row: asyncpg.Record):
            super().__init__()
            self.opp_row = opp_row

        async def on_submit(self, modal_inter: discord.Interaction):
            try:
                wager = int(self.amount.value)
                if wager <= 0:
                    raise ValueError
            except ValueError:
                return await modal_inter.response.send_message("Wager must be a positive integer.", ephemeral=True)
            await handle_challenge(modal_inter, self.opp_row, wager)

    select_view = OpponentSelectView()
    await inter.followup.send(
        "Choose an opponent creature to challenge:", view=select_view, ephemeral=True
    )
    await select_view.wait()

@bot.tree.command(description="Check your cash")
async def cash(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        await inter.response.send_message(
            f"You have {fmt_cash(row['cash'])} cash.", ephemeral=True
        )

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
        return await inter.response.send_message(
            f"Added {fmt_cash(amount)} cash to **all** trainers ({count} rows).",
            ephemeral=True,
        )
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
    await inter.response.send_message(
        f"Added **{fmt_cash(amount)}** cash to **{name}**.", ephemeral=True
    )

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
        msg = (
            f"You have {row['trainer_points']} points. "
            f"Facility Level {level} ({FACILITY_LEVELS[level]['name']}) gives +{bonus} extra per day "
            f"(total {daily}/day)."
        )
        await inter.response.send_message(msg, ephemeral=True, view=TrainerPointsView())


@bot.tree.command(description="Show your friend recruitment code")
async def frc(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    code = await pool.fetchval("SELECT code FROM friend_codes WHERE user_id=$1", inter.user.id)
    if not code:
        while True:
            code = _generate_friend_code()
            exists = await pool.fetchval("SELECT 1 FROM friend_codes WHERE code=$1", code)
            if not exists:
                await pool.execute("INSERT INTO friend_codes(user_id, code) VALUES($1,$2)", inter.user.id, code)
                break
    await inter.response.send_message(f"Your friend code is **{code}**. Share it with friends!", ephemeral=True)


@bot.tree.command(description="Redeem a friend recruitment code")
async def fr(inter: discord.Interaction, code: str):
    row = await ensure_registered(inter)
    if not row:
        return
    code = (code or "").strip().upper()
    pool = await db_pool()
    already = await pool.fetchval("SELECT 1 FROM friend_recruits WHERE new_user_id=$1", inter.user.id)
    if already:
        return await inter.response.send_message("You have already redeemed a friend code.", ephemeral=True)
    rec = await pool.fetchrow("SELECT user_id FROM friend_codes WHERE code=$1", code)
    if not rec:
        return await inter.response.send_message("Invalid friend code.", ephemeral=True)
    inviter_id = rec["user_id"]
    if inviter_id == inter.user.id:
        return await inter.response.send_message("You cannot redeem your own code.", ephemeral=True)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("INSERT INTO friend_recruits(new_user_id, code) VALUES($1,$2)", inter.user.id, code)
            await conn.execute("UPDATE trainers SET cash = cash + 20000 WHERE user_id=$1", inter.user.id)
            await conn.execute("UPDATE trainers SET trainer_points = trainer_points + 5 WHERE user_id=$1", inviter_id)
    inviter_name = await _resolve_trainer_name_from_db(inviter_id) or str(inviter_id)
    await inter.response.send_message(
        f"Friend code redeemed! You received {fmt_cash(20000)} cash and **{inviter_name}** gained 5 trainer points.",
        ephemeral=True,
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
    new_ovr = int(sum(stats.values()))
    await _ensure_record(inter.user.id, c["id"], c["name"], new_ovr)
    asyncio.create_task(update_leaderboard_now(reason="train"))
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
        f"Cost: {fmt_cash(nxt['cost'])} cash",
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
            f"Not enough cash. You need {fmt_cash(cost)} but only have {fmt_cash(row['cash'])}.",
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
        (
            f"✅ Upgraded to **Level {next_level} – {FACILITY_LEVELS[next_level]['name']}**!\n"
            f"Your facility now grants **+{new_bonus} trainer points/day** "
            f"(total {daily_trainer_points_for(next_level)}/day)."
        ),
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
        "SELECT id, name, original_name, rarity, descriptors, stats, current_hp FROM creatures "
        "WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    if c_row["original_name"] and c_row["name"] != c_row["original_name"]:
        return await inter.response.send_message(
            f"{c_row['name']} has been renamed. Rename it back to **{c_row['original_name']}** before adding to the Encyclopedia.",
            ephemeral=True,
        )

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
    name = c_row["original_name"] or c_row["name"]
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


class RenameCreatureModal(discord.ui.Modal):
    """Modal that only asks for a new name for a specific creature."""

    def __init__(self, creature_name: str):
        super().__init__(title=f"Rename {creature_name}")
        self.creature_name = creature_name
        self.new_name: discord.ui.TextInput = discord.ui.TextInput(label="New Name")
        self.add_item(self.new_name)

    async def on_submit(self, interaction: discord.Interaction):
        await _rename_creature(interaction, self.creature_name, self.new_name.value)


async def _use_small_healing_injector(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            SMALL_HEALING_INJECTOR,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message("You don't have any Small Healing Injectors.", ephemeral=True)
        stats = json.loads(c_row["stats"])
        max_hp = int(stats.get("HP", 0)) * 5
        heal_amount = max(1, math.floor(max_hp * 0.25))
        new_hp = min(int(c_row["current_hp"]) + heal_amount, max_hp)
        healed = new_hp - int(c_row["current_hp"])
        await conn.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", new_hp, c_row["id"])
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            SMALL_HEALING_INJECTOR,
        )
    await inter.response.send_message(
        f"Healed **{c_row['name']}** for {healed} HP.", ephemeral=True
    )


async def _use_large_healing_injector(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            LARGE_HEALING_INJECTOR,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message("You don't have any Large Healing Injectors.", ephemeral=True)
        stats = json.loads(c_row["stats"])
        max_hp = int(stats.get("HP", 0)) * 5
        heal_amount = max(1, math.floor(max_hp * 0.5))
        new_hp = min(int(c_row["current_hp"]) + heal_amount, max_hp)
        healed = new_hp - int(c_row["current_hp"])
        await conn.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", new_hp, c_row["id"])
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            LARGE_HEALING_INJECTOR,
        )
    await inter.response.send_message(
        f"Healed **{c_row['name']}** for {healed} HP.", ephemeral=True
    )


async def _use_full_healing_injector(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            FULL_HEALING_INJECTOR,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message("You don't have any Full Healing Injectors.", ephemeral=True)
        stats = json.loads(c_row["stats"])
        max_hp = int(stats.get("HP", 0)) * 5
        new_hp = max_hp
        healed = new_hp - int(c_row["current_hp"])
        await conn.execute("UPDATE creatures SET current_hp=$1 WHERE id=$2", new_hp, c_row["id"])
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            FULL_HEALING_INJECTOR,
        )
    await inter.response.send_message(
        f"Healed **{c_row['name']}** for {healed} HP.", ephemeral=True
    )


async def _use_exhaustion_eliminator(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            EXHAUSTION_ELIMINATOR,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message("You don't have any Exhaustion Eliminators.", ephemeral=True)
        async with conn.transaction():
            day = await conn.fetchval("SELECT (now() AT TIME ZONE 'Europe/London')::date")
            row_cap = await conn.fetchrow(
                "SELECT count FROM battle_caps WHERE creature_id=$1 AND day=$2 FOR UPDATE",
                c_row["id"],
                day,
            )
            if not row_cap or int(row_cap["count"]) <= 0:
                return await inter.response.send_message(
                    f"**{c_row['name']}** already has 2/2 battles remaining today.",
                    ephemeral=True,
                )
            current = int(row_cap["count"])
            new_count = current - 1
            if new_count == 0:
                await conn.execute(
                    "DELETE FROM battle_caps WHERE creature_id=$1 AND day=$2",
                    c_row["id"],
                    day,
                )
            else:
                await conn.execute(
                    "UPDATE battle_caps SET count=$3 WHERE creature_id=$1 AND day=$2",
                    c_row["id"],
                    day,
                    new_count,
                )
            await conn.execute(
                "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
                inter.user.id,
                EXHAUSTION_ELIMINATOR,
            )
    await inter.response.send_message(
        f"Restored one daily battle for **{c_row['name']}**.", ephemeral=True
    )


async def _use_stat_trainer(inter: discord.Interaction, creature_name: str, stat: str):
    row = await ensure_registered(inter)
    if not row:
        return
    stat = stat.upper().strip()
    if stat not in PRIMARY_STATS:
        return await inter.response.send_message(
            f"Stat must be one of {', '.join(PRIMARY_STATS)}.", ephemeral=True
        )
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        listed = await conn.fetchval(
            "SELECT 1 FROM creature_shop WHERE creature_id=$1",
            c_row["id"],
        )
        if listed:
            return await inter.response.send_message(
                f"{c_row['name']} is listed in the shop and cannot be trained.",
                ephemeral=True,
            )
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            STAT_TRAINER,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message(
                "You don't have any Stat Trainers.", ephemeral=True
            )
        stats = json.loads(c_row["stats"])
        stats[stat] = int(stats.get(stat, 0)) + 1
        new_max_hp = stats["HP"] * 5
        new_cur_hp = c_row["current_hp"]
        if stat == "HP":
            new_cur_hp = min(new_cur_hp + 5, new_max_hp)
        await conn.execute(
            "UPDATE creatures SET stats=$1,current_hp=$2 WHERE id=$3",
            json.dumps(stats),
            new_cur_hp,
            c_row["id"],
        )
        new_ovr = int(sum(stats.values()))
        await _ensure_record(inter.user.id, c_row["id"], c_row["name"], new_ovr)
        asyncio.create_task(update_leaderboard_now(reason="stat_trainer"))
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            STAT_TRAINER,
        )
    display_inc = 5 if stat == "HP" else 1
    await inter.response.send_message(
        f"{c_row['name']} trained: +{display_inc} {stat}.", ephemeral=True
    )


async def _use_premium_stat_trainer(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        listed = await conn.fetchval(
            "SELECT 1 FROM creature_shop WHERE creature_id=$1",
            c_row["id"],
        )
        if listed:
            return await inter.response.send_message(
                f"{c_row['name']} is listed in the shop and cannot be trained.",
                ephemeral=True,
            )
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            PREMIUM_STAT_TRAINER,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message(
                "You don't have any Premium Stat Trainers.", ephemeral=True
            )
        stats = json.loads(c_row["stats"])
        for s in PRIMARY_STATS:
            stats[s] = int(stats.get(s, 0)) + 1
        new_max_hp = stats["HP"] * 5
        new_cur_hp = min(int(c_row["current_hp"]) + 5, new_max_hp)
        await conn.execute(
            "UPDATE creatures SET stats=$1,current_hp=$2 WHERE id=$3",
            json.dumps(stats),
            new_cur_hp,
            c_row["id"],
        )
        new_ovr = int(sum(stats.values()))
        await _ensure_record(inter.user.id, c_row["id"], c_row["name"], new_ovr)
        asyncio.create_task(update_leaderboard_now(reason="premium_stat_trainer"))
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            PREMIUM_STAT_TRAINER,
        )
    await inter.response.send_message(
        f"{c_row['name']} trained: +1 to all stats.", ephemeral=True
    )


async def _use_genetic_reshuffler(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            GENETIC_RESHUFFLER,
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message(
                "You don't have any Genetic Reshufflers.", ephemeral=True
            )
        personality = choose_personality()
        await conn.execute(
            "UPDATE creatures SET personality=$1 WHERE id=$2",
            json.dumps(personality),
            c_row["id"],
        )
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            GENETIC_RESHUFFLER,
        )
    stats_list = ",".join(personality.get("stats", []))
    await inter.response.send_message(
        f"{c_row['name']}'s personality is now {personality.get('name')} ({stats_list}).",
        ephemeral=True,
    )


async def _install_augment(inter: discord.Interaction, creature_name: str, augment_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    data = AUGMENTS.get(augment_name.strip().lower())
    if not data:
        return await inter.response.send_message("Unknown augment.", ephemeral=True)
    pool = await db_pool()
    async with pool.acquire() as conn:
        c_row = await conn.fetchrow(
            "SELECT id, name FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            inter.user.id,
            creature_name,
        )
        if not c_row:
            return await inter.response.send_message("Creature not found.", ephemeral=True)
        qty = await conn.fetchval(
            "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            data["name"],
        )
        if not qty or int(qty) <= 0:
            return await inter.response.send_message("You don't have this augment.", ephemeral=True)
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM creature_augments WHERE creature_id=$1",
            c_row["id"],
        )
        if int(count or 0) >= 3:
            return await inter.response.send_message(
                f"{c_row['name']} already has three augments.", ephemeral=True
            )
        glyph = await conn.fetchval(
            "SELECT COALESCE(MAX(CASE WHEN glyph_unlocked THEN tier ELSE 0 END),0) FROM creature_progress WHERE creature_id=$1",
            c_row["id"],
        )
        req = {"A": 6, "B": 3, "C": 0}.get(data["grade"], 0)
        if int(glyph or 0) < req:
            return await inter.response.send_message(
                f"Requires Glyph {req} for grade {data['grade']} augments.", ephemeral=True
            )
        await conn.execute(
            "INSERT INTO creature_augments(creature_id, augment_name, grade, augment_type) VALUES($1,$2,$3,$4)",
            c_row["id"],
            data["name"],
            data["grade"],
            data["type"],
        )
        await conn.execute(
            "UPDATE trainer_items SET quantity=quantity-1 WHERE user_id=$1 AND item_name=$2",
            inter.user.id,
            data["name"],
        )
    await inter.response.send_message(
        f"Installed {data['name']} on **{c_row['name']}**.", ephemeral=True
    )


async def _show_item_menu(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    pool = await db_pool()
    qty_small = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        SMALL_HEALING_INJECTOR,
    )
    qty_large = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        LARGE_HEALING_INJECTOR,
    )
    qty_full = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        FULL_HEALING_INJECTOR,
    )
    qty_exhaust = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        EXHAUSTION_ELIMINATOR,
    )
    qty_stat = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        STAT_TRAINER,
    )
    qty_premium = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        PREMIUM_STAT_TRAINER,
    )
    qty_reshuffler = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        GENETIC_RESHUFFLER,
    )
    qty_ballistics = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        SUBDERMAL_BALLISTICS_GEL,
    )
    qty_foam = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        ABLATIVE_FOAM_PODS,
    )
    qty_flux = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        FLUX_CAPACITOR,
    )
    qty_clot = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        IMPROVED_CLOTTING_MATRIX,
    )
    qty_aegis = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        AEGIS_COUNTER,
    )
    qty_prism = await pool.fetchval(
        "SELECT quantity FROM trainer_items WHERE user_id=$1 AND item_name=$2",
        inter.user.id,
        PRISM_COIL,
    )
    qty_small = int(qty_small or 0)
    qty_large = int(qty_large or 0)
    qty_full = int(qty_full or 0)
    qty_exhaust = int(qty_exhaust or 0)
    qty_stat = int(qty_stat or 0)
    qty_premium = int(qty_premium or 0)
    qty_reshuffler = int(qty_reshuffler or 0)
    qty_ballistics = int(qty_ballistics or 0)
    qty_foam = int(qty_foam or 0)
    qty_flux = int(qty_flux or 0)
    qty_clot = int(qty_clot or 0)
    qty_aegis = int(qty_aegis or 0)
    qty_prism = int(qty_prism or 0)
    if (
        qty_small <= 0
        and qty_large <= 0
        and qty_full <= 0
        and qty_exhaust <= 0
        and qty_stat <= 0
        and qty_premium <= 0
        and qty_reshuffler <= 0
        and qty_ballistics <= 0
        and qty_foam <= 0
        and qty_flux <= 0
        and qty_clot <= 0
        and qty_aegis <= 0
        and qty_prism <= 0
    ):
        return await inter.response.send_message("You have no items.", ephemeral=True)
    await inter.response.send_message(
        f"Choose an item for {creature_name}:",
        ephemeral=True,
        view=UseItemView(
            creature_name,
            qty_small,
            qty_large,
            qty_full,
            qty_exhaust,
            qty_stat,
            qty_premium,
            qty_reshuffler,
            qty_ballistics,
            qty_foam,
            qty_flux,
            qty_clot,
            qty_aegis,
            qty_prism,
        ),
    )


class UseItemView(discord.ui.View):
    def __init__(
        self,
        creature_name: str,
        small_qty: int,
        large_qty: int,
        full_qty: int,
        exhaust_qty: int,
        stat_qty: int,
        premium_qty: int,
        reshuffler_qty: int,
        ballistics_qty: int,
        foam_qty: int,
        flux_qty: int,
        clot_qty: int,
        aegis_qty: int,
        prism_qty: int,
    ):
        super().__init__(timeout=None)
        self.creature_name = creature_name
        self.use_small.label = f"{SMALL_HEALING_INJECTOR} ({small_qty})"
        self.use_small.disabled = small_qty <= 0
        self.use_large.label = f"{LARGE_HEALING_INJECTOR} ({large_qty})"
        self.use_large.disabled = large_qty <= 0
        self.use_full.label = f"{FULL_HEALING_INJECTOR} ({full_qty})"
        self.use_full.disabled = full_qty <= 0
        self.use_exhaust.label = f"{EXHAUSTION_ELIMINATOR} ({exhaust_qty})"
        self.use_exhaust.disabled = exhaust_qty <= 0
        self.use_stat.label = f"{STAT_TRAINER} ({stat_qty})"
        self.use_stat.disabled = stat_qty <= 0
        self.use_premium_stat.label = f"{PREMIUM_STAT_TRAINER} ({premium_qty})"
        self.use_premium_stat.disabled = premium_qty <= 0
        self.use_reshuffler.label = f"{GENETIC_RESHUFFLER} ({reshuffler_qty})"
        self.use_reshuffler.disabled = reshuffler_qty <= 0
        self.use_ballistics.label = f"{SUBDERMAL_BALLISTICS_GEL} ({ballistics_qty})"
        self.use_ballistics.disabled = ballistics_qty <= 0
        self.use_foam.label = f"{ABLATIVE_FOAM_PODS} ({foam_qty})"
        self.use_foam.disabled = foam_qty <= 0
        self.use_flux.label = f"{FLUX_CAPACITOR} ({flux_qty})"
        self.use_flux.disabled = flux_qty <= 0
        self.use_clot.label = f"{IMPROVED_CLOTTING_MATRIX} ({clot_qty})"
        self.use_clot.disabled = clot_qty <= 0
        self.use_aegis.label = f"{AEGIS_COUNTER} ({aegis_qty})"
        self.use_aegis.disabled = aegis_qty <= 0
        self.use_prism.label = f"{PRISM_COIL} ({prism_qty})"
        self.use_prism.disabled = prism_qty <= 0

    @discord.ui.button(label=SMALL_HEALING_INJECTOR, style=discord.ButtonStyle.primary)
    async def use_small(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_small_healing_injector(interaction, self.creature_name)

    @discord.ui.button(label=LARGE_HEALING_INJECTOR, style=discord.ButtonStyle.primary)
    async def use_large(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_large_healing_injector(interaction, self.creature_name)

    @discord.ui.button(label=FULL_HEALING_INJECTOR, style=discord.ButtonStyle.primary)
    async def use_full(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_full_healing_injector(interaction, self.creature_name)

    @discord.ui.button(label=EXHAUSTION_ELIMINATOR, style=discord.ButtonStyle.primary)
    async def use_exhaust(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_exhaustion_eliminator(interaction, self.creature_name)

    @discord.ui.button(label=STAT_TRAINER, style=discord.ButtonStyle.primary)
    async def use_stat(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(StatTrainerModal(self.creature_name))

    @discord.ui.button(label=PREMIUM_STAT_TRAINER, style=discord.ButtonStyle.primary)
    async def use_premium_stat(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_premium_stat_trainer(interaction, self.creature_name)

    @discord.ui.button(label=GENETIC_RESHUFFLER, style=discord.ButtonStyle.primary)
    async def use_reshuffler(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _use_genetic_reshuffler(interaction, self.creature_name)

    @discord.ui.button(label=SUBDERMAL_BALLISTICS_GEL, style=discord.ButtonStyle.primary)
    async def use_ballistics(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, SUBDERMAL_BALLISTICS_GEL)

    @discord.ui.button(label=ABLATIVE_FOAM_PODS, style=discord.ButtonStyle.primary)
    async def use_foam(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, ABLATIVE_FOAM_PODS)

    @discord.ui.button(label=FLUX_CAPACITOR, style=discord.ButtonStyle.primary)
    async def use_flux(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, FLUX_CAPACITOR)

    @discord.ui.button(label=IMPROVED_CLOTTING_MATRIX, style=discord.ButtonStyle.primary)
    async def use_clot(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, IMPROVED_CLOTTING_MATRIX)

    @discord.ui.button(label=AEGIS_COUNTER, style=discord.ButtonStyle.primary)
    async def use_aegis(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, AEGIS_COUNTER)

    @discord.ui.button(label=PRISM_COIL, style=discord.ButtonStyle.primary)
    async def use_prism(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _install_augment(interaction, self.creature_name, PRISM_COIL)


class StatTrainerModal(discord.ui.Modal):
    def __init__(self, creature_name: str):
        super().__init__(title=f"Stat Trainer for {creature_name}")
        self.creature_name = creature_name
        self.stat = discord.ui.TextInput(label="Stat")
        self.add_item(self.stat)

    async def on_submit(self, interaction: discord.Interaction):
        await _use_stat_trainer(interaction, self.creature_name, self.stat.value)


class QuickSellConfirmView(discord.ui.View):
    def __init__(self, user_id: int, creature_name: str):
        super().__init__(timeout=None)
        self.user_id = user_id
        self.creature_name = creature_name

    @discord.ui.button(label="Yes", style=discord.ButtonStyle.success)
    async def yes(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message("This isn't your confirmation.", ephemeral=True)
        await quicksell.callback(interaction, self.creature_name)
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
        await interaction.response.send_message("Quicksell canceled.", ephemeral=True)
        for child in self.children:
            child.disabled = True
        try:
            await interaction.message.edit(view=self)
        except Exception:
            pass


class CreatureView(discord.ui.View):
    """View with rename, item, and quick sell buttons for a single creature."""

    def __init__(self, creature_name: str):
        super().__init__(timeout=None)
        self.creature_name = creature_name

    @discord.ui.button(label="Rename", style=discord.ButtonStyle.success)
    async def btn_rename(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(RenameCreatureModal(self.creature_name))

    @discord.ui.button(label="Item", style=discord.ButtonStyle.primary)
    async def btn_item(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _show_item_menu(interaction, self.creature_name)

    @discord.ui.button(label="Quick Sell", style=discord.ButtonStyle.danger)
    async def btn_quicksell(self, interaction: discord.Interaction, button: discord.ui.Button):
        info = await _get_quicksell_info(interaction.user.id, self.creature_name)
        if not info:
            return await interaction.response.send_message("Creature not found.", ephemeral=True)
        _, price, _ = info
        await interaction.response.send_message(
            f"Are you sure you want to quick sell **{self.creature_name}** for **{fmt_cash(price)}** cash? This cannot be undone.",
            ephemeral=True,
            view=QuickSellConfirmView(interaction.user.id, self.creature_name),
        )


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


class TrainerPointsView(discord.ui.View):
    @discord.ui.button(label="Train", style=discord.ButtonStyle.success)
    async def btn_train(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(TrainModal())


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

controls_view: Optional[ControlsView] = None

@bot.tree.command(description="Show basic info about the game")
async def info(inter: discord.Interaction):
    """Respond with core commands and stat explanations."""
    caps_line = (
        "• Passive income: 60 cash/hour | Creature cap: "
        + str(MAX_CREATURES)
        + "."
    )
    header = (
        "**Game Overview**\n"
        "Collect, train, and battle creatures. Progress through tiers to unlock glyphs.\n"
        + caps_line
        + "\n"
    )
    commands_info = (
        "**Basic Commands**\n"
        "/register — create your trainer account.\n"
        "/record <creature_name> — view a creature's lifetime record.\n"
        "/spawn — spend 10,000 cash to hatch a new creature (max 2/day).\n"
        "/spawnforce — admin: spawn ignoring daily limit.\n"
        "/battle <creature_name> <tier> — fight an AI opponent in a chosen tier.\n"
        "/pvp — challenge another trainer to battle.\n"
        "/frc — show your friend recruitment code.\n"
        "/fr <code> — redeem a friend code for rewards.\n"
    )
    stats_info = (
        "\n**Creature Stats**\n"
        "HP: total health (each point grants 5 HP).\n"
        "AR: armor that mitigates incoming damage.\n"
        "PATK: physical attack power.\n"
        "SATK: special attack power.\n"
        "SPD: speed; acts first and may grant extra swings.\n"
    )
    content = header + "\n" + commands_info + stats_info
    await send_chunks(inter, content, ephemeral=True)

if __name__ == "__main__":
    bot.run(TOKEN)
