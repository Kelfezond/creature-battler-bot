from __future__ import annotations

import asyncio, json, logging, math, os, random, time, re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import discord
from discord.ext import commands, tasks
from openai import OpenAI

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

logger.info("Using TEXT_MODEL=%s, IMAGE_MODEL=%s", TEXT_MODEL, IMAGE_MODEL)

# Optional: channel where the live leaderboard is posted/updated.
LEADERBOARD_CHANNEL_ID_ENV = os.getenv("LEADERBOARD_CHANNEL_ID")

# Admin allow-list for privileged commands
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
intents.message_content = True  # enable content for commands
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

ALTER TABLE trainers
  ADD COLUMN IF NOT EXISTS facility_level INT DEFAULT 1;

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

-- Lifetime win/loss records (persist after death/sale)
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

-- Store the message we keep editing for the live leaderboard
CREATE TABLE IF NOT EXISTS leaderboard_messages (
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

# ─── Game constants & utilities ──────────────────────────────
MAX_CREATURES = 5
DAILY_BATTLE_CAP = 2

SELL_PRICES: Dict[str, int] = {
    "Common": 1_000,
    "Uncommon": 2_000,
    "Rare": 10_000,
    "Epic": 20_000,
    "Legendary": 50_000,
}

def spawn_rarity() -> str:
    """Determine rarity of a new spawn via weighted chance."""
    r = random.random() * 100.0
    if r < 75.0:
        return "Common"
    elif r < 88.0:
        return "Uncommon"
    elif r < 95.0:
        return "Rare"
    elif r < 99.5:
        return "Epic"
    else:
        return "Legendary"

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
    "Common": (25, 50),
    "Uncommon": (50, 100),
    "Rare": (100, 200),
    "Epic": (200, 400),
    "Legendary": (400, 800),
}
TIER_EXTRAS = {
    1: (0, 10),  2: (10, 30), 3: (30, 60), 4: (60, 100),
    5: (100, 140), 6: (140, 180), 7: (180, 220),
    8: (220, 260), 9: (260, 300)
}
PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]
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
    next_log_idx: int = 0
    rounds: int = 0

# Track active battles by user ID (allows multiple battles concurrently across users)
active_battles: Dict[int, BattleState] = {}

# ─── Scheduled tasks ─────────────────────────────────────────
@tasks.loop(hours=1)
async def distribute_cash():
    # Skip immediate run after restart
    if distribute_cash.current_loop == 0:
        logger.info("Skipping first hourly cash distribution after restart")
        return
    await (await db_pool()).execute("UPDATE trainers SET cash = cash + 60")
    logger.info("Distributed 60 cash to all trainers")

@tasks.loop(hours=24)
async def distribute_points():
    if distribute_points.current_loop == 0:
        logger.info("Skipping first daily trainer-point distribution after restart")
        return
    await (await db_pool()).execute("""
        UPDATE trainers
        SET trainer_points = trainer_points
          + (5 + LEAST(GREATEST(facility_level - 1, 0), 5))
    """)
    logger.info("Distributed daily trainer points with facility bonuses")

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
    logger.info("Regenerated 20% HP for all creatures")

# ─── Helper functions ────────────────────────────────────────
async def ensure_registered(inter: discord.Interaction) -> Optional[asyncpg.Record]:
    """Ensure the user has a trainer profile; send error message if not."""
    row = await (await db_pool()).fetchrow(
        "SELECT cash, trainer_points, facility_level FROM trainers WHERE user_id=$1",
        inter.user.id
    )
    if not row:
        await inter.response.send_message("Use /register first.", ephemeral=True)
        return None
    # Update stored display name (if changed) to avoid redundant API calls
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

async def ensure_creature(inter: discord.Interaction, creature_name: str) -> Optional[asyncpg.Record]:
    """Fetch the user's creature by name; send error message if not found."""
    row = await (await db_pool()).fetchrow(
        "SELECT id, name, rarity, descriptors, stats, current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not row:
        await inter.response.send_message("Creature not found.", ephemeral=True)
        return None
    return row

# ─── Leaderboard and records utilities ───────────────────────
async def _resolve_trainer_name_from_db(user_id: int) -> Optional[str]:
    """Fetch trainer's display name from DB (fallback to user_id if not set)."""
    try:
        return await (await db_pool()).fetchval(
            "SELECT COALESCE(display_name, user_id::text) FROM trainers WHERE user_id=$1",
            user_id,
        )
    except Exception:
        return None

async def _backfill_creature_records():
    pool = await db_pool()
    await pool.execute("""
        INSERT INTO creature_records (creature_id, owner_id, name)
        SELECT c.id, c.owner_id, c.name
        FROM creatures c
        LEFT JOIN creature_records r
          ON r.owner_id = c.owner_id AND LOWER(r.name) = LOWER(c.name)
        WHERE r.owner_id IS NULL
    """)
    logger.info("Backfilled creature_records for existing creatures (if any missing).")

async def _ensure_record(owner_id: int, creature_id: int, name: str):
    await (await db_pool()).execute("""
        INSERT INTO creature_records (creature_id, owner_id, name)
        VALUES ($1,$2,$3)
        ON CONFLICT (owner_id, name) DO NOTHING
    """, creature_id, owner_id, name)

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

async def _record_death(owner_id: int, name: str):
    await (await db_pool()).execute(
        "UPDATE creature_records SET is_dead=true, died_at=now() WHERE owner_id=$1 AND LOWER(name)=LOWER($2)",
        owner_id, name
    )

async def _get_progress(conn: asyncpg.Connection, creature_id: int, tier: int) -> Optional[asyncpg.Record]:
    return await conn.fetchrow(
        "SELECT wins, glyph_unlocked FROM creature_progress WHERE creature_id=$1 AND tier=$2",
        creature_id, tier
    )

async def _get_wins_for_tier(creature_id: int, tier: int) -> int:
    async with (await db_pool()).acquire() as conn:
        row = await _get_progress(conn, creature_id, tier)
        return (row["wins"] if row else 0)

async def _max_unlocked_tier(creature_id: int) -> int:
    async with (await db_pool()).acquire() as conn:
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
    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            row = await _get_progress(conn, creature_id, tier)
            if not row:
                wins = 1
                glyph_unlocked = (wins >= 5)
                await conn.execute(
                    "INSERT INTO creature_progress(creature_id, tier, wins, glyph_unlocked) VALUES($1,$2,$3,$4)",
                    creature_id, tier, wins, glyph_unlocked
                )
                return wins, glyph_unlocked
            wins = row["wins"] + 1
            glyph_unlocked = row["glyph_unlocked"]
            if not glyph_unlocked and wins >= 5:
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

# ─── Bot event handlers ───────────────────────────────────────
@bot.event
async def setup_hook():
    conn = await (await db_pool()).acquire()
    try:
        await conn.execute(SCHEMA_SQL)
    finally:
        await (await db_pool()).release(conn)
    await _backfill_creature_records()
    # Register commands to a specific guild if GUILD_ID is set, otherwise globally
    if GUILD_ID:
        guild_obj = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild_obj)
        await bot.tree.sync(guild=guild_obj)
        bot.tree.clear_commands(guild=None)
        await bot.tree.sync()
        logger.info("Synced commands to guild %s and cleared global commands.", GUILD_ID)
    else:
        await bot.tree.sync()
        logger.info("Synced commands globally")

    # Start background tasks
    for task_loop in (distribute_cash, distribute_points, regenerate_hp):
        if not task_loop.is_running():
            task_loop.start()
    # Initialize leaderboard if configured
    chan_id = await _get_leaderboard_channel_id()
    if chan_id:
        await _get_or_create_leaderboard_message(chan_id)
        await update_leaderboard_now(reason="startup")
    else:
        logger.info("No leaderboard channel configured. Use /setleaderboardchannel or set LEADERBOARD_CHANNEL_ID.")

@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)
    # Commands are already synced in setup_hook
    logger.info("Bot is ready. (%d commands active)", len(bot.tree.get_commands()))

# ─── Admin Commands ──────────────────────────────────────────
@bot.tree.command(description="(Admin) Set this channel as the live leaderboard channel")
async def setleaderboardchannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    await (await db_pool()).execute("""
        INSERT INTO leaderboard_messages(channel_id, message_id)
        VALUES ($1, NULL)
        ON CONFLICT (channel_id) DO UPDATE SET message_id = leaderboard_messages.message_id
    """, inter.channel.id)

    await inter.response.send_message(f"Leaderboard channel set to {inter.channel.mention}. Initializing…", ephemeral=True)
    await _get_or_create_leaderboard_message(inter.channel.id)
    await update_leaderboard_now(reason="admin_set_channel")

@bot.tree.command(description="(Admin) Set this channel as the Encyclopedia channel")
async def setencyclopediachannel(inter: discord.Interaction):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if not inter.channel:
        return await inter.response.send_message("Cannot determine channel.", ephemeral=True)

    await (await db_pool()).execute("""
        INSERT INTO encyclopedia_channel(channel_id)
        VALUES ($1)
        ON CONFLICT (channel_id) DO UPDATE SET channel_id = EXCLUDED.channel_id
    """, inter.channel.id)

    await inter.response.send_message(f"Encyclopedia channel set to {inter.channel.mention}.", ephemeral=True)

# ─── User Commands ───────────────────────────────────────────
@bot.tree.command(description="Register as a trainer")
async def register(inter: discord.Interaction):
    pool = await db_pool()
    # Only register if not already present
    if await pool.fetchval("SELECT 1 FROM trainers WHERE user_id=$1", inter.user.id):
        return await inter.response.send_message("Already registered!", ephemeral=True)
    await pool.execute(
        "INSERT INTO trainers(user_id, cash, trainer_points, facility_level, display_name) VALUES($1,$2,$3,$4,$5)",
        inter.user.id, 20000, 5, 1,
        getattr(inter.user, 'global_name', None) or getattr(inter.user, 'display_name', None) or inter.user.name
    )
    await inter.response.send_message(
        "Profile created! You received 20 000 cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a new creature egg (10 000 cash)")
async def spawn(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    if not await enforce_creature_cap(inter):
        return
    if row["cash"] < 10_000:
        return await inter.response.send_message("Not enough cash.", ephemeral=True)

    # Deduct cost and proceed with spawn
    await (await db_pool()).execute("UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", inter.user.id)
    await inter.response.defer(thinking=True)

    rarity = spawn_rarity()
    meta = await generate_creature_meta(rarity)
    stats = allocate_stats(rarity)
    max_hp = stats["HP"] * 5

    rec = await (await db_pool()).fetchrow(
        "INSERT INTO creatures(owner_id, name, rarity, descriptors, stats, current_hp)"
        "VALUES($1,$2,$3,$4,$5,$6) RETURNING id",
        inter.user.id, meta["name"], rarity, meta["descriptors"], json.dumps(stats), max_hp
    )
    await _ensure_record(inter.user.id, rec["id"], meta["name"])

    embed = discord.Embed(
        title=f"{meta['name']} ({rarity})",
        description="Descriptors: " + ", ".join(meta["descriptors"])
    )
    for s, v in stats.items():
        embed.add_field(name=s, value=str(v*5 if s == "HP" else v))
    embed.set_footer(text="Legendary spawn chance: 0.5%")
    await inter.followup.send(embed=embed)
    asyncio.create_task(update_leaderboard_now(reason="spawn"))

@bot.tree.command(description="List your creatures")
async def creatures(inter: discord.Interaction):
    if not await ensure_registered(inter):
        return
    await inter.response.defer(ephemeral=True)
    rows = await (await db_pool()).fetch(
        "SELECT id, name, rarity, descriptors, stats, current_hp FROM creatures WHERE owner_id=$1 ORDER BY id",
        inter.user.id
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

    for r in rows:
        st = json.loads(r["stats"])
        desc = ", ".join(r["descriptors"] or []) if (r["descriptors"] or []) else "None"
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
            f"Battles left today: **{left}/{DAILY_BATTLE_CAP}**",
        ]
        await inter.followup.send("\n".join(lines), ephemeral=True)

@bot.tree.command(description="See your creature's lifetime win/loss record")
async def record(inter: discord.Interaction, creature_name: str):
    if not await ensure_registered(inter):
        return
    row = await (await db_pool()).fetchrow(
        "SELECT name, wins, losses, is_dead, died_at FROM creature_records WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not row:
        return await inter.response.send_message("No record found for that creature name.", ephemeral=True)
    total = row["wins"] + row["losses"]
    wr = (row["wins"] / total * 100.0) if total > 0 else 0.0
    status = "💀 DEAD" if row["is_dead"] else "ALIVE"
    died_line = f"\nDied: {row['died_at']:%Y-%m-%d %H:%M %Z}" if row["is_dead"] and row["died_at"] else ""
    msg = (
        f"**{row['name']} – Lifetime Record**\n"
        f"Wins: **{row['wins']}** | Losses: **{row['losses']}** | Winrate: **{wr:.1f}%**\n"
        f"Status: **{status}**{died_line}"
    )
    await inter.response.send_message(msg, ephemeral=True)

@bot.tree.command(description="Sell one of your creatures for cash (price depends on rarity)")
async def sell(inter: discord.Interaction, creature_name: str):
    row = await ensure_registered(inter)
    if not row:
        return
    creature = await ensure_creature(inter, creature_name)
    if not creature:
        return
    st = active_battles.get(inter.user.id)
    if st and st.creature_id == creature["id"]:
        return await inter.response.send_message(
            f"**{creature['name']}** is currently in a battle. Finish or cancel the battle before selling.",
            ephemeral=True
        )
    rarity = creature["rarity"]
    price = SELL_PRICES.get(rarity, 0)
    await _ensure_record(inter.user.id, creature["id"], creature["name"])
    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM creatures WHERE id=$1", creature["id"])
            await conn.execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", price, inter.user.id)
    await inter.response.send_message(
        f"Sold **{creature['name']}** ({rarity}) for **{price}** cash.",
        ephemeral=True
    )
    asyncio.create_task(update_leaderboard_now(reason="sell"))

@bot.tree.command(description="Show glyphs and tier progress for a creature")
async def glyphs(inter: discord.Interaction, creature_name: str):
    if not await ensure_registered(inter):
        return
    creature = await ensure_creature(inter, creature_name)
    if not creature:
        return
    pool = await db_pool()
    progress: Dict[int, Tuple[int, bool]] = {}
    async with pool.acquire() as conn:
        for t in range(1, 10):
            row = await _get_progress(conn, creature["id"], t)
            wins = row["wins"] if row else 0
            glyph = (row["glyph_unlocked"] if row else False)
            progress[t] = (wins, glyph)
    max_tier = await _max_unlocked_tier(creature["id"])
    lines = [f"**{creature['name']} – Glyphs & Progress**"]
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

    # If multiple matches, attempt to narrow down to an exact match (case-insensitive)
    exact_matches = [r for r in rows if str(r["name"]).lower() == creature_name.lower()]
    if len(rows) > 1 and len(exact_matches) == 1:
        rows = exact_matches
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

    creature = await ensure_creature(inter, creature_name)
    if not creature:
        return
    allowed_tier = await _max_unlocked_tier(creature["id"])
    if tier > allowed_tier:
        prev_tier = tier - 1
        wins_prev = await _get_wins_for_tier(creature["id"], prev_tier)
        return await inter.response.send_message(
            f"Tier {tier} is locked for **{creature['name']}**. Current unlock: **1..{allowed_tier}**. "
            f"You need **5 wins at Tier {prev_tier}** to unlock Tier {tier} (current: {wins_prev}/5).",
            ephemeral=True
        )
    # Enforce daily battle limit for this creature
    allowed, count = await _can_start_battle_and_increment(creature["id"])
    if not allowed:
        return await inter.response.send_message(
            f"Daily battle cap reached for **{creature['name']}**: {DAILY_BATTLE_CAP}/{DAILY_BATTLE_CAP} used. "
            "Try again after midnight Europe/London.", ephemeral=True
        )

    # Begin battle simulation (one battle at a time per user)
    await inter.response.defer(thinking=True)
    user_cre = {"name": creature["name"], "stats": json.loads(creature["stats"])}
    rarity = rarity_for_tier(tier)
    opp_name = await generate_creature_name(rarity)
    extra = random.randint(*TIER_EXTRAS[tier])
    opp_stats = allocate_stats(rarity, extra)
    opp_cre = {"name": opp_name, "stats": opp_stats}
    # Initialize battle state for this user
    st = BattleState(
        inter.user.id, creature["id"], tier,
        user_cre, creature["current_hp"], creature["current_hp"] * 1,  # current HP and max HP (max is stats['HP']*5, but current HP is already full at start)
        opp_cre, opp_stats["HP"] * 5, opp_stats["HP"] * 5,
        logs=[]
    )
    active_battles[inter.user.id] = st
    # Log battle start information
    st.logs += [
        f"Battle start! Tier {tier} (+{extra} pts) — Daily battle use for {user_cre['name']}: {count}/{DAILY_BATTLE_CAP}",
        f"{user_cre['name']} vs {opp_cre['name']}",
        f"Opponent rarity (tier table) → {rarity}",
        "",
        "Your creature:",
        stat_block(user_cre["name"], st.user_current_hp, st.user_current_hp, user_cre["stats"]),
        "Opponent:",
        stat_block(opp_cre["name"], st.opp_current_hp, st.opp_current_hp, opp_stats),
        "",
        "Rules: Action weights A/Ag/Sp/Df = 38/22/22/18, Aggressive +25% dmg, Special ignores AR, "
        "AR softened (halved), extra swing at 1.5× SPD, +10% global damage every 10 rounds.",
        f"Daily cap: Each creature can start at most {DAILY_BATTLE_CAP} battles per Europe/London day.",
    ]
    max_tier = await _max_unlocked_tier(creature["id"])
    st.logs.append(f"Tier gate: {user_cre['name']} can currently queue Tier 1..{max_tier}.")

    # Announce battle start publicly
    start_msg = f"**Battle Start** — {user_cre['name']} vs {opp_cre['name']} (Tier {tier})\nOpponent rarity: **{rarity}**"
    await inter.followup.send(start_msg, ephemeral=False)
    # Simulate rounds in private logs
    st.next_log_idx = len(st.logs)
    while st.user_current_hp > 0 and st.opp_current_hp > 0:
        simulate_round(st)
        new_logs = st.logs[st.next_log_idx:]
        st.next_log_idx = len(st.logs)
        if new_logs:
            await send_chunks(inter, "\n".join(new_logs), ephemeral=True)
            await asyncio.sleep(0.2)

    # Battle ended: update creature's current HP in DB
    await (await db_pool()).execute(
        "UPDATE creatures SET current_hp=$1 WHERE id=$2",
        max(st.user_current_hp, 0), st.creature_id
    )
    # Determine winner and finalize battle outcome
    if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
        winner_name = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
        st.logs.append(f"Winner: {winner_name}")
        outcome = await finalize_battle(inter, st)
        # Remove active battle entry
        active_battles.pop(inter.user.id, None)
        trainer_name = await _resolve_trainer_name_from_db(inter.user.id) or (getattr(inter.user, 'display_name', None) or inter.user.name)
        remaining_logs = st.logs[st.next_log_idx:]
        st.next_log_idx = len(st.logs)
        if remaining_logs:
            await send_chunks(inter, "\n".join(remaining_logs), ephemeral=True)
            await asyncio.sleep(0.35)
        # Announce battle result publicly
        await inter.followup.send(format_public_battle_summary(st, outcome, trainer_name), ephemeral=False)
    # Ensure active battle entry is cleared in case of any errors
    active_battles.pop(inter.user.id, None)

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

    tgt = (target or "").strip().lower()
    if tgt in ("me", "self"):
        user_id = inter.user.id
    elif tgt == "all":
        result = await (await db_pool()).execute("UPDATE trainers SET cash = cash + $1", amount)
        try:
            updated_count = int(str(result).split()[-1])
        except Exception:
            updated_count = 0
        return await inter.response.send_message(f"Added {amount} cash to **all** trainers ({updated_count} records).", ephemeral=True)
    else:
        # Resolve target user by mention, ID, or display name
        user_id = await _extract_user_id_from_mention_or_id(target) or await _find_single_trainer_id_by_display_name(target)
        if user_id is None:
            return await inter.response.send_message(
                "Couldn't uniquely resolve that trainer. Try a mention, raw ID, or exact display name.",
                ephemeral=True
            )

    if not await (await db_pool()).fetchval("SELECT 1 FROM trainers WHERE user_id=$1", user_id):
        return await inter.response.send_message("That user isn't registered.", ephemeral=True)

    await (await db_pool()).execute("UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", amount, user_id)
    name = await _resolve_trainer_name_from_db(user_id) or str(user_id)
    await inter.response.send_message(f"Added **{amount}** cash to **{name}**.", ephemeral=True)

@bot.tree.command(description="(Admin) Add trainer points to a player by name/mention/id, or 'all'")
async def trainerpointsadd(inter: discord.Interaction, amount: int, target: str = "me"):
    if inter.user.id not in ADMIN_USER_IDS:
        return await inter.response.send_message("Not authorized to use this command.", ephemeral=True)
    if amount <= 0:
        return await inter.response.send_message("Positive amounts only.", ephemeral=True)

    tgt = (target or "").strip().lower()
    if tgt in ("me", "self"):
        user_id = inter.user.id
    elif tgt == "all":
        result = await (await db_pool()).execute("UPDATE trainers SET trainer_points = trainer_points + $1", amount)
        try:
            updated_count = int(str(result).split()[-1])
        except Exception:
            updated_count = 0
        return await inter.response.send_message(f"Added {amount} trainer points to **all** trainers ({updated_count} records).", ephemeral=True)
    else:
        user_id = await _extract_user_id_from_mention_or_id(target) or await _find_single_trainer_id_by_display_name(target)
        if user_id is None:
            return await inter.response.send_message(
                "Couldn't uniquely resolve that trainer. Try a mention, raw ID, or exact display name.",
                ephemeral=True
            )

    if not await (await db_pool()).fetchval("SELECT 1 FROM trainers WHERE user_id=$1", user_id):
        return await inter.response.send_message("That user isn't registered.", ephemeral=True)

    await (await db_pool()).execute("UPDATE trainers SET trainer_points = trainer_points + $1 WHERE user_id=$2", amount, user_id)
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

@bot.tree.command(description="Train a creature stat")
async def train(inter: discord.Interaction, creature_name: str, stat: str, increase: int):
    stat = stat.upper()
    if stat not in PRIMARY_STATS:
        return await inter.response.send_message(f"Stat must be one of {', '.join(PRIMARY_STATS)}.", ephemeral=True)
    if increase <= 0:
        return await inter.response.send_message("Increase must be positive.", ephemeral=True)

    row = await ensure_registered(inter)
    if not row or row["trainer_points"] < increase:
        return await inter.response.send_message("Not enough trainer points.", ephemeral=True)

    creature = await ensure_creature(inter, creature_name)
    if not creature:
        return
    stats = json.loads(creature["stats"])
    stats[stat] += increase
    new_max_hp = stats["HP"] * 5
    new_cur_hp = creature["current_hp"]
    if stat == "HP":
        new_cur_hp = min(new_cur_hp + increase * 5, new_max_hp)

    async with (await db_pool()).acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "UPDATE creatures SET stats=$1, current_hp=$2 WHERE id=$3",
                json.dumps(stats), new_cur_hp, creature["id"]
            )
            await conn.execute(
                "UPDATE trainers SET trainer_points = trainer_points - $1 WHERE user_id=$2",
                increase, inter.user.id
            )
    display_inc = increase * 5 if stat == "HP" else increase
    await inter.response.send_message(
        f"{creature['id']} – {creature_name.title()} trained: +{display_inc} {stat}.",
        ephemeral=True
    )

@bot.tree.command(description="Show and confirm upgrading your training facility")
async def upgrade(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    level = row["facility_level"]
    current = FACILITY_LEVELS[level]
    msg_lines = [
        f"**Your Training Facility**",
        f"Level {level}: **{current['name']}**",
        f"Bonus trainer points/day: +{current['bonus']} (total daily = {daily_trainer_points_for(level)})",
        f"Description: {current['desc']}",
        ""
    ]
    if level >= MAX_FACILITY_LEVEL:
        msg_lines.append("You're already at the **maximum level**. No further upgrades available.")
        return await send_chunks(inter, "\n".join(msg_lines), ephemeral=True)

    next_level = level + 1
    nxt = FACILITY_LEVELS[next_level]
    msg_lines += [
        f"**Next Upgrade → Level {next_level}: {nxt['name']}**",
        f"Cost: {nxt['cost']} cash",
        f"New bonus: +{nxt['bonus']} (daily total = {daily_trainer_points_for(next_level)})",
        f"Description: {nxt['desc']}",
        "",
        "Type `/upgradeyes` to confirm the upgrade if you can afford it."
    ]
    await send_chunks(inter, "\n".join(msg_lines), ephemeral=True)

@bot.tree.command(description="Confirm upgrading your training facility (costs cash)")
async def upgradeyes(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    level = row["facility_level"]
    if level >= MAX_FACILITY_LEVEL:
        return await inter.response.send_message("You're already at the maximum facility level.", ephemeral=True)

    next_level = level + 1
    cost = FACILITY_LEVELS[next_level]["cost"]
    if row["cash"] < cost:
        return await inter.response.send_message(
            f"Not enough cash. You need {cost} but only have {row['cash']}.",
            ephemeral=True
        )

    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash - $1, facility_level = facility_level + 1 WHERE user_id=$2",
        cost, inter.user.id
    )
    new_bonus = FACILITY_LEVELS[next_level]["bonus"]
    await inter.response.send_message(
        f"✅ Upgraded to **Level {next_level} – {FACILITY_LEVELS[next_level]['name']}**!\n"
        f"Your facility now grants **+{new_bonus} trainer points/day** "
        f"(total {daily_trainer_points_for(next_level)}/day).",
        ephemeral=True
    )

@bot.tree.command(description="Add one of your creatures to the Encyclopedia")
async def enc(inter: discord.Interaction, creature_name: str):
    if not await ensure_registered(inter):
        return

    creature = await ensure_creature(inter, creature_name)
    if not creature:
        return
    # Gate: Only allow Common/Uncommon if >10 wins or Glyph 3 unlocked
    rarity_val = str(creature["rarity"])
    if rarity_val in ("Common", "Uncommon"):
        conn = await (await db_pool()).acquire()
        try:
            total_wins = await conn.fetchval(
                "SELECT COALESCE(SUM(wins), 0) FROM creature_progress WHERE creature_id=$1",
                creature["id"]
            )
            glyph3_unlocked = await conn.fetchval(
                "SELECT glyph_unlocked FROM creature_progress WHERE creature_id=$1 AND tier=3",
                creature["id"]
            )
        finally:
            await (await db_pool()).release(conn)
        if (total_wins or 0) <= 10 and not bool(glyph3_unlocked):
            return await inter.response.send_message(
                f"Encyclopedia entry denied for **{creature['name']}** ({rarity_val}). "
                "Common/Uncommon creatures must have **>10 total wins** or **Glyph 3 unlocked**.\n"
                f"Current: Wins **{(total_wins or 0)}**, Glyph 3 {'✅' if glyph3_unlocked else '❌'}.",
                ephemeral=True
            )

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
    # Prepare data for encyclopedia entry
    name = creature["name"]
    rarity = creature["rarity"]
    traits = creature["descriptors"] or []
    stats = json.loads(creature["stats"])
    stats_block = _format_stats_block(stats)
    # Generate bio and image via OpenAI
    bio, image_url, image_bytes = await _gpt_generate_bio_and_image(name, rarity, traits, stats)
    # Build embed for the encyclopedia entry
    embed = discord.Embed(title=f"{name} — {rarity}", description=bio)
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
        # Post entry to encyclopedia channel
        if file_to_send:
            msg = await channel.send(embed=embed, file=file_to_send)
        else:
            msg = await channel.send(embed=embed)
    except Exception as e:
        logger.error("Failed to post encyclopedia entry: %s", e)
        return await inter.followup.send("Failed to post to the Encyclopedia channel.", ephemeral=True)

    await inter.followup.send(f"Added **{name}** to the Encyclopedia: {msg.jump_url}", ephemeral=True)

@bot.tree.command(description="Show all commands and what they do")
async def info(inter: discord.Interaction):
    # Brief overview and dynamic command list
    caps_line = f"• Passive income: 60 cash/hour | Creature cap: {MAX_CREATURES} | Daily cap: {DAILY_BATTLE_CAP}/creature/day (Europe/London)."
    header = (
        "**Game Overview**\n"
        "Collect, train, and battle creatures. Progress through tiers to unlock glyphs.\n"
        + caps_line + "\n"
    )
    try:
        command_list_text = _build_command_list(bot)
    except Exception:
        command_list_text = "Failed to build command list."
    content = header + "\n" + command_list_text
    await send_chunks(inter, content, ephemeral=True)

if __name__ == "__main__":
    bot.run(TOKEN)
