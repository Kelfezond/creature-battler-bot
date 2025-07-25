from __future__ import annotations
import asyncio, json, logging, math, os, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import discord
from discord.ext import commands, tasks
import openai

# ─── Basic config & logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN     = os.getenv("DISCORD_TOKEN")
DB_URL    = os.getenv("DATABASE_URL")
GUILD_ID  = os.getenv("GUILD_ID") or None
openai.api_key = os.getenv("OPENAI_API_KEY")

for env_name, env_val in {
    "DISCORD_TOKEN": TOKEN,
    "DATABASE_URL": DB_URL,
    "OPENAI_API_KEY": openai.api_key,
}.items():
    if not env_val:
        raise RuntimeError(f"Missing environment variable: {env_name}")

# ─── Discord client ──────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
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
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game constants ──────────────────────────────────────────
MAX_CREATURES = 5  # hard cap per player

# Used for /spawn eggs – we overrode this with spawn_rarity() below to make Legendary 0.5%
RARITY_TABLE = [
    (1, 75, "Common"), (76, 88, "Uncommon"), (89, 95, "Rare"),
    (96, 98, "Epic"), (99, 100, "Legendary"),
]

# NEW: /spawn rarity distribution with Legendary at **0.5%**
# Common 75%, Uncommon 13%, Rare 7%, Epic 4.5%, Legendary 0.5%
def spawn_rarity() -> str:
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

# Per-tier rarity weights for *battle opponents*
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

# Training Facility progression
MAX_FACILITY_LEVEL = 6
FACILITY_LEVELS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Basic Training Yard",
        "bonus": 0,
        "cost": None,
        "desc": "A patch of land with scattered targets, sand pits, and makeshift climbing posts. Rough, simple, and functional."
    },
    2: {
        "name": "Reinforced Combat Pit",
        "bonus": 1,
        "cost": 18_000,
        "desc": "Expanded grounds with adjustable barriers, weighted obstacles, sky hoops, and water trenches. Built to be tough and versatile."
    },
    3: {
        "name": "Kinetic Optimization Center",
        "bonus": 2,
        "cost": 55_000,
        "desc": "Modular platforms with reactive surfaces, telescoping tracks, and pressure pads. The environment adapts to suit each training style."
    },
    4: {
        "name": "Neuro-Combat Simulator",
        "bonus": 3,
        "cost": 130_000,
        "desc": "Holographic arenas simulate dynamic opponents and shifting terrain. Training is personalized and reactive in real time."
    },
    5: {
        "name": "BioSync Reactor Chamber",
        "bonus": 4,
        "cost": 275_000,
        "desc": "A synchronized chamber tuned to physical and mental rhythms. Terrain and resistance fields shift unpredictably to enhance reflex development."
    },
    6: {
        "name": "SynapseForge Hyperlab",
        "bonus": 5,
        "cost": 500_000,
        "desc": "A high-tech fusion of neural feedback, virtual training microcosms, and time-compressed simulations. Mastery is forged at the speed of thought."
    },
}

def facility_bonus(level: int) -> int:
    level = max(1, min(MAX_FACILITY_LEVEL, level))
    return FACILITY_LEVELS[level]["bonus"]

def daily_trainer_points_for(level: int) -> int:
    # Base 5 + facility bonus (max +5) = max 10/day
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

ACTIONS = ["Attack", "Aggressive", "Special", "Defend"]
ACTION_WEIGHTS = [36, 18, 16, 30]   # sum = 100

# ─── Tier Payouts (Approach A: independently rounded to nearest 10) ───────────
# Structure: tier: (win_cash, loss_cash)
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
    next_log_idx: int = 0
    rounds: int = 0

active_battles: Dict[int, BattleState] = {}

# ─── Scheduled rewards & regen ───────────────────────────────
@tasks.loop(hours=1)
async def distribute_cash():
    if distribute_cash.current_loop == 0:
        logger.info("Skipping first hourly cash distribution after restart")
        return
    # Passive income: 60 cash per hour
    await (await db_pool()).execute("UPDATE trainers SET cash = cash + 60")
    logger.info("Distributed 60 cash to all trainers")

@tasks.loop(hours=24)
async def distribute_points():
    if distribute_points.current_loop == 0:
        logger.info("Skipping first daily trainer‑point distribution after restart")
        return
    # Base 5 + facility bonus (max 5) = up to 10/day.
    await (await db_pool()).execute("""
        UPDATE trainers
        SET trainer_points = trainer_points
          + (5 + LEAST(GREATEST(facility_level - 1, 0), 5))
    """)
    logger.info("Distributed daily trainer points with facility bonuses")

@tasks.loop(hours=12)
async def regenerate_hp():
    """
    Heal every creature by 20 % of its max HP (ceil), up to its max.
    (Now: +1.0 × base HP stat, since max = HP * 5.)
    """
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

def rarity_from_roll(r: int) -> str:
    # kept for backwards compatibility, but /spawn uses spawn_rarity()
    for low, high, name in RARITY_TABLE:
        if low <= r <= high:
            return name
    return "Common"

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
    return (
        f"{name} – HP:{cur_hp}/{max_hp} "
        f"AR:{s['AR']} PATK:{s['PATK']} SATK:{s['SATK']} SPD:{s['SPD']}"
    )

def choose_action() -> str:
    return random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]

async def ensure_registered(inter: discord.Interaction) -> Optional[asyncpg.Record]:
    row = await (await db_pool()).fetchrow(
        "SELECT cash, trainer_points, facility_level FROM trainers WHERE user_id=$1", inter.user.id
    )
    if not row:
        await inter.response.send_message("Use /register first.", ephemeral=True)
        return None
    return row

# Creature-cap helpers
async def get_creature_count(user_id: int) -> int:
    return await (await db_pool()).fetchval(
        "SELECT COUNT(*) FROM creatures WHERE owner_id=$1", user_id
    )

async def enforce_creature_cap(inter: discord.Interaction) -> bool:
    count = await get_creature_count(inter.user.id)
    if count >= MAX_CREATURES:
        await inter.response.send_message(
            f"You already own the maximum of {MAX_CREATURES} creatures. "
            "Release one before spawning a new egg.",
            ephemeral=True
        )
        return False
    return True

async def generate_creature_meta(rarity: str) -> Dict[str, Any]:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    used_names = [r["name"].lower() for r in rows]
    used_words = {w.lower() for r in rows for w in r["descriptors"]}
    prompt = f"""
Invent a creature of rarity **{rarity}**. Return ONLY JSON:
{{"name":"1-3 words","descriptors":["w1","w2","w3"]}}
Avoid names: {', '.join(used_names)}
Avoid words: {', '.join(used_words)}
"""
    for _ in range(3):
        try:
            resp = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0, max_tokens=100,
                )
            )
            data = json.loads(resp.choices[0].message.content.strip())
            if "name" in data and len(data.get("descriptors", [])) == 3:
                data["name"] = data["name"].title()
                return data
        except Exception as e:
            logger.error("OpenAI error: %s", e)
    return {"name": f"Wild{random.randint(1000,9999)}", "descriptors": []}

def simulate_round(st: BattleState):
    st.rounds += 1
    st.logs.append(f"Round {st.rounds}")

    # If both pick Defend, silently re-roll until at least one doesn't.
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
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        if act == "Defend":
            st.logs.append(f"{atk['name']} is defending.")
            continue

        swings = 2 if atk["stats"]["SPD"] >= 2 * dfn["stats"]["SPD"] else 1
        for _ in range(swings):
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                break

            S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
            AR_val = 0 if act == "Special" else dfn["stats"]["AR"]
            rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
            dmg = max(1, math.ceil(sum(rolls) ** 2 / (sum(rolls) + AR_val)))
            if act == "Aggressive":
                dmg = math.ceil(dmg * 1.1)
            if dfn_act == "Defend":
                dmg = max(1, math.ceil(dmg * 0.5))

            if side == "user":
                st.opp_current_hp -= dmg
            else:
                st.user_current_hp -= dmg

            act_word = {
                "Attack": "hits",
                "Aggressive": "aggressively hits",
                "Special": "unleashes a special attack on"
            }[act]
            note = " (defended)" if dfn_act == "Defend" else ""
            st.logs.append(
                f"{atk['name']} {act_word} {dfn['name']} for {dmg} dmg"
                f"{' (rolls '+str(rolls)+')' if act!='Special' else ''}{note}"
            )

            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                st.logs.append(f"{dfn['name']} is down!")
                break

    st.logs.append(
        f"{st.user_creature['name']} HP {max(st.user_current_hp,0)}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {max(st.opp_current_hp,0)}/{st.opp_max_hp}"
    )
    st.logs.append("")

async def send_chunks(inter: discord.Interaction, content: str):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    sender = inter.followup.send if inter.response.is_done() else inter.response.send_message
    await sender(chunks[0])
    for chunk in chunks[1:]:
        await inter.followup.send(chunk)

async def finalize_battle(inter: discord.Interaction, st: BattleState):
    """Handle end-of-battle rewards, death chance, and logging."""
    player_won = st.opp_current_hp <= 0 and st.user_current_hp > 0
    win_cash, loss_cash = TIER_PAYOUTS[st.tier]
    payout = win_cash if player_won else loss_cash
    pool = await db_pool()
    await pool.execute(
        "UPDATE trainers SET cash = cash + $1 WHERE user_id=$2",
        payout, st.user_id
    )
    result_word = "won" if player_won else "lost"
    st.logs.append(f"You {result_word} the Tier {st.tier} battle: +{payout} cash awarded.")
    # 50% death chance if player lost
    if not player_won:
        death_roll = random.random()
        pct = int(death_roll * 100)
        if death_roll < 0.5:
            await pool.execute("DELETE FROM creatures WHERE id=$1", st.creature_id)
            st.logs.append(
                f"💀 Death roll {pct} (<50): Your creature **{st.user_creature['name']}** died and was removed."
            )
        else:
            st.logs.append(
                f"🛡️ Death roll {pct} (≥50): Your creature **{st.user_creature['name']}** survived the defeat."
            )

# ─── Bot events ──────────────────────────────────────────────
@bot.event
async def setup_hook():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    if GUILD_ID:
        bot.tree.copy_global_to(guild=discord.Object(id=int(GUILD_ID)))
        await bot.tree.sync(guild=discord.Object(id=int(GUILD_ID)))
        logger.info("Synced to guild %s", GUILD_ID)
    else:
        await bot.tree.sync()
        logger.info("Synced globally")

    for loop in (distribute_cash, distribute_points, regenerate_hp):
        if not loop.is_running():
            loop.start()

@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)

# ─── Slash commands ─────────────────────────────────────────
@bot.tree.command(description="Register as a trainer")
async def register(inter: discord.Interaction):
    pool = await db_pool()
    if await pool.fetchval("SELECT 1 FROM trainers WHERE user_id=$1", inter.user.id):
        return await inter.response.send_message("Already registered!", ephemeral=True)
    await pool.execute(
        "INSERT INTO trainers(user_id, cash, trainer_points, facility_level) VALUES($1,$2,$3,$4)",
        inter.user.id, 20000, 5, 1
    )
    await inter.response.send_message(
        "Profile created! You received 20 000 cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a new creature egg (10 000 cash)")
async def spawn(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row:
        return
    # Enforce cap BEFORE taking their money
    can_spawn = await enforce_creature_cap(inter)
    if not can_spawn:
        return
    if row["cash"] < 10_000:
        return await inter.response.send_message("Not enough cash.", ephemeral=True)

    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", inter.user.id
    )
    await inter.response.defer(thinking=True)

    rarity = spawn_rarity()  # 0.5% legendary table
    meta = await generate_creature_meta(rarity)
    stats = allocate_stats(rarity)
    max_hp = stats["HP"] * 5

    await (await db_pool()).execute(
        "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats,current_hp)"
        "VALUES($1,$2,$3,$4,$5,$6)",
        inter.user.id, meta["name"], rarity, meta["descriptors"], json.dumps(stats), max_hp
    )
    embed = discord.Embed(
        title=f"{meta['name']} ({rarity})",
        description="Descriptors: " + ", ".join(meta["descriptors"])
    )
    for s, v in stats.items():
        embed.add_field(name=s, value=str(v*5 if s == "HP" else v))
    embed.set_footer(text="Legendary spawn chance: 0.5%")
    await inter.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(inter: discord.Interaction):
    if not await ensure_registered(inter):
        return
    rows = await (await db_pool()).fetch(
        "SELECT id,name,rarity,descriptors,stats,current_hp FROM creatures "
        "WHERE owner_id=$1 ORDER BY id", inter.user.id
    )
    if not rows:
        return await inter.response.send_message("You own no creatures.", ephemeral=True)
    lines = []
    for idx, r in enumerate(rows, 1):
        st = json.loads(r["stats"])
        desc = ", ".join(r["descriptors"]) or "None"
        max_hp = st["HP"] * 5
        lines.append(
            f"{idx}. **{r['name']}** ({r['rarity']}) – {desc} | "
            f"HP:{r['current_hp']}/{max_hp} AR:{st['AR']} PATK:{st['PATK']} "
            f"SATK:{st['SATK']} SPD:{st['SPD']}"
        )
    await inter.response.send_message("\n".join(lines), ephemeral=True)

@bot.tree.command(description="Battle one of your creatures vs. a tiered opponent")
async def battle(inter: discord.Interaction, creature_name: str, tier: int):
    if tier not in TIER_EXTRAS:
        return await inter.response.send_message("Invalid tier (1‑9).", ephemeral=True)
    if inter.user.id in active_battles:
        return await inter.response.send_message(
            "You already have an active battle – use /continue.", ephemeral=True
        )
    if not await ensure_registered(inter):
        return

    c_row = await (await db_pool()).fetchrow(
        "SELECT id,name,stats,current_hp FROM creatures "
        "WHERE owner_id=$1 AND name ILIKE $2", inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    stats = json.loads(c_row["stats"])
    max_hp = stats["HP"] * 5
    if c_row["current_hp"] <= 0:
        return await inter.response.send_message(
            f"{c_row['name']} has fainted and needs healing.", ephemeral=True
        )

    await inter.response.defer(thinking=True)

    user_cre = {"name": c_row["name"], "stats": stats}

    rarity = rarity_for_tier(tier)
    meta = await generate_creature_meta(rarity)
    extra = random.randint(*TIER_EXTRAS[tier])
    opp_stats = allocate_stats(rarity, extra)
    opp_cre = {"name": meta["name"], "stats": opp_stats}

    st = BattleState(
        inter.user.id, c_row["id"], tier,
        user_cre, c_row["current_hp"], max_hp,
        opp_cre, opp_stats["HP"] * 5, opp_stats["HP"] * 5,
        logs=[]
    )
    active_battles[inter.user.id] = st
    st.logs += [
        f"Battle start! Tier {tier} (+{extra} pts)",
        f"{user_cre['name']} vs {opp_cre['name']}",
        f"Opponent rarity (tier table) → {rarity}",
        "",
        "Your creature:",
        stat_block(user_cre["name"], st.user_current_hp, st.user_max_hp, stats),
        "Opponent:",
        stat_block(opp_cre["name"], st.opp_max_hp, st.opp_max_hp, opp_stats),
        ""
    ]

    for _ in range(10):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        simulate_round(st)

    await (await db_pool()).execute(
        "UPDATE creatures SET current_hp=$1 WHERE id=$2",
        max(st.user_current_hp, 0), st.creature_id
    )

    if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
        winner = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
        st.logs.append(f"Winner: {winner}")
        await finalize_battle(inter, st)
        active_battles.pop(inter.user.id, None)
    else:
        st.logs.append("Use /continue to proceed.")
    await send_chunks(inter, "\n".join(st.logs))
    st.next_log_idx = len(st.logs)

@bot.tree.command(name="continue", description="Continue your current battle")
async def continue_battle(inter: discord.Interaction):
    st = active_battles.get(inter.user.id)
    if not st:
        return await inter.response.send_message("No active battle.", ephemeral=True)

    await inter.response.defer(thinking=True)
    # Simulate up to 10 more rounds or until someone faints
    for _ in range(10):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        simulate_round(st)

    # Persist HP (safe even if potential deletion after finalize)
    await (await db_pool()).execute(
        "UPDATE creatures SET current_hp=$1 WHERE id=$2",
        max(st.user_current_hp, 0), st.creature_id
    )

    battle_ended = st.user_current_hp <= 0 or st.opp_current_hp <= 0
    if battle_ended:
        winner = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
        st.logs.append(f"Winner: {winner}")
        await finalize_battle(inter, st)
        active_battles.pop(inter.user.id, None)
        new_logs = st.logs[st.next_log_idx:]
        st.next_log_idx = len(st.logs)
        await send_chunks(inter, "\n".join(new_logs))
        return

    # Not ended: just show incremental rounds
    st.logs.append("Use /continue to proceed.")
    new_logs = st.logs[st.next_log_idx:]
    st.next_log_idx = len(st.logs)
    await send_chunks(inter, "\n".join(new_logs))

@bot.tree.command(description="Check your cash")
async def cash(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        await inter.response.send_message(f"You have {row['cash']} cash.", ephemeral=True)

@bot.tree.command(description="Add cash (dev utility)")
async def cashadd(inter: discord.Interaction, amount: int):
    if amount <= 0:
        return await inter.response.send_message("Positive amounts only.", ephemeral=True)
    if not await ensure_registered(inter):
        return
    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", amount, inter.user.id
    )
    await inter.response.send_message(f"Added {amount} cash.", ephemeral=True)

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
        return await inter.response.send_message(
            f"Stat must be one of {', '.join(PRIMARY_STATS)}.", ephemeral=True
        )
    if increase <= 0:
        return await inter.response.send_message("Increase must be positive.", ephemeral=True)

    row = await ensure_registered(inter)
    if not row or row["trainer_points"] < increase:
        return await inter.response.send_message("Not enough trainer points.", ephemeral=True)

    c = await (await db_pool()).fetchrow(
        "SELECT id,stats,current_hp FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    stats = json.loads(c["stats"])
    stats[stat] += increase
    new_max_hp = stats["HP"] * 5
    new_cur_hp = c["current_hp"]
    if stat == "HP":
        new_cur_hp += increase * 5
        new_cur_hp = min(new_cur_hp, new_max_hp)

    await (await db_pool()).execute(
        "UPDATE creatures SET stats=$1,current_hp=$2 WHERE id=$3",
        json.dumps(stats), new_cur_hp, c["id"]
    )
    await (await db_pool()).execute(
        "UPDATE trainers SET trainer_points = trainer_points - $1 WHERE user_id=$2",
        increase, inter.user.id
    )
    display_inc = increase * 5 if stat == "HP" else increase
    await inter.response.send_message(
        f"{c['id']} – {creature_name.title()} trained: +{display_inc} {stat}.",
        ephemeral=True
    )

# ─── Training Facility Upgrade Commands ──────────────────────
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
        "Type `/upgradeyes` to confirm the upgrade if you can afford it."
    ]
    await inter.response.send_message("\n".join(msg), ephemeral=True)

@bot.tree.command(description="Confirm upgrading your training facility (costs cash)")
async def upgradeyes(inter: discord.Interaction):
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
            ephemeral=True
        )

    pool = await db_pool()
    await pool.execute(
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

# /info command
@bot.tree.command(description="Show game overview and command list")
async def info(inter: discord.Interaction):
    overview = (
        "**Game Overview**\n"
        "Collect creatures, train their stats, and battle tiered opponents.\n"
        "• Passive income: 60 cash/hour.\n"
        "• Creature cap: You can own at most **5 creatures**. Extra spawns are blocked.\n"
        "• **Spawn eggs Legendary chance: 0.5%** (others adjusted accordingly).\n"
        "• **Battle opponent rarity by tier**:\n"
        "  - T1–2: Common only\n"
        "  - T3–4: Common/Uncommon (75/25)\n"
        "  - T5–6: Common/Uncommon/Rare (50/33/16)\n"
        "  - T7–8: Common/Uncommon/Rare/Epic (40/30/20/10)\n"
        "  - T9: Common/Uncommon/Rare/Epic/Legendary (33/26/20/13/6)\n"
        "• If both creatures pick **Defend** in a round, it is silently re‑rolled until at least one doesn't defend.\n"
        "• **Training Facilities**: Start at Level 1 (Basic Training Yard). Each level adds +1 trainer point/day up to +5 (Level 6). Base income is 5/day → max 10/day.\n"
        "  Use `/upgrade` to view & `/upgradeyes` to confirm if you can afford it.\n"
        "• Battles occur in rounds; continue long fights with `/continue`.\n"
        "• Tier payouts scale from 1k/500 (T1 W/L) up to 50k/25k (T9 W/L).\n"
        "• If you *lose* a battle your creature has a 50% chance to **permanently die**.\n"
        "• Trainer points (daily +5 plus facility bonus) are spent to increase stats. HP increases also raise current HP.\n"
        "\n"
        "**Commands**\n"
        "/register – Create your trainer profile (one-time).\n"
        "/spawn – Spend 10,000 cash to hatch a new creature egg (blocked if you already have 5 creatures).\n"
        "/creatures – List your creatures and their stats.\n"
        "/battle <creature_name> <tier> – Start a battle (tiers 1–9).\n"
        "/continue – Continue your current battle (up to 10 more rounds per use).\n"
        "/cash – Show your current cash.\n"
        "/cashadd <amount> – (Dev) Add test cash to your account.\n"
        "/trainerpoints – Show your remaining trainer points and facility bonus.\n"
        "/train <creature_name> <stat> <increase> – Spend trainer points to raise a stat.\n"
        "/upgrade – View your facility and the cost to upgrade.\n"
        "/upgradeyes – Confirm the upgrade and pay the cost.\n"
        "/info – Show this help & overview.\n"
        "\n"
        "**Stats**: HP (health pool*5), AR (defense), PATK, SATK, SPD (initiative; may grant extra swing).\n"
        "**Actions**: Attack, Aggressive (+10% dmg), Special (ignores AR), Defend (halve incoming dmg; double-defend rerolled).\n"
        "**Death**: On a loss, 50% chance (random < 0.5) your creature is deleted.\n"
        "\n"
        "Good luck, Trainer!"
    )
    await inter.response.send_message(overview, ephemeral=True)

# ─── Launch ──────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(TOKEN)
