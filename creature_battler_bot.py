from __future__ import annotations
import asyncio, json, logging, math, os, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import discord
from discord.ext import commands, tasks
import openai              # pre‑1.0 SDK

# ─── Basic config & logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN     = os.getenv("DISCORD_TOKEN")
DB_URL    = os.getenv("DATABASE_URL")
GUILD_ID  = os.getenv("GUILD_ID") or None
openai.api_key = os.getenv("OPENAI_API_KEY")

for name, val in {
    "DISCORD_TOKEN": TOKEN,
    "DATABASE_URL": DB_URL,
    "OPENAI_API_KEY": openai.api_key,
}.items():
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")

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
  trainer_points BIGINT DEFAULT 0
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
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game constants ──────────────────────────────────────────
RARITY_TABLE: List[Tuple[int, int, str]] = [
    (1, 75, "Common"), (76, 88, "Uncommon"), (89, 95, "Rare"),
    (96, 98, "Epic"),  (99, 100, "Legendary"),
]
POINT_POOLS = {
    "Common": (25, 50), "Uncommon": (50, 100), "Rare": (100, 200),
    "Epic": (200, 400), "Legendary": (400, 800),
}
TIER_EXTRAS = {
    1: (0, 10),  2: (10, 30), 3: (30, 60), 4: (60, 100),
    5: (100, 140), 6: (140, 180), 7: (180, 220),
    8: (220, 260), 9: (260, 300)          # fixed overlap with tier 8
}
PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]

# ─── Battle state ────────────────────────────────────────────
@dataclass
class BattleState:
    user_id: int
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

# ─── Scheduled rewards (skip first run) ──────────────────────
@tasks.loop(hours=1)
async def distribute_cash():
    if distribute_cash.current_loop == 0:
        logger.info("Skipping first hourly cash distribution after restart")
        return
    await (await db_pool()).execute("UPDATE trainers SET cash = cash + 400")
    logger.info("Distributed 400 cash to all trainers")

@tasks.loop(hours=24)
async def distribute_points():
    if distribute_points.current_loop == 0:
        logger.info("Skipping first daily trainer‑point distribution after restart")
        return
    await (await db_pool()).execute("UPDATE trainers SET trainer_points = trainer_points + 5")
    logger.info("Distributed 5 trainer points to all trainers")

# ─── Utility functions ───────────────────────────────────────
def roll_d100() -> int: return random.randint(1, 100)

def rarity_from_roll(r: int) -> str:
    for low, high, name in RARITY_TABLE:
        if low <= r <= high:
            return name
    return "Common"

def allocate_stats(rarity: str, extra: int = 0) -> Dict[str, int]:
    pool = random.randint(*POINT_POOLS[rarity]) + extra
    stats = {k: 1 for k in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)
    for _ in range(pool):
        stats[random.choice(PRIMARY_STATS)] += 1
    return stats

def stat_block(cre: Dict[str, Any], max_hp: int) -> str:
    s = cre["stats"]
    return (
        f"{cre['name']} – HP:{max_hp} "
        f"AR:{s['AR']} PATK:{s['PATK']} SATK:{s['SATK']} SPD:{s['SPD']}"
    )

async def fetch_used_lists() -> Tuple[List[str], List[str]]:
    rows = await (await db_pool()).fetch("SELECT name, descriptors FROM creatures")
    names = [r["name"].lower() for r in rows]
    words = {w.lower() for r in rows for w in r["descriptors"]}
    return names, list(words)

async def ensure_registered(inter: discord.Interaction) -> Optional[asyncpg.Record]:
    row = await (await db_pool()).fetchrow(
        "SELECT cash, trainer_points FROM trainers WHERE user_id=$1", inter.user.id
    )
    if not row:
        await inter.response.send_message("Use /register first.", ephemeral=True)
        return None
    return row

async def generate_creature_meta(rarity: str) -> Dict[str, Any]:
    names_used, words_used = await fetch_used_lists()
    prompt = f"""
Invent a creature of rarity **{rarity}**. Return ONLY JSON:
{{"name":"1‑3 words","descriptors":["w1","w2","w3"]}}
Avoid names: {', '.join(names_used)}
Avoid words: {', '.join(words_used)}
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

    # Current HP summary (before attacks)
    st.logs.append(
        f"{st.user_creature['name']} HP {st.user_current_hp}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {st.opp_current_hp}/{st.opp_max_hp}"
    )

    uc, oc = st.user_creature, st.opp_creature
    order = [("user", uc, oc), ("opp", oc, uc)]
    if uc["stats"]["SPD"] < oc["stats"]["SPD"] or (
        uc["stats"]["SPD"] == oc["stats"]["SPD"] and random.choice([0, 1])
    ):
        order.reverse()

    for side, atk, dfn in order:
        attacks = 2 if atk["stats"]["SPD"] >= 2 * dfn["stats"]["SPD"] else 1
        for _ in range(attacks):
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                return
            S = max(atk["stats"]["PATK"], atk["stats"]["SATK"])
            rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
            dmg = max(1, math.ceil(sum(rolls) ** 2 / (sum(rolls) + dfn["stats"]["AR"])))
            if side == "user":
                st.opp_current_hp -= dmg
            else:
                st.user_current_hp -= dmg
            st.logs.append(f"{atk['name']} hits for {dmg} (rolls {rolls})")
            if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
                st.logs.append(f"{dfn['name']} is down!")
                return

    # HP summary after attacks
    st.logs.append(
        f"{st.user_creature['name']} HP {max(st.user_current_hp,0)}/{st.user_max_hp} | "
        f"{st.opp_creature['name']} HP {max(st.opp_current_hp,0)}/{st.opp_max_hp}"
    )
    st.logs.append("")

async def send_chunks(inter: discord.Interaction, content: str):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    first_sender = (
        inter.followup.send if inter.response.is_done() else inter.response.send_message
    )
    await first_sender(chunks[0])
    for chunk in chunks[1:]:
        await inter.followup.send(chunk)

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

    if not distribute_cash.is_running():
        distribute_cash.start()
    if not distribute_points.is_running():
        distribute_points.start()

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
        "INSERT INTO trainers(user_id, cash, trainer_points) VALUES($1,$2,$3)",
        inter.user.id, 20000, 5
    )
    await inter.response.send_message(
        "Profile created! You received 20 000 cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a new creature egg (10 000 cash)")
async def spawn(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if not row or row["cash"] < 10_000:
        return await inter.response.send_message("Not enough cash.", ephemeral=True)

    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", inter.user.id
    )
    await inter.response.defer(thinking=True)

    roll, rarity = roll_d100(), None
    rarity = rarity_from_roll(roll)
    meta   = await generate_creature_meta(rarity)
    stats  = allocate_stats(rarity)

    await (await db_pool()).execute(
        "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats)"
        "VALUES($1,$2,$3,$4,$5)",
        inter.user.id, meta["name"], rarity, meta["descriptors"], json.dumps(stats)
    )
    embed = discord.Embed(
        title=f"{meta['name']} ({rarity})",
        description="Descriptors: " + ", ".join(meta["descriptors"])
    )
    for s,v in stats.items():
        embed.add_field(name=s, value=str(v*5 if s=="HP" else v))
    embed.set_footer(text=f"d100 roll: {roll}")
    await inter.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(inter: discord.Interaction):
    if not await ensure_registered(inter):
        return
    rows = await (await db_pool()).fetch(
        "SELECT name,rarity,descriptors,stats FROM creatures "
        "WHERE owner_id=$1 ORDER BY id", inter.user.id
    )
    if not rows:
        return await inter.response.send_message("You own no creatures.", ephemeral=True)
    lines = []
    for idx, r in enumerate(rows, 1):
        st = json.loads(r["stats"])
        desc = ", ".join(r["descriptors"]) or "None"
        lines.append(
            f"{idx}. **{r['name']}** ({r['rarity']}) – {desc} | "
            f"HP:{st['HP']*5} AR:{st['AR']} PATK:{st['PATK']} SATK:{st['SATK']} SPD:{st['SPD']}"
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
    row = await ensure_registered(inter)
    if not row:
        return

    c_row = await (await db_pool()).fetchrow(
        "SELECT name,stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c_row:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    await inter.response.defer(thinking=True)

    user_cre = {"name": c_row["name"], "stats": json.loads(c_row["stats"])}
    roll     = roll_d100()
    rarity   = rarity_from_roll(roll)
    meta     = await generate_creature_meta(rarity)
    extra    = random.randint(*TIER_EXTRAS[tier])
    opp_cre  = {"name": meta["name"], "stats": allocate_stats(rarity, extra)}

    st = BattleState(
        inter.user.id,
        user_cre, user_cre["stats"]["HP"]*5, user_cre["stats"]["HP"]*5,
        opp_cre, opp_cre["stats"]["HP"]*5,  opp_cre["stats"]["HP"]*5,
        logs=[]
    )
    active_battles[inter.user.id] = st
    # Battle header + stat comparison
    st.logs += [
        f"Battle start! Tier {tier} (+{extra} pts)",
        f"{user_cre['name']} vs {opp_cre['name']}",
        f"Opponent roll {roll} → {rarity}",
        "",
        "Your creature:",
        stat_block(user_cre, st.user_max_hp),
        "Opponent:",
        stat_block(opp_cre, st.opp_max_hp),
        ""
    ]
    for _ in range(10):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        simulate_round(st)

    if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
        winner = user_cre["name"] if st.opp_current_hp <= 0 else opp_cre["name"]
        st.logs.append(f"Winner: {winner}")
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
    for _ in range(10):
        if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
            break
        simulate_round(st)

    new = st.logs[st.next_log_idx:]
    st.next_log_idx = len(st.logs)
    if st.user_current_hp <= 0 or st.opp_current_hp <= 0:
        winner = st.user_creature["name"] if st.opp_current_hp <= 0 else st.opp_creature["name"]
        new.append(f"Winner: {winner}")
        active_battles.pop(inter.user.id, None)
    else:
        new.append("Use /continue to proceed.")
    await send_chunks(inter, "\n".join(new))

@bot.tree.command(description="Check your cash")
async def cash(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        await inter.response.send_message(f"You have {row['cash']} cash.", ephemeral=True)

@bot.tree.command(description="Add cash (dev utility)")
async def cashadd(inter: discord.Interaction, amount: int):
    if amount <= 0:
        return await inter.response.send_message("Positive amounts only.", ephemeral=True)
    row = await ensure_registered(inter)
    if not row:
        return
    await (await db_pool()).execute(
        "UPDATE trainers SET cash = cash + $1 WHERE user_id=$2", amount, inter.user.id
    )
    await inter.response.send_message(f"Added {amount} cash.", ephemeral=True)

@bot.tree.command(description="Check your trainer points")
async def trainerpoints(inter: discord.Interaction):
    row = await ensure_registered(inter)
    if row:
        await inter.response.send_message(f"You have {row['trainer_points']} points.", ephemeral=True)

@bot.tree.command(description="Train a creature stat")
async def train(
    inter: discord.Interaction, creature_name: str, stat: str, increase: int
):
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
        "SELECT id,stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        inter.user.id, creature_name
    )
    if not c:
        return await inter.response.send_message("Creature not found.", ephemeral=True)

    st = json.loads(c["stats"])
    st[stat] += increase
    await (await db_pool()).execute(
        "UPDATE creatures SET stats=$1 WHERE id=$2", json.dumps(st), c["id"]
    )
    await (await db_pool()).execute(
        "UPDATE trainers SET trainer_points = trainer_points - $1 WHERE user_id=$2",
        increase, inter.user.id
    )
    disp = increase*5 if stat=="HP" else increase
    await inter.response.send_message(
        f"{c['id']} – {creature_name.title()} trained: +{disp} {stat}.",
        ephemeral=True
    )

# ─── Launch ──────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(TOKEN)
