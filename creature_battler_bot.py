from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import asyncpg
import discord
from discord.ext import commands, tasks
import openai  # pre-1.0 SDK

# ─── Configuration & Logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("DISCORD_TOKEN")
DB_URL = os.getenv("DATABASE_URL")
GUILD_ID = os.getenv("GUILD_ID") or None
openai.api_key = os.getenv("OPENAI_API_KEY")

for name, val in {
    "DISCORD_TOKEN": TOKEN,
    "DATABASE_URL": DB_URL,
    "OPENAI_API_KEY": openai.api_key,
}.items():
    if not val:
        raise RuntimeError(f"Missing environment variable: {name}")

# ─── Discord Client Setup ─────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ─── Database Schema & Helpers ────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trainers (
  user_id BIGINT PRIMARY KEY,
  joined_at TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE trainers ADD COLUMN IF NOT EXISTS cash BIGINT DEFAULT 0;
ALTER TABLE trainers ADD COLUMN IF NOT EXISTS trainer_points BIGINT DEFAULT 0;

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

# ─── Game Constants ──────────────────────────────────────────
RARITY_TABLE: List[Tuple[int, int, str]] = [
    (1, 75, "Common"),
    (76, 88, "Uncommon"),
    (89, 95, "Rare"),
    (96, 98, "Epic"),
    (99, 100, "Legendary"),
]
POINT_POOLS: Dict[str, Tuple[int, int]] = {
    "Common":    (25, 50),
    "Uncommon":  (50, 100),
    "Rare":      (100, 200),
    "Epic":      (200, 400),
    "Legendary": (400, 800),
}
TIER_EXTRAS: Dict[int, Tuple[int, int]] = {
    1: (0, 10),
    2: (10, 30),
    3: (30, 60),
    4: (60, 100),
    5: (100, 140),
    6: (140, 180),
    7: (180, 220),
    8: (220, 260),
    9: (260, 300),   # adjusted to avoid overlap with tier 8
}
PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]

# ─── In-Memory Battle Store ─────────────────────────────────
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

# ─── Scheduled Tasks ─────────────────────────────────────────
@tasks.loop(hours=1)
async def distribute_cash():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE trainers SET cash = cash + 400")
    logger.info("Distributed 400 cash to all trainers")

@tasks.loop(hours=24)
async def distribute_points():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE trainers SET trainer_points = trainer_points + 5")
    logger.info("Distributed 5 trainer points to all trainers")

# ─── Utility Functions ───────────────────────────────────────
def roll_d100() -> int:
    return random.randint(1, 100)

def rarity_from_roll(r: int) -> str:
    for low, high, name in RARITY_TABLE:
        if low <= r <= high:
            return name
    return "Common"

def allocate_stats(rarity: str, extra: int = 0) -> Dict[str, int]:
    pool = random.randint(*POINT_POOLS[rarity]) + extra
    stats = {s: 1 for s in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)
    for _ in range(pool):
        stats[random.choice(PRIMARY_STATS)] += 1
    return stats

async def fetch_used_lists() -> Tuple[List[str], List[str]]:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    names = [r["name"].lower() for r in rows]
    words = {w.lower() for r in rows for w in r["descriptors"]}
    return names, list(words)

async def ensure_registered(interaction: discord.Interaction) -> Optional[asyncpg.Record]:
    """Ensure the user has a trainer profile. Returns the trainer row or None (after sending an error)."""
    pool = await db_pool()
    row = await pool.fetchrow("SELECT cash, trainer_points FROM trainers WHERE user_id=$1", interaction.user.id)
    if not row:
        await interaction.response.send_message("Use /register first.", ephemeral=True)
        return None
    return row

async def generate_creature_meta(rarity: str) -> Dict[str, Any]:
    # Avoid reusing any existing names or descriptor words:
    names_used, words_used = await fetch_used_lists()
    avoid_names = ", ".join(names_used)
    avoid_words = ", ".join(words_used)
    prompt = f"""
You are inventing a creature name and descriptors for a {rarity} creature.
Reply ONLY with JSON:
{{
  "name": "string (1-3 words)",
  "descriptors": ["word1","word2","word3"]
}}
• Exactly 3 descriptors.
• Avoid names: {avoid_names}
• Avoid descriptors: {avoid_words}
"""
    # Try up to 3 times to get a valid JSON response
    for _ in range(3):
        try:
            resp = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=100,
                )
            )
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            continue
        text = resp.choices[0].message.content.strip()
        try:
            data = json.loads(text)
            if "name" in data and len(data.get("descriptors", [])) == 3:
                data["name"] = data["name"].title()
                return data
        except json.JSONDecodeError:
            continue
    # Fallback: generate a placeholder name if OpenAI fails or returns invalid JSON
    return {"name": f"Wild{random.randint(1000, 9999)}", "descriptors": []}

def simulate_round(state: BattleState):
    state.rounds += 1
    state.logs.append(f"Round {state.rounds}")
    u, o = state.user_creature, state.opp_creature
    u_spd, o_spd = u["stats"]["SPD"], o["stats"]["SPD"]
    order = [("user", u, o), ("opp", o, u)] if (u_spd > o_spd or (u_spd == o_spd and random.choice([True, False]))) else [("opp", o, u), ("user", u, o)]
    for actor, attacker, defender in order:
        attacks = 2 if attacker["stats"]["SPD"] >= 2 * defender["stats"]["SPD"] else 1
        for _ in range(attacks):
            if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
                return
            S = max(attacker["stats"].get("PATK", 0), attacker["stats"].get("SATK", 0))
            rolls = [random.randint(1, 6) for _ in range(math.ceil(S / 10))]
            damage = max(1, math.ceil(sum(rolls) ** 2 / (sum(rolls) + defender["stats"].get("AR", 0))))
            if actor == "user":
                state.opp_current_hp -= damage
            else:
                state.user_current_hp -= damage
            state.logs.append(f"{attacker['name']} attacked for {damage} (rolls {rolls})")
            if state.opp_current_hp <= 0 or state.user_current_hp <= 0:
                state.logs.append(f"{defender['name']} is down!")
                return
    state.logs.append("")

# ─── Bot Lifecycle Events ───────────────────────────────────
@bot.event
async def setup_hook():
    # Initialize database and sync commands
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

async def send_chunks(interaction: discord.Interaction, content: str, use_followup_first: bool=False):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    if use_followup_first:
        await interaction.followup.send(chunks[0])
    else:
        await interaction.response.send_message(chunks[0])
    for c in chunks[1:]:
        await interaction.followup.send(c)

# ─── Slash Commands ─────────────────────────────────────────
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        existing = await conn.fetchrow("SELECT user_id FROM trainers WHERE user_id=$1", uid)
        if existing:
            return await interaction.response.send_message("You are already registered!", ephemeral=True)
        await conn.execute(
            "INSERT INTO trainers(user_id, cash, trainer_points) VALUES($1, $2, $3)",
            uid, 20000, 5
        )
    await interaction.response.send_message(
        "Trainer profile created! You receive 20,000 cash and 5 trainer points.",
        ephemeral=True
    )

@bot.tree.command(description="Spawn a brand-new creature egg (costs 10000 cash)")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    # Ensure user exists and has enough cash
    row = await ensure_registered(interaction)
    if not row:
        return
    if row['cash'] < 10000:
        return await interaction.response.send_message("Not enough cash", ephemeral=True)
    await (await db_pool()).execute("UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", uid)
    await interaction.response.defer(thinking=True)
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    meta = await generate_creature_meta(rarity)
    stats = allocate_stats(rarity)
    await (await db_pool()).execute(
        "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats) VALUES($1,$2,$3,$4,$5)",
        uid, meta["name"], rarity, meta["descriptors"], json.dumps(stats)
    )
    embed = discord.Embed(
        title=f"{meta['name']} ({rarity})",
        description=f"Descriptors: {', '.join(meta.get('descriptors', []))}"
    )
    for stat, value in stats.items():
        display = value * 5 if stat == "HP" else value
        embed.add_field(name=stat, value=str(display), inline=True)
    embed.set_footer(text=f"d100 roll: {roll}")
    await interaction.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    row = await ensure_registered(interaction)
    if not row:
        return
    rows = await (await db_pool()).fetch(
        "SELECT id,name,rarity,stats,descriptors FROM creatures WHERE owner_id=$1 ORDER BY id",
        interaction.user.id
    )
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)
    lines = []
    for idx, r in enumerate(rows, 1):
        stats = json.loads(r['stats'])
        descriptors = ", ".join(r['descriptors']) if r['descriptors'] else "None"
        lines.append(
            f"{idx}. **{r['name']}** ({r['rarity']}) - Descriptors: {descriptors} - "
            f"HP:{stats['HP']*5}, AR:{stats['AR']}, PATK:{stats['PATK']}, SATK:{stats['SATK']}, SPD:{stats['SPD']}"
        )
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

@bot.tree.command(description="Battle your creature against a tiered opponent")
async def battle(interaction: discord.Interaction, creature_name: str, tier: int):
    uid = interaction.user.id
    # Prevent starting a new battle if one is active
    if uid in active_battles:
        return await interaction.response.send_message("Finish your ongoing battle with /continue first.", ephemeral=True)
    # Validate tier and trainer profile
    if not tier in TIER_EXTRAS:
        return await interaction.response.send_message("Invalid tier.", ephemeral=True)
    row_tr = await ensure_registered(interaction)
    if not row_tr:
        return
    # Fetch the chosen creature
    creature_row = await (await db_pool()).fetchrow(
        "SELECT name, stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        uid, creature_name
    )
    if not creature_row:
        return await interaction.response.send_message("You don't own that creature.", ephemeral=True)
    await interaction.response.defer(thinking=True)
    user_stats = json.loads(creature_row['stats'])
    user_creature = {"name": creature_row['name'], "stats": user_stats}
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    meta = await generate_creature_meta(rarity)
    extra = random.randint(*TIER_EXTRAS[tier])
    opp_stats = allocate_stats(rarity, extra)
    opp_creature = {"name": meta['name'], "stats": opp_stats}
    state = BattleState(
        uid, user_creature, user_stats['HP'] * 5, user_stats['HP'] * 5,
        opp_creature, opp_stats['HP'] * 5, opp_stats['HP'] * 5, []
    )
    active_battles[uid] = state
    state.logs.extend([
        "Battle Start",
        f"Tier {tier} (+{extra} points)",
        f"{user_creature['name']} vs {opp_creature['name']}",
        f"Roll {roll} -> {rarity}",
        ""
    ])
    # Simulate up to 10 rounds initially
    for _ in range(10):
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        simulate_round(state)
    if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
        winner = user_creature['name'] if state.opp_current_hp <= 0 else opp_creature['name']
        state.logs.append(f"Winner: {winner}")
        active_battles.pop(uid, None)
    else:
        state.logs.append("Type /continue to proceed.")
    await send_chunks(interaction, "\n".join(state.logs))
    # Update log index so /continue sends only new logs
    state.next_log_idx = len(state.logs)

@bot.tree.command(name="continue", description="Continue your ongoing battle")
async def continue_battle(interaction: discord.Interaction):
    uid = interaction.user.id
    state = active_battles.get(uid)
    if not state:
        return await interaction.response.send_message("No ongoing battle.", ephemeral=True)
    await interaction.response.defer()
    # Play up to 10 more rounds
    for _ in range(10):
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        simulate_round(state)
    new_logs = state.logs[state.next_log_idx:]
    state.next_log_idx = len(state.logs)
    if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
        winner = state.user_creature['name'] if state.opp_current_hp <= 0 else state.opp_creature['name']
        new_logs.append(f"Winner: {winner}")
        active_battles.pop(uid, None)
    else:
        new_logs.append("Type /continue to proceed.")
    await send_chunks(interaction, "\n".join(new_logs), use_followup_first=True)

@bot.tree.command(description="Check your cash balance")
async def cash(interaction: discord.Interaction):
    row = await ensure_registered(interaction)
    if not row:
        return
    await interaction.response.send_message(f"You have {row['cash']} cash.", ephemeral=True)

@bot.tree.command(description="Add cash to your balance")
async def cashadd(interaction: discord.Interaction, amount: int):
    if amount <= 0:
        return await interaction.response.send_message("Amount must be positive.", ephemeral=True)
    row = await ensure_registered(interaction)
    if not row:
        return
    await (await db_pool()).execute("UPDATE trainers SET cash = cash + $1 WHERE user_id = $2", amount, interaction.user.id)
    await interaction.response.send_message(f"Added {amount} cash.", ephemeral=True)

@bot.tree.command(description="Check your trainer points")
async def trainerpoints(interaction: discord.Interaction):
    row = await ensure_registered(interaction)
    if not row:
        return
    await interaction.response.send_message(f"You have {row['trainer_points']} trainer points.", ephemeral=True)

@bot.tree.command(description="Train a creature stat using trainer points")
async def train(interaction: discord.Interaction, creature_name: str, stat: str, increase: int):
    stat = stat.upper()
    if stat not in PRIMARY_STATS:
        return await interaction.response.send_message(f"Invalid stat; choose one of {', '.join(PRIMARY_STATS)}.", ephemeral=True)
    if increase <= 0:
        return await interaction.response.send_message("Increase must be positive.", ephemeral=True)
    uid = interaction.user.id
    # Ensure user exists and has enough points
    row = await ensure_registered(interaction)
    if not row:
        return
    if row['trainer_points'] < increase:
        return await interaction.response.send_message("Not enough trainer points.", ephemeral=True)
    # Fetch the creature to train
    creature = await (await db_pool()).fetchrow(
        "SELECT id, stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
        uid, creature_name
    )
    if not creature:
        return await interaction.response.send_message("Creature not found.", ephemeral=True)
    stats = json.loads(creature['stats'])
    if stat == 'HP':
        stats['HP'] += increase
    else:
        stats[stat] += increase
    await (await db_pool()).execute(
        "UPDATE creatures SET stats = $1 WHERE id = $2",
        json.dumps(stats), creature['id']
    )
    await (await db_pool()).execute(
        "UPDATE trainers SET trainer_points = trainer_points - $1 WHERE user_id = $2",
        increase, uid
    )
    display_inc = increase * 5 if stat == 'HP' else increase
    await interaction.response.send_message(
        f"Trained {creature_name.title()}: +{display_inc} {stat}. You have {row['trainer_points'] - increase} points left.",
        ephemeral=True
    )

if __name__ == "__main__":
    bot.run(TOKEN)
