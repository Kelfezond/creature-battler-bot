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
    9: (200, 300),
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

def allocate_stats(rarity: str, descriptors: List[str], extra: int = 0) -> Dict[str, int]:
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

async def ask_openai(prompt: str, max_tokens: int = 100) -> Optional[str]:
    resp = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=max_tokens,
        )
    )
    return resp.choices[0].message.content.strip()

async def generate_creature_meta(rarity: str) -> Dict[str, Any]:
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
    for _ in range(3):
        text = await ask_openai(prompt)
        try:
            data = json.loads(text)
            if "name" in data and len(data.get("descriptors", [])) == 3:
                data["name"] = data["name"].title()
                return data
        except json.JSONDecodeError:
            continue
    return {"name": f"Wild{random.randint(1000,9999)}", "descriptors": []}

# ─── Battle Simulation ─────────────────────────────────────
def simulate_round(state: BattleState):
    state.rounds += 1
    state.logs.append(f"Round {state.rounds}")
    u, o = state.user_creature, state.opp_creature
    u_spd, o_spd = u["stats"]["SPD"], o["stats"]["SPD"]
    order = [("user", u, o), ("opp", o, u)] if (u_spd > o_spd or (u_spd == o_spd and random.choice([True,False]))) else [("opp", o, u), ("user", u, o)]
    for actor, attacker, defender in order:
        attacks = 2 if attacker["stats"]["SPD"] >= 2 * defender["stats"]["SPD"] else 1
        for _ in range(attacks):
            if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
                return
            S = max(attacker["stats"].get("PATK",0), attacker["stats"].get("SATK",0))
            rolls = [random.randint(1,6) for _ in range(math.ceil(S/10))]
            damage = max(1, math.ceil(sum(rolls)**2 / (sum(rolls) + defender["stats"].get("AR",0))))
            if actor == "user": state.opp_current_hp -= damage
            else: state.user_current_hp -= damage
            state.logs.append(f"{attacker['name']} attacked for {damage} (rolls {rolls})")
            if state.opp_current_hp <= 0 or state.user_current_hp <= 0:
                state.logs.append(f"{defender['name']} is down!")
                return
    state.logs.append("")

# ─── Bot Lifecycle Events ───────────────────────────────────
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
    if not distribute_cash.is_running(): distribute_cash.start()
    if not distribute_points.is_running(): distribute_points.start()

@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)

async def send_chunks(interaction: discord.Interaction, content: str, use_followup_first: bool=False):
    chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
    if not use_followup_first:
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
    async with (await db_pool()).acquire() as conn:
        row = await conn.fetchrow("SELECT cash FROM trainers WHERE user_id=$1", uid)
        if not row:
            return await interaction.response.send_message("Use /register first.", ephemeral=True)
        if row['cash'] < 10000:
            return await interaction.response.send_message("Not enough cash", ephemeral=True)
        await conn.execute("UPDATE trainers SET cash = cash - 10000 WHERE user_id=$1", uid)
    await interaction.response.defer(thinking=True)
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    meta = await generate_creature_meta(rarity)
    stats = allocate_stats(rarity, meta.get("descriptors", []))
    async with (await db_pool()).acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats) VALUES($1,$2,$3,$4,$5)",
            uid, meta["name"], rarity, meta["descriptors"], json.dumps(stats)
        )
    embed = discord.Embed(title=f"{meta['name']} ({rarity})")
    # Add stat fields
    for stat, value in stats.items():
        display = value * 5 if stat == "HP" else value
        embed.add_field(name=stat, value=str(display), inline=True)
    # Include descriptors
    descriptors_list = meta.get('descriptors', [])
    if descriptors_list:
        embed.add_field(name="Descriptors", value=", ".join(descriptors_list), inline=False)
    embed.set_footer(text=f"d100 roll: {roll}")
    await interaction.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    rows = await (await db_pool()).fetch(
        "SELECT id,name,rarity,stats,descriptors FROM creatures WHERE owner_id=$1 ORDER BY id",
        interaction.user.id
    )
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)
    lines = []
    for idx, r in enumerate(rows, 1):
        stats = json.loads(r['stats'])
        descs = r['descriptors'] or []
        desc_str = f" - Descriptors: {', '.join(descs)}" if descs else ""
        lines.append(
            f"{idx}. **{r['name']}** ({r['rarity']}) - "
            f"HP:{stats['HP']*5}, AR:{stats['AR']}, PATK:{stats['PATK']}, SATK:{stats['SATK']}, SPD:{stats['SPD']}"
            f"{desc_str}"
        )
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

# ... rest of commands unchanged ...

if __name__ == "__main__":
    bot.run(TOKEN)
