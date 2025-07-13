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
from discord.ext import commands
import openai  # pre-1.0 SDK
# ─── Configuration & Logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("DISCORD_TOKEN")
DB_URL = os.getenv("DATABASE_URL")
GUILD_ID = os.getenv("GUILD_ID")  # empty string => global
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
POINT_POOLS: Dict[str, Tuple[int,int]] = {
    "Common":    (25, 50),
    "Uncommon":  (50, 100),
    "Rare":      (100, 200),
    "Epic":      (200, 400),
    "Legendary": (400, 800),
}
TIER_EXTRAS: Dict[int, Tuple[int,int]] = {
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

# ─── In-Memory Battle Store ──────────────────────────────────
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
    # descriptors bias removed for simplicity, can re-add if desired
    for _ in range(pool):
        stat = random.choice(PRIMARY_STATS)
        stats[stat] += 1
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
    choice = resp.choices[0]
    return choice.message.content.strip()

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
                return data
        except json.JSONDecodeError:
            continue
    # fallback name
    return {"name": f"Creature{random.randint(1000,9999)}", "descriptors": []}

# ─── Battle Simulation ─────────────────────────────────────
def simulate_round(state: BattleState):
    u = state.user_creature
    o = state.opp_creature
    u_spd = u["stats"]["SPD"]
    o_spd = o["stats"]["SPD"]
    if u_spd > o_spd or (u_spd == o_spd and random.choice([True, False])):
        order = [("user", u, o)]
    else:
        order = [("opp", o, u)]
    for actor, attacker, defender in order:
        attacks = 2 if attacker["stats"]["SPD"] >= 2 * defender["stats"]["SPD"] else 1
        for _ in range(attacks):
            if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
                return
            S = max(attacker["stats"]["PATK"], attacker["stats"]["SATK"])
            N = math.ceil(S / 10)
            roll_sum = sum(random.randint(1, 6) for _ in range(N))
            defense = defender["stats"].get("AR", 0)
            damage = max(1, roll_sum - defense)
            if actor == "user":
                state.opp_current_hp -= damage
            else:
                state.user_current_hp -= damage
            state.logs.append(f"{attacker['name']} attacked and dealt {damage} damage.")
    state.rounds += 1

# ─── Bot Lifecycle Events ────────────────────────────────────
@bot.event
async def setup_hook():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild)
        await bot.tree.sync(guild=guild)
        logger.info("Synced to guild %s", GUILD_ID)
    else:
        await bot.tree.sync()
        logger.info("Synced globally")

@bot.event
async def on_ready():
    logger.info("Logged in as %s", bot.user)

# ─── Slash Commands ─────────────────────────────────────────
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    async with (await db_pool()).acquire() as conn:
        await conn.execute(
            "INSERT INTO trainers(user_id) VALUES($1) ON CONFLICT DO NOTHING", interaction.user.id
        )
    await interaction.response.send_message("Trainer profile created!", ephemeral=True)

@bot.tree.command(description="Spawn a brand-new creature egg")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        if not await conn.fetchrow("SELECT 1 FROM trainers WHERE user_id=$1", uid):
            return await interaction.response.send_message("Use /register first.", ephemeral=True)
    await interaction.response.defer(thinking=True)
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    meta = await generate_creature_meta(rarity)
    name = meta.get("name", f"Creature{roll}{random.randint(1000,9999)}")
    descriptors = meta.get("descriptors", [])
    stats = allocate_stats(rarity, descriptors)
    async with (await db_pool()).acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats)"
            " VALUES($1,$2,$3,$4,$5)",
            uid, name, rarity, descriptors, json.dumps(stats)
        )
    embed = discord.Embed(title=f"{name} ({rarity})", color=0x8B0000)
    hp_display = stats["HP"] * 5
    embed.add_field(name="HP", value=str(hp_display), inline=True)
    embed.add_field(name="AR", value=str(stats["AR"]), inline=True)
    embed.add_field(name="PATK", value=str(stats["PATK"]), inline=True)
    embed.add_field(name="SATK", value=str(stats["SATK"]), inline=True)
    embed.add_field(name="SPD", value=str(stats["SPD"]), inline=True)
    embed.add_field(name="Descriptors", value=", ".join(descriptors) or "None", inline=False)
    embed.set_footer(text=f"d100 roll: {roll}")
    await interaction.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    rows = await (await db_pool()).fetch(
        "SELECT name,rarity,stats,descriptors FROM creatures WHERE owner_id=$1 ORDER BY id", interaction.user.id
    )
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)
    lines: List[str] = []
    for i, r in enumerate(rows, 1):
        stats = r['stats'] if isinstance(r['stats'], dict) else json.loads(r['stats'])
        hp = stats['HP'] * 5
        stat_str = f"HP:{hp}, AR:{stats['AR']}, PATK:{stats['PATK']}, SATK:{stats['SATK']}, SPD:{stats['SPD']}"
        desc_str = ", ".join(r['descriptors']) if r.get('descriptors') else 'None'
        lines.append(f"{i}. **{r['name']}** ({r['rarity']}) – {stat_str} – [{desc_str}]")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

@bot.tree.command(description="Battle your creature against a tiered opponent")
async def battle(interaction: discord.Interaction, creature_name: str, tier: int):
    uid = interaction.user.id
    if tier not in TIER_EXTRAS:
        return await interaction.response.send_message("Invalid tier.", ephemeral=True)
    row = await (await db_pool()).fetchrow(
        "SELECT name,rarity,stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2", uid, creature_name
    )
    if not row:
        return await interaction.response.send_message("You don't own a creature with that name.", ephemeral=True)
    user_stats = row['stats'] if isinstance(row['stats'], dict) else json.loads(row['stats'])
    user_creature = {"name": row['name'], "stats": user_stats}
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    extra = random.randint(*TIER_EXTRAS[tier])
    opp_stats = allocate_stats(rarity, [], extra)
    opp_creature = {"name": f"Wild{roll}{random.randint(1000,9999)}", "stats": opp_stats}
    state = BattleState(
        user_id=uid,
        user_creature=user_creature,
        user_current_hp=user_stats['HP'] * 5,
        user_max_hp=user_stats['HP'] * 5,
        opp_creature=opp_creature,
        opp_current_hp=opp_stats['HP'] * 5,
        opp_max_hp=opp_stats['HP'] * 5,
        logs=[]
    )
    active_battles[uid] = state
    for _ in range(10):
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        simulate_round(state)
    out = state.logs.copy()
    if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
        winner = 'you' if state.opp_current_hp <= 0 else 'opponent'
        out.append(f"Battle over! {winner.capitalize()} won.")
        active_battles.pop(uid, None)
    else:
        out.append("Type /continue to proceed to the next 10 rounds.")
    await interaction.response.send_message("\n".join(out))

@bot.tree.command(name="continue", description="Continue your ongoing battle")
async def continue_battle(interaction: discord.Interaction):
    uid = interaction.user.id
    state = active_battles.get(uid)
    if not state:
        return await interaction.response.send_message("You have no ongoing battle.", ephemeral=True)
    for _ in range(10):
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        simulate_round(state)
    out = state.logs[state.next_log_idx:]
    state.next_log_idx = len(state.logs)
    if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
        winner = 'you' if state.opp_current_hp <= 0 else 'opponent'
        out.append(f"Battle over! {winner.capitalize()} won.")
        active_battles.pop(uid, None)
    else:
        out.append("Type /continue to proceed to the next 10 rounds.")
    await interaction.response.send_message("\n".join(out))

if __name__ == "__main__":
    bot.run(TOKEN)
