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
POINT_POOLS: Dict[str, Tuple[int, int]] = {
    "Common":    (25, 50),
    "Uncommon":  (50, 100),
    "Rare":      (100, 200),
    "Epic":      (200, 400),
    "Legendary": (400, 800),
}
TIER_EXTRAS: Dict[int, Tuple[int, int]] = {
    1: (0, 10), 2: (10, 30), 3: (30, 60), 4: (60, 100),
    5: (100, 140), 6: (140, 180), 7: (180, 220),
    8: (220, 260), 9: (200, 300),
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
    prompt = f"""
You are inventing a creature name and descriptors for a {rarity} creature.
Reply ONLY with JSON:
{{"name": "string","descriptors": ["w1","w2","w3"]}}
Avoid names: {', '.join(names_used)}
Avoid descriptors: {', '.join(words_used)}
"""
    for _ in range(3):
        try:
            data = json.loads(await ask_openai(prompt))
            if data.get("name") and len(data.get("descriptors", [])) == 3:
                return data
        except:
            pass
    return {"name": f"Wild{random.randint(1000,9999)}", "descriptors": []}

# ─── Battle Simulation ─────────────────────────────────────
def simulate_round(state: BattleState):
    u, o = state.user_creature, state.opp_creature
    ordered = sorted((("user", u, o), ("opp", o, u)),
                     key=lambda x: (x[1]["stats"]["SPD"], random.random()),
                     reverse=True)
    for actor, attacker, defender in ordered:
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        attacks = 2 if attacker["stats"]["SPD"] >= 2 * defender["stats"]["SPD"] else 1
        for _ in range(attacks):
            if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
                break
            S = max(attacker["stats"]["PATK"], attacker["stats"]["SATK"])
            N = math.ceil(S / 10)
            rolls = [random.randint(1, 6) for _ in range(N)]
            total = sum(rolls)
            AR = defender["stats"].get("AR", 0)
            dmg = max(1, math.ceil((total**2)/(total + AR)))
            if actor == "user": state.opp_current_hp -= dmg
            else: state.user_current_hp -= dmg
            state.logs.append(f"{attacker['name']} rolled {rolls} (sum {total}), AR {AR} -> {dmg}")
    state.rounds += 1
    state.logs.append(
        f"After round {state.rounds}: {state.user_creature['name']} HP {state.user_current_hp}/{state.user_max_hp}, "+
        f"{state.opp_creature['name']} HP {state.opp_current_hp}/{state.opp_max_hp}"
    )

# ─── Chunked Messaging ───────────────────────────────────────
async def send_chunks(interaction: discord.Interaction, content: str):
    parts = [content[i:i+1900] for i in range(0, len(content), 1900)]
    await interaction.response.send_message(parts[0])
    for p in parts[1:]: await interaction.followup.send(p)

# ─── Commands ──────────────────────────────────────────────
@bot.tree.command(description="Register as trainer")
async def register(interaction: discord.Interaction):
    async with (await db_pool()).acquire() as conn:
        await conn.execute("INSERT INTO trainers(user_id) VALUES($1) ON CONFLICT DO NOTHING", interaction.user.id)
    await interaction.response.send_message("Registered!", ephemeral=True)

@bot.tree.command(description="Spawn a creature egg")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    async with (await db_pool()).acquire() as conn:
        if not await conn.fetchrow("SELECT 1 FROM trainers WHERE user_id=$1", uid):
            return await interaction.response.send_message("Use /register first.", ephemeral=True)
    await interaction.response.defer()
    roll, rar = roll_d100(), rarity_from_roll(roll)
    meta = await generate_creature_meta(rar)
    stats = allocate_stats(rar, meta.get("descriptors", []))
    async with (await db_pool()).acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats)"
            " VALUES($1,$2,$3,$4,$5)", uid, meta['name'], rar, meta['descriptors'], json.dumps(stats)
        )
    emb = discord.Embed(title=f"{meta['name']} ({rar})", color=0x00ff00)
    for k in ['HP','AR','PATK','SATK','SPD']:
        emb.add_field(name=k, value=str(stats[k] if k!='HP' else stats[k]*5), inline=True)
    emb.set_footer(text=f"Roll: {roll}")
    await interaction.followup.send(embed=emb)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    rows = await (await db_pool()).fetch("SELECT name,rarity,stats,descriptors FROM creatures WHERE owner_id=$1", interaction.user.id)
    if not rows: return await interaction.response.send_message("No creatures.", ephemeral=True)
    txt = '\n'.join(
        f"{i+1}. {r['name']} ({r['rarity']}) - " +
        ", ".join(r['descriptors']) + ": " +
        ", ".join(f"{s}:{(v*5 if s=='HP' else v)}" for s,v in (json.loads(r['stats']) if isinstance(r['stats'],str) else r['stats']).items())
        for i,r in enumerate(rows)
    )
    await interaction.response.send_message(txt)

@bot.tree.command(description="Battle your creature")
async def battle(interaction: discord.Interaction, creature_name: str, tier: int):
    uid=interaction.user.id
    if tier not in TIER_EXTRAS: return await interaction.response.send_message("Invalid tier.", ephemeral=True)
    row = await (await db_pool()).fetchrow(
        "SELECT name,stats FROM creatures WHERE owner_id=$1 AND name ILIKE $2", uid, creature_name)
    if not row: return await interaction.response.send_message("No such creature.", ephemeral=True)
    stats = json.loads(row['stats']) if isinstance(row['stats'],str) else row['stats']
    user = {'name':row['name'],'stats':stats}
    roll, rar = roll_d100(), rarity_from_roll(roll)
    opp_meta = await(generate_creature_meta(rar))
    opp_stats = allocate_stats(rar, opp_meta.get('descriptors',[]), random.randint(*TIER_EXTRAS[tier]))
    opp = {'name':opp_meta['name'],'stats':opp_stats}
    state = BattleState(uid,user,stats['HP']*5,stats['HP']*5,opp,opp_stats['HP']*5,opp_stats['HP']*5,[])
    active_battles[uid]=state
    for _ in range(10): simulate_round(state)
    if state.user_current_hp<=0 or state.opp_current_hp<=0:
        winner='you' if state.opp_current_hp<=0 else 'opponent'
        state.logs.append(f"Battle over! {winner} won.")
        active_battles.pop(uid,None)
    else: state.logs.append("Type /continue to continue.")
    await send_chunks(interaction, '\n'.join(state.logs))

@bot.tree.command(name="continue",description="Continue battle")
async def continue_battle(interaction: discord.Interaction):
    uid=interaction.user.id
    state=active_battles.get(uid)
    if not state: return await interaction.response.send_message("No battle.",ephemeral=True)
    for _ in range(10): simulate_round(state)
    logs=state.logs[state.next_log_idx:]
    state.next_log_idx=len(state.logs)
    if state.user_current_hp<=0 or state.opp_current_hp<=0:
        w='you' if state.opp_current_hp<=0 else 'opponent'
        logs.append(f"Battle over! {w} won.")
        active_battles.pop(uid,None)
    else: logs.append("Type /continue to continue.")
    await send_chunks(interaction,'\n'.join(logs))

if __name__ == "__main__":
    bot.run(TOKEN)
