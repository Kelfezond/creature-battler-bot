"""
Creature Battler Discord Bot – scaffolding **v0.3b**
----------------------------------------------------
Rolls back to the pre‑1.0 OpenAI SDK (openai==0.28.*) to avoid the
`Client.__init__() got an unexpected keyword argument 'proxies'` crash that
Railway is showing.  The older SDK still supports `openai.ChatCompletion.create`.

**Change summary**
• requirements.txt should pin `openai==0.28.1` (or latest 0.28.x).
• Code reverts to synchronous `openai.ChatCompletion.create()` inside an
  `run_in_executor` wrapper so the Discord bot loop stays async.

All other behaviour is identical to v0.2.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import textwrap
from functools import partial
from typing import Dict, List

import asyncpg
import discord
import openai  # openai==0.28.x
from discord.ext import commands

# ────────────────────────── Environment ──────────────────────────
TOKEN = os.getenv("DISCORD_TOKEN")
CLIENT_ID = os.getenv("CLIENT_ID")
GUILD_ID = os.getenv("GUILD_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

for var, val in {"DISCORD_TOKEN": TOKEN, "CLIENT_ID": CLIENT_ID, "OPENAI_API_KEY": OPENAI_API_KEY, "DATABASE_URL": DB_URL}.items():
    if not val:
        raise RuntimeError(f"Missing env var {var}")

openai.api_key = OPENAI_API_KEY

# ────────────────────────── Discord client ───────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ────────────────────────── Database helpers ─────────────────────
async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trainers (
    user_id   BIGINT PRIMARY KEY,
    joined_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS creatures (
    id          SERIAL PRIMARY KEY,
    owner_id    BIGINT NOT NULL,
    name        TEXT,
    rarity      TEXT,
    descriptors TEXT[],
    stats       JSONB,
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""

# ────────────────────────── Game constants (unchanged) ───────────
RARITY_TABLE = [
    (1, 75, "Common"),
    (76, 88, "Uncommon"),
    (89, 95, "Rare"),
    (96, 98, "Epic"),
    (99, 100, "Legendary"),
]

POINT_POOLS = {
    "Common": (25, 50),
    "Uncommon": (50, 100),
    "Rare": (100, 200),
    "Epic": (200, 400),
    "Legendary": (400, 800),
}

PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]

BIAS_MAP = {
    "Rocky": {"AR": +0.2, "SPD": -0.2},
    "Lightning-fast": {"SPD": +0.2, "AR": -0.2},
    "Giant": {"HP": +0.2, "SPD": -0.2},
}

# ────────────────────────── Utility functions ────────────────────

def roll_d100() -> int:
    return random.randint(1, 100)


def rarity_from_roll(roll: int) -> str:
    for low, high, rarity in RARITY_TABLE:
        if low <= roll <= high:
            return rarity
    return "Common"


def allocate_stats(rarity: str, descriptors: List[str]) -> Dict[str, int]:
    pts = random.randint(*POINT_POOLS[rarity])
    stats = {s: 1 for s in PRIMARY_STATS}
    pts -= len(PRIMARY_STATS)

    weights = {s: 1.0 for s in PRIMARY_STATS}
    for d in descriptors:
        for stat, delta in BIAS_MAP.get(d, {}).items():
            weights[stat] = max(0.1, weights[stat] + delta)

    total = sum(weights.values())
    for _ in range(pts):
        r = random.uniform(0, total)
        cum = 0
        for stat, w in weights.items():
            cum += w
            if r <= cum:
                stats[stat] += 1
                break
    return stats


async def ask_openai(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    # run blocking call in default executor
    func = partial(openai.ChatCompletion.create, model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}], temperature=0.9, max_tokens=50)
    resp = await loop.run_in_executor(None, func)
    return resp.choices[0].message.content.strip()


async def generate_creature_prompt(rarity: str):
    prompt = textwrap.dedent(f"""
        You are inventing a dark‑fantasy arena monster for a gritty creature‑battler game.
        Output EXACTLY two lines:
        1. A short unique name (1‑3 words).
        2. Exactly THREE descriptor keywords, comma‑separated.
        Creature rarity: {rarity}.
    """)
    text = await ask_openai(prompt)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("Malformed OpenAI response")
    return lines[0], [d.strip() for d in lines[1].split(",")][:3]

# ────────────────────────── Bot lifecycle ────────────────────────
@bot.event
async def setup_hook():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild)
        await bot.tree.sync(guild=guild)
        print(f"✅ Commands synced to guild {GUILD_ID}")
    else:
        await bot.tree.sync()
        print("✅ Commands synced globally")

@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user} ({bot.user.id})")

# ────────────────────────── Slash commands ───────────────────────
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    async with (await db_pool()).acquire() as conn:
        await conn.execute("INSERT INTO trainers (user_id) VALUES ($1) ON CONFLICT DO NOTHING", uid)
    await interaction.response.send_message("Trainer profile created!", ephemeral=True)


@bot.tree.command(description="Spawn a brand‑new creature egg")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        if not await conn.fetchrow("SELECT 1 FROM trainers WHERE user_id=$1", uid):
            return await interaction.response.send_message("Use /register first", ephemeral=True)

    await interaction.response.defer(thinking=True)

    roll = roll_d100()
    rarity = rarity_from_roll(roll)

    try:
        name, descriptors = await generate_creature_prompt(rarity)
    except Exception:
        logging.exception("OpenAI error")
        return await interaction.followup.send("Creature generator failed. Try again later.")

    stats = allocate_stats(rarity, descriptors)

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures (owner_id, name, rarity, descriptors, stats) VALUES ($1,$2,$3,$4,$5)",
            uid, name, rarity, descriptors, json.dumps(stats)
        )

    embed = discord.Embed(title=f"{name} ({rarity})", colour=0x8B0000)
    embed.add_field(name="Descriptors", value=", ".join(descriptors), inline=False)
    for s in PRIMARY_STATS:
        embed.add_field(name=s, value=str(stats[s]), inline=True)
    embed.set_footer(text=f"d100 roll: {roll}")

    await interaction.followup.send(embed=embed)


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    rows = await (await db_pool()).fetch("SELECT name, rarity, stats FROM creatures WHERE owner_id=$1", uid)
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)

    msg = []
