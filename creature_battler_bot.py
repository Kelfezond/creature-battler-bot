"""
Creature Battler Discord Bot – scaffolding **v0.3b**
---------------------------------------------------
This patch upgrades the OpenAI call to the **openai‑python ≥ 1.x API** that Railway
installs from `requirements.txt` (openai==1.25.0).  No other behaviour changes.

Live commands:
• /register   – create trainer profile
• /spawn      – roll rarity, generate creature, store in DB
• /creatures  – list your roster
"""

from __future__ import annotations

import json
import logging
import os
import random
import textwrap
from typing import Dict, List

import asyncpg
import discord
from discord.ext import commands

# --- OpenAI ≥ 1.x ------------------------------------------------------------
import openai
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ────────────────────────── Environment ------------------------------------
TOKEN     = os.environ.get("DISCORD_TOKEN")
CLIENT_ID = os.environ.get("CLIENT_ID")
GUILD_ID  = os.environ.get("GUILD_ID")  # "" = global registration
DB_URL    = os.environ.get("DATABASE_URL")

for key, val in {"DISCORD_TOKEN": TOKEN, "CLIENT_ID": CLIENT_ID, "DATABASE_URL": DB_URL}.items():
    if not val:
        raise RuntimeError(f"Missing env var: {key}")

# ────────────────────────── Discord client ----------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ────────────────────────── DB schema ---------------------------------------
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

async def db_pool() -> asyncpg.pool.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ────────────────────────── Game constants ----------------------------------
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

# ────────────────────────── Helpers -----------------------------------------
def roll_d100() -> int:
    return random.randint(1, 100)

def rarity_from_roll(r: int) -> str:
    for low, high, name in RARITY_TABLE:
        if low <= r <= high:
            return name
    return "Common"

def allocate_stats(rarity: str, descriptors: List[str]) -> Dict[str, int]:
    low, high = POINT_POOLS[rarity]
    pool = random.randint(low, high)
    stats = {s: 1 for s in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)

    weights = {s: 1.0 for s in PRIMARY_STATS}
    for d in descriptors:
        for stat, delta in BIAS_MAP.get(d, {}).items():
            weights[stat] = max(0.1, weights[stat] + delta)
    total = sum(weights.values())
    for _ in range(pool):
        r = random.uniform(0, total)
        acc = 0
        for stat, w in weights.items():
            acc += w
            if r <= acc:
                stats[stat] += 1
                break
    return stats

async def generate_creature(rarity: str) -> Dict[str, any]:
    system_prompt = textwrap.dedent(f"""
        You are inventing a dark‑fantasy monster for an arena creature‑battler game.
        Respond with **exactly two lines**:
        1. A short unique name (1‑3 words).
        2. Exactly THREE evocative descriptor keywords, comma‑separated.
        Creature rarity: {rarity}.
    """)

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.9,
        max_tokens=50,
    )
    text = completion.choices[0].message.content.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 2:
        raise ValueError("OpenAI response malformed")
    return {
        "name": lines[0],
        "descriptors": [d.strip() for d in lines[1].split(",")][:3],
    }

# ────────────────────────── Bot lifecycle ------------------------------------
@bot.event
async def setup_hook():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    if GUILD_ID:
        guild_obj = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild_obj)
        await bot.tree.sync(guild=guild_obj)
        print(f"✅ Slash commands synced to guild {GUILD_ID}")
    else:
        await bot.tree.sync()
        print("✅ Slash commands synced globally")

@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user} ({bot.user.id})")

# ────────────────────────── Commands ----------------------------------------
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO trainers (user_id) VALUES ($1) ON CONFLICT DO NOTHING", uid
        )
    await interaction.response.send_message("Trainer profile created!", ephemeral=True)

@bot.tree.command(description="Spawn a brand‑new creature egg")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        if not await conn.fetchrow("SELECT 1 FROM trainers WHERE user_id=$1", uid):
            return await interaction.response.send_message(
                "You are not registered. Use /register first.", ephemeral=True
            )

    await interaction.response.defer(thinking=True)
    roll = roll_d100()
    rarity = rarity_from_roll(roll)

    try:
        ai = await generate_creature(rarity)
    except Exception:
        logging.exception("OpenAI error")
        return await interaction.followup.send("Creature generator failed. Try again later.")

    stats = allocate_stats(rarity, ai["descriptors"])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures (owner_id, name, rarity, descriptors, stats) VALUES ($1,$2,$3,$4,$5)",
            uid,
            ai["name"],
            rarity,
            ai["descriptors"],
            json.dumps(stats),
        )

    embed = discord.Embed(title=f"{ai['name']} ({rarity})", colour=0x8B0000)
    embed.add_field(name="Descriptors", value=", ".join(ai["descriptors"]), inline=False)
    for s in PRIMARY_STATS:
        embed.add_field(name=s, value=str(stats[s]), inline=True)
    embed.set_footer(text=f"d100 roll: {roll}")

    await interaction.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    rows = await (await db_pool()).fetch(
        "SELECT name, rarity, stats FROM creatures WHERE owner_id=$1 ORDER BY id", uid
    )
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)

    lines = []
    for i, r in enumerate(rows, 1):
        s = r["stats"] if isinstance(r["stats"], dict) else json.loads(r["stats"])
        lines.append(f"{i}. **{r['name']}** ({r['rarity']}) – " + ", ".join(f"{k}:{v}" for k, v in s.items()))

    await interaction.response.send_message("\n".join(lines), ephemeral=True)

# ────────────────────────── Main --------------------------------------------
if __name__ == "__main__":
    logging
