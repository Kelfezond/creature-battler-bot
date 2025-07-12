"""
Creature Battler Discord Bot – scaffolding v0.3b  (OpenAI 0.28 stable)
---------------------------------------------------------------------
Pins the OpenAI SDK to 0.28.x so we avoid the proxy-parameter crash
seen with 1.x on Railway.

Live slash-commands
• /register   – create trainer profile
• /spawn      – roll rarity, generate creature, store in DB
• /creatures  – list your roster
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
from discord.ext import commands

# ─── OpenAI pre-1.0 (openai==0.28.x) ───────────────────────────────
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Environment vars ─────────────────────────────────────────────
TOKEN     = os.getenv("DISCORD_TOKEN")
CLIENT_ID = os.getenv("CLIENT_ID")
GUILD_ID  = os.getenv("GUILD_ID")          # "" = global sync
DB_URL    = os.getenv("DATABASE_URL")

for k, v in {"DISCORD_TOKEN": TOKEN,
             "CLIENT_ID": CLIENT_ID,
             "DATABASE_URL": DB_URL,
             "OPENAI_API_KEY": openai.api_key}.items():
    if not v:
        raise RuntimeError(f"Missing env var: {k}")

# ─── Discord client ───────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─── Postgres helpers ─────────────────────────────────────────────
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

# ─── Game constants ───────────────────────────────────────────────
RARITY_TABLE = [
    (1, 75, "Common"),
    (76, 88, "Uncommon"),
    (89, 95, "Rare"),
    (96, 98, "Epic"),
    (99, 100, "Legendary"),
]

POINT_POOLS = {
    "Common":     (25,  50),
    "Uncommon":   (50, 100),
    "Rare":       (100, 200),
    "Epic":       (200, 400),
    "Legendary":  (400, 800),
}

PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]

BIAS_MAP = {
    "Rocky":           {"AR": +0.2, "SPD": -0.2},
    "Lightning-fast":  {"SPD": +0.2, "AR": -0.2},
    "Giant":           {"HP": +0.2, "SPD": -0.2},
}

# ─── Utility functions ────────────────────────────────────────────
def roll_d100() -> int:
    return random.randint(1, 100)


def rarity_from_roll(roll: int) -> str:
    for low, high, name in RARITY_TABLE:
        if low <= roll <= high:
            return name
    return "Common"


def allocate_stats(rarity: str, descriptors: List[str]) -> Dict[str, int]:
    pool = random.randint(*POINT_POOLS[rarity])
    stats = {s: 1 for s in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)

    weights = {s: 1.0 for s in PRIMARY_STATS}
    for desc in descriptors:
        for stat, delta in BIAS_MAP.get(desc, {}).items():
            weights[stat] = max(0.1, weights[stat] + delta)

    total_w = sum(weights.values())
    for _ in range(pool):
        r = random.uniform(0, total_w)
        acc = 0
        for stat, w in weights.items():
            acc += w
            if r <= acc:
                stats[stat] += 1
                break
    return stats


async def ask_openai(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    fn = partial(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": "Generate now."}],
        temperature=0.9,
        max_tokens=50,
    )
    resp = await loop.run_in_executor(None, fn)
    return resp.choices[0].message.content.strip()


async def generate_creature(rarity: str) -> Dict[str, any]:
    """
    Ask GPT for a creature and return
      { "name": str, "descriptors": [str, str, str] }
    """
    prompt = textwrap.dedent(f"""
        You are inventing a dark-fantasy arena monster for a gritty creature-battler game.

        Reply **only** with valid JSON in this exact schema:

        {{
          "name": "<short unique name (1-3 words)>",
          "descriptors": ["<descriptor1>", "<descriptor2>", "<descriptor3>"]
        }}

        No markdown, no extra keys, no newlines before or after the JSON block.
        Creature rarity: {rarity}.
    """)

    # blocking call -> run in executor
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.9,
            max_tokens=120,
        ),
    )
    txt = resp.choices[0].message.content.strip()

    try:
        data = json.loads(txt)
        assert isinstance(data["descriptors"], list) and len(data["descriptors"]) == 3
        return data
    except Exception as e:
        raise ValueError(f"Bad GPT JSON: {txt}") from e

# ─── Bot lifecycle ────────────────────────────────────────────────
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

# ─── Slash commands ──────────────────────────────────────────────
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    async with (await db_pool()).acquire() as conn:
        await conn.execute(
            "INSERT INTO trainers (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
            uid,
        )
    await interaction.response.send_message(
        "Trainer profile created!", ephemeral=True
    )


@bot.tree.command(description="Spawn a brand-new creature egg")
async def spawn(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()

    # ensure trainer exists
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
        return await interaction.followup.send(
            "Creature generator failed. Try again later."
        )

    stats = allocate_stats(rarity, ai["descriptors"])

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures (owner_id, name, rarity, descriptors, stats)"
            " VALUES ($1,$2,$3,$4,$5)",
            uid,
            ai["name"],
            rarity,
            ai["descriptors"],
            json.dumps(stats),
        )

    embed = discord.Embed(
        title=f"{ai['name']} ({rarity})",
        colour=0x8B0000,
    )
    embed.add_field(name="Descriptors", value=", ".join(ai["descriptors"]), inline=False)
    for s in PRIMARY_STATS:
        embed.add_field(name=s, value=str(stats[s]), inline=True)
    embed.set_footer(text=f"d100 roll: {roll}")

    await interaction.followup.send(embed=embed)


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    rows = await (await db_pool()).fetch(
        "SELECT name, rarity, stats FROM creatures WHERE owner_id=$1 ORDER BY id",
        uid,
    )
    if not rows:
        return await interaction.response.send_message(
            "You own no creatures yet.", ephemeral=True
        )

    lines: List[str] = []
    for i, r in enumerate(rows, 1):
        stat_dict = (
            r["stats"] if isinstance(r["stats"], dict) else json.loads(r["stats"])
        )
        stat_summary = ", ".join(f"{k}:{v}" for k, v in stat_dict.items())
        lines.append(f"{i}. **{r['name']}** ({r['rarity']}) – {stat_summary}")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)

# ─── Main entry ──────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)
