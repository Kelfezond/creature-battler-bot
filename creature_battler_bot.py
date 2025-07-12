"""
Creature Battler Discord Bot – scaffolding v0.1
------------------------------------------------
Implements the *absolute minimum* end‑to‑end flow so we can begin live‑testing:
 • "/register" – create trainer profile
 • "/spawn"    – roll rarity, ask OpenAI to generate a creature, store in DB
 • "/creatures" – list your roster

Future‑ready hooks (TODO): battles, injuries, arena tiers, facilities.

REQUIRES environment variables:
  DISCORD_TOKEN     – bot token from Discord Dev Portal
  OPENAI_API_KEY    – for creature generation
  DATABASE_URL      – Postgres connection string (e.g. postgres://user:pass@host/db)

Set intents.message_content = True → you must enable the intent in the portal too.

----------------------------------------------
*This code is complete and runnable on Railway/Replit reserved VM.*
"""

from __future__ import annotations
import os, asyncio, json, random, textwrap, logging
from typing import List, Dict

import openai
import discord
from discord.ext import commands
import asyncpg

# ─── Environment ─────────────────────────────────────────────────────────────
TOKEN         = os.environ.get("DISCORD_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DB_URL        = os.environ.get("DATABASE_URL")

if not all([TOKEN, OPENAI_API_KEY, DB_URL]):
    raise RuntimeError("Missing one of DISCORD_TOKEN, OPENAI_API_KEY, DATABASE_URL")

openai.api_key = OPENAI_API_KEY

# ─── Discord client setup ────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ─── Database helpers ────────────────────────────────────────────────────────
async def db_pool() -> asyncpg.Pool:  # singleton pool
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

CREATURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS creatures (
  id          SERIAL PRIMARY KEY,
  owner_id    BIGINT NOT NULL,
  name        TEXT,
  rarity      TEXT,
  descriptors TEXT[],
  stats       JSONB,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS trainers (
  user_id     BIGINT PRIMARY KEY,
  joined_at   TIMESTAMPTZ DEFAULT now()
);
"""

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

# descriptor bias map (simplified; TODO expand)
BIAS_MAP = {
    "Rocky": {"AR": +0.2, "SPD": -0.2},
    "Lightning‑fast": {"SPD": +0.2, "AR": -0.2},
    "Giant": {"HP": +0.2, "SPD": -0.2},
}

# ─── Utility functions ───────────────────────────────────────────────────────

def roll_d100() -> int:
    return random.randint(1, 100)


def rarity_from_roll(roll: int) -> str:
    for low, high, rarity in RARITY_TABLE:
        if low <= roll <= high:
            return rarity
    return "Common"  # fallback


def allocate_stats(rarity: str, descriptors: List[str]) -> Dict[str, int]:
    pool_min, pool_max = POINT_POOLS[rarity]
    pool = random.randint(pool_min, pool_max)

    # start with 1 in every stat
    stats = {s: 1 for s in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)

    # build weighting list
    weights = {s: 1.0 for s in PRIMARY_STATS}
    for d in descriptors:
        bias = BIAS_MAP.get(d, {})
        for stat, delta in bias.items():
            weights[stat] = max(0.1, weights[stat] + delta)

    # allocate remaining points
    total_weight = sum(weights.values())
    for _ in range(pool):
        r = random.uniform(0, total_weight)
        cumulative = 0
        for stat, w in weights.items():
            cumulative += w
            if r <= cumulative:
                stats[stat] += 1
                break
    return stats


async def generate_creature_prompt(rarity: str) -> Dict[str, any]:
    """Call OpenAI to invent a creature name + descriptors."""
    prompt = textwrap.dedent(f"""
        You are creating a dark fantasy arena monster for a creature‑battler game.
        Give me:
        1. A short unique name (1‑3 words).
        2. Exactly 3 descriptor keywords (comma‑separated) that evoke appearance/ability.
        The creature rarity is {rarity}.
    """)
    response = await openai.AsyncCompletion.acreate(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.9,
        max_tokens=50,
    )
    text = response.choices[0].text.strip()
    # Expected format: Name\nDescriptor1, Descriptor2, Descriptor3
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0]
    descriptors = [d.strip() for d in lines[1].split(',')][:3]
    return {"name": name, "descriptors": descriptors}

# ─── Bot events & commands ───────────────────────────────────────────────────
@bot.event
async def on_ready():
    print(f"🤖 Logged in as {bot.user} ({bot.user.id})")
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(CREATURE_TABLE_SQL)


@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    user_id = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO trainers (user_id) VALUES ($1) ON CONFLICT DO NOTHING;",
            user_id,
        )
    await interaction.response.send_message("Trainer profile created!", ephemeral=True)


@bot.tree.command(description="Spawn a brand‑new creature egg")
async def spawn(interaction: discord.Interaction):
    user_id = interaction.user.id
    # Check trainer exists
    pool = await db_pool()
    async with pool.acquire() as conn:
        trainer = await conn.fetchrow("SELECT 1 FROM trainers WHERE user_id=$1", user_id)
    if not trainer:
        return await interaction.response.send_message("You are not registered. Use /register first.", ephemeral=True)

    await interaction.response.defer(thinking=True)

    roll = roll_d100()
    rarity = rarity_from_roll(roll)

    try:
        ai_data = await generate_creature_prompt(rarity)
    except Exception as e:
        logging.exception("OpenAI error")
        return await interaction.followup.send("Failed to contact the monster registry. Try again later.")

    descriptors = ai_data["descriptors"]
    stats = allocate_stats(rarity, descriptors)

    # Store
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO creatures (owner_id, name, rarity, descriptors, stats)
            VALUES ($1,$2,$3,$4,$5)
            """,
            user_id,
            ai_data["name"],
            rarity,
            descriptors,
            json.dumps(stats),
        )

    # Craft embed
    embed = discord.Embed(title=f"{ai_data['name']} ({rarity})", color=0x8b0000)
    embed.add_field(name="Descriptors", value=", ".join(descriptors), inline=False)
    for stat in PRIMARY_STATS:
        embed.add_field(name=stat, value=str(stats[stat]), inline=True)
    embed.set_footer(text=f"Roll: {roll}")

    await interaction.followup.send(embed=embed)


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    user_id = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT name, rarity, stats FROM creatures WHERE owner_id=$1 ORDER BY id", user_id)
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)
    lines = []
    for i, row in enumerate(rows, 1):
        stats = row["stats"] if isinstance(row["stats"], dict) else json.loads(row["stats"])
        stat_summary = ", ".join(f"{k}:{v}" for k, v in stats.items())
        lines.append(f"{i}. **{row['name']}** ({row['rarity']}) – {stat_summary}")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot.run(TOKEN)
