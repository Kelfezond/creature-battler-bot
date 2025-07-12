from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import re
import textwrap
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

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
bot = commands.Bot(command_prefix="!", intents=intents)

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
  abilities JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game Constants ──────────────────────────────────────────
RARITY_TABLE = [
    (1, 75, "Common"),
    (76, 88, "Uncommon"),
    (89, 95, "Rare"),
    (96, 98, "Epic"),
    (99, 100, "Legendary"),
]
POINT_POOLS = {
    "Common":    (25, 50),
    "Uncommon":  (50, 100),
    "Rare":      (100, 200),
    "Epic":      (200, 400),
    "Legendary": (400, 800),
}
PRIMARY_STATS = ["HP", "AR", "PATK", "SATK", "SPD"]
BIAS_MAP = {
    "Rocky":          {"AR": +0.2, "SPD": -0.2},
    "Lightning-fast": {"SPD": +0.2, "AR": -0.2},
    "Giant":          {"HP": +0.2, "SPD": -0.2},
}

# ─── Utility Functions ───────────────────────────────────────
def roll_d100() -> int:
    return random.randint(1, 100)


def rarity_from_roll(r: int) -> str:
    for low, high, name in RARITY_TABLE:
        if low <= r <= high:
            return name
    return "Common"


def allocate_stats(rarity: str, descriptors: List[str]) -> Dict[str, int]:
    pool = random.randint(*POINT_POOLS[rarity])
    stats = {s: 1 for s in PRIMARY_STATS}
    pool -= len(PRIMARY_STATS)

    weights = {s: 1.0 for s in PRIMARY_STATS}
    for d in descriptors:
        for stat, delta in BIAS_MAP.get(d, {}).items():
            weights[stat] = max(0.1, weights[stat] + delta)

    total = sum(weights.values())
    for _ in range(pool):
        r = random.uniform(0, total)
        acc = 0.0
        for stat, w in weights.items():
            acc += w
            if r <= acc:
                stats[stat] += 1
                break
    return stats


async def fetch_used_lists() -> tuple[list[str], list[str]]:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    names = [r["name"].lower() for r in rows][:40]
    words = {w.lower() for r in rows for w in r["descriptors"]}
    return names, list(words)[:80]


def _fix_json(txt: str) -> str:
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")
    txt = re.sub(r',\s*([}\]])', r'\1', txt)
    # balance braces/brackets
    txt += '}' * max(0, txt.count('{') - txt.count('}'))
    txt += ']' * max(0, txt.count('[') - txt.count(']'))
    return txt.strip()


async def ask_openai(prompt: str, max_tokens: int = 400) -> Tuple[str, Optional[str]]:
    """
    Sends prompt to OpenAI and returns (content, finish_reason).
    """
    loop = asyncio.get_running_loop()
    fn = partial(
        openai.ChatCompletion.create,
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user",   "content": "Generate now."}],
        temperature=1.2,
        presence_penalty=0.8,
        max_tokens=max_tokens,
    )
    resp = await loop.run_in_executor(None, fn)
    choice = resp.choices[0]
    return choice.message.content.strip(), choice.finish_reason


async def generate_creature_json(rarity: str) -> Dict[str, Any]:
    names_used, words_used = await fetch_used_lists()
    avoid_names = ", ".join(names_used)
    avoid_words = ", ".join(words_used)

    base_prompt = textwrap.dedent(f"""
        You are inventing a dark-fantasy arena monster.

        Reply ONLY with JSON:
        {{
          "name": "string (1-3 words)",
          "descriptors": ["word1","word2","word3"],
          "abilities": [
            {{
              "name":"string",
              "type":"physical|special|utility",
              "damage_mod": int,
              "defense_mod": int,
              "speed_mod": int,
              "weight": int
            }}
          ]
        }}

        • 3–5 abilities total.
        • Lower total effect ⇒ higher weight (1–100).
        • Do NOT repeat exact names: {avoid_names}
        • Try to avoid descriptors already used: {avoid_words}
        • No markdown, no extra keys.

        Creature rarity: {rarity}
    """)

    # First 3 attempts
    for attempt in range(3):
        text, finish_reason = await ask_openai(base_prompt)
        logger.debug("GPT RAW [%s]: %s", finish_reason, text[:200])
        if finish_reason == "length":
            logger.warning("GPT response truncated (length); retrying attempt %d", attempt + 1)
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data = json.loads(_fix_json(text))
            except json.JSONDecodeError:
                logger.warning("JSON decode failed on attempt %d", attempt + 1)
                continue
        # Validate
        if data.get("name", "").lower() in names_used:
            logger.info("Duplicate name detected on attempt %d: %s", attempt + 1, data.get("name"))
            continue
        if not 3 <= len(data.get("abilities", [])) <= 5:
            logger.info("Invalid abilities count on attempt %d: %d", attempt + 1, len(data.get("abilities", [])))
            continue
        return data

    # Final seeded attempts (two retries)
    for seed_attempt in range(2):
        seed = random.randint(1, 1_000_000)
        seed_prompt = base_prompt + f"\nSEED:{seed}"
        text, _ = await ask_openai(seed_prompt)
        logger.debug("GPT RAW [seed %d]: %s", seed, text[:200])
        try:
            return json.loads(_fix_json(text))
        except json.JSONDecodeError:
            logger.warning("Seeded JSON decode failed on seed %d", seed)
            continue

    raise RuntimeError("Failed to parse valid creature JSON after retries")

# ─── Bot Lifecycle Events ─────────────────────────────────────
@bot.event
async def setup_hook():
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)

    if GUILD_ID:
        guild = discord.Object(id=int(GUILD_ID))
        bot.tree.copy_global_to(guild=guild)
        await bot.tree.sync(guild=guild)
        logger.info("Commands synced to guild %s", GUILD_ID)
    else:
        await bot.tree.sync()
        logger.info("Commands synced globally")


@bot.event
async def on_ready():
    logger.info("Logged in as %s (%s)", bot.user, bot.user.id)

# ─── Slash Commands ──────────────────────────────────────────
@bot.tree.command(description="Register yourself as a trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO trainers (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
            uid,
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

    try:
        ai = await generate_creature_json(rarity)
    except Exception as e:
        logger.exception("Creature generation error")
        return await interaction.followup.send(
            "Creature generator failed. Try again later.",
        )

    stats = allocate_stats(rarity, ai["descriptors"])

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures (owner_id,name,rarity,descriptors,stats,abilities)"
            " VALUES ($1,$2,$3,$4,$5,$6)",
            uid,
            ai["name"],
            rarity,
            ai["descriptors"],
            json.dumps(stats),
            json.dumps(ai["abilities"]),
        )

    embed = discord.Embed(title=f"{ai['name']} ({rarity})", colour=0x8B0000)
    embed.add_field(name="Descriptors", value=", ".join(ai["descriptors"]), inline=False)
    for stat in PRIMARY_STATS:
        embed.add_field(name=stat, value=str(stats[stat]), inline=True)

    ability_lines = []
    for ab in ai["abilities"]:
        ability_lines.append(
            f"• {ab['name']} ({ab['type']}, dmg {ab['damage_mod']}%, "
            f"def {ab['defense_mod']}%, spd {ab['speed_mod']}%, w={ab['weight']})"
        )
    embed.add_field(name="Abilities", value="\n".join(ability_lines), inline=False)
    embed.set_footer(text=f"d100 roll: {roll}")

    await interaction.followup.send(embed=embed)


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    pool = await db_pool()
    rows = await pool.fetch(
        "SELECT name, rarity, stats, abilities FROM creatures WHERE owner_id=$1 ORDER BY id",
        uid,
    )
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)

    lines: List[str] = []
    for idx, r in enumerate(rows, start=1):
        stats = r["stats"] if isinstance(r["stats"], dict) else json.loads(r["stats"])
        abilities = (
            r["abilities"] if isinstance(r["abilities"], list) else json.loads(r["abilities"])
        )
        stat_str = ", ".join(f"{k}:{v}" for k, v in stats.items())
        abil_str = ", ".join(ab["name"] for ab in abilities)
        lines.append(f"{idx}. **{r['name']}** ({r['rarity']}) – {stat_str} – [{abil_str}]")

    await interaction.response.send_message("\n".join(lines), ephemeral=True)

# ─── Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(TOKEN)
