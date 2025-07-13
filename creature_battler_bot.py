from __future__ import annotations
import asyncio
import json
import logging
import os
import random
import re
import textwrap
import math
from dataclasses import dataclass
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
TIER_EXTRAS = {
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
BIAS_MAP = {
    "Rocky":          {"AR": +0.2, "SPD": -0.2},
    "Lightning-fast": {"SPD": +0.2, "AR": -0.2},
    "Giant":          {"HP": +0.2, "SPD": -0.2},
}

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


async def fetch_used_lists() -> Tuple[List[str], List[str]]:
    pool = await db_pool()
    rows = await pool.fetch("SELECT name, descriptors FROM creatures")
    names = [r["name"].lower() for r in rows][:40]
    words = {w.lower() for r in rows for w in r["descriptors"]}
    return names, list(words)[:80]


def _fix_json(txt: str) -> str:
    txt = txt.replace("“", '"').replace("”", '"').replace("’", "'")
    txt = re.sub(r',\s*([}\]])', r'\1', txt)
    txt += '}' * max(0, txt.count('{') - txt.count('}'))
    txt += ']' * max(0, txt.count('[') - txt.count(']'))
    return txt.strip()


async def ask_openai(prompt: str, max_tokens: int = 400) -> Tuple[str, Optional[str]]:
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
        You are inventing a creature.

        Reply ONLY with JSON:
        {{
          "name": "string (1-3 words)",
          "descriptors": ["word1","word2","word3"],
          "abilities": [
            {{
              "name": "string",
              "type": "physical|special|utility",
              "damage_mod": int,
              "defense_mod": int,
              "speed_mod": int,
              "weight": int
            }}
          ]
        }}

        • Exactly 3 abilities:
          1) Offensive: physical if PATK ≥ SATK else special.
          2) Utility: type utility.
          3) Ultimate: same type as offensive but high damage_mod, low weight.
        • Lower total effect ⇒ higher weight (1–100).
        • Avoid names: {avoid_names}
        • Avoid descriptors: {avoid_words}
        • No markdown, no extra keys.

        Creature rarity: {rarity}
    """)
    for attempt in range(3):
        text, reason = await ask_openai(base_prompt)
        logger.debug("GPT RAW [%s]: %s", reason, text[:200])
        if reason == "length":
            logger.warning("Truncated reply; retry %d", attempt+1)
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data = json.loads(_fix_json(text))
            except Exception:
                continue
        if data.get("name",""
        ).lower() in names_used or len(data.get("abilities",[])) != 3:
            continue
        return data
    for _ in range(2):
        seed = random.randint(1,1_000_000)
        text,_ = await ask_openai(base_prompt + f"\nSEED:{seed}")
        try:
            data = json.loads(_fix_json(text))
        except Exception:
            continue
        if data.get("name",""
        ).lower() in names_used or len(data.get("abilities",[])) != 3:
            continue
        return data
    raise RuntimeError("Failed to generate creature JSON")

# ─── Battle Simulation ──────────────────────────────────────
def choose_ability(creature: Dict[str, Any]) -> Dict[str, Any]:
    abilities = creature["abilities"]
    weights = [ab["weight"] for ab in abilities]
    return random.choices(abilities, weights=weights, k=1)[0]


def simulate_round(state: BattleState):
    # determine turn order based on SPD
    u_spd = state.user_creature["stats"]["SPD"]
    o_spd = state.opp_creature["stats"]["SPD"]
    if u_spd > o_spd or (u_spd == o_spd and random.choice([True, False])):
        order = [("user", state.user_creature), ("opp", state.opp_creature)]
    else:
        order = [("opp", state.opp_creature), ("user", state.user_creature)]

    for actor, creature in order:
        # break if someone is down
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        chosen = choose_ability(creature)
        name = creature["name"]

        # UTILITY: either buff AR or heal HP
        if chosen["type"] == "utility":
            buff = chosen.get("defense_mod", 0)
            if buff > 0:
                if actor == "user":
                    state.user_creature["stats"]["AR"] += buff
                else:
                    state.opp_creature["stats"]["AR"] += buff
                state.logs.append(f"{name} used {chosen['name']} and increased AR by {buff}.")
            else:
                heal_amt = math.ceil(chosen.get("weight", 0) / 10)
                if actor == "user":
                    state.user_current_hp = min(state.user_max_hp, state.user_current_hp + heal_amt)
                else:
                    state.opp_current_hp = min(state.opp_max_hp, state.opp_current_hp + heal_amt)
                state.logs.append(f"{name} used {chosen['name']} and healed {heal_amt} HP.")
            continue

        # OFFENSIVE: damage roll
        stats = creature["stats"]
        S = stats["PATK"] if chosen["type"] == "physical" else stats["SATK"]
        N = math.ceil(S / 10)
        roll_sum = sum(random.randint(1, 6) for _ in range(N))

        # identify ultimate by highest damage_mod
        max_mod = max(ab["damage_mod"] for ab in creature["abilities"])
        is_ult = chosen["damage_mod"] >= max_mod
        if is_ult:
            if random.random() > 0.3:
                state.logs.append(f"{name} tried {chosen['name']} but missed.")
                continue
            damage_roll = roll_sum * 2
        else:
            damage_roll = roll_sum

        # apply AR for physical attacks
        if chosen["type"] == "physical":
            target_stats = state.opp_creature["stats"] if actor == "user" else state.user_creature["stats"]
            damage = max(1, damage_roll - target_stats.get("AR", 0))
        else:
            damage = damage_roll

        # deduct HP
        if actor == "user":
            state.opp_current_hp -= damage
            state.logs.append(f"{name} used {chosen['name']} and dealt {damage} damage.")
        else:
            state.user_current_hp -= damage
            state.logs.append(f"{name} used {chosen['name']} and dealt {damage} damage.")

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
        await conn.execute("INSERT INTO trainers(user_id) VALUES($1) ON CONFLICT DO NOTHING", interaction.user.id)
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
    except Exception:
        logger.exception("Gen error")
        return await interaction.followup.send("Creature generator failed. Try again later.")
    stats = allocate_stats(rarity, ai["descriptors"])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO creatures(owner_id,name,rarity,descriptors,stats,abilities)"
            " VALUES($1,$2,$3,$4,$5,$6)",
            uid, ai["name"], rarity, ai["descriptors"], json.dumps(stats), json.dumps(ai["abilities"])
        )
    embed = discord.Embed(title=f"{ai['name']} ({rarity})", color=0x8B0000)
    embed.add_field(name="Descriptors", value=", ".join(ai["descriptors"]), inline=False)
    for s in PRIMARY_STATS:
        embed.add_field(name=s, value=str(stats[s]), inline=True)
    ab_lines = [f"• {ab['name']} ({ab['type']}, dmg {ab['damage_mod']}%, def {ab['defense_mod']}%, spd {ab['speed_mod']}%, w={ab['weight']})" for ab in ai['abilities']]
    embed.add_field(name="Abilities", value="\n".join(ab_lines), inline=False)
    embed.set_footer(text=f"d100 roll: {roll}")
    await interaction.followup.send(embed=embed)

@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    rows = await (await db_pool()).fetch("SELECT name,rarity,stats,abilities FROM creatures WHERE owner_id=$1 ORDER BY id", interaction.user.id)
    if not rows:
        return await interaction.response.send_message("You own no creatures yet.", ephemeral=True)
    lines: List[str] = []
    for i, r in enumerate(rows, 1):
        stats = r['stats'] if isinstance(r['stats'], dict) else json.loads(r['stats'])
        abs_list = r['abilities'] if isinstance(r['abilities'], list) else json.loads(r['abilities'])
        stat_str = ", ".join(f"{k}:{v}" for k, v in stats.items())
        abil_str = ", ".join(ab['name'] for ab in abs_list)
        lines.append(f"{i}. **{r['name']}** ({r['rarity']}) – {stat_str} – [{abil_str}]")
    await interaction.response.send_message("\n".join(lines), ephemeral=True)

@bot.tree.command(description="Battle your creature against a tiered opponent")
async def battle(interaction: discord.Interaction, creature_name: str, tier: int):
    uid = interaction.user.id
    if tier not in TIER_EXTRAS:
        return await interaction.response.send_message("Invalid tier.", ephemeral=True)
    # fetch user creature
    row = await (await db_pool()).fetchrow("SELECT name,rarity,descriptors,stats,abilities FROM creatures WHERE owner_id=$1 AND name ILIKE $2", uid, creature_name)
    if not row:
        return await interaction.response.send_message("You don't own a creature with that name.", ephemeral=True)
    # prepare user creature
    user_stats = row['stats'] if isinstance(row['stats'], dict) else json.loads(row['stats'])
    user_abilities = row['abilities'] if isinstance(row['abilities'], list) else json.loads(row['abilities'])
    user_creature = {"name": row['name'], "stats": user_stats, "abilities": user_abilities}
    # spawn opponent
    roll = roll_d100()
    rarity = rarity_from_roll(roll)
    ai = await generate_creature_json(rarity)
    extra = random.randint(*TIER_EXTRAS[tier])
    opp_stats = allocate_stats(rarity, ai['descriptors'], extra)
    opp_creature = {"name": ai['name'], "stats": opp_stats, "abilities": ai['abilities']}
    # init battle state
    state = BattleState(
        user_id=uid,
        user_creature=user_creature,
        user_current_hp=user_stats['HP'],
        user_max_hp=user_stats['HP'],
        opp_creature=opp_creature,
        opp_current_hp=opp_stats['HP'],
        opp_max_hp=opp_stats['HP'],
        logs=[]
    )
    active_battles[uid] = state
    # simulate first chunk
    for _ in range(10):
        if state.user_current_hp <= 0 or state.opp_current_hp <= 0:
            break
        simulate_round(state)
    # send logs
    out = []
    for line in state.logs:
        out.append(line)
    # check winner
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
    out = []
    for line in state.logs[state.next_log_idx:]:
        out.append(line)
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
