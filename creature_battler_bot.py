from __future__ import annotations

"""Creature Battler Discord bot
--------------------------------
Re‑written with the following improvements:
  • Added dotenv support (optional) and clearer env‑var validation.
  • Added graceful shutdown that closes the asyncpg pool.
  • `/battle` now defers its interaction reply and has robust exception handling (mirrors `/spawn`).
  • Prevents users from starting multiple concurrent battles.
  • Utility abilities now handle *positive* buffs, *negative* debuffs, *speed* modifiers, and healing distinctly.
  • Minor refactors (Creature dataclass, helper functions) for clarity while remaining a single‑file script.
This keeps the original behaviour but shores up stability ahead of larger changes.
"""

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import asyncpg
import discord
import dotenv
import openai
from discord import app_commands
from discord.ext import commands

###############################################################################
# Configuration & startup helpers
###############################################################################

# Load .env in local development (ignored in prod if env vars already set)
dotenv.load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GUILD_ID = os.getenv("GUILD_ID")  # optional; speeds up command sync when set

missing = [v for v, name in ((DISCORD_TOKEN, "DISCORD_TOKEN"), (DATABASE_URL, "DATABASE_URL"), (OPENAI_API_KEY, "OPENAI_API_KEY")) if v is None]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

openai.api_key = OPENAI_API_KEY  # pre‑1.0 OpenAI SDK

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creature_battler")

###############################################################################
# Data classes & helpers
###############################################################################

STAT_NAMES = ("HP", "ATK", "DEF", "SPD")

@dataclass
class Ability:
    name: str
    description: str
    type: str  # "attack" | "utility"
    damage_mod: int = 0
    defense_mod: int = 0
    speed_mod: int = 0
    weight: float = 1.0  # used for healing amount if utility heal

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Ability":
        return cls(
            name=d.get("name", "Unnamed"),
            description=d.get("description", ""),
            type=d.get("type", "attack"),
            damage_mod=int(d.get("damage_mod", 0)),
            defense_mod=int(d.get("defense_mod", 0)),
            speed_mod=int(d.get("speed_mod", 0)),
            weight=float(d.get("weight", 1.0)),
        )

@dataclass
class Creature:
    owner_id: int
    name: str
    tier: str
    stats: Dict[str, int]
    ability: Ability
    current_hp: int = field(init=False)
    current_def: int = field(init=False)
    current_spd: int = field(init=False)

    def __post_init__(self):
        self.current_hp = self.stats["HP"]
        self.current_def = self.stats["DEF"]
        self.current_spd = self.stats["SPD"]

    @classmethod
    def from_record(cls, rec: asyncpg.Record) -> "Creature":
        ability = Ability.from_dict(json.loads(rec["ability_json"]))
        stats = {
            "HP": rec["hp"],
            "ATK": rec["atk"],
            "DEF": rec["def"],
            "SPD": rec["spd"],
        }
        return cls(rec["owner_id"], rec["name"], rec["tier"], stats, ability)

###############################################################################
# Discord bot setup
###############################################################################

class CreatureBattlerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True  # privileged intent
        super().__init__(command_prefix="!", intents=intents)
        self.db_pool: asyncpg.Pool | None = None
        self.active_battles: Dict[int, Tuple[Creature, Creature]] = {}

    async def setup_hook(self):
        # DB init
        self.db_pool = await asyncpg.create_pool(DATABASE_URL)
        await self._ensure_schema()

        # Command sync (if GUILD_ID set, use guild‑only for faster updates)
        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Commands synced to guild %s", GUILD_ID)
        else:
            await self.tree.sync()
            logger.info("Commands synced globally (may take up to an hour)")

        logger.info("Creature Battler Bot is ready!")

    async def close(self):
        if self.db_pool:
            await self.db_pool.close()
        await super().close()

    async def _ensure_schema(self):
        assert self.db_pool is not None
        ddl = """
        CREATE TABLE IF NOT EXISTS users (
            id BIGINT PRIMARY KEY,
            username TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS creatures (
            id SERIAL PRIMARY KEY,
            owner_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            tier TEXT NOT NULL,
            hp INT NOT NULL,
            atk INT NOT NULL,
            def INT NOT NULL,
            spd INT NOT NULL,
            ability_json JSONB NOT NULL
        );
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(ddl)

bot = CreatureBattlerBot()

###############################################################################
# Utility functions
###############################################################################

def allocate_stats(tier: str) -> Dict[str, int]:
    base_points = {"bronze": 40, "silver": 55, "gold": 70}.get(tier, 40)
    # Simple random distribution with minimums
    hp = random.randint(10, 20)
    atk = random.randint(5, 15)
    def_ = random.randint(5, 15)
    spd = random.randint(5, 15)
    remaining = base_points - (hp + atk + def_ + spd)
    for _ in range(remaining):
        choice = random.choice(["HP", "ATK", "DEF", "SPD"])
        if choice == "HP":
            hp += 1
        elif choice == "ATK":
            atk += 1
        elif choice == "DEF":
            def_ += 1
        else:
            spd += 1
    return {"HP": hp, "ATK": atk, "DEF": def_, "SPD": spd}

async def generate_creature_from_openai(name: str, tier: str) -> Ability:
    """Generate a single ability via OpenAI and return it as an Ability object."""
    prompt = (
        "You are designing a turn‑based RPG creature ability for a Discord bot. "
        "Return *only* JSON matching this schema: {name, description, type, damage_mod, defense_mod, speed_mod, weight}. "
        "Values: damage_mod, defense_mod, speed_mod are integers (‑20..20). "
        "If type == 'utility', damage_mod must be 0. \n"
        f"Design a balanced ability for a creature named '{name}' in the '{tier}' tier."
    )
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    # Extract JSON from the reply (first code block or raw)
    content = response.choices[0].message.content
    try:
        json_start = content.find("{")
        json_data = json.loads(content[json_start:])
    except Exception as e:  # fallback parse
        logger.error("Failed to parse ability JSON: %s\nRaw content: %s", e, content)
        raise
    return Ability.from_dict(json_data)

###############################################################################
# Battle simulation
###############################################################################

def simulate_round(a: Creature, b: Creature) -> str:
    """Simulate a single round and return a battle log string."""
    log_lines: List[str] = []

    # Determine order each round by *current* SPD
    first, second = (a, b) if a.current_spd >= b.current_spd else (b, a)

    for actor, target in ((first, second), (second, first)):
        if target.current_hp <= 0:
            break  # Target already knocked out earlier in the round.

        ability = actor.ability
        if ability.type == "attack":
            dmg = max(1, ability.damage_mod + actor.stats["ATK"] - target.current_def)
            target.current_hp = max(0, target.current_hp - dmg)
            log_lines.append(f"{actor.name} hits {target.name} for {dmg} damage! ({target.current_hp} HP left)")
        else:  # utility ability
            # Handle defense buff/debuff
            if ability.defense_mod > 0:
                actor.current_def += ability.defense_mod
                log_lines.append(f"{actor.name} fortifies itself (+{ability.defense_mod} DEF)")
            elif ability.defense_mod < 0:
                target.current_def = max(0, target.current_def + ability.defense_mod)  # debuff
                log_lines.append(f"{actor.name} weakens {target.name} ({ability.defense_mod} DEF)")

            # Handle speed buff/debuff
            if ability.speed_mod != 0:
                if ability.speed_mod > 0:
                    actor.current_spd += ability.speed_mod
                else:
                    target.current_spd = max(1, target.current_spd + ability.speed_mod)
                log_lines.append(
                    f"{actor.name} {'boosts' if ability.speed_mod>0 else 'slows'} "
                    f"{'itself' if ability.speed_mod>0 else target.name} ({ability.speed_mod:+} SPD)"
                )

            # Healing (only if ability.weight > 0 and no defense debuff applied)
            if ability.damage_mod == 0 and ability.weight > 0:
                heal_amount = int(actor.stats["HP"] * ability.weight / 10)
                actor.current_hp = min(actor.stats["HP"], actor.current_hp + heal_amount)
                log_lines.append(f"{actor.name} heals {heal_amount} HP! ({actor.current_hp} / {actor.stats['HP']})")

        if target.current_hp <= 0:
            log_lines.append(f"{target.name} is knocked out!")
            break

    return "\n".join(log_lines)

###############################################################################
# Slash commands
###############################################################################

@bot.tree.command(description="Register as a Creature Battler trainer")
async def register(interaction: discord.Interaction):
    uid = interaction.user.id
    username = interaction.user.name
    async with bot.db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users(id, username) VALUES($1, $2) ON CONFLICT (id) DO NOTHING",
            uid,
            username,
        )
    await interaction.response.send_message("Registration complete! You can now /spawn creatures.")


@bot.tree.command(description="Spawn a new creature")
@app_commands.describe(name="Name of your new creature", tier="Tier (bronze/silver/gold)")
async def spawn(interaction: discord.Interaction, name: str, tier: app_commands.Choice[str]):  # type: ignore
    uid = interaction.user.id

    # Ensure user registered
    async with bot.db_pool.acquire() as conn:
        rec = await conn.fetchrow("SELECT 1 FROM users WHERE id=$1", uid)
    if not rec:
        await interaction.response.send_message("You need to /register first!", ephemeral=True)
        return

    await interaction.response.defer(thinking=True, ephemeral=True)

    # Allocate stats and generate ability
    stats = allocate_stats(tier.value)
    try:
        ability = await generate_creature_from_openai(name, tier.value)
    except Exception:
        await interaction.followup.send("AI generation failed. Please try again later.")
        return

    # Persist creature
    async with bot.db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO creatures(owner_id, name, tier, hp, atk, def, spd, ability_json)
            VALUES($1,$2,$3,$4,$5,$6,$7,$8)
            """,
            uid,
            name,
            tier.value,
            stats["HP"],
            stats["ATK"],
            stats["DEF"],
            stats["SPD"],
            json.dumps(ability.__dict__),
        )
    await interaction.followup.send(f"Spawned {name}! Ability: {ability.name} – {ability.description}")


@spawn.autocomplete("tier")
async def tier_autocomplete(_: discord.Interaction, current: str):
    options = [app_commands.Choice(name=t.title(), value=t) for t in ("bronze", "silver", "gold")]
    return [opt for opt in options if current.lower() in opt.name.lower()]  # simple filter


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    async with bot.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT name, tier FROM creatures WHERE owner_id=$1", uid)
    if not rows:
        await interaction.response.send_message("You have no creatures yet. Use /spawn!")
        return
    formatted = "\n".join(f"• {r['name']} ({r['tier']})" for r in rows)
    await interaction.response.send_message(f"Your creatures:\n{formatted}")


@bot.tree.command(description="Start a battle with one of your creatures")
@app_commands.describe(creature="Name of your creature", tier="Opponent tier")
async def battle(interaction: discord.Interaction, creature: str, tier: app_commands.Choice[str]):  # type: ignore
    uid = interaction.user.id

    if uid in bot.active_battles:
        await interaction.response.send_message("You already have an active battle! Finish it with /continue.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    # Fetch player's creature
    async with bot.db_pool.acquire() as conn:
        rec = await conn.fetchrow(
            "SELECT * FROM creatures WHERE owner_id=$1 AND name ILIKE $2", uid, creature
        )
    if not rec:
        await interaction.followup.send("Creature not found.")
        return
    player_creature = Creature.from_record(rec)

    # Generate opponent creature via OpenAI
    opponent_name = random.choice(["Gobblin", "Shadow Drake", "Crystal Lynx", "Thunder Imp"])
    try:
        opp_ability = await generate_creature_from_openai(opponent_name, tier.value)
    except Exception:
        await interaction.followup.send("AI generation failed. Please try again later.")
        return
    opp_stats = allocate_stats(tier.value)
    opponent_creature = Creature(
        owner_id=0,
        name=opponent_name,
        tier=tier.value,
        stats=opp_stats,
        ability=opp_ability,
    )

    bot.active_battles[uid] = (player_creature, opponent_creature)
    await interaction.followup.send(
        f"A wild {opponent_name} appears! Use /continue to play out the battle."
    )


@battle.autocomplete("tier")
async def tier_autocomplete_battle(_: discord.Interaction, current: str):
    return [app_commands.Choice(name=t.title(), value=t) for t in ("bronze", "silver", "gold") if current.lower() in t]


@bot.tree.command(description="Continue your active battle")
async def continue_battle(interaction: discord.Interaction):
    uid = interaction.user.id
    battle_pair = bot.active_battles.get(uid)
    if not battle_pair:
        await interaction.response.send_message("No active battle. Use /battle to start one!")
        return

    player, opponent = battle_pair
    battle_log = []
    # Simulate up to 5 rounds or until someone faints
    for _ in range(5):
        battle_log.append(simulate_round(player, opponent))
        if player.current_hp <= 0 or opponent.current_hp <= 0:
            break

    if player.current_hp <= 0 and opponent.current_hp <= 0:
        result = "It’s a draw! Both creatures are down."
    elif opponent.current_hp <= 0:
        result = f"{player.name} wins!"
    else:
        result = f"{opponent.name} wins!"

    await interaction.response.send_message("\n".join(battle_log + [result]))
    # Clear active battle regardless of result
    bot.active_battles.pop(uid, None)

###############################################################################
# Run the bot
###############################################################################

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
