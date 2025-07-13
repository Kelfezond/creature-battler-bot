from __future__ import annotations

"""Creature Battler Discord bot (v3 – no abilities)
----------------------------------------------------
Battle system simplified:
• No ability system; database column removed.
• Battle HP = HP * 5.
• Turn order by SPD; if a creature’s SPD ≥ 2× opponent’s, it attacks twice.
• Damage each attack = d(ATK) – DEF, floored to 1.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import asyncpg
import discord
import dotenv
from discord import app_commands
from discord.ext import commands

###############################################################################
# Configuration & startup helpers
###############################################################################

dotenv.load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
GUILD_ID = os.getenv("GUILD_ID")  # optional for faster dev sync

missing = [name for val, name in ((DISCORD_TOKEN, "DISCORD_TOKEN"), (DATABASE_URL, "DATABASE_URL")) if val is None]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creature_battler")

###############################################################################
# Data classes & helpers
###############################################################################

STAT_NAMES = ("HP", "ATK", "DEF", "SPD")

@dataclass
class Creature:
    owner_id: int
    name: str
    tier: str
    stats: Dict[str, int]
    current_hp: int = field(init=False)
    current_def: int = field(init=False)
    current_spd: int = field(init=False)

    def __post_init__(self):
        self.current_hp = self.stats["HP"] * 5  # battle HP
        self.current_def = self.stats["DEF"]
        self.current_spd = self.stats["SPD"]

    @classmethod
    def from_record(cls, rec: asyncpg.Record) -> "Creature":
        stats = {"HP": rec["hp"], "ATK": rec["atk"], "DEF": rec["def"], "SPD": rec["spd"]}
        return cls(rec["owner_id"], rec["name"], rec["tier"], stats)

###############################################################################
# Discord bot setup
###############################################################################

class CreatureBattlerBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.db_pool: asyncpg.Pool | None = None
        self.active_battles: Dict[int, Tuple[Creature, Creature]] = {}

    async def setup_hook(self):
        self.db_pool = await asyncpg.create_pool(DATABASE_URL)
        await self._ensure_schema()

        if GUILD_ID:
            guild = discord.Object(id=int(GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("Commands synced to guild %s", GUILD_ID)
        else:
            await self.tree.sync()
            logger.info("Commands synced globally (may take up to an hour)")

        logger.info("Creature Battler Bot ready!")

    async def close(self):
        if self.db_pool:
            await self.db_pool.close()
        await super().close()

    async def _ensure_schema(self):
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
            spd INT NOT NULL
        );
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(ddl)

bot = CreatureBattlerBot()

###############################################################################
# Utility functions
###############################################################################

def allocate_stats(tier: str) -> Dict[str, int]:
    base = {"bronze": 40, "silver": 55, "gold": 70}.get(tier, 40)
    hp = random.randint(10, 20)
    atk = random.randint(5, 15)
    defense = random.randint(5, 15)
    spd = random.randint(5, 15)
    remaining = base - (hp + atk + defense + spd)
    for _ in range(remaining):
        random.choice(["HP", "ATK", "DEF", "SPD"])
        if _ == 0:  # just to suppress lint, nothing special
            pass
    # Simple redistribution for remaining points
    while remaining > 0:
        choice = random.choice(["HP", "ATK", "DEF", "SPD"])
        if choice == "HP":
            hp += 1
        elif choice == "ATK":
            atk += 1
        elif choice == "DEF":
            defense += 1
        else:
            spd += 1
        remaining -= 1
    return {"HP": hp, "ATK": atk, "DEF": defense, "SPD": spd}

###############################################################################
# Battle simulation
###############################################################################

def _single_attack(attacker: Creature, defender: Creature) -> str:
    raw_roll = random.randint(1, attacker.stats["ATK"])
    damage = max(1, raw_roll - defender.current_def)
    defender.current_hp = max(0, defender.current_hp - damage)
    return f"{attacker.name} rolls {raw_roll} ATK ⇒ {defender.name} takes {damage} (HP {defender.current_hp})"

def simulate_round(a: Creature, b: Creature) -> str:
    log: List[str] = []
    first, second = (a, b) if a.current_spd >= b.current_spd else (b, a)

    for attacker, defender in ((first, second), (second, first)):
        if defender.current_hp <= 0:
            break
        attacks = 2 if attacker.current_spd >= defender.current_spd * 2 else 1
        for _ in range(attacks):
            if defender.current_hp <= 0:
                break
            log.append(_single_attack(attacker, defender))
        if defender.current_hp <= 0:
            log.append(f"{defender.name} is knocked out!")
            break
    return "\n".join(log)

###############################################################################
# Slash commands
###############################################################################

@bot.tree.command(description="Register as a Creature Battler trainer")
async def register(interaction: discord.Interaction):
    uid, username = interaction.user.id, interaction.user.name
    async with bot.db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO users (id, username) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
            uid,
            username,
        )
    await interaction.response.send_message("Registration complete! Use /spawn to create a creature.")


@bot.tree.command(description="Spawn a new creature")
@app_commands.describe(name="Creature name", tier="Tier (bronze/silver/gold)")
async def spawn(interaction: discord.Interaction, name: str, tier: str):
    uid = interaction.user.id
    tier_value = tier.lower()

    # Verify user registered
    async with bot.db_pool.acquire() as conn:
        if not await conn.fetchrow("SELECT 1 FROM users WHERE id=$1", uid):
            await interaction.response.send_message("Please /register first!", ephemeral=True)
            return

    await interaction.response.defer(thinking=True, ephemeral=True)
    stats = allocate_stats(tier_value)

    async with bot.db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO creatures (owner_id, name, tier, hp, atk, def, spd)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            uid, name, tier_value, stats["HP"], stats["ATK"], stats["DEF"], stats["SPD"],
        )

    await interaction.followup.send(
        f"Spawned {name}! (HP:{stats['HP']} ATK:{stats['ATK']} DEF:{stats['DEF']} SPD:{stats['SPD']})"
    )


@spawn.autocomplete("tier")
async def tier_autocomplete(_: discord.Interaction, current: str):
    return [app_commands.Choice(name=t.title(), value=t) for t in ("bronze", "silver", "gold") if current.lower() in t]


@bot.tree.command(description="List your creatures")
async def creatures(interaction: discord.Interaction):
    uid = interaction.user.id
    async with bot.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, tier, hp, atk, def, spd FROM creatures WHERE owner_id=$1",
            uid,
        )
    if not rows:
        await interaction.response.send_message("You have no creatures. Use /spawn!")
        return
    lines = [
        f"• {r['name']} ({r['tier']}) HP:{r['hp']} ATK:{r['atk']} DEF:{r['def']} SPD:{r['spd']}"
        for r in rows
    ]
    await interaction.response.send_message("Your creatures:\n" + "\n".join(lines))


@bot.tree.command(description="Challenge a wild opponent")
@app_commands.describe(creature="Your creature's name", tier="Opponent tier (bronze/silver/gold)")
async def battle(interaction: discord.Interaction, creature: str, tier: str):
    uid = interaction.user.id
    tier_value = tier.lower()

    if uid in bot.active_battles:
        await interaction.response.send_message("Finish your current battle with /continue first.", ephemeral=True)
        return

    await interaction.response.defer(thinking=True)

    # Fetch player's creature
    async with bot.db_pool.acquire() as conn:
        rec = await conn.fetchrow(
            "SELECT * FROM creatures WHERE owner_id=$1 AND name ILIKE $2",
            uid,
            creature,
        )
    if not rec:
        await interaction.followup.send("Creature not found.")
        return
    player_creature = Creature.from_record(rec)

    # Generate opponent creature
    opponent_name = random.choice(["Gobblin", "Shadow Drake", "Crystal Lynx", "Thunder Imp"])
    opp_stats = allocate_stats(tier_value)
    opponent_creature = Creature(owner_id=0, name=opponent_name, tier=tier_value, stats=opp_stats)

    bot.active_battles[uid] = (player_creature, opponent_creature)
    await interaction.followup.send(
        f"A wild {opponent_name} appears! (HP:{opp_stats['HP']} ATK:{opp_stats['ATK']} DEF:{opp_stats['DEF']} SPD:{opp_stats['SPD']})
Use /continue to fight."
    )


@battle.autocomplete("tier")
async def tier_autocomplete_battle(_: discord.Interaction, current: str):
    return [app_commands.Choice(name=t.title(), value=t) for t in ("bronze", "silver", "gold") if current.lower() in t]


@bot.tree.command(name="continue", description="Continue your active battle")
async def continue_battle(interaction: discord.Interaction):
    uid = interaction.user.id
    battle_pair = bot.active_battles.get(uid)
    if not battle_pair:
        await interaction.response.send_message("No active battle. Use /battle to start one.")
        return

    player, opponent = battle_pair
    battle_log: List[str] = []

    # Up to five rounds or until knockout
    for _ in range(5):
        battle_log.append(simulate_round(player, opponent))
        if player.current_hp <= 0 or opponent.current_hp <= 0:
            break

    if player.current_hp <= 0 and opponent.current_hp <= 0:
        result = "It's a draw! Both creatures are down."
    elif opponent.current_hp <= 0:
        result = f"{player.name} wins!"
    else:
        result = f"{opponent.name} wins!"

    battle_log.append(result)
    await interaction.response.send_message("
".join(battle_log))

    bot.active_battles.pop(uid, None)

###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
