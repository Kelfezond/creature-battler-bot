from __future__ import annotations
import asyncio, json, logging, math, os, random, time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import discord
from discord.ext import commands, tasks
import openai

# ─── Basic config & logging ──────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN     = os.getenv("DISCORD_TOKEN")
DB_URL    = os.getenv("DATABASE_URL")
GUILD_ID  = os.getenv("GUILD_ID") or None
openai.api_key = os.getenv("OPENAI_API_KEY")

LEADERBOARD_CHANNEL_ID_ENV = os.getenv("LEADERBOARD_CHANNEL_ID")

# Admin allow-list
def _parse_admin_ids(raw: Optional[str]) -> set[int]:
    if not raw: return set()
    ids: set[int] = set()
    for p in raw.split(","):
        p = p.strip()
        try: ids.add(int(p))
        except ValueError: logger.warning("Bad ADMIN_USER_IDS entry %r", p)
    return ids
ADMIN_USER_IDS: set[int] = _parse_admin_ids(os.getenv("ADMIN_USER_IDS"))

for k, v in {"DISCORD_TOKEN":TOKEN,"DATABASE_URL":DB_URL,"OPENAI_API_KEY":openai.api_key}.items():
    if not v: raise RuntimeError(f"Missing env var {k}")

# ─── Discord client ──────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ─── Database helpers and schema  (UNCHANGED except at end) ─
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trainers (
  user_id BIGINT PRIMARY KEY,
  joined_at TIMESTAMPTZ DEFAULT now(),
  cash BIGINT DEFAULT 0,
  trainer_points BIGINT DEFAULT 0,
  facility_level INT DEFAULT 1
);

CREATE TABLE IF NOT EXISTS creatures (
  id SERIAL PRIMARY KEY,
  owner_id BIGINT NOT NULL,
  name TEXT,
  rarity TEXT,
  descriptors TEXT[],
  stats JSONB,
  created_at TIMESTAMPTZ DEFAULT now(),
  current_hp BIGINT
);

CREATE TABLE IF NOT EXISTS battle_caps (
  creature_id INT NOT NULL REFERENCES creatures(id) ON DELETE CASCADE,
  day DATE NOT NULL,
  count INT NOT NULL DEFAULT 0,
  PRIMARY KEY (creature_id, day)
);

CREATE TABLE IF NOT EXISTS creature_progress (
  creature_id INT NOT NULL REFERENCES creatures(id) ON DELETE CASCADE,
  tier INT NOT NULL,
  wins INT NOT NULL DEFAULT 0,
  glyph_unlocked BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (creature_id, tier)
);

CREATE TABLE IF NOT EXISTS creature_records (
  creature_id INT,
  owner_id BIGINT NOT NULL,
  name TEXT NOT NULL,
  wins INT NOT NULL DEFAULT 0,
  losses INT NOT NULL DEFAULT 0,
  is_dead BOOLEAN NOT NULL DEFAULT FALSE,
  died_at TIMESTAMPTZ,
  PRIMARY KEY (owner_id, name)
);

CREATE INDEX IF NOT EXISTS cr_rank_idx ON creature_records (wins DESC, losses ASC, name ASC);

CREATE TABLE IF NOT EXISTS leaderboard_messages (
  channel_id BIGINT PRIMARY KEY,
  message_id BIGINT
);
"""

async def db_pool() -> asyncpg.Pool:
    if not hasattr(bot, "_pool"):
        bot._pool = await asyncpg.create_pool(DB_URL)
    return bot._pool

# ─── Game constants ──────────────────────────────────────────
MAX_CREATURES      = 5
DAILY_BATTLE_CAP   = 2
PRIMARY_STATS      = ["HP", "AR", "PATK", "SATK", "SPD"]

SELL_PRICES = {"Common":1_000,"Uncommon":2_000,"Rare":10_000,"Epic":20_000,"Legendary":50_000}

def spawn_rarity() -> str:
    r = random.random()*100
    return ("Common"      if r<75  else
            "Uncommon"    if r<88  else
            "Rare"        if r<95  else
            "Epic"        if r<99.5 else
            "Legendary")

# (tier rarity weights, facility levels, point pools, payouts – unchanged)

# ─── Helper: apply a random +1 stat on win ───────────────────
async def _grant_random_stat(creature_id: int) -> str:
    """
    Increments one random primary stat by +1.
    Returns the stat name that was increased.
    """
    pool = await db_pool()
    row = await pool.fetchrow("SELECT stats,current_hp FROM creatures WHERE id=$1", creature_id)
    if not row: return "?"
    stats = json.loads(row["stats"])
    stat = random.choice(PRIMARY_STATS)
    stats[stat] += 1
    new_cur_hp = row["current_hp"]
    if stat == "HP":
        new_cur_hp = min(new_cur_hp + 5, stats["HP"]*5)
    await pool.execute(
        "UPDATE creatures SET stats=$1,current_hp=$2 WHERE id=$3",
        json.dumps(stats), new_cur_hp, creature_id
    )
    return stat

# ─── Leaderboard formatting tweaks (Glyph column) ────────────
NAME_W, TRAINER_W, GLYPH_W = 20, 14, 5  # widths for alignment

def _fmt_leaderboard(rows: List[Tuple[str,int,int,bool,str,int]]) -> str:
    """
    rows: (name,w,l,is_dead,trainer,max_glyph)
    """
    head = f"{'#':>3}. {'Name':<{NAME_W}} {'Trainer':<{TRAINER_W}} {'W':>4} {'L':>4} {'Glyph':>{GLYPH_W}} Status"
    out = ["  "+head]
    for i,(n,w,l,d,t,g) in enumerate(rows,1):
        gstr = ("–" if g==0 else str(g))
        line = f"{i:>3}. {n:<{NAME_W}} {t:<{TRAINER_W}} {w:>4} {l:>4} {gstr:>{GLYPH_W}} {'💀 DEAD' if d else ''}"
        out.append((" -"+line if d else "  "+line))
    return "```diff\n"+"\n".join(out)+"\n```"

async def update_leaderboard_now(reason:str="trigger"):
    chan_id = await _get_leaderboard_channel_id()
    if not chan_id: return
    msg   = await _get_or_create_leaderboard_message(chan_id)
    if not msg: return
    pool  = await db_pool()
    rows  = await pool.fetch("""
        SELECT r.name,r.wins,r.losses,r.is_dead,r.owner_id,
               COALESCE(MAX(CASE WHEN cp.glyph_unlocked THEN cp.tier END),0) AS glyph
        FROM creature_records r
        LEFT JOIN creature_progress cp ON cp.creature_id = r.creature_id
        GROUP BY r.name,r.wins,r.losses,r.is_dead,r.owner_id
        ORDER BY r.wins DESC,r.losses ASC,r.name
        LIMIT 20
    """)
    trainers = await asyncio.gather(*[_resolve_trainer_name(r["owner_id"]) for r in rows])
    packed   = [(r["name"][:NAME_W],r["wins"],r["losses"],r["is_dead"],trainers[i][:TRAINER_W],r["glyph"]) for i,r in enumerate(rows)]
    content  = ("**Creature Leaderboard — Top 20**\nUpdated: <t:%d:R>\n" % int(time.time())) + _fmt_leaderboard(packed)
    try: await msg.edit(content=content)
    except Exception as e: logger.error("LB edit fail: %s",e)

# ─── Battle finalisation changed to grant stat ───────────────
async def finalize_battle(inter: discord.Interaction, st):
    player_won = st.opp_current_hp<=0<st.user_current_hp
    win_cash,loss_cash = TIER_PAYOUTS[st.tier]
    payout = win_cash if player_won else loss_cash
    pool   = await db_pool()
    await pool.execute("UPDATE trainers SET cash=cash+$1 WHERE user_id=$2",payout,st.user_id)

    # ensure record row exists
    await _ensure_record(st.user_id, st.creature_id, st.user_creature["name"])
    await _record_result(st.user_id, st.user_creature["name"], player_won)

    if player_won:
        wins, unlocked = await _record_win_and_maybe_unlock(st.creature_id, st.tier)
        gained = await _grant_random_stat(st.creature_id)
        st.logs.append(f"🏆 Victory! {st.user_creature['name']} gained **+1 {gained}**.")
        st.logs.append(f"Progress Tier {st.tier}: {wins}/5 wins.")
        if unlocked:
            st.logs.append(f"🏅 Glyph {st.tier} unlocked! Tier {st.tier+1 if st.tier<9 else 9} available.")
    else:
        death_roll = random.random()
        if death_roll<0.5:
            await _record_death(st.user_id, st.user_creature["name"])
            await pool.execute("DELETE FROM creatures WHERE id=$1", st.creature_id)
            st.logs.append(f"💀 {st.user_creature['name']} died.")
        else:
            st.logs.append("🛡️ Your creature survived the loss.")

    st.logs.append(f"Cash awarded: {payout}.")
    asyncio.create_task(update_leaderboard_now("battle"))

# ─── /creatures shows battles left (unchanged from previous) ─
# (… your existing /creatures command remains the same …)

# ─── Everything else (commands, setup_hook, loops) UNCHANGED ─
# (… keep all previous code, ensuring imports at top include any helpers …)

if __name__ == "__main__":
    bot.run(TOKEN)
