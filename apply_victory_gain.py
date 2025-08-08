#!/usr/bin/env python3
"""
apply_victory_gain.py

Adds a '+1 random stat on win (only if survives)' block inside finalize_battle()
in creature_battler_bot.py. Creates a .bak backup and is idempotent (safe to rerun).
"""

from pathlib import Path
import sys

TARGET = Path("creature_battler_bot.py")
MARKER = "Victory stat gain (+1 random stat)"

CODE_BLOCK = r"""
        # ── NEW: Victory stat gain (+1 random stat) ─────────────────────────────
        # Only on win, and only for the surviving creature (wins never roll death here).
        try:
            # Choose a stat and apply +1 in memory first
            gained_stat = random.choice(PRIMARY_STATS)
            new_stats = dict(st.user_creature["stats"])  # shallow copy
            new_stats[gained_stat] = int(new_stats.get(gained_stat, 0)) + 1

            # Compute new HP values if HP increased
            new_max_hp = int(new_stats["HP"]) * 5
            # st.user_current_hp at this point is the final HP after battle rounds
            new_cur_hp = st.user_current_hp
            if gained_stat == "HP":
                # On HP gain, increase current HP by +5 but cap at new max
                new_cur_hp = min(st.user_current_hp + 5, new_max_hp)

            # Persist to DB
            await pool.execute(
                "UPDATE creatures SET stats=$1, current_hp=$2 WHERE id=$3",
                json.dumps(new_stats), new_cur_hp, st.creature_id
            )

            # Update in-memory snapshot so future logs/logic reflect the change
            st.user_creature["stats"] = new_stats
            st.user_max_hp = new_max_hp
            st.user_current_hp = new_cur_hp

            # Friendly log message
            if gained_stat == "HP":
                st.logs.append(
                    f"✨ **{st.user_creature['name']}** gained **+1 HP** from the victory "
                    f"(Max HP is now {new_max_hp}, current {new_cur_hp}/{new_max_hp})."
                )
            else:
                st.logs.append(
                    f"✨ **{st.user_creature['name']}** gained **+1 {gained_stat}** from the victory!"
                )
        except Exception as e:
            logger.error("Failed to apply victory stat gain: %s", e)
        # ── End NEW ─────────────────────────────────────────────────────────────
"""

def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found. Run this from your repo root.", file=sys.stderr)
        return 1

    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("Already applied. No changes made.")
        return 0

    # Find the finalize_battle function
    def_index = src.find("async def finalize_battle")
    if def_index == -1:
        print("ERROR: Could not find 'async def finalize_battle'.", file=sys.stderr)
        return 1

    # Find the 'if player_won:' line inside finalize_battle
    player_won_index = src.find("\n    if player_won:", def_index)
    if player_won_index == -1:
        print("ERROR: Could not find 'if player_won:' inside finalize_battle.", file=sys.stderr)
        return 1

    # Find the 'if not player_won:' line at the same indentation (end of the win block)
    not_player_won_index = src.find("\n    if not player_won:", player_won_index)
    if not_player_won_index == -1:
        print("ERROR: Could not find 'if not player_won:' after the win block.", file=sys.stderr)
        return 1

    # We will insert right before 'if not player_won:'
    insert_at = not_player_won_index

    # Prepare correctly indented block: the win block inner code is 8 spaces in your file
    # We ensure indentation is 8 spaces to align with surrounding code.
    block = CODE_BLOCK
    # Safety: ensure block starts with a newline and 8 spaces
    if not block.startswith("\n"):
        block = "\n" + block

    new_src = src[:insert_at] + block + src[insert_at:]

    # Backup
    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(src, encoding="utf-8")
    TARGET.write_text(new_src, encoding="utf-8")

    print(f"Patched successfully. Backup saved to {backup.name}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
