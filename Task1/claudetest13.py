"""
Test harness for the merged simulation.py. Picks a fresh random matchup
(different teams, different tiers) every run, rebuilds all Player/Team
objects from scratch each time (no shared global state between runs),
runs a full match, and writes a report + correctness checks to a txt file.

Run: python3 test_simulation.py [seed]
"""

import random
import sys
from task13 import Player, Team, Match, MatchAI, MockCoachModel, Position, Phase, EventType


# Pool of (country, tier) - tier scales base_attack/base_defense so matchups
# aren't all evenly matched. A fresh call to build_team() always returns
# brand-new Player/Team objects, so re-running never leaks stamina/incidents/
# substitutions_remaining state from a previous simulation.
TEAM_POOL = [
    ("ARGENTINA", 1.15), ("FRANCE", 1.15), ("BRAZIL", 1.12), ("ENGLAND", 1.10),
    ("SPAIN", 1.08), ("GERMANY", 1.05), ("PORTUGAL", 1.05), ("NETHERLANDS", 1.0),
    ("MOROCCO", 0.95), ("JAPAN", 0.92), ("MEXICO", 0.9), ("USA", 0.88),
    ("SAUDI_ARABIA", 0.8), ("POLAND", 0.85),
]


def build_team(country: str, tier: float) -> Team:
    """Fresh Player/Team objects every call. 11 active + 15 bench = 26 total,
    matching the spec's own worked example ('roster minus active_lineup = 15
    players')."""
    def stat(base):
        return max(1, min(100, round(base * tier)))

    roster = []
    roster.append(Player(f"{country}_GK1", Position.GOALKEEPER, stat(20), stat(72)))
    for i in range(4):
        roster.append(Player(f"{country}_DEF{i+1}", Position.DEFENDER, stat(32), stat(62)))
    for i in range(4):
        roster.append(Player(f"{country}_MID{i+1}", Position.MIDFIELDER, stat(70), stat(48)))
    for i in range(2):
        roster.append(Player(f"{country}_FWD{i+1}", Position.FORWARD, stat(85), stat(25)))

    # bench: 15 players spanning all positions
    roster.append(Player(f"{country}_GK2", Position.GOALKEEPER, stat(18), stat(65)))
    for i in range(5):
        roster.append(Player(f"{country}_DEF_B{i+1}", Position.DEFENDER, stat(30), stat(58)))
    for i in range(5):
        roster.append(Player(f"{country}_MID_B{i+1}", Position.MIDFIELDER, stat(62), stat(45)))
    for i in range(4):
        roster.append(Player(f"{country}_FWD_B{i+1}", Position.FORWARD, stat(78), stat(22)))

    return Team(country, roster, roster[:11])


def build_random_match(with_ai=True) -> Match:
    """Picks two DIFFERENT teams from the pool at random every call."""
    (home_country, home_tier), (away_country, away_tier) = random.sample(TEAM_POOL, 2)
    home_team = build_team(home_country, home_tier)
    away_team = build_team(away_country, away_tier)

    home_ai = MatchAI(MockCoachModel(), home_team, risk_tolerance=0.5) if with_ai else None
    away_ai = MatchAI(MockCoachModel(), away_team, risk_tolerance=0.5) if with_ai else None

    return Match(home_team, away_team, 0, 0, 0, [], Phase.REGULATION, home_ai, away_ai)


def run_checks(match: Match) -> list[str]:
    failures = []
    for team in (match.home_team, match.away_team):
        for player in team.roster:
            if not (10.0 <= player.stamina <= 100.0):
                failures.append(f"Stamina out of bounds: {player.name} = {player.stamina}")
        if team.substitutions_remaining < 0:
            failures.append(f"{team.country_name} substitutions_remaining went negative")
        if len(set(team.active_lineup)) != len(team.active_lineup):
            failures.append(f"{team.country_name} active_lineup has duplicate players")

    if match.home_score < 0 or match.away_score < 0:
        failures.append("Negative score detected")

    for idx, event in enumerate(match.timeline, start=1):
        if event.event_id != str(idx):
            failures.append(f"Event id out of sequence at position {idx}")
            break

    minutes = [e.minute for e in match.timeline]
    if minutes != sorted(minutes):
        failures.append("Timeline minutes not non-decreasing")

    # verify no red-card / half-time collision: every RED_CARD-labeled event
    # must say "SENT OFF" and no HALF_TIME event should ever mention it
    for event in match.timeline:
        label = event.event_types[event.event_type]
        if label == "HALF_TIME" and "SENT OFF" in event.outcome_text:
            failures.append("HALF_TIME/RED_CARD collision still present!")
        if label == "RED_CARD" and "SENT OFF" not in event.outcome_text:
            failures.append("RED_CARD event missing expected text")

    return failures


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(1, 1_000_000)
    random.seed(seed)

    match = build_random_match(with_ai=True)
    log = []
    all_failures = []

    log.append(f"=== RANDOM MATCHUP (seed={seed}) ===")
    log.append(f"{match.home_team.country_name} (home) vs {match.away_team.country_name} (away)\n")

    tick_count = 0
    while match.phase == Phase.REGULATION:
        match.run_minute_tick()
        tick_count += 1
        all_failures.extend(run_checks(match))
        if tick_count > 200:
            all_failures.append("run_minute_tick did not terminate within 200 ticks")
            break

    log.append(f"Final score: {match.home_team.country_name} {match.home_score} - "
               f"{match.away_score} {match.away_team.country_name}")
    log.append(f"Final phase: {Match.phases[match.phase]}\n")

    log.append("--- EVENT TIMELINE (last 20) ---")
    for event in match.timeline[-20:]:
        log.append(event.to_string())

    log.append("\n--- PENALTY BOARD ---")
    log.append(match.render_penalty_board())

    log.append("\n--- AI DECISION SAMPLE (home team, first 8) ---")
    if match.home_ai:
        for entry in match.home_ai.decision_log[:8]:
            log.append(f"minute {entry['minute']:>3} | {entry['action']:<16} | {entry['reasoning']}")

    log.append("\n--- CORRECTNESS CHECKS ---")
    if not all_failures:
        log.append(f"ALL CHECKS PASSED ({tick_count} ticks simulated)")
    else:
        log.append(f"{len(all_failures)} CHECK(S) FAILED:")
        for f in set(all_failures):
            log.append(f"  - {f}")

    report = "\n".join(log)
    print(report)

    # with open("/mnt/user-data/outputs/simulation_log.txt", "w") as f:
    #     f.write(report)

    print("\nOVERALL RESULT:", "PASS" if not all_failures else f"FAIL ({len(all_failures)} issue(s))")
    print(f"(re-run with a different/no seed to get a different matchup - this run used seed={seed})")


if __name__ == "__main__":
    main()
