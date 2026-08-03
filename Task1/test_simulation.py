"""
Test harness for simulation.py.

Runs full matches (regulation + penalties when needed), checks a set of
logical-correctness assertions after every tick, and writes a full report
(event timeline, penalty board, pass/fail results) to a .txt file.
"""

import random
import sys
from task13 import Player, Team, Match


def make_roster(prefix: str) -> list[Player]:
    """1 GK, 4 DEF, 4 MID, 2 FWD active = 11, plus 2 bench players.
    Attack is set high enough relative to the 1.3x defense multiplier
    in process_goal_attempt that goals actually happen during the test run."""
    roster = []
    roster.append(Player(f"{prefix}_GK1", 1, base_attack=20, base_defense=70))
    for i in range(4):
        roster.append(Player(f"{prefix}_DEF{i+1}", 2, base_attack=30, base_defense=60))
    for i in range(4):
        roster.append(Player(f"{prefix}_MID{i+1}", 3, base_attack=85, base_defense=40))
    for i in range(2):
        roster.append(Player(f"{prefix}_FWD{i+1}", 4, base_attack=95, base_defense=20))
    # bench
    roster.append(Player(f"{prefix}_BENCH1", 4, base_attack=88, base_defense=25))
    roster.append(Player(f"{prefix}_BENCH2", 3, base_attack=80, base_defense=45))
    return roster


def build_match() -> Match:
    home_roster = make_roster("HOME")
    # mirror stats exactly so the match is a near coin-flip and is likely
    # to land level after 90 minutes, forcing the penalty shootout path
    away_roster = make_roster("AWAY")
    for p, hp in zip(away_roster, home_roster):
        p.base_attack = hp.base_attack
        p.base_defense = hp.base_defense
    home_team = Team("HOME_FC", home_roster, home_roster[:11])
    away_team = Team("AWAY_FC", away_roster, away_roster[:11])
    return Match(home_team, away_team, 0, 0, 0, [], 1)


def run_checks(match: Match, log: list[str]) -> list[str]:
    """Assertion-style checks. Returns list of failure messages (empty = all pass)."""
    failures = []

    for team in (match.home_team, match.away_team):
        for player in team.roster:
            if not (10.0 <= player.stamina <= 100.0):
                failures.append(f"Stamina out of bounds: {player.name} = {player.stamina}")
        if team.substitutions_remaining < 0:
            failures.append(f"{team.country_name} substitutions_remaining went negative")
        if len(team.active_lineup) > 11:
            failures.append(f"{team.country_name} active_lineup exceeds 11 players")
        # no duplicate players in active lineup
        if len(set(team.active_lineup)) != len(team.active_lineup):
            failures.append(f"{team.country_name} active_lineup has duplicate players")

    if match.home_score < 0 or match.away_score < 0:
        failures.append("Negative score detected")

    # event ids must be sequential strings "1","2","3"...
    for idx, event in enumerate(match.timeline, start=1):
        if event.event_id != str(idx):
            failures.append(f"Event id out of sequence at position {idx}: got {event.event_id}")
            break

    # minutes in timeline should be non-decreasing
    minutes = [e.minute for e in match.timeline]
    if minutes != sorted(minutes):
        failures.append("Timeline minutes are not in non-decreasing order")

    if match.phase == 3:
        failures.append("Match ended tick loop still in PENALTIES phase (should resolve to FINISHED)")

    if match.phase == 2 and match.penalty_kicks:
        if match.penalty_home_score == match.penalty_away_score:
            failures.append("Penalty shootout ended level - no winner determined")

    return failures


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    random.seed(seed)
    match = build_match()
    log = []
    all_failures = []

    log.append("=== MATCH SIMULATION LOG ===")
    log.append(f"{match.home_team.country_name} vs {match.away_team.country_name}\n")

    tick_count = 0
    incident_test_done = False
    while match.phase == 1:
        match.run_minute_tick()
        tick_count += 1

        # exercise the discipline system once, mid-match: two incidents -> send-off
        if not incident_test_done and match.current_minute == 20:
            target = match.away_team.active_lineup[0]
            match.record_incident(match.away_team, target)  # incident #1, no send-off yet
            match.record_incident(match.away_team, target)  # incident #2, should trigger send-off
            if target in match.away_team.active_lineup:
                all_failures.append("Player was not removed from active_lineup after 2nd incident")
            if not target.sent_off:
                all_failures.append("Player.sent_off flag not set after 2nd incident")
            incident_test_done = True

        failures = run_checks(match, log)
        if failures:
            all_failures.extend([f"[minute {match.current_minute}] {f}" for f in failures])
        if tick_count > 200:  # safety valve against infinite loops
            all_failures.append("run_minute_tick did not terminate within 200 ticks")
            break

    log.append(f"Final score after regulation/penalties: "
               f"{match.home_team.country_name} {match.home_score} - "
               f"{match.away_score} {match.away_team.country_name}")
    log.append(f"Final phase: {Match.phases[match.phase]}\n")

    log.append("--- EVENT TIMELINE ---")
    for event in match.timeline:
        log.append(event.to_string())

    log.append("\n--- PENALTY BOARD ---")
    log.append(match.render_penalty_board())

    log.append("\n--- SUBSTITUTIONS REMAINING ---")
    log.append(f"{match.home_team.country_name}: {match.home_team.substitutions_remaining}")
    log.append(f"{match.away_team.country_name}: {match.away_team.substitutions_remaining}")

    log.append("\n--- CORRECTNESS CHECKS ---")
    if not all_failures:
        log.append(f"ALL CHECKS PASSED ({tick_count} ticks simulated)")
    else:
        log.append(f"{len(all_failures)} CHECK(S) FAILED:")
        for f in all_failures:
            log.append(f"  - {f}")

    # --- dedicated shootout test: don't rely on regulation ending level by chance ---
    log.append("\n\n=== DEDICATED PENALTY SHOOTOUT TEST ===")
    shootout_match = build_match()
    shootout_match.phase = 3
    shootout_match.current_minute = 90
    shootout_match.run_penalty_shootout()

    shootout_failures = []
    if shootout_match.phase != 2:
        shootout_failures.append("Match did not resolve to FINISHED after shootout")
    if shootout_match.penalty_home_score == shootout_match.penalty_away_score:
        shootout_failures.append("Shootout ended level - no winner")
    if len(shootout_match.penalty_kicks) < 10:
        shootout_failures.append("Fewer than 10 kicks recorded in a shootout")
    for kick in shootout_match.penalty_kicks:
        if kick["result"] not in ("SCORED", "MISSED"):
            shootout_failures.append(f"Invalid kick result: {kick['result']}")

    log.append(shootout_match.render_penalty_board())
    log.append("")
    if shootout_failures:
        log.append(f"{len(shootout_failures)} SHOOTOUT CHECK(S) FAILED:")
        for f in shootout_failures:
            log.append(f"  - {f}")
        all_failures.extend(shootout_failures)
    else:
        log.append("ALL SHOOTOUT CHECKS PASSED")

    report = "\n".join(log)
    print(report)

    print("\n\nOVERALL RESULT:", "PASS" if not all_failures else f"FAIL ({len(all_failures)} issue(s))")


def run_ai_coached_match():
    """Runs a full match with a MockCoachModel driving the home team, and
    verifies formation changes/substitutions never break lineup invariants."""
    from task13 import MatchAI, MockCoachModel

    log = []
    failures = []

    match = build_match()
    ai = MatchAI(model=MockCoachModel(), controlled_team=match.home_team, risk_tolerance=0.5)
    match.home_ai = ai

    log.append("=== AI-COACHED MATCH TEST ===")
    tick_count = 0
    while match.phase == 1:
        match.run_minute_tick()
        tick_count += 1

        if len(match.home_team.active_lineup) != 11:
            failures.append(f"[minute {match.current_minute}] home active_lineup size drifted to "
                             f"{len(match.home_team.active_lineup)}")
        if len(set(match.home_team.active_lineup)) != len(match.home_team.active_lineup):
            failures.append(f"[minute {match.current_minute}] duplicate players in home active_lineup after AI action")
        if not (0.0 <= ai.risk_tolerance <= 1.0):
            failures.append(f"[minute {match.current_minute}] risk_tolerance out of bounds: {ai.risk_tolerance}")

        if tick_count > 200:
            failures.append("AI-coached match did not terminate within 200 ticks")
            break

    log.append(f"Final score: HOME {match.home_score} - {match.away_score} AWAY, phase={Match.phases[match.phase]}")
    log.append(f"Final formation: {ai.current_formation}, final risk_tolerance: {ai.risk_tolerance:.2f}")
    log.append(f"Total AI decisions logged: {len(ai.decision_log)}\n")

    log.append("--- SAMPLE OF DECISION LOG (first 10) ---")
    for entry in ai.decision_log[:10]:
        log.append(f"minute {entry['minute']:>3} | {entry['action']:<16} | {entry['reasoning']}")

    action_counts = {}
    for entry in ai.decision_log:
        action_counts[entry["action"]] = action_counts.get(entry["action"], 0) + 1
    log.append(f"\nAction frequency across match: {action_counts}")

    log.append("\n--- AI TEST RESULT ---")
    if failures:
        log.append(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            log.append(f"  - {f}")
    else:
        log.append(f"ALL AI CHECKS PASSED ({tick_count} ticks, {len(ai.decision_log)} decisions)")

    report = "\n".join(log)
    print("\n\n" + report)



    return not failures


if __name__ == "__main__":
    main()
    run_ai_coached_match()
