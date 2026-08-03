"""
Task 1.3 - FIFA World Cup Simulation System
Merged version: base is the user's sim.py; incorporates the parts of the
Gemini-generated version that were actually correct, fixes bugs found in
both, and wires up every bonus item from the spec PDF. Every change from
the original sim.py is flagged with a "# CHANGED:" comment explaining why.
"""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


# CHANGED: adopted from the Gemini version. Spec says `position: Enum -
# Restricted to [FORWARD, MIDFIELDER, DEFENDER, GOALKEEPER]`. sim.py used a
# plain int plus a lookup dict, which silently accepts any integer (e.g.
# Player("x", 99, ...) would not raise). IntEnum still compares/equals like
# a plain int everywhere the rest of the code expects one (position in (3,4)
# still works), but Position(99) now raises ValueError - the class actually
# enforces the "Restricted to" requirement instead of just documenting it.
class Position(IntEnum):
    GOALKEEPER = 1
    DEFENDER = 2
    MIDFIELDER = 3
    FORWARD = 4


# CHANGED: same reasoning - EventType restricted to [GOAL, SUBSTITUTION,
# HALF_TIME, FULL_TIME] per the base spec, extended with PENALTY_KICK and
# RED_CARD for the bonus features. Crucially, RED_CARD gets its OWN number
# (6) - in sim.py, record_incident() reused event_type=3, which collided
# with HALF_TIME (3 was already "HALF_TIME" in the event_types dict), so a
# sent-off player's event printed as "HALF_TIME" in the timeline. This is
# the single most important bug fix in this merge.
class EventType(IntEnum):
    GOAL = 1
    SUBSTITUTION = 2
    HALF_TIME = 3
    FULL_TIME = 4
    PENALTY_KICK = 5
    RED_CARD = 6


class Phase(IntEnum):
    REGULATION = 1
    FINISHED = 2
    PENALTIES = 3


class Player:
    def __init__(self, name: str, position: int, base_attack: int, base_defense: int, stamina: float = 100.0):
        self.name = name
        self.position = Position(position)  # CHANGED: cast through the enum so invalid values raise immediately
        self.base_attack = base_attack
        self.base_defense = base_defense
        self.stamina = stamina
        self.incidents = 0
        self.sent_off = False

    def deplete_stamina(self, rate):
        # Spec Section 3: Stamina(t) = Max(10.0, Stamina(t-1) - Base_Decay). Unchanged from sim.py - this was correct.
        self.stamina -= rate
        if self.stamina < 10.0:
            self.stamina = 10.0

    def get_effective_attack(self):
        return self.base_attack * (self.stamina / 100.0)

    def get_effective_defense(self):
        return self.base_defense * (self.stamina / 100.0)

    def add_incident(self):
        self.incidents += 1
        if self.incidents >= 2:
            self.sent_off = True
        return self.sent_off


class Team:
    def __init__(self, country_name: str, roster: list[Player], active_lineup: list[Player],
                 substitutions_remaining: int = 5):
        self.country_name = country_name
        self.roster = roster
        self.active_lineup = active_lineup
        self.substitutions_remaining = substitutions_remaining

    @property
    def bench(self):
        return [player for player in self.roster if player not in self.active_lineup]

    def get_aggregate_attack(self):
        attackers = [player for player in self.active_lineup if player.position in (Position.MIDFIELDER, Position.FORWARD)]
        if not attackers:  # bonus requirement: aggregate calc must not break if a position group ends up empty
            return 0.0
        return sum(player.get_effective_attack() for player in attackers) / len(attackers)

    def get_aggregate_defense(self):
        defenders = [player for player in self.active_lineup if player.position in (Position.GOALKEEPER, Position.DEFENDER)]
        if not defenders:
            return 0.0
        return sum(player.get_effective_defense() for player in defenders) / len(defenders)

    def execute_substitution(self, player_out: Player, player_in: Player):
        if self.substitutions_remaining > 0 and player_out in self.active_lineup and player_in in self.bench:
            i = self.active_lineup.index(player_out)
            self.active_lineup[i] = player_in
            self.substitutions_remaining -= 1
            return True
        print("Substitution not possible")
        return False

    def send_off_player(self, player: Player):
        if player in self.active_lineup:
            self.active_lineup.remove(player)


@dataclass(frozen=True)
class MatchEvent:
    event_types = {
        1: "GOAL",
        2: "SUBSTITUTION",
        3: "HALF_TIME",
        4: "FULL_TIME",
        5: "PENALTY_KICK",
        6: "RED_CARD",  # CHANGED: added so sent-off events stop colliding with HALF_TIME (see EventType comment above)
    }

    event_id: str
    event_type: int
    minute: int
    team: Optional[Team]
    player: Optional[Player]
    outcome_text: str

    def to_string(self):
        team_str = self.team.country_name if self.team else "MATCH"
        player_str = self.player.name if self.player else "-"
        return f"{self.event_types[self.event_type]} - {self.minute} - {team_str} - {player_str}"


class Match:
    phases = {1: "REGULATION", 2: "FINISHED", 3: "PENALTIES"}

    def __init__(self, home_team: Team, away_team: Team, home_score: int, away_score: int,
                 current_minute: int, timeline: list[MatchEvent], phase: int,
                 home_ai: Optional["MatchAI"] = None, away_ai: Optional["MatchAI"] = None):
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.current_minute = current_minute
        self.timeline = timeline
        self.phase = phase
        self.penalty_home_score = 0
        self.penalty_away_score = 0
        self.penalty_kicks = []
        self.home_ai = home_ai
        self.away_ai = away_ai

    def record(self, event_type: int, team: Optional[Team], player: Optional[Player], outcome_text: str):
        event = MatchEvent(
            event_id=str(len(self.timeline) + 1),
            event_type=event_type,
            minute=self.current_minute,
            team=team,
            player=player,
            outcome_text=outcome_text
        )
        self.timeline.append(event)
        return event

    def substitute(self, team: Team, player_out: Player, player_in: Player):
        if team.execute_substitution(player_out, player_in):
            self.record(EventType.SUBSTITUTION, team, player_in, f"{player_out.name} OFF, {player_in.name} ON")

    def auto_substitute(self, team: Team):
        """Bonus (base, non-AI): auto-swap whichever active player has the lowest
        stamina. CHANGED from sim.py: `player_in = team.bench[0]` used to run
        BEFORE checking whether the bench was even non-empty, so an empty bench
        crashed with IndexError on every tick. Now guarded up front. Also now
        picks the highest-stamina bench player (matching the AI's own SUBSTITUTE
        behavior) instead of an arbitrary bench[0], so both substitution paths
        behave consistently."""
        if team.substitutions_remaining <= 0 or not team.active_lineup or not team.bench:
            return
        player_out = min(team.active_lineup, key=lambda player: player.stamina)
        if player_out.stamina <= 30.0:
            player_in = max(team.bench, key=lambda player: player.stamina)  # CHANGED: was bench[0]
            self.substitute(team, player_out, player_in)

    def record_incident(self, team: Team, player: Player):
        sent_off = player.add_incident()
        if sent_off:
            team.send_off_player(player)
            # CHANGED: was event_type=3 (collided with HALF_TIME) - now EventType.RED_CARD
            self.record(EventType.RED_CARD, team, player, f"{player.name} SENT OFF!")

    def _stamina_multiplier(self, ai: Optional["MatchAI"]) -> float:
        """CHANGED: entirely new. Neither sim.py nor the Gemini version wired
        this up. Spec bonus AI table: stamina decay multiplier =
        (1 + (risk_tolerance - 0.5) * 0.4) -> 0.8x at risk_tolerance=0.0
        (energy conserved), 1.2x at risk_tolerance=1.0 (extra fatigue).
        Teams without an attached AI use risk_tolerance=0.5 -> multiplier 1.0,
        i.e. plain Base_Decay, matching the non-AI base spec exactly."""
        risk_tolerance = ai.risk_tolerance if ai else 0.5
        return 1 + (risk_tolerance - 0.5) * 0.4

    def run_minute_tick(self):
        if self.phase == Phase.FINISHED:
            return

        self.current_minute += 1

        # CHANGED: stamina decay now scaled by each team's own AI risk_tolerance
        # (previously a flat 0.5 for everyone, ignoring the bonus AI table entirely)
        for player in self.home_team.active_lineup:
            player.deplete_stamina(0.5 * self._stamina_multiplier(self.home_ai))
        for player in self.away_team.active_lineup:
            player.deplete_stamina(0.5 * self._stamina_multiplier(self.away_ai))

        # CHANGED: baseline auto-substitute now only runs for teams WITHOUT an
        # attached AI. In sim.py this ran unconditionally for both teams every
        # tick, which could compete with an AI's own SUBSTITUTE action for the
        # same low-stamina player and the same substitutions_remaining budget
        # in the same minute. Splitting responsibility this way means: no AI ->
        # baseline auto-sub handles fatigue; AI attached -> the AI's own
        # SUBSTITUTE decision is the only thing managing subs for that team.
        for team, ai in ((self.home_team, self.home_ai), (self.away_team, self.away_ai)):
            if ai is None:
                self.auto_substitute(team)

        # CHANGED: AI coaching integration point (spec: "call observe_state() ->
        # decide_action() -> apply_decision() once per run_minute_tick()").
        # Entirely missing from sim.py - the MatchAI class existed but nothing
        # ever called it during the simulation loop.
        for ai in (self.home_ai, self.away_ai):
            if ai is not None:
                state = ai.observe_state(self)
                action = ai.decide_action(state)
                ai.apply_decision(action, self)

        self.process_goal_attempt(self.home_team, self.away_team, self.home_ai)
        self.process_goal_attempt(self.away_team, self.home_team, self.away_ai)

        if self.current_minute == 45:
            self.record(EventType.HALF_TIME, None, None, f"HALF TIME - {self.home_score} - {self.away_score}")

        if self.current_minute >= 90:
            if self.home_score != self.away_score:
                self.phase = Phase.FINISHED
                winner = self.home_team if self.home_score > self.away_score else self.away_team
                self.record(EventType.FULL_TIME, None, None, f"{winner.country_name} WINS")
            else:
                self.phase = Phase.PENALTIES
                # CHANGED: was event_type=3 (HALF_TIME) - this is a full-time /
                # phase-transition announcement, not a half-time one, so it
                # should be EventType.FULL_TIME, not collide with HALF_TIME.
                self.record(EventType.FULL_TIME, None, None, "FULL TIME - SCORES LEVEL - PENALTIES")
                self.run_penalty_shootout()

    def process_goal_attempt(self, attacking_team: Team, defending_team: Team, ai: Optional["MatchAI"] = None):
        """CHANGED: signature now accepts the attacking team's AI (if any).
        Two fixes vs sim.py:
        1. Base random range corrected from (0.85, 1.15) to (0.75, 1.25) to
           match the PDF's "Full Simulation Formula Matrix" exactly:
           Goal scored if (Aggregate_Attack * Random(0.75, 1.25)) >
           (Aggregate_Defense * 1.3 * Random(0.80, 1.20)).
        2. The bonus AI table's live risk multiplier - upper bound =
           1.25 + (risk_tolerance - 0.5) * 0.4 (1.05 at risk_tolerance=0.0,
           1.45 at risk_tolerance=1.0) - is now actually wired in. sim.py's
           process_goal_attempt didn't take an `ai` argument at all, so this
           entire bonus table was unreachable dead spec. (The Gemini version
           attempted this but applied it as a separate flat multiplier
           alongside a still-fixed (0.75,1.25) range, which isn't what the
           table describes - it specifically says "upper bound", i.e. the
           random range itself should shift, not get an extra factor bolted on.)
        """
        if random.random() >= 0.10 or not attacking_team.active_lineup:
            return

        risk_tolerance = ai.risk_tolerance if ai else 0.5
        attack_upper = 1.25 + (risk_tolerance - 0.5) * 0.4

        attack_score = attacking_team.get_aggregate_attack() * random.uniform(0.75, attack_upper)
        defense_score = defending_team.get_aggregate_defense() * 1.3 * random.uniform(0.80, 1.20)

        if attack_score > defense_score:
            if attacking_team is self.home_team:
                self.home_score += 1
            else:
                self.away_score += 1
            scorer = max(attacking_team.active_lineup, key=lambda player: player.get_effective_attack())
            self.record(EventType.GOAL, attacking_team, scorer, "GOAL!")

    def penalty_sort(self, team: Team):
        outfield = [p for p in team.active_lineup if p.position != Position.GOALKEEPER]
        keeper = [p for p in team.active_lineup if p.position == Position.GOALKEEPER]
        return outfield + keeper

    def take_penalty(self, team: Team, kicker: Player, is_home: bool):
        opponent = self.away_team if is_home else self.home_team
        keeper = next((p for p in opponent.active_lineup if p.position == Position.GOALKEEPER), None)
        keeper_defense = keeper.get_effective_defense() if keeper else 50.0

        scored = kicker.get_effective_attack() * random.uniform(0.85, 1.15) > keeper_defense * random.uniform(0.8, 1.2)

        if scored:
            if is_home:
                self.penalty_home_score += 1
            else:
                self.penalty_away_score += 1

        outcome = "SCORED" if scored else "MISSED"
        self.record(EventType.PENALTY_KICK, team, kicker,
                     f"PENALTY {outcome} ({self.penalty_home_score}-{self.penalty_away_score})")

        # CHANGED: sim.py had a render_penalty_board() method but never populated
        # self.penalty_kicks anywhere - it would always print "(no penalty
        # shootout took place)" even after a real shootout. Recording each kick
        # here is what actually feeds the board.
        self.penalty_kicks.append({
            "kick": len(self.penalty_kicks) + 1,
            "team": team.country_name,
            "kicker": kicker.name,
            "result": outcome,
            "home": self.penalty_home_score,
            "away": self.penalty_away_score
        })

    def run_penalty_shootout(self):
        home_kickers = self.penalty_sort(self.home_team)
        away_kickers = self.penalty_sort(self.away_team)

        if not home_kickers or not away_kickers:  # CHANGED: guard against a fully sent-off team
            self.phase = Phase.FINISHED
            self.record(EventType.FULL_TIME, None, None, "PENALTIES ABANDONED - a team has no eligible kickers")
            return

        self.penalty_home_score = 0
        self.penalty_away_score = 0

        for i in range(5):
            if i < len(home_kickers):
                self.take_penalty(self.home_team, home_kickers[i], is_home=True)
            if i < len(away_kickers):
                self.take_penalty(self.away_team, away_kickers[i], is_home=False)

            remaining_home = 5 - (i + 1)
            remaining_away = 5 - (i + 1)
            if self.penalty_home_score > self.penalty_away_score + remaining_away or \
               self.penalty_away_score > self.penalty_home_score + remaining_home:
                break

        sdi = 5
        while self.penalty_home_score == self.penalty_away_score:
            home_kicker = home_kickers[sdi % len(home_kickers)]
            away_kicker = away_kickers[sdi % len(away_kickers)]
            self.take_penalty(self.home_team, home_kicker, is_home=True)
            self.take_penalty(self.away_team, away_kicker, is_home=False)
            sdi += 1

        self.phase = Phase.FINISHED
        winner = self.home_team if self.penalty_home_score > self.penalty_away_score else self.away_team
        self.record(EventType.FULL_TIME, None, None,
                    f"{winner.country_name} WINS ON PENALTIES ({self.penalty_home_score}-{self.penalty_away_score})")

    def render_penalty_board(self):
        """Kick-by-kick board, styled like the Task 1.1 standings table."""
        if not self.penalty_kicks:
            return "(no penalty shootout took place)"
        header = f"{'Kick':<5}{'Team':<6}{'Kicker':<18}{'Result':<9}{'Score':<7}"
        divider = "-" * len(header)
        rows = [header, divider]
        for k in self.penalty_kicks:
            code = k["team"][:3].upper()
            score_str = f"{k['home']}-{k['away']}"
            rows.append(f"{k['kick']:<5}{code:<6}{k['kicker']:<18}{k['result']:<9}{score_str:<7}")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# BONUS: AI Model - Controlled Match Engine
# ---------------------------------------------------------------------------

class LLMModel:
    """Pluggable interface - swap in a real client (Gemini, OpenAI, etc.) by
    implementing generate(). Per the spec hint: 'implement decide_action() by
    calling a real LLM API ... and parsing its reply into one of the four
    actions.' Kept identical to sim.py; this part was already correct."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Provide a real API-backed implementation of generate().")


class MockCoachModel(LLMModel):
    """Deterministic stand-in for testing without a live API."""

    def generate(self, prompt: str) -> str:
        own_score = _extract_number(prompt, "own_score")
        opp_score = _extract_number(prompt, "opponent_score")
        lowest_stamina = _extract_number(prompt, "lowest_active_stamina")

        if lowest_stamina is not None and lowest_stamina <= 35:
            return "SUBSTITUTE"
        if own_score is not None and opp_score is not None:
            if own_score < opp_score:
                return "PUSH_ATTACK"
            if own_score > opp_score:
                return "HOLD"
        return "CHANGE_FORMATION"


def _extract_number(text: str, key: str):
    marker = f"'{key}':"
    idx = text.find(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker):].strip()
    num = ""
    for ch in tail:
        if ch.isdigit() or ch == "." or (ch == "-" and not num):
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return None


class MatchAI:
    """Class: MatchAI - extends coaching logic onto a Team."""

    VALID_ACTIONS = {"SUBSTITUTE", "CHANGE_FORMATION", "HOLD", "PUSH_ATTACK"}

    # Kept as a 2-bucket table (defense_bucket/attack_bucket) matching exactly
    # what get_aggregate_attack()/get_aggregate_defense() read from - this was
    # already correct in sim.py and is the part of sim.py's design that's
    # functionally superior to the Gemini version (see next comment).
    FORMATIONS = {
        "DEFENSIVE_5_3_2": {"defense_bucket": 6, "attack_bucket": 5},
        "BALANCED_4_4_2":  {"defense_bucket": 5, "attack_bucket": 6},
        "ATTACKING_3_4_3": {"defense_bucket": 4, "attack_bucket": 7},
    }

    def __init__(self, model: LLMModel, controlled_team: Team, risk_tolerance: float = 0.5):
        self.model = model
        self.controlled_team = controlled_team
        self.risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self.decision_log = []
        self.current_formation = "BALANCED_4_4_2"

    def observe_state(self, match: "Match") -> dict:
        team = self.controlled_team
        is_home = team is match.home_team
        opponent = match.away_team if is_home else match.home_team

        active_lineup = [
            {"name": p.name, "position": p.position.name, "stamina": round(p.stamina, 1)}
            for p in team.active_lineup
        ]
        lowest_active_stamina = min((p.stamina for p in team.active_lineup), default=100.0)

        return {
            "minute": match.current_minute,
            "phase": Match.phases.get(match.phase, str(match.phase)),
            "own_score": match.home_score if is_home else match.away_score,
            "opponent_score": match.away_score if is_home else match.home_score,
            "own_aggregate_attack": round(team.get_aggregate_attack(), 2),
            "own_aggregate_defense": round(team.get_aggregate_defense(), 2),
            "opponent_aggregate_attack": round(opponent.get_aggregate_attack(), 2),
            "opponent_aggregate_defense": round(opponent.get_aggregate_defense(), 2),
            "active_lineup": active_lineup,
            "lowest_active_stamina": round(lowest_active_stamina, 1),
            "bench": [
                {"name": p.name, "position": p.position.name, "stamina": round(p.stamina, 1)}
                for p in team.bench
            ],
            "substitutions_remaining": team.substitutions_remaining,
            "current_formation": self.current_formation,
            "risk_tolerance": round(self.risk_tolerance, 2),
        }

    def decide_action(self, state: dict) -> str:
        prompt = self._build_prompt(state)
        raw_response = self.model.generate(prompt)
        return self._parse_action(raw_response)

    def _build_prompt(self, state: dict) -> str:
        return (
            "You are the head coach AI for a football match simulation.\n"
            f"Current match state: {state}\n\n"
            "Choose exactly one tactical action from: SUBSTITUTE, CHANGE_FORMATION, HOLD, PUSH_ATTACK.\n"
            "Reply with only the action name, nothing else."
        )

    def _parse_action(self, raw_response: str) -> str:
        if not raw_response:
            return "HOLD"
        upper = raw_response.upper()
        for action in self.VALID_ACTIONS:
            if action in upper:
                return action
        return "HOLD"

    def apply_decision(self, action: str, match: "Match"):
        team = self.controlled_team
        detail = ""

        if action == "SUBSTITUTE":
            detail = self._apply_substitute(team, match)
        elif action == "CHANGE_FORMATION":
            detail = self._apply_formation_change(team)
        elif action == "PUSH_ATTACK":
            before = self.risk_tolerance
            self.risk_tolerance = min(1.0, round(self.risk_tolerance + 0.2, 2))
            detail = f"risk_tolerance {before:.2f} -> {self.risk_tolerance:.2f} (more aggressive)"
        elif action == "HOLD":
            before = self.risk_tolerance
            self.risk_tolerance = max(0.0, round(self.risk_tolerance - 0.2, 2))
            detail = f"risk_tolerance {before:.2f} -> {self.risk_tolerance:.2f} (more conservative)"
        else:
            detail = "unrecognized action - no-op"

        self.decision_log.append({"minute": match.current_minute, "action": action, "reasoning": detail})
        return action

    def _apply_substitute(self, team: Team, match: "Match") -> str:
        if not team.bench:
            return "no bench players available - skipped"
        if team.substitutions_remaining <= 0:
            return "no substitutions remaining - skipped"
        if not team.active_lineup:
            return "no active players to substitute - skipped"

        player_out = min(team.active_lineup, key=lambda p: p.stamina)
        player_in = max(team.bench, key=lambda p: p.stamina)
        match.substitute(team, player_out, player_in)
        return f"{player_out.name} (stamina {player_out.stamina:.1f}) OFF, {player_in.name} (stamina {player_in.stamina:.1f}) ON"

    def _apply_formation_change(self, team: Team) -> str:
        """This method in sim.py was ALREADY functionally correct (unlike the
        Gemini version, whose equivalent method only updated a label string
        and never touched player.position at all, making CHANGE_FORMATION a
        total no-op there). Kept as-is other than switching to Position enum
        comparisons for consistency with the rest of this merged file."""
        next_formation = self._pick_next_formation(team)
        target = self.FORMATIONS[next_formation]

        gk = next((p for p in team.active_lineup if p.position == Position.GOALKEEPER), None)
        outfield = [p for p in team.active_lineup if p.position != Position.GOALKEEPER]

        needed_defense_outfield = target["defense_bucket"] - (1 if gk else 0)
        needed_attack = target["attack_bucket"]

        if needed_defense_outfield < 0 or needed_defense_outfield + needed_attack != len(outfield):
            return f"formation {next_formation} incompatible with current lineup size - skipped"

        ranked = sorted(outfield, key=lambda p: p.base_defense - p.base_attack, reverse=True)
        for i, p in enumerate(ranked):
            p.position = Position.DEFENDER if i < needed_defense_outfield else Position.MIDFIELDER

        previous = self.current_formation
        self.current_formation = next_formation
        return f"formation {previous} -> {next_formation} (defense_bucket={target['defense_bucket']}, attack_bucket={target['attack_bucket']})"

    def _pick_next_formation(self, team: Team) -> str:
        if self.risk_tolerance >= 0.65:
            return "ATTACKING_3_4_3"
        if self.risk_tolerance <= 0.35:
            return "DEFENSIVE_5_3_2"
        return "BALANCED_4_4_2"
