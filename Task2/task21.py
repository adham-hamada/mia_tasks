import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("Task2/data/results.csv", parse_dates=["date"])
goals = pd.read_csv("Task2/data/goalscorers.csv", parse_dates=["date"])
shootouts = pd.read_csv("Task2/data/shootouts.csv", parse_dates=["date"])

#Data cleaning (They can't tell us anything about "performance", so we drop them.)
before = len(results)
results = results.dropna(subset=["home_score", "away_score"]).copy()
dropped = before - len(results)
print(f"[clean] results.csv: dropped {dropped} unplayed/scoreless fixtures")

results["home_score"] = results["home_score"].astype(int)
results["away_score"] = results["away_score"].astype(int)

# Trim stray whitespace in team names so "Brazil " and "Brazil" don't
# end up as two different teams.
for col in ["home_team", "away_team", "tournament", "city", "country"]:
    results[col] = results[col].astype(str).str.strip()

# A very small number of matches share the exact same date/home/away
# We still need a unique key per match for merging goalscorers/shootouts onto
# results, so we number duplicates instead of dropping them.
results = results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
results["match_seq"] = results.groupby(["date", "home_team", "away_team"]).cumcount()
results["match_id"] = (
    results["date"].dt.strftime("%Y-%m-%d") + "_" +
    results["home_team"] + "_" + results["away_team"] + "_" +
    results["match_seq"].astype(str)
)

for df in (goals, shootouts):
        for col in ["home_team", "away_team"]:
            df[col] = df[col].astype(str).str.strip()

goals["team"] = goals["team"].astype(str).str.strip()
goals["own_goal"] = goals["own_goal"].astype(bool)
goals["penalty"] = goals["penalty"].astype(bool)

shootouts["winner"] = shootouts["winner"].astype(str).str.strip()

print(f"[clean] results: {len(results):,} matches | "
        f"goals: {len(goals):,} goal events | "
        f"shootouts: {len(shootouts):,} shootouts")

results = results.copy()
results["year"] = results["date"].dt.year
results["decade"] = (results["year"] // 10) * 10
results["total_goals"] = results["home_score"] + results["away_score"]
results["goal_margin"] = (results["home_score"] - results["away_score"]).abs()

conditions = [
    results["home_score"] > results["away_score"],
    results["home_score"] < results["away_score"],
]
results["outcome"] = np.select(conditions, ["home_win", "away_win"], default="draw")

keep_cols = ["match_id", "date", "year", "decade", "tournament", "neutral"]

home = results.rename(columns={
    "home_team": "team", "away_team": "opponent",
    "home_score": "goals_for", "away_score": "goals_against",
})[keep_cols + ["team", "opponent", "goals_for", "goals_against"]].copy()
home["venue"] = np.where(results["neutral"], "neutral", "home")

away = results.rename(columns={
    "away_team": "team", "home_team": "opponent",
    "away_score": "goals_for", "home_score": "goals_against",
})[keep_cols + ["team", "opponent", "goals_for", "goals_against"]].copy()
away["venue"] = np.where(results["neutral"], "neutral", "away")

team_matches = pd.concat([home, away], ignore_index=True)

# Standard football points: win = 3, draw = 1, loss = 0
team_matches["points"] = np.select(
    [team_matches["goals_for"] > team_matches["goals_against"],
        team_matches["goals_for"] == team_matches["goals_against"]],
    [3, 1],
    default=0,
)
team_matches["goal_margin"] = team_matches["goals_for"] - team_matches["goals_against"]


def top_performers(team_matches, goals):
    # --- Top 10 teams by total goals scored across history ---
    top_scoring_teams = (
        team_matches.groupby("team")["goals_for"].sum().sort_values(ascending=False).head(10).reset_index(name="total_goals")
    )

    # --- Top 10 individual goalscorers ---
    # Own goals are credited to the benefiting team in this dataset, but an
    # own goal is not a "goal scored" by that player's own record, so we
    # exclude own_goal rows before counting individual totals.
    non_og = goals[~goals["own_goal"]]
    top_scorers = (
        non_og.groupby("scorer")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="goals")
    )
    return top_scoring_teams, top_scorers

def team_efficiency(team_matches):
    team_stats = team_matches.groupby("team").agg(
        matches=("match_id", "count"),
        points=("points", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    ).reset_index()
    team_stats["points_per_match"] = team_stats["points"] / team_stats["matches"]
    team_stats["goals_per_match"] = team_stats["goals_for"] / team_stats["matches"]

    # Unfiltered top 10 — this is misleading, since teams that have played
    # only a handful of games can post a perfect record by chance (e.g. a
    # micronation team that has played 2 matches and won both).
    unfiltered_top10 = team_stats.sort_values("points_per_match", ascending=False).head(10)

    # Filtered top 10 — only teams with a meaningful sample size of matches.
    qualified = team_stats[team_stats["matches"] >= 100]
    filtered_top10 = qualified.sort_values("points_per_match", ascending=False).head(10)

    print(f"[efficiency] {(team_stats['matches'] < 100).sum()} of "
          f"{len(team_stats)} teams have fewer than 100 matches "
          f"and are excluded from the filtered ranking")

    return team_stats, unfiltered_top10, filtered_top10

def drama_analysis(results, shootouts):
    """
    shootouts.csv has no tournament/date-context columns beyond date and
    the two teams, so per the task note we MERGE it onto results.csv to
    recover the tournament (and therefore decade) each shootout happened in.
    """
    merged = shootouts.merge(
        results[["date", "home_team", "away_team", "tournament", "decade"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    unmatched = merged["tournament"].isna().sum()
    print(f"[drama] {unmatched} of {len(merged)} shootouts could not be "
          f"matched back to a result row (kept, but without tournament info)")

    by_decade = (
        merged.dropna(subset=["decade"])
        .groupby("decade").size()
        .reset_index(name="shootouts")
        .sort_values("decade")
    )

    by_tournament = (
        merged.groupby("tournament").size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="shootouts")
    )

    # Raw counts favour whichever tournament simply has the most matches
    # (Friendly, with 18k+ matches). A fairer view of "which tournaments
    # cause shootouts most often" is the RATE: shootouts per match played
    # in that tournament.
    matches_played = results.groupby("tournament").size().rename("matches_played")
    shootout_counts = merged.groupby("tournament").size().rename("shootouts")
    rate = pd.concat([shootout_counts, matches_played], axis=1).dropna()
    rate["shootout_rate"] = rate["shootouts"] / rate["matches_played"]
    # only tournaments with at least 10 shootouts, so the rate isn't noisy
    rate_top10 = (
        rate[rate["shootouts"] >= 10]
        .sort_values("shootout_rate", ascending=False)
        .head(10)
        .reset_index()
    )

    return merged, by_decade, by_tournament, rate_top10

def worst_performers(results, goals, team_matches):
    """
    "Scored first" requires knowing the chronological order of goals in a
    match, which needs the `minute` column. Roughly half of matches in
    goalscorers.csv have no scorer/minute data at all (older/minor games),
    and a 0-0 draw has no first goal by definition — those matches are
    naturally excluded. On top of that, if ANY goal in a match is missing
    its minute we cannot trust the ordering, so we drop that match too
    rather than risk mislabeling the "first" scorer.
    """
    goals_with_id = goals.merge(
        results[["match_id", "date", "home_team", "away_team"]],
        on=["date", "home_team", "away_team"],
        how="left",
    ).dropna(subset=["match_id"])

    unreliable_matches = goals_with_id.loc[goals_with_id["minute"].isna(), "match_id"].unique()
    usable_goals = goals_with_id[~goals_with_id["match_id"].isin(unreliable_matches)]

    first_goal = (
        usable_goals.sort_values("minute")
        .groupby("match_id").first()[["team"]]
        .rename(columns={"team": "first_scoring_team"})
        .reset_index()
    )
    print(f"[worst_performers] first goal determinable for {len(first_goal):,} of "
          f"{results['match_id'].nunique():,} matches")

    # Bring in that team's own match outcome (points) for the match in question.
    check = first_goal.merge(
        team_matches[["match_id", "team", "points"]],
        left_on=["match_id", "first_scoring_team"],
        right_on=["match_id", "team"],
        how="left",
    )
    check["failed_to_win"] = check["points"] != 3  # draw (1pt) or loss (0pt)

    summary = check.groupby("first_scoring_team").agg(
        times_scored_first=("match_id", "count"),
        times_failed_to_win=("failed_to_win", "sum"),
    ).reset_index()
    summary["fail_rate"] = summary["times_failed_to_win"] / summary["times_scored_first"]

    # Raw counts (naturally favours teams that play — and score first — a lot)
    by_count = summary.sort_values("times_failed_to_win", ascending=False).head(10)

    # Rate, restricted to teams with enough "scored first" occasions to be meaningful
    qualified = summary[summary["times_scored_first"] >= 20]
    by_rate = qualified.sort_values("fail_rate", ascending=False).head(10)

    return summary, by_count, by_rate

def era_comparison(results):
    era = results.groupby("decade").agg(
        matches=("match_id", "count"),
        avg_goals_per_match=("total_goals", "mean"),
        draw_rate=("outcome", lambda s: (s == "draw").mean()),
        # winning margin only makes sense for decisive (non-draw) matches
        avg_winning_margin=("goal_margin", lambda s: s[s > 0].mean()),
    ).reset_index()
    return era

def outlier_detection(results, z_thresh=6):
    # "Most lopsided match" = largest goal_margin. This directly answers
    # the task's question.
    top_blowouts = (
        results.sort_values("goal_margin", ascending=False)
        .head(10)[["date", "home_team", "away_team", "home_score",
                    "away_score", "tournament", "goal_margin"]]
    )

    # Statistical outlier check: how many standard deviations does each
    # match's goal_margin sit above the historical mean? This is a sanity
    # check for possible data-entry errors, not a claim these games didn't
    # happen. goal_margin (not total_goals) is used because "lopsided"
    # is specifically about the gap between the two teams, not raw scoring.
    mean_m, std_m = results["goal_margin"].mean(), results["goal_margin"].std()
    results = results.copy()
    results["margin_zscore"] = (results["goal_margin"] - mean_m) / std_m
    extreme = results[results["margin_zscore"] > z_thresh].sort_values(
        "margin_zscore", ascending=False
    )[["date", "home_team", "away_team", "home_score", "away_score",
       "tournament", "margin_zscore"]]

    print(f"[outliers] mean goal_margin={mean_m:.2f}, std={std_m:.2f}; "
          f"{len(extreme)} matches sit more than {z_thresh} std above the mean "
          f"({len(extreme) / len(results):.2%} of all matches)")

    return top_blowouts, extreme

# helper function
def categorize_tournament(name):
    t = name.lower()
    if "world cup qualification" in t:
        return "World Cup Qualifying"
    if t == "fifa world cup":
        return "World Cup Finals"
    if "qualification" in t or "qualifying" in t:
        return "Continental Qualifying"
    if "friendly" in t:
        return "Friendly"
    if any(k in t for k in ["cup of nations", "euro", "copa am",
                             "asian cup", "gold cup", "nations league"]):
        return "Continental Finals"
    return "Other"



def make_visualizations(results, top_teams, top_scorers, unfilt_eff, filt_eff,
                         sh_decade, sh_tourn, sh_rate, wp_rate, era):
    COLOR = "#c1121f"  # single accent color used across charts

    # 1. Top 10 scoring teams (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = top_teams.sort_values("total_goals")
    ax.barh(data["team"], data["total_goals"], color=COLOR)
    ax.set_title("Top 10 National Teams by Total Goals Scored (1872–2026)")
    ax.set_xlabel("Total goals")
    plt.tight_layout()
    plt.savefig("Task2/charts/01_top10_scoring_teams.png")
    plt.close()

    # 2. Top 10 individual scorers (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = top_scorers.sort_values("goals")
    ax.barh(data["scorer"], data["goals"], color=COLOR)
    ax.set_title("Top 10 International Goalscorers of All Time")
    ax.set_xlabel("Goals (own goals excluded)")
    plt.tight_layout()
    plt.savefig("Task2/charts/02_top10_individual_scorers.png")
    plt.close()

    # 3. Team efficiency — unfiltered vs filtered top 10 (points per match)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    d1 = unfilt_eff.sort_values("points_per_match")
    axes[0].barh(d1["team"] + " (" + d1["matches"].astype(str) + " m)", d1["points_per_match"], color="#adb5bd")
    axes[0].set_title(f"Unfiltered Top 10\n(no minimum matches)")
    axes[0].set_xlabel("Points per match")

    d2 = filt_eff.sort_values("points_per_match")
    axes[1].barh(d2["team"] + " (" + d2["matches"].astype(str) + " m)", d2["points_per_match"], color=COLOR)
    axes[1].set_title(f"Filtered Top 10\n(min. 100 matches played)")
    axes[1].set_xlabel("Points per match")
    fig.suptitle("Team Efficiency: Why a Minimum-Matches Threshold Matters")
    plt.tight_layout()
    plt.savefig("Task2/charts/03_team_efficiency_comparison.png")
    plt.close()

    # 4. Penalty shootouts by decade (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sh_decade["decade"].astype(int).astype(str), sh_decade["shootouts"], color=COLOR)
    ax.set_title("Penalty Shootouts by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Number of shootouts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Task2/charts/04_shootouts_by_decade.png")
    plt.close()

    # 5. Penalty shootouts by tournament (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = sh_tourn.sort_values("shootouts")
    ax.barh(data["tournament"], data["shootouts"], color=COLOR)
    ax.set_title("Top 10 Tournaments by Number of Penalty Shootouts (Raw Count)")
    ax.set_xlabel("Number of shootouts")
    plt.tight_layout()
    plt.savefig("Task2/charts/05_shootouts_by_tournament.png")
    plt.close()

    # 5b. Shootout RATE by tournament (shootouts per match played) — this
    # is the fairer answer to "which tournaments cause them most often"
    fig, ax = plt.subplots(figsize=(8, 5))
    data = sh_rate.sort_values("shootout_rate")
    ax.barh(data["tournament"], data["shootout_rate"] * 100, color=COLOR)
    ax.set_title("Tournaments Most Likely to End in a Shootout\n(shootouts per 100 matches, min. 10 shootouts)")
    ax.set_xlabel("Shootouts per 100 matches")
    plt.tight_layout()
    plt.savefig("Task2/charts/05b_shootout_rate_by_tournament.png")
    plt.close()

    # 6. Worst performers — highest "scored first but failed to win" rate
    fig, ax = plt.subplots(figsize=(8, 5))
    data = wp_rate.sort_values("fail_rate")
    ax.barh(data["first_scoring_team"], data["fail_rate"] * 100, color=COLOR)
    ax.set_title(f"Teams Most Likely to Fail to Win After Scoring First\n"
                 f"(min. 20 occasions)")
    ax.set_xlabel("% of the time a lead did not turn into a win")
    plt.tight_layout()
    plt.savefig("Task2/charts/06_worst_performers_fail_rate.png")
    plt.close()

    # 7. Era comparison — goals/match and draw rate trend lines
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(era["decade"], era["avg_goals_per_match"], marker="o", color=COLOR)
    axes[0].set_ylabel("Avg. goals per match")
    axes[0].set_title("Goals per Match by Decade")

    axes[1].plot(era["decade"], era["draw_rate"] * 100, marker="o", color="#495057")
    axes[1].set_ylabel("Draw rate (%)")
    axes[1].set_title("Draw Rate by Decade")
    axes[1].set_xlabel("Decade")
    plt.tight_layout()
    plt.savefig("Task2/charts/07_era_trends.png")
    plt.close()

    # 8. Heatmap: avg goals per match, decade x tournament category
    results = results.copy()
    results["tourn_cat"] = results["tournament"].apply(categorize_tournament)
    pivot = results.pivot_table(
        index="decade", columns="tourn_cat", values="total_goals", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "Avg goals/match"})
    ax.set_title("Average Goals per Match — Decade vs. Tournament Type")
    plt.tight_layout()
    plt.savefig("Task2/charts/08_goals_heatmap_decade_tournament.png")
    plt.close()

    print(f"[viz] saved 9 charts to Task2/charts")


def main():

    print("\n" + "=" * 70)
    print("QUESTION 1: Top Performers")
    print("=" * 70)
    top_teams, top_scorers = top_performers(team_matches, goals)
    print(top_teams.to_string(index=False))
    print(top_scorers.to_string(index=False))

    print("\n" + "=" * 70)
    print("QUESTION 2: Team Efficiency")
    print("=" * 70)
    team_stats, unfilt_eff, filt_eff = team_efficiency(team_matches)
    print(unfilt_eff[["team", "matches", "points_per_match"]].to_string(index=False))
    print(filt_eff[["team", "matches", "points_per_match"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("QUESTION 3: Drama Analysis (shootouts)")
    print("=" * 70)
    sh_merged, sh_decade, sh_tourn, sh_rate = drama_analysis(results, shootouts)
    print(sh_decade.to_string(index=False))
    print(sh_tourn.to_string(index=False))
    print(sh_rate.to_string(index=False))

    print("\n" + "=" * 70)
    print("QUESTION 4: Worst Performers")
    print("=" * 70)
    wp_summary, wp_count, wp_rate = worst_performers(results, goals, team_matches)
    print(wp_count.to_string(index=False))
    print(wp_rate.to_string(index=False))

    print("\n" + "=" * 70)
    print("QUESTION 5: Era Comparison")
    print("=" * 70)
    era = era_comparison(results)
    print(era.to_string(index=False))

    print("\n" + "=" * 70)
    print("OUTLIER DETECTION")
    print("=" * 70)
    blowouts, extreme = outlier_detection(results)
    print(blowouts.to_string(index=False))

    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)
    make_visualizations(results, top_teams, top_scorers, unfilt_eff, filt_eff,
                         sh_decade, sh_tourn, sh_rate, wp_rate, era)

    print("\nDone. charts in ./charts/")
    return {
        "results": results, "goals": goals, "shootouts": shootouts,
        "team_matches": team_matches, "top_teams": top_teams,
        "top_scorers": top_scorers, "team_stats": team_stats,
        "unfilt_eff": unfilt_eff, "filt_eff": filt_eff,
        "sh_decade": sh_decade, "sh_tourn": sh_tourn,
        "wp_summary": wp_summary, "wp_count": wp_count, "wp_rate": wp_rate,
        "era": era, "blowouts": blowouts, "extreme": extreme,
    }


if __name__ == "__main__":
    main()
