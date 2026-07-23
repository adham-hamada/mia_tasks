standings = {
    "ARG": {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0},
    "MEX": {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0},
    "POL": {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0},
    "KSA": {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
            }
def process_match(standings, team1, team2, team1_goals, team2_goals):
    standings[team1]["P"] += 1
    standings[team2]["P"] += 1

    standings[team1]["GF"] += team1_goals
    standings[team2]["GF"] += team2_goals

    standings[team1]["GA"] += team2_goals
    standings[team2]["GA"] += team1_goals

    standings[team1]["GD"] = standings[team1]["GF"] - standings[team1]["GA"]
    if standings[team1]["GD"] > 0:
        standings[team1]["GD"] = "+" + str(standings[team1]["GD"])
    standings[team2]["GD"] = standings[team2]["GF"] - standings[team2]["GA"]
    if standings[team2]["GD"] > 0:
        standings[team2]["GD"] = "+" + str(standings[team2]["GD"])

    if team1_goals > team2_goals:
        standings[team1]["W"] += 1
        standings[team2]["L"] += 1
        standings[team1]["Pts"] += 3
    elif team1_goals < team2_goals:
        standings[team2]["W"] += 1
        standings[team1]["L"] += 1
        standings[team2]["Pts"] += 3
    else:
        standings[team1]["D"] += 1
        standings[team2]["D"] += 1
        standings[team1]["Pts"] += 1
        standings[team2]["Pts"] += 1

def sort_standings(standings):
    return dict(sorted(standings.items(), key=lambda x: (x[1]["Pts"], x[1]["GD"], x[1]["GF"]), reverse=True))

def print_standings(standings):
    print(f"{'Team':<5} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'Pts':<4}")
    for team, stats in sort_standings(standings).items():
        print(f"{team:<5} {stats['P']:<3} {stats['W']:<3} {stats['D']:<3} {stats['L']:<3} {stats['GF']:<3} {stats['GA']:<3} {stats['GD']:<4} {stats['Pts']:<4}")

def input_handling():
    global standings
    for match in [("ARG", "MEX"), ("ARG", "POL"), ("ARG", "KSA"), ("MEX", "POL"), ("MEX", "KSA"), ("POL", "KSA")]:
        valid_input = False
        while not valid_input:
            try:
                result = input(f"Enter score for {match[0]} vs {match[1]} (format: 2-0): ")
                team1_goals, team2_goals = map(int, result.split('-')[:2])
                if team1_goals < 0 or team2_goals < 0 or (team1_goals == 0 and team2_goals == 0):
                    continue
                valid_input = True
                process_match(standings, match[0], match[1], team1_goals, team2_goals)
            except ValueError:
                continue

def main():
    input_handling()
    print()
    print_standings(sort_standings(standings))

if __name__ == "__main__":
    main()