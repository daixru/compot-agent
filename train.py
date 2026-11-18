"""Запускает обучение обоих методов и сохраняет результаты эксперимента."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

from compote import train, save_policy


def save_csv(path, rows):
    """Записывает список одинаковых по структуре словарей в CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def learning_curve(rewards, window=500):
    """Скользящая средняя за 500 эпизодов; точка на графике каждые 100 эпизодов."""
    window = min(window, len(rewards))
    total = 0
    for index, reward in enumerate(rewards):
        total += reward
        if index >= window:
            total -= rewards[index - window]
        episode = index + 1
        if episode >= window and (episode % 100 == 0 or episode == len(rewards)):
            yield {"episode": episode, "mean_return": total / window}


def summarize(runs):
    """Сводит результаты отдельных seed, сохраняя различие обучения и проверки."""
    final = [run["checkpoints"][-1] for run in runs]
    stable95 = []
    for run in runs:
        points = run["checkpoints"]
        stable = [point["episode"] for i, point in enumerate(points)
                  if all(p["optimal_rate"] >= 0.95 for p in points[i:])]
        stable95.append(stable[0] if stable else None)
    reached = [episode for episode in stable95 if episode is not None]
    return {
        "final_mean": {key: statistics.mean(row[key] for row in final)
                       for key in final[0] if key != "episode"},
        "final_sd": {key: statistics.stdev(row[key] for row in final) if len(runs) > 1 else 0.0
                     for key in ("optimal_rate", "mean_lifts")},
        "stable95_episodes": stable95,
        "stable95_median": statistics.median(reached) if reached else None,
        "training_seconds_mean": statistics.mean(r["training_seconds"] for r in runs),
        "interactions_mean": statistics.mean(sum(r["steps"]) for r in runs),
        "all_training_return": statistics.mean(statistics.mean(r["rewards"]) for r in runs),
        "last_1000_return": statistics.mean(statistics.mean(r["rewards"][-1000:]) for r in runs),
        "truncation_rate": statistics.mean(1 - statistics.mean(r["success"]) for r in runs),
    }


def run_experiment(output, episodes=60000, seeds=20):
    """Одинаковые старты и настройки для Q-learning и SARSA, по нескольку запусков."""
    output.mkdir(parents=True, exist_ok=True)
    runs = {"q_learning": [], "sarsa": []}
    evaluation_rows, curve_rows = [], []
    with gzip.open(output / "training.csv.gz", "wt", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["method", "seed", "episode", "return", "steps", "success"])
        for seed in range(seeds):
            # Чередуем порядок методов, чтобы время не зависело только от порядка запуска.
            methods = ["q_learning", "sarsa"] if seed % 2 == 0 else ["sarsa", "q_learning"]
            for method in methods:
                run = train(method, episodes, seed)
                final = run["checkpoints"][-1]
                print(f"{method}, seed={seed}: успех {final['success_rate']:.1%}, "
                      f"оптимум {final['optimal_rate']:.1%}", flush=True)
                writer.writerows((method, seed, i + 1, reward, steps, int(success))
                                 for i, (reward, steps, success) in enumerate(
                                     zip(run["rewards"], run["steps"], run["success"])))
                evaluation_rows.extend({"method": method, "seed": seed, **row}
                                       for row in run["checkpoints"])
                curve_rows.extend({"method": method, "seed": seed, **row}
                                  for row in learning_curve(run["rewards"]))
                if seed == 0:
                    save_policy(run["q"], output / f"{method}_policy.json")
                del run["q"]  # Для сводки нужны награды и метрики, а не все таблицы Q.
                runs[method].append(run)

    summary = {"episodes": episodes, "seeds": seeds, "alpha": 0.2, "gamma": 1,
               "epsilon": "1 -> 0.05 за первые 80% эпизодов, затем 0.05",
               "max_steps": 200, "check_every": 1000,
               "oracle_mean_lifts": 388 / 230,
               "methods": {method: summarize(rows) for method, rows in runs.items()}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    save_csv(output / "evaluation.csv", evaluation_rows)
    save_csv(output / "learning.csv", curve_rows)
    from plot_results import plot_results
    plot_results(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seeds < 1:
        parser.error("Число эпизодов и число seed должны быть положительными")
    run_experiment(args.output, args.episodes, args.seeds)
