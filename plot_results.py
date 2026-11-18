"""Строит графики по сохранённым CSV, без повторного обучения."""

import argparse
import csv
from pathlib import Path
import statistics


def read_curves(path, method, metric):
    """Объединяет значения разных seed по номеру эпизода."""
    by_episode = {}
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["method"] == method:
                by_episode.setdefault(int(row["episode"]), []).append(float(row[metric]))
    episodes = sorted(by_episode)
    means = [statistics.mean(by_episode[e]) for e in episodes]
    deviations = [statistics.stdev(by_episode[e]) if len(by_episode[e]) > 1 else 0.0 for e in episodes]
    return episodes, means, deviations


def draw_curves(axis, path, metric, title, ylabel, percent=False):
    """Линия — среднее по seed, полоса — одно стандартное отклонение."""
    for method, label, color in [("q_learning", "Q-learning", "#2463a6"),
                                 ("sarsa", "SARSA", "#c26426")]:
        x, mean, sd = read_curves(path, method, metric)
        factor = 100 if percent else 1
        mean = [m * factor for m in mean]
        low = [(m - s * factor) for m, s in zip(mean, sd)]
        high = [(m + s * factor) for m, s in zip(mean, sd)]
        if percent:
            low, high = [max(0, v) for v in low], [min(100, v) for v in high]
        axis.plot(x, mean, label=label, color=color)
        axis.fill_between(x, low, high, color=color, alpha=0.16)
    if not percent:
        axis.axhline(-388 / 230, color="#555555", linestyle="--", linewidth=1,
                     label="Точный оптимум (средний)")
    axis.set(title=title, xlabel="Эпизоды обучения", ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)


def plot_results(folder):
    """Сохраняет награду обучения, награду готовой стратегии и долю оптимумов."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 8), layout="constrained")
    draw_curves(axes[0], folder / "learning.csv", "mean_return",
                "Награда при обучении · скользящее среднее", "Награда за эпизод (−подъёмы)")
    draw_curves(axes[1], folder / "evaluation.csv", "mean_return",
                "Награда жадной стратегии · все 230 стартов", "Средняя награда (−подъёмы)")
    figure.savefig(folder / "learning_curves.png", dpi=220)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.8, 4.4), layout="constrained")
    draw_curves(axis, folder / "evaluation.csv", "optimal_rate",
                "Доля оптимальных решений на всех 230 начальных состояниях",
                "Оптимальные решения, %", percent=True)
    axis.set_ylim(-2, 102)
    figure.savefig(folder / "optimal_rate.png", dpi=220)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    plot_results(parser.parse_args().results)
