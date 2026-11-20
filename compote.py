"""Среда с четырьмя стаканами, Q-learning и SARSA."""

import argparse
from itertools import product
import json
from pathlib import Path
import random
import time

CAPACITY = 6
TOTAL = 12
TARGET = TOTAL // 4
ALPHA = 0.2
MAX_STEPS = 200
CHECK_EVERY = 1000


def initial_states():
    """Все неравномерные распределения; -1 означает, что стакана в руке нет."""
    return [(*volumes, -1) for volumes in product(range(CAPACITY + 1), repeat=4)
            if sum(volumes) == TOTAL and len(set(volumes)) != 1]


def is_goal(state):
    """Цель достигнута, когда во всех стаканах поровну."""
    return state[:4] == (TARGET,) * 4


def possible_actions(state):
    """Можно лить из непустого стакана в другой, в котором есть место."""
    if is_goal(state):
        return []
    return [(source, target) for source in range(4) for target in range(4)
            if source != target and state[source] > 0 and state[target] < CAPACITY]


def pour(state, action):
    """Переливает единицу. Новый источник стоит один подъём, прежний — ноль."""
    source, target = action
    if (is_goal(state) or source == target or not 0 <= source < 4
            or not 0 <= target < 4 or state[source] == 0 or state[target] == CAPACITY):
        raise ValueError("Недопустимое переливание")
    volumes = list(state[:4])
    volumes[source] -= 1
    volumes[target] += 1
    reward = -int(source != state[4])
    return (*volumes, source), reward


def action_values(q, state):
    """Создаёт нулевые оценки при первом посещении состояния."""
    if state not in q:
        q[state] = {action: 0.0 for action in possible_actions(state)}
    return q[state]


def choose_action(q, state, rng, epsilon):
    """С вероятностью epsilon исследует, иначе выбирает лучший известный ход."""
    values = action_values(q, state)
    if rng.random() < epsilon:
        return rng.choice(list(values))
    best = max(values.values())
    tied = [action for action, value in values.items() if abs(value - best) < 1e-12]
    return rng.choice(tied)


def greedy_action(q, state):
    """При проверке выбирает лучший ход; равенства разрешает в порядке пар."""
    values = action_values(q, state)
    return max(values, key=values.get)


def evaluate(q, starts):
    """Проверяет готовую стратегию на всех стартах, без случайных действий."""
    successes = optimal = lifts_total = steps_total = 0
    for start in starts:
        state, lifts = start, 0
        for steps in range(1, MAX_STEPS + 1):
            action = greedy_action(q, state)
            state, reward = pour(state, action)
            lifts -= reward
            if is_goal(state):
                break
        success = is_goal(state)
        # Каждый стакан выше среднего обязан хотя бы раз стать источником.
        minimum = sum(volume > TARGET for volume in start[:4])
        successes += success
        optimal += success and lifts == minimum
        lifts_total += lifts
        steps_total += steps
    count = len(starts)
    return {"success_rate": successes / count, "optimal_rate": optimal / count,
            "mean_lifts": lifts_total / count, "mean_return": -lifts_total / count,
            "mean_steps": steps_total / count}


def train(method, episodes, seed):
    """Обучает один метод. Различие Q-learning и SARSA находится в цели обновления."""
    if method not in ("q_learning", "sarsa"):
        raise ValueError("Метод должен быть q_learning или sarsa")
    starts = initial_states()
    starts_rng = random.Random(100000 + seed)
    episode_starts = [starts_rng.choice(starts) for _ in range(episodes)]
    rng = random.Random(seed)
    q, rewards, steps_history, success_history = {}, [], [], []
    checkpoints = [{"episode": 0, **evaluate(q, starts)}]
    training_seconds = 0.0

    for episode, state in enumerate(episode_starts):
        began = time.perf_counter()
        fraction = min(1.0, episode / max(1.0, 0.8 * (episodes - 1)))
        epsilon = 1.0 + fraction * (0.05 - 1.0)
        episode_reward = 0
        if method == "sarsa":
            action = choose_action(q, state, rng, epsilon)

        for steps in range(1, MAX_STEPS + 1):
            if method == "q_learning":
                action = choose_action(q, state, rng, epsilon)
            next_state, reward = pour(state, action)
            episode_reward += reward
            success = is_goal(next_state)

            if success:
                target = reward
            elif method == "q_learning":
                # Q-learning оценивает лучший следующий ход.
                target = reward + max(action_values(q, next_state).values())
            else:
                # SARSA оценивает тот ход, который действительно будет сделан.
                next_action = choose_action(q, next_state, rng, epsilon)
                target = reward + q[next_state][next_action]

            old_value = q[state][action]
            q[state][action] += ALPHA * (target - old_value)
            state = next_state
            if success:
                break
            if method == "sarsa":
                action = next_action

        # Лимит шагов не является целью: обновление выше сохраняет будущую оценку.
        training_seconds += time.perf_counter() - began
        rewards.append(episode_reward)
        steps_history.append(steps)
        success_history.append(success)
        if (episode + 1) % CHECK_EVERY == 0 or episode + 1 == episodes:
            checkpoints.append({"episode": episode + 1, **evaluate(q, starts)})

    return {"q": q, "rewards": rewards, "steps": steps_history,
            "success": success_history, "checkpoints": checkpoints,
            "training_seconds": training_seconds}


def save_policy(q, path):
    """Сохраняет только готовые действия, в читаемый JSON."""
    rows = [{"state": state, "action": max(values, key=values.get)}
            for state, values in sorted(q.items()) if values]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def demonstrate(volumes, policy_path):
    """Показывает переливания с нумерацией с 1 и возвращает признак достижения цели."""
    if len(volumes) != 4 or sum(volumes) != TOTAL or any(v < 0 or v > CAPACITY for v in volumes):
        raise ValueError("Нужны 4 целых объёма от 0 до 6 с суммой 12")
    rows = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = {tuple(row["state"]): tuple(row["action"]) for row in rows}
    state, lifts = (*volumes, -1), 0
    print(f"Начало: {state[:4]}")
    for _ in range(MAX_STEPS):
        if is_goal(state):
            break
        action = policy[state]
        state, reward = pour(state, action)
        lifts -= reward
        print(f"Стакан {action[0] + 1} в стакан {action[1] + 1}: "
              f"{state[:4]}, новых подъёмов: {-reward}")
    if is_goal(state):
        print(f"Готово. Подъёмов: {lifts}")
        return True
    print(f"Цель не достигнута за {MAX_STEPS} переливаний")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Применить сохранённую стратегию")
    parser.add_argument("volumes", nargs=4, type=int, help="Начальные объёмы четырёх стаканов")
    parser.add_argument("--policy", type=Path, default=Path("results/q_learning_policy.json"))
    args = parser.parse_args()
    try:
        if not demonstrate(args.volumes, args.policy):
            parser.exit(1)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
