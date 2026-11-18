"""Проверки физики переливания и независимый поиск точного минимума."""

from collections import deque
from itertools import product
import unittest

from compote import CAPACITY, TOTAL, TARGET, initial_states, is_goal, possible_actions, pour
from train import learning_curve


class CompoteTests(unittest.TestCase):
    def test_one_lift_for_two_recipients(self):
        state, reward1 = pour((5, 3, 2, 2, -1), (0, 2))
        state, reward2 = pour(state, (0, 3))
        self.assertTrue(is_goal(state))
        self.assertEqual(reward1 + reward2, -1)

    def test_changing_source_costs_another_lift(self):
        state, reward1 = pour((4, 4, 3, 1, -1), (0, 3))
        state, reward2 = pour(state, (1, 3))
        self.assertTrue(is_goal(state))
        self.assertEqual(reward1 + reward2, -2)

    def test_volumes_and_legal_actions(self):
        for volumes in product(range(CAPACITY + 1), repeat=4):
            if sum(volumes) != TOTAL:
                continue
            for held in range(-1, 4):
                state = (*volumes, held)
                for action in possible_actions(state):
                    next_state, _ = pour(state, action)
                    self.assertEqual(sum(next_state[:4]), TOTAL)
                    self.assertTrue(all(0 <= v <= CAPACITY for v in next_state[:4]))
        self.assertEqual(possible_actions((3, 3, 3, 3, -1)), [])
        with self.assertRaises(ValueError):
            pour((6, 6, 0, 0, -1), (2, 0))

    def test_minimum_against_shortest_paths(self):
        # 0-1 BFS считает цену пути, не используя формулу числа переполненных стаканов.
        # Запускаем поиск назад от цели сразу для всех возможных стартов.
        reverse = {}
        states = [(*v, h) for v in product(range(CAPACITY + 1), repeat=4)
                  if sum(v) == TOTAL for h in range(-1, 4)]
        for state in states:
            for action in possible_actions(state):
                next_state, reward = pour(state, action)
                reverse.setdefault(next_state, []).append((state, -reward))
        distances = {state: 0 for state in states if is_goal(state)}
        queue = deque(distances)
        while queue:
            state = queue.popleft()
            for previous, cost in reverse.get(state, []):
                candidate = distances[state] + cost
                if candidate < distances.get(previous, float("inf")):
                    distances[previous] = candidate
                    if cost == 0:
                        queue.appendleft(previous)
                    else:
                        queue.append(previous)
        starts = initial_states()
        self.assertEqual(len(starts), 230)
        for state in starts:
            self.assertEqual(distances[state], sum(v > TARGET for v in state[:4]))
        self.assertEqual(sum(distances[state] for state in starts), 388)

    def test_curve_uses_last_window(self):
        rewards = [-10] * 500 + [-2] * 500
        points = list(learning_curve(rewards))
        self.assertEqual(points[0], {"episode": 500, "mean_return": -10})
        self.assertEqual(points[-1], {"episode": 1000, "mean_return": -2})


if __name__ == "__main__":
    unittest.main()
