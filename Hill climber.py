import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 23
STEADY_STATE = 1000


class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0]) ** 2 +
                                                         (pos1[1] - pos2[1]) ** 2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph


def tweak_insert(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        if i1 > i2:
            i1, i2 = i2, i1
        # swap 2 nodes
        # for index in range(int((i2-i1)/2)):
        #   new_solution[index+i1], new_solution[i2-index]= new_solution[i2-index], new_solution[index+i1]
        for index in range(i2 - i1):
            new_solution[i1 + index], new_solution[i1 + index + 1] = new_solution[i1 + index + 1], new_solution[
                i1 + index]
        # temp = new_solution[i1]
        # new_solution[i1] = new_solution[i2]
        # new_solution[i2] = temp
        p = np.random.random()
    return new_solution


def tweak_swap(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        if i1 > i2:
            i1, i2 = i2, i1
        # swap 2 nodes
        for index in range(int((i2 - i1) / 2)):
            new_solution[index + i1], new_solution[i2 - index] = new_solution[i2 - index], new_solution[index + i1]
        # for index in range (i2-i1):
        #    new_solution[i1+index], new_solution[i1+index+1] = new_solution[i1+index+1], new_solution[i1+index]
        # temp = new_solution[i1]
        # new_solution[i1] = new_solution[i2]
        # new_solution[i2] = temp
        p = np.random.random()
    return new_solution


def main():
    swap = 0
    insert = 0
    chosen = ""
    problem = Tsp(NUM_CITIES)

    solution = np.array(range(NUM_CITIES))
    np.random.shuffle(solution)
    solution_cost = problem.evaluate_solution(solution)
    problem.plot(solution)
    print(problem.evaluate_solution(solution))
    history = [(0, solution_cost)]
    steady_state = 0
    step = 0
    while steady_state < STEADY_STATE:
        step += 1
        steady_state += 1
        new_solution_swap = tweak_swap(solution, pm=.5)
        new_solution_insert = tweak_insert(solution, pm=.85)
        new_solution_cost_swap = problem.evaluate_solution(new_solution_swap)
        new_solution_cost_insert = problem.evaluate_solution(new_solution_insert)
        if new_solution_cost_swap < new_solution_cost_insert:
            new_solution_cost = new_solution_cost_swap
            new_solution = new_solution_swap
            chosen = "swap"
        else:
            new_solution_cost = new_solution_cost_insert
            new_solution = new_solution_insert
            chosen = "insert"
        if new_solution_cost < solution_cost:
            solution = new_solution
            solution_cost = new_solution_cost
            history.append((step, solution_cost))
            steady_state = 0
            if chosen == "swap":
                swap += 1
            else:
                insert += 1
    problem.plot(solution)
    print(problem.evaluate_solution(solution))
    print("number of swaps offspirng selected =")
    print(swap)
    print("number of insert offspirng selected =")
    print(insert)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()
