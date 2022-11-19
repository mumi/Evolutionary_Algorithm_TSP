import numpy as np
import matplotlib.pyplot as plt
import random
from parameters import population_size, children_size, problem, numbers_of_generations, mut_rate, best_known_solution
class Node:  # Node = Location = Point
    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)

dataset = []
for id, (x, y) in problem.node_coords.items():
    dataset.append(Node(id, x, y))

# This function will be run once at the beginning of the program to create a distance matrix
def create_distance_matrix(node_list: list[Node]) -> list[list[float]]:
    matrix = [[0 for _ in range(len(dataset))] for _ in range(len(dataset))]
    for i in range(0, len(matrix)-1):
        for j in range(0, len(matrix[0])-1):
            matrix[node_list[i].id][node_list[j].id] = problem.get_weight(node_list[i].id, node_list[j].id)
    return matrix

matrix = create_distance_matrix(dataset)  # calculate all distances among all points and create a matrix

# Chromosome = Solution = Path
class Chromosome:
    def __init__(self, node_list: list[Node]):
        self.chromosome = node_list

        chr_representation = []
        for i in range(0, len(node_list)):
            chr_representation.append(self.chromosome[i].id)
        chr_representation = chr_representation

        distance = 0
        for j in range(1, len(chr_representation) - 1):  # get distances from the matrix
            distance += matrix[chr_representation[j]-1][chr_representation[j + 1]-1]
        self.cost = distance

        self.fitness_value = 1 / self.cost

# create a random chromosome --> shuffle node list randomly
def create_random_list(node_list: list[Node]) -> list[Chromosome]:
    start = node_list[0]  # start and end points should be same, so keep the first point before shuffling

    temp = node_list[1:]
    temp = random.sample(temp, len(temp))  # shuffle the node list

    temp.insert(0, start)  # add start point to the beginning of the chromosome
    temp.append(start)  # add start point to the end, because route should be ended where it started
    return temp

def initialization(data: list[Node], pop_size: int) -> list[Chromosome]:
    initial_population = []
    for i in range(0, pop_size):  # create chromosomes as much as population size
        temp = create_random_list(data)
        new_ch = Chromosome(temp)
        initial_population.append(new_ch)
    return initial_population

# Two points crossover
def crossover(p_1: Chromosome, p_2: Chromosome) -> tuple[Chromosome, Chromosome]:
    point_1, point_2 = random.sample(range(1, len(p_1.chromosome)-1), 2)
    begin = min(point_1, point_2)
    end = max(point_1, point_2)

    child_1 = p_1.chromosome[begin:end+1]
    child_2 = p_2.chromosome[begin:end+1]

    child_1_remain = [item for item in p_2.chromosome[1:-1] if item not in child_1]
    child_2_remain = [item for item in p_1.chromosome[1:-1] if item not in child_2]

    child_1 += child_1_remain
    child_2 += child_2_remain

    child_1.insert(0, p_1.chromosome[0])
    child_1.append(p_1.chromosome[0])

    child_2.insert(0, p_2.chromosome[0])
    child_2.append(p_2.chromosome[0])

    return child_1, child_2


def mutation(chromosome: Chromosome) -> Chromosome:  # swap two nodes of the chromosome
    mutation_index_1, mutation_index_2 = random.sample(range(1, 19), 2)
    chromosome[mutation_index_1], chromosome[mutation_index_2] = chromosome[mutation_index_2], chromosome[mutation_index_1]
    return chromosome

def get_best(generation: list[Chromosome]) -> Chromosome:
    return generation[0]

# Find the best chromosomes of the generation based on the cost
def find_best_population(generation: list[Chromosome]) -> list[Chromosome]:
    return sorted(generation, key=lambda x: x.fitness_value, reverse=True)[:population_size]

def create_new_generation(previous_generation: list[Chromosome], mutation_rate: float) -> list[Chromosome]:
    previous_best = find_best_population(previous_generation)
    new_generation = previous_best

    for a in range(0, int(children_size/2)):
        parent_1 = random.choice(previous_best)
        parent_2 = random.choice(previous_best)

        while parent_1 == parent_2:
            parent_2 = random.choice(previous_best)

        child_1, child_2 = crossover(parent_1, parent_2)  # This will create node lists, we need Chromosome objects
        child_1 = Chromosome(child_1)
        child_2 = Chromosome(child_2)

        if random.random() < mutation_rate:
            mutated = mutation(child_1.chromosome)
            child_1 = Chromosome(mutated)

        new_generation.append(child_1)
        new_generation.append(child_2)

    return find_best_population(new_generation)  # return the best chromosomes of the new generation

def genetic_algorithm(num_of_generations: int, pop_size: int, mutation_rate: float, data_list: list[Node]) -> (Chromosome, list[float]):
    new_gen = initialization(data_list, pop_size)

    costs_for_plot = []

    for iteration in range(0, num_of_generations):
        new_gen = create_new_generation(new_gen, mutation_rate)
        print(f"{str(iteration)}. generation --> {str(new_gen[0].cost)} ({round(best_known_solution / new_gen[0].cost *100, 1)} %)")
        costs_for_plot.append(get_best(new_gen).cost)
    return new_gen, costs_for_plot

def draw_cost_generation(y_list: list[float]):
    x_list = np.arange(1, len(y_list)+1)

    plt.plot(x_list, y_list)

    plt.title("Streckenkosten über Generationen")
    plt.xlabel("Generationen")
    plt.ylabel("Kosten")
    plt.show()

def draw_path(solution: Chromosome):
    x_list = []
    y_list = []

    for m in range(0, len(solution.chromosome)):
        x_list.append(solution.chromosome[m].x)
        y_list.append(solution.chromosome[m].y)

    fig, ax = plt.subplots()
    plt.scatter(x_list, y_list)

    ax.plot(x_list, y_list, '--', lw=2, color='black', ms=10)
    ax.set_xlim(0, 1650)
    ax.set_ylim(0, 1300)
    plt.show()

if __name__ == '__main__':
    last_generation, y_axis = genetic_algorithm(numbers_of_generations, population_size, mut_rate, dataset)
    #draw_path(get_best(last_generation))
    best_solution = [i.id for i in get_best(last_generation).chromosome]
    draw_cost_generation(y_axis)

    print(best_solution)