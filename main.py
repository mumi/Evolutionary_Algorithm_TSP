import numpy as np
import matplotlib.pyplot as plt
import random
from parameters import population_size, children_size, problem, numbers_of_generations, mut_rate

class Node:  # Node = Location = Point
    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)

dataset = []
for id, (x, y) in problem.node_coords.items():
    dataset.append(Node(id, x, y))

# This function will be run once at the beginning of the program to create a distance matrix
def create_distance_matrix(node_list: list) -> list:
    matrix = [[0 for _ in range(len(dataset))] for _ in range(len(dataset))]

    # classical matrix creation with two for loops
    for i in range(0, len(matrix)-1):
        for j in range(0, len(matrix[0])-1):
            matrix[node_list[i].id][node_list[j].id] = problem.get_weight(node_list[i].id, node_list[j].id)
    return matrix


matrix = create_distance_matrix(dataset)  # calculate all distances among all points and create a matrix

# Chromosome = Solution = Path
# Chromosome will contain Node list. This will be used in crossover, mutation operations etc,
# Chromosome representation --> chr_representation is only for displaying the route in a simple/clear way
# Chromosome cost will be used to compare the chromosomes
# We want to minimize the cost. So, lower cost is better!
class Chromosome:
    def __init__(self, node_list: list):
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
def create_random_list(n_list: list) -> list[Chromosome]:
    start = n_list[0]  # start and end points should be same, so keep the first point before shuffling

    temp = n_list[1:]
    temp = random.sample(temp, len(temp))  # shuffle the node list

    temp.insert(0, start)  # add start point to the beginning of the chromosome
    temp.append(start)  # add start point to the end, because route should be ended where it started
    return temp


def initialization(data: list, pop_size: int) -> list:
    initial_population = []
    for i in range(0, pop_size):  # create chromosomes as much as population size
        temp = create_random_list(data)
        new_ch = Chromosome(temp)
        initial_population.append(new_ch)
    return initial_population

# Two points crossover
def crossover_two(p_1: Chromosome, p_2: Chromosome) -> tuple[Chromosome, Chromosome]:  # two points crossover
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


# Mixed two points crossover
def crossover_mix(p_1: Chromosome, p_2: Chromosome) -> tuple[Chromosome, Chromosome]:
    point_1, point_2 = random.sample(range(1, len(p_1.chromosome)-1), 2)
    begin = min(point_1, point_2)
    end = max(point_1, point_2)

    child_1_1 = p_1.chromosome[:begin]
    child_1_2 = p_1.chromosome[end:]
    child_1 = child_1_1 + child_1_2
    child_2 = p_2.chromosome[begin:end+1]

    child_1_remain = [item for item in p_2.chromosome[1:-1] if item not in child_1]
    child_2_remain = [item for item in p_1.chromosome[1:-1] if item not in child_2]

    child_1 = child_1_1 + child_1_remain + child_1_2
    child_2 += child_2_remain

    child_2.insert(0, p_2.chromosome[0])
    child_2.append(p_2.chromosome[0])

    return child_1, child_2

def mutation(chromosome: Chromosome) -> Chromosome:  # swap two nodes of the chromosome
    mutation_index_1, mutation_index_2 = random.sample(range(1, 19), 2)
    chromosome[mutation_index_1], chromosome[mutation_index_2] = chromosome[mutation_index_2], chromosome[mutation_index_1]
    return chromosome

def get_best(generation: list[Chromosome]) -> Chromosome:
    return generation[0]

# Find the best chromosome of the generation based on the cost
def find_best_population(generation: list[Chromosome]) -> list[Chromosome]:
    return sorted(generation, key=lambda i: i.fitness_value, reverse=True)[:population_size]

# Major function!
# Use elitism, crossover, mutation operators to create a new generation based on a previous generation
def create_new_generation(previous_generation: list[Chromosome], mutation_rate: float) -> list[Chromosome]:
    previous_best = find_best_population(previous_generation)  # find the best 100 chromosomes of the previous generation
    new_generation = previous_best

    # Use two chromosomes and create two chromosomes. So, iteration size will be half of the population size!
    for a in range(0, int(children_size/2)):

        parent_1 = random.choice(previous_best)  # select a parent chromosome randomly
        parent_2 = random.choice(previous_best)  # select a parent chromosome randomly

        while parent_1 == parent_2:
            parent_2 = random.choice(previous_best)

        child_1, child_2 = crossover_mix(parent_1, parent_2)  # This will create node lists, we need Chromosome objects
        child_1 = Chromosome(child_1)
        child_2 = Chromosome(child_2)

        if random.random() < mutation_rate:
            mutated = mutation(child_1.chromosome)
            child_1 = Chromosome(mutated)

        new_generation.append(child_1)
        new_generation.append(child_2)

    return find_best_population(new_generation)  # return the best 100 chromosomes of the new generation

# main function for genetic algorithm
def genetic_algorithm(num_of_generations: int, pop_size: int, mutation_rate: float, data_list: list[Node]) -> (Chromosome, list[float]):
    new_gen = initialization(data_list, pop_size)  # first generation is created with initialization function

    costs_for_plot = []  # this list is only for Cost-Generations graph. it will constitute y-axis of the graph

    for iteration in range(0, num_of_generations):
        new_gen = create_new_generation(new_gen, mutation_rate)  # create a new generation in each iteration
        # print the cost of first chromosome of each new generation to observe the change over generations
        print(str(iteration) + ". generation --> " + "cost --> " + str(new_gen[0].cost))
        costs_for_plot.append(get_best(new_gen).cost)  # append the best chromosome's cost of each new generation
        # to the list to plot in the graph

    return new_gen, costs_for_plot

def draw_cost_generation(y_list: list[float]):
    x_list = np.arange(1, len(y_list)+1)  # create a numpy list from 1 to the numbers of generations

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
    best_solution = get_best(last_generation)
    draw_cost_generation(y_axis)
    #draw_path(best_solution)
    for i in best_solution.chromosome:
        print(i.id, end=",")