import tsplib95
import random
from typing import Tuple, Any

from parameters import population_size

problem = tsplib95.load('TSPLIB/a280.tsp/a280.tsp')

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
# This matrix is needed to decrease the runtime and complexity of general flow.


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


# initialization
def initialization(data: list, pop_size: int) -> list:
    initial_population = []
    for i in range(0, pop_size):  # create chromosomes as much as population size
        temp = create_random_list(data)
        new_ch = Chromosome(temp)
        initial_population.append(new_ch)
    return initial_population


# selection of parent chromosomes to create child chromosomes
def selection(population: list[Chromosome]) -> Chromosome:  # tournament selection
    ticket_1, ticket_2, ticket_3, ticket_4 = random.sample(range(0, 99), 4)  # random 4 tickets

    # create candidate chromosomes based on ticket numbers
    candidate_1 = population[ticket_1]
    candidate_2 = population[ticket_2]
    candidate_3 = population[ticket_3]
    candidate_4 = population[ticket_4]

    # select the winner according to their costs
    if candidate_1.fitness_value > candidate_2.fitness_value:
        winner = candidate_1
    else:
        winner = candidate_2

    if candidate_3.fitness_value > winner.fitness_value:
        winner = candidate_3

    if candidate_4.fitness_value > winner.fitness_value:
        winner = candidate_4

    return winner  # winner = chromosome


# Three different crossover methods
# One point crossover
def crossover(p_1, p_2) -> tuple[Chromosome, Chromosome]:
    one_point = random.randint(2, 14)

    child_1 = p_1.chromosome[1:one_point]
    child_2 = p_2.chromosome[1:one_point]

    child_1_remain = [item for item in p_2.chromosome[1:-1] if item not in child_1]
    child_2_remain = [item for item in p_1.chromosome[1:-1] if item not in child_2]

    child_1 += child_1_remain
    child_2 += child_2_remain

    child_1.insert(0, p_1.chromosome[0])
    child_1.append(p_1.chromosome[0])

    child_2.insert(0, p_2.chromosome[0])
    child_2.append(p_2.chromosome[0])

    return child_1, child_2


# Two points crossover
def crossover_two(p_1, p_2) -> tuple[Chromosome, Chromosome]:  # two points crossover
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
def crossover_mix(p_1, p_2) -> tuple[Chromosome, Chromosome]:
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


# Mutation operation
def mutation(chromosome: Chromosome) -> Chromosome:  # swap two nodes of the chromosome
    mutation_index_1, mutation_index_2 = random.sample(range(1, 19), 2)
    chromosome[mutation_index_1], chromosome[mutation_index_2] = chromosome[mutation_index_2], chromosome[mutation_index_1]
    return chromosome

# Find the best chromosome of the generation based on the cost
def find_best(generation) -> Chromosome:
    best = generation[0]
    for n in range(1, len(generation)):
        if generation[n].cost < best.cost:
            best = generation[n]
    return best

# Find the best chromosome of the generation based on the cost
def find_best_population(generation: list[Chromosome]) -> list[Chromosome]:
    return sorted(generation, key=lambda i: i.fitness_value, reverse=True)[:population_size]

# Major function!
# Use elitism, crossover, mutation operators to create a new generation based on a previous generation
def create_new_generation(previous_generation: list[Chromosome], mutation_rate: float) -> list[Chromosome]:
    previous_best = find_best_population(previous_generation)  # find the best 100 chromosomes of the previous generation
    new_generation = []

    # Use two chromosomes and create two chromosomes. So, iteration size will be half of the population size!
    for a in range(0, int(len(previous_generation)/2)):
        parent_1 = random.choice(previous_best)  # select a parent chromosome randomly
        previous_best.remove(parent_1)  # remove the selected chromosome from the list
        parent_2 = random.choice(previous_best)  # select a parent chromosome randomly
        previous_best.remove(parent_2)  # remove the selected chromosome from the list

        child_1, child_2 = crossover_two(parent_1, parent_2)  # This will create node lists, we need Chromosome objects
        child_1 = Chromosome(child_1)
        child_2 = Chromosome(child_2)

        if random.random() < mutation_rate:
            mutated = mutation(child_1.chromosome)
            child_1 = Chromosome(mutated)

        new_generation.append(child_1)
        new_generation.append(child_2)

    return new_generation