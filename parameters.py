import tsplib95

problem = tsplib95.load('TSPLIB/berlin52.tsp/berlin52.tsp')
numbers_of_generations = 1000
population_size = 100
children_size = 100
mut_rate = 0.2