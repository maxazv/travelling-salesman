from travelling_salesman import *


def hill_climb_avg(its):
    pass


def sim_annealing_avg(its):
    pass


def loc_beam_avg(its):
    pass


def genetic_alg_avg(its):
    pass



# travelling salesman problem initialisation
tsp = TSP(15)
tsp.init_table(500, 0.001)
#print(f'Adjacency Matrix:\n{tsp.adj_matrix}\n')

# hill climbing solution
hc_opt, hc_cost = tsp.compute_hill_climb()
print(f'(HC)[Final]\t Route is hc_opt with cost \t{hc_cost}\n')

# simulated annealing solution
sa_opt, sa_cost = tsp.compute_sim_annealing()
print(f'(SA)[Final]\t Route is sa_opt with cost \t{sa_cost}\n')

# local beam search solution
lb_cost, lb_opt = tsp.compute_loc_beam_search(3, lb_its=500)
print(f'(LB)[Final]\t Route is sa_opt with cost \t{lb_cost}')