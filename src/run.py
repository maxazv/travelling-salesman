import time
from travelling_salesman import *


# travelling salesman problem initialisation
tsp = TSP(25)
tsp.init_table(1000, 0.001)
#print(f'Adjacency Matrix:\n{tsp.adj_matrix}\n')


def hill_climb_avg(its):
    avg_cost, avg_time = 0, 0
    for i in range(its):
        # measure time for each iteration
        start = time.time()
        _, hc_cost = tsp.compute_hill_climb()
        end = time.time()

        avg_time += (end - start)
        avg_cost += hc_cost

    return avg_cost/its, avg_time/its


def sim_annealing_avg(its):
    avg_cost, avg_time = 0, 0
    for i in range(its):
        # measure time for each iteration
        start = time.time()
        _, sa_cost = tsp.compute_sim_annealing()
        end = time.time()

        avg_time += (end - start)
        avg_cost += sa_cost
    
    return avg_cost/its, avg_time/its


def loc_beam_avg(its):
    avg_cost, avg_time = 0, 0
    for i in range(its):
        # measure time for each iteration
        start = time.time()
        lb_ret = tsp.compute_loc_beam_search(4, lb_its=50)
        end = time.time()

        avg_time += (end - start)
        avg_cost += lb_ret[0]

    return avg_cost/its, avg_time/its


def genetic_alg_avg(its):
    pass



# algorithm measurements
msr_iters = 20
print(hill_climb_avg(msr_iters))
print(sim_annealing_avg(msr_iters))
print(loc_beam_avg(msr_iters))

'''
# hill climbing solution
hc_opt, hc_cost = tsp.compute_hill_climb()
print(f'(HC)[Final]\t Route is hc_opt with cost \t{hc_cost}\n')

# simulated annealing solution
sa_opt, sa_cost = tsp.compute_sim_annealing()
print(f'(SA)[Final]\t Route is sa_opt with cost \t{sa_cost}\n')

# local beam search solution
lb_ret[0], lb_opt = tsp.compute_loc_beam_search(4, lb_its=100)
print(f'(LB)[Final]\t Route is sa_opt with cost \t{lb_cost}')
'''