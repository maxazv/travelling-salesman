import time
from travelling_salesman import *


# travelling salesman problem initialisation
tsp = TSP(15, 1000)
tsp.calc_points_dist()
#tsp.init_rand_table(1000, 0.001)
#print(f'Adjacency Matrix:\n{tsp.adj_matrix}\n')


def algorithm_avg(its, alg=0):
    avg_cost, avg_time = 0, 0
    for i in range(its):
        # measure time for each iteration
        if alg == 0:
            start = time.time()
            _, c_cost = tsp.compute_hill_climb()
            end = time.time()
        
        elif alg == 1:
            start = time.time()
            _, c_cost = tsp.compute_sim_annealing()
            end = time.time()

        else:
            start = time.time()
            c_cost = tsp.compute_loc_beam_search(4, lb_its=40)
            end = time.time()
            c_cost = c_cost[0]

        avg_time += (end - start)
        avg_cost += c_cost

    return avg_cost/its, avg_time/its




# algorithm measurements
'''
msr_iters = 20
print(algorithm_avg(msr_iters, alg=0))
print(algorithm_avg(msr_iters, alg=1))
print(algorithm_avg(msr_iters, alg=2))
'''

# hill climbing solution
hc_opt, hc_cost = tsp.compute_hill_climb()
print(f'(HC)[Final]\t Route is hc_opt with cost \t{hc_cost}\n')

# simulated annealing solution
sa_opt, sa_cost = tsp.compute_sim_annealing()
print(f'(SA)[Final]\t Route is sa_opt with cost \t{sa_cost}\n')

# local beam search solution
lb_ret = tsp.compute_loc_beam_search(4, lb_its=100)
print(f'(LB)[Final]\t Route is sa_opt with cost \t{lb_ret[0]}')


# plot routes
tsp.tsp_plot()
tsp.tsp_plot(hc_opt)
tsp.tsp_plot(sa_opt)
tsp.tsp_plot(lb_ret[1])