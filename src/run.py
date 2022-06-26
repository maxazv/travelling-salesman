import time
from travelling_salesman import *


# travelling salesman problem initialisation
num_pnts = 100               # default: 25
tsp = TSP(num_pnts, 2200)
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
            c_cost = tsp.compute_loc_beam_search(4, lb_its=100)  # 3, 75
            end = time.time()
            c_cost = c_cost[0]

        avg_time += (end - start)
        avg_cost += c_cost

    return avg_cost/its, avg_time/its


def tsp_plot(tsp, order=None):
        pnt_x, pnt_y = [], []

        if order is None:
            pnt_x = tsp.points[:, 0]
            pnt_y = tsp.points[:, 1]
        
        else:
            for i in order:
                pnt_x.append(tsp.points[i][0])
                pnt_y.append(tsp.points[i][1])

        plt.scatter(pnt_x, pnt_y)
        plt.scatter(pnt_x[0], pnt_y[0], color='red')    # starting point
        plt.plot(pnt_x, pnt_y, color='red')


        plt.show()



# algorithm measurements

msr_iters = 1
print(algorithm_avg(msr_iters, alg=0))
print(algorithm_avg(msr_iters, alg=1))
print(algorithm_avg(msr_iters, alg=2))


# hill climbing solution
s_hc = time.time()
hc_opt, hc_cost = tsp.compute_hill_climb()
e_hc = time.time()
print(f'(HC)[Final]\t Route is hc_opt with cost \t{hc_cost}\n')

# simulated annealing solution
s_sa = time.time()
sa_opt, sa_cost = tsp.compute_sim_annealing()
e_sa = time.time()
print(f'(SA)[Final]\t Route is sa_opt with cost \t{sa_cost}\n')

# local beam search solution
s_lb = time.time()
lb_ret = tsp.compute_loc_beam_search(8, lb_its=100)
e_lb = time.time()
print(f'(LB)[Final]\t Route is lb_opt with cost \t{lb_ret[0]}')


# plot data
rand_route, rand_cost = tsp.gen_rand_route()
rand_cost = round(rand_cost, 2)
plt.title(f'Random Route for {num_pnts} Points\n')
plt.figtext(.5, .9, f'path cost: {rand_cost}, time: 0s', ha="center")
tsp_plot(tsp, rand_route)

hc_time = round(e_hc - s_hc, 4)
hc_cost = round(hc_cost, 2)
plt.title(f'Hill Climbing for {num_pnts} Points\n')
plt.figtext(.5, .9, f'path cost: {hc_cost}, time: {hc_time}s', ha="center")
tsp_plot(tsp, order=hc_opt)

sa_time = round(e_sa - s_sa, 4)
sa_cost = round(sa_cost, 2)
plt.title(f'Simulated Annealing for {num_pnts} Points\n')
plt.figtext(.5, .9, f'path cost: {sa_cost}, time: {sa_time}s', ha="center")
tsp_plot(tsp, order=sa_opt)

lb_time = round(e_lb - s_lb, 4)
lb_cost = round(lb_ret[0], 2)
plt.title(f'Local Beam Search for {num_pnts} Points\n')
plt.figtext(.5, .9, f'path cost: {lb_cost}, time: {lb_time}s', ha="center")
tsp_plot(tsp, lb_ret[1])
