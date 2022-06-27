from travelling_salesman import *
import time
import matplotlib.pyplot as plt


# travelling salesman problem initialisation
num_pnts = 100
tsp = TSP(num_pnts, 2200)
tsp.calc_points_dist()
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

        elif alg == 2:
            start = time.time()
            c_cost = tsp.compute_loc_beam_search(3, lb_its=200)  # 4, 100
            end = time.time()
            c_cost = c_cost[0]

        else:
            start = time.time()
            c_cost = tsp.compute_gen_algorithm(40, ga_its=2000) # 20, 1000
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
# print(algorithm_avg(msr_iters, alg=1))
# print(algorithm_avg(msr_iters, alg=2))
# print(algorithm_avg(msr_iters, alg=3))

'''
# hill climbing solution
s_hc = time.time()
hc_opt, hc_cost = tsp.compute_hill_climb()
e_hc = time.time()
print(f'(HC)[Final]\t Route is hc_opt with cost \t{hc_cost}\n')

# simulated annealing solution
s_sa = time.time()
sa_opt, sa_cost= tsp.compute_sim_annealing()
e_sa = time.time()
print(f'(SA)[Final]\t Route is sa_opt with cost \t{sa_cost}\n')

# local beam search solution
s_lb = time.time()
lb_ret = tsp.compute_loc_beam_search(4, lb_its=150)
e_lb = time.time()
print(f'(LB)[Final]\t Route is lb_opt with cost \t{lb_ret[0]}')

# genetic algorithm search solution
s_ga = time.time()
ga_ret = tsp.compute_gen_algorithm(40, ga_its=2000)
e_ga = time.time()
print(f'(GA)[Final]\t Route is ga_ret[1] with cost \t{ga_ret[0]}')


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

ga_time = round(e_ga - s_ga, 4)
ga_cost = round(ga_ret[0], 2)
plt.title(f'Genetic Algorithm for {num_pnts} Points\n')
plt.figtext(.5, .9, f'path cost: {ga_cost}, time: {ga_time}s', ha="center")
tsp_plot(tsp, ga_ret[1])


# iteration measure
plt.title(f'Hill Climbing')
plt.plot(hc_x, hc_y, marker='|', markevery=1.25)
plt.xlabel("Iterations")
plt.ylabel("Path Cost")
plt.show()

plt.title(f'Simulated Annealing')
plt.plot(sa_x, sa_y, marker='|', markevery=50000)
plt.xlabel("Iterations")
plt.ylabel("Path Cost")
plt.show()

plt.title(f'Local Beam Search')
plt.plot(lb_x, lb_y, marker='|', markevery=15)
plt.xlabel("Iterations")
plt.ylabel("Path Cost")
plt.show()

plt.title(f'Genetic Algorithm')
plt.plot(ga_x, ga_y, marker='|', markevery=100)
plt.xlabel("Iterations")
plt.ylabel("Path Cost")
plt.show()
'''