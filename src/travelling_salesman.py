import numpy as np
import random
import math


class TSP:
    def __init__(self, size):
        self.size = size
        self.adj_matrix = None


    def init_table(self, max_value, null_freq):
        # create a random matrix with random edge weights
        self.adj_matrix = np.random.randint(low=1, high=max_value, size=(self.size, self.size))
        
        # increase weights of random connections extremely
        inf_val = max_value*(10**4)
        mask = np.random.choice([0, 1], size=self.adj_matrix.shape, p=((1 - null_freq), null_freq)).astype(bool)
        zeros = np.random.rand(self.size, self.size)*inf_val
        self.adj_matrix[mask] = zeros[mask]

        # remove connections to oneself
        np.fill_diagonal(self.adj_matrix, 0)

        # symmetrize matrix
        self.adj_matrix = self.adj_matrix + self.adj_matrix.T - np.diag(self.adj_matrix.diagonal())


    def tsp_plot(self):
        pass


    def successor_states(self, route, dist_old):
        # simple way of generating successor nodes
        succs = []
        dist_new = []
        for i in range(0, len(route)-2):
            for j in range(i+1, len(route)-1):
                swap = route[:]     # pass by value
                s_i, s_j = i+1, j+1

                dist_aft = self.swap_dist_cost(swap, s_i, s_j)

                # swap nodes
                swap[s_i], swap[s_j] = route[s_j], route[s_i]

                dist_bef = self.swap_dist_cost(swap, s_i, s_j)
                new_cost = dist_old - dist_aft + dist_bef
                d_err = new_cost - dist_old

                test = self.calc_route_cost(swap)
                #print("A", test, dist_old - dist_aft + dist_bef)

                succs.append(swap)
                dist_new.append(new_cost)

        return succs, dist_new


    def calc_route_cost(self, route):
        # sum over values of arr[i][i+1] for i in route. (i, i+1) is an edge from route[i] to route[i+1]
        return sum([self.adj_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])


    def swap_dist_cost(self, route, i, j):
        p_1 = min(i, j)     # smallest index of swapped elems
        p_2 = max(i, j)     # biggest index of swapped elems

        dist = []

        dist.append(self.adj_matrix[route[p_2-1]][route[p_2]])
        if p_2 < self.size-1:
            dist.append(self.adj_matrix[route[p_2]][route[p_2+1]])

        if p_2-1 != p_1:    # otherwise we calculate this distance twice
            dist.append(self.adj_matrix[route[p_1]][route[p_1+1]])

        dist.append(self.adj_matrix[route[p_1-1]][route[p_1]])    # always greater than 0

        return sum(dist)


    def gen_rand_route(self):
        rand_route = list(np.random.permutation(self.size))
        return rand_route, self.calc_route_cost(rand_route)


    def compute_hill_climb(self, variant='naive', hc_iter=1):
        best_route = None
        best_route_cost = float("inf")

        for i in range(hc_iter):
            # generate array with unique random integers (random initial route)
            route, route_cost = self.gen_rand_route()
            #print(f'(HC)[Init]\t Route is route with cost \t{route_cost}')

            min_route = route_cost
            moved = True

            while moved:
                moved = False
                # generate successor nodes
                successors, succ_costs = self.successor_states(route, min_route)

                # find successor node with minimum route cost
                for i, s in enumerate(successors):
                    if succ_costs[i] < min_route:
                        route = s
                        min_route = succ_costs[i]

                        moved = True

            # compute best route among multiple hill climb attempts (random-restart hill climbing)
            if min_route < best_route_cost:
                best_route = route
                best_route_cost = min_route

        return best_route, best_route_cost


    def compute_sim_annealing(self, init_temp=500, thr=0.07, d_t=0.075):
        route, route_cost = self.gen_rand_route()
        #print(f'(SA)[Init]\t Route is route with cost \t{route_cost}')

        temp = init_temp    # starting temperature

        while temp > thr:
            # schedule
            temp = temp - d_t   # different schedules: a/math.log(1+t), a*temp : for some constant a, counter t

            # generate random successor node by swapping some i, j > 1, and calculate its value
            rand_scc = route[:]
            i, j = random.sample(range(len(route) - 1), 2)
            i, j = i+1, j+1

            # used a few lines ahead
            dist_aft = self.swap_dist_cost(rand_scc, i, j)

            rand_scc[i], rand_scc[j] = rand_scc[j], rand_scc[i]


            # compute error efficiently by only checking swap distances
            dist_bef = self.swap_dist_cost(rand_scc, i, j)
            new_cost = route_cost + dist_aft - dist_bef
            d_err = -(new_cost - route_cost)
            #d_err_bad = self.calc_route_cost(rand_scc) - route_cost 

            # accept node if better or if worse by some probability
            if d_err < 0 or math.exp(-d_err/temp) > random.random():
                route = rand_scc
                route_cost = d_err + route_cost

        return route, route_cost


    def compute_loc_beam_search(self, k, lb_its=100):
        # generate start positions with random permutation of cities (random initial routes)
        curr_routes = []
        for i in range(k):
            route, route_cost = self.gen_rand_route()
            #print(f'(LB)[Init]\t Route {i} is route with cost \t{route_cost}')

            curr_routes.append((route_cost, route))

        moved = True
        cnt = 0
        while moved and cnt < lb_its:

            moved = True
            k_best_routes = []
            routes_arr = [i[1] for i in curr_routes]

            # find k best successor nodes
            for i, route in enumerate(routes_arr):
                # generate successor nodes
                successors, succ_costs = self.successor_states(route, curr_routes[i][0])   # ii

                # find successor node with minimum route cost
                for i, s in enumerate(successors):
                    #route_cost = self.calc_route_cost(s)
                    route_cost = succ_costs[i]
                    
                    # check if current node is better than the highest val of the best
                    k_best_routes.append((route_cost, s))
                    if len(k_best_routes) > k:
                        k_best_routes.remove(max(k_best_routes))
            
            curr_routes = k_best_routes[:]
            cnt += 1

        min_key = min(curr_routes, key=lambda t : t[0])
        return min_key

    
    def compute_gen_algorithm(self):
        route, route_cost = self.gen_rand_route()