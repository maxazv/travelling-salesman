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


    def get_neighbour_nodes(self, node):
        # return list with j values where (node, j) was not zero and not node
        return [i for i in range(self.size) if self.adj_matrix[node][i] and i != node]


    def successor_states(self, route):
        # simple way of generating successor nodes
        succs = []
        for i in range(len(route)):
            for j in range(i+1, len(route)):
                swap = route[:]     # pass by value
                swap[i], swap[j] = route[j], route[i]
                succs.append(swap)

        return succs


    def calc_route_cost(self, route):
        # sum over values of arr[i][i+1] for i in route. (i, i+1) is an edge from route[i] to route[i+1]
        return sum([self.adj_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])


    def gen_rand_route(self):
        rand_route = list(np.random.permutation(self.size))
        return rand_route, self.calc_route_cost(rand_route)


    def compute_hill_climb(self, variant='naive'):
        # generate array with unique random integers (random initial route)
        route, route_cost = self.gen_rand_route()
        print(f'(HC)[Init]\t Route is route with cost \t{route_cost}')

        min_route = route_cost
        moved = True

        while(moved):
            moved = False
            # generate successor nodes
            successors = self.successor_states(route)

            # find successor node with minimum route cost
            for s in successors:
                route_cost = self.calc_route_cost(s)
                if route_cost < min_route:
                    route = s
                    min_route = route_cost

                    moved = True

        return route, min_route


    def compute_sim_annealing(self, init_temp=500, thr=0.07, d_t=0.075):
        route, route_cost = self.gen_rand_route()
        print(f'(SA)[Init]\t Route is route with cost \t{route_cost}')

        temp = init_temp    # starting temperature

        while temp > thr:
            # schedule
            temp = temp - d_t   # different schedules: a/math.log(1+t), a*temp : for some constant a, counter t

            # generate random successor node by swapping some i, j > 1, and calculate its value
            rand_scc = route[:]
            i, j = random.sample(range(len(route) - 1), 2)
            i, j = i+1, j+1
            rand_scc[i], rand_scc[j] = rand_scc[j], rand_scc[i]

            # compute error (very expensive)
            d_err = self.calc_route_cost(rand_scc) - route_cost

            # accept node if better or if worse by some probability
            if d_err < 0 or math.exp(-d_err/temp) > random.random():
                route = rand_scc
                route_cost = d_err + route_cost

        return route, route_cost


    # <TO DO>
    def compute_loc_beam_search(self, k):
        # generate array with unique random integers (random initial route)
        routes = []
        route_costs = []
        for i in range(k):
            route, route_cost = self.gen_rand_route()
            routes.append(route)
            route_costs.append(route_cost)

        print(f'(HC)[Init]\t Route is route with cost \t{route_cost}')

        min_routes = route_costs
        moved = True

        while(moved):
            moved = False
            # generate successor nodes
            successors = self.successor_states(route)

            # find successor node with minimum route cost
            for s in successors:
                route_cost = self.calc_route_cost(s)
                if route_cost < min_route:
                    route = s
                    min_route = route_cost

                    moved = True

        return route, min_route

    
    def compute_gen_algorithm(self):
        route, route_cost = self.gen_rand_route()