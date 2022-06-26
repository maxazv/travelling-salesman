import numpy as np
import random
import math
import heapq
from matplotlib import pyplot as plt


class TSP:
    def __init__(self, size, max_val):
        self.size = size
        self.points = np.random.rand(self.size, 2) * max_val
        self.adj_matrix = np.zeros(shape=(self.size, self.size))

    
    def calc_points_dist(self):
        for i in range(self.size):
            p_1 = self.points[i]
            for j in range(i+1, self.size):
                p_2 = self.points[j]
                dist = math.sqrt(math.pow((p_1[0] - p_2[0]), 2) + math.pow((p_1[1] - p_2[1]), 2))

                self.adj_matrix[i][j] = dist
                self.adj_matrix[j][i] = dist


    def init_rand_table(self, max_value, null_freq):
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


    def successor_states(self, route, dist_old):    # O(n^2)
        # simple way of generating successor nodes
        succs = []
        for i in range(0, len(route)-2):        # O( (n^2-n)/2 - (n+1) ) or O(n^2)
            for j in range(i+1, len(route)-1):
                swap = route[:]     # pass by value
                s_i, s_j = i+1, j+1

                dist_aft = self.swap_dist_cost(swap, s_i, s_j)  # O(1)

                # swap nodes
                swap[s_i], swap[s_j] = route[s_j], route[s_i]

                dist_bef = self.swap_dist_cost(swap, s_i, s_j)
                new_cost = dist_old - dist_aft + dist_bef

                succs.append((new_cost, swap))

        heapq.heapify(succs)    # O(n)
        return succs


    def calc_route_cost(self, route):   # O(n)
        # sum over values of arr[i][i+1] for i in route. (i, i+1) is an edge from route[i] to route[i+1]
        return sum([self.adj_matrix[route[i]][route[i+1]] for i in range(len(route)-1)])


    def swap_dist_cost(self, route, i, j):  # O(1)
        p_1 = min(i, j)     # smallest index of swapped elems
        p_2 = max(i, j)     # biggest index of swapped elems

        dist = 0

        dist += self.adj_matrix[route[p_2-1]][route[p_2]]
        if p_2 < self.size-1:
            dist += self.adj_matrix[route[p_2]][route[p_2+1]]

        if p_2-1 != p_1:    # otherwise we calculate this distance twice
            dist += self.adj_matrix[route[p_1]][route[p_1+1]]

        dist += self.adj_matrix[route[p_1-1]][route[p_1]]    # always greater than 0

        return dist


    def gen_rand_route(self):
        rand_route = list(np.random.permutation(self.size))
        return rand_route, self.calc_route_cost(rand_route)


    # O(n^2 * its)
    def compute_hill_climb(self, variant='naive', hc_iter=1):
        best_route = None
        best_route_cost = float("inf")

        for i in range(hc_iter):
            # generate array with unique random integers (random initial route)
            route, route_cost = self.gen_rand_route()

            min_route_cost = route_cost
            moved = True

            # cnt = 0
            # msr_x = [cnt]
            # msr_y = [min_route_cost]
            while moved:
                #cnt += 1

                moved = False
                # generate successor nodes
                successors = self.successor_states(route, min_route_cost)   # O(n^2)
                # find successor node with minimum route cost
                s_cost, s = successors[0]
                if s_cost < min_route_cost: moved = True
                min_route_cost, route = s_cost, s

                # msr_x.append(cnt)
                # msr_y.append(min_route_cost)

            # compute best route among multiple hill climb attempts (random-restart hill climbing)
            if min_route_cost < best_route_cost:
                best_route = route
                best_route_cost = min_route_cost

        return best_route, best_route_cost#, msr_x, msr_y


    def swap_points(self, route, route_cost):
        rand_scc = route[:]
        i, j = random.sample(range(len(route) - 1), 2)  # choose two random indices two swap
        i, j = i+1, j+1

        # used a few lines ahead
        dist_aft = self.swap_dist_cost(rand_scc, i, j)

        rand_scc[i], rand_scc[j] = rand_scc[j], rand_scc[i]

        # compute error efficiently by only checking swap distances
        dist_bef = self.swap_dist_cost(rand_scc, i, j)
        new_cost = route_cost - dist_aft + dist_bef
        d_err = new_cost - route_cost

        return rand_scc, new_cost


    # O(its)
    def compute_sim_annealing(self, init_temp=500, thr=0.07, d_t=0.0075):    # default: 500, 0.07, 0.075
        route, route_cost = self.gen_rand_route()

        temp = init_temp    # starting temperature

        # cnt = 0
        # msr_x = [cnt]
        # msr_y = [route_cost]
        while temp > thr:
            #cnt += 1

            # schedule
            temp = temp - d_t   # different schedules: a/math.log(1+t), a*temp : for some constant a, counter t

            # generate random successor node by swapping some i, j > 1, and calculate its value
            rand_scc, new_cost = self.swap_points(route, route_cost)
            d_err = new_cost - route_cost

            # accept node if better or if worse by some probability
            if d_err < 0 or math.exp(-d_err/temp) > random.random():
                route = rand_scc
                route_cost = new_cost

            # msr_x.append(cnt)
            # msr_y.append(route_cost)

        return route, route_cost#, msr_x, msr_y


    # O(k*n^2 * its)
    def compute_loc_beam_search(self, k, lb_its=100):
        # generate start positions with random permutation of cities (random initial routes)
        curr_routes = []
        for i in range(k):
            route, route_cost = self.gen_rand_route()
            curr_routes.append((route_cost, route))

        # tmp = lb_its
        # msr_x = [0]
        # msr_y = [curr_routes[0][0]]
        while lb_its > 0:   # O(k*n^2)
            all_succs = []

            # add all successor nodes to a list
            for (cost, route) in curr_routes:
                # generate successor nodes
                successors = self.successor_states(route, cost)             # O(n^2)
                all_succs += [heapq.heappop(successors) for _ in range(k)]  # O(log(m)) = O(k*log(m)) for m heap size

            # find successor node with minimum route cost with heap
            heapq.heapify(all_succs)    # O(k^2) but k is often small
            for i in range(k):          # O(k*log(k^2)) = O(k^2)
                curr_routes.append(heapq.heappop(all_succs))

            curr_routes = curr_routes[-k:]

            # msr_x.append(tmp-lb_its)
            # msr_y.append(curr_routes[0][0])
            lb_its -= 1

        return curr_routes[0]#, msr_x, msr_y

    
    def crossover(self, indiv_1, indiv_2):
        crossover_pnt = random.randint(1, self.size-1)

        offspring_1 = indiv_1[:crossover_pnt]
        seen = set(offspring_1)

        for genome in indiv_2:      # no duplicate points are to be visited
            if genome not in seen:
                offspring_1.append(genome)

        return offspring_1


    def mutate(self, indiv, swap_rot_prob):
        mutated = indiv

        if random.random() > swap_rot_prob:
            # swap two indices like in simulated annealing (swap mutation)
            cost = self.calc_route_cost(indiv)
            mutated, cost = self.swap_points(indiv, cost)

        else:
            # rotate a sequence of our genome (rotation mutation)
            i, j = random.sample(range(len(mutated) - 1), 2)
            i, j = i+1, j+1

            start = min(i, j)
            end = max(i, j)

            # calculate cost in sections
            bef_mut_cost = self.calc_route_cost(mutated[:start])

            rot_seq = mutated[start:end][::-1]
            mut_cost = self.calc_route_cost(rot_seq)

            aft_mut_cost = self.calc_route_cost(mutated[end:])

            cost = bef_mut_cost + mut_cost + aft_mut_cost

            mutated[start:end] = mutated[start:end][::-1]

        return mutated, cost

    
    def compute_gen_algorithm(self, pop_size, mutation_prob=.05, swap_rot_prob=.5, ga_its=5000):
        curr_pop = []
        total_fitness = 0
        for i in range(pop_size):
            indiv, cost = self.gen_rand_route()
            curr_pop.append((cost, indiv))

            total_fitness += cost

        while ga_its > 0:   # O(pop_size * x)
            next_pop = []

            # selection method with probability of selection being increasing function of fitness
            # based on `biased roulette wheel`
            inv_prop = [total_fitness/cost for (cost, _) in curr_pop]
            inv_prop_sum = sum(inv_prop)
            norm_inv_prop = [inv_p/inv_prop_sum for inv_p in inv_prop]

            accum_prop = []
            accum_total = 0
            for prop in norm_inv_prop:
                accum_total += prop
                accum_prop.append(accum_total)

            # select `pop_size` amount of parents
            parents = []
            for _ in range(pop_size):
                rand_select = random.random()

                for i, val in enumerate(accum_prop):
                    if val >= rand_select:
                        parents.append(curr_pop[i][1])

            # crossover between parents and mutation
            for i in range(pop_size-2):
                offspring_a = self.crossover(parents[i], parents[i+1])
                offspring_b = self.crossover(parents[i+1], parents[i])

                # introduce mutation into the next population inspired by annealing
                mutated_a, mutated_b = offspring_a, offspring_b
                cost_a, cost_b = 0, 0
                if random.random() < mutation_prob:
                    mutated_a, cost_a = self.mutate(offspring_a, swap_rot_prob)
                else:
                    cost_a = self.calc_route_cost(mutated_a)

                if random.random() < mutation_prob:
                    mutated_b, cost_b = self.mutate(offspring_b, swap_rot_prob)
                else:
                    cost_b = self.calc_route_cost(mutated_b)

                curr_pop.append((cost_a, mutated_a))
                curr_pop.append((cost_b, mutated_b))
                
            # select best individuals of total population (including parents)
            heapq.heapify(curr_pop)
            for _ in range(pop_size):
                next_pop.append(heapq.heappop(curr_pop))

            curr_pop = next_pop
            ga_its -= 1

        return curr_pop[0]