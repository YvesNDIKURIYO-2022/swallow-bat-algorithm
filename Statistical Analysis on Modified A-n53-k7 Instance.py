import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from itertools import combinations
import pulp
import scipy.stats as scipy_stats

# --- Modified A-n53-k7 Dataset ---
coords = [
    (24, 63),  # depot index 0
    (35, 60), (79, 46), (3, 45), (42, 50), (3, 40), (29, 96), (47, 30), (54, 77), (36, 30), (83, 86),
    (30, 6), (55, 29), (13, 2), (1, 19), (98, 1), (75, 10), (39, 23), (62, 91), (96, 9), (27, 87),
    (14, 16), (52, 49), (95, 21), (30, 6), (18, 40), (82, 90), (50, 79), (48, 49), (82, 73), (64, 62),
    (34, 78), (83, 6), (3, 77), (18, 8)
]

demands = [
    0,  # depot
    2, 12, 14, 2, 17, 20, 2, 26, 7, 24, 23, 13, 25, 20, 3, 18, 23, 6, 2, 13,
    22, 3, 6, 7, 1, 18, 18, 10, 2, 9, 10, 8, 30, 16
]

capacity = 100
max_vehicles = 5

# --- Threat Zones ---
threat_zones = [
    {"center": (60, 80), "radius": 3},
    {"center": (60, 20), "radius": 3},
    {"center": (15, 70), "radius": 3},
    {"center": (25, 25), "radius": 3},
    {"center": (80, 60), "radius": 3}
]

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    return any(math.hypot(point[0]-zone['center'][0], point[1]-zone['center'][1]) < zone['radius'] 
              for zone in zones)

def calculate_route_cost(route, coords):
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    return sum(calculate_route_cost(route, coords) for route in routes)

def calculate_threat_penalty(route, coords, zones, penalty=1000, segments=5):
    exposures = 0
    for i in range(len(route)-1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]
        for s in range(segments+1):
            t = s/segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(point, zones):
                exposures += 1
                break
    return exposures, exposures * penalty

def segments_intersect(A, B, C, D):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def count_route_crossings(routes, coords):
    segments = []
    for route in routes:
        for i in range(len(route)-1):
            segments.append((coords[route[i]], coords[route[i+1]]))
    
    crossings = 0
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if segments_intersect(*segments[i], *segments[j]):
                crossings += 1
    return crossings

def decode_routes(permutation, demands, capacity, max_vehicles):
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    routes = []
    route = [0]
    load = 0
    
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route = [0, cust]
            load = demands[cust]
    
    route.append(0)
    routes.append(route)
    
    while len(routes) > max_vehicles:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]
    
    return routes

def validate_solution(routes, demands, capacity, num_customers):
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

def format_table(headers, data):
    col_widths = [len(str(h)) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    col_widths = [w + 2 for w in col_widths]
    
    header_line = "│".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
    separator = "┼".join("─" * w for w in col_widths)
    
    table = []
    table.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")
    table.append("│" + header_line + "│")
    table.append("├" + separator + "┤")
    
    for row in data:
        row_line = "│".join(f"{str(cell):^{w}}" for cell, w in zip(row, col_widths))
        table.append("│" + row_line + "│")
    
    table.append("└" + "┴".join("─" * w for w in col_widths) + "┘")
    
    return "\n".join(table)

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=40, max_iter=300):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        # Optimized parameters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.crossing_penalty = 200
        self.threat_penalty = 1000
        
        # Enhanced adaptive parameters
        self.initial_loudness = 0.8
        self.initial_pulse_rate = 0.2
        self.loudness_decay = 0.97
        self.pulse_rate_increase = 1.05
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []
        self.loudness = [self.initial_loudness] * self.pop_size
        self.pulse_rate = [self.initial_pulse_rate] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2.0
        
        # Enhanced local search
        self.local_search_prob = 0.5
        self.improvement_count = 0

    def _initialize_population(self):
        """Enhanced population initialization"""
        population = []
        
        # 1. ALNS-style solutions (50%)
        for _ in range(int(self.pop_size * 0.5)):
            population.append(self._alns_style_initialization())
        
        # 2. Enhanced savings algorithm (30%)
        for _ in range(int(self.pop_size * 0.3)):
            population.append(self._enhanced_savings_algorithm())
        
        # 3. Threat-aware solutions (20%)
        for _ in range(int(self.pop_size * 0.2)):
            population.append(self._advanced_threat_aware_init())
        
        # Fill remaining spots if needed
        while len(population) < self.pop_size:
            population.append(self._alns_style_initialization())
        
        return population

    def _alns_style_initialization(self):
        """ALNS-inspired initialization"""
        unserved = set(self.customers)
        routes = []
        
        while unserved:
            route = [0]
            load = 0
            current = 0
            
            while unserved and load < self.capacity:
                best_cust = None
                best_score = float('inf')
                
                for cust in unserved:
                    if load + self.demands[cust] <= self.capacity:
                        dist = euclidean(self.coords[current], self.coords[cust])
                        threat_risk = 50 if is_in_threat_zone(self.coords[cust], self.zones) else 0
                        score = dist + threat_risk
                        
                        if score < best_score:
                            best_score = score
                            best_cust = cust
                
                if best_cust is None:
                    break
                    
                route.append(best_cust)
                load += self.demands[best_cust]
                unserved.remove(best_cust)
                current = best_cust
            
            route.append(0)
            routes.append(route)
        
        # Convert to permutation
        solution = []
        for route in routes:
            solution.extend(route[1:-1])
        
        return solution

    def _enhanced_savings_algorithm(self):
        """Improved savings algorithm"""
        routes = [[0, cust, 0] for cust in self.customers]
        
        savings = []
        for i in range(len(self.customers)):
            for j in range(i+1, len(self.customers)):
                cust_i = self.customers[i]
                cust_j = self.customers[j]
                
                saving = (euclidean(self.coords[0], self.coords[cust_i]) + 
                         euclidean(self.coords[0], self.coords[cust_j]) - 
                         euclidean(self.coords[cust_i], self.coords[cust_j]))
                savings.append((saving, cust_i, cust_j))
        
        savings.sort(reverse=True, key=lambda x: x[0])
        
        for saving, cust_i, cust_j in savings:
            route_i = None
            route_j = None
            
            for route in routes:
                if cust_i in route:
                    route_i = route
                if cust_j in route:
                    route_j = route
            
            if (route_i and route_j and route_i != route_j and 
                len(routes) > self.max_vehicles):
                
                total_load = (sum(self.demands[c] for c in route_i[1:-1]) + 
                            sum(self.demands[c] for c in route_j[1:-1]))
                
                if total_load <= self.capacity:
                    if route_i[-2] == cust_i and route_j[1] == cust_j:
                        new_route = route_i[:-1] + route_j[1:]
                        routes.remove(route_i)
                        routes.remove(route_j)
                        routes.append(new_route)
        
        solution = []
        for route in routes:
            solution.extend(route[1:-1])
        
        return solution

    def _advanced_threat_aware_init(self):
        """Threat-aware initialization"""
        high_risk = [c for c in self.customers if is_in_threat_zone(self.coords[c], self.zones)]
        low_risk = [c for c in self.customers if c not in high_risk]
        
        solution = low_risk.copy()
        random.shuffle(solution)
        
        for cust in high_risk:
            best_pos = len(solution)
            best_cost = float('inf')
            
            for pos in range(len(solution) + 1):
                test_sol = solution[:pos] + [cust] + solution[pos:]
                cost, _ = self.fitness(test_sol)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            solution.insert(best_pos, cust)
        
        return solution

    def fitness(self, permutation):
        """Optimized fitness function"""
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            if not routes or len(routes) > self.max_vehicles:
                return float('inf'), []
                
            distance_cost = calculate_total_cost(routes, self.coords)
            
            threat_cost = 0
            for route in routes:
                count, cost = calculate_threat_penalty(route, self.coords, self.zones, self.threat_penalty)
                threat_cost += cost
            
            crossing_cost = count_route_crossings(routes, self.coords) * self.crossing_penalty
            
            total_cost = distance_cost + threat_cost + crossing_cost
            return total_cost, routes
        except Exception as e:
            return float('inf'), []

    def _enhanced_local_search(self, solution):
        """Enhanced local search with multiple operators"""
        best_solution = solution.copy()
        best_cost, _ = self.fitness(solution)
        
        operators = ['swap', 'insert', 'reverse', 'threat_swap']
        
        for _ in range(10):
            new_solution = best_solution.copy()
            operator = random.choice(operators)
            
            if operator == 'swap':
                i, j = random.sample(range(len(new_solution)), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            elif operator == 'insert':
                i = random.randint(0, len(new_solution)-1)
                j = random.randint(0, len(new_solution)-1)
                if i != j:
                    cust = new_solution.pop(i)
                    new_solution.insert(j, cust)
            elif operator == 'reverse':
                i, j = random.sample(range(len(new_solution)), 2)
                if i > j:
                    i, j = j, i
                new_solution[i:j+1] = reversed(new_solution[i:j+1])
            elif operator == 'threat_swap':
                threat_indices = [i for i, cust in enumerate(new_solution) 
                                if is_in_threat_zone(self.coords[cust], self.zones)]
                safe_indices = [i for i, cust in enumerate(new_solution) 
                              if not is_in_threat_zone(self.coords[cust], self.zones)]
                
                if threat_indices and safe_indices:
                    i = random.choice(threat_indices)
                    j = random.choice(safe_indices)
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            new_cost, _ = self.fitness(new_solution)
            
            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost
        
        return best_solution

    def run(self):
        """Optimized SBA process"""
        start_time = time.time()
        
        # Initialize population and evaluate
        solutions = self.population.copy()
        fitness_vals = []
        
        for i, sol in enumerate(solutions):
            fit, routes = self.fitness(sol)
            fitness_vals.append(fit)
            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes
                self.improvement_count += 1

        # Main optimization loop
        for t in range(self.max_iter):
            improvements_this_iter = 0
            
            for i in range(self.pop_size):
                # Frequency-based movement
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = max(1, int(freq * len(self.customers) * 0.3))
                
                if random.random() > self.pulse_rate[i]:
                    # Enhanced search around best solution
                    new_sol = self._enhanced_local_search(self.best_solution)
                else:
                    # Exploration with frequency-based moves
                    new_sol = solutions[i].copy()
                    for _ in range(vel):
                        op = random.choice(['swap', 'insert'])
                        if op == 'swap':
                            i1, i2 = random.sample(range(len(new_sol)), 2)
                            new_sol[i1], new_sol[i2] = new_sol[i2], new_sol[i1]
                        else:
                            i1 = random.randint(0, len(new_sol)-1)
                            i2 = random.randint(0, len(new_sol)-1)
                            if i1 != i2:
                                cust = new_sol.pop(i1)
                                new_sol.insert(i2, cust)
                
                # Apply local search with probability
                if random.random() < self.local_search_prob:
                    new_sol = self._enhanced_local_search(new_sol)
                
                # Evaluate and update
                new_fit, new_routes = self.fitness(new_sol)
                
                # Enhanced acceptance criterion
                if (new_fit < fitness_vals[i] or 
                    (random.random() < self.loudness[i] and new_fit < fitness_vals[i] * 1.1)):
                    
                    solutions[i] = new_sol
                    fitness_vals[i] = new_fit
                    
                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
                        self.improvement_count += 1
                        improvements_this_iter += 1
            
            # Adaptive parameter update
            for i in range(self.pop_size):
                self.loudness[i] *= self.loudness_decay
                self.pulse_rate[i] = min(0.9, self.pulse_rate[i] * self.pulse_rate_increase)
            
            # Population management
            if t % 50 == 0 and improvements_this_iter == 0:
                # Replace worst solution
                worst_idx = np.argmax(fitness_vals)
                solutions[worst_idx] = self._alns_style_initialization()
                fitness_vals[worst_idx], _ = self.fitness(solutions[worst_idx])
            
            self.history.append(self.best_cost)

        # Final intensification
        if self.best_solution:
            final_improved = self._enhanced_local_search(self.best_solution)
            final_cost, final_routes = self.fitness(final_improved)
            if final_cost < self.best_cost:
                self.best_solution = final_improved
                self.best_cost = final_cost
                self.best_routes = final_routes
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'SBA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Bat Algorithm (BA) ---
class BatAlgorithm:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=30, max_iter=200):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.freq_min = 0
        self.freq_max = 2
        self.loudness = 0.5
        self.pulse_rate = 0.5
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(random.sample(self.customers, len(self.customers)))
        return population

    def fitness(self, permutation):
        routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        total_cost = distance_cost + threat_cost + crossing_cost
        return total_cost, routes

    def run(self):
        start_time = time.time()
        
        fitness_vals = []
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            fitness_vals.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                
                new_sol = self.population[i].copy()
                for j in range(len(new_sol)):
                    if random.random() < abs(freq):
                        swap_idx = random.randint(0, len(new_sol)-1)
                        new_sol[j], new_sol[swap_idx] = new_sol[swap_idx], new_sol[j]
                
                if random.random() > self.pulse_rate:
                    new_sol = self._local_search(new_sol)
                
                new_cost, new_routes = self.fitness(new_sol)
                
                if new_cost < fitness_vals[i] and random.random() < self.loudness:
                    self.population[i] = new_sol
                    fitness_vals[i] = new_cost
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
            
            self.history.append(self.best_cost)
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'BA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

    def _local_search(self, solution):
        new_sol = solution.copy()
        i, j = random.sample(range(len(new_sol)), 2)
        new_sol[i], new_sol[j] = new_sol[j], new_sol[i]
        return new_sol

# --- Particle Swarm Optimization (PSO) ---
class PSO:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=30, max_iter=200, w=0.7, c1=1.4, c2=1.4):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        self.velocity = [[0]*len(self.customers) for _ in range(self.pop_size)]
        self.pbest = self.population.copy()
        self.pbest_cost = [float('inf')] * self.pop_size
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(random.sample(self.customers, len(self.customers)))
        return population

    def fitness(self, permutation):
        routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        total_cost = distance_cost + threat_cost + crossing_cost
        return total_cost, routes

    def run(self):
        start_time = time.time()
        
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            self.pbest_cost[i] = cost
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (self.pbest[i][j] - self.population[i][j])
                    social = self.c2 * r2 * (self.best_solution[j] - self.population[i][j])
                    self.velocity[i][j] = self.w * self.velocity[i][j] + cognitive + social
                    
                    if random.random() < abs(self.velocity[i][j]):
                        swap_idx = random.randint(0, len(self.customers)-1)
                        self.population[i][j], self.population[i][swap_idx] = \
                            self.population[i][swap_idx], self.population[i][j]
                
                new_cost, new_routes = self.fitness(self.population[i])
                
                if new_cost < self.pbest_cost[i]:
                    self.pbest[i] = self.population[i].copy()
                    self.pbest_cost[i] = new_cost
                    
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = self.population[i].copy()
                        self.best_routes = new_routes
            
            self.history.append(self.best_cost)
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'PSO',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Harris Hawks Optimization (HHO) ---
class HHO:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=30, max_iter=200):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            population.append(random.sample(self.customers, len(self.customers)))
        return population

    def fitness(self, permutation):
        routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        total_cost = distance_cost + threat_cost + crossing_cost
        return total_cost, routes

    def run(self):
        start_time = time.time()
        
        fitness_vals = []
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            fitness_vals.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            E0 = 2 * random.random() - 1
            E = 2 * E0 * (1 - iteration / self.max_iter)
            
            for i in range(self.pop_size):
                if abs(E) >= 1:
                    # Exploration phase
                    k = random.randint(0, self.pop_size-1)
                    new_sol = []
                    for j in range(len(self.customers)):
                        if random.random() >= 0.5:
                            new_val = (self.population[k][j] - random.random() * 
                                      abs(self.population[k][j] - 2 * random.random() * self.population[i][j]))
                        else:
                            new_val = ((self.best_solution[j] - self.population[i][j]) - 
                                      random.random() * ((self.max_iter - iteration) / self.max_iter) * 
                                      (self.coords[0][0] + random.random() * (self.coords[0][1] - self.coords[0][0])))
                        new_val = max(1, min(len(self.customers), int(new_val)))
                        new_sol.append(new_val)
                    
                    # Ensure all customers are present
                    new_sol = list(dict.fromkeys(new_sol))
                    missing = set(self.customers) - set(new_sol)
                    new_sol.extend(missing)
                    new_sol = new_sol[:len(self.customers)]
                else:
                    # Exploitation phase
                    new_sol = self.population[i].copy()
                    delta = random.random() * abs(self.best_solution[0] - self.population[i][0])
                    
                    if abs(E) >= 0.5:
                        # Soft besiege
                        for j in range(len(new_sol)):
                            new_sol[j] = int(self.best_solution[j] - E * abs(delta))
                    else:
                        # Hard besiege
                        for j in range(len(new_sol)):
                            new_sol[j] = int(self.best_solution[j] - E * abs(delta))
                
                # Ensure valid solution
                new_sol = [max(1, min(len(self.customers), x)) for x in new_sol]
                new_sol = list(dict.fromkeys(new_sol))
                missing = set(self.customers) - set(new_sol)
                new_sol.extend(missing)
                new_sol = new_sol[:len(self.customers)]
                
                new_cost, new_routes = self.fitness(new_sol)
                
                if new_cost < fitness_vals[i]:
                    self.population[i] = new_sol
                    fitness_vals[i] = new_cost
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
            
            self.history.append(self.best_cost)
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'HHO',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Adaptive Large Neighborhood Search (ALNS) ---
class ALNS:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 max_iter=1000, adaptive_period=100):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.max_iter = max_iter
        self.adaptive_period = adaptive_period
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        
        self.destroy_operators = [
            self._random_removal,
            self._worst_removal,
            self._route_removal
        ]
        self.repair_operators = [
            self._greedy_insertion,
            self._regret_insertion
        ]
        
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.repair_weights = [1.0] * len(self.repair_operators)
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def fitness(self, routes):
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        total_cost = distance_cost + threat_cost + crossing_cost
        return total_cost

    def _generate_initial_solution(self):
        unserved = self.customers.copy()
        routes = []
        
        while unserved:
            route = [0]
            load = 0
            
            while unserved and load < self.capacity:
                best_cust = None
                best_dist = float('inf')
                current_pos = self.coords[route[-1]]
                
                for cust in unserved:
                    if load + self.demands[cust] <= self.capacity:
                        dist = euclidean(current_pos, self.coords[cust])
                        if dist < best_dist:
                            best_dist = dist
                            best_cust = cust
                
                if best_cust is None:
                    break
                    
                route.append(best_cust)
                load += self.demands[best_cust]
                unserved.remove(best_cust)
            
            route.append(0)
            routes.append(route)
        
        return routes

    def _random_removal(self, routes, q=5):
        customers_in_routes = []
        for route in routes:
            customers_in_routes.extend(route[1:-1])
        
        if not customers_in_routes:
            return routes, []
        
        q = min(q, len(customers_in_routes))
        removed = random.sample(customers_in_routes, q)
        
        new_routes = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            if len(new_route) > 2:
                new_routes.append(new_route)
        
        return new_routes, removed

    def _worst_removal(self, routes, q=5):
        if not routes:
            return routes, []
        
        customer_costs = {}
        for route in routes:
            for i in range(1, len(route)-1):
                cust = route[i]
                prev_cost = euclidean(self.coords[route[i-1]], self.coords[cust])
                next_cost = euclidean(self.coords[cust], self.coords[route[i+1]])
                direct_cost = euclidean(self.coords[route[i-1]], self.coords[route[i+1]])
                cost_contribution = prev_cost + next_cost - direct_cost
                customer_costs[cust] = cost_contribution
        
        if not customer_costs:
            return routes, []
        
        q = min(q, len(customer_costs))
        worst_customers = sorted(customer_costs.keys(), key=lambda x: customer_costs[x], reverse=True)[:q]
        
        new_routes = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in worst_customers] + [0]
            if len(new_route) > 2:
                new_routes.append(new_route)
        
        return new_routes, worst_customers

    def _route_removal(self, routes, q=3):
        if len(routes) <= 1:
            return routes, []
        
        q = min(q, len(routes))
        routes_to_remove = random.sample(range(len(routes)), q)
        
        removed = []
        new_routes = []
        for i, route in enumerate(routes):
            if i in routes_to_remove:
                removed.extend(route[1:-1])
            else:
                new_routes.append(route)
        
        return new_routes, removed

    def _greedy_insertion(self, routes, removed_customers):
        for cust in removed_customers:
            best_cost = float('inf')
            best_route_idx = -1
            best_position = -1
            
            for r_idx, route in enumerate(routes):
                current_load = sum(self.demands[c] for c in route[1:-1])
                if current_load + self.demands[cust] > self.capacity:
                    continue
                
                for pos in range(1, len(route)):
                    new_route = route[:pos] + [cust] + route[pos:]
                    cost_increase = self._calculate_route_cost_increase(route, cust, pos)
                    
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_route_idx = r_idx
                        best_position = pos
            
            if best_route_idx != -1:
                routes[best_route_idx].insert(best_position, cust)
            else:
                routes.append([0, cust, 0])
        
        return routes

    def _regret_insertion(self, routes, removed_customers):
        for cust in removed_customers:
            insertion_costs = []
            
            for r_idx, route in enumerate(routes):
                current_load = sum(self.demands[c] for c in route[1:-1])
                if current_load + self.demands[cust] > self.capacity:
                    insertion_costs.append((float('inf'), r_idx, -1))
                    continue
                
                best_cost = float('inf')
                best_pos = -1
                
                for pos in range(1, len(route)):
                    cost_increase = self._calculate_route_cost_increase(route, cust, pos)
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_pos = pos
                
                insertion_costs.append((best_cost, r_idx, best_pos))
            
            insertion_costs.append((self._calculate_new_route_cost(cust), len(routes), 1))
            
            insertion_costs.sort(key=lambda x: x[0])
            
            if len(insertion_costs) > 1 and insertion_costs[0][0] < float('inf'):
                cost, r_idx, pos = insertion_costs[0]
                if r_idx == len(routes):
                    routes.append([0, cust, 0])
                else:
                    routes[r_idx].insert(pos, cust)
        
        return routes

    def _calculate_route_cost_increase(self, route, cust, pos):
        if pos < 1 or pos > len(route)-1:
            return float('inf')
        
        prev_cust = route[pos-1]
        next_cust = route[pos] if pos < len(route) else 0
        
        current_cost = euclidean(self.coords[prev_cust], self.coords[next_cust])
        new_cost = (euclidean(self.coords[prev_cust], self.coords[cust]) + 
                   euclidean(self.coords[cust], self.coords[next_cust]))
        
        return new_cost - current_cost

    def _calculate_new_route_cost(self, cust):
        return (euclidean(self.coords[0], self.coords[cust]) + 
                euclidean(self.coords[cust], self.coords[0]))

    def run(self):
        start_time = time.time()
        
        current_routes = self._generate_initial_solution()
        current_cost = self.fitness(current_routes)
        self.best_routes = deepcopy(current_routes)
        self.best_cost = current_cost
        
        scores = [0] * (len(self.destroy_operators) * len(self.repair_operators))
        usage = [0] * (len(self.destroy_operators) * len(self.repair_operators))
        
        for iteration in range(self.max_iter):
            destroy_idx = random.choices(range(len(self.destroy_operators)), weights=self.destroy_weights)[0]
            repair_idx = random.choices(range(len(self.repair_operators)), weights=self.repair_weights)[0]
            
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]
            
            destroyed_routes, removed = destroy_op(deepcopy(current_routes))
            new_routes = repair_op(destroyed_routes, removed)
            
            if not validate_solution(new_routes, self.demands, self.capacity, len(self.demands)):
                new_routes = self._generate_initial_solution()
            
            new_cost = self.fitness(new_routes)
            
            temperature = 1000 * (1 - iteration / self.max_iter)
            accept = (new_cost < current_cost or 
                     random.random() < math.exp((current_cost - new_cost) / max(0.1, temperature)))
            
            if accept:
                current_routes = new_routes
                current_cost = new_cost
                
                if new_cost < self.best_cost:
                    self.best_routes = deepcopy(new_routes)
                    self.best_cost = new_cost
                    scores[destroy_idx * len(self.repair_operators) + repair_idx] += 3
                else:
                    scores[destroy_idx * len(self.repair_operators) + repair_idx] += 2
            else:
                scores[destroy_idx * len(self.repair_operators) + repair_idx] += 1
            
            usage[destroy_idx * len(self.repair_operators) + repair_idx] += 1
            
            if iteration > 0 and iteration % self.adaptive_period == 0:
                for i in range(len(self.destroy_operators)):
                    for j in range(len(self.repair_operators)):
                        idx = i * len(self.repair_operators) + j
                        if usage[idx] > 0:
                            self.destroy_weights[i] = (0.8 * self.destroy_weights[i] + 
                                                      0.2 * scores[idx] / usage[idx])
                            self.repair_weights[j] = (0.8 * self.repair_weights[j] + 
                                                     0.2 * scores[idx] / usage[idx])
                
                scores = [0] * len(scores)
                usage = [0] * len(usage)
            
            self.history.append(self.best_cost)
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'ALNS',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Hybrid Genetic Algorithm (HGA) ---
class HGA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=7,
                 population_size=50, max_iter=200, crossover_rate=0.8, mutation_rate=0.2):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        population = []
        
        for _ in range(self.pop_size - 2):
            population.append(random.sample(self.customers, len(self.customers)))
        
        nn_solution = self._nearest_neighbor()
        population.append(nn_solution)
        
        savings_solution = self._savings_algorithm()
        population.append(savings_solution)
        
        return population

    def _nearest_neighbor(self):
        unvisited = set(self.customers)
        solution = []
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
            solution.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return solution

    def _savings_algorithm(self):
        routes = [[0, cust, 0] for cust in self.customers]
        
        savings = []
        for i in range(len(self.customers)):
            for j in range(i+1, len(self.customers)):
                cust_i = self.customers[i]
                cust_j = self.customers[j]
                
                saving = (euclidean(self.coords[0], self.coords[cust_i]) + 
                         euclidean(self.coords[0], self.coords[cust_j]) - 
                         euclidean(self.coords[cust_i], self.coords[cust_j]))
                savings.append((saving, cust_i, cust_j))
        
        savings.sort(reverse=True, key=lambda x: x[0])
        
        for saving, cust_i, cust_j in savings:
            route_i = None
            route_j = None
            pos_i = -1
            pos_j = -1
            
            for idx, route in enumerate(routes):
                if cust_i in route:
                    route_i = route
                    pos_i = route.index(cust_i)
                if cust_j in route:
                    route_j = route
                    pos_j = route.index(cust_j)
            
            if (route_i is not None and route_j is not None and 
                route_i != route_j and len(routes) > self.max_vehicles):
                
                total_load = (sum(self.demands[c] for c in route_i[1:-1]) + 
                            sum(self.demands[c] for c in route_j[1:-1]))
                
                if total_load <= self.capacity:
                    if pos_i == len(route_i)-2:
                        if pos_j == 1:
                            new_route = route_i[:-1] + route_j[1:]
                            routes.remove(route_i)
                            routes.remove(route_j)
                            routes.append(new_route)
        
        solution = []
        for route in routes:
            solution.extend(route[1:-1])
        
        return solution

    def fitness(self, permutation):
        routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        total_cost = distance_cost + threat_cost + crossing_cost
        return total_cost, routes

    def _selection(self, fitness_vals):
        tournament_size = 3
        tournament = random.sample(range(len(fitness_vals)), tournament_size)
        winner = min(tournament, key=lambda x: fitness_vals[x])
        return winner

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [None] * size
        child2 = [None] * size
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        current_pos1 = end
        current_pos2 = end
        
        for i in range(size):
            pos = (end + i) % size
            if parent2[pos] not in child1:
                child1[current_pos1 % size] = parent2[pos]
                current_pos1 += 1
            
            if parent1[pos] not in child2:
                child2[current_pos2 % size] = parent1[pos]
                current_pos2 += 1
        
        return child1, child2

    def _mutation(self, individual):
        if random.random() > self.mutation_rate:
            return individual
        
        new_individual = individual.copy()
        i, j = random.sample(range(len(new_individual)), 2)
        new_individual[i], new_individual[j] = new_individual[j], new_individual[i]
        return new_individual

    def _local_search(self, individual):
        new_individual = individual.copy()
        
        for _ in range(5):
            i, j = random.sample(range(len(new_individual)), 2)
            if i > j:
                i, j = j, i
            
            new_individual[i:j+1] = reversed(new_individual[i:j+1])
            
            current_cost, _ = self.fitness(individual)
            new_cost, _ = self.fitness(new_individual)
            
            if new_cost < current_cost:
                individual = new_individual.copy()
            else:
                new_individual = individual.copy()
        
        return individual

    def run(self):
        start_time = time.time()
        
        fitness_vals = []
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            fitness_vals.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            new_population = []
            
            new_population.append(self.best_solution.copy())
            
            while len(new_population) < self.pop_size:
                parent1_idx = self._selection(fitness_vals)
                parent2_idx = self._selection(fitness_vals)
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                
                if random.random() < 0.3:
                    child1 = self._local_search(child1)
                if random.random() < 0.3:
                    child2 = self._local_search(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
            
            fitness_vals = []
            for i in range(self.pop_size):
                cost, routes = self.fitness(self.population[i])
                fitness_vals.append(cost)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = self.population[i].copy()
                    self.best_routes = routes
            
            self.history.append(self.best_cost)
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'HGA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Statistical Analysis Functions ---
def run_multiple_trials(algorithm_class, coords, demands, capacity, threat_zones, max_vehicles, num_runs=30):
    costs = []
    times = []
    valid_count = 0
    best_solution = None
    best_cost = float('inf')
    
    print(f"Running {num_runs} trials for {algorithm_class.__name__}...")
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}", end="\r")
        
        random.seed(run)
        np.random.seed(run)
        
        algorithm = algorithm_class(coords, demands, capacity, threat_zones, max_vehicles)
        result = algorithm.run()
        
        costs.append(result['cost'])
        times.append(result['time'])
        
        if result['valid']:
            valid_count += 1
        
        if result['cost'] < best_cost and result['valid']:
            best_cost = result['cost']
            best_solution = result
    
    print()
    
    costs_array = np.array(costs)
    times_array = np.array(times)
    
    stats = {
        'algorithm': algorithm_class.__name__,
        'mean_cost': np.mean(costs_array),
        'std_cost': np.std(costs_array),
        'best_cost': np.min(costs_array),
        'worst_cost': np.max(costs_array),
        'cv_cost': (np.std(costs_array) / np.mean(costs_array)) * 100 if np.mean(costs_array) > 0 else 0,
        'mean_time': np.mean(times_array),
        'success_rate': (valid_count / num_runs) * 100,
        'all_costs': costs_array,
        'all_times': times_array,
        'best_solution': best_solution
    }
    
    return stats

def statistical_comparison(algorithms, coords, demands, capacity, threat_zones, max_vehicles, num_runs=30):
    print("=" * 80)
    print("STATISTICAL COMPARISON (Multiple Runs)")
    print("=" * 80)
    print(f"Number of runs per algorithm: {num_runs}")
    print()
    
    all_stats = []
    
    for alg_class in algorithms:
        stats = run_multiple_trials(alg_class, coords, demands, capacity, threat_zones, max_vehicles, num_runs)
        all_stats.append(stats)
    
    headers = ["Algorithm", "Mean Cost", "Std Dev", "Best", "Worst", "CV (%)", "Mean Time (s)", "Success Rate"]
    table_data = []
    
    for stats in all_stats:
        table_data.append([
            stats['algorithm'],
            f"{stats['mean_cost']:.2f}",
            f"{stats['std_cost']:.2f}",
            f"{stats['best_cost']:.2f}",
            f"{stats['worst_cost']:.2f}",
            f"{stats['cv_cost']:.2f}",
            f"{stats['mean_time']:.2f}",
            f"{stats['success_rate']:.1f}%"
        ])
    
    print(format_table(headers, table_data))
    
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    if len(all_stats) > 1:
        cost_arrays = [stats['all_costs'] for stats in all_stats]
        algorithm_names = [stats['algorithm'] for stats in all_stats]
        
        f_stat, p_value = scipy_stats.f_oneway(*cost_arrays)
        print(f"One-way ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("→ Statistically significant differences exist between algorithms (p < 0.05)")
        else:
            print("→ No statistically significant differences between algorithms (p ≥ 0.05)")
        
        print(f"\nPairwise t-tests (Bonferroni corrected):")
        alpha = 0.05
        num_comparisons = len(algorithm_names) * (len(algorithm_names) - 1) // 2
        corrected_alpha = alpha / num_comparisons
        
        for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
                t_stat, p_val = scipy_stats.ttest_ind(cost_arrays[i], cost_arrays[j])
                significance = "✓" if p_val < corrected_alpha else "✗"
                print(f"  {algorithm_names[i]} vs {algorithm_names[j]}: p = {p_val:.4f} {significance}")
    
    best_alg = min(all_stats, key=lambda x: x['mean_cost'])
    print(f"\n🏆 BEST OVERALL ALGORITHM: {best_alg['algorithm']}")
    print(f"   Mean Cost: {best_alg['mean_cost']:.2f} ± {best_alg['std_cost']:.2f}")
    print(f"   Success Rate: {best_alg['success_rate']:.1f}%")
    print(f"   Computation Time: {best_alg['mean_time']:.2f}s")
    
    return all_stats

def plot_statistical_results(all_stats):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    algorithm_names = [stats['algorithm'] for stats in all_stats]
    cost_data = [stats['all_costs'] for stats in all_stats]
    
    ax1.boxplot(cost_data, labels=algorithm_names)
    ax1.set_title('Cost Distribution Across Algorithms')
    ax1.set_ylabel('Total Cost')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    success_rates = [stats['success_rate'] for stats in all_stats]
    bars = ax2.bar(algorithm_names, success_rates, color='lightgreen', alpha=0.7)
    ax2.set_title('Algorithm Success Rates')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    mean_times = [stats['mean_time'] for stats in all_stats]
    time_std = [np.std(stats['all_times']) for stats in all_stats]
    
    bars = ax3.bar(algorithm_names, mean_times, yerr=time_std, capsize=5, 
                   color='lightcoral', alpha=0.7)
    ax3.set_title('Average Computation Time')
    ax3.set_ylabel('Time (seconds)')
    ax3.tick_params(axis='x', rotation=45)
    
    cv_values = [stats['cv_cost'] for stats in all_stats]
    bars = ax4.bar(algorithm_names, cv_values, color='gold', alpha=0.7)
    ax4.set_title('Coefficient of Variation (Stability)')
    ax4.set_ylabel('CV (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='10% CV threshold')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for stats in all_stats:
        if stats['best_solution'] and 'convergence' in stats['best_solution']:
            convergence = stats['best_solution']['convergence']
            plt.plot(convergence, label=stats['algorithm'], linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('Convergence Characteristics (Best Run)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Main Experiment ---
def main():
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF OPTIMIZATION ALGORITHMS")
    print("Vehicle Routing Problem with Threat Zones")
    print("A-n53-k7 Dataset (53 nodes, 7 vehicles)")
    print("=" * 80)
    
    total_demand = sum(demands[1:])
    min_vehicles_required = math.ceil(total_demand / capacity)
    print(f"Dataset Statistics:")
    print(f"- Total customers: {len(coords) - 1}")
    print(f"- Total demand: {total_demand}")
    print(f"- Vehicle capacity: {capacity}")
    print(f"- Minimum vehicles required: {min_vehicles_required}")
    print(f"- Maximum vehicles allowed: {max_vehicles}")
    print()
    
    algorithms = [SBA, ALNS, HGA, BatAlgorithm, PSO, HHO]
    
    all_stats = statistical_comparison(
        algorithms, coords, demands, capacity, threat_zones, max_vehicles, num_runs=30
    )
    
    print("\nGenerating statistical visualizations...")
    plot_statistical_results(all_stats)
    
    best_overall = min(all_stats, key=lambda x: x['mean_cost'])
    best_solution = best_overall['best_solution']
    
    if best_solution:
        print("\n" + "=" * 80)
        print("BEST SOLUTION DETAILS")
        print("=" * 80)
        print(f"Algorithm: {best_solution['name']}")
        print(f"Cost: {best_solution['cost']:.2f}")
        print(f"Valid: {best_solution['valid']}")
        print(f"Computation Time: {best_solution['time']:.2f}s")
        
        routes = best_solution['routes']
        total_distance = calculate_total_cost(routes, coords)
        threat_exp = sum(calculate_threat_penalty(r, coords, threat_zones, 1)[0] for r in routes)
        crossings = count_route_crossings(routes, coords)
        
        print(f"\nDetailed Metrics:")
        print(f"- Total Distance: {total_distance:.2f}")
        print(f"- Threat Exposures: {threat_exp}")
        print(f"- Route Crossings: {crossings}")
        print(f"- Number of Vehicles: {len(routes)}")
        
        print(f"\nRoute Statistics:")
        for i, route in enumerate(routes):
            distance = calculate_route_cost(route, coords)
            load = sum(demands[node] for node in route[1:-1])
            customers = len(route) - 2
            print(f"  Vehicle {i+1}: {customers} customers, Load {load}/{capacity}, Distance {distance:.2f}")

if __name__ == "__main__":
    main()