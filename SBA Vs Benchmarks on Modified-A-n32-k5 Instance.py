import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from itertools import combinations

# ---Modified A-n32-k5 Dataset ---
coords = [
    (82, 76),  # depot (index 0)
    (96, 44), (50, 5), (49, 8), (13, 7), (29, 89),  # Nodes 2-6
    (58, 30), (84, 39), (14, 24), (2, 39), (3, 82),  # Nodes 7-11
    (5, 74), (61, 50), (50, 30), (13, 40), (90, 60), # Nodes 12-16
    (91, 90), (25, 17), (67, 64), (70, 14), (36, 82), # Nodes 17-21
    (41, 94)   # (Index 21)
]

demands = [
    0,   # depot
    19, 21, 6, 19, 7,    # Nodes 2-6
    12, 16, 6, 16, 8,     # Nodes 7-11
    14, 21, 16, 3, 22,    # Nodes 12-16
    18, 19, 1, 24, 8,     # Nodes 17-21
    5    #  (Index 21)
]

capacity = 100
max_vehicles = 3

# --- Your Threat Zones ---
threat_zones = [
    {"center": (50, 50), "radius": 3},
    {"center": (60, 80), "radius": 3},
    {"center": (40, 15), "radius": 3},
    {"center": (20, 80), "radius": 3},
    {"center": (90, 30), "radius": 3}
]

# --- Utility Functions ---
def euclidean(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_threat_zone(point, zones):
    """Check if a point is inside any threat zone"""
    return any(math.hypot(point[0]-zone['center'][0], point[1]-zone['center'][1]) < zone['radius'] 
              for zone in zones)

def calculate_route_cost(route, coords):
    """Calculate total distance of a route"""
    return sum(euclidean(coords[route[i]], coords[route[i+1]]) for i in range(len(route)-1))

def calculate_total_cost(routes, coords):
    """Calculate total distance of all routes"""
    return sum(calculate_route_cost(route, coords) for route in routes)

def calculate_threat_penalty(route, coords, zones, penalty=1000, segments=5):
    """Calculate threat exposure count and penalty with multiple segment checks"""
    exposures = 0
    for i in range(len(route)-1):
        p1 = coords[route[i]]
        p2 = coords[route[i+1]]
        # Check multiple points along the segment
        for s in range(segments+1):
            t = s/segments
            point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
            if is_in_threat_zone(point, zones):
                exposures += 1
                break  # Only count once per segment
    return exposures, exposures * penalty

def segments_intersect(A, B, C, D):
    """Check if line segments AB and CD intersect using CCW method"""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def count_route_crossings(routes, coords):
    """Count intersections between route segments"""
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
    """Convert permutation to valid routes with capacity constraints"""
    # Filter invalid customer indices
    valid_customers = [i for i in permutation if 1 <= i < len(demands)]
    routes = []
    route = [0]  # Start with depot
    load = 0
    
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)  # Return to depot
            routes.append(route)
            route = [0, cust]  # New route starting at depot
            load = demands[cust]
    
    route.append(0)  # Final return to depot
    routes.append(route)
    
    # Ensure vehicle limit
    while len(routes) > max_vehicles:
        last = routes.pop()
        routes[-1] = routes[-1][:-1] + last[1:]  # Merge last route
    
    return routes

def two_opt(route, coords):
    """2-opt route optimization"""
    improved = True
    best_route = route
    best_cost = calculate_route_cost(route, coords)
    
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                if j-i == 1: continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = calculate_route_cost(new_route, coords)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
        route = best_route
    return route

def validate_solution(routes, demands, capacity, num_customers):
    """Validate solution feasibility"""
    served = set()
    for route in routes:
        # Check depot start/end
        if route[0] != 0 or route[-1] != 0:
            return False
        
        # Check capacity
        if sum(demands[c] for c in route[1:-1]) > capacity:
            return False
        
        served.update(route[1:-1])
    
    # Check all customers served
    return len(served) == num_customers - 1  # -1 for depot

def route_statistics(routes, coords, demands):
    """Generate statistics for each route"""
    stats = []
    for i, route in enumerate(routes):
        dist = calculate_route_cost(route, coords)
        load = sum(demands[c] for c in route[1:-1])
        stats.append({
            'vehicle': i+1,
            'distance': dist,
            'load': load,
            'customers': len(route)-2,
            'route': route
        })
    return stats

# --- Enhanced SBA Optimizer (Swallow Bat Algorithm) ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=50, max_iter=500, alpha=0.97, gamma=0.97):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        # Algorithm parameters
        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.crossing_penalty = 300
        self.threat_penalty = 1000
        self.balance_weight = 50
        
        # Initialize population with valid customers
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        # Tracking
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []
        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        """Initialize population with heuristic solutions"""
        population = []
        valid_customers = self.customers.copy()
        
        # 1. Random permutations
        for _ in range(self.pop_size // 2):
            population.append(random.sample(valid_customers, len(valid_customers)))
        
        # 2. Nearest neighbor heuristic
        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(valid_customers)
            current = 0  # depot
            
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)
        
        # 3. Demand-sorted solutions
        for _ in range(self.pop_size // 4):
            # Sort by demand (high to low)
            sol = sorted(valid_customers, key=lambda x: -self.demands[x])
            population.append(sol)
            
            # Sort by demand (low to high)
            sol = sorted(valid_customers, key=lambda x: self.demands[x])
            population.append(sol)
        
        # Fill remaining with random
        while len(population) < self.pop_size:
            population.append(random.sample(valid_customers, len(valid_customers)))
        
        return population[:self.pop_size]

    def fitness(self, permutation):
        """Evaluate solution fitness with multiple cost components"""
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            
            # Base components
            distance_cost = calculate_total_cost(routes, self.coords)
            threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, self.threat_penalty) 
                            for r in routes])
            threat_cost = sum(threat_cost)
            threat_count = sum(threat_count)
            crossing_cost = count_route_crossings(routes, self.coords) * self.crossing_penalty
            
            # Balance penalty
            lengths = [calculate_route_cost(r, self.coords) for r in routes]
            balance_cost = np.std(lengths) * self.balance_weight
            
            # Vehicle count penalty
            vehicle_cost = max(0, len(routes) - self.max_vehicles) * 1000          
            total_cost = distance_cost + threat_cost + crossing_cost + balance_cost + vehicle_cost
            
            return total_cost, routes, distance_cost, threat_count
        except Exception as e:
            print(f"Fitness error: {e}")
            return float('inf'), [], float('inf'), float('inf')

    def run(self, stopping_threshold=None, max_time=300):
        """Run the optimization process"""
        # Initial evaluation
        solutions = self.population.copy()
        fitness_vals = []
        routes_list = []
        
        for sol in solutions:
            fit, routes, _, _ = self.fitness(sol)
            fitness_vals.append(fit)
            routes_list.append(routes)
            if fit < self.best_cost:
                self.best_cost = fit
                self.best_solution = sol.copy()
                self.best_routes = routes.copy()
        
        start_time = time.time()
        last_improvement = 0
        
        # Optimization loop
        for t in range(self.max_iter):
            if time.time() - start_time > max_time:
                print(f"Time limit reached at iteration {t}")
                break
                
            for i in range(self.pop_size):
                # Generate new solution
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * len(self.customers))
                
                if random.random() > self.pulse_rate[i]:
                    # Local search around best solution
                    new_sol = self._enhanced_local_search(self.best_solution)
                else:
                    # Random walk with velocity
                    new_sol = self._apply_velocity(solutions[i], 
                                                self._random_velocity(vel))
                
                # Evaluate new solution
                new_fit, new_routes, _, _ = self.fitness(new_sol)
                
                # Update if better or with small probability (diversification)
                if (new_fit < fitness_vals[i] or 
                    (random.random() < 0.05 and new_fit < 1.5 * fitness_vals[i])):
                    solutions[i] = new_sol
                    fitness_vals[i] = new_fit
                    routes_list[i] = new_routes
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] = 0.95 * (1 - math.exp(-self.gamma * t/self.max_iter))
                    
                    if new_fit < self.best_cost:
                        self.best_cost = new_fit
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes.copy()
                        last_improvement = t
            
            self.history.append(self.best_cost)
            
            # Early stopping if no improvement
            if stopping_threshold and t - last_improvement > stopping_threshold:
                print(f"Early stopping at iteration {t} (no improvement for {stopping_threshold} iterations)")
                break
                
            if (t+1) % 50 == 0:
                print(f"Iter {t+1}: Best Cost = {self.best_cost:.2f}")
        
        # Final validation and statistics
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        stats = route_statistics(self.best_routes, self.coords, self.demands)
        
        print(f"\nSolution Validation: {'Valid' if is_valid else 'Invalid'}")
        print("\nRoute Statistics:")
        for s in stats:
            print(f"Vehicle {s['vehicle']}: {s['customers']} customers, "
                  f"Load {s['load']}/{self.capacity}, Distance {s['distance']:.2f}")
        
        # Cost breakdown
        total_distance = sum(s['distance'] for s in stats)
        threat_exp = sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes)
        crossings = count_route_crossings(self.best_routes, self.coords)
        
        print(f"\nCost Breakdown:")
        print(f"- Distance Cost: {total_distance:.2f}")
        print(f"- Threat Exposures: {threat_exp} (Penalty: {threat_exp * self.threat_penalty})")
        print(f"- Route Crossings: {crossings} (Penalty: {crossings * self.crossing_penalty})")
        print(f"- Vehicle Usage: {len(self.best_routes)}/{self.max_vehicles}")
        print(f"Total Cost: {self.best_cost:.2f}")
        
        return {
            'name': 'SBA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': total_distance,
            'threat_exposure': threat_exp,
            'crossings': crossings,
            'convergence': self.history,
            'stats': stats,
            'valid': is_valid,
            'time': time.time() - start_time
        }

    def _random_velocity(self, length):
        """Generate random velocity for solution modification"""
        return [random.sample(range(len(self.customers)), 2) 
               for _ in range(random.randint(1, max(1, length)))]

    def _apply_velocity(self, perm, velocity):
        """Apply velocity to permutation with bounds checking"""
        perm = perm.copy()
        n = len(perm)
        for i, j in velocity:
            if 0 <= i < n and 0 <= j < n:
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _enhanced_local_search(self, perm):
        """Enhanced local search with multiple operators"""
        perm = perm.copy()
        r = random.random()
        
        if r < 0.4:  # 2-opt style reversal (40%)
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:  # Swap mutation (30%)
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        elif r < 0.9:  # Insertion mutation (20%)
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)
        else:  # Scramble mutation (10%)
            i, j = sorted(random.sample(range(len(perm)), 2))
            segment = perm[i:j+1]
            random.shuffle(segment)
            perm[i:j+1] = segment
            
        return perm

# --- ALNS with Robust Repair Operators ---
class ALNS:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 max_iter=200, adaptive_period=50, initial_temperature=1000):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.max_iter = max_iter
        self.adaptive_period = adaptive_period
        self.temperature = initial_temperature
        self.cooling_rate = 0.995
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        
        # Destroy and repair operators with weights
        self.destroy_operators = [
            self._random_removal,
            self._worst_removal,
            self._route_removal,
            self._threat_zone_removal
        ]
        
        self.repair_operators = [
            self._robust_greedy_insertion,
            self._robust_regret_insertion
        ]
        
        # Initialize weights and scores
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.repair_weights = [1.0] * len(self.repair_operators)
        self.destroy_scores = [0] * len(self.destroy_operators)
        self.repair_scores = [0] * len(self.repair_operators)
        self.operator_usage = [0] * len(self.destroy_operators)
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initial_solution(self):
        """Create initial solution using sequential insertion"""
        routes = []
        unserved = self.customers.copy()
        random.shuffle(unserved)
        
        while unserved:
            # Start new route
            route = [0]
            load = 0
            temp_unserved = unserved.copy()
            
            # Add customers until capacity is reached
            for customer in temp_unserved:
                if load + self.demands[customer] <= self.capacity:
                    route.append(customer)
                    load += self.demands[customer]
                    unserved.remove(customer)
                if load >= self.capacity * 0.8:  # Leave some room
                    break
            
            route.append(0)
            routes.append(route)
            
            # Check vehicle limit
            if len(routes) >= self.max_vehicles:
                break
        
        # Handle remaining customers
        while unserved:
            customer = unserved.pop(0)
            best_cost = float('inf')
            best_route_idx = None
            best_position = None
            
            for idx, route in enumerate(routes):
                route_load = sum(self.demands[c] for c in route[1:-1])
                if route_load + self.demands[customer] <= self.capacity:
                    for pos in range(1, len(route)):
                        cost_increase = (euclidean(self.coords[route[pos-1]], self.coords[customer]) +
                                       euclidean(self.coords[customer], self.coords[route[pos]]) -
                                       euclidean(self.coords[route[pos-1]], self.coords[route[pos]]))
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_route_idx = idx
                            best_position = pos
            
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_position, customer)
            else:
                # Create new route if possible, otherwise force insertion
                if len(routes) < self.max_vehicles:
                    routes.append([0, customer, 0])
                else:
                    # Insert in least loaded route
                    least_loaded_idx = min(range(len(routes)), 
                                         key=lambda i: sum(self.demands[c] for c in routes[i][1:-1]))
                    # Find best position in least loaded route
                    best_pos = 1
                    best_cost = float('inf')
                    for pos in range(1, len(routes[least_loaded_idx])):
                        cost_increase = (euclidean(self.coords[routes[least_loaded_idx][pos-1]], self.coords[customer]) +
                                       euclidean(self.coords[customer], self.coords[routes[least_loaded_idx][pos]]) -
                                       euclidean(self.coords[routes[least_loaded_idx][pos-1]], self.coords[routes[least_loaded_idx][pos]]))
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_pos = pos
                    routes[least_loaded_idx].insert(best_pos, customer)
        
        return routes

    def fitness(self, routes):
        """Evaluate solution fitness"""
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        
        # Vehicle count penalty
        vehicle_cost = max(0, len(routes) - self.max_vehicles) * 1000
        
        total_cost = distance_cost + threat_cost + crossing_cost + vehicle_cost
        return total_cost

    # Destroy operators
    def _random_removal(self, routes, q=5):
        """Randomly remove q customers"""
        removed_customers = []
        all_customers = []
        for route in routes:
            all_customers.extend([c for c in route[1:-1]])
        
        if len(all_customers) == 0:
            return routes, removed_customers
            
        q = min(q, len(all_customers))
        removed = random.sample(all_customers, q)
        
        new_routes = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            if len(new_route) > 2:  # Only keep non-empty routes
                new_routes.append(new_route)
        removed_customers.extend(removed)
        
        return new_routes, removed_customers

    def _worst_removal(self, routes, q=5):
        """Remove customers with highest cost contribution"""
        removed_customers = []
        
        # Calculate cost contribution for each customer
        customer_costs = {}
        for route in routes:
            for i, cust in enumerate(route[1:-1], 1):
                # Cost of removing this customer
                prev_cust = route[i-1]
                next_cust = route[i+1]
                original_cost = (euclidean(self.coords[prev_cust], self.coords[cust]) +
                               euclidean(self.coords[cust], self.coords[next_cust]))
                new_cost = euclidean(self.coords[prev_cust], self.coords[next_cust])
                cost_saving = original_cost - new_cost
                customer_costs[cust] = cost_saving
        
        # Sort by cost saving (descending)
        if not customer_costs:
            return self._random_removal(routes, q)
            
        sorted_customers = sorted(customer_costs.keys(), key=lambda x: customer_costs[x], reverse=True)
        q = min(q, len(sorted_customers))
        removed = sorted_customers[:q]
        
        new_routes = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            if len(new_route) > 2:
                new_routes.append(new_route)
        removed_customers.extend(removed)
        
        return new_routes, removed_customers

    def _route_removal(self, routes, q=3):
        """Remove entire routes"""
        if len(routes) <= 1:
            return self._random_removal(routes, q)
        
        removed_customers = []
        q_routes = min(q, len(routes) - 1)  # Keep at least one route
        routes_to_remove = random.sample(routes, q_routes)
        
        for route in routes_to_remove:
            removed_customers.extend(route[1:-1])
        
        new_routes = [route for route in routes if route not in routes_to_remove]
        return new_routes, removed_customers

    def _threat_zone_removal(self, routes, q=5):
        """Remove customers in threat zones"""
        removed_customers = []
        threatened_customers = []
        
        for route in routes:
            for cust in route[1:-1]:
                if is_in_threat_zone(self.coords[cust], self.zones):
                    threatened_customers.append(cust)
        
        if not threatened_customers:
            return self._random_removal(routes, q)
        
        q = min(q, len(threatened_customers))
        removed = random.sample(threatened_customers, q)
        
        new_routes = []
        for route in routes:
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            if len(new_route) > 2:
                new_routes.append(new_route)
        removed_customers.extend(removed)
        
        return new_routes, removed_customers

    # Robust repair operators
    def _robust_greedy_insertion(self, routes, removed_customers):
        """Robust greedy insertion that handles edge cases"""
        unserved = removed_customers.copy()
        
        # Safety check
        if not unserved:
            return routes
            
        # Randomize insertion order for diversity
        random.shuffle(unserved)
        
        while unserved:
            customer = unserved[0]  # Always work with first customer to avoid removal issues
            best_cost = float('inf')
            best_route_idx = None
            best_position = None
            
            # Try existing routes first
            for route_idx, route in enumerate(routes):
                route_load = sum(self.demands[c] for c in route[1:-1])
                if route_load + self.demands[customer] <= self.capacity:
                    for pos in range(1, len(route)):
                        cost_increase = (euclidean(self.coords[route[pos-1]], self.coords[customer]) +
                                       euclidean(self.coords[customer], self.coords[route[pos]]) -
                                       euclidean(self.coords[route[pos-1]], self.coords[route[pos]]))
                        
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_route_idx = route_idx
                            best_position = pos
            
            # Insert customer
            if best_route_idx is not None:
                routes[best_route_idx].insert(best_position, customer)
                unserved.remove(customer)
            else:
                # Create new route if possible
                if len(routes) < self.max_vehicles:
                    routes.append([0, customer, 0])
                    unserved.remove(customer)
                else:
                    # Force insertion in least loaded route (violates capacity but keeps solution valid)
                    least_loaded_idx = min(range(len(routes)), 
                                         key=lambda i: sum(self.demands[c] for c in routes[i][1:-1]))
                    # Find best position
                    best_pos = 1
                    best_cost = float('inf')
                    for pos in range(1, len(routes[least_loaded_idx])):
                        cost_increase = (euclidean(self.coords[routes[least_loaded_idx][pos-1]], self.coords[customer]) +
                                       euclidean(self.coords[customer], self.coords[routes[least_loaded_idx][pos]]) -
                                       euclidean(self.coords[routes[least_loaded_idx][pos-1]], self.coords[routes[least_loaded_idx][pos]]))
                        if cost_increase < best_cost:
                            best_cost = cost_increase
                            best_pos = pos
                    routes[least_loaded_idx].insert(best_pos, customer)
                    unserved.remove(customer)
        
        return routes

    def _robust_regret_insertion(self, routes, removed_customers):
        """Simplified regret insertion that uses robust greedy as fallback"""
        return self._robust_greedy_insertion(routes, removed_customers)

    def run(self):
        """Run ALNS algorithm"""
        start_time = time.time()
        
        # Initial solution
        current_routes = self._initial_solution()
        current_cost = self.fitness(current_routes)
        self.best_routes = deepcopy(current_routes)
        self.best_cost = current_cost
        
        print(f"Initial solution cost: {current_cost:.2f}")
        
        for iteration in range(self.max_iter):
            # Adaptive operator selection
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)
            
            destroy_operator = self.destroy_operators[destroy_idx]
            repair_operator = self.repair_operators[repair_idx]
            
            # Determine degree of destruction (adaptive)
            degree = max(2, min(6, int(len(self.customers) * 0.1)))
            
            try:
                # Destroy and repair
                destroyed_routes, removed = destroy_operator(current_routes, degree)
                
                # Ensure we have routes to work with
                if not destroyed_routes:
                    destroyed_routes = [[0, 0]]  # At least one empty route
                
                new_routes = repair_operator(destroyed_routes, removed)
                
                # Remove any empty routes (except one if all are empty)
                new_routes = [route for route in new_routes if len(route) > 2]
                if not new_routes:
                    new_routes = [[0, random.choice(self.customers), 0]]
                
                # Optimize routes with 2-opt
                for i in range(len(new_routes)):
                    if len(new_routes[i]) > 4:  # Only optimize routes with at least 2 customers
                        new_routes[i] = two_opt(new_routes[i], self.coords)
                
                new_cost = self.fitness(new_routes)
                
                # Simulated annealing acceptance
                if new_cost < current_cost or random.random() < math.exp((current_cost - new_cost) / max(0.1, self.temperature)):
                    current_routes = new_routes
                    current_cost = new_cost
                    
                    # Update scores based on solution quality
                    if new_cost < self.best_cost:
                        self.best_routes = deepcopy(new_routes)
                        self.best_cost = new_cost
                        self.destroy_scores[destroy_idx] += 3
                        self.repair_scores[repair_idx] += 3
                        print(f"ALNS Iter {iteration+1}: New Best Cost = {self.best_cost:.2f}")
                    else:
                        self.destroy_scores[destroy_idx] += 2
                        self.repair_scores[repair_idx] += 2
                else:
                    self.destroy_scores[destroy_idx] += 1
                    self.repair_scores[repair_idx] += 1
                
                self.operator_usage[destroy_idx] += 1
                
            except Exception as e:
                # If operator fails, continue with current solution
                print(f"ALNS operator error at iteration {iteration}: {e}")
                continue
            
            # Update weights periodically
            if iteration > 0 and iteration % self.adaptive_period == 0:
                self._update_weights()
            
            # Cooling
            self.temperature *= self.cooling_rate
            
            self.history.append(self.best_cost)
        
        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'ALNS',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': calculate_total_cost(self.best_routes, self.coords),
            'threat_exposure': sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes),
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

    def _select_operator(self, weights):
        """Select operator using roulette wheel selection"""
        total = sum(weights)
        if total == 0:
            return random.randint(0, len(weights)-1)
        r = random.uniform(0, total)
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1

    def _update_weights(self):
        """Update operator weights based on performance"""
        for i in range(len(self.destroy_operators)):
            if self.operator_usage[i] > 0:
                self.destroy_weights[i] = (0.8 * self.destroy_weights[i] + 
                                         0.2 * self.destroy_scores[i] / self.operator_usage[i])
        
        for i in range(len(self.repair_operators)):
            if self.operator_usage[i] > 0:
                self.repair_weights[i] = (0.8 * self.repair_weights[i] + 
                                        0.2 * self.repair_scores[i] / self.operator_usage[i])
        
        # Reset scores and usage
        self.destroy_scores = [0] * len(self.destroy_operators)
        self.repair_scores = [0] * len(self.repair_operators)
        self.operator_usage = [0] * len(self.destroy_operators)

# --- Hybrid Genetic Algorithm (HGA) ---
class HGA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=30, max_generations=300, crossover_rate=0.8, 
                 mutation_rate=0.1, elite_size=2):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []

    def _initialize_population(self):
        """Initialize population with diverse solutions"""
        population = []
        
        # Random solutions
        for _ in range(self.pop_size):
            population.append(random.sample(self.customers, len(self.customers)))
        
        return population

    def fitness(self, permutation):
        """Evaluate solution fitness"""
        routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
        distance_cost = calculate_total_cost(routes, self.coords)
        threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, 1000) 
                        for r in routes])
        threat_cost = sum(threat_cost)
        crossing_cost = count_route_crossings(routes, self.coords) * 300
        
        # Vehicle count penalty
        vehicle_cost = max(0, len(routes) - self.max_vehicles) * 1000
        
        total_cost = distance_cost + threat_cost + crossing_cost + vehicle_cost
        return total_cost, routes

    def _selection(self, fitness_vals):
        """Tournament selection"""
        tournament_size = 3
        tournament = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [fitness_vals[i] for i in tournament]
        return self.population[tournament[np.argmin(tournament_fitness)]]

    def _crossover(self, parent1, parent2):
        """Order crossover (OX)"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy segment
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Fill remaining positions
        current_pos1 = (end + 1) % size
        current_pos2 = (end + 1) % size
        
        for i in range(size):
            pos = (end + 1 + i) % size
            if parent2[pos] not in child1:
                child1[current_pos1] = parent2[pos]
                current_pos1 = (current_pos1 + 1) % size
            
            if parent1[pos] not in child2:
                child2[current_pos2] = parent1[pos]
                current_pos2 = (current_pos2 + 1) % size
        
        return child1, child2

    def _mutate(self, individual):
        """Mutation operators"""
        if random.random() > self.mutation_rate:
            return individual
        
        individual = individual.copy()
        mutation_type = random.choice(['swap', 'inversion', 'insertion'])
        
        if mutation_type == 'swap':
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        
        elif mutation_type == 'inversion':
            i, j = sorted(random.sample(range(len(individual)), 2))
            individual[i:j+1] = reversed(individual[i:j+1])
        
        elif mutation_type == 'insertion':
            i, j = random.sample(range(len(individual)), 2)
            customer = individual.pop(i)
            individual.insert(j, customer)
        
        return individual

    def run(self):
        """Run hybrid genetic algorithm"""
        start_time = time.time()
        
        # Initial evaluation
        fitness_vals = []
        routes_list = []
        
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            fitness_vals.append(cost)
            routes_list.append(routes)
            
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for generation in range(self.max_generations):
            new_population = []
            
            # Elitism: keep best solutions
            elite_indices = np.argsort(fitness_vals)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(self.population[idx])
            
            # Generate new population
            while len(new_population) < self.pop_size:
                parent1 = self._selection(fitness_vals)
                parent2 = self._selection(fitness_vals)
                
                child1, child2 = self._crossover(parent1, parent2)
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            self.population = new_population[:self.pop_size]
            
            # Evaluate new population
            fitness_vals = []
            for i in range(self.pop_size):
                cost, routes = self.fitness(self.population[i])
                fitness_vals.append(cost)
                
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = self.population[i].copy()
                    self.best_routes = routes
            
            self.history.append(self.best_cost)
            
            if (generation + 1) % 50 == 0:
                print(f"HGA Generation {generation+1}: Best Cost = {self.best_cost:.2f}")
        
        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'HGA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': calculate_total_cost(self.best_routes, self.coords),
            'threat_exposure': sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes),
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Simple Bat Algorithm (BA) ---
class BatAlgorithm:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=20, max_iter=200):
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
        
        # Initial evaluation
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
                # Generate new solution
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                new_sol = self.population[i].copy()
                
                # Apply frequency-based modification
                for j in range(len(new_sol)):
                    if random.random() < freq:
                        swap_idx = random.randint(0, len(new_sol)-1)
                        new_sol[j], new_sol[swap_idx] = new_sol[swap_idx], new_sol[j]
                
                # Local search with pulse rate
                if random.random() > self.pulse_rate:
                    # Simple local search: swap two random customers
                    idx1, idx2 = random.sample(range(len(new_sol)), 2)
                    new_sol[idx1], new_sol[idx2] = new_sol[idx2], new_sol[idx1]
                
                new_cost, new_routes = self.fitness(new_sol)
                
                # Update if better solution found
                if new_cost < fitness_vals[i] and random.random() < self.loudness:
                    self.population[i] = new_sol
                    fitness_vals[i] = new_cost
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
            
            # Update parameters
            self.loudness *= 0.95
            self.pulse_rate = 0.5 * (1 - math.exp(-0.1 * iteration))
            
            self.history.append(self.best_cost)
            
            if (iteration + 1) % 50 == 0:
                print(f"BA Iter {iteration+1}: Best Cost = {self.best_cost:.2f}")
        
        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'BA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': calculate_total_cost(self.best_routes, self.coords),
            'threat_exposure': sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes),
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Particle Swarm Optimization (PSO) ---
class PSO:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=20, max_iter=200, w=0.7, c1=1.4, c2=1.4):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
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
        
        # Initial evaluation
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            self.pbest_cost[i] = cost
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # Update velocity and position
                for j in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (self.pbest[i][j] - self.population[i][j])
                    social = self.c2 * r2 * (self.best_solution[j] - self.population[i][j])
                    self.velocity[i][j] = self.w * self.velocity[i][j] + cognitive + social
                    
                    # Update position (discrete PSO)
                    if random.random() < abs(self.velocity[i][j]):
                        swap_idx = random.randint(0, len(self.customers)-1)
                        self.population[i][j], self.population[i][swap_idx] = \
                            self.population[i][swap_idx], self.population[i][j]
                
                # Evaluate new solution
                new_cost, new_routes = self.fitness(self.population[i])
                
                # Update personal best
                if new_cost < self.pbest_cost[i]:
                    self.pbest[i] = self.population[i].copy()
                    self.pbest_cost[i] = new_cost
                    
                    # Update global best
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = self.population[i].copy()
                        self.best_routes = new_routes
            
            self.history.append(self.best_cost)
            
            if (iteration + 1) % 50 == 0:
                print(f"PSO Iter {iteration+1}: Best Cost = {self.best_cost:.2f}")
        
        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'PSO',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': calculate_total_cost(self.best_routes, self.coords),
            'threat_exposure': sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes),
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Harris Hawks Optimization (HHO) ---
class HHO:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=20, max_iter=200):
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
        
        # Initial evaluation
        fitness_vals = []
        for i in range(self.pop_size):
            cost, routes = self.fitness(self.population[i])
            fitness_vals.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_solution = self.population[i].copy()
                self.best_routes = routes
        
        for iteration in range(self.max_iter):
            E0 = 2 * random.random() - 1  # -1 to 1
            E = 2 * E0 * (1 - iteration / self.max_iter)  # Escaping energy
            
            for i in range(self.pop_size):
                new_sol = self.population[i].copy()
                
                if abs(E) >= 1:
                    # Exploration phase
                    q = random.random()
                    if q >= 0.5:
                        # Random walk based on other solutions
                        k = random.randint(0, self.pop_size-1)
                        for j in range(len(new_sol)):
                            if random.random() < 0.5:
                                new_sol[j] = self.population[k][j]
                    else:
                        # Random permutation
                        random.shuffle(new_sol)
                else:
                    # Exploitation phase
                    r = random.random()
                    if r >= 0.5 and abs(E) < 0.5:
                        # Soft besiege
                        for j in range(len(new_sol)):
                            if random.random() < 0.7:
                                new_sol[j] = self.best_solution[j]
                    else:
                        # Hard besiege - guided local search
                        idx1, idx2 = random.sample(range(len(new_sol)), 2)
                        new_sol[idx1], new_sol[idx2] = new_sol[idx2], new_sol[idx1]
                
                # Ensure valid solution
                if len(set(new_sol)) != len(self.customers):
                    new_sol = random.sample(self.customers, len(self.customers))
                
                new_cost, new_routes = self.fitness(new_sol)
                
                if new_cost < fitness_vals[i]:
                    self.population[i] = new_sol
                    fitness_vals[i] = new_cost
                    if new_cost < self.best_cost:
                        self.best_cost = new_cost
                        self.best_solution = new_sol.copy()
                        self.best_routes = new_routes
            
            self.history.append(self.best_cost)
            
            if (iteration + 1) % 50 == 0:
                print(f"HHO Iter {iteration+1}: Best Cost = {self.best_cost:.2f}")
        
        # Final validation
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        return {
            'name': 'HHO',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': calculate_total_cost(self.best_routes, self.coords),
            'threat_exposure': sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes),
            'crossings': count_route_crossings(self.best_routes, self.coords),
            'convergence': self.history,
            'valid': is_valid,
            'time': time.time() - start_time
        }

# --- Visualization Functions ---
def plot_routes(coords, routes, zones, title):
    colors = plt.cm.tab20.colors
    plt.figure(figsize=(12, 8))

    # Depot
    depot_handle = plt.scatter(coords[0][0], coords[0][1], c='black', s=200, marker='s')

    # Customers
    for i in range(1, len(coords)):
        plt.scatter(coords[i][0], coords[i][1], c='blue', s=100)
        plt.text(coords[i][0], coords[i][1], f"{i}({demands[i]})", fontsize=8)

    # Vehicles
    vehicle_handles = []
    vehicle_labels = []
    for i, route in enumerate(routes):
        load = sum(demands[n] for n in route[1:-1])
        x = [coords[n][0] for n in route]
        y = [coords[n][1] for n in route]
        vh, = plt.plot(x, y, marker='o', color=colors[i % 20], linewidth=2)
        vehicle_handles.append(vh)
        vehicle_labels.append(f'Vehicle {i+1}: (Load {load}/{capacity})')

        # Arrows
        for j in range(len(route) - 1):
            dx = x[j+1] - x[j]
            dy = y[j+1] - y[j]
            plt.arrow(x[j], y[j], dx*0.8, dy*0.8,
                      shape='full', color=colors[i % 20],
                      length_includes_head=True, head_width=2)

    # Threat zones (label only once)
    threat_zone_handles = []
    for idx, zone in enumerate(zones):
        circle = plt.Circle(zone['center'], zone['radius'], color='red', alpha=0.2)
        plt.gca().add_patch(circle)
        th, = plt.plot(zone['center'][0], zone['center'][1], 'rx', markersize=10)
        threat_zone_handles.append(th)

    # Legend handles and labels
    handles = [depot_handle] + vehicle_handles + [threat_zone_handles[0]]
    labels = ['Depot'] + vehicle_labels + ['Threat Zone']
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1))

    plt.title(title)
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_convergence(histories, algorithm_names):
    """Plot optimization convergence for multiple algorithms"""
    plt.figure(figsize=(12, 8))
    for i, history in enumerate(histories):
        if history:  # Only plot if history exists
            plt.plot(history, label=algorithm_names[i], linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Cost')
    plt.title('Algorithm Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Main Experiment ---
def run_comprehensive_experiment():
    """Run comprehensive comparative experiment"""
    print("Running Comprehensive Algorithm Comparison...")
    print("=" * 80)
    
    algorithms = [
        ("SBA (Proposed)", lambda: SBA(coords, demands, capacity, threat_zones, max_vehicles, population_size=30, max_iter=200).run(max_time=60)),
        ("ALNS", lambda: ALNS(coords, demands, capacity, threat_zones, max_vehicles, max_iter=200).run()),
        ("Hybrid GA", lambda: HGA(coords, demands, capacity, threat_zones, max_vehicles, population_size=20, max_generations=150).run()),
        ("Bat Algorithm", lambda: BatAlgorithm(coords, demands, capacity, threat_zones, max_vehicles, population_size=15, max_iter=150).run()),
        ("Particle Swarm", lambda: PSO(coords, demands, capacity, threat_zones, max_vehicles, population_size=15, max_iter=150).run()),
        ("Harris Hawks", lambda: HHO(coords, demands, capacity, threat_zones, max_vehicles, population_size=15, max_iter=150).run()),
    ]
    
    results = []
    
    # Run metaheuristic algorithms
    for name, algorithm in algorithms:
        print(f"\n--- Running {name} ---")
        try:
            result = algorithm()
            # Calculate additional metrics
            if 'routes' in result:
                result['distance'] = calculate_total_cost(result['routes'], coords)
                result['threat_exposure'] = sum(calculate_threat_penalty(r, coords, threat_zones, 1)[0] 
                                             for r in result['routes'])
                result['crossings'] = count_route_crossings(result['routes'], coords)
                result['vehicles'] = len(result['routes'])
                result['valid'] = validate_solution(result['routes'], demands, capacity, len(coords))
            results.append(result)
            print(f"{name} completed in {result['time']:.2f}s with cost {result['cost']:.2f}")
        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'name': name, 'cost': float('inf'), 'time': 0, 'valid': False})
    
    # Display comprehensive results
    print("\n" + "=" * 100)
    print("COMPREHENSIVE ALGORITHM COMPARISON RESULTS")
    print("=" * 100)
    print(f"{'Algorithm':<25} | {'Cost':>10} | {'Time(s)':>8} | {'Distance':>10} | "
          f"{'Threats':>7} | {'Crossings':>9} | {'Vehicles':>8} | {'Valid':>5}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['name']:<25} | {r.get('cost',0):10.2f} | {r.get('time',0):8.2f} | "
              f"{r.get('distance',0):10.2f} | {r.get('threat_exposure',0):7.0f} | "
              f"{r.get('crossings',0):9d} | {r.get('vehicles',0):8d} | "
              f"{'Yes' if r.get('valid', False) else 'No':>5}")
    
    # Find best solution
    valid_results = [r for r in results if r.get('valid', False)]
    if valid_results:
        best_solution = min(valid_results, key=lambda x: x['cost'])
        print(f"\n*** BEST SOLUTION: {best_solution['name']} with cost {best_solution['cost']:.2f} ***")
    else:
        print("\n*** NO VALID SOLUTIONS FOUND ***")
        best_solution = min(results, key=lambda x: x.get('cost', float('inf')))
    
    # Visualizations - Fixed to handle different result lengths
    print("\nGenerating visualizations...")
    
    # Plot routes for each algorithm
    for result in results:
        if result.get('routes') and result.get('valid', False):
            plot_routes(coords, result['routes'], threat_zones, 
                       f"{result['name']} Solution\nCost: {result['cost']:.2f}")
    
    # Plot convergence comparison
    convergence_data = [r.get('convergence', []) for r in results if 'convergence' in r and r.get('convergence')]
    convergence_names = [r['name'] for r in results if 'convergence' in r and r.get('convergence')]
    if convergence_data and len(convergence_data) > 1:
        plot_convergence(convergence_data, convergence_names)
    
    # Performance comparison bar chart - Fixed to handle different array lengths
    valid_results_for_viz = [r for r in results if r.get('cost', float('inf')) < float('inf')]
    if len(valid_results_for_viz) > 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        names = [r['name'] for r in valid_results_for_viz]
        
        # Cost comparison
        costs = [r.get('cost', 0) for r in valid_results_for_viz]
        bars1 = ax1.bar(range(len(names)), costs, color='skyblue')
        ax1.set_title('Total Cost Comparison')
        ax1.set_ylabel('Cost')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45)
        # Add value labels on bars
        for bar, cost in zip(bars1, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{cost:.0f}', 
                    ha='center', va='bottom')
        
        # Time comparison
        times = [r.get('time', 0) for r in valid_results_for_viz]
        bars2 = ax2.bar(range(len(names)), times, color='lightcoral')
        ax2.set_title('Computation Time Comparison')
        ax2.set_ylabel('Time (s)')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45)
        # Add value labels on bars
        for bar, time_val in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{time_val:.1f}s', 
                    ha='center', va='bottom')
        
        # Distance comparison
        distances = [r.get('distance', 0) for r in valid_results_for_viz]
        bars3 = ax3.bar(range(len(names)), distances, color='lightgreen')
        ax3.set_title('Total Distance Comparison')
        ax3.set_ylabel('Distance')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45)
        # Add value labels on bars
        for bar, dist in zip(bars3, distances):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{dist:.1f}', 
                    ha='center', va='bottom')
        
        # Threat exposure comparison
        threats = [r.get('threat_exposure', 0) for r in valid_results_for_viz]
        bars4 = ax4.bar(range(len(names)), threats, color='gold')
        ax4.set_title('Threat Exposure Comparison')
        ax4.set_ylabel('Threat Exposures')
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(names, rotation=45)
        # Add value labels on bars
        for bar, threat in zip(bars4, threats):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{threat}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    return best_solution, results

if __name__ == "__main__":
    # Run comprehensive experiment
    print("A-n32-k5 Instance - Algorithm Comparison")
    print("Algorithms: SBA, ALNS, HGA, BA, PSO, HHO")
    print("=" * 70)
    
    best_solution, all_results = run_comprehensive_experiment()
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    valid_results = [r for r in all_results if r.get('valid', False)]
    print(f"Valid solutions: {len(valid_results)}/{len(all_results)}")
    
    if valid_results:
        avg_cost = np.mean([r['cost'] for r in valid_results])
        avg_time = np.mean([r['time'] for r in valid_results])
        print(f"Average cost: {avg_cost:.2f}")
        print(f"Average time: {avg_time:.2f}s")
        
        # Rank solutions
        ranked = sorted(valid_results, key=lambda x: x['cost'])
        print("\nSolution Ranking:")
        for i, result in enumerate(ranked, 1):
            print(f"{i}. {result['name']}: {result['cost']:.2f} "
                  f"(Time: {result['time']:.2f}s, Vehicles: {result.get('vehicles', 0)})")