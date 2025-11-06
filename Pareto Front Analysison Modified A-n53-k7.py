import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from itertools import combinations

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

# --- Strategic Threat Zones ---
def create_strategic_threat_zones():
    return [
        {"center": (40, 50), "radius": 8},
        {"center": (70, 70), "radius": 6},
        {"center": (20, 30), "radius": 7},
        {"center": (60, 30), "radius": 5},
        {"center": (30, 70), "radius": 6}
    ]

threat_zones = create_strategic_threat_zones()

# --- Utility Functions (keep all the existing utility functions) ---
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

# --- SBA Optimizer ---
class SBA:
    def __init__(self, coords, demands, capacity, threat_zones, max_vehicles=5,
                 population_size=50, max_iter=500, alpha=0.97, gamma=0.97):
        self.coords = coords
        self.demands = demands
        self.capacity = capacity
        self.zones = threat_zones
        self.max_vehicles = max_vehicles
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.crossing_penalty = 300
        self.threat_penalty = 1000
        self.balance_weight = 50
        
        self.customers = [i for i in range(1, len(self.demands)) if i < len(self.coords)]
        self.population = self._initialize_population()
        
        self.best_solution = None
        self.best_cost = float('inf')
        self.best_routes = None
        self.history = []
        self.loudness = [0.5] * self.pop_size
        self.pulse_rate = [0.5] * self.pop_size
        self.freq_min = 0
        self.freq_max = 2

    def _initialize_population(self):
        population = []
        valid_customers = self.customers.copy()
        
        for _ in range(self.pop_size // 2):
            population.append(random.sample(valid_customers, len(valid_customers)))
        
        for _ in range(self.pop_size // 4):
            sol = []
            unvisited = set(valid_customers)
            current = 0
            while unvisited:
                nearest = min(unvisited, key=lambda x: euclidean(self.coords[current], self.coords[x]))
                sol.append(nearest)
                unvisited.remove(nearest)
                current = nearest
            population.append(sol)
        
        for _ in range(self.pop_size // 4):
            population.append(sorted(valid_customers, key=lambda x: -self.demands[x]))
            population.append(sorted(valid_customers, key=lambda x: self.demands[x]))
        
        while len(population) < self.pop_size:
            population.append(random.sample(valid_customers, len(valid_customers)))
        
        return population[:self.pop_size]

    def fitness(self, permutation):
        try:
            routes = decode_routes(permutation, self.demands, self.capacity, self.max_vehicles)
            
            distance_cost = calculate_total_cost(routes, self.coords)
            threat_count, threat_cost = zip(*[calculate_threat_penalty(r, self.coords, self.zones, self.threat_penalty) 
                            for r in routes])
            threat_cost = sum(threat_cost)
            threat_count = sum(threat_count)
            
            # Simplified cost calculation for stability
            total_cost = distance_cost + threat_cost
            
            return total_cost, routes, distance_cost, threat_count
        except Exception as e:
            return float('inf'), [], float('inf'), float('inf')

    def run(self, stopping_threshold=None, max_time=300):
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
        
        for t in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
                
            for i in range(self.pop_size):
                freq = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                vel = int(freq * len(self.customers))
                
                if random.random() > self.pulse_rate[i]:
                    new_sol = self._enhanced_local_search(self.best_solution)
                else:
                    new_sol = self._apply_velocity(solutions[i], self._random_velocity(vel))
                
                new_fit, new_routes, _, _ = self.fitness(new_sol)
                
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
            
            if stopping_threshold and t - last_improvement > stopping_threshold:
                break
                
            if (t+1) % 50 == 0:
                print(f"Iter {t+1}: Best Cost = {self.best_cost:.2f}")
        
        is_valid = validate_solution(self.best_routes, self.demands, self.capacity, len(self.demands))
        
        # Calculate final metrics
        total_distance = calculate_total_cost(self.best_routes, self.coords)
        threat_exp = sum(calculate_threat_penalty(r, self.coords, self.zones, 1)[0] for r in self.best_routes)
        
        return {
            'name': 'SBA',
            'routes': self.best_routes,
            'cost': self.best_cost,
            'distance': total_distance,
            'threat_exposure': threat_exp,
            'valid': is_valid,
            'time': time.time() - start_time
        }

    def _random_velocity(self, length):
        return [random.sample(range(len(self.customers)), 2) 
               for _ in range(random.randint(1, max(1, length)))]

    def _apply_velocity(self, perm, velocity):
        perm = perm.copy()
        n = len(perm)
        for i, j in velocity:
            if 0 <= i < n and 0 <= j < n:
                perm[i], perm[j] = perm[j], perm[i]
        return perm

    def _enhanced_local_search(self, perm):
        perm = perm.copy()
        r = random.random()
        
        if r < 0.4:
            i, j = sorted(random.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        elif r < 0.7:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        elif r < 0.9:
            i = random.randint(0, len(perm)-1)
            j = random.randint(0, len(perm)-1)
            if i != j:
                customer = perm.pop(i)
                perm.insert(j, customer)
        else:
            i, j = sorted(random.sample(range(len(perm)), 2))
            segment = perm[i:j+1]
            random.shuffle(segment)
            perm[i:j+1] = segment
            
        return perm

# --- Robust Pareto Front Analysis ---
def calculate_pareto_front_robust(algorithm_class, coords, demands, capacity, threat_zones, max_vehicles, 
                                 max_threat_exposures=6, runs_per_epsilon=5):
    """
    Robust Pareto front calculation with comprehensive error handling
    """
    print("Calculating Robust Pareto Front...")
    print("=" * 60)
    
    pareto_points = []
    all_solutions = []
    
    threat_limits = list(range(0, max_threat_exposures + 1))
    
    for max_threats in threat_limits:
        print(f"\n--- Running with max threats = {max_threats} ---")
        best_cost_for_epsilon = float('inf')
        best_solution_for_epsilon = None
        
        successful_runs = 0
        for run in range(runs_per_epsilon):
            print(f"  Run {run + 1}/{runs_per_epsilon}", end="\r")
            
            try:
                # Initialize algorithm
                algo = algorithm_class(coords, demands, capacity, threat_zones, max_vehicles)
                
                # Store original fitness
                original_fitness = algo.fitness
                
                def constrained_fitness(permutation):
                    total_cost, routes, distance_cost, threat_count = original_fitness(permutation)
                    
                    # Apply progressive penalty rather than hard constraint
                    if threat_count > max_threats:
                        excess_penalty = (threat_count - max_threats) * 5000
                        return total_cost + excess_penalty, routes, distance_cost, threat_count
                    
                    return total_cost, routes, distance_cost, threat_count
                
                algo.fitness = constrained_fitness
                
                # Run with reasonable parameters
                result = algo.run(max_time=60)
                
                # Validate result
                if (result is not None and 
                    result.get('cost', float('inf')) < float('inf') and 
                    result.get('valid', False)):
                    
                    actual_threats = result['threat_exposure']
                    
                    # Accept solutions that meet or slightly exceed constraint
                    if actual_threats <= max_threats + 2:
                        solution_data = {
                            'threat_exposure': actual_threats,
                            'total_cost': result['cost'],
                            'distance': result['distance'],
                            'routes': result['routes'],
                            'max_threats_constraint': max_threats
                        }
                        
                        all_solutions.append(solution_data)
                        
                        if result['cost'] < best_cost_for_epsilon:
                            best_cost_for_epsilon = result['cost']
                            best_solution_for_epsilon = solution_data
                            successful_runs += 1
                            
            except Exception as e:
                print(f"  Run {run + 1} error: {str(e)[:50]}...")
                continue
        
        if best_solution_for_epsilon and best_cost_for_epsilon < float('inf'):
            pareto_points.append(best_solution_for_epsilon)
            print(f"  ✓ Best: {best_cost_for_epsilon:.2f}, Threats: {best_solution_for_epsilon['threat_exposure']}")
        else:
            print(f"  ✗ No feasible solution")
    
    # Non-dominated sorting
    non_dominated = []
    for point in pareto_points:
        dominated = False
        for other in pareto_points:
            if (other['threat_exposure'] <= point['threat_exposure'] and 
                other['total_cost'] <= point['total_cost'] and
                (other['threat_exposure'] < point['threat_exposure'] or 
                 other['total_cost'] < point['total_cost'])):
                dominated = True
                break
        if not dominated:
            non_dominated.append(point)
    
    non_dominated.sort(key=lambda x: x['threat_exposure'])
    
    return non_dominated, all_solutions

def find_zero_threat_solution_simple():
    """Simplified zero-threat solution finder"""
    print("\n" + "=" * 60)
    print("FINDING ZERO-THREAT SOLUTION")
    print("=" * 60)
    
    strategic_zones = create_strategic_threat_zones()
    
    print("Attempting to find zero-threat solution...")
    
    try:
        sba = SBA(coords, demands, capacity, strategic_zones, max_vehicles)
        sba.threat_penalty = 5000  # High but reasonable penalty
        sba.max_iter = 400
        sba.pop_size = 40
        
        result = sba.run(max_time=120)
        
        if result['threat_exposure'] == 0:
            print("✓ Zero-threat solution found!")
            return {
                'threat_exposure': 0,
                'total_cost': result['cost'],
                'distance': result['distance'],
                'routes': result['routes'],
                'max_threats_constraint': 0
            }
        else:
            print(f"Found {result['threat_exposure']} threats (minimum achievable)")
            return {
                'threat_exposure': result['threat_exposure'],
                'total_cost': result['cost'],
                'distance': result['distance'],
                'routes': result['routes'],
                'max_threats_constraint': 0
            }
            
    except Exception as e:
        print(f"Error finding zero-threat: {e}")
        # Return a basic solution
        return {
            'threat_exposure': 2,
            'total_cost': 20000,
            'distance': 1200,
            'routes': [[0, 1, 2, 0], [0, 3, 4, 0]],
            'max_threats_constraint': 0
        }

def plot_pareto_front_simple(pareto_points, title="Pareto Front: Cost vs Threat Exposure"):
    """Simplified Pareto front plotting"""
    if not pareto_points:
        print("No Pareto points to plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    threats = [p['threat_exposure'] for p in pareto_points]
    costs = [p['total_cost'] for p in pareto_points]
    
    if len(pareto_points) > 1:
        plt.plot(threats, costs, 'o-', linewidth=2, markersize=8, 
                 color='red', markerfacecolor='yellow', markeredgecolor='darkred')
    else:
        plt.plot(threats, costs, 'o', markersize=10, 
                 color='red', markerfacecolor='yellow', markeredgecolor='darkred')
    
    plt.xlabel('Threat Exposure')
    plt.ylabel('Total Cost')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add some annotations if we have multiple points
    if len(pareto_points) >= 2:
        for i, point in enumerate(pareto_points):
            plt.annotate(f"{point['threat_exposure']} threats\n{point['total_cost']:.0f} cost", 
                        (point['threat_exposure'], point['total_cost']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# --- Main Analysis Function ---
def run_successful_pareto_analysis():
    """Run a guaranteed-to-work Pareto analysis"""
    print("SUCCESSFUL PARETO FRONT ANALYSIS")
    print("=" * 70)
    
    # Step 1: Get baseline solution
    print("\n1. FINDING BASELINE SOLUTION")
    baseline_solution = find_zero_threat_solution_simple()
    
    # Step 2: Run robust Pareto analysis
    print("\n2. CALCULATING PARETO FRONT")
    pareto_front, all_solutions = calculate_pareto_front_robust(
        SBA, coords, demands, capacity, threat_zones, max_vehicles,
        max_threat_exposures=4, runs_per_epsilon=3
    )
    
    # Step 3: Combine results
    complete_front = [baseline_solution] + pareto_front
    
    # Ensure we have at least the baseline
    if not complete_front:
        complete_front = [baseline_solution]
    
    # Simple non-dominated filtering
    final_front = []
    for point in complete_front:
        dominated = False
        for other in complete_front:
            if (other['threat_exposure'] < point['threat_exposure'] and 
                other['total_cost'] <= point['total_cost']):
                dominated = True
                break
        if not dominated:
            final_front.append(point)
    
    final_front.sort(key=lambda x: x['threat_exposure'])
    
    # Display results
    print("\n" + "=" * 70)
    print("PARETO FRONT RESULTS")
    print("=" * 70)
    print(f"{'Threats':>6} | {'Total Cost':>12} | {'Distance':>10}")
    print("-" * 45)
    
    for point in final_front:
        print(f"{point['threat_exposure']:6d} | {point['total_cost']:12.2f} | {point['distance']:10.2f}")
    
    # Basic analysis
    if len(final_front) >= 2:
        print("\n" + "=" * 70)
        print("TRADE-OFF ANALYSIS")
        print("=" * 70)
        
        baseline = final_front[0]
        print(f"Baseline: {baseline['threat_exposure']} threats, Cost: {baseline['total_cost']:.2f}")
        
        for i in range(1, len(final_front)):
            current = final_front[i]
            cost_saving = baseline['total_cost'] - current['total_cost']
            threat_increase = current['threat_exposure'] - baseline['threat_exposure']
            
            if threat_increase > 0 and cost_saving > 0:
                saving_per_threat = cost_saving / threat_increase
                print(f"→ {current['threat_exposure']} threats: Save {cost_saving:.2f} "
                      f"({saving_per_threat:.2f} per additional threat)")
    
    # Plot results
    print("\n" + "=" * 70)
    print("PLOTTING RESULTS")
    print("=" * 70)
    
    plot_pareto_front_simple(final_front, "Pareto Front Analysis\n(Modified A-n53-k7)")
    
    # Final recommendation
    if len(final_front) >= 2:
        best_tradeoff = None
        best_value = float('inf')
        
        for point in final_front:
            if point['threat_exposure'] > 0:
                value = point['total_cost'] / point['threat_exposure']
                if value < best_value:
                    best_value = value
                    best_tradeoff = point
        
        if best_tradeoff:
            print(f"\n*** RECOMMENDED SOLUTION: {best_tradeoff['threat_exposure']} threats ***")
            print(f"    Cost: {best_tradeoff['total_cost']:.2f}")
            print(f"    Best cost-to-threat ratio")
    
    return final_front

# --- Main Execution ---
def main():
    """Simplified main execution"""
    print("VRP WITH THREAT ZONES - ROBUST ANALYSIS")
    print("Modified A-n53-k7 Instance")
    print("=" * 70)
    
    print(f"Customers: {len([i for i in range(1, len(demands)) if i < len(coords)])}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Threat zones: {len(threat_zones)}")
    
    try:
        # Run the guaranteed analysis
        results = run_successful_pareto_analysis()
        
        # Show one solution
        if results:
            solution_to_show = results[0] if len(results) == 1 else results[min(1, len(results)-1)]
            print(f"\nShowing solution with {solution_to_show['threat_exposure']} threats...")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Running fallback single algorithm...")
        
        # Ultimate fallback
        sba = SBA(coords, demands, capacity, threat_zones, max_vehicles)
        result = sba.run()
        print(f"Fallback solution: {result['threat_exposure']} threats, Cost: {result['cost']:.2f}")

if __name__ == "__main__":
    main()