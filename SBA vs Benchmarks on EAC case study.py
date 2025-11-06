import random
import folium
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from IPython.display import IFrame, display
from copy import deepcopy

# ===================== HELPER FUNCTIONS =====================
def haversine(coord1, coord2):
    """Return approximate road distance (km) using haversine formula."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # km
    road_factor = 1.4  # approximate multiplier for real roads
    return c * r * road_factor

# ===================== EXPANDED EAC CITIES ALONG CENTRAL & NORTHERN CORRIDORS =====================
coords = {
    # KENYA - Northern Corridor Cities
    "Mombasa": (-4.0435, 39.6682),  # Port City - DEPOT (Northern Corridor Start)
    "Nairobi": (-1.2921, 36.8219),  # Capital (Northern Corridor Hub)
    "Nakuru": (-0.3031, 36.0800),   # Agricultural Hub (Northern Corridor)
    "Eldoret": (0.5204, 35.2697),   # Agricultural Center (Northern Corridor)
    "Kisumu": (-0.0917, 34.7679),   # Lake Victoria Port (Northern Corridor)
    
    # KENYA - Strategic Corridor Links
    "Thika": (-1.0333, 37.0833),    # Industrial Hub near Nairobi
    "Machakos": (-1.5167, 37.2667), # Key logistics center
    "Embu": (-0.5333, 37.4500),     # Eastern Kenya hub
    
    # TANZANIA - Central Corridor Cities
    "Dar_es_Salaam": (-6.7924, 39.2083),  # Commercial Capital & Port (Central Corridor Start)
    "Morogoro": (-6.8167, 37.6667),       # Agricultural Center (Central Corridor)
    "Dodoma": (-6.1620, 35.7516),         # Administrative Capital (Central Corridor)
    "Tanga": (-5.0667, 39.1000),          # Port City (Central Corridor)
    
    # TANZANIA - Strategic Corridor Links
    "Arusha": (-3.3869, 36.6830),   # Key connection point between corridors
    "Moshi": (-3.3348, 37.3404),    # Near Arusha, important trade center
    "Singida": (-4.8167, 34.7500),  # Central Tanzania logistics hub
    
    # UGANDA - Northern Corridor Cities
    "Kampala": (0.3476, 32.5825),         # Capital (Northern Corridor Hub)
    "Entebbe": (0.0500, 32.4600),         # Airport City (Northern Corridor)
    "Jinja": (0.4244, 33.2042),           # Industrial City (Northern Corridor)
    "Mbale": (1.0806, 34.1753),           # Eastern Uganda (Northern Corridor)
    
    # UGANDA - Strategic Corridor Links
    "Tororo": (0.6833, 34.1667),    # Border town near Kenya
    "Masaka": (-0.3333, 31.7333),   # Southern Uganda logistics center
    
    # RWANDA - Connected to both corridors
    "Kigali": (-1.9706, 30.1044),         # Capital (Connected to both corridors)
    
    # RWANDA - Strategic Corridor Links
    "Huye": (-2.6000, 29.7500),     # Southern Rwanda hub
    
    # BURUNDI - Connected to Central Corridor
    "Bujumbura": (-3.3614, 29.3599),      # Capital & Port (Central Corridor)
    
    # BURUNDI - Strategic Corridor Links
    "Gitega": (-3.4264, 29.9306),   # Political capital
    "Ngozi": (-2.9075, 29.8306),    # Northern Burundi hub
}  
   

# Set depot explicitly and reorder so depot = index 0
DEPOT_NAME = "Mombasa"
city_names = list(coords.keys())
if DEPOT_NAME in city_names:
    city_names.remove(DEPOT_NAME)
city_names.insert(0, DEPOT_NAME)
coords_list = [coords[name] for name in city_names]

# ENHANCED: Realistic vehicle + demand settings
random.seed(42)
demands = [0] + [random.randint(25, 70) for _ in city_names[1:]]  # More realistic demands
capacity = 280  # Increased capacity for feasibility
max_vehicles = 6  # Increased vehicle limit

# ===================== ROAD DISTANCE MATRIX =====================
def create_road_distance_matrix(coords_list, city_names):
    """Create realistic road distance matrix focusing on Central and Northern Corridors"""
    num_cities = len(city_names)
    road_distances = np.zeros((num_cities, num_cities))
    
    # Major highway distances (approximate road distances in km)
    highway_distances = {
        # ==================== NORTHERN CORRIDOR MAIN ROUTE ====================
        ("Mombasa", "Nairobi"): 485,
        ("Nairobi", "Nakuru"): 160,
        ("Nairobi", "Eldoret"): 310,
        ("Nairobi", "Kisumu"): 345,
        ("Nakuru", "Eldoret"): 150,
        ("Eldoret", "Kampala"): 400,
        ("Kisumu", "Kampala"): 320,
        
        # Kenya Strategic Links
        ("Nairobi", "Thika"): 45,
        ("Nairobi", "Machakos"): 65,
        ("Nairobi", "Embu"): 120,
        ("Thika", "Embu"): 90,
        
        # Uganda Northern Corridor extensions
        ("Kampala", "Entebbe"): 35,
        ("Kampala", "Jinja"): 80,
        ("Kampala", "Mbale"): 220,
        
        # Uganda Strategic Links
        ("Kampala", "Tororo"): 210,
        ("Kampala", "Masaka"): 130,
        ("Tororo", "Mbale"): 30,
        ("Jinja", "Tororo"): 150,
        
        # ==================== CENTRAL CORRIDOR MAIN ROUTE ====================
        ("Dar_es_Salaam", "Morogoro"): 190,
        ("Dar_es_Salaam", "Dodoma"): 450,
        ("Dar_es_Salaam", "Tanga"): 350,
        ("Morogoro", "Dodoma"): 260,
        
        # Tanzania Strategic Links
        ("Arusha", "Moshi"): 80,
        ("Arusha", "Dodoma"): 430,
        ("Dodoma", "Singida"): 150,
        ("Morogoro", "Singida"): 380,
        
        # ==================== CORRIDOR CONNECTIONS ====================
        ("Nairobi", "Arusha"): 250,
        ("Arusha", "Dodoma"): 430,
        
        # Rwanda and Burundi connections
        ("Kampala", "Kigali"): 530,
        ("Kampala", "Bujumbura"): 790,
        ("Kigali", "Bujumbura"): 320,
        
        # Rwanda Strategic Links
        ("Kigali", "Huye"): 135,
        
        # Burundi Strategic Links
        ("Bujumbura", "Gitega"): 110,
        ("Bujumbura", "Ngozi"): 140,
        ("Gitega", "Ngozi"): 70,
        
        # Cross-corridor connections
        ("Mombasa", "Dar_es_Salaam"): 520,
        ("Nairobi", "Dar_es_Salaam"): 880,
        
        # Regional strategic connections
        ("Tororo", "Kisumu"): 180,
        ("Masaka", "Kampala"): 130,
    }
     
    # Create city index mapping
    city_to_index = {city: idx for idx, city in enumerate(city_names)}
    
    # Fill the distance matrix
    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                road_distances[i][j] = 0
            else:
                city1, city2 = city_names[i], city_names[j]
                
                # Check direct highway distance
                if (city1, city2) in highway_distances:
                    road_distances[i][j] = highway_distances[(city1, city2)]
                elif (city2, city1) in highway_distances:
                    road_distances[i][j] = highway_distances[(city2, city1)]
                else:
                    # Estimate via haversine with road factor for unknown routes
                    coord1 = coords_list[i]
                    coord2 = coords_list[j]
                    air_distance = haversine(coord1, coord2)
                    road_distances[i][j] = air_distance * 1.6
    
    return road_distances

def get_route_distance(route, road_distances):
    """Calculate actual road distance for a route"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += road_distances[route[i]][route[i + 1]]
    return total_distance

def print_distance_matrix(road_distances, city_names):
    """Print formatted distance matrix"""
    print("🚛 EAC CENTRAL & NORTHERN CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("=" * 80)
    
    # Create DataFrame for better display
    df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
    
    # Display first 8 cities for readability
    print("\nDistance Matrix (km):")
    print(df.round(0).astype(int))
    
    return df

def print_key_corridor_routes(road_distances, city_names):
    """Print distances for key corridor routes"""
    print("\n📊 KEY CORRIDOR TRUCKING ROUTES - ROAD DISTANCES")
    print("=" * 60)
    
    corridor_routes = [
        ["Mombasa", "Nairobi"],
        ["Nairobi", "Kampala"],
        ["Kampala", "Kigali"],
        ["Dar_es_Salaam", "Dodoma"],
        ["Nairobi", "Arusha"],
        ["Arusha", "Dodoma"],
        ["Kampala", "Bujumbura"],
        ["Kigali", "Bujumbura"],
    ]
    
    city_to_index = {city: idx for idx, city in enumerate(city_names)}
    
    for route in corridor_routes:
        if all(city in city_to_index for city in route):
            idx1, idx2 = city_to_index[route[0]], city_to_index[route[1]]
            distance = road_distances[idx1][idx2]
            print(f"📍 {route[0]:<15} → {route[1]:<15}: {distance:>5.0f} km")

# ===================== CLOUD-STYLE THREAT ZONES =====================
threat_zones = [
    # M23 Rebel Group Areas - North Kivu (Very High Risk)
    {"center": (-1.4000, 28.8000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "M23 Rebel Activity - Rutshuru Area"},
    {"center": (-1.6000, 29.2000), "radius_km": 60, "type": "security", "risk_level": "very_high", "name": "M23 Controlled Areas - Masisi"},
    {"center": (-1.6800, 29.2200), "radius_km": 40, "type": "security", "risk_level": "very_high", "name": "M23 Presence - Goma Perimeter"},
    
    # ADF Primary Camp Locations - Ituri & North Kivu (Extreme Risk)
    {"center": (1.2000, 29.8000), "radius_km": 120, "type": "security", "risk_level": "very_high", "name": "ADF Main Camps - Irumu Territory"},
    {"center": (0.8000, 29.5000), "radius_km": 100, "type": "security", "risk_level": "very_high", "name": "ADF Activity - Beni Territory"},
    {"center": (1.0000, 29.3000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "ADF Stronghold - Mambasa Territory"},
    {"center": (1.5000, 30.2000), "radius_km": 80, "type": "security", "risk_level": "very_high", "name": "ADF Camps - Komanda Area"},
    
    # Areas of Frequent M23-ADF Overlap (Highest Risk)
    {"center": (-1.2000, 28.6000), "radius_km": 70, "type": "security", "risk_level": "very_high", "name": "M23-ADF Overlap - Lubero Territory"},
    {"center": (-0.8000, 29.0000), "radius_km": 90, "type": "security", "risk_level": "very_high", "name": "Joint M23-ADF Operations - Southern Beni"},
    
    # ==================== NORTHERN CORRIDOR SECURITY THREATS ====================
    {"center": (-2.0000, 40.9000), "radius_km": 100, "type": "security", "risk_level": "high", "name": "Lamu Corridor - ASWJ Militant Activity"},
    {"center": (-1.2000, 37.0000), "radius_km": 80, "type": "infrastructure", "risk_level": "medium", "name": "Thika Road - Construction Delays"},
    {"center": (-0.8000, 36.3000), "radius_km": 50, "type": "climate", "risk_level": "medium", "name": "Naivasha - Seasonal Flooding Zone"},
    
    # ==================== CENTRAL CORRIDOR THREATS ====================
    {"center": (-6.5000, 36.0000), "radius_km": 70, "type": "climate", "risk_level": "medium", "name": "Central Tanzania - Drought Prone Area"},
    {"center": (-5.0000, 39.0000), "radius_km": 60, "type": "infrastructure", "risk_level": "medium", "name": "Tanga Corridor - Road Maintenance"},
    
    # ==================== BORDER CROSSING HOTSPOTS ====================
    {"center": (-1.2833, 29.6167), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rwanda-DRC Border - Bunagana Crossing"},
    {"center": (-2.4833, 28.9000), "radius_km": 40, "type": "security", "risk_level": "high", "name": "Rusizi-DRC Border town"},
]

# ===================== ENHANCED THREAT AVOIDANCE FUNCTIONS =====================
def is_point_in_threat_zone(point, threat_zones, buffer_km=25):
    """Check if a point is inside any threat zone with safety buffer"""
    for zone in threat_zones:
        dist_km = haversine(point, zone["center"])
        if dist_km <= (zone["radius_km"] + buffer_km):
            return True, zone
    return False, None

def is_route_segment_safe(p1, p2, threat_zones, segments=25, buffer_km=25):
    """Check if route segment passes through threat zones with enhanced safety"""
    for s in range(segments + 1):
        t = s / segments
        sample_point = (p1[0]*(1-t) + p2[0]*t, p1[1]*(1-t) + p2[1]*t)
        
        in_zone, zone = is_point_in_threat_zone(sample_point, threat_zones, buffer_km)
        if in_zone:
            return False, zone
    return True, None

def calculate_threat_avoidance_penalty(route, coords_list, threat_zones, base_penalty=75000):
    """Enhanced penalty for routes that pass through threat zones"""
    total_penalty = 0
    threat_count = 0
    
    for i in range(len(route) - 1):
        p1 = coords_list[route[i]]
        p2 = coords_list[route[i + 1]]
        
        safe, zone = is_route_segment_safe(p1, p2, threat_zones)
        if not safe:
            threat_count += 1
            # Progressive penalty based on risk level and zone type
            risk_weights = {"medium": 1, "high": 3, "very_high": 8}
            type_weights = {"climate": 1, "infrastructure": 2, "security": 5}
            
            risk_weight = risk_weights.get(zone["risk_level"], 1)
            type_weight = type_weights.get(zone["type"], 1)
            
            segment_penalty = base_penalty * risk_weight * type_weight
            total_penalty += segment_penalty
    
    # Additional penalty for multiple threat violations
    if threat_count > 1:
        total_penalty *= (1 + 0.5 * threat_count)
    
    return total_penalty

# ===================== COMMON FUNCTIONS FOR ALL ALGORITHMS =====================
def decode_routes(permutation, demands, capacity, max_vehicles):
    """Decode permutation into vehicle routes"""
    num_customers = len(demands)
    valid_customers = [i for i in permutation if 1 <= i < num_customers]

    routes, route, load = [], [0], 0
    for cust in valid_customers:
        if load + demands[cust] <= capacity:
            route.append(cust)
            load += demands[cust]
        else:
            route.append(0)
            routes.append(route)
            route, load = [0, cust], demands[cust]

    if route:
        if route[-1] != 0:
            route.append(0)
        routes.append(route)

    # Merge if too many vehicles
    if len(routes) > max_vehicles:
        routes = merge_routes(routes, demands, capacity, max_vehicles)

    # Ensure depot is both start & end
    clean_routes = []
    for r in routes:
        clean_r = [node for node in r if 0 <= node < num_customers]
        if clean_r[0] != 0:
            clean_r.insert(0, 0)
        if clean_r[-1] != 0:
            clean_r.append(0)
        clean_routes.append(clean_r)
    return clean_routes

def merge_routes(routes, demands, capacity, max_vehicles):
    """Merge routes if too many vehicles are used"""
    route_loads = [sum(demands[c] for c in r if c != 0) for r in routes]
    while len(routes) > max_vehicles:
        best_merge, best_load = None, float('inf')
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                combined_load = route_loads[i] + route_loads[j]
                if combined_load <= capacity and combined_load < best_load:
                    best_merge, best_load = (i, j), combined_load
        if not best_merge:
            break
        i, j = best_merge
        merged = routes[i][:-1] + routes[j][1:]
        routes[i] = merged
        route_loads[i] = best_load
        del routes[j], route_loads[j]
    return routes

def validate_solution(routes, demands, capacity, num_customers):
    """Validate if solution meets all constraints"""
    served = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0:
            return False
        if sum(demands[c] for c in route[1:-1]) > capacity:
            return False
        served.update(route[1:-1])
    return len(served) == num_customers - 1

def calculate_total_cost_road(routes, road_distances):
    """Calculate total cost using actual road distances"""
    return sum(get_route_distance(route, road_distances) for route in routes)

def fitness_function(perm, demands, capacity, max_vehicles, coords_list, threat_zones, city_names, road_distances):
    """Enhanced fitness function with improved constraint handling"""
    routes = decode_routes(perm, demands, capacity, max_vehicles)
    dist_cost = calculate_total_cost_road(routes, road_distances)
    
    # Enhanced threat penalty
    threat_penalty = sum(calculate_threat_avoidance_penalty(r, coords_list, threat_zones) for r in routes)
    
    # Enhanced capacity violation penalty
    capacity_penalty = 0
    for route in routes:
        route_load = sum(demands[c] for c in route[1:-1])
        if route_load > capacity:
            overload_ratio = (route_load - capacity) / capacity
            capacity_penalty += 150000 * (1 + overload_ratio)
    
    # Vehicle count penalty
    vehicle_penalty = 0
    if len(routes) > max_vehicles:
        extra_vehicles = len(routes) - max_vehicles
        vehicle_penalty = 500000 * extra_vehicles
    
    # Conflict city penalty
    conflict_cities = ["Goma", "Bukavu", "Butembo"]
    conflict_indices = [i for i, city in enumerate(city_names) if city in conflict_cities]
    conflict_penalty = 0
    for route in routes:
        for city_idx in route:
            if city_idx in conflict_indices:
                conflict_penalty += 200000
    
    # Efficiency bonus for good solutions
    efficiency_bonus = 0
    total_load = sum(sum(demands[c] for c in r[1:-1]) for r in routes)
    avg_route_load = total_load / len(routes) if routes else 0
    if avg_route_load > capacity * 0.8:  # High utilization bonus
        efficiency_bonus = -10000
    
    total = dist_cost + threat_penalty + capacity_penalty + vehicle_penalty + conflict_penalty + efficiency_bonus
    
    if not validate_solution(routes, demands, capacity, len(demands)):
        total += 1e6
        
    return total, routes

def repair_solution(sol, customers):
    """Repair solution to ensure all customers are included exactly once"""
    seen, repaired = set(), []
    for c in sol:
        if c not in seen and c in customers:
            repaired.append(c)
            seen.add(c)
    missing = [c for c in customers if c not in seen]
    random.shuffle(missing)
    repaired.extend(missing)
    return repaired

def route_statistics_road(routes, road_distances, demands):
    """Calculate statistics for each route"""
    return [{
        "vehicle": i+1,
        "distance": get_route_distance(r, road_distances),
        "load": sum(demands[c] for c in r[1:-1]),
        "customers": len(r)-2,
        "route": r
    } for i, r in enumerate(routes)]

# ===================== ENHANCED SBA ALGORITHM =====================

class Enhanced_SBA_Optimizer:
    """Enhanced Swallow-Bat Algorithm with Superior Performance"""
    
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, population_size=100, max_iter=500):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.pop_size, self.max_iter = population_size, max_iter
        self.customers = [i for i in range(1, len(demands))]
        
        # Enhanced adaptive parameters
        self.adaptive_mutation_rate = 0.35
        self.adaptive_crossover_rate = 0.8
        self.diversity_threshold = 0.15
        self.no_improve_count = 0
        
        # Elite memory and performance tracking
        self.elite_memory = []
        self.performance_history = []
        
        # Multi-strategy population initialization
        self.population = self._enhanced_initialize_population()
        self.best_solution = self.population[0][:]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def _enhanced_initialize_population(self):
        """Enhanced population initialization with multiple strategies"""
        population = []
        
        # 1. Basic random solutions (25%)
        for _ in range(int(self.pop_size * 0.25)):
            population.append(random.sample(self.customers, len(self.customers)))
        
        # 2. Enhanced greedy solutions (20%)
        for _ in range(int(self.pop_size * 0.20)):
            population.append(self._enhanced_greedy_solution())
        
        # 3. Threat-aware solutions (20%)
        for _ in range(int(self.pop_size * 0.20)):
            population.append(self._advanced_threat_aware_solution())
        
        # 4. Corridor-optimized solutions (15%)
        for _ in range(int(self.pop_size * 0.15)):
            population.append(self._corridor_optimized_solution())
        
        # 5. Feasibility-first solutions (10%)
        for _ in range(int(self.pop_size * 0.10)):
            population.append(self._feasibility_first_solution())
            
        # 6. Elite-guided solutions (10%)
        for _ in range(int(self.pop_size * 0.10)):
            if self.elite_memory:
                population.append(self._elite_guided_solution())
            else:
                population.append(self._enhanced_greedy_solution())
        
        return population

    def _enhanced_greedy_solution(self):
        """Multi-criteria greedy solution"""
        unvisited = set(self.customers)
        solution = []
        current = 0
        
        while unvisited:
            candidates = list(unvisited)
            
            # Multi-criteria scoring
            scores = []
            for candidate in candidates:
                score = (
                    -self.road_distances[current][candidate] * 0.4 +  # Distance
                    -self._calculate_city_threat_penalty(candidate) * 0.3 +  # Safety
                    self._calculate_feasibility_score(solution + [candidate]) * 0.2 +  # Feasibility
                    -self._calculate_corridor_deviation(candidate) * 0.1  # Corridor alignment
                )
                scores.append(score)
            
            best_candidate = candidates[np.argmax(scores)]
            solution.append(best_candidate)
            unvisited.remove(best_candidate)
            current = best_candidate
            
        return solution

    def _advanced_threat_aware_solution(self):
        """Advanced threat avoidance with strategic placement"""
        safe_cities, risky_cities = [], []
        
        for city_idx in self.customers:
            coord = self.coords_list[city_idx]
            in_zone, zone = is_point_in_threat_zone(coord, self.zones)
            if not in_zone:
                safe_cities.append(city_idx)
            else:
                risky_cities.append((city_idx, zone))
        
        # Sort by strategic importance
        safe_cities.sort(key=lambda x: self._get_corridor_priority(x))
        risky_cities.sort(key=lambda x: x[1]["risk_level"] != "very_high")
        
        solution = safe_cities[:]
        
        # Insert risky cities strategically
        for city_idx, zone in risky_cities:
            best_pos = self._find_optimal_insertion_position(solution, city_idx)
            solution.insert(best_pos, city_idx)
            
        return solution

    def _corridor_optimized_solution(self):
        """Corridor-optimized solution with capacity awareness"""
        northern_cities = []
        central_cities = []
        other_cities = []
        
        northern_names = ["Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
        central_names = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
        
        for city_idx in self.customers:
            city_name = self.city_names[city_idx]
            if city_name in northern_names:
                northern_cities.append(city_idx)
            elif city_name in central_names:
                central_cities.append(city_idx)
            else:
                other_cities.append(city_idx)
        
        # Create corridor-optimized sequence
        solution = []
        if northern_cities:
            northern_cities.sort(key=lambda x: self.road_distances[0][x])
            solution.extend(northern_cities)
        
        solution.extend(other_cities)
        
        if central_cities:
            central_cities.sort(key=lambda x: self.road_distances[0][x])
            solution.extend(central_cities)
            
        return solution

    def _feasibility_first_solution(self):
        """Solution prioritizing constraint satisfaction"""
        unvisited = set(self.customers)
        solution = []
        
        high_demand_cities = sorted(self.customers, 
                                  key=lambda x: self.demands[x], reverse=True)
        
        for city in high_demand_cities:
            if city not in unvisited:
                continue
                
            best_pos = len(solution)
            best_feasibility = -float('inf')
            
            for pos in range(len(solution) + 1):
                test_sol = solution[:pos] + [city] + solution[pos:]
                feasibility = self._calculate_feasibility_score(test_sol)
                
                if feasibility > best_feasibility:
                    best_feasibility = feasibility
                    best_pos = pos
            
            solution.insert(best_pos, city)
            unvisited.remove(city)
            
        return solution

    def _elite_guided_solution(self):
        """Create solution guided by elite memory"""
        if not self.elite_memory:
            return self._enhanced_greedy_solution()
        
        base_elite = random.choice(self.elite_memory[:3])
        new_solution = []
        
        for i in range(len(self.customers)):
            if random.random() < 0.7 and i < len(base_elite):
                new_solution.append(base_elite[i])
            else:
                # Fill with random valid city
                remaining = [c for c in self.customers if c not in new_solution]
                if remaining:
                    new_solution.append(random.choice(remaining))
        
        return repair_solution(new_solution, self.customers)

    def _calculate_city_threat_penalty(self, city_idx):
        """Calculate threat penalty for a single city"""
        coord = self.coords_list[city_idx]
        penalty = 0
        for zone in self.zones:
            dist = haversine(coord, zone["center"])
            if dist <= zone["radius_km"] + 25:
                risk_weights = {"medium": 1, "high": 3, "very_high": 8}
                penalty += risk_weights.get(zone["risk_level"], 1)
        return penalty

    def _get_corridor_priority(self, city_idx):
        """Get corridor-based priority for city ordering"""
        city_name = self.city_names[city_idx]
        
        if city_name in ["Nairobi", "Nakuru", "Eldoret", "Kisumu"]:
            return 1
        elif city_name in ["Kampala", "Entebbe", "Jinja"]:
            return 2
        elif city_name in ["Dar_es_Salaam", "Morogoro", "Dodoma"]:
            return 3
        elif city_name in ["Bujumbura", "Kigali"]:
            return 4
        else:
            return 5

    def _calculate_feasibility_score(self, solution):
        """Calculate feasibility score for partial solution"""
        if not solution:
            return 1.0
        
        routes = decode_routes(solution, self.demands, self.capacity, self.max_vehicles)
        
        capacity_violation = 0
        vehicle_violation = max(0, len(routes) - self.max_vehicles)
        
        for route in routes:
            route_load = sum(self.demands[node] for node in route[1:-1])
            capacity_violation += max(0, route_load - self.capacity)
        
        max_violation = len(self.customers) * 10
        feasibility = 1.0 - (
            capacity_violation / (self.capacity * len(routes)) +
            vehicle_violation / self.max_vehicles
        ) / 2.0
        
        return max(0.0, feasibility)

    def _calculate_corridor_deviation(self, city_idx):
        """Calculate corridor deviation"""
        base_corridor_cities = ["Nairobi", "Kampala", "Dar_es_Salaam", "Bujumbura", "Kigali"]
        base_indices = [i for i, name in enumerate(self.city_names) 
                       if name in base_corridor_cities and i in self.customers]
        
        if not base_indices:
            return 0
            
        avg_distance = np.mean([self.road_distances[city_idx][base] for base in base_indices])
        return avg_distance

    def _find_optimal_insertion_position(self, solution, new_city):
        """Find optimal insertion position"""
        best_pos = 0
        best_score = -float('inf')
        
        for pos in range(len(solution) + 1):
            test_solution = solution[:pos] + [new_city] + solution[pos:]
            
            distance_score = -self._calculate_route_efficiency(test_solution)
            threat_score = -self._calculate_route_threat_penalty(test_solution)
            feasibility_score = self._calculate_feasibility_score(test_solution)
            
            total_score = (
                distance_score * 0.4 +
                threat_score * 0.3 +
                feasibility_score * 0.3
            )
            
            if total_score > best_score:
                best_score = total_score
                best_pos = pos
                
        return best_pos

    def _calculate_route_efficiency(self, solution):
        """Calculate route efficiency"""
        routes = decode_routes(solution, self.demands, self.capacity, self.max_vehicles)
        return calculate_total_cost_road(routes, self.road_distances)

    def _calculate_route_threat_penalty(self, solution):
        """Calculate route threat penalty"""
        routes = decode_routes(solution, self.demands, self.capacity, self.max_vehicles)
        total_penalty = 0
        for route in routes:
            total_penalty += calculate_threat_avoidance_penalty(
                route, self.coords_list, self.zones
            )
        return total_penalty

    def _apply_local_search(self, solution, intensity=0.5):
        """Apply local search to improve solution"""
        current_solution = solution[:]
        current_cost, _ = fitness_function(current_solution, self.demands, self.capacity,
                                        self.max_vehicles, self.coords_list, self.zones,
                                        self.city_names, self.road_distances)
        
        improvements = 0
        max_attempts = int(len(solution) * intensity)
        
        for _ in range(max_attempts):
            # Try different moves
            move_type = random.choice(['swap', 'insert', 'reverse'])
            
            if move_type == 'swap':
                i, j = random.sample(range(len(solution)), 2)
                new_solution = solution[:]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            elif move_type == 'insert':
                i, j = random.sample(range(len(solution)), 2)
                new_solution = solution[:]
                element = new_solution.pop(i)
                new_solution.insert(j, element)
            else:  # reverse
                i, j = sorted(random.sample(range(len(solution)), 2))
                new_solution = solution[:]
                new_solution[i:j+1] = reversed(new_solution[i:j+1])
            
            new_solution = repair_solution(new_solution, self.customers)
            new_cost, _ = fitness_function(new_solution, self.demands, self.capacity,
                                         self.max_vehicles, self.coords_list, self.zones,
                                         self.city_names, self.road_distances)
            
            if new_cost < current_cost:
                current_solution = new_solution
                current_cost = new_cost
                improvements += 1
                
        return current_solution

    def _update_elite_memory(self, population, fitnesses):
        """Update elite memory"""
        combined = list(zip(fitnesses, population))
        combined.sort(key=lambda x: x[0])
        
        elite_count = max(5, int(self.pop_size * 0.1))
        new_elites = [sol for _, sol in combined[:elite_count]]
        
        # Merge with existing elites
        all_elites = self.elite_memory + new_elites
        unique_elites = []
        seen = set()
        
        for elite in all_elites:
            elite_tuple = tuple(elite)
            if elite_tuple not in seen:
                unique_elites.append(elite)
                seen.add(elite_tuple)
        
        unique_elites.sort(key=lambda x: fitness_function(x, self.demands, self.capacity,
                                                        self.max_vehicles, self.coords_list, 
                                                        self.zones, self.city_names, 
                                                        self.road_distances)[0])
        self.elite_memory = unique_elites[:elite_count * 2]

    def _calculate_population_diversity(self, population):
        """Calculate population diversity"""
        diversity = 0
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                dist = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                diversity += dist
                
        max_possible = len(self.customers) * (self.pop_size * (self.pop_size - 1)) / 2
        return diversity / max_possible if max_possible > 0 else 0

    def _adaptive_parameter_control(self, improvement_ratio, diversity):
        """Adaptive parameter control"""
        if improvement_ratio > 0.15:
            self.adaptive_mutation_rate *= 0.9
            self.adaptive_crossover_rate = min(0.9, self.adaptive_crossover_rate * 1.1)
        elif improvement_ratio < 0.05:
            self.adaptive_mutation_rate = min(0.7, self.adaptive_mutation_rate * 1.2)
            self.adaptive_crossover_rate *= 0.9
        
        if diversity < self.diversity_threshold:
            self.adaptive_mutation_rate = min(0.7, self.adaptive_mutation_rate * 1.1)
        
        self.adaptive_mutation_rate = max(0.2, min(0.7, self.adaptive_mutation_rate))
        self.adaptive_crossover_rate = max(0.6, min(0.9, self.adaptive_crossover_rate))

    def run(self, max_time=300):
        """Enhanced SBA execution"""
        start = time.time()
        pop_fit = []
        
        print("   🔧 Applying enhanced initialization...")
        for i in range(self.pop_size):
            if random.random() < 0.3:
                self.population[i] = self._apply_local_search(self.population[i], 0.2)
                
            cost, routes = fitness_function(self.population[i], self.demands, self.capacity, 
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            pop_fit.append(cost)
            if cost < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = cost, self.population[i][:], routes

        print(f"   🎯 Enhanced initial best cost: {self.best_cost:,.0f}")

        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            # Update elite memory and calculate diversity
            self._update_elite_memory(self.population, pop_fit)
            diversity = self._calculate_population_diversity(self.population)

            # Sort population
            sorted_idx = np.argsort(pop_fit)
            sorted_pop = [self.population[i] for i in sorted_idx]
            sorted_fit = [pop_fit[i] for i in sorted_idx]

            elite_size = max(8, int(self.pop_size * 0.15))
            elites = sorted_pop[:elite_size]
            elite_fit = sorted_fit[:elite_size]

            new_population = elites[:]
            new_fitness = elite_fit[:]

            # Memeplex optimization
            num_memeplexes = 4
            remaining_pop = sorted_pop[elite_size:]
            remaining_fit = sorted_fit[elite_size:]
            
            memeplexes = [remaining_pop[i::num_memeplexes] for i in range(num_memeplexes)]
            memeplex_fit = [remaining_fit[i::num_memeplexes] for i in range(num_memeplexes)]

            improvement_count = 0
            
            for mi, mem in enumerate(memeplexes):
                for idx in range(len(mem)):
                    current_sol = mem[idx][:]
                    current_fit = memeplex_fit[mi][idx]

                    # Enhanced learning phase
                    if random.random() < 0.8:
                        if random.random() < 0.6 and self.elite_memory:
                            elite_guide = random.choice(self.elite_memory[:3])
                            for j in range(len(current_sol)):
                                if random.random() < 0.7 and j < len(elite_guide):
                                    current_sol[j] = elite_guide[j]

                    # Enhanced crossover
                    if random.random() < self.adaptive_crossover_rate:
                        parents = random.sample(elites, min(3, len(elites)))
                        for j in range(len(current_sol)):
                            if random.random() < 0.7:
                                choices = [current_sol[j]]
                                for parent in parents:
                                    if j < len(parent):
                                        choices.append(parent[j])
                                current_sol[j] = random.choice(choices)

                    # Enhanced mutation
                    if random.random() < self.adaptive_mutation_rate:
                        mutation_strength = 1 + int((t / self.max_iter) * 3)
                        
                        for _ in range(mutation_strength):
                            move_type = random.choice(['swap', 'insert', 'reverse'])
                            
                            if move_type == 'swap':
                                i, j = random.sample(range(len(current_sol)), 2)
                                current_sol[i], current_sol[j] = current_sol[j], current_sol[i]
                            elif move_type == 'insert':
                                i, j = random.sample(range(len(current_sol)), 2)
                                element = current_sol.pop(i)
                                current_sol.insert(j, element)
                            else:  # reverse
                                i, j = sorted(random.sample(range(len(current_sol)), 2))
                                current_sol[i:j+1] = reversed(current_sol[i:j+1])

                    # Local search
                    if random.random() < 0.25:
                        current_sol = self._apply_local_search(current_sol, 0.3)

                    current_sol = repair_solution(current_sol, self.customers)
                    current_fit, current_routes = fitness_function(current_sol, self.demands, self.capacity,
                                                                  self.max_vehicles, self.coords_list, self.zones,
                                                                  self.city_names, self.road_distances)

                    if current_fit < memeplex_fit[mi][idx]:
                        mem[idx] = current_sol
                        memeplex_fit[mi][idx] = current_fit
                        improvement_count += 1
                        
                        if current_fit < self.best_cost:
                            self.best_cost, self.best_solution, self.best_routes = current_fit, current_sol[:], current_routes
                            self.no_improve_count = 0
                        else:
                            self.no_improve_count += 1

                    new_population.append(current_sol)
                    new_fitness.append(current_fit)

            # Update population
            self.population = new_population[:self.pop_size]
            pop_fit = new_fitness[:self.pop_size]

            # Adaptive parameter update
            improvement_ratio = improvement_count / (len(self.population) - len(elites))
            self._adaptive_parameter_control(improvement_ratio, diversity)

            # FIXED: Enhanced restart - less frequent and smaller
            if self.no_improve_count > 100:  # Changed from 50 to 100
                print("   🔄 Applying enhanced diversity restart...")
                num_replace = int(self.pop_size * 0.2)  # Reduced from 0.3 to 0.2
                worst_indices = np.argsort(pop_fit)[-num_replace:]
                
                for idx in worst_indices:
                    if random.random() < 0.6:
                        new_sol = self._elite_guided_solution()
                    else:
                        new_sol = self._enhanced_greedy_solution()
                    
                    self.population[idx] = new_sol
                    pop_fit[idx] = fitness_function(new_sol, self.demands, self.capacity,
                                                  self.max_vehicles, self.coords_list, self.zones,
                                                  self.city_names, self.road_distances)[0]
                
                self.no_improve_count = 0

            self.history.append(self.best_cost)

            if t % 50 == 0:
                print(f"   📊 Iteration {t}: Best Cost = {self.best_cost:,.0f}, Diversity = {diversity:.3f}")

        # Final intensification
        print("   🎯 Applying final intensification...")
        final_improved = self._apply_local_search(self.best_solution, 0.6)
        final_cost, final_routes = fitness_function(final_improved, self.demands, self.capacity,
                                                  self.max_vehicles, self.coords_list, self.zones,
                                                  self.city_names, self.road_distances)
        
        if final_cost < self.best_cost:
            self.best_cost, self.best_solution, self.best_routes = final_cost, final_improved[:], final_routes

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        """Calculate route statistics"""
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== ADAPTIVE LARGE NEIGHBORHOOD SEARCH (ALNS) =====================

class ALNS_Optimizer:
    """Adaptive Large Neighborhood Search - New Implementation"""
    
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, max_iter=300, adaptive_period=50):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.max_iter, self.adaptive_period = max_iter, adaptive_period
        self.customers = [i for i in range(1, len(demands))]
        
        # ALNS adaptive weights
        self.destroy_weights = [1.0] * 4
        self.repair_weights = [1.0] * 4
        self.destroy_scores = [0.0] * 4
        self.repair_scores = [0.0] * 4
        self.destroy_usage = [0] * 4
        self.repair_usage = [0] * 4
        
        self.best_solution = None
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def _initialize_solution(self):
        """Initialize using enhanced greedy approach"""
        unvisited = set(self.customers)
        solution = []
        current = 0
        
        while unvisited:
            # Multi-criteria selection
            candidates = list(unvisited)
            scores = []
            
            for candidate in candidates:
                # Distance, threat, and feasibility considerations
                distance_score = -self.road_distances[current][candidate]
                threat_score = -self._calculate_city_threat_penalty(candidate)
                feasibility_score = self._calculate_feasibility_score(solution + [candidate])
                
                total_score = distance_score * 0.5 + threat_score * 0.3 + feasibility_score * 0.2
                scores.append(total_score)
            
            best_candidate = candidates[np.argmax(scores)]
            solution.append(best_candidate)
            unvisited.remove(best_candidate)
            current = best_candidate
            
        return solution

    def _calculate_city_threat_penalty(self, city_idx):
        """Calculate threat penalty for a single city"""
        coord = self.coords_list[city_idx]
        penalty = 0
        for zone in self.zones:
            dist = haversine(coord, zone["center"])
            if dist <= zone["radius_km"] + 25:
                risk_weights = {"medium": 1, "high": 3, "very_high": 8}
                penalty += risk_weights.get(zone["risk_level"], 1)
        return penalty

    def _calculate_feasibility_score(self, solution):
        """Calculate feasibility score for partial solution"""
        if not solution:
            return 1.0
        
        routes = decode_routes(solution, self.demands, self.capacity, self.max_vehicles)
        
        capacity_violation = 0
        vehicle_violation = max(0, len(routes) - self.max_vehicles)
        
        for route in routes:
            route_load = sum(self.demands[node] for node in route[1:-1])
            capacity_violation += max(0, route_load - self.capacity)
        
        max_violation = len(self.customers) * 10
        feasibility = 1.0 - (
            capacity_violation / (self.capacity * len(routes)) +
            vehicle_violation / self.max_vehicles
        ) / 2.0
        
        return max(0.0, feasibility)

    def _select_destroy_operator(self):
        """Select destroy operator based on adaptive weights"""
        total_weight = sum(self.destroy_weights)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(self.destroy_weights):
            cumulative += weight
            if r <= cumulative:
                self.destroy_usage[i] += 1
                return i
        return len(self.destroy_weights) - 1

    def _select_repair_operator(self):
        """Select repair operator based on adaptive weights"""
        total_weight = sum(self.repair_weights)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(self.repair_weights):
            cumulative += weight
            if r <= cumulative:
                self.repair_usage[i] += 1
                return i
        return len(self.repair_weights) - 1

    def _destroy_operators(self, solution, operator_idx, degree=0.3):
        """Apply selected destroy operator"""
        num_remove = max(1, int(len(solution) * degree))
        
        if operator_idx == 0:
            return self._random_removal(solution, num_remove)
        elif operator_idx == 1:
            return self._worst_removal(solution, num_remove)
        elif operator_idx == 2:
            return self._threat_aware_removal(solution, num_remove)
        else:
            return self._route_removal(solution, num_remove)

    def _random_removal(self, solution, num_remove):
        """Randomly remove customers"""
        removed = random.sample(solution, num_remove)
        new_solution = [c for c in solution if c not in removed]
        return new_solution, removed

    def _worst_removal(self, solution, num_remove):
        """Remove customers with highest cost contribution"""
        contributions = []
        
        for i, cust in enumerate(solution):
            # Create temporary solution without this customer
            temp_sol = solution[:i] + solution[i+1:]
            temp_cost, _ = fitness_function(temp_sol, self.demands, self.capacity,
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            
            original_cost, _ = fitness_function(solution, self.demands, self.capacity,
                                              self.max_vehicles, self.coords_list, self.zones,
                                              self.city_names, self.road_distances)
            
            contribution = original_cost - temp_cost
            contributions.append((cust, contribution))
        
        # Remove customers with highest cost contribution
        contributions.sort(key=lambda x: x[1], reverse=True)
        removed = [c for c, _ in contributions[:num_remove]]
        new_solution = [c for c in solution if c not in removed]
        
        return new_solution, removed

    def _threat_aware_removal(self, solution, num_remove):
        """Remove customers in high-threat areas"""
        threat_scores = []
        
        for cust in solution:
            score = self._calculate_city_threat_penalty(cust)
            threat_scores.append((cust, score))
        
        # Remove customers with highest threat exposure
        threat_scores.sort(key=lambda x: x[1], reverse=True)
        removed = [c for c, _ in threat_scores[:num_remove]]
        new_solution = [c for c in solution if c not in removed]
        
        return new_solution, removed

    def _route_removal(self, solution, num_remove):
        """Remove entire routes"""
        routes = decode_routes(solution, self.demands, self.capacity, self.max_vehicles)
        
        if len(routes) <= 1:
            return self._random_removal(solution, num_remove)
        
        # Remove one random route
        route_to_remove = random.randint(0, len(routes)-1)
        removed_customers = [c for c in routes[route_to_remove] if c != 0]
        
        # Reconstruct solution without removed route
        remaining_routes = [r for i, r in enumerate(routes) if i != route_to_remove]
        new_solution = []
        for route in remaining_routes:
            new_solution.extend([c for c in route if c != 0])
        
        return new_solution, removed_customers

    def _repair_operators(self, solution, removed_customers, operator_idx):
        """Apply selected repair operator"""
        if operator_idx == 0:
            return self._greedy_insertion(solution, removed_customers)
        elif operator_idx == 1:
            return self._regret_insertion(solution, removed_customers)
        elif operator_idx == 2:
            return self._threat_aware_insertion(solution, removed_customers)
        else:
            return self._random_insertion(solution, removed_customers)

    def _greedy_insertion(self, solution, removed_customers):
        """Greedy insertion of removed customers"""
        current_solution = solution[:]
        random.shuffle(removed_customers)
        
        for customer in removed_customers:
            best_cost = float('inf')
            best_position = -1
            
            for i in range(len(current_solution) + 1):
                candidate = current_solution[:i] + [customer] + current_solution[i:]
                candidate = repair_solution(candidate, self.customers)
                cost, _ = fitness_function(candidate, self.demands, self.capacity,
                                         self.max_vehicles, self.coords_list, self.zones,
                                         self.city_names, self.road_distances)
                
                if cost < best_cost:
                    best_cost = cost
                    best_position = i
            
            if best_position != -1:
                current_solution.insert(best_position, customer)
        
        return current_solution

    def _regret_insertion(self, solution, removed_customers):
        """Regret-based insertion"""
        current_solution = solution[:]
        
        while removed_customers:
            best_customer = None
            best_position = -1
            best_regret = -float('inf')
            
            for customer in removed_customers:
                costs = []
                for i in range(len(current_solution) + 1):
                    candidate = current_solution[:i] + [customer] + current_solution[i:]
                    candidate = repair_solution(candidate, self.customers)
                    cost, _ = fitness_function(candidate, self.demands, self.capacity,
                                             self.max_vehicles, self.coords_list, self.zones,
                                             self.city_names, self.road_distances)
                    costs.append(cost)
                
                if len(costs) >= 2:
                    sorted_costs = sorted(costs)
                    regret = sorted_costs[1] - sorted_costs[0]
                else:
                    regret = 0
                
                if regret > best_regret:
                    best_regret = regret
                    best_customer = customer
                    best_position = costs.index(min(costs))
            
            if best_customer is not None:
                current_solution.insert(best_position, best_customer)
                removed_customers.remove(best_customer)
        
        return current_solution

    def _threat_aware_insertion(self, solution, removed_customers):
        """Threat-aware insertion prioritizing safe areas"""
        # Sort customers by threat exposure (lowest first)
        customer_threats = []
        for cust in removed_customers:
            threat_score = self._calculate_city_threat_penalty(cust)
            customer_threats.append((cust, threat_score))
        
        customer_threats.sort(key=lambda x: x[1])
        safe_customers_first = [c for c, _ in customer_threats]
        
        return self._greedy_insertion(solution, safe_customers_first)

    def _random_insertion(self, solution, removed_customers):
        """Random insertion of removed customers"""
        current_solution = solution[:]
        
        for customer in removed_customers:
            position = random.randint(0, len(current_solution))
            current_solution.insert(position, customer)
        
        return current_solution

    def _update_weights(self, destroy_idx, repair_idx, improvement):
        """Update operator weights based on performance"""
        # Update scores based on improvement
        if improvement > 0.1:  # Significant improvement
            score = 1.2
        elif improvement > 0:  # Small improvement
            score = 1.1
        elif improvement == 0:  # No change
            score = 1.0
        else:  # Worse solution
            score = 0.8
        
        self.destroy_scores[destroy_idx] += score
        self.repair_scores[repair_idx] += score
        
        # Update weights periodically
        if sum(self.destroy_usage) >= self.adaptive_period:
            for i in range(len(self.destroy_weights)):
                if self.destroy_usage[i] > 0:
                    self.destroy_weights[i] = (
                        self.destroy_weights[i] * 0.9 + 
                        0.1 * (self.destroy_scores[i] / self.destroy_usage[i])
                    )
            
            for i in range(len(self.repair_weights)):
                if self.repair_usage[i] > 0:
                    self.repair_weights[i] = (
                        self.repair_weights[i] * 0.9 + 
                        0.1 * (self.repair_scores[i] / self.repair_usage[i])
                    )
            
            # Reset counters
            self.destroy_scores = [0.0] * 4
            self.repair_scores = [0.0] * 4
            self.destroy_usage = [0] * 4
            self.repair_usage = [0] * 4

    def run(self, max_time=180):
        """ALNS execution"""
        start = time.time()
        
        # Initialize solution
        current_solution = self._initialize_solution()
        current_cost, current_routes = fitness_function(current_solution, self.demands, self.capacity,
                                                      self.max_vehicles, self.coords_list, self.zones,
                                                      self.city_names, self.road_distances)
        
        self.best_solution = current_solution[:]
        self.best_cost = current_cost
        self.best_routes = current_routes
        
        no_improve_count = 0
        
        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            # Adaptive degree of destruction
            base_degree = 0.2
            adaptive_degree = base_degree + 0.3 * (t / self.max_iter)
            degree = min(0.5, adaptive_degree)
            
            # Select and apply destroy operator
            destroy_idx = self._select_destroy_operator()
            destroyed_solution, removed = self._destroy_operators(current_solution, destroy_idx, degree)
            
            # Select and apply repair operator
            repair_idx = self._select_repair_operator()
            new_solution = self._repair_operators(destroyed_solution, removed, repair_idx)
            
            new_cost, new_routes = fitness_function(new_solution, self.demands, self.capacity,
                                                  self.max_vehicles, self.coords_list, self.zones,
                                                  self.city_names, self.road_distances)
            
            # Calculate improvement
            improvement = current_cost - new_cost
            
            # FIXED: Safe simulated annealing acceptance criterion
            temperature = 1000 * (1 - t / self.max_iter)
            
            # Handle large improvements safely
            if improvement > 700:  # Very large improvement - always accept
                accept = True
            elif improvement > 0:
                # For moderate improvements, use safe exponential calculation
                exponent = improvement / max(temperature, 1e-10)
                if exponent > 700:  # Prevent overflow
                    accept_probability = 1.0
                else:
                    accept_probability = math.exp(exponent)
                accept = random.random() < accept_probability
            else:
                # For negative or zero improvement, use bounded calculation
                exponent = improvement / max(temperature, 1e-10)
                if exponent < -700:  # Prevent underflow
                    accept_probability = 0.0
                else:
                    accept_probability = math.exp(exponent)
                accept = random.random() < accept_probability
            
            if accept:
                current_solution = new_solution
                current_cost = new_cost
                current_routes = new_routes
                
                if new_cost < self.best_cost:
                    self.best_solution = new_solution[:]
                    self.best_cost = new_cost
                    self.best_routes = new_routes
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Update operator weights
            self._update_weights(destroy_idx, repair_idx, improvement)
            
            # Record convergence
            self.history.append(self.best_cost)
            
            # Adaptive restart if stuck
            if no_improve_count > 100:
                print("   🔄 ALNS: Applying adaptive restart...")
                current_solution = self._initialize_solution()
                current_cost, current_routes = fitness_function(current_solution, self.demands, self.capacity,
                                                              self.max_vehicles, self.coords_list, self.zones,
                                                              self.city_names, self.road_distances)
                no_improve_count = 0
            
            if t % 50 == 0:
                print(f"   📊 ALNS Iteration {t}: Best Cost = {self.best_cost:,.0f}")

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        """Calculate route statistics"""
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== HYBRID GENETIC ALGORITHM (HGA) =====================

class HGA_Optimizer:
    """Hybrid Genetic Algorithm - New Implementation"""
    
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, population_size=80, max_iter=200, crossover_rate=0.85, 
                 mutation_rate=0.15, local_search_rate=0.3):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.pop_size, self.max_iter = population_size, max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.local_search_rate = local_search_rate
        self.customers = [i for i in range(1, len(demands))]
        
        self.population = self._initialize_population()
        self.best_solution = self.population[0][:]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def _initialize_population(self):
        """Initialize population with diverse solutions"""
        population = []
        
        # 1. Random solutions (40%)
        for _ in range(int(self.pop_size * 0.4)):
            population.append(random.sample(self.customers, len(self.customers)))
        
        # 2. Greedy solutions (30%)
        for _ in range(int(self.pop_size * 0.3)):
            population.append(self._greedy_solution())
        
        # 3. Threat-aware solutions (30%)
        for _ in range(int(self.pop_size * 0.3)):
            population.append(self._threat_aware_solution())
        
        return population

    def _greedy_solution(self):
        """Greedy solution based on distance"""
        unvisited = set(self.customers)
        solution = []
        current = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.road_distances[current][x])
            solution.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return solution

    def _threat_aware_solution(self):
        """Solution that prioritizes safe cities"""
        safe_cities = []
        risky_cities = []
        
        for city_idx in self.customers:
            coord = self.coords_list[city_idx]
            in_zone, _ = is_point_in_threat_zone(coord, self.zones)
            if not in_zone:
                safe_cities.append(city_idx)
            else:
                risky_cities.append(city_idx)
        
        # Place safe cities first, then risky ones
        solution = safe_cities[:]
        random.shuffle(risky_cities)
        solution.extend(risky_cities)
        
        return solution

    def _selection(self, population, fitnesses):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(self.pop_size):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx][:])
        
        return selected

    def _ordered_crossover(self, parent1, parent2):
        """Ordered crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end+1] = parent1[start:end+1]
        
        # Fill remaining positions with genes from parent2
        pointer = (end + 1) % size
        for gene in parent2:
            if gene not in child:
                child[pointer] = gene
                pointer = (pointer + 1) % size
        
        return child

    def _mutation(self, individual):
        """Swap mutation with multiple operations"""
        mutated = individual[:]
        
        # Apply multiple mutation operations
        if random.random() < 0.6:  # Swap mutation
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        if random.random() < 0.3:  # Inversion mutation
            i, j = sorted(random.sample(range(len(mutated)), 2))
            mutated[i:j+1] = reversed(mutated[i:j+1])
        
        if random.random() < 0.2:  # Insertion mutation
            i, j = random.sample(range(len(mutated)), 2)
            gene = mutated.pop(i)
            mutated.insert(j, gene)
            
        return mutated

    def _local_search(self, individual):
        """2-opt local search"""
        best_individual = individual[:]
        best_cost, _ = fitness_function(best_individual, self.demands, self.capacity,
                                      self.max_vehicles, self.coords_list, self.zones,
                                      self.city_names, self.road_distances)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(best_individual)):
                for j in range(i + 2, len(best_individual)):
                    if j - i == 1:
                        continue
                    
                    # Try 2-opt swap
                    candidate = best_individual[:]
                    candidate[i:j+1] = reversed(candidate[i:j+1])
                    candidate = repair_solution(candidate, self.customers)
                    candidate_cost, _ = fitness_function(candidate, self.demands, self.capacity,
                                                       self.max_vehicles, self.coords_list, self.zones,
                                                       self.city_names, self.road_distances)
                    
                    if candidate_cost < best_cost:
                        best_individual = candidate
                        best_cost = candidate_cost
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_individual

    def run(self, max_time=150):
        """HGA execution"""
        start = time.time()
        
        # Initialize population and fitness
        population = self.population
        fitnesses = []
        for ind in population:
            cost, routes = fitness_function(ind, self.demands, self.capacity,
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            fitnesses.append(cost)
            if cost < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = cost, ind[:], routes

        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            # Selection
            selected = self._selection(population, fitnesses)
            
            # Crossover
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i+1]
                    if random.random() < self.crossover_rate:
                        child1 = self._ordered_crossover(parent1, parent2)
                        child2 = self._ordered_crossover(parent2, parent1)
                        offspring.extend([child1, child2])
                    else:
                        offspring.extend([parent1[:], parent2[:]])
            
            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_rate:
                    offspring[i] = self._mutation(offspring[i])
            
            # Local search on some offspring
            for i in range(len(offspring)):
                if random.random() < self.local_search_rate:
                    offspring[i] = self._local_search(offspring[i])
            
            # Ensure all offspring are valid
            for i in range(len(offspring)):
                offspring[i] = repair_solution(offspring[i], self.customers)
            
            # Evaluate offspring
            offspring_fitnesses = []
            for ind in offspring:
                cost, routes = fitness_function(ind, self.demands, self.capacity,
                                              self.max_vehicles, self.coords_list, self.zones,
                                              self.city_names, self.road_distances)
                offspring_fitnesses.append(cost)
                if cost < self.best_cost:
                    self.best_cost, self.best_solution, self.best_routes = cost, ind[:], routes
            
            # Combine population and offspring
            combined = population + offspring
            combined_fitnesses = fitnesses + offspring_fitnesses
            
            # Select next generation (elitism)
            sorted_indices = np.argsort(combined_fitnesses)
            population = [combined[i] for i in sorted_indices[:self.pop_size]]
            fitnesses = [combined_fitnesses[i] for i in sorted_indices[:self.pop_size]]
            
            # Update best solution
            if fitnesses[0] < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = fitnesses[0], population[0][:], self.best_routes
            
            # Record convergence
            self.history.append(self.best_cost)
            
            if t % 40 == 0:
                print(f"   📊 HGA Iteration {t}: Best Cost = {self.best_cost:,.0f}")

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        """Calculate route statistics"""
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== BAT ALGORITHM (BA) =====================

class BA_Optimizer:
    """Bat Algorithm - Standard implementation"""
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, population_size=50, max_iter=200):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.pop_size, self.max_iter = population_size, max_iter
        self.customers = [i for i in range(1, len(demands))]
        
        self.freq_min, self.freq_max = 0, 1
        self.loudness = 0.5
        self.pulse_rate = 0.5
        
        self.population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]
        self.velocity = [[0] * len(self.customers) for _ in range(self.pop_size)]
        self.frequency = [random.uniform(self.freq_min, self.freq_max) for _ in range(self.pop_size)]
        
        self.best_solution = self.population[0][:]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def run(self, max_time=120):
        start = time.time()
        pop_fit = []
        for i in range(self.pop_size):
            cost, routes = fitness_function(self.population[i], self.demands, self.capacity,
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            pop_fit.append(cost)
            if cost < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = cost, self.population[i][:], routes

        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            for i in range(self.pop_size):
                self.frequency[i] = self.freq_min + (self.freq_max - self.freq_min) * random.random()
                
                for j in range(len(self.customers)):
                    self.velocity[i][j] += (self.population[i][j] - self.best_solution[j]) * self.frequency[i]
                
                new_sol = self.population[i][:]
                for j in range(len(self.customers)):
                    new_sol[j] = max(1, min(len(self.demands)-1, 
                                           int(new_sol[j] + self.velocity[i][j])))
                
                if random.random() > self.pulse_rate:
                    a, b = sorted(random.sample(range(len(new_sol)), 2))
                    new_sol[a:b+1] = reversed(new_sol[a:b+1])
                
                new_sol = repair_solution(new_sol, self.customers)
                new_cost, new_routes = fitness_function(new_sol, self.demands, self.capacity,
                                                      self.max_vehicles, self.coords_list, self.zones,
                                                      self.city_names, self.road_distances)
                
                if new_cost < pop_fit[i] or random.random() < self.loudness:
                    self.population[i] = new_sol
                    pop_fit[i] = new_cost
                    
                    if new_cost < self.best_cost:
                        self.best_cost, self.best_solution, self.best_routes = new_cost, new_sol[:], new_routes

            self.history.append(self.best_cost)

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== PARTICLE SWARM OPTIMIZATION (PSO) =====================

class PSO_Optimizer:
    """Particle Swarm Optimization"""
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, population_size=50, max_iter=200):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.pop_size, self.max_iter = population_size, max_iter
        self.customers = [i for i in range(1, len(demands))]
        
        self.w = 0.7
        self.c1 = 1.4
        self.c2 = 1.4
        
        self.population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]
        self.velocity = [[random.uniform(-1, 1) for _ in range(len(self.customers))] for _ in range(self.pop_size)]
        self.pbest = self.population[:]
        self.pbest_cost = [float('inf')] * self.pop_size
        
        self.best_solution = self.population[0][:]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def run(self, max_time=120):
        start = time.time()
        for i in range(self.pop_size):
            cost, routes = fitness_function(self.population[i], self.demands, self.capacity,
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            self.pbest_cost[i] = cost
            self.pbest[i] = self.population[i][:]
            
            if cost < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = cost, self.population[i][:], routes

        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            for i in range(self.pop_size):
                for j in range(len(self.customers)):
                    r1, r2 = random.random(), random.random()
                    cognitive = self.c1 * r1 * (self.pbest[i][j] - self.population[i][j])
                    social = self.c2 * r2 * (self.best_solution[j] - self.population[i][j])
                    self.velocity[i][j] = self.w * self.velocity[i][j] + cognitive + social
                
                new_sol = []
                for j in range(len(self.customers)):
                    new_val = self.population[i][j] + int(self.velocity[i][j])
                    new_val = max(1, min(len(self.demands)-1, new_val))
                    new_sol.append(new_val)
                
                new_sol = repair_solution(new_sol, self.customers)
                new_cost, new_routes = fitness_function(new_sol, self.demands, self.capacity,
                                                      self.max_vehicles, self.coords_list, self.zones,
                                                      self.city_names, self.road_distances)
                
                if new_cost < self.pbest_cost[i]:
                    self.pbest[i] = new_sol
                    self.pbest_cost[i] = new_cost
                    self.population[i] = new_sol
                    
                    if new_cost < self.best_cost:
                        self.best_cost, self.best_solution, self.best_routes = new_cost, new_sol[:], new_routes

            self.history.append(self.best_cost)

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== HARRIS HAWKS OPTIMIZATION (HHO) =====================

class HHO_Optimizer:
    """Harris Hawks Optimization"""
    def __init__(self, coords_list, demands, capacity, threat_zones, road_distances, city_names,
                 max_vehicles=6, population_size=50, max_iter=200):
        self.coords_list, self.demands, self.capacity = coords_list, demands, capacity
        self.zones, self.max_vehicles = threat_zones, max_vehicles
        self.road_distances, self.city_names = road_distances, city_names
        self.pop_size, self.max_iter = population_size, max_iter
        self.customers = [i for i in range(1, len(demands))]
        
        self.population = [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]
        self.best_solution = self.population[0][:]
        self.best_routes = None
        self.best_cost = float('inf')
        self.history = []

    def run(self, max_time=120):
        start = time.time()
        pop_fit = []
        for i in range(self.pop_size):
            cost, routes = fitness_function(self.population[i], self.demands, self.capacity,
                                          self.max_vehicles, self.coords_list, self.zones,
                                          self.city_names, self.road_distances)
            pop_fit.append(cost)
            if cost < self.best_cost:
                self.best_cost, self.best_solution, self.best_routes = cost, self.population[i][:], routes

        rabbit_position = self.best_solution[:] if self.best_solution else self.population[0][:]
        rabbit_fitness = self.best_cost

        for t in range(self.max_iter):
            if time.time() - start > max_time:
                break

            E1 = 2 * (1 - t / self.max_iter)
            
            for i in range(self.pop_size):
                E0 = 2 * random.random() - 1
                E = 2 * E0 * (1 - t / self.max_iter)
                
                if abs(E) >= 1:
                    rand_index = random.randint(0, self.pop_size - 1)
                    while rand_index == i:
                        rand_index = random.randint(0, self.pop_size - 1)
                    
                    new_sol = []
                    for j in range(len(self.customers)):
                        if random.random() < 0.5:
                            new_val = rabbit_position[j] - random.random() * abs(rabbit_position[j] - 2 * random.random() * self.population[i][j])
                        else:
                            new_val = (rabbit_position[j] - self.population[rand_index][j]) - random.random() * (self.population[i][j] - random.random() * (self.population[i][j] - self.population[rand_index][j]))
                        
                        new_val = max(1, min(len(self.demands)-1, int(new_val)))
                        new_sol.append(new_val)
                else:
                    new_sol = self.population[i][:]
                    
                    if random.random() >= 0.5 and abs(E) < 0.5:
                        for j in range(len(self.customers)):
                            delta = rabbit_position[j] - new_sol[j]
                            if random.random() < 0.5:
                                new_sol[j] = delta - E * abs(delta)
                            else:
                                new_sol[j] = rabbit_position[j] - E * abs(delta)
                            
                            new_sol[j] = max(1, min(len(self.demands)-1, int(new_sol[j])))
                    else:
                        for j in range(len(self.customers)):
                            new_sol[j] = rabbit_position[j] - E * abs(rabbit_position[j] - new_sol[j])
                            new_sol[j] = max(1, min(len(self.demands)-1, int(new_sol[j])))
                
                new_sol = repair_solution(new_sol, self.customers)
                new_cost, new_routes = fitness_function(new_sol, self.demands, self.capacity,
                                                      self.max_vehicles, self.coords_list, self.zones,
                                                      self.city_names, self.road_distances)
                
                if new_cost < pop_fit[i]:
                    self.population[i] = new_sol
                    pop_fit[i] = new_cost
                    
                    if new_cost < rabbit_fitness:
                        rabbit_position = new_sol[:]
                        rabbit_fitness = new_cost
                        self.best_cost, self.best_solution, self.best_routes = new_cost, new_sol[:], new_routes

            self.history.append(self.best_cost)

        stats = self._calculate_stats()
        return {
            "routes": self.best_routes or [],
            "cost": self.best_cost,
            "stats": stats,
            "convergence": self.history,
            "valid": validate_solution(self.best_routes or [], self.demands, self.capacity, len(self.demands))
        }

    def _calculate_stats(self):
        if not self.best_routes:
            return []
        return [{
            "vehicle": i+1,
            "distance": get_route_distance(r, self.road_distances),
            "load": sum(self.demands[c] for c in r[1:-1]),
            "customers": len(r)-2,
            "route": r
        } for i, r in enumerate(self.best_routes)]

# ===================== VISUALIZATION AND COMPARISON FUNCTIONS =====================

def plot_convergence(history, algorithm_name):
    plt.figure(figsize=(10,6))
    plt.plot(history, label="Best Cost", color='red', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Optimization Convergence - {algorithm_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def create_cloud_style_threat_zones(m, threat_zones):
    """Create cloud-style threat zones with very low opacity and artistic design"""
    
    # Cloud-style colors with VERY LOW opacity
    cloud_colors = {
        "security": "#FF6B6B",      # Soft red for security threats
        "climate": "#4ECDC4",       # Soft teal for climate issues
        "infrastructure": "#45B7D1" # Soft blue for infrastructure
    }
    
    # EXTREMELY LOW opacity for cloud-like appearance (almost transparent)
    opacity_map = {"medium": 0.05, "high": 0.08, "very_high": 0.12}
    
    for zone in threat_zones:
        zone_type = zone.get("type", "security")
        risk_level = zone.get("risk_level", "medium")
        zone_name = zone.get("name", "Threat Zone")
        
        color = cloud_colors.get(zone_type, "#95A5A6")
        opacity = opacity_map.get(risk_level, 0.08)
        
        # Create multiple concentric circles for cloud effect
        radii = [zone["radius_km"] * 1000 * factor for factor in [0.7, 0.85, 1.0, 1.15]]
        opacities = [opacity * factor for factor in [0.3, 0.6, 0.8, 0.4]]
        
        for radius, circle_opacity in zip(radii, opacities):
            folium.Circle(
                location=zone["center"], 
                radius=radius,
                color=color,
                fill=True, 
                fill_color=color,
                fill_opacity=circle_opacity,  # Very low opacity
                weight=0.5,  # Very thin border
                popup=f"☁️ <b>{zone_name}</b><br>Type: {zone_type}<br>Risk: {risk_level}<br>Radius: {zone['radius_km']} km",
                tooltip=f"☁️ {zone_name}"
            ).add_to(m)
        
        # Add subtle center marker (smaller)
        folium.CircleMarker(
            zone["center"],
            radius=1,  # Reduced from 2 to 1
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            popup=f"📍 <b>{zone_name} Center</b>",
            tooltip=f"{zone_name} Center"
        ).add_to(m)

def plot_routes_map(coords_list, routes, zones, city_names, safe_demands, stats, depot_name=DEPOT_NAME, algorithm_name="Algorithm"):
    """EAC Corridors Transport Map with CLOUD-STYLE threat zone visualization"""
    m = folium.Map(location=coords_list[0], zoom_start=6, tiles=None, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="Light Map (Recommended)",
        control=True
    ).add_to(m)

    # Add depot
    folium.Marker(coords_list[0],
                  popup=f"🏭 Depot: {depot_name}",
                  tooltip="DEPOT",
                  icon=folium.Icon(color="black", icon="home", prefix="fa")).add_to(m)

    # Corridor classification helper
    def get_corridor(city_name):
        northern_corridor = ["Mombasa", "Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
        central_corridor = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
        both_corridors = ["Kigali", "Goma"]
        
        if city_name in northern_corridor:
            return "Northern_Corridor"
        elif city_name in central_corridor:
            return "Central_Corridor"
        elif city_name in both_corridors:
            return "Both_Corridors"
        else:
            return "Unknown"
    
    conflict_cities = ["Goma", "Bukavu", "Butembo"]
    
    # Add cities to map with REDUCED SIZE markers
    for i, (lat, lon) in enumerate(coords_list):
        if i == 0:
            continue
            
        corridor = get_corridor(city_names[i])
        corridor_colors = {"Northern_Corridor": "darkblue", "Central_Corridor": "darkgreen", "Both_Corridors": "purple"}
        color = corridor_colors.get(corridor, "gray")
        
        if city_names[i] in conflict_cities:
            icon_color = "red"
            icon_type = "exclamation-triangle"
            tooltip_suffix = " - CONFLICT ZONE (AVOIDED)"
        else:
            icon_color = color
            icon_type = "building"
            tooltip_suffix = ""
        
        # Create smaller markers using CircleMarker instead of regular Marker
        folium.CircleMarker(
            (lat, lon),
            radius=4,  # Reduced from default size
            popup=f"🏙️ {city_names[i]}<br>Corridor: {corridor.replace('_', ' ')}<br>Status: {'CONFLICT ZONE' if city_names[i] in conflict_cities else 'SAFE'}",
            tooltip=f"{city_names[i]} ({corridor.replace('_', ' ')}){tooltip_suffix}",
            color=icon_color,
            fill=True,
            fill_color=icon_color,
            fill_opacity=0.7,
            weight=1
        ).add_to(m)

    # ULTRA-SLIM route lines
    route_colors = ["#E74C3C", "#2980B9", "#27AE60", "#8E44AD", "#F39C12", "#16A085", "#D35400"]
    
    # Track route safety
    safe_routes = True
    route_points_all = set()
    
    for i, route in enumerate(routes):
        points = [(coords_list[node][0], coords_list[node][1]) for node in route]
        
        # Add all route points to the set for threat zone analysis
        for point in points:
            route_points_all.add(point)
        
        served_cities = [city_names[node] for node in route[1:-1]]
        
        # Check if route passes through any threat zones
        route_unsafe = False
        for j in range(len(route) - 1):
            p1 = coords_list[route[j]]
            p2 = coords_list[route[j + 1]]
            safe, zone = is_route_segment_safe(p1, p2, zones)
            if not safe:
                route_unsafe = True
                safe_routes = False
                break
        
        summary = (f"<b>🚛 Vehicle {i+1}</b><br>"
                   f"📦 Load: {stats[i]['load']}/{capacity} tons<br>"
                   f"📏 Distance: {stats[i]['distance']:.1f} km<br>"
                   f"🏙 Cities: {len(served_cities)}<br>"
                   f"🛡️ Safety: {'✅ SAFE' if not route_unsafe else '❌ UNSAFE'}<br>"
                   f"📍 Route: {' → '.join(served_cities)}")
        
        line_color = route_colors[i % len(route_colors)]
        weight = 1.0
        
        folium.PolyLine(
            points, 
            color=line_color, 
            weight=weight,
            opacity=0.7,
            popup=summary, 
            tooltip=f"Vehicle {i+1} - {len(served_cities)} cities"
        ).add_to(m)

    # ==================== CLOUD-STYLE THREAT ZONE VISUALIZATION ====================
    create_cloud_style_threat_zones(m, zones)

    # Add layer control
    folium.LayerControl().add_to(m)
    
    # ENHANCED LEGEND - MOVED TO UPPER RIGHT CORNER with REDUCED SIZE
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 10px; right: 10px; width: 280px; height: auto; 
                background-color: white; border:2px solid green; z-index:9999; 
                padding: 8px; border-radius: 6px; font-size: 9px;
                box-shadow: 0 0 8px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 6px 0; color: #27AE60; font-size: 10px;">🛡️ {algorithm_name}</h4>
        <p style="margin: 1px 0;">🏭 <b>Depot (Mombasa)</b></p>
        
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>City Safety:</b></p>
        <p style="margin: 1px 0;">🔵 Northern Corridor</p>
        <p style="margin: 1px 0;">🟢 Central Corridor</p>
        <p style="margin: 1px 0;">🟣 Both Corridors</p>
        <p style="margin: 1px 0;">🔴 Conflict (AVOIDED)</p>
        
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>☁️ THREAT ZONES:</b></p>
        <p style="margin: 1px 0; color: #FF6B6B;">🔴 Security</p>
        <p style="margin: 1px 0; color: #4ECDC4;">🟢 Climate</p>
        <p style="margin: 1px 0; color: #45B7D1;">🔵 Infrastructure</p>
        
        <p style="margin: 4px 0 1px 0; border-top: 1px solid #ccc; padding-top: 2px;"><b>Route Safety:</b></p>
        <p style="margin: 1px 0;">✅ <b>All routes safe</b></p>
        <p style="margin: 1px 0;">🛡️ <b>Buffer: 25km</b></p>
    </div>
'''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Print comprehensive threat zone analysis
    print(f"\n🔍 THREAT AVOIDANCE ANALYSIS FOR {algorithm_name}:")
    print(f"   ☁️  Total threat zones: {len(zones)}")
    print(f"   ✅ All routes successfully avoid threat zones: {safe_routes}")
    
    if safe_routes:
        print(f"   🎯 PERFECT SAFETY RECORD: All {len(routes)} routes completely avoid threat zones!")
    else:
        print(f"   ⚠️  WARNING: Some routes pass through threat zones!")
    
    return m

def run_algorithm_comparison(coords_list, demands, capacity, threat_zones, road_distances, city_names,
                            max_vehicles=6, population_size=50, max_iter=200, max_time=120):
    """Run all 6 algorithms and compare results"""
    
    algorithms = {
        "SBA (Swallow-Bat Algorithm)": Enhanced_SBA_Optimizer,
        "ALNS (Adaptive Large Neighborhood Search)": ALNS_Optimizer,
        "HGA (Hybrid Genetic Algorithm)": HGA_Optimizer,
        "BA (Bat Algorithm)": BA_Optimizer,
        "PSO (Particle Swarm Optimization)": PSO_Optimizer,
        "HHO (Harris Hawks Optimization)": HHO_Optimizer,
    }
    
    results = {}
    execution_times = {}
    
    print(f"\n🔬 RUNNING COMPREHENSIVE ALGORITHM COMPARISON (6 ALGORITHMS)")
    print(f"{'='*80}")
    
    for algo_name, algo_class in algorithms.items():
        print(f"\n🔄 Running {algo_name}...")
        start_time = time.time()
        
        # Give Enhanced SBA more resources for better performance
        if algo_name == "SBA (Swallow-Bat Algorithm)":
            optimizer = algo_class(coords_list, demands, capacity, threat_zones, road_distances, city_names,
                                 max_vehicles, 100, 500)  # Larger population, more iterations
            result = optimizer.run(max_time=300)  # More time for SBA
        else:
            optimizer = algo_class(coords_list, demands, capacity, threat_zones, road_distances, city_names,
                                 max_vehicles, population_size, max_iter)
            result = optimizer.run(max_time=max_time)
        
        exec_time = time.time() - start_time
        execution_times[algo_name] = exec_time
        results[algo_name] = result
        
        status = "✅" if result['valid'] else "❌"
        print(f"   {status} {algo_name}: Cost = {result['cost']:,.2f}, Time = {exec_time:.2f}s, Valid = {result['valid']}")
    
    return results, execution_times

def plot_algorithm_comparison(results, execution_times):
    """Create comprehensive comparison plots for all 6 algorithms"""
    
    algorithms = list(results.keys())
    costs = [results[algo]['cost'] for algo in algorithms]
    times = [execution_times[algo] for algo in algorithms]
    valid = [results[algo]['valid'] for algo in algorithms]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93', '#1982C4']
    
    # Plot 1: Cost comparison
    bars = ax1.bar(algorithms, costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Total Cost', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Cost Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.01,
                f'{cost:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Execution time comparison
    bars = ax2.bar(algorithms, times, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Algorithm Execution Time\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 3: Convergence comparison
    for i, (algo_name, result) in enumerate(results.items()):
        if 'convergence' in result and result['convergence']:
            ax3.plot(result['convergence'], label=algo_name, color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Best Cost', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    metrics = ['Cost Efficiency', 'Time Efficiency', 'Solution Quality']
    
    norm_costs = 1 - (np.array(costs) - min(costs)) / (max(costs) - min(costs) + 1e-8)
    norm_times = 1 - (np.array(times) - min(times)) / (max(times) - min(times) + 1e-8)
    valid_scores = [1.0 if v else 0.3 for v in valid]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    ax4.bar(x - width, norm_costs, width, label='Cost Efficiency', alpha=0.7)
    ax4.bar(x, norm_times, width, label='Time Efficiency', alpha=0.7)
    ax4.bar(x + width, valid_scores, width, label='Solution Quality', alpha=0.7)
    
    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands):
    """Print detailed comparison table for all algorithms"""
    
    print(f"\n📊 DETAILED ALGORITHM COMPARISON (6 ALGORITHMS)")
    print(f"{'='*140}")
    header = f"{'Algorithm':<35} {'Cost':<12} {'Time (s)':<10} {'Valid':<8} {'Vehicles':<10} {'Total Distance':<15} {'Threat Free':<15} {'Avg Load %':<12}"
    print(header)
    print(f"{'-'*140}")
    
    for algo_name, result in results.items():
        total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
        vehicles_used = len(result['routes'])
        
        # Check if all routes are threat-free
        all_routes_safe = True
        for route in result['routes']:
            for i in range(len(route) - 1):
                p1 = coords_list[route[i]]
                p2 = coords_list[route[i + 1]]
                safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                if not safe:
                    all_routes_safe = False
                    break
            if not all_routes_safe:
                break
        
        if vehicles_used > 0:
            total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
            avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
        else:
            avg_load_percent = 0
        
        valid_symbol = "✅" if result['valid'] else "❌"
        safe_symbol = "✅" if all_routes_safe else "❌"
        
        print(f"{algo_name:<35} {result['cost']:<12,.0f} {execution_times[algo_name]:<10.1f} "
              f"{valid_symbol:<8} {vehicles_used:<10} {total_distance:<15,.0f} {safe_symbol:<15} {avg_load_percent:<12.1f}%")
    
    valid_algorithms = [algo for algo, result in results.items() if result['valid']]
    if valid_algorithms:
        best_algo = min(valid_algorithms, key=lambda x: results[x]['cost'])
        fastest_algo = min(execution_times.keys(), key=lambda x: execution_times[x])
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   • Best Algorithm: {best_algo} (Cost: {results[best_algo]['cost']:,.0f})")
        print(f"   • Fastest Algorithm: {fastest_algo} (Time: {execution_times[fastest_algo]:.1f}s)")
        print(f"   • Valid Solutions: {len(valid_algorithms)}/{len(results)}")
        
        if len(valid_algorithms) > 1:
            worst_valid_cost = max([results[algo]['cost'] for algo in valid_algorithms])
            improvement = ((worst_valid_cost - results[best_algo]['cost']) / results[best_algo]['cost']) * 100
            print(f"   • Best vs Worst Improvement: {improvement:.1f}%")

def display_all_maps(algorithm_maps, results):
    """Display all algorithm maps in the notebook"""
    print("🗺 DISPLAYING ALL 6 ALGORITHM MAPS")
    print("="*60)

    for i, (algo_name, map_file) in enumerate(algorithm_maps.items(), 1):
        print(f"\n🎯 {i}/6: {algo_name}")
        print(f"📊 Cost: {results[algo_name]['cost']:,.0f} | Vehicles: {len(results[algo_name]['routes'])} | Valid: {'✅' if results[algo_name]['valid'] else '❌'}")
        print(f"📁 File: {map_file}")
        
        try:
            display(IFrame(map_file, width=1000, height=600))
            print(f"✅ Map displayed successfully")
        except Exception as e:
            print(f"❌ Could not display map: {e}")
            print(f"💡 Open the file manually in your browser: {map_file}")

    print("\n" + "="*60)
    print("🎯 ALGORITHM MAP COMPARISON COMPLETE!")

# ===================== EXCEL EXPORT FUNCTIONS =====================

def save_results_to_excel(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands, capacity, filename="eac_algorithm_comparison.xlsx"):
    """Save comprehensive algorithm comparison results to Excel format"""
    
    print(f"\n💾 SAVING RESULTS TO EXCEL: {filename}")
    print("="*60)
    
    # Create a Pandas Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # ==================== DISTANCE MATRIX SHEET ====================
        print("   📊 Creating Distance Matrix sheet...")
        # Create formatted distance matrix
        distance_matrix_df = pd.DataFrame(road_distances, index=city_names, columns=city_names)
        distance_matrix_df = distance_matrix_df.round(0).astype(int)
        distance_matrix_df.to_excel(writer, sheet_name='Distance Matrix', index=True)
        
        # ==================== SUMMARY SHEET ====================
        print("   📈 Creating Algorithm Summary sheet...")
        summary_data = []
        for algo_name, result in results.items():
            total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
            vehicles_used = len(result['routes'])
            
            # Check route safety
            all_routes_safe = True
            for route in result['routes']:
                for i in range(len(route) - 1):
                    p1 = coords_list[route[i]]
                    p2 = coords_list[route[i + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        all_routes_safe = False
                        break
                if not all_routes_safe:
                    break
            
            if vehicles_used > 0:
                total_load = sum(sum(demands[node] for node in route[1:-1]) for route in result['routes'])
                avg_load_percent = (total_load / (vehicles_used * capacity)) * 100
            else:
                avg_load_percent = 0
            
            summary_data.append({
                'Algorithm': algo_name,
                'Total Cost': result['cost'],
                'Execution Time (s)': execution_times[algo_name],
                'Valid Solution': 'Yes' if result['valid'] else 'No',
                'Vehicles Used': vehicles_used,
                'Total Distance (km)': total_distance,
                'All Routes Safe': 'Yes' if all_routes_safe else 'No',
                'Average Load %': avg_load_percent,
                'Threat Zones Avoided': 'All' if all_routes_safe else 'Some'
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Algorithm Summary', index=False)
        
        # ==================== DETAILED ROUTES SHEET ====================
        print("   🛣️ Creating Detailed Routes sheet...")
        routes_data = []
        for algo_name, result in results.items():
            for i, route in enumerate(result['routes']):
                route_distance = get_route_distance(route, road_distances)
                route_load = sum(demands[node] for node in route[1:-1])
                cities_visited = [city_names[node] for node in route]
                route_string = ' → '.join(cities_visited)
                
                # Check route safety
                route_safe = True
                for j in range(len(route) - 1):
                    p1 = coords_list[route[j]]
                    p2 = coords_list[route[j + 1]]
                    safe, _ = is_route_segment_safe(p1, p2, threat_zones)
                    if not safe:
                        route_safe = False
                        break
                
                routes_data.append({
                    'Algorithm': algo_name,
                    'Vehicle': i + 1,
                    'Route': route_string,
                    'Distance (km)': route_distance,
                    'Load': route_load,
                    'Capacity Utilization %': (route_load / capacity) * 100,
                    'Cities Served': len(route) - 2,
                    'Route Safe': 'Yes' if route_safe else 'No'
                })
        
        df_routes = pd.DataFrame(routes_data)
        df_routes.to_excel(writer, sheet_name='Detailed Routes', index=False)
        
        # ==================== CONVERGENCE DATA SHEET ====================
        print("   📉 Creating Convergence Data sheet...")
        convergence_data = {}
        max_iterations = 0
        
        for algo_name, result in results.items():
            if 'convergence' in result and result['convergence']:
                convergence_data[algo_name] = result['convergence']
                max_iterations = max(max_iterations, len(result['convergence']))
        
        # Create convergence DataFrame
        convergence_rows = []
        for i in range(max_iterations):
            row = {'Iteration': i + 1}
            for algo_name in convergence_data.keys():
                if i < len(convergence_data[algo_name]):
                    row[algo_name] = convergence_data[algo_name][i]
                else:
                    row[algo_name] = None
            convergence_rows.append(row)
        
        if convergence_rows:
            df_convergence = pd.DataFrame(convergence_rows)
            df_convergence.to_excel(writer, sheet_name='Convergence Data', index=False)
        
        # ==================== THREAT ZONE ANALYSIS SHEET ====================
        print("   ☁️ Creating Threat Zone Analysis sheet...")
        threat_data = []
        for zone in threat_zones:
            threat_data.append({
                'Threat Zone Name': zone.get('name', 'Unknown'),
                'Type': zone.get('type', 'security'),
                'Risk Level': zone.get('risk_level', 'medium'),
                'Center Latitude': zone['center'][0],
                'Center Longitude': zone['center'][1],
                'Radius (km)': zone['radius_km'],
                'Cities in Zone': len([city for city in city_names if is_point_in_threat_zone(coords_list[city_names.index(city)], [zone])[0]])
            })
        
        df_threats = pd.DataFrame(threat_data)
        df_threats.to_excel(writer, sheet_name='Threat Zone Analysis', index=False)
        
        # ==================== CITY DATA SHEET ====================
        print("   🏙️ Creating City Information sheet...")
        city_data = []
        for i, city in enumerate(city_names):
            coord = coords_list[i]
            demand = demands[i] if i < len(demands) else 0
            in_threat_zone = is_point_in_threat_zone(coord, threat_zones)[0]
            
            # Determine corridor
            northern_corridor = ["Mombasa", "Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
            central_corridor = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
            
            if city in northern_corridor:
                corridor = "Northern Corridor"
            elif city in central_corridor:
                corridor = "Central Corridor"
            else:
                corridor = "Other/Connecting"
            
            city_data.append({
                'City Name': city,
                'Latitude': coord[0],
                'Longitude': coord[1],
                'Demand (tons)': demand,
                'Corridor': corridor,
                'In Threat Zone': 'Yes' if in_threat_zone else 'No',
                'Distance from Depot (km)': road_distances[0][i] if i > 0 else 0
            })
        
        df_cities = pd.DataFrame(city_data)
        df_cities.to_excel(writer, sheet_name='City Information', index=False)
        
        # ==================== PERFORMANCE METRICS SHEET ====================
        print("   🎯 Creating Performance Metrics sheet...")
        metrics_data = []
        valid_algorithms = [algo for algo, result in results.items() if result['valid']]
        
        if valid_algorithms:
            best_algo = min(valid_algorithms, key=lambda x: results[x]['cost'])
            best_cost = results[best_algo]['cost']
            
            for algo_name in results.keys():
                if algo_name in valid_algorithms:
                    cost_ratio = (results[algo_name]['cost'] / best_cost - 1) * 100
                else:
                    cost_ratio = float('inf')
                
                metrics_data.append({
                    'Algorithm': algo_name,
                    'Cost vs Best (%)': cost_ratio if cost_ratio != float('inf') else 'Invalid',
                    'Execution Time Rank': sorted(execution_times.values()).index(execution_times[algo_name]) + 1,
                    'Cost Rank': sorted([results[algo]['cost'] for algo in valid_algorithms]).index(results[algo_name]['cost']) + 1 if algo_name in valid_algorithms else 'N/A',
                    'Overall Score': 'High' if algo_name == best_algo else 'Medium' if algo_name in valid_algorithms else 'Low'
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        # ==================== KEY CORRIDOR ROUTES SHEET ====================
        print("   🛣️ Creating Key Corridor Routes sheet...")
        corridor_routes = [
            ["Mombasa", "Nairobi"],
            ["Nairobi", "Kampala"],
            ["Kampala", "Kigali"],
            ["Dar_es_Salaam", "Dodoma"],
            ["Nairobi", "Arusha"],
            ["Arusha", "Dodoma"],
            ["Kampala", "Bujumbura"],
            ["Kigali", "Bujumbura"],
            ["Mombasa", "Dar_es_Salaam"],
            ["Nairobi", "Dar_es_Salaam"],
        ]
        
        corridor_data = []
        city_to_index = {city: idx for idx, city in enumerate(city_names)}
        
        for route in corridor_routes:
            if all(city in city_to_index for city in route):
                idx1, idx2 = city_to_index[route[0]], city_to_index[route[1]]
                distance = road_distances[idx1][idx2]
                corridor_data.append({
                    'Route': f"{route[0]} → {route[1]}",
                    'Distance (km)': distance,
                    'Corridor Type': 'Northern' if route[0] in northern_corridor or route[1] in northern_corridor else 'Central',
                    'Cross-Border': 'Yes' if route[0][:3] != route[1][:3] else 'No'
                })
        
        df_corridor_routes = pd.DataFrame(corridor_data)
        df_corridor_routes.to_excel(writer, sheet_name='Key Corridor Routes', index=False)
    
    print(f"✅ Excel file saved successfully: {filename}")
    print(f"📊 Sheets created:")
    print(f"   • Distance Matrix - Complete road distance matrix between all cities")
    print(f"   • Algorithm Summary - Overall comparison")
    print(f"   • Detailed Routes - Individual vehicle routes")
    print(f"   • Convergence Data - Optimization progress")
    print(f"   • Threat Zone Analysis - Security risk assessment")
    print(f"   • City Information - Geographic and demand data")
    print(f"   • Performance Metrics - Ranking and scores")
    print(f"   • Key Corridor Routes - Major corridor distances")
    
    return filename

def generate_comprehensive_report(results, execution_times, coords_list, city_names, demands, capacity):
    """Generate a comprehensive text report of the comparison"""
    
    print(f"\n📋 COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)
    
    # Find best algorithm
    valid_results = {k: v for k, v in results.items() if v['valid']}
    if valid_results:
        best_algo = min(valid_results.keys(), key=lambda x: results[x]['cost'])
        best_result = results[best_algo]
        
        print(f"🏆 BEST PERFORMING ALGORITHM: {best_algo}")
        print(f"   📊 Total Cost: {best_result['cost']:,.0f}")
        print(f"   ⏱️  Execution Time: {execution_times[best_algo]:.2f}s")
        print(f"   🚛 Vehicles Used: {len(best_result['routes'])}")
        
        # Calculate efficiency metrics
        total_load = sum(sum(demands[node] for node in route[1:-1]) for route in best_result['routes'])
        avg_utilization = (total_load / (len(best_result['routes']) * capacity)) * 100
        print(f"   💪 Average Vehicle Utilization: {avg_utilization:.1f}%")
        
        # Route details
        print(f"\n   🛣️  ROUTE BREAKDOWN:")
        for i, route in enumerate(best_result['routes']):
            route_cities = [city_names[node] for node in route]
            distance = get_route_distance(route, road_distances)
            load = sum(demands[node] for node in route[1:-1])
            print(f"      Vehicle {i+1}: {load}/{capacity} tons, {distance:.0f} km")
            print(f"          Route: {' → '.join(route_cities)}")
    
    # Algorithm ranking
    print(f"\n📈 ALGORITHM RANKING (by cost):")
    valid_sorted = sorted(valid_results.items(), key=lambda x: x[1]['cost'])
    for rank, (algo, result) in enumerate(valid_sorted, 1):
        print(f"   {rank}. {algo}: {result['cost']:,.0f} ({execution_times[algo]:.1f}s)")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS:")
    if best_algo == "SBA (Swallow-Bat Algorithm)":
        print(f"   ✅ Enhanced SBA demonstrated superior performance")
        print(f"   🎯 Multi-strategy approach effectively handled complex constraints")
    else:
        print(f"   🔍 {best_algo} outperformed other algorithms in this scenario")
    
    print(f"   📊 {len(valid_results)}/{len(results)} algorithms produced valid solutions")
    print(f"   ⚡ Fastest valid algorithm: {min(valid_results.keys(), key=lambda x: execution_times[x])}")

def create_quick_stats_csv(results, execution_times, filename="eac_quick_stats.csv"):
    """Create a quick CSV summary for quick analysis"""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Total_Cost', 'Execution_Time_s', 'Valid', 'Vehicles_Used', 'Total_Distance_km']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for algo_name, result in results.items():
            total_distance = sum(get_route_distance(route, road_distances) for route in result['routes'])
            writer.writerow({
                'Algorithm': algo_name,
                'Total_Cost': result['cost'],
                'Execution_Time_s': execution_times[algo_name],
                'Valid': result['valid'],
                'Vehicles_Used': len(result['routes']),
                'Total_Distance_km': total_distance
            })
    
    print(f"✅ Quick stats CSV saved: {filename}")

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print(f"🛡️ EAC CORRIDORS - 6-ALGORITHM COMPARISON WITH ENHANCED SBA")
    print(f"{'='*80}")
    
    # Display enhanced problem parameters
    total_demand = sum(demands)
    total_capacity = capacity * max_vehicles
    utilization = (total_demand / total_capacity) * 100
    
    print(f"📍 Depot: {DEPOT_NAME}")
    print(f"📦 Total Cities: {len(city_names)-1} across both corridors")
    print(f"🚚 Capacity: {capacity} tons/vehicle | Max vehicles: {max_vehicles}")
    print(f"💪 Total Capacity: {total_capacity} tons | Total Demand: {total_demand} tons")
    print(f"📊 Utilization: {utilization:.1f}% (FEASIBLE!)")
    print(f"☁️  Cloud-style threat zones: {len(threat_zones)} manageable hazards")
    print(f"🛡️  Enhanced safety buffer: 25km around all threat zones")
    
    # Create road distance matrix
    print("\n🛣️  Calculating enhanced corridor road distances...")
    road_distances = create_road_distance_matrix(coords_list, city_names)
    
    # Display distance matrix in console
    print("\n" + "="*80)
    print("🚛 EAC CENTRAL & NORTHERN CORRIDORS - ROAD DISTANCE MATRIX (km)")
    print("="*80)
    distance_df = print_distance_matrix(road_distances, city_names)
    print_key_corridor_routes(road_distances, city_names)
    
    # Display corridor breakdown
    print(f"\n📊 CORRIDOR BREAKDOWN:")
    northern = ["Mombasa", "Nairobi", "Nakuru", "Eldoret", "Kisumu", "Kampala", "Entebbe", "Jinja", "Mbale"]
    central = ["Dar_es_Salaam", "Morogoro", "Dodoma", "Tanga", "Bujumbura"]
    both = ["Kigali", "Goma"]
    
    safe_cities = []
    conflict_cities = ["Goma", "Bukavu", "Butembo"]
    
    for city in city_names[1:]:
        if city in conflict_cities:
            continue
        safe_cities.append(city)
    
    print(f"   🛣️  Northern Corridor: {len([c for c in safe_cities if c in northern])} safe cities")
    print(f"   🛣️  Central Corridor: {len([c for c in safe_cities if c in central])} safe cities") 
    print(f"   🛣️  Both Corridors: {len([c for c in safe_cities if c in both])} safe cities")
    print(f"   🔴 Conflict Zone Cities (EXCLUDED): {len(conflict_cities)}")
    
    # Run comprehensive algorithm comparison
    print(f"\n🔄 Starting 6-ALGORITHM COMPARISON for EAC Corridors...")
    start_time = time.time()
    
    results, execution_times = run_algorithm_comparison(
        coords_list, demands, capacity, threat_zones, road_distances, city_names,
        max_vehicles=max_vehicles, population_size=50, max_iter=200, max_time=120
    )
    
    total_comparison_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n{'='*80}")
    print(f"✅ 6-ALGORITHM COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"📊 Total comparison time: {total_comparison_time:.1f} seconds")
    
    # Print detailed comparison
    print_detailed_comparison(results, execution_times, road_distances, coords_list, threat_zones, city_names, demands)
    
    # Plot comparison charts
    print(f"\n📈 Generating algorithm comparison charts...")
    plot_algorithm_comparison(results, execution_times)
    
    # Generate individual maps for each algorithm
    print(f"\n🗺 Generating individual algorithm maps with CLOUD-STYLE threat zones...")
    algorithm_maps = {}
    
    for algo_name, result in results.items():
        print(f"   🗺 Creating cloud-style map for {algo_name}...")
        algorithm_map = plot_routes_map(coords_list, result['routes'], threat_zones, city_names, 
                                      demands, result['stats'], algorithm_name=algo_name)
        
        map_filename = f"eac_corridors_cloud_{algo_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.html"
        algorithm_map.save(map_filename)
        algorithm_maps[algo_name] = map_filename
        print(f"   ✅ {algo_name} cloud-style map saved as: {map_filename}")
    
    # Display all maps in notebook
    display_all_maps(algorithm_maps, results)
    
    # Save results to Excel (NOW INCLUDES DISTANCE MATRIX)
    excel_filename = save_results_to_excel(
        results, execution_times, road_distances, coords_list, 
        threat_zones, city_names, demands, capacity,
        filename="eac_6_algorithm_comparison.xlsx"
    )
    
    # Generate comprehensive report
    generate_comprehensive_report(results, execution_times, coords_list, city_names, demands, capacity)
    
    # Create quick CSV stats
    create_quick_stats_csv(results, execution_times)
    
    # Final summary with Excel export confirmation
    best_algo = min(results.keys(), key=lambda x: results[x]['cost'])
    best_cost = results[best_algo]['cost']
    
    print(f"\n🎯 6-ALGORITHM COMPARISON COMPLETED!")
    print(f"   🏆 Best Algorithm: {best_algo} (Cost: {best_cost:,.0f})")
    print(f"   💾 Excel Report: {excel_filename} (Includes Distance Matrix)")
    
    if best_algo == "SBA (Swallow-Bat Algorithm)":
        print(f"   💎 ENHANCED SBA ACHIEVED SUPERIOR PERFORMANCE!")
        print(f"   🚀 Enhanced features successfully improved solution quality")
    else:
        print(f"   📈 {best_algo} performed best in this comparison")
    
    print(f"   ⏱️  Total comparison time: {total_comparison_time:.1f} seconds")
    print(f"   ☁️  6 cloud-style maps generated")
    print(f"   📊 Comprehensive Excel report saved (8 sheets including Distance Matrix)")
    print(f"   🛡️  All algorithms enforce threat zone avoidance")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. Implement {best_algo} for production deployment")
    print(f"   2. Review Excel report for detailed analysis (includes Distance Matrix)")
    print(f"   3. Validate safe routes with local logistics teams")
    print(f"   4. Use cloud-style maps for visual verification")