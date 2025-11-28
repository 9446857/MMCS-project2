#!/usr/bin/env python3
"""
Bike Station MCLP with Weight Sensitivity Analysis

This script finds optimal locations for new bike-sharing stations in Edinburgh
by maximizing weighted coverage of demand points (bus stops, hospitals, shops, SIMD areas, etc.) prioritising transport demand points, within service radii, while accounting for station costs.

Configuration is now loaded from config.csv file.
"""

import csv
import numpy as np
import xpress as xp
import folium
from collections import defaultdict
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from folium.plugins import MeasureControl
import time
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config_csv(config_file='config.csv'):
    """Load configuration file which contains the parameter values."""
    print(f"Loading configuration from {config_file}...")
    
    config = {
        'optimization_parameters': {},
        'analysis_settings': {},
        'sensitivity_analysis': {},
        'pareto_front_analysis': {},
        'service_radii': {},
        'demand_weights': {},
        'data_files': {},
        'shop_clustering': {}
    }
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                section = row['section']
                param = row['parameter']
                value = row['value']
                
                # load the values for each parameter
                if section == 'optimization':
                    if param in ['n_stations', 'min_distance_between_stations']:
                        config['optimization_parameters'][param] = int(value)
                    else:
                        config['optimization_parameters'][param] = float(value)
                
                elif section == 'analysis':
                    config['analysis_settings'][param] = value.lower() in ['true', '1', 'yes']
                
                elif section == 'sensitivity':
                    if param == 'weight_multipliers':
                        config['sensitivity_analysis'][param] = [float(x) for x in value.split(';')]
                    elif param == 'demand_types_to_test':
                        config['sensitivity_analysis'][param] = [x.strip() for x in value.split(';')]
                
                elif section == 'pareto':
                    if param == 'n_stations_range':
                        config['pareto_front_analysis'][param] = [int(x) for x in value.split(';')]
                
                elif section == 'service_radius':
                    config['service_radii'][param] = int(value)
                
                elif section == 'demand_weight':
                    config['demand_weights'][param] = int(value)
                
                elif section == 'data_file':
                    config['data_files'][param] = value
                
                elif section == 'shop_clustering':
                    if param == 'min_shops':
                        config['shop_clustering'][param] = int(value)
                    else:
                        config['shop_clustering'][param] = int(value)
        
        print("✓ Configuration loaded successfully")
        return config
        
    except FileNotFoundError:
        print(f"ERROR: Configuration file '{config_file}' not found!")
        raise
    except Exception as e:
        print(f"ERROR: Failed to parse configuration file: {e}")
        raise

# Load configuration globally
CONFIG = load_config_csv()

# Extract frequently used parameters
N_STATIONS = CONFIG['optimization_parameters']['n_stations']
MIN_DISTANCE_BETWEEN_STATIONS = CONFIG['optimization_parameters']['min_distance_between_stations']
GRID_RESOLUTION = CONFIG['optimization_parameters']['grid_resolution']
STATION_COST = CONFIG['optimization_parameters']['station_cost']
SERVICE_RADII = CONFIG['service_radii']
DEMAND_WEIGHTS = CONFIG['demand_weights']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points (in meters)."""
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def load_csv_points(filename, required_cols=None):
    """Load points from CSV file with encoding handling."""
    if required_cols is None:
        required_cols = ['lat', 'lon']
    
    points = []
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding, newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                if not fieldnames or not all(col in fieldnames for col in required_cols):
                    continue
                
                for row_dict in reader:
                    try:
                        point = {'lat': float(row_dict['lat']), 'lon': float(row_dict['lon'])}
                        for key, value in row_dict.items():
                            if key not in ['lat', 'lon'] and value:
                                point[key] = str(value).strip()
                        points.append(point)
                    except (ValueError, KeyError):
                        continue
            
            return points
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    return []


def load_boundary(filename):
    """Load Edinburgh boundary from CSV file."""
    boundary_points = load_csv_points(filename)
    if not boundary_points:
        return None
    coords = [(p['lon'], p['lat']) for p in boundary_points]
    return Polygon(coords)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_data():
    """Load all required data from CSV files using paths from config."""
    print("Loading data from CSV files...")
    
    files = CONFIG['data_files']
    
    # Load combined transport locations and split by type
    transport_data = load_csv_points(files['transport_locations'])
    bus_stops = [p for p in transport_data if p.get('type') == 'bus']
    tram_stops = [p for p in transport_data if p.get('type') == 'tram']
    train_stations = [p for p in transport_data if p.get('type') == 'train']
    park_rides = [p for p in transport_data if p.get('type') == 'park_n_ride']
    airports = [p for p in transport_data if p.get('type') == 'airport']
    
    # Load combined cycle routes and split by type
    cycle_data = load_csv_points(files['cycle_routes'])
    cycle_network = [p for p in cycle_data if p.get('type') == 'cycle_route']
    quiet_routes = [p for p in cycle_data if p.get('type') == 'quiet_route']
    
    data = {
        'boundary': load_boundary(files['boundary']),
        'city_centre': load_csv_points(files['city_centre'])[0] if load_csv_points(files['city_centre']) else {'lat': 55.9533, 'lon': -3.1883},
        'bus_stops': bus_stops,
        'tram_stops': tram_stops,
        'train_stations': train_stations,
        'park_rides': park_rides,
        'airports': airports,
        'pois': load_csv_points(files['pois']),
        'shops': load_csv_points(files['shops']),
        'cycle_network': cycle_network,
        'quiet_routes': quiet_routes,
        'simd_areas': load_csv_points(files['simd_areas']),
    }
    
    # Map limits
    map_limit_points = load_csv_points(files['map_limits'])
    if len(map_limit_points) >= 2:
        data['map_limits'] = {
            'min_lat': min(p['lat'] for p in map_limit_points),
            'max_lat': max(p['lat'] for p in map_limit_points),
            'min_lon': min(p['lon'] for p in map_limit_points),
            'max_lon': max(p['lon'] for p in map_limit_points),
        }
    else:
        data['map_limits'] = {'min_lat': 55.85, 'max_lat': 56.00, 'min_lon': -3.45, 'max_lon': -3.05}
    
    print(f"Data loaded from combined transport file: {len(bus_stops)} bus stops, {len(tram_stops)} tram stops, {len(train_stations)} train stations, {len(park_rides)} park & rides, {len(airports)} airport")
    print(f"Data loaded from combined cycle routes: {len(cycle_network)} cycle route points, {len(quiet_routes)} quiet route points")
    print(f"Other data: {len(data['simd_areas'])} SIMD areas, {len(data['pois'])} POIs, {len(data['shops'])} shops")
    return data


def cluster_shops(shops, min_shops=None, max_distance=None):
    """Cluster shops into groups."""
    if min_shops is None:
        min_shops = CONFIG['shop_clustering']['min_shops']
    if max_distance is None:
        max_distance = CONFIG['shop_clustering']['max_distance']
    
    if not shops:
        return []
    
    coords = np.array([[s['lat'], s['lon']] for s in shops])
    tree = cKDTree(coords)
    
    clusters = []
    processed = set()
    
    for i, shop in enumerate(shops):
        if i in processed:
            continue
        
        nearby_indices = tree.query_ball_point([shop['lat'], shop['lon']], max_distance / 111320)
        
        if len(nearby_indices) >= min_shops:
            cluster_coords = coords[nearby_indices]
            clusters.append({
                'lat': np.mean(cluster_coords[:, 0]),
                'lon': np.mean(cluster_coords[:, 1]),
                'shop_count': len(nearby_indices),
                'type': 'shop_cluster'
            })
            processed.update(nearby_indices)
    
    return clusters


def create_demand_points(data, custom_weights=None):
    """
    Create weighted demand points from all data sources.
    
    Args:
        data: Dictionary of data sources
        custom_weights: Optional dict to override default weights
    
    Returns:
        List of demand points with lat, lon, weight, type, and radius
    """
    weights = custom_weights if custom_weights else DEMAND_WEIGHTS
    
    demand_points = []
    
    # Add bus stops
    for stop in data['bus_stops']:
        demand_points.append({
            'lat': stop['lat'],
            'lon': stop['lon'],
            'weight': weights['bus_stop'],
            'type': 'bus_stop',
            'radius': SERVICE_RADII['bus_stop']
        })
    
    # Add tram stops
    for stop in data['tram_stops']:
        demand_points.append({
            'lat': stop['lat'],
            'lon': stop['lon'],
            'weight': weights['tram_stop'],
            'type': 'tram_stop',
            'radius': SERVICE_RADII['tram_stop']
        })
    
    # Add train stations
    for station in data['train_stations']:
        demand_points.append({
            'lat': station['lat'],
            'lon': station['lon'],
            'weight': weights['train_station'],
            'type': 'train_station',
            'radius': SERVICE_RADII['train_station']
        })
    
    # Add park & rides
    for pr in data['park_rides']:
        demand_points.append({
            'lat': pr['lat'],
            'lon': pr['lon'],
            'weight': weights['park_ride'],
            'type': 'park_ride',
            'radius': SERVICE_RADII['park_ride']
        })
    
    # Add airport
    for airport in data['airports']:
        demand_points.append({
            'lat': airport['lat'],
            'lon': airport['lon'],
            'weight': weights['airport'],
            'type': 'airport',
            'radius': SERVICE_RADII['airport']
        })
    
    # Add POIs (hospitals, schools, libraries, universities)
    for poi in data['pois']:
        poi_type = poi.get('category', '').lower()
        
        if 'hospital' in poi_type or 'clinic' in poi_type:
            demand_type = 'hospital'
        elif 'school' in poi_type:
            demand_type = 'school'
        elif 'library' in poi_type:
            demand_type = 'library'
        elif 'university' in poi_type or 'college' in poi_type:
            demand_type = 'university'
        else:
            continue
        
        demand_points.append({
            'lat': poi['lat'],
            'lon': poi['lon'],
            'weight': weights[demand_type],
            'type': demand_type,
            'radius': SERVICE_RADII[demand_type]
        })
    
    # Add shop clusters
    shop_clusters = cluster_shops(data['shops'])
    for cluster in shop_clusters:
        demand_points.append({
            'lat': cluster['lat'],
            'lon': cluster['lon'],
            'weight': weights['shop_cluster'],
            'type': 'shop_cluster',
            'radius': SERVICE_RADII['shop_cluster'],
            'shop_count': cluster['shop_count']
        })
    
    # Add cycle network points (sample every 10th point to reduce density)
    # Weight is based on distance from city centre: w = 0.01 * distance_in_meters
    city_lat = data['city_centre']['lat']
    city_lon = data['city_centre']['lon']
    
    for i, point in enumerate(data['cycle_network']):
        if i % 10 == 0:
            dist_from_centre = haversine_distance(point['lat'], point['lon'], city_lat, city_lon)
            weight = 0.0002 * dist_from_centre  # Weight increases with distance from centre
            demand_points.append({
                'lat': point['lat'],
                'lon': point['lon'],
                'weight': weight,
                'type': 'cycle_network_point',
                'radius': SERVICE_RADII['cycle_network_point']
            })
    
    # Add quiet route points
    # Weight is based on distance from city centre: w = 0.01 * distance_in_meters
    for i, point in enumerate(data['quiet_routes']):
        if i % 10 == 0:
            dist_from_centre = haversine_distance(point['lat'], point['lon'], city_lat, city_lon)
            weight = 0.0002 * dist_from_centre  # Weight increases with distance from centre
            demand_points.append({
                'lat': point['lat'],
                'lon': point['lon'],
                'weight': weight,
                'type': 'quiet_route_point',
                'radius': SERVICE_RADII['quiet_route_point']
            })
    
    # Add SIMD areas (most deprived)
    for area in data['simd_areas']:
        demand_points.append({
            'lat': area['lat'],
            'lon': area['lon'],
            'weight': weights['simd_area'],
            'type': 'simd_area',
            'radius': SERVICE_RADII['simd_area']
        })
    
    print(f"Created {len(demand_points)} demand points")
    
    # Print breakdown
    type_counts = defaultdict(int)
    type_total_weights = defaultdict(float)
    for dp in demand_points:
        type_counts[dp['type']] += 1
        type_total_weights[dp['type']] += dp['weight']
    
    print("Demand point breakdown:")
    for dtype in sorted(type_counts.keys()):
        count = type_counts[dtype]
        total_weight = type_total_weights[dtype]
        avg_weight = total_weight / count if count > 0 else 0
        
        # Special note for cycle routes (distance-based weighting)
        if dtype in ['cycle_network_point', 'quiet_route_point']:
            print(f"  {dtype}: {count} points (avg weight={avg_weight:.1f}, distance-based)")
        else:
            print(f"  {dtype}: {count} points (weight={weights.get(dtype, 'N/A')})")
    
    return demand_points


# =============================================================================
# CANDIDATE LOCATION GENERATION
# =============================================================================

def generate_candidate_locations(map_limits, boundary):
    """Generate grid of candidate locations within Edinburgh boundary."""
    print("\nGenerating candidate locations...")
    
    lats = np.arange(map_limits['min_lat'], map_limits['max_lat'], GRID_RESOLUTION)
    lons = np.arange(map_limits['min_lon'], map_limits['max_lon'], GRID_RESOLUTION)
    
    candidates = []
    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if boundary and boundary.contains(point):
                candidates.append({'lat': lat, 'lon': lon})
    
    print(f"Generated {len(candidates)} candidate locations")
    return candidates


# =============================================================================
# OPTIMIZATION MODEL
# =============================================================================

def build_coverage_matrix(demand_points, candidates):
    """Build binary coverage matrix: candidates x demand_points."""
    print("Building coverage matrix...")
    
    n_candidates = len(candidates)
    n_demands = len(demand_points)
    
    coverage_matrix = np.zeros((n_candidates, n_demands), dtype=bool)
    
    # Build KD-tree for candidates
    candidate_coords = np.array([[c['lat'], c['lon']] for c in candidates])
    tree = cKDTree(candidate_coords)
    
    for j, demand in enumerate(demand_points):
        # Find candidates within service radius
        radius_degrees = demand['radius'] / 111320  # rough conversion
        nearby_indices = tree.query_ball_point([demand['lat'], demand['lon']], radius_degrees)
        
        # Verify with haversine distance
        for i in nearby_indices:
            dist = haversine_distance(
                candidates[i]['lat'], candidates[i]['lon'],
                demand['lat'], demand['lon']
            )
            if dist <= demand['radius']:
                coverage_matrix[i, j] = True
    
    print(f"Coverage matrix built: {n_candidates} x {n_demands}")
    return coverage_matrix


def identify_cycle_dominant_candidates(candidates, demand_points, coverage_matrix, threshold=0.5):
    """
    Identify candidates whose main contribution comes from cycle/quiet routes.
    
    A candidate is cycle-dominant if >threshold of its potential coverage weight
    comes from cycle_network_point or quiet_route_point demand types.
    
    Returns:
        set: Indices of cycle-dominant candidates
    """
    print(f"Identifying cycle-route-dominant candidates (threshold={threshold*100:.0f}%)...")
    
    n_candidates = len(candidates)
    n_demands = len(demand_points)
    
    cycle_types = {'cycle_network_point', 'quiet_route_point'}
    cycle_dominant = set()
    
    for i in range(n_candidates):
        total_weight = 0
        cycle_weight = 0
        
        # Sum weights of all demand points this candidate can cover
        for j in range(n_demands):
            if coverage_matrix[i, j]:
                weight = demand_points[j]['weight']
                total_weight += weight
                if demand_points[j]['type'] in cycle_types:
                    cycle_weight += weight
        
        # Check if cycle-dominant
        if total_weight > 0 and (cycle_weight / total_weight) > threshold:
            cycle_dominant.add(i)
    
    print(f"  Found {len(cycle_dominant)} cycle-dominant candidates (out of {n_candidates})")
    return cycle_dominant


def build_distance_constraint_pairs(candidates, min_distance, cycle_dominant_indices=None, cycle_min_distance=1000):
    """
    Build list of candidate pairs that are too close together.
    Uses grid structure for efficiency.
    
    If cycle_dominant_indices is provided, applies cycle_min_distance (default 500m)
    for any pair involving a cycle-dominant candidate.
    
    Returns:
        list: List of tuples (i, j) where candidates i and j are too close
    """
    print(f"Building minimum distance constraints...")
    print(f"  Standard min distance: {min_distance}m")
    if cycle_dominant_indices:
        print(f"  Cycle-dominant min distance: {cycle_min_distance}m")
    
    n_candidates = len(candidates)
    constraint_pairs = []
    
    # Build KD-tree for efficient proximity search
    candidate_coords = np.array([[c['lat'], c['lon']] for c in candidates])
    tree = cKDTree(candidate_coords)
    
    # Use larger search radius to catch both constraints
    max_distance = cycle_min_distance if cycle_dominant_indices else min_distance
    search_radius_degrees = max_distance / 111320
    
    for i in range(n_candidates):
        # Find all candidates within search radius
        nearby_indices = tree.query_ball_point(candidate_coords[i], search_radius_degrees)
        
        # Check actual distance for candidates with index > i (to avoid duplicates)
        for j in nearby_indices:
            if j > i:  # Only add each pair once
                dist = haversine_distance(
                    candidates[i]['lat'], candidates[i]['lon'],
                    candidates[j]['lat'], candidates[j]['lon']
                )
                
                # Determine which minimum distance applies
                if cycle_dominant_indices and (i in cycle_dominant_indices or j in cycle_dominant_indices):
                    # At least one is cycle-dominant: use 500m
                    if dist < cycle_min_distance:
                        constraint_pairs.append((i, j))
                else:
                    # Neither is cycle-dominant: use standard 300m
                    if dist < min_distance:
                        constraint_pairs.append((i, j))
    
    print(f"  Found {len(constraint_pairs)} constraint pairs")
    return constraint_pairs


def solve_mclp(demand_points, candidates, n_stations=None, station_cost=None, verbose=False):
    """
    Solve MCLP with cost constraint and minimum distance constraints using FICO Xpress.
    
    The minimum distance constraint is enforced in the optimization model itself,
    not as post-processing. Special rule: if a candidate's main contribution comes
    from cycle/quiet routes, it must be 500m from other stations (not 300m).
    
    Returns:
        dict: Solution details including selected candidates and coverage info
    """
    if n_stations is None:
        n_stations = N_STATIONS
    if station_cost is None:
        station_cost = STATION_COST
    
    if verbose:
        print(f"\nSolving MCLP for {n_stations} stations (cost={station_cost})...")
    
    n_candidates = len(candidates)
    n_demands = len(demand_points)
    
    # Build coverage matrix
    coverage_matrix = build_coverage_matrix(demand_points, candidates)
    
    # Identify cycle-dominant candidates
    cycle_dominant_indices = identify_cycle_dominant_candidates(candidates, demand_points, coverage_matrix)
    
    # Build minimum distance constraint pairs (with conditional 500m for cycle-dominant)
    constraint_pairs = build_distance_constraint_pairs(
        candidates, 
        MIN_DISTANCE_BETWEEN_STATIONS,
        cycle_dominant_indices,
        cycle_min_distance=500
    )
    
    # Create Xpress problem
    prob = xp.problem()
    
    # Decision variables
    x = [xp.var(vartype=xp.binary, name=f'x_{i}') for i in range(n_candidates)]
    y = [xp.var(vartype=xp.binary, name=f'y_{j}') for j in range(n_demands)]
    
    prob.addVariable(x, y)
    
    # Objective: maximize weighted coverage minus cost
    weights = [demand_points[j]['weight'] for j in range(n_demands)]
    prob.setObjective(xp.Sum(weights[j] * y[j] for j in range(n_demands)) - station_cost * xp.Sum(x[i] for i in range(n_candidates)), sense=xp.maximize)
    
    # Constraints: coverage definition
    for j in range(n_demands):
        covering_candidates = [i for i in range(n_candidates) if coverage_matrix[i, j]]
        if covering_candidates:
            prob.addConstraint(y[j] <= xp.Sum(x[i] for i in covering_candidates))
    
    # Constraint: at most n stations
    prob.addConstraint(xp.Sum(x[i] for i in range(n_candidates)) <= n_stations)
    
    # Constraints: minimum distance between stations
    # For each pair of candidates that are too close: x[i] + x[j] <= 1
    if verbose and len(constraint_pairs) > 0:
        print(f"  Adding {len(constraint_pairs)} minimum distance constraints...")
    
    for i, j in constraint_pairs:
        prob.addConstraint(x[i] + x[j] <= 1)
    
    # Solve
    if verbose:
        print("  Solving optimization model...")
    prob.solve()
    
    if prob.getProbStatus() != xp.mip_optimal:
        print("Warning: Optimal solution not found")
        return {'solved': False}
    
    # Extract solution
    x_vals = prob.getSolution(x)
    y_vals = prob.getSolution(y)
    
    selected_indices = [i for i in range(n_candidates) if x_vals[i] > 0.5]
    selected_candidates = [candidates[i].copy() for i in selected_indices]
    
    # Verify minimum distance constraints are satisfied
    if verbose:
        min_dist_violations = 0
        cycle_min_dist_violations = 0
        for i_idx, i in enumerate(selected_indices):
            for j_idx in range(i_idx + 1, len(selected_indices)):
                j = selected_indices[j_idx]
                dist = haversine_distance(
                    candidates[i]['lat'], candidates[i]['lon'],
                    candidates[j]['lat'], candidates[j]['lon']
                )
                
                # Check appropriate constraint
                if i in cycle_dominant_indices or j in cycle_dominant_indices:
                    if dist < 500:
                        cycle_min_dist_violations += 1
                else:
                    if dist < MIN_DISTANCE_BETWEEN_STATIONS:
                        min_dist_violations += 1
        
        if min_dist_violations > 0 or cycle_min_dist_violations > 0:
            print(f"  WARNING: Constraint violations detected!")
            if min_dist_violations > 0:
                print(f"    - {min_dist_violations} standard distance violations (< {MIN_DISTANCE_BETWEEN_STATIONS}m)")
            if cycle_min_dist_violations > 0:
                print(f"    - {cycle_min_dist_violations} cycle-dominant distance violations (< 500m)")
        else:
            print(f"  ✓ All stations satisfy minimum distance constraints")
            print(f"    - Standard: {MIN_DISTANCE_BETWEEN_STATIONS}m, Cycle-dominant: 500m")
    
    # Calculate coverage info for each station
    for station in selected_candidates:
        station['covers_count'] = 0
        station['welfare_contribution'] = 0
    
    # Calculate coverage with selected stations
    covered_demands = set()
    for j in range(n_demands):
        for station in selected_candidates:
            dist = haversine_distance(station['lat'], station['lon'],
                                     demand_points[j]['lat'], demand_points[j]['lon'])
            if dist <= demand_points[j]['radius']:
                covered_demands.add(j)
                station['covers_count'] += 1
                station['welfare_contribution'] += demand_points[j]['weight']
                break
    
    # Calculate welfare
    gross_welfare = sum(demand_points[j]['weight'] for j in covered_demands)
    net_welfare = gross_welfare - (station_cost * len(selected_candidates))
    coverage_rate = len(covered_demands) / n_demands if n_demands > 0 else 0
    
    # Coverage by type
    coverage_by_type = defaultdict(lambda: {'covered': 0, 'total': 0, 'welfare': 0})
    for j in range(n_demands):
        dtype = demand_points[j]['type']
        coverage_by_type[dtype]['total'] += 1
        if j in covered_demands:
            coverage_by_type[dtype]['covered'] += 1
            coverage_by_type[dtype]['welfare'] += demand_points[j]['weight']
    
    if verbose:
        print(f"Solution found: {len(selected_candidates)} stations")
        print(f"  Gross welfare: {gross_welfare:.2f}")
        print(f"  Cost: {station_cost * len(selected_candidates):.2f}")
        print(f"  Net welfare: {net_welfare:.2f}")
        print(f"  Coverage: {len(covered_demands)}/{n_demands} ({coverage_rate*100:.1f}%)")
    
    return {
        'solved': True,
        'selected_candidates': selected_candidates,
        'n_stations': len(selected_candidates),
        'gross_welfare': gross_welfare,
        'net_welfare': net_welfare,
        'cost': station_cost * len(selected_candidates),
        'coverage_rate': coverage_rate,
        'n_covered': len(covered_demands),
        'n_demands': n_demands,
        'coverage_by_type': dict(coverage_by_type)
    }


# =============================================================================
# WEIGHT SENSITIVITY ANALYSIS
# =============================================================================

def run_weight_sensitivity_analysis(data, candidates, n_stations=None):
    """
    Run weight sensitivity analysis for specified demand types.
    """
    if n_stations is None:
        n_stations = N_STATIONS
    
    sensitivity_config = CONFIG['sensitivity_analysis']
    weight_multipliers = sensitivity_config['weight_multipliers']
    demand_types = sensitivity_config['demand_types_to_test']
    
    print(f"\nTesting {len(demand_types)} demand types with {len(weight_multipliers)} multipliers each")
    print(f"Demand types: {', '.join(demand_types)}")
    print(f"Multipliers: {weight_multipliers}")
    
    results = []
    total_runs = len(demand_types) * len(weight_multipliers)
    run_count = 0
    
    for demand_type in demand_types:
        print(f"\n{'='*70}")
        print(f"Testing: {demand_type}")
        print(f"{'='*70}")
        
        for multiplier in weight_multipliers:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] {demand_type} with weight × {multiplier}")
            
            # Create custom weights
            custom_weights = DEMAND_WEIGHTS.copy()
            custom_weights[demand_type] = int(DEMAND_WEIGHTS[demand_type] * multiplier)
            
            # Create demand points with custom weights
            demand_points = create_demand_points(data, custom_weights=custom_weights)
            
            # Solve
            result = solve_mclp(demand_points, candidates, n_stations=n_stations, verbose=False)
            
            if result['solved']:
                result['demand_type'] = demand_type
                result['multiplier'] = multiplier
                result['test_weight'] = custom_weights[demand_type]
                results.append(result)
                
                print(f"  Net welfare: {result['net_welfare']:.2f}")
                print(f"  Coverage: {result['coverage_rate']*100:.1f}%")
    
    # Save results
    print(f"\nSaving detailed results...")
    with open('weight_sensitivity_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'demand_type', 'weight_multiplier', 'test_weight', 'n_stations',
            'gross_welfare', 'net_welfare', 'overall_coverage_rate',
            'type_covered', 'type_total', 'type_coverage_rate', 'type_welfare'
        ])
        
        for r in results:
            type_cov = r['coverage_by_type'].get(r['demand_type'], {'covered': 0, 'total': 0, 'welfare': 0})
            type_cov_rate = type_cov['covered'] / type_cov['total'] if type_cov['total'] > 0 else 0
            
            writer.writerow([
                r['demand_type'],
                r['multiplier'],
                r['test_weight'],
                r['n_stations'],
                r['gross_welfare'],
                r['net_welfare'],
                r['coverage_rate'],
                type_cov['covered'],
                type_cov['total'],
                type_cov_rate,
                type_cov['welfare']
            ])
    
    print(f"✓ Saved detailed results: weight_sensitivity_results.csv")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nTested {len(demand_types)} demand types")
    print(f"Tested {len(weight_multipliers)} weight multipliers: {weight_multipliers}")
    print(f"Total optimization runs: {len(results)}")
    
    # Find most sensitive demand types
    print(f"\nWeight Sensitivity Ranking (by welfare variance):")
    welfare_variance = {}
    for demand_type in demand_types:
        type_results = [r for r in results if r['demand_type'] == demand_type]
        if len(type_results) > 1:
            welfares = [r['net_welfare'] for r in type_results]
            welfare_variance[demand_type] = np.std(welfares)
    
    for rank, (dtype, variance) in enumerate(sorted(welfare_variance.items(), 
                                                     key=lambda x: x[1], reverse=True), 1):
        print(f"  {rank}. {dtype}: σ = {variance:.2f}")
    
    return results


# =============================================================================
# PARETO FRONT ANALYSIS
# =============================================================================

def run_pareto_front_analysis(data, candidates):
    """
    Run Pareto front analysis exploring the relationship between:
    - Number of stations
    - Total welfare (gross welfare)
    """
    pareto_config = CONFIG['pareto_front_analysis']
    n_stations_range = pareto_config['n_stations_range']
    
    print(f"\nRunning Pareto Front Analysis")
    print(f"Station counts: {n_stations_range}")
    print(f"Fixed station cost: {STATION_COST}")
    
    demand_points = create_demand_points(data)
    results = []
    total_runs = len(n_stations_range)
    run_count = 0
    
    for n_stat in n_stations_range:
        run_count += 1
        print(f"\n[{run_count}/{total_runs}] n_stations={n_stat}")
        
        result = solve_mclp(demand_points, candidates, 
                          n_stations=n_stat, station_cost=STATION_COST, verbose=False)
        
        if result['solved']:
            result['params'] = {'n_stations': n_stat}
            results.append(result)
            print(f"  Gross welfare: {result['gross_welfare']:.2f}")
            print(f"  Net welfare: {result['net_welfare']:.2f}")
    
    # Save results
    with open('pareto_front_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n_stations', 'gross_welfare', 'total_cost', 
                        'net_welfare', 'coverage_rate', 'n_covered', 'n_demands'])
        
        for r in results:
            writer.writerow([
                r['params']['n_stations'],
                r['gross_welfare'],
                r['cost'],
                r['net_welfare'],
                r['coverage_rate'],
                r['n_covered'],
                r['n_demands']
            ])
    
    print(f"\n✓ Saved Pareto front results: pareto_front_results.csv")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Stations vs Gross Welfare (Total Welfare)
    stations = [r['params']['n_stations'] for r in results]
    gross_welfare = [r['gross_welfare'] for r in results]
    axes[0].plot(stations, gross_welfare, marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Number of Stations', fontsize=12)
    axes[0].set_ylabel('Total Welfare (Gross)', fontsize=12)
    axes[0].set_title('Pareto Front: Stations vs Total Welfare', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Stations vs Net Welfare
    net_welfare = [r['net_welfare'] for r in results]
    axes[1].plot(stations, net_welfare, marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_xlabel('Number of Stations', fontsize=12)
    axes[1].set_ylabel('Net Welfare (Gross - Cost)', fontsize=12)
    axes[1].set_title('Stations vs Net Welfare', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_front_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto front plot: pareto_front_analysis.png")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_result_map(selected_candidates, demand_points, city_centre):
    """Create an interactive map of the solution with layers."""
    print("\nCreating interactive map...")
    
    # Create map without showing base layer in control
    m = folium.Map(location=[city_centre['lat'], city_centre['lon']], 
                   zoom_start=12, tiles='OpenStreetMap', 
                   control_scale=True)
    
    m.add_child(MeasureControl(position='bottomright', primary_length_unit='meters'))
    
    # Create feature groups (layers)
    layers = {
        'stations': folium.FeatureGroup(name='Bike Stations', show=True),
        'simd_areas': folium.FeatureGroup(name='SIMD Areas', show=False),
        'hospitals': folium.FeatureGroup(name='Hospitals', show=False),
        'train_stations': folium.FeatureGroup(name='Train Stations', show=False),
        'universities': folium.FeatureGroup(name='Universities', show=False),
        'schools': folium.FeatureGroup(name='Schools', show=False),
        'libraries': folium.FeatureGroup(name='Libraries', show=False),
        'bus_stops': folium.FeatureGroup(name='Bus Stops', show=False),
        'tram_stops': folium.FeatureGroup(name='Tram Stops', show=False),
        'park_rides': folium.FeatureGroup(name='Park & Ride', show=False),
        'shop_clusters': folium.FeatureGroup(name='Shop Clusters', show=False),
        'cycle_routes': folium.FeatureGroup(name='Cycling Infrastructure', show=False),
        'airport': folium.FeatureGroup(name='Airport', show=False),
    }
    
    # Add bike stations
    for i, station in enumerate(selected_candidates):
        folium.Marker(
            location=[station['lat'], station['lon']],
            popup=f"<b>Station {i+1}</b><br>Covers: {station['covers_count']} points<br>Welfare: {station['welfare_contribution']:.1f}",
            tooltip=f"Station {i+1}",
            icon=folium.Icon(color='red', icon='bicycle', prefix='fa')
        ).add_to(layers['stations'])
    
    # Add demand points by type
    type_layer_map = {
        'simd_area': 'simd_areas',
        'hospital': 'hospitals',
        'train_station': 'train_stations',
        'university': 'universities',
        'school': 'schools',
        'library': 'libraries',
        'bus_stop': 'bus_stops',
        'tram_stop': 'tram_stops',
        'park_ride': 'park_rides',
        'shop_cluster': 'shop_clusters',
        'cycle_network_point': 'cycle_routes',
        'quiet_route_point': 'cycle_routes',
        'airport': 'airport',
    }
    
    # Color and icon scheme - stronger, more vibrant colors
    type_colors = {
        'simd_area': 'darkorange',
        'hospital': 'red',
        'train_station': 'darkblue',
        'university': 'darkviolet',
        'school': 'darkgreen',
        'library': 'blue',
        'bus_stop': 'darkgray',
        'tram_stop': 'gray',
        'park_ride': 'navy',
        'shop_cluster': 'deeppink',
        'cycle_network_point': 'green',
        'quiet_route_point': 'limegreen',
        'airport': 'darkred',
    }
    
    for demand in demand_points:
        dtype = demand['type']
        if dtype in type_layer_map:
            layer_name = type_layer_map[dtype]
            color = type_colors.get(dtype, 'blue')
            
            # Use CircleMarker for better performance with many points
            if dtype in ['bus_stop', 'tram_stop', 'cycle_network_point', 'quiet_route_point']:
                folium.CircleMarker(
                    location=[demand['lat'], demand['lon']],
                    radius=3,
                    popup=f"<b>{dtype.replace('_', ' ').title()}</b><br>Weight: {demand['weight']}",
                    color=color,
                    fill=True,
                    fillOpacity=0.4
                ).add_to(layers[layer_name])
            else:
                # Use markers for important locations
                folium.CircleMarker(
                    location=[demand['lat'], demand['lon']],
                    radius=5,
                    popup=f"<b>{dtype.replace('_', ' ').title()}</b><br>Weight: {demand['weight']}",
                    tooltip=dtype.replace('_', ' ').title(),
                    color=color,
                    fill=True,
                    fillOpacity=0.6
                ).add_to(layers[layer_name])
    
    # Add all layers to map
    for layer in layers.values():
        layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    return m


def save_solution_csv(result, filename='selected_stations.csv'):
    """Save solution to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['station_id', 'lat', 'lon', 'covers_count', 'welfare_contribution'])
        
        for i, station in enumerate(result['selected_candidates']):
            writer.writerow([i + 1, station['lat'], station['lon'], 
                           station['covers_count'], station['welfare_contribution']])
    
    print(f"✓ Saved solution: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("BIKE STATION OPTIMIZATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of stations: {N_STATIONS}")
    print(f"  Station cost: {STATION_COST}")
    print(f"  Sensitivity analysis: {CONFIG['analysis_settings']['run_sensitivity_analysis']}")
    print(f"  Pareto front analysis: {CONFIG['analysis_settings']['run_pareto_front_analysis']}")
    
    start_time = time.time()
    
    # Load data
    data = load_all_data()
    demand_points = create_demand_points(data)
    candidates = generate_candidate_locations(data['map_limits'], data['boundary'])
    
    # Run standard optimization with baseline weights
    print("\n" + "="*70)
    print("STEP 1: BASELINE OPTIMIZATION")
    print("="*70)
    result = solve_mclp(demand_points, candidates, n_stations=N_STATIONS, 
                       station_cost=STATION_COST, verbose=True)
    
    if result['solved']:
        # Save baseline solution
        save_solution_csv(result, 'selected_stations_baseline.csv')
        result_map = create_result_map(result['selected_candidates'], demand_points, data['city_centre'])
        result_map.save('bike_stations_map_baseline.html')
        print("\n✓ Saved baseline map: bike_stations_map_baseline.html")
        
        # Print baseline coverage by type
        print("\n  Coverage by demand type:")
        for dtype, cov in sorted(result['coverage_by_type'].items()):
            cov_rate = 100 * cov['covered'] / cov['total'] if cov['total'] > 0 else 0
            print(f"    {dtype}: {cov['covered']}/{cov['total']} ({cov_rate:.1f}%) - Welfare: {cov['welfare']:.1f}")
    
    # Run weight sensitivity analysis if enabled
    if CONFIG['analysis_settings']['run_sensitivity_analysis']:
        print("\n" + "="*70)
        print("STEP 2: WEIGHT SENSITIVITY ANALYSIS")
        print("="*70)
        sensitivity_results = run_weight_sensitivity_analysis(data, candidates, n_stations=N_STATIONS)
    
    # Run Pareto front analysis if enabled
    if CONFIG['analysis_settings']['run_pareto_front_analysis']:
        print("\n" + "="*70)
        print("STEP 3: PARETO FRONT ANALYSIS")
        print("="*70)
        pareto_results = run_pareto_front_analysis(data, candidates)
    
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print(f"COMPLETE - Total time: {elapsed:.1f} seconds")
    print("="*70)


if __name__ == '__main__':
    main()
