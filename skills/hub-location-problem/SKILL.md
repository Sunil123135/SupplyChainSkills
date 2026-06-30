---
name: hub-location-problem
description: "When the user wants to solve hub location problems, design hub-and-spoke networks, optimize hub placement for logistics networks. Also use when the user mentions \"hub location,\" \"hub-and-spoke,\" \"p-hub median,\" \"hub covering,\" \"single allocation hub,\" \"multiple allocation hub,\" \"airline hub network,\" \"postal hub network,\" or \"consolidation center location.\" For general facility location, see facility-location-problem. For distribution centers, see distribution-center-network."
---

# Hub Location Problem

You are an expert in hub location problems and hub-and-spoke network design. Your goal is to help design efficient hub-and-spoke networks where flows between origin-destination pairs are consolidated through hub facilities, minimizing total transportation and routing costs.

## Initial Assessment

Before solving hub location problems, understand:

1. **Network Type**
   - Hub-and-spoke network? (consolidation through hubs)
   - How many hubs to locate? (p-hub problem)
   - Hub covering problem? (coverage requirements)
   - Hub hierarchy? (multi-level hubs)

2. **Allocation Type**
   - Single allocation? (each node assigned to exactly one hub)
   - Multiple allocation? (nodes can connect to multiple hubs)
   - r-allocation? (each node connects to at most r hubs)

3. **Flow Characteristics**
   - Origin-destination (O-D) flow matrix?
   - Demand between all pairs?
   - Flow volumes/patterns?
   - Directionality (symmetric or asymmetric)?

4. **Hub Characteristics**
   - Hub capacity constraints?
   - Fixed costs to establish hubs?
   - Hub processing/handling costs?
   - Economies of scale at hubs?
   - Discount factor (α) for inter-hub connections?

5. **Network Objectives**
   - Minimize total transportation cost?
   - Minimize maximum travel time? (p-hub center)
   - Minimize number of hubs?
   - Balance hub loads?

---

## Hub Location Problem Framework

### Hub-and-Spoke Network Concept

**Structure:**
```
Origin → Access Hub → Hub-to-Hub → Egress Hub → Destination

Collection   |  Consolidation  |   Distribution
   Leg       |      Leg        |      Leg
```

**Key Features:**
- **Consolidation**: Flows between O-D pairs routed through hubs
- **Economies of Scale**: Inter-hub transport cheaper (discount factor α)
- **Hub Facilities**: Special facilities with sorting/consolidation capability
- **Reduced Connections**: Not all nodes directly connected

**Applications:**
- Airline passenger networks
- Cargo/freight networks
- Postal/package delivery
- Telecommunications networks
- Supply chain distribution
- Public transportation

---

## Problem Classification

### 1. p-Hub Median Problem (pHMP)

**Description:**
- Locate exactly p hubs
- Minimize total transportation cost
- Most common hub location variant

**Variants:**
- **Single Allocation (SA-pHMP)**: Each non-hub node connected to exactly one hub
- **Multiple Allocation (MA-pHMP)**: Non-hub nodes can connect to multiple hubs

**Characteristics:**
- Fixed number of hubs (p)
- Cost minimization
- Network efficiency focus

### 2. Hub Covering Problem

**Description:**
- Cover all O-D pairs within distance/time threshold
- Minimize number of hubs needed

**Applications:**
- Emergency service networks
- Time-sensitive delivery
- Service quality guarantees

### 3. p-Hub Center Problem

**Description:**
- Minimize maximum O-D distance/cost
- Equity-focused objective
- Ensure no O-D pair has excessive cost

**Applications:**
- Emergency response
- Equal service quality
- Fair network design

### 4. Hub Arc Location Problem

**Description:**
- Decide which inter-hub connections to establish
- Not all hubs necessarily connected
- Trade-off connectivity vs. cost

### 5. Hierarchical Hub Location

**Description:**
- Multiple levels of hubs (regional, national, international)
- Flows cascade through hierarchy
- Complex multi-tier networks

---

## Mathematical Formulations

### Single Allocation p-Hub Median Problem (SA-pHMP)

**Sets:**
- N = {1, ..., n}: Set of nodes (potential hubs and non-hubs)

**Parameters:**
- w_{ij}: Flow from node i to node j
- c_{ij}: Unit cost from node i to node j
- α: Inter-hub discount factor (typically 0.2 - 0.8)
- p: Number of hubs to locate

**Decision Variables:**
- y_k ∈ {0,1}: 1 if node k is a hub, 0 otherwise
- x_{ik} ∈ {0,1}: 1 if node i is allocated to hub k, 0 otherwise

**Cost Structure:**
```
For flow from i to j allocated to hubs k and m:
- Collection: c_{ik} (origin to hub)
- Transfer: α × c_{km} (hub to hub, discounted)
- Distribution: c_{mj} (hub to destination)

Total: c_{ik} + α × c_{km} + c_{mj}
```

**Objective Function:**
```
Minimize: Σ_i Σ_j Σ_k Σ_m w_{ij} × (c_{ik} + α×c_{km} + c_{mj}) × x_{ik} × x_{jm}
```

**Constraints:**
```
1. Each node allocated to exactly one hub:
   Σ_k x_{ik} = 1,  ∀i ∈ N

2. Exactly p hubs located:
   Σ_k y_k = p

3. Allocation only to hubs:
   x_{ik} ≤ y_k,  ∀i ∈ N, ∀k ∈ N

4. Hubs allocated to themselves:
   x_{kk} = y_k,  ∀k ∈ N

5. Binary variables:
   y_k ∈ {0,1},  ∀k ∈ N
   x_{ik} ∈ {0,1},  ∀i,k ∈ N
```

**Complexity:** NP-hard

### Multiple Allocation p-Hub Median Problem (MA-pHMP)

**Modified Variables:**
- x_{ikm} ∈ [0,1]: Fraction of flow from i to j through hubs k and m

**Objective:**
```
Minimize: Σ_i Σ_j Σ_k Σ_m w_{ij} × (c_{ik} + α×c_{km} + c_{mj}) × x_{ikm}
```

**Key Constraints:**
```
1. Flow conservation:
   Σ_k Σ_m x_{ikm} = 1,  ∀i,j ∈ N

2. Exactly p hubs:
   Σ_k y_k = p

3. Flow only through hubs:
   x_{ikm} ≤ y_k,  ∀i,j,k,m
   x_{ikm} ≤ y_m,  ∀i,j,k,m
```

---

## Exact Solution Methods

### 1. Single Allocation p-Hub Median (PuLP)

```python
from pulp import *
import numpy as np

def solve_single_allocation_phub(flows, costs, p, alpha=0.75):
    """
    Solve Single Allocation p-Hub Median Problem

    Args:
        flows: n x n matrix of flows between O-D pairs
        costs: n x n matrix of unit transportation costs
        p: number of hubs to locate
        alpha: inter-hub discount factor (0 < α < 1)

    Returns:
        optimal hub location and allocation solution
    """
    n = len(flows)

    # Create problem
    prob = LpProblem("Single_Allocation_pHub_Median", LpMinimize)

    # Decision variables
    # y[k] = 1 if node k is a hub
    y = LpVariable.dicts("hub", range(n), cat='Binary')

    # x[i,k] = 1 if node i is allocated to hub k
    x = LpVariable.dicts("allocation",
                         [(i, k) for i in range(n) for k in range(n)],
                         cat='Binary')

    # Objective: Minimize total weighted transportation cost
    # Cost for flow from i to j through hubs k and m:
    # c_ik + alpha*c_km + c_mj
    objective = []

    for i in range(n):
        for j in range(n):
            if i == j or flows[i][j] == 0:
                continue

            for k in range(n):
                for m in range(n):
                    cost_ij_via_km = (costs[i][k] +
                                     alpha * costs[k][m] +
                                     costs[m][j])

                    objective.append(
                        flows[i][j] * cost_ij_via_km * x[i,k] * x[j,m]
                    )

    prob += lpSum(objective), "Total_Transportation_Cost"

    # Constraints

    # 1. Each node allocated to exactly one hub
    for i in range(n):
        prob += (
            lpSum([x[i,k] for k in range(n)]) == 1,
            f"Allocation_{i}"
        )

    # 2. Exactly p hubs located
    prob += (
        lpSum([y[k] for k in range(n)]) == p,
        "p_Hubs"
    )

    # 3. Can only allocate to hubs
    for i in range(n):
        for k in range(n):
            prob += (
                x[i,k] <= y[k],
                f"Allocation_to_Hub_{i}_{k}"
            )

    # 4. Hubs must be allocated to themselves
    for k in range(n):
        prob += (
            x[k,k] == y[k],
            f"Hub_Self_Allocation_{k}"
        )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        hubs = [k for k in range(n) if y[k].varValue > 0.5]

        allocations = {}
        for i in range(n):
            for k in range(n):
                if x[i,k].varValue > 0.5:
                    allocations[i] = k
                    break

        # Calculate flows through each hub
        hub_inflow = {k: 0 for k in hubs}
        hub_outflow = {k: 0 for k in hubs}

        for i in range(n):
            for j in range(n):
                if i != j and flows[i][j] > 0:
                    hub_i = allocations[i]
                    hub_j = allocations[j]

                    hub_outflow[hub_i] += flows[i][j]
                    hub_inflow[hub_j] += flows[i][j]

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'hubs': hubs,
            'num_hubs': len(hubs),
            'allocations': allocations,
            'hub_inflow': hub_inflow,
            'hub_outflow': hub_outflow,
            'solve_time': solve_time,
            'alpha': alpha
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # 10-node network
    n = 10

    # Generate random coordinates
    coords = np.random.rand(n, 2) * 100

    # Calculate Euclidean distance matrix
    costs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            costs[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Generate flow matrix (random demands)
    flows = np.random.uniform(10, 100, (n, n))
    np.fill_diagonal(flows, 0)  # No self-flow

    # Parameters
    p = 3  # Number of hubs
    alpha = 0.75  # Inter-hub discount factor

    print(f"{'='*70}")
    print(f"SINGLE ALLOCATION p-HUB MEDIAN PROBLEM")
    print(f"{'='*70}")
    print(f"Nodes: {n}")
    print(f"Hubs to locate: {p}")
    print(f"Inter-hub discount factor: {alpha}")
    print(f"Total O-D flow: {flows.sum():.2f}")

    result = solve_single_allocation_phub(flows, costs, p, alpha)

    print(f"\n{'='*70}")
    print(f"SOLUTION")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Total Cost: {result['total_cost']:,.2f}")
    print(f"Hubs Located: {result['hubs']}")
    print(f"Solve Time: {result['solve_time']:.2f} seconds")

    print(f"\nNode Allocations:")
    for node, hub in result['allocations'].items():
        hub_indicator = " (HUB)" if node in result['hubs'] else ""
        print(f"  Node {node} → Hub {hub}{hub_indicator}")

    print(f"\nHub Traffic:")
    for hub in result['hubs']:
        print(f"  Hub {hub}:")
        print(f"    Inflow: {result['hub_inflow'][hub]:.2f}")
        print(f"    Outflow: {result['hub_outflow'][hub]:.2f}")
        print(f"    Total: {result['hub_inflow'][hub] + result['hub_outflow'][hub]:.2f}")
```

### 2. Multiple Allocation p-Hub Median

```python
def solve_multiple_allocation_phub(flows, costs, p, alpha=0.75):
    """
    Solve Multiple Allocation p-Hub Median Problem

    Allows each node to send/receive flows through multiple hubs

    Args:
        flows: n x n flow matrix
        costs: n x n cost matrix
        p: number of hubs
        alpha: inter-hub discount factor

    Returns:
        optimal solution
    """
    n = len(flows)

    prob = LpProblem("Multiple_Allocation_pHub_Median", LpMinimize)

    # Decision variables
    y = LpVariable.dicts("hub", range(n), cat='Binary')

    # x[i,j,k,m] = fraction of flow from i to j through hubs k and m
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j and flows[i][j] > 0:
                for k in range(n):
                    for m in range(n):
                        x[i,j,k,m] = LpVariable(f"flow_{i}_{j}_{k}_{m}",
                                               lowBound=0, upBound=1,
                                               cat='Continuous')

    # Objective: Minimize total cost
    objective = []
    for i in range(n):
        for j in range(n):
            if i != j and flows[i][j] > 0:
                for k in range(n):
                    for m in range(n):
                        cost_via_km = (costs[i][k] +
                                      alpha * costs[k][m] +
                                      costs[m][j])

                        objective.append(
                            flows[i][j] * cost_via_km * x[i,j,k,m]
                        )

    prob += lpSum(objective), "Total_Cost"

    # Constraints

    # 1. Flow conservation: all flow from i to j routed through some hub pair
    for i in range(n):
        for j in range(n):
            if i != j and flows[i][j] > 0:
                prob += (
                    lpSum([x[i,j,k,m] for k in range(n) for m in range(n)]) == 1,
                    f"Flow_{i}_{j}"
                )

    # 2. Exactly p hubs
    prob += (
        lpSum([y[k] for k in range(n)]) == p,
        "p_Hubs"
    )

    # 3. Flow only through open hubs
    for i in range(n):
        for j in range(n):
            if i != j and flows[i][j] > 0:
                for k in range(n):
                    for m in range(n):
                        prob += (
                            x[i,j,k,m] <= y[k],
                            f"Hub_k_{i}_{j}_{k}_{m}"
                        )
                        prob += (
                            x[i,j,k,m] <= y[m],
                            f"Hub_m_{i}_{j}_{k}_{m}"
                        )

    # Solve
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        hubs = [k for k in range(n) if y[k].varValue > 0.5]

        # Extract flow routing
        flow_routing = {}
        for i in range(n):
            for j in range(n):
                if i != j and flows[i][j] > 0:
                    flow_routing[i,j] = []
                    for k in range(n):
                        for m in range(n):
                            if x[i,j,k,m].varValue > 0.01:
                                flow_routing[i,j].append({
                                    'via_hubs': (k, m),
                                    'fraction': x[i,j,k,m].varValue,
                                    'flow': flows[i][j] * x[i,j,k,m].varValue
                                })

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'hubs': hubs,
            'flow_routing': flow_routing,
            'solve_time': solve_time,
            'alpha': alpha
        }
    else:
        return {'status': LpStatus[prob.status]}


# Example usage
result_ma = solve_multiple_allocation_phub(flows, costs, p=3, alpha=0.75)

print(f"\n{'='*70}")
print(f"MULTIPLE ALLOCATION p-HUB MEDIAN SOLUTION")
print(f"{'='*70}")
print(f"Status: {result_ma['status']}")
print(f"Total Cost: {result_ma['total_cost']:,.2f}")
print(f"Hubs: {result_ma['hubs']}")

print(f"\nSample Flow Routing (first 5 O-D pairs):")
count = 0
for (i, j), routing in result_ma['flow_routing'].items():
    if count >= 5:
        break
    print(f"  Flow {i}→{j} (total={flows[i][j]:.2f}):")
    for route in routing:
        via_k, via_m = route['via_hubs']
        print(f"    via hubs ({via_k}, {via_m}): "
              f"{route['fraction']*100:.1f}% ({route['flow']:.2f} units)")
    count += 1
```

---

## Greedy Heuristics

### 1. Greedy Hub Selection

```python
def greedy_hub_selection(flows, costs, p, alpha=0.75):
    """
    Greedy heuristic for p-Hub Median

    Iteratively select hub that gives maximum cost reduction

    Args:
        flows: flow matrix
        costs: cost matrix
        p: number of hubs
        alpha: inter-hub discount

    Returns:
        heuristic solution
    """
    n = len(flows)

    hubs = []
    allocations = {}

    def calculate_cost(current_hubs):
        """Calculate total cost for given hub set"""
        if not current_hubs:
            return float('inf')

        # Allocate each node to nearest hub
        node_allocations = {}
        for i in range(n):
            nearest_hub = min(current_hubs, key=lambda k: costs[i][k])
            node_allocations[i] = nearest_hub

        # Calculate total cost
        total_cost = 0
        for i in range(n):
            for j in range(n):
                if i != j and flows[i][j] > 0:
                    hub_i = node_allocations[i]
                    hub_j = node_allocations[j]

                    cost = (costs[i][hub_i] +
                           alpha * costs[hub_i][hub_j] +
                           costs[hub_j][j])

                    total_cost += flows[i][j] * cost

        return total_cost, node_allocations

    # Greedily select p hubs
    for iteration in range(p):
        best_hub = None
        best_cost = float('inf')
        best_allocations = None

        # Try adding each remaining node as a hub
        for k in range(n):
            if k in hubs:
                continue

            test_hubs = hubs + [k]
            cost, allocations_test = calculate_cost(test_hubs)

            if cost < best_cost:
                best_cost = cost
                best_hub = k
                best_allocations = allocations_test

        if best_hub is not None:
            hubs.append(best_hub)
            allocations = best_allocations

    return {
        'hubs': hubs,
        'allocations': allocations,
        'total_cost': best_cost,
        'method': 'Greedy Hub Selection'
    }
```

### 2. Concentration-Based Heuristic

```python
def concentration_heuristic(flows, costs, p, alpha=0.75):
    """
    Concentration-based heuristic for hub location

    Select nodes with highest total flow (originating + terminating)

    Args:
        flows: flow matrix
        costs: cost matrix
        p: number of hubs
        alpha: inter-hub discount

    Returns:
        heuristic solution
    """
    n = len(flows)

    # Calculate total flow for each node
    node_flows = []
    for i in range(n):
        total_flow = flows[i, :].sum() + flows[:, i].sum()
        node_flows.append((total_flow, i))

    # Sort by flow (descending)
    node_flows.sort(reverse=True)

    # Select top p nodes as hubs
    hubs = [node for _, node in node_flows[:p]]

    # Allocate each node to nearest hub
    allocations = {}
    for i in range(n):
        nearest_hub = min(hubs, key=lambda k: costs[i][k])
        allocations[i] = nearest_hub

    # Calculate total cost
    total_cost = 0
    for i in range(n):
        for j in range(n):
            if i != j and flows[i][j] > 0:
                hub_i = allocations[i]
                hub_j = allocations[j]

                cost = (costs[i][hub_i] +
                       alpha * costs[hub_i][hub_j] +
                       costs[hub_j][j])

                total_cost += flows[i][j] * cost

    return {
        'hubs': hubs,
        'allocations': allocations,
        'total_cost': total_cost,
        'method': 'Concentration Heuristic'
    }
```

---

## Local Search Improvements

### 1. Hub Swap Local Search

```python
def hub_swap_local_search(flows, costs, initial_hubs, alpha=0.75,
                         max_iterations=100):
    """
    Local search with hub swap moves

    Swap a hub with a non-hub if it improves objective

    Args:
        flows: flow matrix
        costs: cost matrix
        initial_hubs: initial hub locations
        alpha: inter-hub discount
        max_iterations: maximum iterations

    Returns:
        improved solution
    """
    n = len(flows)

    current_hubs = set(initial_hubs)
    non_hubs = set(range(n)) - current_hubs

    def evaluate_solution(hub_set):
        """Evaluate cost for given hub configuration"""
        # Allocate nodes
        allocations = {}
        for i in range(n):
            allocations[i] = min(hub_set, key=lambda k: costs[i][k])

        # Calculate cost
        total_cost = 0
        for i in range(n):
            for j in range(n):
                if i != j and flows[i][j] > 0:
                    hub_i = allocations[i]
                    hub_j = allocations[j]

                    cost = (costs[i][hub_i] +
                           alpha * costs[hub_i][hub_j] +
                           costs[hub_j][j])

                    total_cost += flows[i][j] * cost

        return total_cost, allocations

    current_cost, current_allocations = evaluate_solution(current_hubs)
    best_hubs = current_hubs.copy()
    best_cost = current_cost
    best_allocations = current_allocations

    for iteration in range(max_iterations):
        improved = False

        # Try all possible swaps
        for hub_out in list(current_hubs):
            for hub_in in non_hubs:
                # Test swap
                test_hubs = (current_hubs - {hub_out}) | {hub_in}
                test_cost, test_allocations = evaluate_solution(test_hubs)

                if test_cost < best_cost - 1e-6:
                    best_cost = test_cost
                    best_hubs = test_hubs
                    best_allocations = test_allocations
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

        # Update current solution
        current_hubs = best_hubs.copy()
        non_hubs = set(range(n)) - current_hubs

    return {
        'hubs': list(best_hubs),
        'allocations': best_allocations,
        'total_cost': best_cost,
        'method': 'Hub Swap Local Search'
    }
```

---

## Metaheuristics

### 1. Simulated Annealing for Hub Location

```python
import random
import math

def simulated_annealing_hub(flows, costs, p, alpha=0.75,
                           initial_temp=1000, cooling_rate=0.95,
                           max_iterations=5000):
    """
    Simulated Annealing for p-Hub Median

    Args:
        flows: flow matrix
        costs: cost matrix
        p: number of hubs
        alpha: inter-hub discount
        initial_temp: starting temperature
        cooling_rate: cooling factor
        max_iterations: max iterations

    Returns:
        best solution found
    """
    n = len(flows)

    def evaluate(hub_set):
        """Calculate cost for hub configuration"""
        allocations = {}
        for i in range(n):
            allocations[i] = min(hub_set, key=lambda k: costs[i][k])

        total_cost = 0
        for i in range(n):
            for j in range(n):
                if i != j and flows[i][j] > 0:
                    hub_i = allocations[i]
                    hub_j = allocations[j]

                    cost = (costs[i][hub_i] +
                           alpha * costs[hub_i][hub_j] +
                           costs[hub_j][j])

                    total_cost += flows[i][j] * cost

        return total_cost, allocations

    # Initial solution: random hubs
    current_hubs = set(random.sample(range(n), p))
    current_cost, current_allocations = evaluate(current_hubs)

    best_hubs = current_hubs.copy()
    best_cost = current_cost
    best_allocations = current_allocations

    temperature = initial_temp

    for iteration in range(max_iterations):
        # Generate neighbor: swap one hub
        non_hubs = list(set(range(n)) - current_hubs)

        hub_out = random.choice(list(current_hubs))
        hub_in = random.choice(non_hubs)

        neighbor_hubs = (current_hubs - {hub_out}) | {hub_in}
        neighbor_cost, neighbor_allocations = evaluate(neighbor_hubs)

        delta = neighbor_cost - current_cost

        # Accept or reject
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_hubs = neighbor_hubs
            current_cost = neighbor_cost
            current_allocations = neighbor_allocations

            if current_cost < best_cost:
                best_hubs = current_hubs.copy()
                best_cost = current_cost
                best_allocations = current_allocations

        # Cool down
        temperature *= cooling_rate

        if temperature < 0.1:
            break

    return {
        'hubs': list(best_hubs),
        'allocations': best_allocations,
        'total_cost': best_cost,
        'method': 'Simulated Annealing'
    }
```

---

## Complete Hub Location Solver

```python
class HubLocationSolver:
    """
    Comprehensive Hub Location Problem Solver

    Supports single/multiple allocation, various solution methods
    """

    def __init__(self):
        self.flows = None
        self.costs = None
        self.coords = None
        self.n = None

    def load_problem(self, flows, costs, coords=None):
        """
        Load problem data

        Args:
            flows: n x n flow matrix
            costs: n x n cost matrix
            coords: optional node coordinates for visualization
        """
        self.flows = np.array(flows)
        self.costs = np.array(costs)
        self.coords = np.array(coords) if coords is not None else None
        self.n = len(flows)

        print(f"Loaded hub location problem:")
        print(f"  Nodes: {self.n}")
        print(f"  Total O-D flow: {self.flows.sum():.2f}")
        print(f"  Average cost: {self.costs.mean():.2f}")

    def solve_exact_sa(self, p, alpha=0.75, time_limit=600):
        """Solve single allocation with exact MIP"""
        print(f"\nSolving Single Allocation p-Hub Median (exact)...")
        return solve_single_allocation_phub(self.flows, self.costs, p, alpha)

    def solve_exact_ma(self, p, alpha=0.75, time_limit=600):
        """Solve multiple allocation with exact MIP"""
        print(f"\nSolving Multiple Allocation p-Hub Median (exact)...")
        return solve_multiple_allocation_phub(self.flows, self.costs, p, alpha)

    def solve_heuristic(self, method, p, alpha=0.75):
        """
        Solve with heuristic method

        Args:
            method: 'greedy', 'concentration', 'sa' (simulated annealing)
            p: number of hubs
            alpha: inter-hub discount

        Returns:
            heuristic solution
        """
        print(f"\nSolving with {method} heuristic...")

        if method == 'greedy':
            return greedy_hub_selection(self.flows, self.costs, p, alpha)
        elif method == 'concentration':
            return concentration_heuristic(self.flows, self.costs, p, alpha)
        elif method == 'sa':
            return simulated_annealing_hub(self.flows, self.costs, p, alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(self, p, alpha=0.75,
                       methods=['greedy', 'concentration', 'sa', 'exact_sa']):
        """
        Compare multiple solution methods

        Args:
            p: number of hubs
            alpha: inter-hub discount
            methods: list of methods to compare

        Returns:
            comparison dataframe
        """
        import pandas as pd
        import time

        results = []

        for method in methods:
            start_time = time.time()

            try:
                if method == 'exact_sa':
                    solution = self.solve_exact_sa(p, alpha)
                elif method == 'exact_ma':
                    solution = self.solve_exact_ma(p, alpha)
                else:
                    solution = self.solve_heuristic(method, p, alpha)

                solve_time = time.time() - start_time

                results.append({
                    'Method': method,
                    'Total Cost': solution['total_cost'],
                    'Hubs': str(solution['hubs']),
                    'Time (s)': f"{solve_time:.3f}"
                })

            except Exception as e:
                print(f"  Error with {method}: {e}")

        df = pd.DataFrame(results)

        # Calculate gap from best
        if len(df) > 0:
            best_cost = df['Total Cost'].min()
            df['Gap %'] = ((df['Total Cost'] - best_cost) / best_cost * 100).round(2)

        return df

    def visualize_hub_network(self, solution, title="Hub Network"):
        """
        Visualize hub-and-spoke network

        Args:
            solution: solution dictionary with hubs and allocations
            title: plot title
        """
        import matplotlib.pyplot as plt

        if self.coords is None:
            print("No coordinates available for visualization")
            return

        plt.figure(figsize=(12, 8))

        hubs = solution['hubs']
        allocations = solution['allocations']

        # Plot non-hub nodes (small blue circles)
        for i in range(self.n):
            if i not in hubs:
                plt.plot(self.coords[i, 0], self.coords[i, 1],
                        'o', color='lightblue', markersize=8)

        # Plot connections (spoke lines)
        for node, hub in allocations.items():
            if node not in hubs:
                plt.plot([self.coords[node, 0], self.coords[hub, 0]],
                        [self.coords[node, 1], self.coords[hub, 1]],
                        'k-', alpha=0.2, linewidth=0.5)

        # Plot inter-hub connections (thick red lines)
        for i, hub_i in enumerate(hubs):
            for hub_j in hubs[i+1:]:
                plt.plot([self.coords[hub_i, 0], self.coords[hub_j, 0]],
                        [self.coords[hub_i, 1], self.coords[hub_j, 1]],
                        'r-', alpha=0.6, linewidth=2)

        # Plot hub nodes (large red squares)
        for hub in hubs:
            plt.plot(self.coords[hub, 0], self.coords[hub, 1],
                    's', color='red', markersize=15, label='Hub' if hub == hubs[0] else '')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Complete example
if __name__ == "__main__":
    print("="*70)
    print("HUB LOCATION PROBLEM - COMPREHENSIVE EXAMPLE")
    print("="*70)

    # Generate problem data
    np.random.seed(42)
    n = 15

    # Node coordinates
    coords = np.random.rand(n, 2) * 100

    # Cost matrix (Euclidean distances)
    costs = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            costs[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Flow matrix (random demands)
    flows = np.random.uniform(10, 100, (n, n))
    np.fill_diagonal(flows, 0)

    # Parameters
    p = 3  # Number of hubs
    alpha = 0.75  # Inter-hub discount (25% discount on hub-to-hub travel)

    # Create solver
    solver = HubLocationSolver()
    solver.load_problem(flows, costs, coords)

    # Compare methods
    print(f"\n{'='*70}")
    print(f"COMPARING SOLUTION METHODS (p={p}, α={alpha})")
    print(f"{'='*70}")

    comparison = solver.compare_methods(p, alpha,
                                       methods=['greedy', 'concentration', 'sa'])
    print("\n" + comparison.to_string(index=False))

    # Solve with best method and visualize
    print(f"\n{'='*70}")
    print(f"DETAILED SOLUTION")
    print(f"{'='*70}")

    solution = solver.solve_heuristic('greedy', p, alpha)

    print(f"\nHubs Located: {solution['hubs']}")
    print(f"Total Cost: {solution['total_cost']:,.2f}")

    print(f"\nNode Allocations:")
    for node, hub in solution['allocations'].items():
        hub_indicator = " (HUB)" if node in solution['hubs'] else ""
        distance = costs[node][hub]
        print(f"  Node {node} → Hub {hub}{hub_indicator} (distance={distance:.2f})")

    # Visualize
    solver.visualize_hub_network(solution,
                                title=f"Hub Network: {solution['method']} (p={p}, α={alpha})")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **PuLP**: MIP modeling for hub location
- **Pyomo**: Advanced optimization
- **OR-Tools**: Google optimization tools
- **Gurobi/CPLEX**: Commercial solvers

**Network Analysis:**
- **NetworkX**: Graph algorithms and visualization
- **igraph**: Fast network analysis

**Visualization:**
- **matplotlib**: Basic plotting
- **plotly**: Interactive network visualization
- **folium**: Geographic maps

### Specialized Tools

- **HubLocator**: Academic hub location software
- **SITATION**: Network design software

---

## Common Challenges & Solutions

### Challenge: Problem Size

**Problem:**
- Large networks (100+ nodes)
- Exact methods too slow
- Many O-D pairs

**Solutions:**
- Use heuristics (greedy, concentration)
- Metaheuristics (SA, GA, Tabu Search)
- Decomposition approaches
- Lagrangian relaxation
- Column generation

### Challenge: Multiple Allocation Complexity

**Problem:**
- MA-pHMP has more variables than SA-pHMP
- Harder to solve optimally
- Better solutions but higher computational cost

**Solutions:**
- Use SA-pHMP as approximation
- Restricted multiple allocation (limit connections per node)
- Start with SA solution, refine to MA
- Time limits with commercial solvers

### Challenge: Determining Number of Hubs (p)

**Problem:**
- Don't know optimal p
- Trade-off between hub costs and routing efficiency

**Solutions:**
- Solve for multiple values of p
- Plot cost vs. p curve
- Consider fixed costs explicitly
- Sensitivity analysis
- Use uncapacitated hub location (no fixed p)

### Challenge: Uncertain Demand

**Problem:**
- Flow patterns uncertain or time-varying
- Hub locations long-term strategic decisions

**Solutions:**
- Robust optimization
- Stochastic programming
- Scenario-based analysis
- Flexible/modular hub design
- Multi-period models

---

## Output Format

### Hub Location Solution Report

**Problem Instance:**
- Network: 25 nodes
- Hubs to locate: 4
- Inter-hub discount: α = 0.75 (25% discount)
- Allocation: Single Allocation
- Total O-D flow: 12,450 units

**Optimal Solution:**

| Metric | Value |
|--------|-------|
| Total Cost | $1,247,385 |
| Solution Method | MIP Optimal |
| Hubs Located | {5, 12, 18, 23} |
| Solution Time | 45.3 seconds |
| Status | Optimal |

**Hub Details:**

| Hub ID | Location | Nodes Allocated | Total Inflow | Total Outflow |
|--------|----------|-----------------|--------------|---------------|
| 5 | (34.5, 67.2) | 7 | 3,240 | 3,180 |
| 12 | (78.3, 23.8) | 6 | 2,890 | 2,950 |
| 18 | (12.7, 89.1) | 5 | 2,450 | 2,520 |
| 23 | (56.4, 45.3) | 7 | 3,870 | 3,800 |

**Cost Breakdown:**
- Collection costs (nodes → hubs): $412,450 (33.1%)
- Inter-hub transfer: $298,120 (23.9%)
- Distribution costs (hubs → nodes): $536,815 (43.0%)

**Network Statistics:**
- Average distance to nearest hub: 15.3 km
- Maximum distance to hub: 28.7 km
- Average inter-hub distance: 52.4 km

---

## Questions to Ask

1. What type of network? (freight, passenger, postal, telecom)
2. How many nodes in the network?
3. Do you have O-D flow data?
4. How many hubs should be located? (or should this be optimized?)
5. Single allocation or multiple allocation?
6. What is the inter-hub discount factor (α)?
7. Are there capacity constraints at hubs?
8. Fixed costs to establish hubs?
9. Do you have coordinates or distance/cost matrix?
10. Are flows symmetric or directional?
11. Time-sensitive constraints?
12. Is this strategic (long-term) or tactical planning?

---

## Related Skills

- **facility-location-problem**: General facility location
- **distribution-center-network**: DC network design
- **warehouse-location-optimization**: Warehouse siting
- **network-design**: General supply chain network
- **network-flow-optimization**: Flow optimization
- **set-covering-problem**: Coverage-based location
- **vehicle-routing-problem**: Last-mile routing from hubs
- **optimization-modeling**: MIP formulation techniques
- **multi-objective-optimization**: Multi-criteria hub location


---
name: traveling-salesman-problem
description: When the user wants to solve the Traveling Salesman Problem (TSP), find the shortest route visiting all cities, or optimize tour sequences. Also use when the user mentions "TSP," "shortest tour," "Hamiltonian cycle," "tour optimization," "route sequencing," "optimal visit order," "traveling salesperson," or "minimum distance tour." For vehicle routing with capacities, see vehicle-routing-problem.
---

# Traveling Salesman Problem (TSP)

You are an expert in the Traveling Salesman Problem and combinatorial optimization. Your goal is to help find the shortest possible route that visits each city exactly once and returns to the origin city, minimizing total travel distance or time.

## Initial Assessment

Before solving TSP instances, understand:

1. **Problem Characteristics**
   - How many cities/locations need to be visited?
   - Symmetric TSP (distance i→j = distance j→i) or asymmetric?
   - What metric? (Euclidean distance, travel time, cost)
   - Any special constraints? (time windows → see vrp-time-windows)

2. **Problem Scale**
   - Small (< 20 cities): Exact methods feasible
   - Medium (20-100 cities): Advanced exact or good heuristics
   - Large (100-1000 cities): Metaheuristics required
   - Very large (> 1000 cities): Specialized algorithms

3. **Solution Requirements**
   - Need proven optimal solution?
   - Acceptable optimality gap? (e.g., within 5% of optimal)
   - Time constraint for solution?
   - Single tour or multiple tours needed?

4. **Data Format**
   - Coordinates (lat/lon, x/y)?
   - Distance matrix provided?
   - Need to calculate distances?
   - Real road network or Euclidean?

---

## Mathematical Formulation

### Classic TSP Formulation (MTZ)

**Decision Variables:**
- x_{ij} ∈ {0,1}: 1 if arc (i,j) is in tour, 0 otherwise
- u_i ∈ ℝ: Position of city i in tour (subtour elimination)

**Parameters:**
- c_{ij}: Cost/distance from city i to city j
- n: Number of cities

**Objective Function:**
```
Minimize: Σ_{i=1}^n Σ_{j=1}^n c_{ij} * x_{ij}
```

**Constraints:**
```
1. Each city has exactly one outgoing arc:
   Σ_{j=1}^n x_{ij} = 1,  ∀i

2. Each city has exactly one incoming arc:
   Σ_{i=1}^n x_{ij} = 1,  ∀j

3. Subtour elimination (Miller-Tucker-Zemlin):
   u_i - u_j + n*x_{ij} ≤ n-1,  ∀i,j ∈ {2,...,n}, i≠j

4. Variable bounds:
   x_{ij} ∈ {0,1}
   2 ≤ u_i ≤ n,  ∀i ∈ {2,...,n}
```

### Alternative: DFJ (Dantzig-Fulkerson-Johnson) Formulation

**Subtour Elimination Constraints:**
```
Σ_{i∈S} Σ_{j∈S} x_{ij} ≤ |S| - 1,  ∀S ⊂ V, 2 ≤ |S| ≤ n-1
```

This formulation has exponentially many constraints but produces tighter LP relaxations.

---

## Exact Algorithms

### 1. Branch-and-Bound with Dynamic Programming

```python
import numpy as np
from itertools import combinations

def tsp_dynamic_programming(dist_matrix):
    """
    Held-Karp algorithm for TSP using dynamic programming

    Time complexity: O(n^2 * 2^n)
    Space complexity: O(n * 2^n)

    Optimal for small instances (n ≤ 20)

    Args:
        dist_matrix: n x n distance matrix

    Returns:
        tuple: (optimal_cost, optimal_tour)
    """
    n = len(dist_matrix)

    # C[S][i] = minimum cost path visiting all nodes in S ending at i
    C = {}

    # Initialize: paths of length 1
    for i in range(1, n):
        C[(1 << i, i)] = (dist_matrix[0][i], 0)

    # Iterate over subset sizes
    for subset_size in range(2, n):
        # Generate all subsets of size subset_size
        for subset in combinations(range(1, n), subset_size):
            # Convert to bitmask
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find minimum cost to reach each node in subset
            for i in subset:
                # Previous subset without node i
                prev_bits = bits & ~(1 << i)

                min_cost = float('inf')
                min_prev = None

                # Try all possible previous nodes
                for j in subset:
                    if j == i:
                        continue

                    if (prev_bits, j) in C:
                        cost = C[(prev_bits, j)][0] + dist_matrix[j][i]
                        if cost < min_cost:
                            min_cost = cost
                            min_prev = j

                if min_prev is not None:
                    C[(bits, i)] = (min_cost, min_prev)

    # Find optimal tour
    bits = (1 << n) - 2  # All nodes except 0
    min_cost = float('inf')
    last_node = None

    for i in range(1, n):
        if (bits, i) in C:
            cost = C[(bits, i)][0] + dist_matrix[i][0]
            if cost < min_cost:
                min_cost = cost
                last_node = i

    # Reconstruct tour
    tour = [0]
    bits = (1 << n) - 2

    while last_node is not None:
        tour.append(last_node)
        prev_bits = bits & ~(1 << last_node)

        if (bits, last_node) in C:
            last_node = C[(bits, last_node)][1]
            bits = prev_bits
        else:
            break

    tour.append(0)

    return min_cost, tour


# Example usage
if __name__ == "__main__":
    # Small 5-city example
    dist_matrix = np.array([
        [0, 10, 15, 20, 25],
        [10, 0, 35, 25, 30],
        [15, 35, 0, 30, 20],
        [20, 25, 30, 0, 15],
        [25, 30, 20, 15, 0]
    ])

    optimal_cost, optimal_tour = tsp_dynamic_programming(dist_matrix)
    print(f"Optimal Cost: {optimal_cost}")
    print(f"Optimal Tour: {optimal_tour}")
```

### 2. MIP Formulation with PuLP

```python
from pulp import *
import numpy as np

def tsp_mip_mtz(dist_matrix, city_names=None):
    """
    TSP using Miller-Tucker-Zemlin formulation

    Suitable for medium-sized instances (up to 50-100 cities)

    Args:
        dist_matrix: n x n distance matrix
        city_names: optional list of city names

    Returns:
        dict with optimal_cost, tour, and solve_time
    """
    n = len(dist_matrix)

    if city_names is None:
        city_names = [f"City_{i}" for i in range(n)]

    # Create problem
    prob = LpProblem("TSP_MTZ", LpMinimize)

    # Decision variables
    # x[i,j] = 1 if arc from city i to city j is in tour
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i,j] = LpVariable(f"x_{i}_{j}", cat='Binary')

    # u[i] = position of city i in tour (for subtour elimination)
    u = {}
    for i in range(1, n):
        u[i] = LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat='Continuous')

    # Objective: Minimize total distance
    prob += lpSum([dist_matrix[i][j] * x[i,j]
                   for i in range(n) for j in range(n) if i != j]), \
            "Total_Distance"

    # Constraints

    # 1. Each city has exactly one outgoing arc
    for i in range(n):
        prob += lpSum([x[i,j] for j in range(n) if j != i]) == 1, \
                f"Out_{i}"

    # 2. Each city has exactly one incoming arc
    for j in range(n):
        prob += lpSum([x[i,j] for i in range(n) if i != j]) == 1, \
                f"In_{j}"

    # 3. Miller-Tucker-Zemlin subtour elimination
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n*x[i,j] <= n-1, \
                        f"MTZ_{i}_{j}"

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] == 'Optimal':
        tour = [0]
        current_city = 0

        for _ in range(n-1):
            for j in range(n):
                if j != current_city and x[current_city,j].varValue > 0.5:
                    tour.append(j)
                    current_city = j
                    break

        tour.append(0)  # Return to start

        return {
            'status': 'Optimal',
            'optimal_cost': value(prob.objective),
            'tour': tour,
            'tour_names': [city_names[i] for i in tour],
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'optimal_cost': None,
            'tour': None,
            'solve_time': solve_time
        }


# Example usage
dist_matrix = np.array([
    [0, 29, 20, 21, 16, 31],
    [29, 0, 15, 29, 28, 40],
    [20, 15, 0, 15, 14, 25],
    [21, 29, 15, 0, 4, 12],
    [16, 28, 14, 4, 0, 16],
    [31, 40, 25, 12, 16, 0]
])

city_names = ['Depot', 'Customer_A', 'Customer_B',
              'Customer_C', 'Customer_D', 'Customer_E']

result = tsp_mip_mtz(dist_matrix, city_names)
print(f"\nStatus: {result['status']}")
print(f"Optimal Cost: {result['optimal_cost']:.2f}")
print(f"Optimal Tour: {result['tour_names']}")
print(f"Solve Time: {result['solve_time']:.2f} seconds")
```

---

## Constructive Heuristics

### 1. Nearest Neighbor

```python
import numpy as np

def nearest_neighbor_tsp(dist_matrix, start_city=0):
    """
    Nearest Neighbor heuristic for TSP

    Time complexity: O(n^2)
    Quality: Typically within 25% of optimal

    Args:
        dist_matrix: n x n distance matrix
        start_city: starting city (default 0)

    Returns:
        tuple: (total_cost, tour)
    """
    n = len(dist_matrix)
    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)

    current_city = start_city
    total_cost = 0

    while unvisited:
        # Find nearest unvisited city
        nearest_city = min(unvisited,
                          key=lambda city: dist_matrix[current_city][city])

        total_cost += dist_matrix[current_city][nearest_city]
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city

    # Return to start
    total_cost += dist_matrix[current_city][start_city]
    tour.append(start_city)

    return total_cost, tour


def multi_start_nearest_neighbor(dist_matrix, num_starts=None):
    """
    Run Nearest Neighbor from multiple starting cities
    and return best solution

    Args:
        dist_matrix: n x n distance matrix
        num_starts: number of different starts (default: all cities)

    Returns:
        tuple: (best_cost, best_tour)
    """
    n = len(dist_matrix)

    if num_starts is None:
        num_starts = n

    best_cost = float('inf')
    best_tour = None

    for start in range(min(num_starts, n)):
        cost, tour = nearest_neighbor_tsp(dist_matrix, start)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_cost, best_tour
```

### 2. Cheapest Insertion

```python
def cheapest_insertion_tsp(dist_matrix):
    """
    Cheapest Insertion heuristic for TSP

    Time complexity: O(n^3)
    Quality: Generally better than Nearest Neighbor

    Args:
        dist_matrix: n x n distance matrix

    Returns:
        tuple: (total_cost, tour)
    """
    n = len(dist_matrix)

    # Start with smallest edge
    min_dist = float('inf')
    best_pair = (0, 1)

    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i][j] < min_dist:
                min_dist = dist_matrix[i][j]
                best_pair = (i, j)

    # Initial tour
    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)

    # Insert remaining cities
    while unvisited:
        best_insertion = None
        best_cost_increase = float('inf')
        best_position = None

        # Try inserting each unvisited city
        for city in unvisited:
            # Try inserting between each pair of adjacent cities
            for i in range(len(tour)):
                j = (i + 1) % len(tour)

                # Cost increase of inserting city between tour[i] and tour[j]
                cost_increase = (dist_matrix[tour[i]][city] +
                               dist_matrix[city][tour[j]] -
                               dist_matrix[tour[i]][tour[j]])

                if cost_increase < best_cost_increase:
                    best_cost_increase = cost_increase
                    best_insertion = city
                    best_position = j

        # Insert best city
        tour.insert(best_position, best_insertion)
        unvisited.remove(best_insertion)

    # Calculate total cost
    total_cost = sum(dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
                    for i in range(len(tour)))

    tour.append(tour[0])  # Close tour

    return total_cost, tour


def farthest_insertion_tsp(dist_matrix):
    """
    Farthest Insertion heuristic (variant)

    Similar to cheapest insertion but selects farthest city
    from tour at each step
    """
    n = len(dist_matrix)

    # Start with two farthest cities
    max_dist = 0
    best_pair = (0, 1)

    for i in range(n):
        for j in range(i+1, n):
            if dist_matrix[i][j] > max_dist:
                max_dist = dist_matrix[i][j]
                best_pair = (i, j)

    tour = list(best_pair)
    unvisited = set(range(n)) - set(tour)

    while unvisited:
        # Find farthest city from current tour
        farthest_city = None
        max_min_dist = 0

        for city in unvisited:
            min_dist_to_tour = min(dist_matrix[city][tour_city]
                                  for tour_city in tour)
            if min_dist_to_tour > max_min_dist:
                max_min_dist = min_dist_to_tour
                farthest_city = city

        # Find best insertion position
        best_position = 0
        best_cost_increase = float('inf')

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost_increase = (dist_matrix[tour[i]][farthest_city] +
                           dist_matrix[farthest_city][tour[j]] -
                           dist_matrix[tour[i]][tour[j]])

            if cost_increase < best_cost_increase:
                best_cost_increase = cost_increase
                best_position = j

        tour.insert(best_position, farthest_city)
        unvisited.remove(farthest_city)

    total_cost = sum(dist_matrix[tour[i]][tour[(i+1)%len(tour)]]
                    for i in range(len(tour)))
    tour.append(tour[0])

    return total_cost, tour
```

---

## Improvement Heuristics

### 1. 2-Opt Local Search

```python
def two_opt(tour, dist_matrix):
    """
    2-opt improvement heuristic

    Iteratively removes two edges and reconnects the tour
    in a different way if it improves the solution

    Args:
        tour: initial tour (list of city indices)
        dist_matrix: n x n distance matrix

    Returns:
        tuple: (improved_cost, improved_tour)
    """
    n = len(tour) - 1  # Exclude duplicate last city
    improved = True
    best_tour = tour[:-1]  # Remove last city (duplicate of first)

    while improved:
        improved = False

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1])
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1])

                current_cost = (dist_matrix[best_tour[i-1]][best_tour[i]] +
                               dist_matrix[best_tour[j]][best_tour[(j+1)%n]])

                new_cost = (dist_matrix[best_tour[i-1]][best_tour[j]] +
                           dist_matrix[best_tour[i]][best_tour[(j+1)%n]])

                if new_cost < current_cost:
                    # Reverse the segment between i and j
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    improved = True
                    break

            if improved:
                break

    # Calculate total cost
    total_cost = sum(dist_matrix[best_tour[i]][best_tour[(i+1)%n]]
                    for i in range(n))
    total_cost += dist_matrix[best_tour[n-1]][best_tour[0]]

    best_tour.append(best_tour[0])  # Close tour

    return total_cost, best_tour


def two_opt_optimized(tour, dist_matrix, max_iterations=1000):
    """
    Optimized 2-opt with early stopping
    """
    n = len(tour) - 1
    best_tour = tour[:-1]

    def calculate_tour_cost(t):
        return sum(dist_matrix[t[i]][t[(i+1)%len(t)]] for i in range(len(t)))

    best_cost = calculate_tour_cost(best_tour)
    iterations = 0
    no_improvement_count = 0

    while iterations < max_iterations and no_improvement_count < 50:
        iterations += 1
        improved = False

        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate delta (change in cost)
                delta = (dist_matrix[best_tour[i-1]][best_tour[j]] +
                        dist_matrix[best_tour[i]][best_tour[(j+1)%n]] -
                        dist_matrix[best_tour[i-1]][best_tour[i]] -
                        dist_matrix[best_tour[j]][best_tour[(j+1)%n]])

                if delta < -1e-10:  # Improvement found
                    best_tour[i:j+1] = reversed(best_tour[i:j+1])
                    best_cost += delta
                    improved = True
                    no_improvement_count = 0
                    break

            if improved:
                break

        if not improved:
            no_improvement_count += 1

    best_tour.append(best_tour[0])
    return best_cost, best_tour
```

### 2. 3-Opt Local Search

```python
def three_opt(tour, dist_matrix, max_iterations=100):
    """
    3-opt improvement heuristic

    More powerful than 2-opt but slower
    Considers removing 3 edges and reconnecting

    Args:
        tour: initial tour
        dist_matrix: n x n distance matrix
        max_iterations: maximum iterations

    Returns:
        tuple: (improved_cost, improved_tour)
    """
    n = len(tour) - 1
    best_tour = tour[:-1]

    def calculate_cost(t):
        return sum(dist_matrix[t[i]][t[(i+1)%len(t)]] for i in range(len(t)))

    best_cost = calculate_cost(best_tour)

    for iteration in range(max_iterations):
        improved = False

        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                for k in range(j + 2, n):
                    # Try all possible 3-opt reconnections
                    # There are 8 ways to reconnect 3 segments

                    # Current tour segments: A-B-C where
                    # A = tour[0:i+1]
                    # B = tour[i+1:j+1]
                    # C = tour[j+1:k+1]
                    # D = tour[k+1:]

                    # Try different reconnections
                    segments = [
                        best_tour[:i+1],
                        best_tour[i+1:j+1],
                        best_tour[j+1:k+1],
                        best_tour[k+1:]
                    ]

                    # Case 1: A + reverse(B) + C + D
                    new_tour = segments[0] + segments[1][::-1] + segments[2] + segments[3]
                    new_cost = calculate_cost(new_tour)

                    if new_cost < best_cost:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    best_tour.append(best_tour[0])
    return best_cost, best_tour
```

### 3. Or-Opt

```python
def or_opt(tour, dist_matrix):
    """
    Or-opt improvement heuristic

    Relocates sequences of 1, 2, or 3 consecutive cities

    Args:
        tour: initial tour
        dist_matrix: n x n distance matrix

    Returns:
        tuple: (improved_cost, improved_tour)
    """
    n = len(tour) - 1
    best_tour = tour[:-1]
    improved = True

    def calculate_cost(t):
        return sum(dist_matrix[t[i]][t[(i+1)%len(t)]] for i in range(len(t)))

    best_cost = calculate_cost(best_tour)

    while improved:
        improved = False

        # Try relocating sequences of length 1, 2, and 3
        for seq_length in [1, 2, 3]:
            if seq_length >= n:
                continue

            for i in range(n):
                if i + seq_length > n:
                    continue

                # Extract sequence
                sequence = best_tour[i:i+seq_length]

                # Try inserting at all other positions
                for j in range(n):
                    if j >= i and j < i + seq_length:
                        continue

                    # Create new tour with sequence relocated
                    new_tour = best_tour[:i] + best_tour[i+seq_length:]
                    new_tour = new_tour[:j] + sequence + new_tour[j:]

                    new_cost = calculate_cost(new_tour)

                    if new_cost < best_cost - 1e-10:
                        best_tour = new_tour
                        best_cost = new_cost
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

    best_tour.append(best_tour[0])
    return best_cost, best_tour
```

---

## Metaheuristics

### 1. Simulated Annealing

```python
import random
import math

def simulated_annealing_tsp(dist_matrix, initial_temp=1000,
                           cooling_rate=0.995, max_iterations=10000):
    """
    Simulated Annealing for TSP

    Probabilistically accepts worse solutions to escape local optima

    Args:
        dist_matrix: n x n distance matrix
        initial_temp: starting temperature
        cooling_rate: temperature reduction factor
        max_iterations: maximum iterations

    Returns:
        tuple: (best_cost, best_tour)
    """
    n = len(dist_matrix)

    # Generate initial solution
    current_tour = list(range(n))
    random.shuffle(current_tour)
    current_tour.append(current_tour[0])

    def calculate_cost(tour):
        return sum(dist_matrix[tour[i]][tour[i+1]]
                  for i in range(len(tour)-1))

    current_cost = calculate_cost(current_tour)
    best_tour = current_tour.copy()
    best_cost = current_cost

    temperature = initial_temp

    for iteration in range(max_iterations):
        # Generate neighbor solution (2-opt move)
        i = random.randint(1, n - 2)
        j = random.randint(i + 1, n - 1)

        new_tour = current_tour.copy()
        new_tour[i:j+1] = reversed(new_tour[i:j+1])

        new_cost = calculate_cost(new_tour)
        delta = new_cost - current_cost

        # Accept or reject new solution
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_tour = new_tour
            current_cost = new_cost

            # Update best solution
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost

        # Cool down
        temperature *= cooling_rate

        if temperature < 0.01:
            break

    return best_cost, best_tour
```

### 2. Genetic Algorithm

```python
def genetic_algorithm_tsp(dist_matrix, population_size=100,
                         generations=500, mutation_rate=0.01):
    """
    Genetic Algorithm for TSP

    Uses order crossover (OX) and swap mutation

    Args:
        dist_matrix: n x n distance matrix
        population_size: number of individuals
        generations: number of generations
        mutation_rate: probability of mutation

    Returns:
        tuple: (best_cost, best_tour)
    """
    n = len(dist_matrix)

    def calculate_fitness(tour):
        cost = sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
        return 1.0 / (1.0 + cost)  # Higher fitness = better tour

    def create_individual():
        tour = list(range(n))
        random.shuffle(tour)
        return tour

    def order_crossover(parent1, parent2):
        """Order crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))

        child = [-1] * size
        child[start:end] = parent1[start:end]

        # Fill remaining positions from parent2
        pos = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                if pos >= size:
                    pos = 0
                child[pos] = city
                pos += 1

        return child

    def mutate(tour):
        """Swap mutation"""
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour

    def tournament_selection(population, fitnesses, k=3):
        """Tournament selection"""
        selected = random.sample(list(zip(population, fitnesses)), k)
        return max(selected, key=lambda x: x[1])[0]

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_tour = None
    best_cost = float('inf')

    for generation in range(generations):
        # Calculate fitness
        fitnesses = [calculate_fitness(ind) for ind in population]

        # Track best
        for tour, fitness in zip(population, fitnesses):
            cost = sum(dist_matrix[tour[i]][tour[(i+1)%n]] for i in range(n))
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()

        # Create next generation
        new_population = []

        # Elitism: keep best individuals
        elite_count = int(0.1 * population_size)
        elite_indices = sorted(range(len(fitnesses)),
                              key=lambda i: fitnesses[i],
                              reverse=True)[:elite_count]
        new_population = [population[i].copy() for i in elite_indices]

        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            child = order_crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    best_tour.append(best_tour[0])
    return best_cost, best_tour
```

### 3. Ant Colony Optimization

```python
def ant_colony_optimization_tsp(dist_matrix, n_ants=20, n_iterations=100,
                                alpha=1.0, beta=2.0, evaporation=0.5, Q=100):
    """
    Ant Colony Optimization for TSP

    Args:
        dist_matrix: n x n distance matrix
        n_ants: number of ants
        n_iterations: number of iterations
        alpha: pheromone importance
        beta: distance importance
        evaporation: pheromone evaporation rate
        Q: pheromone deposit factor

    Returns:
        tuple: (best_cost, best_tour)
    """
    n = len(dist_matrix)

    # Initialize pheromone matrix
    pheromone = np.ones((n, n)) / n

    best_tour = None
    best_cost = float('inf')

    for iteration in range(n_iterations):
        all_tours = []
        all_costs = []

        # Each ant constructs a tour
        for ant in range(n_ants):
            tour = [0]  # Start from city 0
            unvisited = set(range(1, n))

            current_city = 0

            while unvisited:
                # Calculate probabilities for next city
                probabilities = []

                for city in unvisited:
                    tau = pheromone[current_city][city] ** alpha
                    eta = (1.0 / dist_matrix[current_city][city]) ** beta
                    probabilities.append(tau * eta)

                # Normalize probabilities
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]

                # Select next city
                next_city = np.random.choice(list(unvisited), p=probabilities)

                tour.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city

            tour.append(0)  # Return to start

            # Calculate tour cost
            cost = sum(dist_matrix[tour[i]][tour[i+1]]
                      for i in range(len(tour)-1))

            all_tours.append(tour)
            all_costs.append(cost)

            # Update best solution
            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()

        # Evaporate pheromone
        pheromone *= (1 - evaporation)

        # Deposit pheromone
        for tour, cost in zip(all_tours, all_costs):
            deposit = Q / cost

            for i in range(len(tour) - 1):
                city1, city2 = tour[i], tour[i+1]
                pheromone[city1][city2] += deposit
                pheromone[city2][city1] += deposit

    return best_cost, best_tour
```

---

## Using OR-Tools (Google Optimization)

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_tsp_ortools(dist_matrix, time_limit_seconds=30):
    """
    Solve TSP using Google OR-Tools

    Very efficient for practical instances

    Args:
        dist_matrix: n x n distance matrix
        time_limit_seconds: time limit for solver

    Returns:
        dict with solution details
    """
    n = len(dist_matrix)

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Create distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        tour = []
        index = routing.Start(0)

        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        tour.append(manager.IndexToNode(index))

        return {
            'status': 'Optimal' if solution.status == 1 else 'Feasible',
            'optimal_cost': solution.ObjectiveValue(),
            'tour': tour,
            'computation_time': routing.solver().WallTime() / 1000.0
        }
    else:
        return {
            'status': 'No solution found',
            'optimal_cost': None,
            'tour': None
        }


# Example: Solve TSP with coordinates
def solve_tsp_from_coordinates(coordinates, time_limit=30):
    """
    Solve TSP given (x, y) coordinates

    Args:
        coordinates: list of (x, y) tuples
        time_limit: time limit in seconds

    Returns:
        solution dictionary
    """
    import math

    n = len(coordinates)

    # Calculate Euclidean distance matrix
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                dist_matrix[i][j] = math.sqrt(dx*dx + dy*dy)

    return solve_tsp_ortools(dist_matrix, time_limit)


# Example usage
coordinates = [
    (0, 0),    # Depot
    (2, 4),    # City 1
    (5, 1),    # City 2
    (8, 3),    # City 3
    (6, 7),    # City 4
    (3, 6),    # City 5
    (7, 9),    # City 6
    (1, 8)     # City 7
]

result = solve_tsp_from_coordinates(coordinates)
print(f"Status: {result['status']}")
print(f"Optimal Cost: {result['optimal_cost']:.2f}")
print(f"Optimal Tour: {result['tour']}")
print(f"Computation Time: {result['computation_time']:.2f} seconds")
```

---

## Complete Solution Framework

```python
class TSPSolver:
    """
    Complete TSP solver with multiple algorithms
    """

    def __init__(self, dist_matrix, city_names=None):
        self.dist_matrix = np.array(dist_matrix)
        self.n = len(dist_matrix)

        if city_names is None:
            self.city_names = [f"City_{i}" for i in range(self.n)]
        else:
            self.city_names = city_names

    def solve(self, method='ortools', **kwargs):
        """
        Solve TSP using specified method

        Methods:
        - 'ortools': Google OR-Tools (recommended)
        - 'dynamic_programming': Exact (small instances only)
        - 'nearest_neighbor': Fast heuristic
        - 'cheapest_insertion': Good heuristic
        - '2opt': Local search improvement
        - 'simulated_annealing': Metaheuristic
        - 'genetic': Genetic algorithm
        - 'aco': Ant colony optimization
        """

        if method == 'ortools':
            result = solve_tsp_ortools(self.dist_matrix,
                                      kwargs.get('time_limit', 30))

        elif method == 'dynamic_programming':
            if self.n > 20:
                raise ValueError("Dynamic programming only for n ≤ 20")
            cost, tour = tsp_dynamic_programming(self.dist_matrix)
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Optimal'}

        elif method == 'nearest_neighbor':
            cost, tour = multi_start_nearest_neighbor(
                self.dist_matrix,
                kwargs.get('num_starts', self.n))
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Heuristic'}

        elif method == 'cheapest_insertion':
            cost, tour = cheapest_insertion_tsp(self.dist_matrix)
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Heuristic'}

        elif method == '2opt':
            # Need initial solution
            _, initial_tour = nearest_neighbor_tsp(self.dist_matrix)
            cost, tour = two_opt_optimized(initial_tour, self.dist_matrix)
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Local Search'}

        elif method == 'simulated_annealing':
            cost, tour = simulated_annealing_tsp(
                self.dist_matrix,
                kwargs.get('initial_temp', 1000),
                kwargs.get('cooling_rate', 0.995),
                kwargs.get('max_iterations', 10000))
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Metaheuristic'}

        elif method == 'genetic':
            cost, tour = genetic_algorithm_tsp(
                self.dist_matrix,
                kwargs.get('population_size', 100),
                kwargs.get('generations', 500),
                kwargs.get('mutation_rate', 0.01))
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Metaheuristic'}

        elif method == 'aco':
            cost, tour = ant_colony_optimization_tsp(
                self.dist_matrix,
                kwargs.get('n_ants', 20),
                kwargs.get('n_iterations', 100))
            result = {'optimal_cost': cost, 'tour': tour, 'status': 'Metaheuristic'}

        else:
            raise ValueError(f"Unknown method: {method}")

        # Add city names to result
        if result['tour']:
            result['tour_names'] = [self.city_names[i] for i in result['tour']]

        return result

    def compare_methods(self, methods=['nearest_neighbor', 'cheapest_insertion',
                                      '2opt', 'ortools']):
        """
        Compare multiple solution methods
        """
        import time

        results = []

        for method in methods:
            print(f"Solving with {method}...")
            start_time = time.time()

            try:
                result = self.solve(method)
                solve_time = time.time() - start_time

                results.append({
                    'method': method,
                    'cost': result['optimal_cost'],
                    'tour': result['tour'],
                    'status': result['status'],
                    'time': solve_time
                })
            except Exception as e:
                print(f"  Error: {e}")

        # Create comparison dataframe
        import pandas as pd
        df = pd.DataFrame([{
            'Method': r['method'],
            'Cost': r['cost'],
            'Status': r['status'],
            'Time (s)': f"{r['time']:.3f}"
        } for r in results])

        # Add % from best
        best_cost = df['Cost'].min()
        df['Gap %'] = ((df['Cost'] - best_cost) / best_cost * 100).round(2)

        return df


# Example usage
if __name__ == "__main__":
    # Example distance matrix
    dist_matrix = [
        [0, 29, 20, 21, 16, 31, 100],
        [29, 0, 15, 29, 28, 40, 72],
        [20, 15, 0, 15, 14, 25, 81],
        [21, 29, 15, 0, 4, 12, 92],
        [16, 28, 14, 4, 0, 16, 94],
        [31, 40, 25, 12, 16, 0, 95],
        [100, 72, 81, 92, 94, 95, 0]
    ]

    city_names = ['Depot', 'Customer A', 'Customer B', 'Customer C',
                  'Customer D', 'Customer E', 'Customer F']

    solver = TSPSolver(dist_matrix, city_names)

    # Solve with OR-Tools
    result = solver.solve('ortools', time_limit=10)
    print(f"\nOR-Tools Solution:")
    print(f"Cost: {result['optimal_cost']:.2f}")
    print(f"Tour: {' → '.join(result['tour_names'])}")

    # Compare methods
    print("\n" + "="*60)
    print("Comparing Methods:")
    print("="*60)
    comparison = solver.compare_methods()
    print(comparison.to_string(index=False))
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **OR-Tools (Google)**: Industrial-strength routing solver
- **PuLP**: MIP modeling
- **Pyomo**: Advanced optimization modeling
- **NetworkX**: Graph algorithms
- **python-tsp**: Specialized TSP library

**Heuristics:**
- **scikit-opt**: Genetic algorithms, simulated annealing
- **pymoo**: Multi-objective optimization

**Visualization:**
- **matplotlib**: Basic plotting
- **plotly**: Interactive visualizations
- **folium**: Map-based visualization

### Commercial Solvers

- **Gurobi**: State-of-art MIP solver
- **CPLEX**: IBM optimization solver
- **Concorde**: Best exact TSP solver (specialized)

### Online Tools

- **TSPLIB**: Standard benchmark instances
- **Concorde TSP Solver**: Free online solver
- **NEOS Server**: Remote optimization server

---

## Common Challenges & Solutions

### Challenge: Problem Too Large for Exact Methods

**Problem:**
- 100+ cities makes exact methods impractical
- Need good solutions quickly

**Solutions:**
- Use OR-Tools with time limits
- Multi-start nearest neighbor + 2-opt
- Metaheuristics (SA, GA, ACO)
- Problem decomposition (cluster first, route second)

### Challenge: Asymmetric TSP

**Problem:**
- Distance i→j ≠ distance j→i (e.g., one-way streets)
- Different formulation needed

**Solutions:**
- Modify MTZ formulation for asymmetric case
- OR-Tools handles asymmetric naturally
- Heuristics adapt easily

### Challenge: Real-World Constraints

**Problem:**
- Time windows, capacities, multiple vehicles
- Pure TSP insufficient

**Solutions:**
- See **vrp-time-windows** for time constraints
- See **vehicle-routing-problem** for capacities
- See **multi-depot-vrp** for multiple depots

### Challenge: Dynamic TSP

**Problem:**
- New customers arrive during execution
- Need to reoptimize

**Solutions:**
- Use fast heuristics (nearest neighbor, insertion)
- Implement re-optimization triggers
- Keep partial tours feasible

---

## Output Format

### TSP Solution Report

**Problem Summary:**
- Number of cities: 25
- Distance metric: Euclidean
- Solution method: OR-Tools with Guided Local Search
- Computation time: 2.3 seconds

**Solution Quality:**

| Metric | Value |
|--------|-------|
| Total Distance | 427.85 km |
| Number of Cities | 25 |
| Solution Status | Optimal |
| Optimality Gap | 0% |

**Optimal Tour:**
```
Depot → City_A → City_F → City_C → ... → City_Z → Depot
```

**Tour Visualization:**
[Map or graph showing tour]

**Statistics:**
- Average distance between consecutive cities: 17.1 km
- Longest leg: City_A → City_B (45.2 km)
- Shortest leg: City_F → City_G (3.8 km)

---

## Questions to Ask

If you need more context:
1. How many cities/locations need to be visited?
2. Do you have coordinates or a distance matrix?
3. Is distance symmetric (i→j = j→i)?
4. Do you need the proven optimal solution or is a good solution acceptable?
5. What's the acceptable computation time?
6. Are there any additional constraints (time windows, capacities)?
7. Is this a one-time solve or recurring problem?
8. Do you have access to commercial solvers (Gurobi, CPLEX)?

---

## Related Skills

- **vehicle-routing-problem**: For multiple vehicles with capacities
- **vrp-time-windows**: For TSP with time constraints
- **pickup-delivery-problem**: For pickup and delivery pairs
- **route-optimization**: For practical routing applications
- **network-flow-optimization**: For underlying graph theory
- **metaheuristic-optimization**: For advanced solution methods
- **optimization-modeling**: For general MIP formulation


---
name: supply-chain-automation
description: When the user wants to automate supply chain processes, build automation systems, or implement robotic process automation (RPA). Also use when the user mentions "process automation," "RPA," "workflow automation," "task automation," "supply chain bots," "automated replenishment," "auto-ordering," "automated decision-making," or "process orchestration." For analytics dashboards, see supply-chain-analytics. For optimization, see optimization-modeling.
---

# Supply Chain Automation

You are an expert in supply chain automation and process optimization. Your goal is to help design, implement, and manage automated systems that reduce manual work, improve efficiency, eliminate errors, and enable faster, data-driven decision-making across supply chain operations.

## Initial Assessment

Before implementing automation, understand:

1. **Process Analysis**
   - What processes need automation? (ordering, replenishment, allocation, invoicing)
   - Current pain points? (manual data entry, errors, delays, inconsistency)
   - Process volume and frequency? (1000 orders/day, hourly replenishment)
   - Process complexity? (simple rules vs. complex logic)

2. **Current State**
   - How is the process done today? (manual, semi-automated, spreadsheets)
   - Time spent on manual tasks? (hours per week)
   - Error rates? (% of errors, cost of errors)
   - Systems involved? (ERP, WMS, TMS, spreadsheets)

3. **Business Value**
   - Expected benefits? (time savings, cost reduction, accuracy improvement)
   - ROI targets?
   - Criticality to business? (mission-critical vs. nice-to-have)
   - Compliance or audit requirements?

4. **Technical Environment**
   - System access? (APIs available, database access, screen scraping needed)
   - IT support and governance?
   - Security and data privacy requirements?
   - Infrastructure? (cloud, on-premise, hybrid)

---

## Automation Framework

### Automation Maturity Levels

**Level 0: Manual**
- All tasks done manually
- Spreadsheet-based processes
- Email and phone communication
- No integration

**Level 1: Task Automation**
- Individual tasks automated
- Basic scripts and macros
- Simple data transfers
- Minimal integration

**Level 2: Process Automation**
- End-to-end processes automated
- Workflow orchestration
- Multi-system integration
- Rule-based decision logic

**Level 3: Intelligent Automation**
- ML-powered decision-making
- Predictive automation
- Exception handling
- Adaptive learning

**Level 4: Autonomous**
- Self-optimizing systems
- Real-time adaptation
- Closed-loop control
- Full autonomy

### Automation Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│              User Interface Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Web Apps   │  │    Mobile    │  │  Dashboards  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│           Intelligent Automation Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ML Models   │  │     Rules    │  │  Workflows   │ │
│  │  Prediction  │  │    Engine    │  │ Orchestration│ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│             Process Automation Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     RPA      │  │  API Gateway │  │  Schedulers  │ │
│  │    Bots      │  │ Integration  │  │   Triggers   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│               Data Integration Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  ETL/ELT     │  │   Data Lake  │  │   Message    │ │
│  │  Pipelines   │  │  Warehouse   │  │    Queue     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│               System Integration Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     ERP      │  │     WMS      │  │     TMS      │ │
│  │   SAP/Oracle │  │  Manhattan   │  │  Blue Yonder │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Automated Replenishment

### Automated Inventory Replenishment System

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AutomatedReplenishment:
    """
    Automated inventory replenishment system

    Monitors inventory levels, calculates reorder needs,
    generates purchase orders automatically
    """

    def __init__(self, config):
        """
        config: dict with system parameters
        - reorder_point_method: 'static', 'dynamic', 'ml'
        - order_approval_threshold: $ value requiring approval
        - supplier_api_endpoints: dict of supplier APIs
        - email_config: SMTP settings for notifications
        """
        self.config = config
        self.inventory_data = None
        self.demand_forecast = None
        self.supplier_info = None
        self.orders_generated = []

    def load_inventory_data(self, data_source):
        """
        Load current inventory levels

        data_source: database connection, API, or file path
        """

        # Example: Load from database
        # In production, connect to ERP/WMS system
        self.inventory_data = pd.DataFrame({
            'sku': ['SKU_001', 'SKU_002', 'SKU_003'],
            'on_hand': [150, 80, 250],
            'on_order': [100, 0, 50],
            'reorder_point': [200, 150, 300],
            'order_quantity': [500, 300, 400],
            'supplier': ['Supplier_A', 'Supplier_B', 'Supplier_A'],
            'lead_time_days': [14, 10, 14],
            'unit_cost': [25.00, 45.00, 15.00],
            'last_order_date': [datetime.now() - timedelta(days=20),
                               datetime.now() - timedelta(days=30),
                               datetime.now() - timedelta(days=15)]
        })

    def calculate_reorder_needs(self):
        """
        Determine which SKUs need reordering

        Checks inventory position (on_hand + on_order) against reorder point
        """

        self.inventory_data['inventory_position'] = (
            self.inventory_data['on_hand'] +
            self.inventory_data['on_order']
        )

        self.inventory_data['needs_reorder'] = (
            self.inventory_data['inventory_position'] <
            self.inventory_data['reorder_point']
        )

        # Calculate order quantity
        # Can use EOQ, fixed quantity, or dynamic calculation
        self.inventory_data['recommended_order_qty'] = np.where(
            self.inventory_data['needs_reorder'],
            self.inventory_data['order_quantity'],
            0
        )

        reorder_items = self.inventory_data[
            self.inventory_data['needs_reorder']
        ].copy()

        return reorder_items

    def optimize_order_quantities(self, reorder_items):
        """
        Optimize order quantities considering multiple factors

        - MOQ (minimum order quantity)
        - Price breaks / volume discounts
        - Container/pallet fill optimization
        - Lead time variability
        """

        for idx, row in reorder_items.iterrows():
            sku = row['sku']

            # Get demand forecast
            forecast = self.get_demand_forecast(sku)

            # Calculate optimal order quantity
            # Simplified EOQ calculation
            annual_demand = forecast['annual_demand']
            order_cost = 100  # $ per order
            holding_cost_rate = 0.25  # 25% of unit cost

            eoq = np.sqrt(
                (2 * annual_demand * order_cost) /
                (row['unit_cost'] * holding_cost_rate)
            )

            # Adjust for MOQ, pack size, etc.
            optimal_qty = max(eoq, row['order_quantity'])

            # Round to case pack
            case_pack = 50  # units per case
            optimal_qty = np.ceil(optimal_qty / case_pack) * case_pack

            reorder_items.at[idx, 'recommended_order_qty'] = optimal_qty

        return reorder_items

    def get_demand_forecast(self, sku):
        """
        Get demand forecast for SKU

        Could integrate with ML forecasting system
        """

        # Placeholder: return dummy forecast
        return {
            'annual_demand': 10000,
            'next_30_days': 850,
            'forecast_error': 0.15
        }

    def generate_purchase_orders(self, reorder_items):
        """
        Generate purchase orders for reorder items

        Groups by supplier for consolidation
        """

        pos_created = []

        # Group by supplier
        for supplier, items in reorder_items.groupby('supplier'):
            po_number = self.generate_po_number()

            po = {
                'po_number': po_number,
                'supplier': supplier,
                'order_date': datetime.now(),
                'items': [],
                'total_value': 0,
                'status': 'draft'
            }

            for idx, item in items.iterrows():
                line_item = {
                    'sku': item['sku'],
                    'quantity': item['recommended_order_qty'],
                    'unit_cost': item['unit_cost'],
                    'line_total': item['recommended_order_qty'] * item['unit_cost']
                }

                po['items'].append(line_item)
                po['total_value'] += line_item['line_total']

            # Check if approval needed
            if po['total_value'] > self.config['order_approval_threshold']:
                po['status'] = 'pending_approval'
                self.send_approval_request(po)
            else:
                po['status'] = 'approved'
                self.submit_po_to_supplier(po)

            pos_created.append(po)
            self.orders_generated.append(po)

        return pos_created

    def generate_po_number(self):
        """Generate unique PO number"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"PO-{timestamp}"

    def submit_po_to_supplier(self, po):
        """
        Submit approved PO to supplier

        Methods:
        - API integration (preferred)
        - EDI
        - Email
        - Portal upload
        """

        supplier = po['supplier']

        # Check if supplier has API integration
        if supplier in self.config.get('supplier_api_endpoints', {}):
            self.submit_via_api(po, supplier)
        else:
            self.submit_via_email(po, supplier)

        print(f"PO {po['po_number']} submitted to {supplier}")

    def submit_via_api(self, po, supplier):
        """Submit PO via supplier API"""

        import requests

        api_endpoint = self.config['supplier_api_endpoints'][supplier]

        payload = {
            'po_number': po['po_number'],
            'order_date': po['order_date'].isoformat(),
            'items': po['items']
        }

        response = requests.post(
            api_endpoint,
            json=payload,
            headers={'Authorization': f"Bearer {self.config['api_keys'][supplier]}"}
        )

        if response.status_code == 200:
            po['submission_status'] = 'success'
            po['supplier_confirmation'] = response.json().get('confirmation_number')
        else:
            po['submission_status'] = 'failed'
            self.send_alert(f"PO submission failed: {po['po_number']}")

    def submit_via_email(self, po, supplier):
        """Send PO via email"""

        # Get supplier email
        supplier_email = self.config['supplier_emails'].get(supplier)

        if not supplier_email:
            self.send_alert(f"No email found for supplier: {supplier}")
            return

        # Create email
        msg = MIMEMultipart()
        msg['From'] = self.config['email_config']['from_address']
        msg['To'] = supplier_email
        msg['Subject'] = f"Purchase Order {po['po_number']}"

        # Email body
        body = self.format_po_email(po)
        msg.attach(MIMEText(body, 'html'))

        # Send
        try:
            with smtplib.SMTP(self.config['email_config']['smtp_server'],
                            self.config['email_config']['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email_config']['username'],
                           self.config['email_config']['password'])
                server.send_message(msg)

            po['submission_status'] = 'success'
        except Exception as e:
            po['submission_status'] = 'failed'
            self.send_alert(f"Email sending failed: {e}")

    def format_po_email(self, po):
        """Format PO as HTML email"""

        html = f"""
        <html>
        <body>
            <h2>Purchase Order {po['po_number']}</h2>
            <p><strong>Order Date:</strong> {po['order_date'].strftime('%Y-%m-%d')}</p>

            <h3>Items:</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr>
                    <th>SKU</th>
                    <th>Quantity</th>
                    <th>Unit Cost</th>
                    <th>Line Total</th>
                </tr>
        """

        for item in po['items']:
            html += f"""
                <tr>
                    <td>{item['sku']}</td>
                    <td>{item['quantity']}</td>
                    <td>${item['unit_cost']:.2f}</td>
                    <td>${item['line_total']:.2f}</td>
                </tr>
            """

        html += f"""
            </table>

            <p><strong>Total:</strong> ${po['total_value']:,.2f}</p>

            <p>Please confirm receipt of this order.</p>
        </body>
        </html>
        """

        return html

    def send_approval_request(self, po):
        """Send approval request for high-value PO"""

        approval_url = f"{self.config['approval_portal_url']}/approve/{po['po_number']}"

        msg = MIMEText(f"""
        Purchase Order {po['po_number']} requires approval.

        Supplier: {po['supplier']}
        Total Value: ${po['total_value']:,.2f}
        Items: {len(po['items'])}

        Approve here: {approval_url}
        """)

        msg['Subject'] = f"PO Approval Required: {po['po_number']}"
        msg['From'] = self.config['email_config']['from_address']
        msg['To'] = self.config['approver_email']

        # Send email (simplified)
        print(f"Approval request sent for PO {po['po_number']}")

    def send_alert(self, message):
        """Send alert notification"""
        print(f"ALERT: {message}")
        # In production: send to monitoring system, email, Slack, etc.

    def run_daily_replenishment(self):
        """
        Main execution: daily automated replenishment cycle
        """

        print(f"\n{'='*60}")
        print(f"Starting Automated Replenishment Cycle")
        print(f"Time: {datetime.now()}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Load inventory data
            print("1. Loading inventory data...")
            self.load_inventory_data('erp_database')

            # Step 2: Calculate reorder needs
            print("2. Calculating reorder needs...")
            reorder_items = self.calculate_reorder_needs()
            print(f"   Found {len(reorder_items)} items needing reorder")

            if len(reorder_items) == 0:
                print("\nNo items need reordering. Cycle complete.")
                return

            # Step 3: Optimize order quantities
            print("3. Optimizing order quantities...")
            reorder_items = self.optimize_order_quantities(reorder_items)

            # Step 4: Generate purchase orders
            print("4. Generating purchase orders...")
            pos_created = self.generate_purchase_orders(reorder_items)
            print(f"   Created {len(pos_created)} purchase orders")

            # Step 5: Summary report
            print("\n" + "="*60)
            print("Replenishment Cycle Summary")
            print("="*60)
            for po in pos_created:
                print(f"PO {po['po_number']}: {po['supplier']} - "
                      f"${po['total_value']:,.2f} - {po['status']}")

            print(f"\nTotal Value: ${sum(po['total_value'] for po in pos_created):,.2f}")
            print("\nCycle completed successfully.\n")

        except Exception as e:
            print(f"\nERROR: Replenishment cycle failed: {e}")
            self.send_alert(f"Replenishment automation failed: {e}")

# Example usage
config = {
    'reorder_point_method': 'dynamic',
    'order_approval_threshold': 10000,
    'supplier_api_endpoints': {
        'Supplier_A': 'https://api.suppliera.com/orders'
    },
    'api_keys': {
        'Supplier_A': 'api_key_123'
    },
    'supplier_emails': {
        'Supplier_B': 'orders@supplierb.com'
    },
    'email_config': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'from_address': 'automation@company.com',
        'username': 'automation@company.com',
        'password': 'password'
    },
    'approver_email': 'manager@company.com',
    'approval_portal_url': 'https://portal.company.com'
}

# Create and run automation
replenishment = AutomatedReplenishment(config)
replenishment.run_daily_replenishment()
```

---

## Robotic Process Automation (RPA)

### Invoice Processing Automation

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytesseract
from PIL import Image
import pandas as pd
import time

class InvoiceProcessingBot:
    """
    RPA bot to automate invoice processing

    Tasks:
    1. Download invoices from email/portal
    2. Extract data using OCR
    3. Validate against POs
    4. Enter into ERP system
    5. Route for approval if needed
    """

    def __init__(self, config):
        self.config = config
        self.driver = None
        self.invoices_processed = []

    def initialize_browser(self):
        """Initialize web browser for automation"""

        options = webdriver.ChromeOptions()
        if self.config.get('headless', False):
            options.add_argument('--headless')

        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)

    def login_to_portal(self, url, username, password):
        """Login to supplier portal"""

        self.driver.get(url)

        # Wait for login form
        username_field = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )

        username_field.send_keys(username)
        self.driver.find_element(By.ID, "password").send_keys(password)
        self.driver.find_element(By.ID, "login-button").click()

        # Wait for dashboard to load
        time.sleep(2)

    def download_invoices(self):
        """Download new invoices from portal"""

        # Navigate to invoices page
        self.driver.find_element(By.LINK_TEXT, "Invoices").click()

        # Find new invoices
        invoice_rows = self.driver.find_elements(
            By.CSS_SELECTOR,
            "tr.invoice-row.status-new"
        )

        downloaded_files = []

        for row in invoice_rows:
            invoice_number = row.find_element(
                By.CSS_SELECTOR,
                ".invoice-number"
            ).text

            download_button = row.find_element(
                By.CSS_SELECTOR,
                ".download-button"
            )

            download_button.click()
            time.sleep(1)  # Wait for download

            downloaded_files.append({
                'invoice_number': invoice_number,
                'file_path': f"/downloads/{invoice_number}.pdf"
            })

        return downloaded_files

    def extract_invoice_data(self, file_path):
        """
        Extract data from invoice using OCR

        Returns structured invoice data
        """

        # Convert PDF to image (simplified)
        # In production, use pdf2image library
        image = Image.open(file_path)

        # OCR
        text = pytesseract.image_to_string(image)

        # Parse extracted text
        invoice_data = self.parse_invoice_text(text)

        return invoice_data

    def parse_invoice_text(self, text):
        """
        Parse OCR text to extract structured data

        Uses regex and NLP to find key fields
        """

        import re

        data = {}

        # Extract invoice number
        inv_match = re.search(r'Invoice #:?\s*(\w+)', text, re.IGNORECASE)
        if inv_match:
            data['invoice_number'] = inv_match.group(1)

        # Extract date
        date_match = re.search(
            r'Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            text,
            re.IGNORECASE
        )
        if date_match:
            data['invoice_date'] = date_match.group(1)

        # Extract PO number
        po_match = re.search(r'PO #:?\s*(\w+)', text, re.IGNORECASE)
        if po_match:
            data['po_number'] = po_match.group(1)

        # Extract total amount
        total_match = re.search(
            r'Total:?\s*\$?\s*([\d,]+\.?\d*)',
            text,
            re.IGNORECASE
        )
        if total_match:
            data['total_amount'] = float(total_match.group(1).replace(',', ''))

        # Extract line items (simplified)
        # In production, use more sophisticated parsing
        data['line_items'] = []

        return data

    def validate_invoice(self, invoice_data):
        """
        Validate invoice against PO and business rules

        Checks:
        - PO exists
        - Amounts match
        - Items match
        - Not a duplicate
        """

        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check PO exists
        po = self.lookup_po(invoice_data.get('po_number'))

        if not po:
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"PO {invoice_data.get('po_number')} not found"
            )
            return validation_results

        # Check amount matches (within tolerance)
        tolerance = 0.01  # 1%
        if abs(invoice_data['total_amount'] - po['total']) / po['total'] > tolerance:
            validation_results['warnings'].append(
                f"Amount mismatch: Invoice ${invoice_data['total_amount']:.2f} "
                f"vs PO ${po['total']:.2f}"
            )

        # Check for duplicate
        if self.is_duplicate_invoice(invoice_data['invoice_number']):
            validation_results['valid'] = False
            validation_results['errors'].append(
                f"Duplicate invoice: {invoice_data['invoice_number']}"
            )

        return validation_results

    def lookup_po(self, po_number):
        """Look up PO in ERP system"""
        # In production: query ERP database or API
        # Placeholder return
        return {
            'po_number': po_number,
            'total': 1250.00,
            'items': []
        }

    def is_duplicate_invoice(self, invoice_number):
        """Check if invoice already processed"""
        # In production: query invoice database
        return False

    def enter_invoice_in_erp(self, invoice_data):
        """
        Enter invoice into ERP system

        Uses RPA to navigate ERP interface
        """

        # Navigate to AP module
        self.driver.get(self.config['erp_url'])

        # Wait for page load
        time.sleep(2)

        # Click on Accounts Payable
        self.driver.find_element(By.LINK_TEXT, "Accounts Payable").click()

        # Click New Invoice
        self.driver.find_element(By.ID, "new-invoice-button").click()

        # Fill in invoice fields
        self.driver.find_element(By.ID, "invoice-number").send_keys(
            invoice_data['invoice_number']
        )

        self.driver.find_element(By.ID, "invoice-date").send_keys(
            invoice_data['invoice_date']
        )

        self.driver.find_element(By.ID, "po-number").send_keys(
            invoice_data['po_number']
        )

        self.driver.find_element(By.ID, "total-amount").send_keys(
            str(invoice_data['total_amount'])
        )

        # Add line items
        for item in invoice_data.get('line_items', []):
            # Click add line
            self.driver.find_element(By.ID, "add-line-button").click()

            # Fill line details
            # ... (detailed field entry)

        # Submit invoice
        self.driver.find_element(By.ID, "submit-button").click()

        # Wait for confirmation
        confirmation = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "success-message"))
        )

        return confirmation.text

    def run_invoice_processing(self):
        """
        Main automation workflow
        """

        print("\nStarting Invoice Processing Automation")
        print("="*60)

        try:
            # Initialize browser
            print("1. Initializing browser...")
            self.initialize_browser()

            # Login to supplier portal
            print("2. Logging in to supplier portal...")
            self.login_to_portal(
                self.config['supplier_portal_url'],
                self.config['portal_username'],
                self.config['portal_password']
            )

            # Download invoices
            print("3. Downloading invoices...")
            invoices = self.download_invoices()
            print(f"   Downloaded {len(invoices)} invoices")

            # Process each invoice
            for invoice_file in invoices:
                print(f"\nProcessing {invoice_file['invoice_number']}...")

                # Extract data
                invoice_data = self.extract_invoice_data(invoice_file['file_path'])

                # Validate
                validation = self.validate_invoice(invoice_data)

                if not validation['valid']:
                    print(f"   FAILED validation: {validation['errors']}")
                    # Route for manual review
                    continue

                if validation['warnings']:
                    print(f"   WARNINGS: {validation['warnings']}")
                    # Route for approval
                    continue

                # Enter in ERP
                result = self.enter_invoice_in_erp(invoice_data)
                print(f"   SUCCESS: {result}")

                self.invoices_processed.append({
                    'invoice_number': invoice_data['invoice_number'],
                    'status': 'processed',
                    'timestamp': datetime.now()
                })

            print(f"\n{'='*60}")
            print(f"Processing complete: {len(self.invoices_processed)} invoices")
            print("="*60 + "\n")

        except Exception as e:
            print(f"\nERROR: Automation failed: {e}")

        finally:
            if self.driver:
                self.driver.quit()

# Configuration
config = {
    'headless': False,
    'supplier_portal_url': 'https://portal.supplier.com',
    'portal_username': 'user@company.com',
    'portal_password': 'password',
    'erp_url': 'https://erp.company.com',
    'download_path': '/downloads'
}

# Run automation
bot = InvoiceProcessingBot(config)
# bot.run_invoice_processing()
```

---

## Workflow Orchestration

### Apache Airflow DAG Example

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Define workflow tasks
def extract_orders_from_erp():
    """Extract new orders from ERP system"""
    print("Extracting orders from ERP...")
    # Implementation
    return {'orders_count': 150}

def validate_inventory():
    """Check inventory availability"""
    print("Validating inventory...")
    # Implementation
    return {'available': True}

def allocate_inventory():
    """Allocate inventory to orders"""
    print("Allocating inventory...")
    # Implementation
    return {'allocated_orders': 145}

def generate_pick_lists():
    """Generate warehouse pick lists"""
    print("Generating pick lists...")
    # Implementation
    return {'pick_lists_created': 145}

def send_to_wms():
    """Send pick lists to WMS"""
    print("Sending to WMS...")
    # Implementation
    return {'status': 'success'}

# Define DAG
default_args = {
    'owner': 'supply-chain',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'order_fulfillment_automation',
    default_args=default_args,
    description='Automated order fulfillment workflow',
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    start_date=days_ago(1),
    catchup=False,
    tags=['supply-chain', 'fulfillment']
)

# Define tasks
t1 = PythonOperator(
    task_id='extract_orders',
    python_callable=extract_orders_from_erp,
    dag=dag
)

t2 = PythonOperator(
    task_id='validate_inventory',
    python_callable=validate_inventory,
    dag=dag
)

t3 = PythonOperator(
    task_id='allocate_inventory',
    python_callable=allocate_inventory,
    dag=dag
)

t4 = PythonOperator(
    task_id='generate_pick_lists',
    python_callable=generate_pick_lists,
    dag=dag
)

t5 = PythonOperator(
    task_id='send_to_wms',
    python_callable=send_to_wms,
    dag=dag
)

t6 = EmailOperator(
    task_id='send_completion_email',
    to='operations@company.com',
    subject='Order Fulfillment Workflow Complete',
    html_content='<p>The order fulfillment workflow has completed successfully.</p>',
    dag=dag
)

# Define dependencies
t1 >> t2 >> t3 >> t4 >> t5 >> t6
```

---

## API Integration & Middleware

### REST API Integration

```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
from datetime import datetime

class SupplyChainAPIIntegration:
    """
    Middleware for integrating multiple supply chain systems
    """

    def __init__(self, config):
        self.config = config
        self.session = self.create_session()

    def create_session(self):
        """
        Create requests session with retry logic
        """

        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
            backoff_factor=1
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def get_erp_orders(self, date_from, date_to):
        """
        Fetch orders from ERP system

        ERP: SAP, Oracle, NetSuite, etc.
        """

        url = f"{self.config['erp_base_url']}/api/orders"

        headers = {
            'Authorization': f"Bearer {self.config['erp_api_token']}",
            'Content-Type': 'application/json'
        }

        params = {
            'date_from': date_from.isoformat(),
            'date_to': date_to.isoformat(),
            'status': 'open'
        }

        response = self.session.get(url, headers=headers, params=params)

        response.raise_for_status()

        return response.json()['orders']

    def check_wms_inventory(self, sku):
        """
        Check inventory levels in WMS

        WMS: Manhattan, Blue Yonder, HighJump, etc.
        """

        url = f"{self.config['wms_base_url']}/api/inventory/{sku}"

        headers = {
            'X-API-Key': self.config['wms_api_key']
        }

        response = self.session.get(url, headers=headers)

        response.raise_for_status()

        inventory_data = response.json()

        return {
            'sku': sku,
            'on_hand': inventory_data['quantity_on_hand'],
            'allocated': inventory_data['quantity_allocated'],
            'available': inventory_data['quantity_available']
        }

    def create_wms_shipment(self, order_data):
        """
        Create shipment in WMS
        """

        url = f"{self.config['wms_base_url']}/api/shipments"

        headers = {
            'X-API-Key': self.config['wms_api_key'],
            'Content-Type': 'application/json'
        }

        payload = {
            'order_number': order_data['order_number'],
            'customer': order_data['customer'],
            'ship_to_address': order_data['shipping_address'],
            'line_items': order_data['items'],
            'priority': order_data.get('priority', 'normal'),
            'requested_ship_date': order_data['requested_ship_date']
        }

        response = self.session.post(url, headers=headers, json=payload)

        response.raise_for_status()

        return response.json()

    def get_tms_rates(self, shipment_details):
        """
        Get shipping rates from TMS

        TMS: MercuryGate, Oracle TMS, Manhattan TMS, etc.
        """

        url = f"{self.config['tms_base_url']}/api/rates/quote"

        headers = {
            'Authorization': f"Bearer {self.config['tms_api_token']}",
            'Content-Type': 'application/json'
        }

        payload = {
            'origin': shipment_details['origin'],
            'destination': shipment_details['destination'],
            'weight': shipment_details['weight'],
            'dimensions': shipment_details['dimensions'],
            'service_type': shipment_details.get('service_type', 'ground')
        }

        response = self.session.post(url, headers=headers, json=payload)

        response.raise_for_status()

        rates = response.json()['rates']

        # Return best rate
        return min(rates, key=lambda x: x['total_cost'])

    def sync_order_status(self, order_number):
        """
        Sync order status across systems

        Ensures ERP, WMS, TMS all have current status
        """

        # Get status from WMS
        wms_status = self.get_wms_order_status(order_number)

        # Update ERP
        self.update_erp_order_status(order_number, wms_status)

        # If shipped, update TMS
        if wms_status['status'] == 'shipped':
            self.update_tms_tracking(order_number, wms_status)

        return wms_status

    def orchestrate_order_fulfillment(self, order_number):
        """
        End-to-end order fulfillment orchestration

        Coordinates across ERP, WMS, TMS
        """

        print(f"\nOrchestrating fulfillment for order {order_number}")

        try:
            # 1. Get order details from ERP
            order = self.get_order_from_erp(order_number)
            print(f"  Order retrieved from ERP")

            # 2. Check inventory in WMS
            inventory_available = True
            for item in order['items']:
                inventory = self.check_wms_inventory(item['sku'])
                if inventory['available'] < item['quantity']:
                    inventory_available = False
                    print(f"  WARNING: Insufficient inventory for {item['sku']}")

            if not inventory_available:
                return {'status': 'backorder'}

            # 3. Create shipment in WMS
            shipment = self.create_wms_shipment(order)
            print(f"  Shipment created in WMS: {shipment['shipment_id']}")

            # 4. Get shipping rates from TMS
            rate = self.get_tms_rates({
                'origin': shipment['origin'],
                'destination': order['shipping_address'],
                'weight': shipment['total_weight'],
                'dimensions': shipment['dimensions']
            })
            print(f"  Best shipping rate: ${rate['total_cost']:.2f}")

            # 5. Update ERP with shipping info
            self.update_erp_order(order_number, {
                'shipment_id': shipment['shipment_id'],
                'carrier': rate['carrier'],
                'freight_cost': rate['total_cost'],
                'status': 'in_fulfillment'
            })
            print(f"  ERP updated with shipment details")

            return {
                'status': 'success',
                'shipment_id': shipment['shipment_id'],
                'freight_cost': rate['total_cost']
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            return {'status': 'error', 'message': str(e)}

# Configuration
config = {
    'erp_base_url': 'https://erp.company.com',
    'erp_api_token': 'erp_token_123',
    'wms_base_url': 'https://wms.company.com',
    'wms_api_key': 'wms_key_456',
    'tms_base_url': 'https://tms.company.com',
    'tms_api_token': 'tms_token_789'
}

# Example usage
integration = SupplyChainAPIIntegration(config)
result = integration.orchestrate_order_fulfillment('ORDER-12345')
print(f"\nResult: {result}")
```

---

## Tools & Technologies

### RPA Platforms

**Commercial:**
- **UiPath**: Leading RPA platform
- **Blue Prism**: Enterprise RPA
- **Automation Anywhere**: Cloud-native RPA
- **Microsoft Power Automate**: Microsoft ecosystem integration
- **WorkFusion**: AI-powered automation

**Open Source:**
- **Robot Framework**: Generic automation framework
- **Selenium**: Web browser automation
- **Puppeteer**: Node.js browser automation
- **TagUI**: RPA tool for automating websites

### Workflow Orchestration

**Apache Airflow**: Python-based workflow orchestration
**Prefect**: Modern workflow orchestration
**Luigi (Spotify)**: Python workflow engine
**Dagster**: Data orchestration platform
**n8n**: Workflow automation (low-code)
**Zapier**: No-code automation (SaaS)
**Make (Integromat)**: Visual automation platform

### API Integration

**Python Libraries:**
- `requests`: HTTP library
- `httpx`: Async HTTP client
- `aiohttp`: Async HTTP client/server
- `fastapi`: Build APIs
- `celery`: Distributed task queue

**iPaaS (Integration Platform as a Service):**
- **MuleSoft**: Enterprise integration
- **Dell Boomi**: Cloud integration
- **Informatica**: Data integration
- **Jitterbit**: Integration platform
- **Workato**: Enterprise automation

---

## Common Challenges & Solutions

### Challenge: System Downtime and Failures

**Problem:**
- Automated processes fail when systems are down
- No manual fallback
- Data inconsistency

**Solutions:**
- Implement retry logic with exponential backoff
- Circuit breaker pattern
- Health checks and monitoring
- Fallback to manual process
- Queue-based processing (can resume after downtime)
- Comprehensive error logging

### Challenge: Change Management

**Problem:**
- UI changes break RPA bots
- API versioning issues
- Business process changes

**Solutions:**
- Use APIs instead of UI automation when possible
- Modular design (easy to update components)
- Version control for automation scripts
- Regular maintenance schedule
- Monitoring for automation failures
- Documentation of dependencies

### Challenge: Data Quality Issues

**Problem:**
- Bad data causes automation failures
- Garbage in, garbage out

**Solutions:**
- Input validation before processing
- Data quality checks
- Exception handling and alerts
- Human review for edge cases
- Data cleansing preprocessing
- Clear business rules for data standards

### Challenge: Security and Compliance

**Problem:**
- Bots have access to sensitive systems
- Audit trail concerns
- Regulatory compliance

**Solutions:**
- Principle of least privilege (minimal access)
- Credential vaulting (no hardcoded passwords)
- Comprehensive logging of all actions
- Regular security audits
- Encryption of sensitive data
- Compliance with SOX, GDPR, etc.

---

## Output Format

### Automation Project Report

**Executive Summary:**
- Process automated
- Expected benefits (time savings, cost reduction, accuracy)
- Implementation timeline
- ROI analysis

**Current State Analysis:**
- Process description
- Volume and frequency
- Current pain points
- Time and cost metrics

**Solution Design:**
- Automation approach
- Systems integrated
- Workflow diagram
- Exception handling

**Implementation Plan:**

| Phase | Activities | Duration | Resources |
|-------|-----------|----------|-----------|
| 1. Setup | Tool installation, access provisioning | 2 weeks | IT, 1 developer |
| 2. Development | Bot development, testing | 4 weeks | 2 developers |
| 3. UAT | User acceptance testing | 2 weeks | Business users |
| 4. Deployment | Production deployment, monitoring | 1 week | IT, developers |
| 5. Support | Hypercare, optimization | 4 weeks | Support team |

**Business Case:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Process Time | 4 hours/day | 30 min/day | 88% reduction |
| Error Rate | 5% | 0.5% | 90% reduction |
| FTE Required | 1.0 | 0.25 | 0.75 FTE saved |
| Annual Cost | $80,000 | $20,000 | $60,000 savings |

**ROI:**
- Investment: $150,000 (development + infrastructure)
- Annual Savings: $60,000
- Payback Period: 2.5 years
- 3-Year ROI: 20%

---

## Questions to Ask

If you need more context:
1. What process needs automation?
2. What's the current process? (manual steps, systems involved)
3. What's the volume and frequency? (1000 transactions/day, hourly)
4. What systems are involved? (ERP, WMS, TMS, spreadsheets)
5. Are APIs available or screen scraping needed?
6. What's the expected ROI and timeline?
7. Are there compliance or security requirements?
8. Who will maintain the automation?

---

## Related Skills

- **supply-chain-analytics**: For monitoring automation performance
- **digital-twin-modeling**: For simulating automated processes
- **ml-supply-chain**: For intelligent automation with ML
- **prescriptive-analytics**: For automated decision-making
- **optimization-modeling**: For optimizing automated workflows
- **demand-forecasting**: For automated replenishment
- **inventory-optimization**: For automated reorder triggers
- **order-fulfillment**: For fulfillment automation

---
name: supply-chain-analytics
description: When the user wants to analyze supply chain performance, build analytics dashboards, or track KPIs. Also use when the user mentions "supply chain metrics," "performance analytics," "KPI tracking," "dashboard design," "business intelligence," "data visualization," "supply chain reporting," "metrics analysis," or "performance measurement." For forecasting specifically, see demand-forecasting. For optimization, see optimization-modeling.
---

# Supply Chain Analytics

You are an expert in supply chain analytics and performance measurement. Your goal is to help design comprehensive analytics frameworks, build insightful dashboards, track meaningful KPIs, and derive actionable insights from supply chain data.

## Initial Assessment

Before building analytics solutions, understand:

1. **Business Context**
   - What supply chain processes need tracking? (procurement, inventory, fulfillment, transportation)
   - Who are the stakeholders? (executives, operations, planners, analysts)
   - What decisions will these analytics support?
   - Current pain points? (lack of visibility, poor data quality, manual reporting)

2. **Current State**
   - Existing analytics tools? (Excel, Tableau, Power BI, custom dashboards)
   - Data sources available? (ERP, WMS, TMS, spreadsheets)
   - Reporting frequency? (real-time, daily, weekly, monthly)
   - Known data quality issues?

3. **Strategic Objectives**
   - Primary goals? (cost reduction, service improvement, efficiency gains)
   - Target KPIs and benchmarks?
   - Compliance or regulatory requirements?
   - Integration needs with existing systems?

4. **Technical Environment**
   - IT infrastructure? (cloud, on-premise, hybrid)
   - Database systems? (SQL Server, PostgreSQL, Snowflake)
   - Programming capabilities? (Python, R, SQL)
   - BI tool preferences?

---

## Supply Chain Analytics Framework

### Analytics Hierarchy (Gartner Model)

**1. Descriptive Analytics** (What happened?)
- Historical performance reporting
- KPI dashboards
- Trend analysis
- Example: Last month's on-time delivery was 92%

**2. Diagnostic Analytics** (Why did it happen?)
- Root cause analysis
- Variance analysis
- Correlation studies
- Example: Delivery delays caused by supplier issues in 65% of cases

**3. Predictive Analytics** (What will happen?)
- Forecasting models
- Demand prediction
- Risk scoring
- Example: 80% probability of stockout next week

**4. Prescriptive Analytics** (What should we do?)
- Optimization recommendations
- Decision support
- Scenario planning
- Example: Expedite shipment from alternate supplier to prevent stockout

### Key Performance Areas

**1. Cost Metrics**
- Total supply chain cost
- Cost-to-serve by customer/product
- Transportation cost per unit
- Warehousing cost per order
- Procurement savings

**2. Service Metrics**
- On-time delivery (OTD)
- Order fill rate
- Perfect order rate
- Order cycle time
- Customer satisfaction (CSAT)

**3. Efficiency Metrics**
- Inventory turnover
- Days sales outstanding (DSO)
- Cash-to-cash cycle time
- Asset utilization
- Productivity (units per labor hour)

**4. Quality Metrics**
- Order accuracy
- Damage rate
- Returns rate
- Compliance rate
- Supplier quality score

---

## Core Supply Chain KPIs

### Comprehensive KPI Library

#### Procurement & Supplier Management

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_procurement_kpis(orders_df):
    """
    Calculate key procurement KPIs

    orders_df should have: order_date, delivery_date, po_number,
                          supplier, order_value, defect_rate
    """
    kpis = {}

    # 1. On-Time Delivery Rate
    orders_df['on_time'] = (orders_df['delivery_date'] <=
                            orders_df['promised_date']).astype(int)
    kpis['on_time_delivery_rate'] = orders_df['on_time'].mean() * 100

    # 2. Supplier Lead Time (average)
    orders_df['lead_time'] = (orders_df['delivery_date'] -
                              orders_df['order_date']).dt.days
    kpis['avg_lead_time_days'] = orders_df['lead_time'].mean()

    # 3. Supplier Defect Rate
    kpis['defect_rate_ppm'] = orders_df['defect_rate'].mean() * 1000000

    # 4. Purchase Order Cycle Time
    kpis['po_cycle_time_days'] = (orders_df['po_approved_date'] -
                                   orders_df['requisition_date']).dt.days.mean()

    # 5. Spend Under Management
    total_spend = orders_df['order_value'].sum()
    contracted_spend = orders_df[orders_df['has_contract']]['order_value'].sum()
    kpis['spend_under_management_pct'] = (contracted_spend / total_spend) * 100

    # 6. Supplier Concentration (top 3 suppliers % of spend)
    supplier_spend = orders_df.groupby('supplier')['order_value'].sum().sort_values(ascending=False)
    top3_pct = supplier_spend.head(3).sum() / total_spend * 100
    kpis['supplier_concentration_top3_pct'] = top3_pct

    # 7. Cost Savings vs. Baseline
    if 'baseline_cost' in orders_df.columns:
        actual_cost = orders_df['order_value'].sum()
        baseline_cost = orders_df['baseline_cost'].sum()
        kpis['cost_savings'] = baseline_cost - actual_cost
        kpis['cost_savings_pct'] = ((baseline_cost - actual_cost) / baseline_cost) * 100

    return kpis

# Example usage
procurement_kpis = calculate_procurement_kpis(orders_df)
print(f"On-Time Delivery: {procurement_kpis['on_time_delivery_rate']:.1f}%")
print(f"Average Lead Time: {procurement_kpis['avg_lead_time_days']:.1f} days")
```

#### Inventory Management

```python
def calculate_inventory_kpis(inventory_df, sales_df, cogs):
    """
    Calculate inventory KPIs

    Parameters:
    - inventory_df: columns = ['date', 'sku', 'on_hand_qty', 'unit_cost']
    - sales_df: columns = ['date', 'sku', 'quantity_sold']
    - cogs: Cost of Goods Sold (annual)
    """
    kpis = {}

    # 1. Inventory Turnover
    avg_inventory_value = (inventory_df['on_hand_qty'] *
                          inventory_df['unit_cost']).mean()
    kpis['inventory_turns'] = cogs / avg_inventory_value

    # 2. Days on Hand (DOH)
    kpis['days_on_hand'] = 365 / kpis['inventory_turns']

    # 3. Inventory Accuracy
    if 'physical_count' in inventory_df.columns:
        inventory_df['accurate'] = (
            abs(inventory_df['on_hand_qty'] - inventory_df['physical_count'])
            / inventory_df['physical_count'] < 0.02
        ).astype(int)
        kpis['inventory_accuracy_pct'] = inventory_df['accurate'].mean() * 100

    # 4. Stock-Out Rate
    total_sku_days = len(inventory_df)
    stockout_days = (inventory_df['on_hand_qty'] == 0).sum()
    kpis['stockout_rate_pct'] = (stockout_days / total_sku_days) * 100

    # 5. Excess & Obsolete Inventory
    # Items with >180 days of inventory
    daily_sales = sales_df.groupby('sku')['quantity_sold'].mean()
    inventory_latest = inventory_df.groupby('sku')['on_hand_qty'].last()

    days_of_supply = inventory_latest / daily_sales
    excess_value = inventory_df[
        inventory_df['sku'].isin(days_of_supply[days_of_supply > 180].index)
    ]['on_hand_qty'] * inventory_df['unit_cost']

    kpis['excess_obsolete_value'] = excess_value.sum()
    kpis['excess_obsolete_pct'] = (excess_value.sum() /
                                    avg_inventory_value) * 100

    # 6. Inventory Fill Rate
    total_demand = sales_df['quantity_sold'].sum()
    stockouts = sales_df[sales_df['sku'].isin(
        inventory_df[inventory_df['on_hand_qty'] == 0]['sku']
    )]['quantity_sold'].sum()

    kpis['fill_rate_pct'] = ((total_demand - stockouts) / total_demand) * 100

    # 7. Carrying Cost
    carrying_cost_rate = 0.25  # 25% annual carrying cost
    kpis['annual_carrying_cost'] = avg_inventory_value * carrying_cost_rate

    return kpis

# Example
inventory_kpis = calculate_inventory_kpis(inventory_df, sales_df, cogs=50_000_000)
print(f"Inventory Turns: {inventory_kpis['inventory_turns']:.2f}")
print(f"Days on Hand: {inventory_kpis['days_on_hand']:.1f}")
print(f"Fill Rate: {inventory_kpis['fill_rate_pct']:.1f}%")
```

#### Warehouse & Fulfillment

```python
def calculate_warehouse_kpis(orders_df, shipments_df, warehouse_sqft, labor_hours):
    """
    Calculate warehouse and fulfillment KPIs

    Parameters:
    - orders_df: order details with timestamps
    - shipments_df: shipment details
    - warehouse_sqft: total warehouse square footage
    - labor_hours: total labor hours in period
    """
    kpis = {}

    # 1. Order Cycle Time (order to ship)
    orders_df['cycle_time'] = (orders_df['ship_date'] -
                               orders_df['order_date']).dt.total_seconds() / 3600
    kpis['avg_order_cycle_time_hours'] = orders_df['cycle_time'].mean()

    # 2. Perfect Order Rate
    orders_df['perfect'] = (
        (orders_df['on_time'] == 1) &
        (orders_df['complete'] == 1) &
        (orders_df['damage_free'] == 1) &
        (orders_df['accurate'] == 1)
    ).astype(int)
    kpis['perfect_order_rate_pct'] = orders_df['perfect'].mean() * 100

    # 3. Order Picking Accuracy
    kpis['picking_accuracy_pct'] = (1 - orders_df['picking_errors'].sum() /
                                    orders_df['lines_picked'].sum()) * 100

    # 4. Warehouse Utilization
    avg_inventory_cube = orders_df['inventory_cubic_ft'].mean()
    kpis['space_utilization_pct'] = (avg_inventory_cube / warehouse_sqft) * 100

    # 5. Units per Labor Hour
    total_units = orders_df['units_shipped'].sum()
    kpis['units_per_labor_hour'] = total_units / labor_hours

    # 6. Cost per Order
    total_warehouse_cost = 500000  # example monthly cost
    total_orders = len(orders_df)
    kpis['cost_per_order'] = total_warehouse_cost / total_orders

    # 7. On-Time Shipment Rate
    shipments_df['on_time'] = (shipments_df['actual_ship_date'] <=
                               shipments_df['promised_ship_date']).astype(int)
    kpis['on_time_shipment_pct'] = shipments_df['on_time'].mean() * 100

    # 8. Dock-to-Stock Time
    if 'receipt_date' in shipments_df.columns:
        shipments_df['dock_to_stock_hours'] = (
            shipments_df['put_away_date'] -
            shipments_df['receipt_date']
        ).dt.total_seconds() / 3600
        kpis['avg_dock_to_stock_hours'] = shipments_df['dock_to_stock_hours'].mean()

    # 9. Order Accuracy Rate
    kpis['order_accuracy_pct'] = (1 - orders_df['errors'].sum() /
                                   len(orders_df)) * 100

    return kpis

warehouse_kpis = calculate_warehouse_kpis(orders_df, shipments_df,
                                          warehouse_sqft=200000,
                                          labor_hours=10000)
print(f"Perfect Order Rate: {warehouse_kpis['perfect_order_rate_pct']:.1f}%")
print(f"Units/Labor Hour: {warehouse_kpis['units_per_labor_hour']:.1f}")
```

#### Transportation & Logistics

```python
def calculate_transportation_kpis(shipments_df):
    """
    Calculate transportation and logistics KPIs

    shipments_df: columns = ['shipment_id', 'origin', 'destination',
                             'ship_date', 'delivery_date', 'promised_date',
                             'freight_cost', 'miles', 'weight_lbs', 'mode']
    """
    kpis = {}

    # 1. On-Time Delivery (OTD)
    shipments_df['on_time'] = (shipments_df['delivery_date'] <=
                               shipments_df['promised_date']).astype(int)
    kpis['on_time_delivery_pct'] = shipments_df['on_time'].mean() * 100

    # 2. Transit Time
    shipments_df['transit_days'] = (shipments_df['delivery_date'] -
                                    shipments_df['ship_date']).dt.days
    kpis['avg_transit_days'] = shipments_df['transit_days'].mean()

    # 3. Freight Cost per Mile
    total_cost = shipments_df['freight_cost'].sum()
    total_miles = shipments_df['miles'].sum()
    kpis['cost_per_mile'] = total_cost / total_miles

    # 4. Cost per Pound
    total_weight = shipments_df['weight_lbs'].sum()
    kpis['cost_per_lb'] = total_cost / total_weight

    # 5. Freight as % of Revenue
    revenue = shipments_df['order_value'].sum()
    kpis['freight_as_pct_revenue'] = (total_cost / revenue) * 100

    # 6. Damage Rate
    damaged_shipments = shipments_df['damaged'].sum()
    kpis['damage_rate_pct'] = (damaged_shipments / len(shipments_df)) * 100

    # 7. Load Factor / Utilization
    if 'truck_capacity_lbs' in shipments_df.columns:
        shipments_df['utilization'] = (shipments_df['weight_lbs'] /
                                       shipments_df['truck_capacity_lbs'])
        kpis['avg_load_utilization_pct'] = shipments_df['utilization'].mean() * 100

    # 8. Carrier Performance Score
    # Weighted score: OTD (50%), Transit Time (30%), Damage (20%)
    carrier_metrics = shipments_df.groupby('carrier').agg({
        'on_time': 'mean',
        'transit_days': 'mean',
        'damaged': 'mean',
        'shipment_id': 'count'
    })

    # Normalize and score
    carrier_metrics['otd_score'] = carrier_metrics['on_time'] * 50
    carrier_metrics['transit_score'] = (1 - carrier_metrics['transit_days'] /
                                        carrier_metrics['transit_days'].max()) * 30
    carrier_metrics['damage_score'] = (1 - carrier_metrics['damaged']) * 20
    carrier_metrics['total_score'] = (carrier_metrics['otd_score'] +
                                      carrier_metrics['transit_score'] +
                                      carrier_metrics['damage_score'])

    kpis['carrier_performance'] = carrier_metrics[['total_score']].to_dict()

    # 9. Empty Miles Percentage
    if 'empty_miles' in shipments_df.columns:
        kpis['empty_miles_pct'] = (shipments_df['empty_miles'].sum() /
                                   total_miles) * 100

    return kpis

transportation_kpis = calculate_transportation_kpis(shipments_df)
print(f"On-Time Delivery: {transportation_kpis['on_time_delivery_pct']:.1f}%")
print(f"Cost per Mile: ${transportation_kpis['cost_per_mile']:.2f}")
```

#### Overall Supply Chain Performance

```python
def calculate_scor_metrics(data_dict):
    """
    Calculate SCOR (Supply Chain Operations Reference) Model metrics

    SCOR Level 1 Metrics across 5 performance attributes:
    - Reliability: Perfect Order Fulfillment
    - Responsiveness: Order Fulfillment Cycle Time
    - Agility: Upside Supply Chain Flexibility
    - Costs: Total Supply Chain Management Cost
    - Assets: Cash-to-Cash Cycle Time
    """
    scor_metrics = {}

    # RELIABILITY
    # Perfect Order Fulfillment = % orders delivered complete, on-time, damage-free, with accurate docs
    orders = data_dict['orders']
    perfect_orders = orders[
        (orders['complete'] == 1) &
        (orders['on_time'] == 1) &
        (orders['damage_free'] == 1) &
        (orders['docs_accurate'] == 1)
    ]
    scor_metrics['RL.1.1_perfect_order_pct'] = (len(perfect_orders) / len(orders)) * 100

    # RESPONSIVENESS
    # Order Fulfillment Cycle Time = avg time from order receipt to customer receipt
    scor_metrics['RS.1.1_order_cycle_time_days'] = (
        orders['delivery_date'] - orders['order_date']
    ).dt.days.mean()

    # AGILITY
    # Upside Supply Chain Flexibility = % increase in deliverable quantity (30 days notice)
    # Typically requires historical capacity data
    scor_metrics['AG.1.1_upside_flexibility_pct'] = 20  # Example: 20% flex capacity

    # COSTS
    # Total SC Management Cost = sum of all SC costs as % of revenue
    sc_costs = {
        'plan': data_dict.get('planning_cost', 0),
        'source': data_dict.get('procurement_cost', 0),
        'make': data_dict.get('manufacturing_cost', 0),
        'deliver': data_dict.get('logistics_cost', 0),
        'return': data_dict.get('returns_cost', 0)
    }
    total_sc_cost = sum(sc_costs.values())
    revenue = data_dict['revenue']
    scor_metrics['CO.1.1_total_sc_cost_pct_revenue'] = (total_sc_cost / revenue) * 100

    # ASSETS
    # Cash-to-Cash Cycle Time = DSO + DIO - DPO
    dso = data_dict.get('days_sales_outstanding', 45)  # Days Sales Outstanding
    dio = data_dict.get('days_inventory_outstanding', 60)  # Days Inventory Outstanding
    dpo = data_dict.get('days_payable_outstanding', 30)  # Days Payable Outstanding

    scor_metrics['AM.1.1_cash_to_cash_cycle_days'] = dso + dio - dpo

    # Return on Supply Chain Fixed Assets
    scor_metrics['AM.1.2_rosc_fixed_assets'] = (
        revenue / data_dict['sc_fixed_assets']
    )

    # Working Capital
    scor_metrics['AM.1.3_working_capital'] = (
        data_dict['current_assets'] - data_dict['current_liabilities']
    )

    return scor_metrics

# Example
data_dict = {
    'orders': orders_df,
    'planning_cost': 1_000_000,
    'procurement_cost': 5_000_000,
    'manufacturing_cost': 30_000_000,
    'logistics_cost': 8_000_000,
    'returns_cost': 1_000_000,
    'revenue': 100_000_000,
    'days_sales_outstanding': 45,
    'days_inventory_outstanding': 60,
    'days_payable_outstanding': 30,
    'sc_fixed_assets': 50_000_000,
    'current_assets': 25_000_000,
    'current_liabilities': 15_000_000
}

scor_metrics = calculate_scor_metrics(data_dict)
print(f"Perfect Order: {scor_metrics['RL.1.1_perfect_order_pct']:.1f}%")
print(f"Cash-to-Cash: {scor_metrics['AM.1.1_cash_to_cash_cycle_days']:.0f} days")
```

---

## Analytics Dashboard Design

### Executive Dashboard Example

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class SupplyChainDashboard:
    """
    Build interactive supply chain analytics dashboard
    """

    def __init__(self, data):
        self.data = data

    def create_executive_dashboard(self):
        """
        Create executive-level dashboard with key metrics
        """

        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Perfect Order Rate', 'On-Time Delivery Trend', 'Inventory Turns',
                'Supply Chain Cost', 'Order Volume by Region', 'Top 10 SKUs',
                'Supplier Performance', 'Warehouse Utilization', 'Freight Cost/Mile'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'scatter'}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )

        # 1. Perfect Order Rate (Gauge)
        perfect_order_rate = 94.5
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=perfect_order_rate,
                title={'text': "Perfect Order %"},
                delta={'reference': 90, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 90], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=1, col=1
        )

        # 2. On-Time Delivery Trend
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        otd_trend = np.random.normal(92, 3, 12)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=otd_trend,
                mode='lines+markers',
                name='OTD %',
                line=dict(color='blue', width=3),
                fill='tozeroy'
            ),
            row=1, col=2
        )

        # 3. Inventory Turns (Indicator)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=6.2,
                title={'text': "Inventory Turns"},
                delta={'reference': 5.5, 'increasing': {'color': "green"}},
                number={'suffix': "x"}
            ),
            row=1, col=3
        )

        # 4. Supply Chain Cost Breakdown
        cost_categories = ['Procurement', 'Manufacturing', 'Warehousing',
                          'Transportation', 'Returns']
        costs = [5.0, 30.0, 4.0, 8.0, 1.0]

        fig.add_trace(
            go.Bar(
                x=cost_categories,
                y=costs,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                text=[f'${c}M' for c in costs],
                textposition='auto'
            ),
            row=2, col=1
        )

        # 5. Order Volume by Region (Pie)
        regions = ['North', 'South', 'East', 'West']
        volumes = [2500, 1800, 2200, 1500]

        fig.add_trace(
            go.Pie(
                labels=regions,
                values=volumes,
                hole=0.3
            ),
            row=2, col=2
        )

        # 6. Top 10 SKUs by Revenue
        skus = [f'SKU_{i}' for i in range(1, 11)]
        revenues = np.random.uniform(500, 2000, 10)

        fig.add_trace(
            go.Bar(
                y=skus,
                x=revenues,
                orientation='h',
                marker_color='lightblue'
            ),
            row=2, col=3
        )

        # 7. Supplier Performance Scores
        suppliers = ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E']
        scores = [92, 88, 95, 85, 90]

        fig.add_trace(
            go.Bar(
                x=suppliers,
                y=scores,
                marker_color=['green' if s >= 90 else 'orange' if s >= 85 else 'red'
                             for s in scores]
            ),
            row=3, col=1
        )

        # 8. Warehouse Utilization Trend
        utilization = np.random.uniform(75, 90, 12)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=utilization,
                mode='lines+markers',
                line=dict(color='green'),
                fill='tozeroy'
            ),
            row=3, col=2
        )

        # 9. Freight Cost per Mile Trend
        freight_cost = np.random.uniform(2.20, 2.80, 12)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=freight_cost,
                mode='lines+markers',
                line=dict(color='red')
            ),
            row=3, col=3
        )

        # Update layout
        fig.update_layout(
            title_text="Supply Chain Executive Dashboard",
            showlegend=False,
            height=1200,
            width=1600
        )

        return fig

    def create_operational_dashboard(self):
        """
        Operational dashboard with detailed metrics
        """
        # Similar structure but more detailed, real-time focused
        pass

# Usage
dashboard = SupplyChainDashboard(data)
fig = dashboard.create_executive_dashboard()
fig.show()
# or save: fig.write_html("sc_dashboard.html")
```

### Real-Time Monitoring Dashboard

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def create_realtime_dashboard():
    """
    Streamlit-based real-time monitoring dashboard
    """

    st.set_page_config(layout="wide", page_title="SC Real-Time Monitor")

    st.title("Supply Chain Real-Time Monitoring")

    # Refresh every 30 seconds
    st.markdown("Auto-refresh: 30 seconds")

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Orders Today",
            value="1,247",
            delta="12% vs yesterday"
        )

    with col2:
        st.metric(
            label="On-Time Delivery",
            value="94.2%",
            delta="1.5%"
        )

    with col3:
        st.metric(
            label="Active Shipments",
            value="456",
            delta="-23"
        )

    with col4:
        st.metric(
            label="Warehouse Fill",
            value="82%",
            delta="3%"
        )

    with col5:
        st.metric(
            label="Alerts",
            value="7",
            delta="-2",
            delta_color="inverse"
        )

    # Alerts Section
    st.subheader("Active Alerts")
    alerts_df = pd.DataFrame({
        'Time': ['10:23 AM', '09:45 AM', '09:12 AM'],
        'Severity': ['HIGH', 'MEDIUM', 'LOW'],
        'Type': ['Stockout', 'Delayed Shipment', 'Low Inventory'],
        'Description': [
            'SKU_1234 out of stock at DC_West',
            'Shipment #45678 delayed by 4 hours',
            'SKU_5678 below reorder point'
        ]
    })

    # Color code by severity
    def highlight_severity(val):
        color = {'HIGH': 'background-color: #ffcccc',
                'MEDIUM': 'background-color: #fff4cc',
                'LOW': 'background-color: #e6f3ff'}
        return color.get(val, '')

    st.dataframe(
        alerts_df.style.applymap(highlight_severity, subset=['Severity']),
        use_container_width=True
    )

    # Charts Row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Hourly Order Volume")

        # Generate sample data
        hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
        volumes = np.random.poisson(50, 24)

        fig = px.line(
            x=hours,
            y=volumes,
            labels={'x': 'Time', 'y': 'Orders'},
            title='Last 24 Hours'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Shipment Status")

        status_data = pd.DataFrame({
            'Status': ['In Transit', 'Delivered', 'Processing', 'Delayed'],
            'Count': [156, 289, 45, 11]
        })

        fig = px.pie(
            status_data,
            values='Count',
            names='Status',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

    # Warehouse Status
    st.subheader("Warehouse Status")

    warehouse_data = pd.DataFrame({
        'Warehouse': ['DC_East', 'DC_West', 'DC_Central', 'DC_South'],
        'Utilization': [85, 78, 91, 72],
        'Inbound': [45, 32, 67, 28],
        'Outbound': [123, 89, 145, 67],
        'Alerts': [2, 0, 3, 1]
    })

    st.dataframe(warehouse_data, use_container_width=True)

    # Auto-refresh
    import time
    time.sleep(30)
    st.experimental_rerun()

# Run with: streamlit run dashboard.py
```

---

## Advanced Analytics Techniques

### ABC-XYZ Segmentation Analysis

```python
def abc_xyz_segmentation(sales_df):
    """
    Combined ABC-XYZ segmentation for inventory analytics

    ABC: Value classification (revenue contribution)
    XYZ: Variability classification (demand predictability)
    """

    # Calculate annual value per SKU
    sku_analysis = sales_df.groupby('sku').agg({
        'quantity': 'sum',
        'revenue': 'sum',
        'quantity': ['mean', 'std']
    })

    sku_analysis.columns = ['total_qty', 'total_revenue', 'avg_qty', 'std_qty']

    # ABC Classification (by revenue)
    sku_analysis = sku_analysis.sort_values('total_revenue', ascending=False)
    sku_analysis['cumulative_revenue_pct'] = (
        sku_analysis['total_revenue'].cumsum() /
        sku_analysis['total_revenue'].sum() * 100
    )

    def abc_class(pct):
        if pct <= 80:
            return 'A'
        elif pct <= 95:
            return 'B'
        else:
            return 'C'

    sku_analysis['abc'] = sku_analysis['cumulative_revenue_pct'].apply(abc_class)

    # XYZ Classification (by variability)
    sku_analysis['coefficient_of_variation'] = (
        sku_analysis['std_qty'] / sku_analysis['avg_qty']
    )

    def xyz_class(cv):
        if cv < 0.5:
            return 'X'  # Predictable
        elif cv < 1.0:
            return 'Y'  # Variable
        else:
            return 'Z'  # Erratic

    sku_analysis['xyz'] = sku_analysis['coefficient_of_variation'].apply(xyz_class)

    # Combined classification
    sku_analysis['segment'] = sku_analysis['abc'] + sku_analysis['xyz']

    # Strategy recommendations by segment
    strategy_map = {
        'AX': 'Tight control, daily review, high service level (99%)',
        'AY': 'Frequent review, safety stock, service level (98%)',
        'AZ': 'Close monitoring, high safety stock, service level (95%)',
        'BX': 'Weekly review, moderate safety stock, service level (97%)',
        'BY': 'Weekly review, standard policies, service level (95%)',
        'BZ': 'Bi-weekly review, higher safety stock, service level (90%)',
        'CX': 'Monthly review, min/max rules, service level (90%)',
        'CY': 'Monthly review, simple policies, service level (85%)',
        'CZ': 'Order on demand or don\'t stock, service level (80%)'
    }

    sku_analysis['strategy'] = sku_analysis['segment'].map(strategy_map)

    # Segment summary
    segment_summary = sku_analysis.groupby('segment').agg({
        'sku': 'count',
        'total_revenue': 'sum',
        'total_qty': 'sum'
    }).round(0)

    segment_summary['pct_skus'] = (
        segment_summary['sku'] / segment_summary['sku'].sum() * 100
    )
    segment_summary['pct_revenue'] = (
        segment_summary['total_revenue'] / segment_summary['total_revenue'].sum() * 100
    )

    return sku_analysis, segment_summary

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_abc_xyz(sku_analysis):
    """Create heatmap of ABC-XYZ segmentation"""

    # Create pivot table
    pivot = sku_analysis.pivot_table(
        values='sku',
        index='abc',
        columns='xyz',
        aggfunc='count',
        fill_value=0
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('ABC-XYZ Segmentation Matrix (SKU Count)')
    plt.ylabel('ABC Class (Value)')
    plt.xlabel('XYZ Class (Variability)')
    plt.show()
```

### Cost-to-Serve Analysis

```python
def cost_to_serve_analysis(orders_df, customers_df):
    """
    Calculate cost-to-serve by customer

    Helps identify profitable vs. unprofitable customers
    """

    # Join order and customer data
    analysis = orders_df.merge(customers_df, on='customer_id')

    # Calculate various cost components per customer
    customer_cts = analysis.groupby('customer_id').agg({
        # Revenue
        'order_value': 'sum',

        # Order processing costs
        'order_id': 'count',  # number of orders

        # Transportation costs
        'freight_cost': 'sum',

        # Warehouse costs (could be activity-based)
        'pick_cost': 'sum',
        'pack_cost': 'sum',

        # Returns costs
        'return_cost': 'sum',

        # Service costs
        'customer_service_calls': 'sum'
    })

    # Cost assumptions
    ORDER_PROCESSING_COST = 15  # $ per order
    CS_CALL_COST = 25  # $ per call

    # Calculate total costs
    customer_cts['order_processing_cost'] = (
        customer_cts['order_id'] * ORDER_PROCESSING_COST
    )
    customer_cts['customer_service_cost'] = (
        customer_cts['customer_service_calls'] * CS_CALL_COST
    )

    customer_cts['total_costs'] = (
        customer_cts['freight_cost'] +
        customer_cts['pick_cost'] +
        customer_cts['pack_cost'] +
        customer_cts['return_cost'] +
        customer_cts['order_processing_cost'] +
        customer_cts['customer_service_cost']
    )

    customer_cts['revenue'] = customer_cts['order_value']

    # Calculate profitability
    customer_cts['gross_margin'] = customer_cts['revenue'] * 0.35  # 35% margin
    customer_cts['net_profit'] = (
        customer_cts['gross_margin'] - customer_cts['total_costs']
    )
    customer_cts['profit_margin_pct'] = (
        customer_cts['net_profit'] / customer_cts['revenue'] * 100
    )

    # Cost-to-serve percentage
    customer_cts['cost_to_serve_pct'] = (
        customer_cts['total_costs'] / customer_cts['revenue'] * 100
    )

    # Classify customers
    def classify_customer(row):
        if row['profit_margin_pct'] > 10:
            return 'Profitable'
        elif row['profit_margin_pct'] > 0:
            return 'Marginally Profitable'
        else:
            return 'Unprofitable'

    customer_cts['classification'] = customer_cts.apply(classify_customer, axis=1)

    # Customer profitability quadrant
    # High Revenue + High Profit = Protect
    # High Revenue + Low Profit = Improve
    # Low Revenue + High Profit = Grow
    # Low Revenue + Low Profit = Rationalize

    revenue_median = customer_cts['revenue'].median()
    profit_median = customer_cts['net_profit'].median()

    def quadrant(row):
        if row['revenue'] > revenue_median and row['net_profit'] > profit_median:
            return 'Protect'
        elif row['revenue'] > revenue_median:
            return 'Improve'
        elif row['net_profit'] > profit_median:
            return 'Grow'
        else:
            return 'Rationalize'

    customer_cts['quadrant'] = customer_cts.apply(quadrant, axis=1)

    return customer_cts

# Visualization
def plot_cost_to_serve(customer_cts):
    """Visualize cost-to-serve analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Revenue vs. Cost scatter
    axes[0, 0].scatter(
        customer_cts['revenue'],
        customer_cts['total_costs'],
        c=customer_cts['profit_margin_pct'],
        cmap='RdYlGn',
        s=100,
        alpha=0.6
    )
    axes[0, 0].plot([0, customer_cts['revenue'].max()],
                    [0, customer_cts['revenue'].max()],
                    'r--', label='Break-even')
    axes[0, 0].set_xlabel('Revenue ($)')
    axes[0, 0].set_ylabel('Total Costs ($)')
    axes[0, 0].set_title('Revenue vs. Costs by Customer')
    axes[0, 0].legend()

    # 2. Cost-to-Serve Distribution
    customer_cts['cost_to_serve_pct'].hist(bins=30, ax=axes[0, 1])
    axes[0, 1].axvline(customer_cts['cost_to_serve_pct'].median(),
                      color='r', linestyle='--', label='Median')
    axes[0, 1].set_xlabel('Cost-to-Serve (%)')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Cost-to-Serve Distribution')
    axes[0, 1].legend()

    # 3. Profitability by Classification
    classification_summary = customer_cts.groupby('classification').agg({
        'customer_id': 'count',
        'revenue': 'sum',
        'net_profit': 'sum'
    })

    classification_summary.plot(kind='bar', y='net_profit', ax=axes[1, 0])
    axes[1, 0].set_title('Profit by Customer Classification')
    axes[1, 0].set_xlabel('Classification')
    axes[1, 0].set_ylabel('Net Profit ($)')

    # 4. Quadrant Analysis
    quadrant_counts = customer_cts['quadrant'].value_counts()
    axes[1, 1].pie(quadrant_counts.values, labels=quadrant_counts.index,
                   autopct='%1.1f%%')
    axes[1, 1].set_title('Customer Segmentation Quadrants')

    plt.tight_layout()
    plt.show()
```

### Pareto Analysis (80/20 Rule)

```python
def pareto_analysis(df, item_col, value_col, top_n=None):
    """
    Perform Pareto analysis (80/20 rule)

    Useful for identifying vital few vs. trivial many
    """

    # Aggregate by item
    pareto_df = df.groupby(item_col)[value_col].sum().reset_index()
    pareto_df = pareto_df.sort_values(value_col, ascending=False)

    # Calculate cumulative percentages
    total = pareto_df[value_col].sum()
    pareto_df['cumulative_value'] = pareto_df[value_col].cumsum()
    pareto_df['cumulative_pct'] = (pareto_df['cumulative_value'] / total) * 100
    pareto_df['pct_of_total'] = (pareto_df[value_col] / total) * 100

    # Find 80% threshold
    threshold_idx = (pareto_df['cumulative_pct'] <= 80).sum()
    pct_items_for_80 = (threshold_idx / len(pareto_df)) * 100

    print(f"Pareto Insight: {threshold_idx} items ({pct_items_for_80:.1f}%) "
          f"account for 80% of {value_col}")

    # Visualization
    if top_n is None:
        top_n = min(20, len(pareto_df))

    plot_df = pareto_df.head(top_n)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Bar chart
    ax1.bar(range(len(plot_df)), plot_df[value_col], color='steelblue', alpha=0.7)
    ax1.set_xlabel(item_col)
    ax1.set_ylabel(value_col, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Line chart for cumulative %
    ax2 = ax1.twinx()
    ax2.plot(range(len(plot_df)), plot_df['cumulative_pct'],
             color='red', marker='o', linewidth=2)
    ax2.axhline(y=80, color='green', linestyle='--', label='80% threshold')
    ax2.set_ylabel('Cumulative %', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 100)
    ax2.legend()

    plt.title(f'Pareto Analysis: Top {top_n} {item_col}')
    plt.xticks(range(len(plot_df)), plot_df[item_col], rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return pareto_df

# Example usage
pareto_results = pareto_analysis(sales_df, 'sku', 'revenue', top_n=20)
```

---

## Tools & Technologies

### Python Libraries

**Data Manipulation & Analysis:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `polars`: High-performance DataFrames (faster than pandas)
- `dask`: Parallel computing for large datasets

**Visualization:**
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualizations
- `plotly`: Interactive charts and dashboards
- `altair`: Declarative visualization

**Dashboard & BI:**
- `streamlit`: Quick web apps and dashboards
- `dash (Plotly)`: Enterprise dashboards
- `panel (HoloViz)`: Flexible dashboards
- `voila`: Jupyter notebooks as web apps

**Database & Data Warehouse:**
- `sqlalchemy`: SQL toolkit
- `psycopg2`: PostgreSQL adapter
- `pyodbc`: ODBC database connection
- `snowflake-connector-python`: Snowflake connection

**Statistical Analysis:**
- `scipy`: Scientific computing and statistics
- `statsmodels`: Statistical models
- `scikit-learn`: Machine learning metrics

### Business Intelligence Tools

**Commercial:**
- **Tableau**: Leading visualization and BI platform
- **Power BI (Microsoft)**: Integrated with Microsoft ecosystem
- **Qlik Sense**: Associative analytics engine
- **Looker (Google)**: Cloud-native BI
- **Domo**: Cloud-based BI platform
- **ThoughtSpot**: AI-powered analytics

**Open Source:**
- **Apache Superset**: Modern data exploration platform
- **Metabase**: Simple BI for everyone
- **Redash**: Connect and visualize data
- **Grafana**: Monitoring and observability dashboards

### Supply Chain Specific Platforms

**Enterprise:**
- **SAP Analytics Cloud**: Integrated with SAP S/4HANA
- **Oracle Analytics**: BI for Oracle ecosystem
- **Blue Yonder (JDA)**: Supply chain analytics suite
- **Kinaxis RapidResponse**: Concurrent planning platform
- **o9 Solutions**: Digital brain platform
- **LLamasoft**: Supply chain analytics (now Coupa)

**Specialized:**
- **Llamasoft Supply Chain Guru**: Network modeling and analytics
- **FourKites**: Real-time supply chain visibility
- **project44**: Supply chain visibility platform
- **Shippeo**: Transport visibility

---

## Data Integration & ETL

### Building Data Pipelines

```python
import pandas as pd
from sqlalchemy import create_engine
import logging

class SupplyChainETL:
    """
    Extract, Transform, Load pipeline for supply chain data
    """

    def __init__(self, source_configs, target_config):
        self.source_configs = source_configs
        self.target_config = target_config
        self.logger = logging.getLogger(__name__)

    def extract_from_sql(self, config):
        """Extract data from SQL database"""

        try:
            engine = create_engine(config['connection_string'])

            query = config.get('query') or f"SELECT * FROM {config['table']}"

            df = pd.read_sql(query, engine)

            self.logger.info(f"Extracted {len(df)} rows from {config['source']}")

            return df

        except Exception as e:
            self.logger.error(f"Error extracting from {config['source']}: {e}")
            raise

    def extract_from_api(self, config):
        """Extract data from REST API"""

        import requests

        try:
            response = requests.get(
                config['url'],
                headers=config.get('headers', {}),
                params=config.get('params', {})
            )

            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data[config['data_key']])

            self.logger.info(f"Extracted {len(df)} rows from {config['source']}")

            return df

        except Exception as e:
            self.logger.error(f"Error extracting from API {config['source']}: {e}")
            raise

    def transform_orders(self, orders_df):
        """Transform orders data"""

        # Data cleaning
        orders_df = orders_df.drop_duplicates(subset=['order_id'])

        # Data type conversions
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df['ship_date'] = pd.to_datetime(orders_df['ship_date'])
        orders_df['delivery_date'] = pd.to_datetime(orders_df['delivery_date'])

        # Derived fields
        orders_df['order_to_ship_days'] = (
            orders_df['ship_date'] - orders_df['order_date']
        ).dt.days

        orders_df['ship_to_delivery_days'] = (
            orders_df['delivery_date'] - orders_df['ship_date']
        ).dt.days

        orders_df['on_time'] = (
            orders_df['delivery_date'] <= orders_df['promised_date']
        ).astype(int)

        # Categorization
        orders_df['order_size_category'] = pd.cut(
            orders_df['order_value'],
            bins=[0, 100, 500, 1000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'XLarge']
        )

        # Data quality checks
        null_counts = orders_df.isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found: {null_counts[null_counts > 0]}")

        return orders_df

    def load_to_warehouse(self, df, table_name):
        """Load transformed data to data warehouse"""

        try:
            engine = create_engine(self.target_config['connection_string'])

            df.to_sql(
                table_name,
                engine,
                if_exists='replace',  # or 'append'
                index=False,
                chunksize=1000
            )

            self.logger.info(f"Loaded {len(df)} rows to {table_name}")

        except Exception as e:
            self.logger.error(f"Error loading to {table_name}: {e}")
            raise

    def run_pipeline(self):
        """Execute full ETL pipeline"""

        self.logger.info("Starting ETL pipeline...")

        # Extract
        orders_df = self.extract_from_sql(self.source_configs['orders'])
        shipments_df = self.extract_from_sql(self.source_configs['shipments'])
        inventory_df = self.extract_from_api(self.source_configs['inventory_api'])

        # Transform
        orders_clean = self.transform_orders(orders_df)

        # Combine/join datasets as needed
        # ... additional transformations

        # Load
        self.load_to_warehouse(orders_clean, 'fact_orders')
        self.load_to_warehouse(shipments_df, 'fact_shipments')
        self.load_to_warehouse(inventory_df, 'fact_inventory')

        self.logger.info("ETL pipeline completed successfully")

# Configuration
source_configs = {
    'orders': {
        'source': 'ERP',
        'connection_string': 'postgresql://user:pass@host:5432/erp_db',
        'table': 'orders'
    },
    'shipments': {
        'source': 'TMS',
        'connection_string': 'postgresql://user:pass@host:5432/tms_db',
        'table': 'shipments'
    },
    'inventory_api': {
        'source': 'WMS_API',
        'url': 'https://api.wms.com/v1/inventory',
        'headers': {'Authorization': 'Bearer token'},
        'data_key': 'inventory'
    }
}

target_config = {
    'connection_string': 'postgresql://user:pass@host:5432/analytics_dw'
}

# Run ETL
etl = SupplyChainETL(source_configs, target_config)
etl.run_pipeline()
```

---

## Common Challenges & Solutions

### Challenge: Data Quality Issues

**Problem:**
- Missing data, duplicates, inconsistencies
- Different data formats across systems
- Stale or inaccurate data

**Solutions:**
- Implement data quality checks in ETL pipeline
- Create data quality dashboards showing completeness, accuracy
- Establish data governance policies
- Use data profiling tools to identify issues
- Automate data validation rules

```python
def data_quality_report(df):
    """Generate data quality report"""

    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }

    # Missing values
    missing = df.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    report['completeness_pct'] = (1 - df.isnull().sum().sum() /
                                   (len(df) * len(df.columns))) * 100

    # Duplicates
    report['duplicate_rows'] = df.duplicated().sum()

    # Data types
    report['data_types'] = df.dtypes.value_counts().to_dict()

    # Outliers (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers[col] = ((df[col] < (Q1 - 1.5 * IQR)) |
                        (df[col] > (Q3 + 1.5 * IQR))).sum()
    report['outliers'] = outliers

    return report
```

### Challenge: Siloed Data Systems

**Problem:**
- Data scattered across ERP, WMS, TMS, etc.
- No single source of truth
- Manual data consolidation

**Solutions:**
- Build centralized data warehouse or data lake
- Implement master data management (MDM)
- Use ETL/ELT tools (Apache Airflow, Fivetran, Stitch)
- Create data integration layer (API gateway)
- Standardize data models across systems

### Challenge: Real-Time Analytics Requirements

**Problem:**
- Batch processing too slow for operational decisions
- Need for up-to-the-minute visibility

**Solutions:**
- Implement streaming data pipelines (Apache Kafka, AWS Kinesis)
- Use real-time databases (Redis, TimescaleDB)
- Create operational dashboards with auto-refresh
- Set up event-driven architecture
- Use change data capture (CDC) for database synchronization

### Challenge: Actionable Insights from Data

**Problem:**
- Data available but not actionable
- Overwhelming amount of metrics
- Difficulty identifying root causes

**Solutions:**
- Focus on leading vs. lagging indicators
- Implement automated alerting for exceptions
- Use drill-down capabilities in dashboards
- Apply root cause analysis methodologies
- Create action-oriented reports (not just descriptive stats)

---

## Output Format

### Analytics Report Template

**Executive Summary:**
- Key findings (3-5 bullet points)
- Critical issues requiring attention
- Improvement opportunities
- Recommended actions

**Performance Overview:**

| KPI | Current | Target | Last Period | YoY Change | Status |
|-----|---------|--------|-------------|------------|--------|
| Perfect Order Rate | 94.2% | 95% | 93.5% | +2.1% | ⚠️ |
| On-Time Delivery | 96.8% | 98% | 95.2% | +3.4% | ⚠️ |
| Inventory Turns | 6.2x | 7.0x | 5.8x | +6.9% | ⚠️ |
| Order Cycle Time | 2.1 days | 2.0 days | 2.3 days | -8.7% | ✅ |
| Freight Cost/Revenue | 8.2% | 7.5% | 8.5% | -3.5% | ⚠️ |

**Trend Analysis:**
- Time series charts for key metrics
- Seasonality patterns identified
- Anomaly detection highlights

**Root Cause Analysis:**
- Deep dive into underperforming areas
- Pareto analysis of issues
- Correlation analysis

**Recommendations:**
1. **Immediate Actions** (0-30 days)
   - Specific, actionable items
   - Owner and deadline

2. **Short-Term Initiatives** (1-3 months)
   - Process improvements
   - System enhancements

3. **Strategic Programs** (3-12 months)
   - Long-term transformations
   - Investment requirements

**Appendix:**
- Detailed methodology
- Data sources and quality notes
- Assumptions and calculations

---

## Questions to Ask

If you need more context:
1. What supply chain processes need analytics? (procurement, inventory, transportation, etc.)
2. Who will use these analytics? (executives, planners, analysts)
3. What are the key business questions you need answered?
4. What data sources are available? (ERP, WMS, TMS, spreadsheets)
5. What's the current state of analytics capabilities?
6. What tools are you using or prefer? (Tableau, Power BI, Python)
7. What are the critical KPIs you track today?
8. What reporting frequency is needed? (real-time, daily, weekly, monthly)

---

## Related Skills

- **demand-forecasting**: For predictive analytics and forecasting models
- **optimization-modeling**: For prescriptive analytics and optimization
- **ml-supply-chain**: For advanced machine learning applications
- **prescriptive-analytics**: For decision support and recommendations
- **digital-twin-modeling**: For simulation and scenario analysis
- **inventory-optimization**: For inventory-specific KPIs and analysis
- **network-design**: For network performance analytics
- **freight-optimization**: For transportation analytics


---
name: supplier-selection
description: When the user wants to evaluate suppliers, select vendors, or perform supplier scoring and qualification. Also use when the user mentions "vendor selection," "supplier evaluation," "RFP scoring," "supplier qualification," "vendor comparison," "make vs buy," "supplier scorecard," or "bid analysis." For ongoing supplier risk monitoring, see supplier-risk-management. For contract negotiation, see contract-management.
---

# Supplier Selection

You are an expert in supplier selection and vendor evaluation. Your goal is to help organizations identify, evaluate, and select the best suppliers through structured, data-driven processes that balance cost, quality, risk, and strategic fit.

## Initial Assessment

Before starting supplier selection, understand:

1. **Sourcing Context**
   - What category/commodity is being sourced?
   - Spend volume and strategic importance?
   - Current supplier situation? (single, multi, new sourcing)
   - Urgency of selection decision?

2. **Business Requirements**
   - Critical requirements? (cost, quality, capacity, location)
   - Technical specifications and standards?
   - Volume requirements and growth plans?
   - Service level expectations?

3. **Selection Criteria**
   - Key evaluation factors? (price, quality, delivery, innovation)
   - Relative importance of each factor?
   - Must-have vs. nice-to-have requirements?
   - Deal-breakers or knockout criteria?

4. **Process & Timeline**
   - RFP/RFQ/RFI process requirements?
   - Number of suppliers to evaluate?
   - Decision makers and stakeholders?
   - Selection timeline and go-live date?

---

## Supplier Selection Framework

### Selection Process Stages

**1. Need Identification**
- Define requirements and specifications
- Estimate demand volumes
- Determine sourcing strategy
- Build business case

**2. Market Research**
- Identify potential suppliers
- Conduct market analysis
- Assess supply market dynamics
- Benchmark pricing and terms

**3. Supplier Pre-Qualification**
- Screen for basic requirements
- Verify financial stability
- Check certifications and compliance
- Assess capacity and capability

**4. RFx Development & Issuance**
- Create RFP/RFQ/RFI documents
- Define evaluation criteria
- Issue to qualified suppliers
- Manage Q&A process

**5. Proposal Evaluation**
- Score supplier responses
- Conduct technical evaluations
- Perform cost analysis
- Site visits and audits

**6. Negotiation**
- Discuss terms and conditions
- Negotiate pricing and volumes
- Finalize service levels
- Address contingencies

**7. Final Selection**
- Make recommendation
- Obtain approvals
- Award contract
- Transition planning

---

## Supplier Evaluation Methods

### Weighted Scoring Model

**Most Common Approach:**
- Define evaluation criteria
- Assign weights based on importance
- Score each supplier on each criterion
- Calculate weighted total score

```python
import pandas as pd
import numpy as np

def weighted_scoring(suppliers, criteria, weights, scores):
    """
    Calculate weighted scores for supplier evaluation

    Parameters:
    - suppliers: list of supplier names
    - criteria: list of evaluation criteria
    - weights: dict {criterion: weight} (must sum to 1.0)
    - scores: dict {(supplier, criterion): score} (0-10 scale)

    Returns:
    - DataFrame with scores and rankings
    """

    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight}, normalizing...")
        weights = {k: v/total_weight for k, v in weights.items()}

    results = []

    for supplier in suppliers:
        weighted_score = 0
        criterion_scores = {}

        for criterion in criteria:
            score = scores.get((supplier, criterion), 0)
            weight = weights.get(criterion, 0)
            weighted_value = score * weight

            criterion_scores[criterion] = score
            weighted_score += weighted_value

        results.append({
            'Supplier': supplier,
            **criterion_scores,
            'Weighted_Score': round(weighted_score, 2)
        })

    # Create DataFrame and rank
    df = pd.DataFrame(results)
    df['Rank'] = df['Weighted_Score'].rank(ascending=False, method='min')
    df = df.sort_values('Weighted_Score', ascending=False)

    return df


# Example usage
suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C']

criteria = ['Price', 'Quality', 'Delivery', 'Service', 'Innovation']

weights = {
    'Price': 0.30,
    'Quality': 0.25,
    'Delivery': 0.20,
    'Service': 0.15,
    'Innovation': 0.10
}

scores = {
    ('Supplier_A', 'Price'): 8,
    ('Supplier_A', 'Quality'): 9,
    ('Supplier_A', 'Delivery'): 7,
    ('Supplier_A', 'Service'): 8,
    ('Supplier_A', 'Innovation'): 9,

    ('Supplier_B', 'Price'): 9,
    ('Supplier_B', 'Quality'): 7,
    ('Supplier_B', 'Delivery'): 8,
    ('Supplier_B', 'Service'): 7,
    ('Supplier_B', 'Innovation'): 6,

    ('Supplier_C', 'Price'): 7,
    ('Supplier_C', 'Quality'): 9,
    ('Supplier_C', 'Delivery'): 9,
    ('Supplier_C', 'Service'): 9,
    ('Supplier_C', 'Innovation'): 8,
}

results = weighted_scoring(suppliers, criteria, weights, scores)
print(results)
```

### Total Cost of Ownership (TCO) Analysis

**Beyond Price:**
- Purchase price
- Transportation and logistics
- Quality costs (defects, returns)
- Inventory carrying costs
- Administrative costs
- Risk and disruption costs

```python
class TCOCalculator:
    """Total Cost of Ownership calculator for supplier comparison"""

    def __init__(self, supplier_name):
        self.supplier_name = supplier_name
        self.costs = {}

    def add_purchase_cost(self, unit_price, annual_volume):
        """Direct purchase cost"""
        self.costs['purchase'] = unit_price * annual_volume
        return self

    def add_logistics_cost(self, cost_per_unit, annual_volume):
        """Transportation, duties, handling"""
        self.costs['logistics'] = cost_per_unit * annual_volume
        return self

    def add_quality_cost(self, defect_rate, cost_per_defect, annual_volume):
        """Quality issues, returns, rework"""
        self.costs['quality'] = defect_rate * cost_per_defect * annual_volume
        return self

    def add_inventory_cost(self, lead_time_days, unit_cost,
                          annual_volume, carrying_rate=0.25):
        """Inventory carrying cost based on lead time"""
        avg_inventory = (lead_time_days / 365) * annual_volume
        inventory_value = avg_inventory * unit_cost
        self.costs['inventory'] = inventory_value * carrying_rate
        return self

    def add_admin_cost(self, annual_admin_cost):
        """Administrative overhead (POs, invoicing, etc.)"""
        self.costs['admin'] = annual_admin_cost
        return self

    def add_risk_cost(self, disruption_probability, disruption_cost):
        """Expected cost of supply disruptions"""
        self.costs['risk'] = disruption_probability * disruption_cost
        return self

    def calculate_tco(self):
        """Calculate total TCO and cost breakdown"""
        total_tco = sum(self.costs.values())

        return {
            'supplier': self.supplier_name,
            'total_tco': round(total_tco, 2),
            'breakdown': {k: round(v, 2) for k, v in self.costs.items()},
            'breakdown_pct': {
                k: round(v/total_tco*100, 1)
                for k, v in self.costs.items()
            }
        }


# Example: Compare two suppliers
annual_volume = 100000  # units

# Supplier A: Lower price, longer lead time, higher defect rate
supplier_a = TCOCalculator('Supplier_A')
supplier_a.add_purchase_cost(unit_price=10.00, annual_volume=annual_volume)
supplier_a.add_logistics_cost(cost_per_unit=1.50, annual_volume=annual_volume)
supplier_a.add_quality_cost(defect_rate=0.02, cost_per_defect=50, annual_volume=annual_volume)
supplier_a.add_inventory_cost(lead_time_days=45, unit_cost=10.00, annual_volume=annual_volume)
supplier_a.add_admin_cost(annual_admin_cost=25000)
supplier_a.add_risk_cost(disruption_probability=0.10, disruption_cost=200000)

tco_a = supplier_a.calculate_tco()

# Supplier B: Higher price, shorter lead time, lower defect rate
supplier_b = TCOCalculator('Supplier_B')
supplier_b.add_purchase_cost(unit_price=10.50, annual_volume=annual_volume)
supplier_b.add_logistics_cost(cost_per_unit=1.00, annual_volume=annual_volume)
supplier_b.add_quality_cost(defect_rate=0.005, cost_per_defect=50, annual_volume=annual_volume)
supplier_b.add_inventory_cost(lead_time_days=21, unit_cost=10.50, annual_volume=annual_volume)
supplier_b.add_admin_cost(annual_admin_cost=20000)
supplier_b.add_risk_cost(disruption_probability=0.03, disruption_cost=200000)

tco_b = supplier_b.calculate_tco()

# Compare
print(f"\n{tco_a['supplier']} TCO: ${tco_a['total_tco']:,.0f}")
print(f"{tco_b['supplier']} TCO: ${tco_b['total_tco']:,.0f}")
print(f"\nDifference: ${abs(tco_a['total_tco'] - tco_b['total_tco']):,.0f}")
```

### Analytical Hierarchy Process (AHP)

**For Complex Decisions:**
- Pairwise comparison of criteria
- Consistency checking
- Hierarchical decision structure
- Handles both quantitative and qualitative factors

```python
import numpy as np
from numpy.linalg import eig

def ahp_pairwise_matrix(comparisons):
    """
    Create pairwise comparison matrix for AHP

    comparisons: dict of tuples {(criterion_a, criterion_b): importance}
    importance scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
    """

    # Extract unique criteria
    criteria = set()
    for (a, b) in comparisons.keys():
        criteria.add(a)
        criteria.add(b)
    criteria = sorted(list(criteria))
    n = len(criteria)

    # Build matrix
    matrix = np.ones((n, n))

    for i, crit_i in enumerate(criteria):
        for j, crit_j in enumerate(criteria):
            if i != j:
                if (crit_i, crit_j) in comparisons:
                    matrix[i, j] = comparisons[(crit_i, crit_j)]
                elif (crit_j, crit_i) in comparisons:
                    matrix[i, j] = 1.0 / comparisons[(crit_j, crit_i)]

    return matrix, criteria


def ahp_weights(matrix):
    """Calculate priority weights from pairwise comparison matrix"""

    # Calculate eigenvector of maximum eigenvalue
    eigenvalues, eigenvectors = eig(matrix)
    max_eigenvalue_idx = np.argmax(eigenvalues.real)
    principal_eigenvector = eigenvectors[:, max_eigenvalue_idx].real

    # Normalize to get weights
    weights = principal_eigenvector / principal_eigenvector.sum()

    # Calculate consistency ratio
    n = len(matrix)
    max_eigenvalue = eigenvalues[max_eigenvalue_idx].real
    ci = (max_eigenvalue - n) / (n - 1)

    # Random index values
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_values.get(n, 1.49)

    cr = ci / ri if ri > 0 else 0

    return weights, cr


# Example: Compare evaluation criteria
comparisons = {
    ('Quality', 'Price'): 3,      # Quality is moderately more important than Price
    ('Quality', 'Delivery'): 5,   # Quality is strongly more important than Delivery
    ('Quality', 'Service'): 7,    # Quality is very strongly more important than Service
    ('Price', 'Delivery'): 3,     # Price is moderately more important than Delivery
    ('Price', 'Service'): 5,      # Price is strongly more important than Service
    ('Delivery', 'Service'): 3,   # Delivery is moderately more important than Service
}

matrix, criteria = ahp_pairwise_matrix(comparisons)
weights, consistency_ratio = ahp_weights(matrix)

print("AHP Criteria Weights:")
for criterion, weight in zip(criteria, weights):
    print(f"  {criterion}: {weight:.3f} ({weight*100:.1f}%)")

print(f"\nConsistency Ratio: {consistency_ratio:.3f}")
if consistency_ratio < 0.10:
    print("  ✓ Acceptable consistency (CR < 0.10)")
else:
    print("  ✗ Inconsistent judgments (CR >= 0.10), review needed")
```

---

## Supplier Qualification Criteria

### Financial Stability

**Key Metrics:**
- Credit rating (D&B, S&P)
- Years in business
- Annual revenue and growth
- Profit margins
- Debt-to-equity ratio
- Days sales outstanding (DSO)
- Working capital ratio

**Red Flags:**
- Recent bankruptcy or restructuring
- Declining revenues (>20% YoY)
- Negative cash flow
- High debt levels
- Frequent management turnover

```python
def assess_financial_health(financials):
    """
    Assess supplier financial health

    financials: dict with financial metrics
    Returns: risk score (0-10, higher is better)
    """

    score = 10.0
    flags = []

    # Years in business
    years = financials.get('years_in_business', 0)
    if years < 2:
        score -= 3
        flags.append("Limited operating history")
    elif years < 5:
        score -= 1

    # Revenue trend
    revenue_growth = financials.get('revenue_growth_yoy', 0)
    if revenue_growth < -0.20:
        score -= 2
        flags.append("Significant revenue decline")
    elif revenue_growth < 0:
        score -= 1

    # Profitability
    profit_margin = financials.get('profit_margin', 0)
    if profit_margin < 0:
        score -= 2
        flags.append("Unprofitable")
    elif profit_margin < 0.05:
        score -= 1

    # Liquidity
    current_ratio = financials.get('current_ratio', 0)
    if current_ratio < 1.0:
        score -= 2
        flags.append("Liquidity concerns")
    elif current_ratio < 1.5:
        score -= 0.5

    # Leverage
    debt_to_equity = financials.get('debt_to_equity', 0)
    if debt_to_equity > 2.0:
        score -= 1.5
        flags.append("High leverage")

    score = max(0, score)  # Floor at 0

    risk_level = 'Low' if score >= 7 else 'Medium' if score >= 4 else 'High'

    return {
        'score': round(score, 1),
        'risk_level': risk_level,
        'flags': flags
    }
```

### Operational Capability

**Assessment Areas:**
- Production capacity and utilization
- Technology and equipment
- Quality management systems (ISO 9001, etc.)
- Workforce skills and stability
- Process maturity
- Continuous improvement culture

**Evaluation Methods:**
- Site visits and audits
- Capability studies
- Reference checks
- Trial orders/samples

### Quality & Compliance

**Quality Indicators:**
- Certifications (ISO 9001, AS9100, IATF 16949, etc.)
- Defect rates (PPM)
- Process capability (Cpk)
- Quality management practices
- Testing and inspection procedures
- Corrective action processes

**Compliance Requirements:**
- Industry-specific regulations
- Safety standards (OSHA, etc.)
- Environmental (ISO 14001, RoHS, REACH)
- Social responsibility (SA8000)
- Conflict minerals (Dodd-Frank)
- Anti-bribery/corruption

### Delivery Performance

**Key Metrics:**
- On-time delivery rate (OTIF)
- Lead time consistency
- Order fill rate
- Flexibility and responsiveness
- Minimum order quantities
- Geographic coverage

### Innovation & Technology

**Evaluation Factors:**
- R&D investment
- Patent portfolio
- Technology roadmap
- Digital capabilities
- Collaboration on new products
- Industry leadership

---

## RFP/RFQ Process

### RFP Development Best Practices

**Document Structure:**

1. **Introduction & Overview**
   - Company background
   - Purpose and scope
   - Timeline and process
   - Contact information

2. **Requirements Specification**
   - Technical specifications
   - Volume requirements
   - Quality standards
   - Delivery requirements
   - Packaging and labeling

3. **Commercial Terms**
   - Pricing format (unit, volume tiers)
   - Payment terms
   - Incoterms
   - Contract duration
   - Price adjustment mechanisms

4. **Evaluation Criteria**
   - Weighted scoring factors
   - Must-have requirements
   - Preferred qualifications
   - Evaluation process

5. **Supplier Information Required**
   - Company profile
   - Financial statements
   - References
   - Certifications
   - Insurance coverage
   - Quality management

6. **Instructions to Bidders**
   - Submission format
   - Deadline
   - Q&A process
   - Confidentiality
   - Conditions and disclaimers

### RFP Response Scoring

```python
import pandas as pd

class RFPScorer:
    """Automated RFP response scoring system"""

    def __init__(self, criteria_weights):
        """
        Initialize with weighted criteria

        criteria_weights: dict {category: {criterion: weight}}
        """
        self.criteria_weights = criteria_weights
        self.supplier_scores = {}

    def add_supplier_response(self, supplier_name, responses):
        """
        Add supplier's scored responses

        responses: dict {category: {criterion: score}}
        scores on 0-10 scale
        """
        self.supplier_scores[supplier_name] = responses

    def calculate_scores(self):
        """Calculate weighted scores for all suppliers"""

        results = []

        for supplier, responses in self.supplier_scores.items():
            category_scores = {}
            total_weighted_score = 0
            total_weight = 0

            for category, criteria in self.criteria_weights.items():
                category_score = 0
                category_weight = sum(criteria.values())

                for criterion, weight in criteria.items():
                    score = responses.get(category, {}).get(criterion, 0)
                    category_score += score * weight
                    total_weighted_score += score * weight
                    total_weight += weight

                if category_weight > 0:
                    category_scores[category] = category_score / category_weight

            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0

            results.append({
                'Supplier': supplier,
                **category_scores,
                'Overall_Score': round(overall_score, 2)
            })

        df = pd.DataFrame(results)
        df['Rank'] = df['Overall_Score'].rank(ascending=False, method='min')
        df = df.sort_values('Overall_Score', ascending=False)

        return df

    def generate_report(self):
        """Generate detailed scoring report"""
        df = self.calculate_scores()

        report = []
        report.append("=" * 80)
        report.append("RFP EVALUATION SUMMARY")
        report.append("=" * 80)
        report.append("")

        for _, row in df.iterrows():
            report.append(f"Rank #{int(row['Rank'])}: {row['Supplier']}")
            report.append(f"  Overall Score: {row['Overall_Score']:.2f}/10")
            report.append("")

        return "\n".join(report)


# Example usage
criteria = {
    'Technical': {
        'Specifications_Met': 0.15,
        'Quality_Certifications': 0.10,
        'Technical_Capability': 0.10
    },
    'Commercial': {
        'Price_Competitiveness': 0.20,
        'Payment_Terms': 0.05,
        'Volume_Flexibility': 0.05
    },
    'Operational': {
        'Lead_Time': 0.10,
        'Capacity': 0.08,
        'Geographic_Location': 0.07
    },
    'Strategic': {
        'Innovation': 0.05,
        'Sustainability': 0.03,
        'References': 0.02
    }
}

scorer = RFPScorer(criteria)

# Add supplier responses
scorer.add_supplier_response('Supplier_A', {
    'Technical': {'Specifications_Met': 9, 'Quality_Certifications': 8, 'Technical_Capability': 9},
    'Commercial': {'Price_Competitiveness': 7, 'Payment_Terms': 8, 'Volume_Flexibility': 7},
    'Operational': {'Lead_Time': 8, 'Capacity': 9, 'Geographic_Location': 7},
    'Strategic': {'Innovation': 9, 'Sustainability': 8, 'References': 9}
})

scorer.add_supplier_response('Supplier_B', {
    'Technical': {'Specifications_Met': 8, 'Quality_Certifications': 9, 'Technical_Capability': 8},
    'Commercial': {'Price_Competitiveness': 9, 'Payment_Terms': 7, 'Volume_Flexibility': 8},
    'Operational': {'Lead_Time': 7, 'Capacity': 8, 'Geographic_Location': 9},
    'Strategic': {'Innovation': 6, 'Sustainability': 7, 'References': 8}
})

results_df = scorer.calculate_scores()
print(results_df)
print("\n" + scorer.generate_report())
```

---

## Advanced Selection Techniques

### Multi-Attribute Utility Theory (MAUT)

**For Complex Trade-offs:**
- Convert attributes to common utility scale
- Handle non-linear preferences
- Risk attitudes incorporated

```python
def utility_function(value, min_val, max_val, risk_aversion=0.5):
    """
    Calculate utility for a value

    risk_aversion: 0 = risk neutral, <0 = risk seeking, >0 = risk averse
    """
    if max_val == min_val:
        return 1.0

    normalized = (value - min_val) / (max_val - min_val)

    # Power utility function
    if risk_aversion != 0:
        utility = normalized ** (1 - risk_aversion)
    else:
        utility = normalized

    return max(0, min(1, utility))
```

### Monte Carlo Simulation for Uncertainty

**When There's Uncertainty in Scores:**
- Model score distributions
- Run thousands of scenarios
- Calculate probability of best choice

```python
import numpy as np

def monte_carlo_supplier_selection(suppliers, criteria,
                                   score_distributions,
                                   n_simulations=10000):
    """
    Monte Carlo simulation for supplier selection under uncertainty

    score_distributions: dict {(supplier, criterion): (mean, std)}
    """

    results = {supplier: 0 for supplier in suppliers}

    for _ in range(n_simulations):
        scores = {}

        for supplier in suppliers:
            weighted_score = 0

            for criterion, weight in criteria.items():
                mean, std = score_distributions.get((supplier, criterion), (5, 1))
                score = np.random.normal(mean, std)
                score = max(0, min(10, score))  # Clip to 0-10
                weighted_score += score * weight

            scores[supplier] = weighted_score

        # Winner of this simulation
        winner = max(scores, key=scores.get)
        results[winner] += 1

    # Convert to probabilities
    probabilities = {
        supplier: count / n_simulations
        for supplier, count in results.items()
    }

    return probabilities


# Example with uncertainty
suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C']
criteria = {'Price': 0.4, 'Quality': 0.4, 'Delivery': 0.2}

# (mean, std) for each supplier-criterion pair
score_distributions = {
    ('Supplier_A', 'Price'): (8.0, 0.5),
    ('Supplier_A', 'Quality'): (9.0, 0.8),
    ('Supplier_A', 'Delivery'): (7.0, 1.0),

    ('Supplier_B', 'Price'): (9.0, 1.2),
    ('Supplier_B', 'Quality'): (7.0, 0.6),
    ('Supplier_B', 'Delivery'): (8.0, 0.8),

    ('Supplier_C', 'Price'): (7.0, 0.8),
    ('Supplier_C', 'Quality'): (9.0, 0.5),
    ('Supplier_C', 'Delivery'): (9.0, 0.7),
}

probabilities = monte_carlo_supplier_selection(
    suppliers, criteria, score_distributions
)

print("Probability of being best choice:")
for supplier, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
    print(f"  {supplier}: {prob*100:.1f}%")
```

---

## Tools & Libraries

### Python Libraries

**Data Analysis & Scoring:**
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `scipy`: Scientific computing, optimization
- `scikit-learn`: Machine learning for predictive scoring

**Optimization:**
- `pulp`: Linear programming for multi-sourcing optimization
- `cvxpy`: Convex optimization
- `pyomo`: Mathematical optimization modeling

**Visualization:**
- `matplotlib`, `seaborn`: Statistical visualizations
- `plotly`: Interactive dashboards
- `networkx`: Supplier network visualization

### Commercial Software

**Sourcing Platforms:**
- **Coupa**: Source-to-pay platform
- **SAP Ariba**: Procurement and sourcing
- **Jaggaer**: Strategic sourcing suite
- **GEP SMART**: Unified procurement platform
- **Ivalua**: Source-to-pay solution
- **Zycus**: Source-to-pay suite

**Supplier Management:**
- **Dun & Bradstreet**: Supplier risk & financial data
- **RapidRatings**: Financial health ratings
- **EcoVadis**: Sustainability ratings
- **Resilinc**: Supply chain risk management

**Analytics:**
- **Tableau**, **Power BI**: Supplier analytics dashboards
- **Qlik**: Data visualization
- **ThoughtSpot**: Search & AI-driven analytics

---

## Common Challenges & Solutions

### Challenge: Lack of Objective Data

**Problem:**
- Limited supplier information
- Subjective evaluations
- Incomplete proposals

**Solutions:**
- Request verifiable data (certifications, test reports)
- Use industry benchmarks and standards
- Conduct site visits and audits
- Reference checks with existing customers
- Trial orders or pilot programs
- Third-party assessments (D&B, audits)

### Challenge: Price vs. Quality Trade-off

**Problem:**
- Lowest price often not the best value
- Difficult to quantify quality impact
- Stakeholder pressure on cost

**Solutions:**
- Use Total Cost of Ownership (TCO) analysis
- Quantify quality costs (defects, returns, downtime)
- Include risk and disruption costs
- Show long-term value vs. initial price
- Present multiple scenarios (low cost, balanced, premium)

### Challenge: Too Many Suppliers to Evaluate

**Problem:**
- RFP fatigue
- Resource constraints
- Time pressure

**Solutions:**
- Pre-qualification screening (knockout criteria)
- Two-stage process (RFI then RFP)
- Limit RFP to 3-5 qualified suppliers
- Use automated scoring for initial filtering
- Focus resources on top candidates

### Challenge: Single Source Dependency

**Problem:**
- Risk of supply disruption
- Limited negotiating power
- No backup option

**Solutions:**
- Dual sourcing strategy (70/30 split)
- Qualify backup suppliers
- Regional diversification
- Build inventory buffers
- Long-term agreements with protections
- Develop alternate specifications

### Challenge: Incumbent Advantage

**Problem:**
- Existing supplier has information advantage
- Switching costs and risks
- Relationship bias

**Solutions:**
- Level playing field in RFP (same info to all)
- Explicitly quantify switching costs
- Blind evaluation (anonymous proposals initially)
- Focus on future capabilities, not past
- Clear evaluation criteria upfront

### Challenge: Global Sourcing Complexity

**Problem:**
- Cultural and language barriers
- Time zones and communication
- Currency and payment terms
- Legal and compliance differences

**Solutions:**
- Use local representatives or agents
- Engage interpreters for technical discussions
- Standardize evaluation templates
- Legal review of international contracts
- Consider total landed cost (duties, freight)
- Factor in lead time and inventory impact

---

## Output Format

### Supplier Selection Report

**Executive Summary:**
- Recommended supplier(s) and rationale
- Key differentiators
- Total value and expected savings
- Implementation timeline and risks

**Evaluation Summary:**

| Rank | Supplier | Overall Score | Price | Quality | Delivery | Service | Total Cost (Annual) | Recommendation |
|------|----------|---------------|-------|---------|----------|---------|---------------------|----------------|
| 1 | Supplier B | 8.5 | 9 | 9 | 8 | 8 | $2.1M | Award 60% |
| 2 | Supplier C | 8.2 | 7 | 9 | 9 | 9 | $2.3M | Award 40% |
| 3 | Supplier A | 7.8 | 8 | 7 | 8 | 7 | $2.0M | Backup |

**Detailed Scoring:**

```
Supplier B: 8.5/10

Technical (35%): 8.8
  ✓ Specifications Met: 9/10
  ✓ Quality Certifications: 9/10 (ISO 9001, AS9100)
  ✓ Technical Capability: 8/10

Commercial (30%): 8.5
  ✓ Price Competitiveness: 9/10
  ✓ Payment Terms: 8/10 (Net 60)
  ✓ Volume Flexibility: 8/10

Operational (25%): 8.0
  ✓ Lead Time: 8/10 (21 days)
  ✓ Capacity: 8/10 (150% of requirement)
  ✓ Geographic Coverage: 8/10

Strategic (10%): 9.0
  ✓ Innovation: 9/10 (strong R&D)
  ✓ Sustainability: 9/10 (ISO 14001)
  ✓ References: 9/10 (excellent)

Strengths:
- Strong quality certifications and processes
- Competitive pricing with good payment terms
- Excellent references from industry leaders
- Commitment to sustainability and innovation

Weaknesses:
- Moderate lead time (21 days vs. 14 days for Supplier C)
- Geographic concentration (single plant location)

Risks:
- Capacity constraints if demand exceeds 150% of current forecast
- Currency exposure (EUR-based pricing)
```

**Total Cost of Ownership Comparison:**

| Cost Component | Supplier A | Supplier B | Supplier C |
|----------------|------------|------------|------------|
| Purchase Price | $1,800,000 | $2,000,000 | $2,150,000 |
| Logistics | $150,000 | $100,000 | $80,000 |
| Quality Costs | $100,000 | $50,000 | $40,000 |
| Inventory Carrying | $80,000 | $60,000 | $50,000 |
| Administrative | $30,000 | $25,000 | $25,000 |
| Risk Premium | $40,000 | $20,000 | $15,000 |
| **Total TCO** | **$2,200,000** | **$2,255,000** | **$2,360,000** |

**Recommendation:**
- Award 60% to Supplier B (best balance of cost, quality, and capability)
- Award 40% to Supplier C (dual sourcing, best delivery and service)
- Qualify Supplier A as backup
- Implement quarterly performance reviews
- Re-bid in 2 years or if performance issues arise

**Implementation Plan:**
1. Contract negotiation and finalization (Weeks 1-2)
2. Purchase orders and production planning (Weeks 3-4)
3. First shipments and quality validation (Weeks 5-8)
4. Full production ramp-up (Weeks 9-12)
5. Performance review (Month 3)

---

## Questions to Ask

If you need more context:
1. What product/service category are you sourcing?
2. What's the annual spend volume and strategic importance?
3. What are the must-have requirements and evaluation criteria?
4. How many suppliers should be evaluated?
5. What's the timeline for supplier selection and go-live?
6. Are there incumbent suppliers or is this new sourcing?
7. What's the current pain point (cost, quality, risk, capacity)?
8. Any regulatory or compliance requirements?
9. Single source or multi-source strategy preferred?
10. What's the RFP process and who are the decision makers?

---

## Related Skills

- **strategic-sourcing**: For overall category sourcing strategy
- **procurement-optimization**: For optimal order allocation across suppliers
- **supplier-risk-management**: For ongoing supplier monitoring and risk assessment
- **contract-management**: For negotiating and managing supplier contracts
- **spend-analysis**: For category spend analysis and savings opportunities
- **quality-management**: For supplier quality audits and improvement


---
name: set-covering-problem
description: When the user wants to solve set covering problems, determine minimum coverage sets, or optimize facility coverage. Also use when the user mentions "set cover," "minimum set cover," "coverage optimization," "facility coverage problem," "service coverage," "location set covering," "maximal covering location problem," or "covering design." For general facility location, see facility-location-problem. For specific applications, see warehouse-location-optimization or hub-location-problem.
---

# Set Covering Problem

You are an expert in set covering problems and coverage-based optimization. Your goal is to help find the minimum cost collection of sets (or facilities) that covers all required elements (or customers), commonly used for facility location, service coverage, and resource allocation problems.

## Initial Assessment

Before solving set covering problems, understand:

1. **Problem Type**
   - Set Covering Problem (SCP)? (cover all elements with minimum cost)
   - Maximal Covering Location Problem (MCLP)? (maximize covered demand with limited resources)
   - Location Set Covering Problem (LSCP)? (minimum facilities for full coverage)
   - Partial Set Covering? (cover a percentage of elements)
   - Redundant Coverage? (elements covered multiple times)

2. **Coverage Requirements**
   - Must cover all elements? (100% coverage)
   - Partial coverage acceptable? (e.g., 95%)
   - Coverage distance/time threshold?
   - Redundancy requirements? (backup coverage)
   - Quality of coverage (single vs. multiple cover)?

3. **Elements to Cover**
   - What needs to be covered? (customers, demand points, areas)
   - How many elements?
   - Weights/priorities for elements?
   - Geographic locations?
   - Time-dependent coverage needs?

4. **Coverage Sets/Facilities**
   - How many potential covering sets/facilities?
   - Cost structure? (fixed costs, variable costs)
   - Coverage radius or service area?
   - Capacity constraints?
   - Can sets/facilities overlap in coverage?

5. **Objectives**
   - Minimize number of facilities?
   - Minimize total cost?
   - Maximize covered demand (with budget)?
   - Balance coverage and cost?
   - Ensure redundancy for reliability?

---

## Set Covering Problem Framework

### Problem Variants

**1. Basic Set Covering Problem (SCP)**
- **Goal**: Cover all elements with minimum cost
- **Constraint**: Every element must be covered at least once
- **Application**: Emergency service location, sensor placement

**2. Location Set Covering Problem (LSCP)**
- **Goal**: Minimize number of facilities for full coverage
- **Constraint**: All demand points within coverage distance
- **Application**: Fire station, ambulance location

**3. Maximal Covering Location Problem (MCLP)**
- **Goal**: Maximize demand covered with limited facilities
- **Constraint**: Can only open p facilities
- **Application**: Retail location with budget constraint

**4. Partial Set Covering**
- **Goal**: Minimize cost while covering at least α% of elements
- **Constraint**: α ≤ coverage ≤ 100%
- **Application**: Cost-effective service coverage

**5. Redundant Coverage (Backup Coverage)**
- **Goal**: Ensure elements covered by multiple facilities
- **Constraint**: Each element covered by at least k facilities
- **Application**: Reliable emergency response, fault-tolerant networks

---

## Mathematical Formulations

### Basic Set Covering Problem (SCP)

**Sets:**
- I = {1, ..., n}: Set of elements to be covered
- J = {1, ..., m}: Set of potential covering sets/facilities

**Parameters:**
- c_j: Cost of selecting set/facility j
- a_{ij}: Coverage coefficient (1 if set j covers element i, 0 otherwise)

**Decision Variables:**
- x_j ∈ {0,1}: 1 if set j is selected, 0 otherwise

**Objective Function:**
```
Minimize: Σ_{j=1}^m c_j × x_j
```

**Constraints:**
```
1. Coverage: Every element must be covered
   Σ_{j:a_{ij}=1} x_j ≥ 1,  ∀i ∈ I

   Or equivalently:
   Σ_{j=1}^m a_{ij} × x_j ≥ 1,  ∀i ∈ I

2. Binary variables:
   x_j ∈ {0,1},  ∀j ∈ J
```

**Complexity:** NP-complete

### Location Set Covering Problem (LSCP)

**Parameters:**
- d_{ij}: Distance from facility site j to demand point i
- S: Maximum service distance (coverage radius)

**Coverage Matrix:**
```
a_{ij} = 1  if  d_{ij} ≤ S
a_{ij} = 0  otherwise
```

**Objective:**
```
Minimize: Σ_{j=1}^m x_j  (minimize number of facilities)
```

**Constraints:**
```
Same coverage constraints as SCP
```

### Maximal Covering Location Problem (MCLP)

**Additional Parameters:**
- w_i: Weight/importance of demand point i (e.g., population, demand)
- p: Number of facilities to locate (budget constraint)

**Decision Variables:**
- x_j ∈ {0,1}: 1 if facility j is opened
- y_i ∈ {0,1}: 1 if demand point i is covered

**Objective Function:**
```
Maximize: Σ_{i=1}^n w_i × y_i  (maximize covered demand)
```

**Constraints:**
```
1. Coverage definition:
   y_i ≤ Σ_{j:a_{ij}=1} x_j,  ∀i ∈ I

2. Facility limit:
   Σ_{j=1}^m x_j ≤ p

3. Binary variables:
   x_j, y_i ∈ {0,1}
```

### Redundant Coverage (k-Coverage)

**Constraint:**
```
Each element must be covered by at least k facilities:
Σ_{j:a_{ij}=1} x_j ≥ k,  ∀i ∈ I
```

---

## Exact Solution Methods

### 1. Set Covering Problem with PuLP

```python
from pulp import *
import numpy as np

def solve_set_covering(costs, coverage_matrix, element_names=None,
                      set_names=None, redundancy=1):
    """
    Solve Set Covering Problem

    Args:
        costs: list of costs for each set/facility
        coverage_matrix: binary matrix [elements x sets]
                        coverage_matrix[i][j] = 1 if set j covers element i
        element_names: optional element names
        set_names: optional set names
        redundancy: coverage redundancy (k-coverage), default 1

    Returns:
        optimal solution
    """
    n_elements = len(coverage_matrix)
    n_sets = len(coverage_matrix[0]) if n_elements > 0 else 0

    if element_names is None:
        element_names = [f"Element_{i}" for i in range(n_elements)]

    if set_names is None:
        set_names = [f"Set_{j}" for j in range(n_sets)]

    # Create problem
    prob = LpProblem("Set_Covering", LpMinimize)

    # Decision variables: x[j] = 1 if set j is selected
    x = LpVariable.dicts("select", range(n_sets), cat='Binary')

    # Objective: Minimize total cost
    prob += lpSum([costs[j] * x[j] for j in range(n_sets)]), "Total_Cost"

    # Constraints: Each element covered at least 'redundancy' times
    for i in range(n_elements):
        prob += (
            lpSum([coverage_matrix[i][j] * x[j] for j in range(n_sets)]) >= redundancy,
            f"Coverage_{i}"
        )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        selected_sets = [j for j in range(n_sets) if x[j].varValue > 0.5]

        # Determine which sets cover each element
        element_coverage = {}
        for i in range(n_elements):
            covering_sets = [j for j in selected_sets
                           if coverage_matrix[i][j] == 1]
            element_coverage[i] = covering_sets

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'num_sets': len(selected_sets),
            'selected_sets': selected_sets,
            'selected_set_names': [set_names[j] for j in selected_sets],
            'element_coverage': element_coverage,
            'solve_time': solve_time,
            'redundancy': redundancy
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
if __name__ == "__main__":
    # Example: Emergency service coverage
    # 12 demand points, 8 potential facility locations

    # Costs to establish each facility
    costs = [100, 120, 90, 110, 105, 115, 95, 108]

    # Coverage matrix: coverage[i][j] = 1 if facility j covers demand i
    # Rows = demand points, Columns = facilities
    coverage_matrix = [
        [1, 1, 0, 0, 0, 0, 0, 0],  # Demand 0 covered by facilities 0, 1
        [1, 0, 1, 0, 0, 0, 0, 0],  # Demand 1 covered by facilities 0, 2
        [0, 1, 1, 1, 0, 0, 0, 0],  # Demand 2 covered by facilities 1, 2, 3
        [0, 0, 1, 1, 0, 0, 0, 0],  # Demand 3
        [0, 0, 0, 1, 1, 0, 0, 0],  # Demand 4
        [0, 0, 0, 0, 1, 1, 0, 0],  # Demand 5
        [0, 0, 0, 0, 1, 1, 1, 0],  # Demand 6
        [0, 0, 0, 0, 0, 1, 1, 1],  # Demand 7
        [0, 0, 0, 0, 0, 0, 1, 1],  # Demand 8
        [1, 0, 0, 0, 0, 0, 0, 1],  # Demand 9
        [1, 0, 0, 0, 0, 0, 1, 0],  # Demand 10
        [0, 1, 0, 0, 1, 0, 0, 0],  # Demand 11
    ]

    demand_names = [f"Demand_{i}" for i in range(12)]
    facility_names = [f"Facility_{i}" for i in range(8)]

    print("="*70)
    print("SET COVERING PROBLEM")
    print("="*70)
    print(f"Elements to cover: {len(coverage_matrix)}")
    print(f"Potential facilities: {len(costs)}")

    # Solve with single coverage
    result = solve_set_covering(costs, coverage_matrix,
                               demand_names, facility_names,
                               redundancy=1)

    print(f"\n{'='*70}")
    print(f"OPTIMAL SOLUTION (Single Coverage)")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:,.2f}")
    print(f"Facilities Selected: {result['num_sets']}")
    print(f"Facility IDs: {result['selected_sets']}")
    print(f"Facility Names: {result['selected_set_names']}")

    print(f"\nCoverage Details:")
    for i, covering_facilities in result['element_coverage'].items():
        print(f"  {demand_names[i]}: covered by facilities {covering_facilities}")

    print(f"\nSolve Time: {result['solve_time']:.2f} seconds")

    # Solve with redundant coverage (backup coverage)
    print(f"\n{'='*70}")
    print(f"REDUNDANT COVERAGE SOLUTION (k=2)")
    print(f"{'='*70}")

    result_redundant = solve_set_covering(costs, coverage_matrix,
                                         demand_names, facility_names,
                                         redundancy=2)

    print(f"Status: {result_redundant['status']}")
    print(f"Total Cost: ${result_redundant['total_cost']:,.2f}")
    print(f"Facilities Selected: {result_redundant['num_sets']}")
    print(f"Selected: {result_redundant['selected_set_names']}")

    print(f"\nCoverage Verification (each point covered ≥ 2 times):")
    for i, covering_facilities in result_redundant['element_coverage'].items():
        coverage_count = len(covering_facilities)
        status = "✓" if coverage_count >= 2 else "✗"
        print(f"  {demand_names[i]}: {status} {coverage_count} facilities "
              f"{covering_facilities}")
```

### 2. Location Set Covering Problem (LSCP)

```python
def solve_location_set_covering(facility_coords, demand_coords,
                               service_radius, facility_costs=None):
    """
    Solve Location Set Covering Problem

    Minimize number of facilities to cover all demand within service radius

    Args:
        facility_coords: array of potential facility coordinates
        demand_coords: array of demand point coordinates
        service_radius: maximum service distance
        facility_costs: optional costs (if None, minimize count)

    Returns:
        optimal facility locations
    """
    n_facilities = len(facility_coords)
    n_demands = len(demand_coords)

    # Calculate coverage matrix based on distances
    coverage_matrix = np.zeros((n_demands, n_facilities))

    for i in range(n_demands):
        for j in range(n_facilities):
            distance = np.linalg.norm(demand_coords[i] - facility_coords[j])
            if distance <= service_radius:
                coverage_matrix[i][j] = 1

    # If no costs provided, minimize number of facilities
    if facility_costs is None:
        facility_costs = [1] * n_facilities

    # Solve as set covering problem
    result = solve_set_covering(facility_costs, coverage_matrix)

    # Add distance information
    if result['status'] in ['Optimal', 'Feasible']:
        # Calculate average and max distance for each demand
        demand_distances = {}
        for i in range(n_demands):
            covering_facilities = result['element_coverage'][i]
            if covering_facilities:
                distances = [
                    np.linalg.norm(demand_coords[i] - facility_coords[j])
                    for j in covering_facilities
                ]
                demand_distances[i] = {
                    'min_distance': min(distances),
                    'covering_facilities': covering_facilities
                }

        result['demand_distances'] = demand_distances
        result['service_radius'] = service_radius

    return result


# Example usage
np.random.seed(42)

# Generate random coordinates
n_facilities = 15
n_demands = 30

facility_coords = np.random.rand(n_facilities, 2) * 100
demand_coords = np.random.rand(n_demands, 2) * 100

service_radius = 25  # Maximum service distance

print("\n" + "="*70)
print("LOCATION SET COVERING PROBLEM")
print("="*70)
print(f"Potential facilities: {n_facilities}")
print(f"Demand points: {n_demands}")
print(f"Service radius: {service_radius}")

result = solve_location_set_covering(facility_coords, demand_coords,
                                    service_radius)

print(f"\n{'='*70}")
print(f"OPTIMAL SOLUTION")
print(f"{'='*70}")
print(f"Status: {result['status']}")
print(f"Facilities Needed: {result['num_sets']}")
print(f"Facility IDs: {result['selected_sets']}")

print(f"\nService Statistics:")
all_min_distances = [d['min_distance'] for d in result['demand_distances'].values()]
print(f"  Average distance to nearest facility: {np.mean(all_min_distances):.2f}")
print(f"  Maximum distance to nearest facility: {np.max(all_min_distances):.2f}")
print(f"  All demands within service radius: {np.max(all_min_distances) <= service_radius}")
```

### 3. Maximal Covering Location Problem (MCLP)

```python
def solve_maximal_covering(facility_coords, demand_coords, demand_weights,
                          service_radius, max_facilities):
    """
    Solve Maximal Covering Location Problem

    Maximize covered demand with limited number of facilities

    Args:
        facility_coords: potential facility coordinates
        demand_coords: demand point coordinates
        demand_weights: demand weights (population, demand volume, etc.)
        service_radius: coverage radius
        max_facilities: maximum number of facilities to open

    Returns:
        optimal solution maximizing covered demand
    """
    n_facilities = len(facility_coords)
    n_demands = len(demand_coords)

    # Calculate coverage matrix
    coverage_matrix = np.zeros((n_demands, n_facilities))
    for i in range(n_demands):
        for j in range(n_facilities):
            distance = np.linalg.norm(demand_coords[i] - facility_coords[j])
            if distance <= service_radius:
                coverage_matrix[i][j] = 1

    # Create problem
    prob = LpProblem("Maximal_Covering", LpMaximize)

    # Decision variables
    x = LpVariable.dicts("facility", range(n_facilities), cat='Binary')
    y = LpVariable.dicts("covered", range(n_demands), cat='Binary')

    # Objective: Maximize total covered demand
    prob += (
        lpSum([demand_weights[i] * y[i] for i in range(n_demands)]),
        "Total_Covered_Demand"
    )

    # Constraints

    # 1. Coverage definition: demand covered only if within radius of open facility
    for i in range(n_demands):
        prob += (
            y[i] <= lpSum([coverage_matrix[i][j] * x[j]
                          for j in range(n_facilities)]),
            f"Coverage_{i}"
        )

    # 2. Facility limit
    prob += (
        lpSum([x[j] for j in range(n_facilities)]) <= max_facilities,
        "Facility_Limit"
    )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        open_facilities = [j for j in range(n_facilities)
                          if x[j].varValue > 0.5]

        covered_demands = [i for i in range(n_demands)
                          if y[i].varValue > 0.5]

        uncovered_demands = [i for i in range(n_demands)
                            if y[i].varValue < 0.5]

        total_demand = sum(demand_weights)
        covered_demand = sum(demand_weights[i] for i in covered_demands)
        coverage_percentage = (covered_demand / total_demand) * 100

        return {
            'status': LpStatus[prob.status],
            'covered_demand': covered_demand,
            'total_demand': total_demand,
            'coverage_percentage': coverage_percentage,
            'num_facilities': len(open_facilities),
            'max_facilities': max_facilities,
            'open_facilities': open_facilities,
            'covered_demands': covered_demands,
            'uncovered_demands': uncovered_demands,
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
demand_weights = np.random.uniform(50, 500, n_demands)  # Population or demand
max_facilities = 5  # Budget allows only 5 facilities

print("\n" + "="*70)
print("MAXIMAL COVERING LOCATION PROBLEM")
print("="*70)
print(f"Potential facilities: {n_facilities}")
print(f"Demand points: {n_demands}")
print(f"Total demand: {demand_weights.sum():,.2f}")
print(f"Maximum facilities: {max_facilities}")
print(f"Service radius: {service_radius}")

result = solve_maximal_covering(facility_coords, demand_coords,
                               demand_weights, service_radius,
                               max_facilities)

print(f"\n{'='*70}")
print(f"OPTIMAL SOLUTION")
print(f"{'='*70}")
print(f"Status: {result['status']}")
print(f"Facilities Opened: {result['num_facilities']} / {result['max_facilities']}")
print(f"Open Facility IDs: {result['open_facilities']}")
print(f"Covered Demand: {result['covered_demand']:,.2f} / {result['total_demand']:,.2f}")
print(f"Coverage: {result['coverage_percentage']:.2f}%")
print(f"Covered Demand Points: {len(result['covered_demands'])} / {n_demands}")
print(f"Uncovered Demand Points: {len(result['uncovered_demands'])}")

if result['uncovered_demands']:
    print(f"\nUncovered Demand Points:")
    for i in result['uncovered_demands'][:10]:  # Show first 10
        print(f"  Demand {i}: weight={demand_weights[i]:.2f}")
```

---

## Greedy Heuristics

### 1. Greedy Set Covering

```python
def greedy_set_covering(costs, coverage_matrix):
    """
    Greedy heuristic for set covering

    Iteratively select set with best cost-effectiveness ratio

    Args:
        costs: set costs
        coverage_matrix: coverage matrix

    Returns:
        heuristic solution
    """
    n_elements = len(coverage_matrix)
    n_sets = len(coverage_matrix[0])

    uncovered_elements = set(range(n_elements))
    selected_sets = []
    total_cost = 0

    while uncovered_elements:
        best_set = None
        best_ratio = float('inf')

        # Find set with best cost per newly covered element
        for j in range(n_sets):
            if j in selected_sets:
                continue

            # Count how many uncovered elements this set covers
            newly_covered = sum(1 for i in uncovered_elements
                              if coverage_matrix[i][j] == 1)

            if newly_covered > 0:
                ratio = costs[j] / newly_covered
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_set = j

        if best_set is None:
            # No set can cover remaining elements
            break

        # Select this set
        selected_sets.append(best_set)
        total_cost += costs[best_set]

        # Remove covered elements
        newly_covered_elements = {i for i in uncovered_elements
                                 if coverage_matrix[i][best_set] == 1}
        uncovered_elements -= newly_covered_elements

    return {
        'selected_sets': selected_sets,
        'num_sets': len(selected_sets),
        'total_cost': total_cost,
        'all_covered': len(uncovered_elements) == 0,
        'method': 'Greedy'
    }
```

### 2. Greedy Location for LSCP

```python
def greedy_location_covering(facility_coords, demand_coords, service_radius):
    """
    Greedy heuristic for location set covering

    Iteratively select facility that covers most uncovered demands

    Args:
        facility_coords: facility coordinates
        demand_coords: demand coordinates
        service_radius: service radius

    Returns:
        heuristic facility selection
    """
    n_facilities = len(facility_coords)
    n_demands = len(demand_coords)

    # Calculate coverage
    coverage = {}
    for j in range(n_facilities):
        coverage[j] = set()
        for i in range(n_demands):
            distance = np.linalg.norm(demand_coords[i] - facility_coords[j])
            if distance <= service_radius:
                coverage[j].add(i)

    uncovered_demands = set(range(n_demands))
    selected_facilities = []

    while uncovered_demands:
        # Select facility covering most uncovered demands
        best_facility = None
        max_new_coverage = 0

        for j in range(n_facilities):
            if j in selected_facilities:
                continue

            new_coverage = len(coverage[j] & uncovered_demands)
            if new_coverage > max_new_coverage:
                max_new_coverage = new_coverage
                best_facility = j

        if best_facility is None:
            # Cannot cover all demands
            break

        selected_facilities.append(best_facility)
        uncovered_demands -= coverage[best_facility]

    return {
        'selected_facilities': selected_facilities,
        'num_facilities': len(selected_facilities),
        'all_covered': len(uncovered_demands) == 0,
        'uncovered_count': len(uncovered_demands),
        'method': 'Greedy Location'
    }
```

---

## Complete Set Covering Solver

```python
class SetCoveringSolver:
    """
    Comprehensive Set Covering Problem Solver
    """

    def __init__(self):
        self.problem_type = None
        self.loaded = False

    def load_set_covering(self, costs, coverage_matrix,
                         element_names=None, set_names=None):
        """Load basic set covering problem"""
        self.costs = np.array(costs)
        self.coverage_matrix = np.array(coverage_matrix)
        self.element_names = element_names
        self.set_names = set_names
        self.problem_type = 'SCP'
        self.loaded = True

        print(f"Loaded Set Covering Problem:")
        print(f"  Elements: {len(coverage_matrix)}")
        print(f"  Sets: {len(costs)}")

    def load_location_covering(self, facility_coords, demand_coords,
                              service_radius, demand_weights=None):
        """Load location-based covering problem"""
        self.facility_coords = np.array(facility_coords)
        self.demand_coords = np.array(demand_coords)
        self.service_radius = service_radius
        self.demand_weights = demand_weights
        self.problem_type = 'LSCP'
        self.loaded = True

        print(f"Loaded Location Set Covering Problem:")
        print(f"  Facilities: {len(facility_coords)}")
        print(f"  Demands: {len(demand_coords)}")
        print(f"  Service radius: {service_radius}")

    def solve_exact(self, redundancy=1, max_facilities=None):
        """Solve with exact MIP"""
        if not self.loaded:
            raise ValueError("Problem not loaded")

        if self.problem_type == 'SCP':
            return solve_set_covering(self.costs, self.coverage_matrix,
                                     self.element_names, self.set_names,
                                     redundancy)

        elif self.problem_type == 'LSCP':
            if max_facilities:
                # Solve as MCLP
                weights = self.demand_weights if self.demand_weights is not None \
                         else np.ones(len(self.demand_coords))
                return solve_maximal_covering(
                    self.facility_coords, self.demand_coords,
                    weights, self.service_radius, max_facilities
                )
            else:
                # Solve as LSCP
                return solve_location_set_covering(
                    self.facility_coords, self.demand_coords,
                    self.service_radius
                )

    def solve_heuristic(self, method='greedy'):
        """Solve with heuristic"""
        if not self.loaded:
            raise ValueError("Problem not loaded")

        if method == 'greedy':
            if self.problem_type == 'SCP':
                return greedy_set_covering(self.costs, self.coverage_matrix)
            elif self.problem_type == 'LSCP':
                return greedy_location_covering(
                    self.facility_coords, self.demand_coords,
                    self.service_radius
                )

    def compare_methods(self, methods=['greedy', 'exact']):
        """Compare solution methods"""
        import pandas as pd

        results = []

        for method in methods:
            import time
            start = time.time()

            try:
                if method == 'exact':
                    sol = self.solve_exact()
                else:
                    sol = self.solve_heuristic(method)

                solve_time = time.time() - start

                results.append({
                    'Method': method,
                    'Sets/Facilities': sol.get('num_sets') or sol.get('num_facilities'),
                    'Cost': sol.get('total_cost', 'N/A'),
                    'Time (s)': f"{solve_time:.3f}"
                })

            except Exception as e:
                print(f"Error with {method}: {e}")

        return pd.DataFrame(results)

    def visualize_coverage(self, solution):
        """Visualize coverage solution"""
        if self.problem_type != 'LSCP':
            print("Visualization only for location problems")
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Plot demand points
        plt.scatter(self.demand_coords[:, 0], self.demand_coords[:, 1],
                   c='blue', s=50, alpha=0.6, label='Demand Points')

        # Plot all potential facilities (gray)
        plt.scatter(self.facility_coords[:, 0], self.facility_coords[:, 1],
                   c='lightgray', s=200, alpha=0.3, marker='s',
                   label='Potential Facilities')

        # Plot selected facilities (red)
        selected = solution.get('selected_sets') or solution.get('open_facilities') or \
                  solution.get('selected_facilities')

        if selected:
            selected_coords = self.facility_coords[selected]
            plt.scatter(selected_coords[:, 0], selected_coords[:, 1],
                       c='red', s=300, alpha=0.8, marker='s',
                       label='Selected Facilities', edgecolors='black', linewidths=2)

            # Draw coverage circles
            for idx in selected:
                circle = plt.Circle(
                    self.facility_coords[idx],
                    self.service_radius,
                    fill=False, edgecolor='red', alpha=0.3, linewidth=1
                )
                plt.gca().add_patch(circle)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Set Covering Solution - Facility Coverage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


# Complete example
if __name__ == "__main__":
    print("="*70)
    print("SET COVERING - COMPREHENSIVE EXAMPLE")
    print("="*70)

    # Example: Emergency service location
    np.random.seed(42)

    n_facilities = 20
    n_demands = 50

    facility_coords = np.random.rand(n_facilities, 2) * 100
    demand_coords = np.random.rand(n_demands, 2) * 100
    service_radius = 20

    # Create solver
    solver = SetCoveringSolver()
    solver.load_location_covering(facility_coords, demand_coords, service_radius)

    # Compare methods
    print("\n" + "="*70)
    print("COMPARING SOLUTION METHODS")
    print("="*70)

    comparison = solver.compare_methods(['greedy', 'exact'])
    print("\n" + comparison.to_string(index=False))

    # Detailed optimal solution
    print("\n" + "="*70)
    print("DETAILED OPTIMAL SOLUTION")
    print("="*70)

    optimal = solver.solve_exact()

    print(f"Facilities Needed: {optimal['num_sets']}")
    print(f"Facility IDs: {optimal['selected_sets']}")
    print(f"Average distance: {np.mean([d['min_distance'] for d in optimal['demand_distances'].values()]):.2f}")

    # Visualize
    solver.visualize_coverage(optimal)
```

---

## Tools & Libraries

**Python:**
- PuLP, Pyomo, OR-Tools
- NetworkX for graph-based coverage
- scikit-learn for clustering

**Applications:**
- Emergency services
- Retail location
- Sensor placement
- Network design

---

## Common Challenges & Solutions

**Large Problems:** Use greedy, column generation, Lagrangian relaxation

**Dynamic Coverage:** Update coverage sets, re-optimize periodically

**Probabilistic Demand:** Stochastic models, robust optimization

---

## Output Format

**Coverage Solution:**
- Sets Selected: X
- Total Cost: $Y
- Coverage: 100% (or Z%)
- Redundancy: k-coverage

---

## Questions to Ask

1. What needs to be covered?
2. Coverage requirements (100%, partial)?
3. Redundancy needed?
4. Budget constraints?
5. Coverage radius/distance?
6. Costs to consider?

---

## Related Skills

- **facility-location-problem**: General facility location
- **warehouse-location-optimization**: Warehouse coverage
- **hub-location-problem**: Hub coverage
- **network-flow-optimization**: Network-based coverage
- **optimization-modeling**: MIP formulation

---
name: shelf-life-management
description: When the user wants to manage product shelf life, implement FEFO (First-Expired-First-Out), optimize freshness, or handle perishable products. Also use when the user mentions "expiration management," "date code tracking," "FEFO," "freshness optimization," "waste reduction," "markdown management," or "spoilage prevention." For food supply chain, see food-beverage-supply-chain. For pharmaceutical expiry, see pharmaceutical-supply-chain.
---

# Shelf Life Management

You are an expert in shelf life management and perishable product supply chain optimization. Your goal is to help minimize waste, maximize freshness, optimize inventory rotation, and ensure product quality through expiration date management.

## Initial Assessment

Before implementing shelf life management, understand:

1. **Product Characteristics**
   - What products have shelf life concerns? (food, pharma, cosmetics)
   - What are the shelf lives? (days, weeks, months)
   - Storage requirements? (ambient, refrigerated, frozen)
   - Regulatory requirements? (FDA, USDA, EU regulations)
   - Date code format? (use-by, sell-by, best-before, manufacturing date)

2. **Current State**
   - Current waste/spoilage rate? (% of inventory)
   - Inventory rotation method? (FIFO, FEFO, manual)
   - Date code tracking capability? (WMS, manual)
   - Markdown/clearance process?
   - Customer complaints about freshness?

3. **Supply Chain Characteristics**
   - Lead times from production to shelf?
   - Number of nodes (plants, DCs, stores)?
   - Replenishment frequency?
   - Promotional activity impact?

4. **Business Impact**
   - Annual waste cost (spoilage + markdown)?
   - Lost sales from stockouts?
   - Customer satisfaction issues?
   - Compliance penalties or recalls?

---

## Shelf Life Management Framework

### Shelf Life Definitions

**Key Date Types:**

1. **Manufacturing Date**
   - When product was produced
   - Starting point for shelf life calculation

2. **Expiration Date / Use-By Date**
   - Last date product should be used/consumed
   - Safety concern (especially food, pharma)
   - Regulatory requirement

3. **Best-Before Date**
   - Quality date (not safety)
   - Product may still be safe but quality degrades
   - Common in food products

4. **Sell-By Date**
   - Last date retailer should sell product
   - Provides buffer before expiration
   - Typical: expiration date minus X days

**Remaining Shelf Life (RSL):**
```
RSL = Expiration Date - Current Date
RSL % = (Expiration Date - Current Date) / (Expiration Date - Manufacturing Date) × 100
```

### Shelf Life Zones

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ShelfLifeManager:
    """
    Manage shelf life and expiration dates
    """

    def __init__(self, shelf_life_days):
        self.shelf_life_days = shelf_life_days

        # Define shelf life zones
        self.zones = {
            'green': {'min_pct': 67, 'max_pct': 100, 'action': 'Normal sales'},
            'yellow': {'min_pct': 33, 'max_pct': 67, 'action': 'Priority sales'},
            'red': {'min_pct': 10, 'max_pct': 33, 'action': 'Markdown/clearance'},
            'expired': {'min_pct': 0, 'max_pct': 10, 'action': 'Pull from shelf'}
        }

    def calculate_rsl(self, manufacturing_date, current_date=None):
        """Calculate remaining shelf life"""

        if current_date is None:
            current_date = datetime.now()

        # Convert to datetime if strings
        if isinstance(manufacturing_date, str):
            manufacturing_date = pd.to_datetime(manufacturing_date)
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)

        expiration_date = manufacturing_date + timedelta(days=self.shelf_life_days)
        rsl_days = (expiration_date - current_date).days
        rsl_pct = (rsl_days / self.shelf_life_days) * 100

        return {
            'manufacturing_date': manufacturing_date,
            'expiration_date': expiration_date,
            'current_date': current_date,
            'rsl_days': max(0, rsl_days),
            'rsl_pct': max(0, rsl_pct),
            'expired': rsl_days <= 0
        }

    def classify_zone(self, rsl_pct):
        """Classify product into shelf life zone"""

        for zone_name, zone_info in self.zones.items():
            if zone_info['min_pct'] <= rsl_pct < zone_info['max_pct']:
                return {
                    'zone': zone_name,
                    'action': zone_info['action']
                }

        return {'zone': 'expired', 'action': 'Pull from shelf'}

    def generate_shelf_life_report(self, inventory_df):
        """
        Generate shelf life report for inventory

        Parameters:
        - inventory_df: DataFrame with columns ['sku', 'lot', 'manufacturing_date',
                       'quantity', 'location']

        Returns:
        - report with expiration analysis
        """

        current_date = datetime.now()

        # Calculate RSL for each lot
        inventory_df['rsl_info'] = inventory_df['manufacturing_date'].apply(
            lambda x: self.calculate_rsl(x, current_date)
        )

        # Extract RSL values
        inventory_df['rsl_days'] = inventory_df['rsl_info'].apply(lambda x: x['rsl_days'])
        inventory_df['rsl_pct'] = inventory_df['rsl_info'].apply(lambda x: x['rsl_pct'])
        inventory_df['expiration_date'] = inventory_df['rsl_info'].apply(
            lambda x: x['expiration_date']
        )
        inventory_df['expired'] = inventory_df['rsl_info'].apply(lambda x: x['expired'])

        # Classify zones
        inventory_df['zone_info'] = inventory_df['rsl_pct'].apply(self.classify_zone)
        inventory_df['zone'] = inventory_df['zone_info'].apply(lambda x: x['zone'])
        inventory_df['action'] = inventory_df['zone_info'].apply(lambda x: x['action'])

        # Summary by zone
        zone_summary = inventory_df.groupby('zone').agg({
            'quantity': 'sum',
            'lot': 'count'
        }).rename(columns={'lot': 'num_lots'})

        # Expiring soon (next 7 days)
        expiring_soon = inventory_df[
            (inventory_df['rsl_days'] <= 7) &
            (inventory_df['rsl_days'] > 0)
        ]

        # Expired inventory
        expired_inventory = inventory_df[inventory_df['expired'] == True]

        report = {
            'total_inventory': inventory_df['quantity'].sum(),
            'total_lots': len(inventory_df),
            'zone_summary': zone_summary,
            'expiring_soon_7days': {
                'quantity': expiring_soon['quantity'].sum(),
                'lots': len(expiring_soon),
                'details': expiring_soon[['sku', 'lot', 'quantity', 'rsl_days', 'location']]
            },
            'expired': {
                'quantity': expired_inventory['quantity'].sum(),
                'lots': len(expired_inventory),
                'details': expired_inventory[['sku', 'lot', 'quantity', 'expiration_date', 'location']]
            }
        }

        return report


# Example usage
manager = ShelfLifeManager(shelf_life_days=120)  # 120-day shelf life

inventory = pd.DataFrame({
    'sku': ['SKU_A', 'SKU_A', 'SKU_A', 'SKU_B', 'SKU_B'],
    'lot': ['LOT001', 'LOT002', 'LOT003', 'LOT004', 'LOT005'],
    'manufacturing_date': [
        datetime.now() - timedelta(days=100),  # Old
        datetime.now() - timedelta(days=60),   # Medium
        datetime.now() - timedelta(days=10),   # Fresh
        datetime.now() - timedelta(days=125),  # Expired
        datetime.now() - timedelta(days=80)    # Medium
    ],
    'quantity': [500, 1000, 1500, 200, 800],
    'location': ['DC1', 'DC1', 'DC2', 'DC1', 'DC2']
})

report = manager.generate_shelf_life_report(inventory)

print("Zone Summary:")
print(report['zone_summary'])
print(f"\nExpiring in 7 days: {report['expiring_soon_7days']['quantity']} units")
print(f"Expired: {report['expired']['quantity']} units")
```

---

## FEFO (First-Expired-First-Out) Implementation

### FEFO Allocation Logic

```python
class FEFOInventoryManager:
    """
    Implement FEFO (First-Expired-First-Out) inventory allocation
    """

    def __init__(self, inventory_df):
        """
        Initialize with inventory

        Parameters:
        - inventory_df: DataFrame with columns ['sku', 'lot', 'expiration_date',
                       'quantity', 'location']
        """
        self.inventory = inventory_df.copy()

    def allocate_order(self, sku, quantity_needed, location=None,
                        min_rsl_days=None):
        """
        Allocate inventory using FEFO logic

        Parameters:
        - sku: product SKU
        - quantity_needed: quantity to allocate
        - location: preferred location (None = any)
        - min_rsl_days: minimum remaining shelf life (customer requirement)

        Returns:
        - allocation list of lots
        """

        # Filter to SKU
        available = self.inventory[
            (self.inventory['sku'] == sku) &
            (self.inventory['quantity'] > 0)
        ].copy()

        # Filter by location if specified
        if location:
            available = available[available['location'] == location]

        # Filter by minimum RSL if specified
        if min_rsl_days:
            current_date = datetime.now()
            available = available[
                (available['expiration_date'] - current_date).dt.days >= min_rsl_days
            ]

        # Sort by expiration date (earliest first) - FEFO
        available = available.sort_values('expiration_date')

        # Allocate
        allocation = []
        remaining_need = quantity_needed

        for idx, row in available.iterrows():
            if remaining_need <= 0:
                break

            # Allocate from this lot
            allocate_qty = min(remaining_need, row['quantity'])

            allocation.append({
                'sku': sku,
                'lot': row['lot'],
                'location': row['location'],
                'expiration_date': row['expiration_date'],
                'quantity': allocate_qty,
                'rsl_days': (row['expiration_date'] - datetime.now()).days
            })

            # Update remaining need
            remaining_need -= allocate_qty

            # Update inventory
            self.inventory.loc[idx, 'quantity'] -= allocate_qty

        # Check if fully allocated
        allocated_qty = sum(a['quantity'] for a in allocation)
        shortage = quantity_needed - allocated_qty

        return {
            'allocated': allocation,
            'total_allocated': allocated_qty,
            'shortage': shortage,
            'fill_rate': allocated_qty / quantity_needed if quantity_needed > 0 else 0
        }

    def get_inventory_summary(self):
        """Get current inventory summary"""

        summary = self.inventory.groupby(['sku', 'location']).agg({
            'quantity': 'sum',
            'lot': 'count',
            'expiration_date': ['min', 'max']
        })

        return summary


# Example
inventory = pd.DataFrame({
    'sku': ['SKU_A', 'SKU_A', 'SKU_A', 'SKU_A'],
    'lot': ['LOT001', 'LOT002', 'LOT003', 'LOT004'],
    'expiration_date': pd.to_datetime([
        '2025-03-15',
        '2025-04-20',
        '2025-02-10',  # Oldest - should allocate first
        '2025-05-01'
    ]),
    'quantity': [500, 800, 300, 1000],
    'location': ['DC1', 'DC1', 'DC1', 'DC2']
})

fefo = FEFOInventoryManager(inventory)

# Allocate order
order = fefo.allocate_order(
    sku='SKU_A',
    quantity_needed=1000,
    location='DC1',
    min_rsl_days=30  # Customer requires 30 days min shelf life
)

print("Allocation:")
for alloc in order['allocated']:
    print(f"  Lot {alloc['lot']}: {alloc['quantity']} units, "
          f"RSL: {alloc['rsl_days']} days")

print(f"\nTotal Allocated: {order['total_allocated']}")
print(f"Shortage: {order['shortage']}")
```

---

## Waste Reduction Strategies

### Dynamic Markdown Optimization

```python
import numpy as np
from scipy.optimize import minimize_scalar

def optimize_markdown_timing(current_rsl_days, regular_price, cost,
                              demand_elasticity=-2.0):
    """
    Optimize when to markdown product to minimize waste

    Parameters:
    - current_rsl_days: remaining shelf life
    - regular_price: normal selling price
    - cost: product cost
    - demand_elasticity: price elasticity of demand

    Returns:
    - optimal markdown timing and price
    """

    def expected_profit(markdown_day):
        """Calculate expected profit if markdown starts on given day"""

        # Days at full price
        days_full_price = min(markdown_day, current_rsl_days)

        # Days at markdown price
        days_markdown = max(0, current_rsl_days - markdown_day)

        # Demand curves (simplified)
        daily_demand_full = 10  # Base demand at full price
        markdown_pct = min(0.5, days_markdown / current_rsl_days)  # Up to 50% off
        markdown_price = regular_price * (1 - markdown_pct)

        # Increased demand due to markdown
        demand_lift = (markdown_pct / 0.5) ** (-demand_elasticity)
        daily_demand_markdown = daily_demand_full * demand_lift

        # Total sales
        sales_full_price = days_full_price * daily_demand_full * regular_price
        sales_markdown = days_markdown * daily_demand_markdown * markdown_price

        # Costs
        units_sold = (days_full_price * daily_demand_full +
                     days_markdown * daily_demand_markdown)
        total_cost = units_sold * cost

        # Profit
        profit = sales_full_price + sales_markdown - total_cost

        # Penalty for waste (unsold inventory)
        # Assume some units don't sell even with markdown
        waste = max(0, 100 - units_sold)  # Assume started with 100 units
        waste_cost = waste * cost

        return profit - waste_cost

    # Optimize markdown day
    result = minimize_scalar(
        lambda x: -expected_profit(x),  # Negative for maximization
        bounds=(0, current_rsl_days),
        method='bounded'
    )

    optimal_day = int(result.x)
    optimal_profit = -result.fun

    # Calculate optimal markdown percentage
    markdown_pct = min(0.5, (current_rsl_days - optimal_day) / current_rsl_days)

    return {
        'optimal_markdown_day': optimal_day,
        'days_until_markdown': optimal_day,
        'markdown_pct': markdown_pct * 100,
        'markdown_price': regular_price * (1 - markdown_pct),
        'expected_profit': optimal_profit
    }


# Example
markdown_strategy = optimize_markdown_timing(
    current_rsl_days=30,
    regular_price=10.00,
    cost=6.00,
    demand_elasticity=-2.0
)

print(f"Start markdown in: {markdown_strategy['days_until_markdown']} days")
print(f"Markdown %: {markdown_strategy['markdown_pct']:.0f}%")
print(f"Markdown Price: ${markdown_strategy['markdown_price']:.2f}")
```

### Waste Tracking and Analysis

```python
class WasteAnalyzer:
    """
    Track and analyze waste from expiration
    """

    def __init__(self):
        self.waste_records = []

    def record_waste(self, waste_data):
        """Record waste event"""
        self.waste_records.append(waste_data)

    def analyze_waste(self):
        """Analyze waste patterns"""

        if not self.waste_records:
            return None

        df = pd.DataFrame(self.waste_records)

        analysis = {
            'total_waste_units': df['quantity'].sum(),
            'total_waste_value': (df['quantity'] * df['unit_cost']).sum(),
            'waste_by_sku': df.groupby('sku').agg({
                'quantity': 'sum',
                'unit_cost': lambda x: (df.loc[x.index, 'quantity'] * x).sum()
            }),
            'waste_by_location': df.groupby('location')['quantity'].sum(),
            'waste_by_reason': df.groupby('reason')['quantity'].sum(),
            'avg_rsl_at_waste': df['rsl_at_waste'].mean()
        }

        # Root cause analysis
        analysis['top_waste_skus'] = analysis['waste_by_sku'].nlargest(10, 'quantity')

        # Calculate waste rate
        if 'total_demand' in df.columns:
            analysis['waste_rate'] = (
                df['quantity'].sum() / df['total_demand'].sum() * 100
            )

        return analysis

    def identify_waste_drivers(self):
        """Identify key drivers of waste"""

        df = pd.DataFrame(self.waste_records)

        drivers = {}

        # 1. Overstocking
        overstock_waste = df[df['reason'] == 'overstock']
        drivers['overstock'] = {
            'waste_pct': len(overstock_waste) / len(df) * 100,
            'value': (overstock_waste['quantity'] * overstock_waste['unit_cost']).sum()
        }

        # 2. Long lead times
        long_lt_waste = df[df['lead_time_days'] > 14]
        drivers['long_lead_time'] = {
            'waste_pct': len(long_lt_waste) / len(df) * 100,
            'value': (long_lt_waste['quantity'] * long_lt_waste['unit_cost']).sum()
        }

        # 3. Poor forecasting
        forecast_error_waste = df[df['forecast_error_pct'].abs() > 20]
        drivers['forecast_error'] = {
            'waste_pct': len(forecast_error_waste) / len(df) * 100,
            'value': (forecast_error_waste['quantity'] *
                     forecast_error_waste['unit_cost']).sum()
        }

        # 4. Improper rotation (should be FEFO but wasn't)
        rotation_waste = df[df['reason'] == 'improper_rotation']
        drivers['improper_rotation'] = {
            'waste_pct': len(rotation_waste) / len(df) * 100,
            'value': (rotation_waste['quantity'] * rotation_waste['unit_cost']).sum()
        }

        return drivers


# Example
analyzer = WasteAnalyzer()

# Record waste events
analyzer.record_waste({
    'date': '2025-01-15',
    'sku': 'SKU_A',
    'location': 'DC1',
    'quantity': 100,
    'unit_cost': 5.00,
    'reason': 'overstock',
    'rsl_at_waste': 0,
    'lead_time_days': 21,
    'forecast_error_pct': 35,
    'total_demand': 500
})

analysis = analyzer.analyze_waste()
drivers = analyzer.identify_waste_drivers()

print(f"Total Waste Value: ${analysis['total_waste_value']:,.0f}")
print(f"Waste Rate: {analysis.get('waste_rate', 0):.1f}%")
print("\nWaste Drivers:")
for driver, data in drivers.items():
    print(f"  {driver}: {data['waste_pct']:.0f}% of waste, ${data['value']:,.0f}")
```

---

## Freshness Optimization

### Supplier Selection Based on Age

```python
def select_supplier_by_freshness(suppliers, demand, min_rsl_required):
    """
    Select suppliers to maximize freshness

    Parameters:
    - suppliers: list of suppliers with available product and RSL
    - demand: total demand to fulfill
    - min_rsl_required: minimum RSL acceptable

    Returns:
    - optimal supplier selection
    """

    from pulp import *

    # Create problem
    prob = LpProblem("Freshness_Optimization", LpMaximize)

    # Decision variables: quantity from each supplier
    x = LpVariable.dicts("Quantity",
                          [s['supplier_id'] for s in suppliers],
                          lowBound=0,
                          cat='Continuous')

    # Objective: Maximize weighted freshness
    # Higher RSL = better
    objective = lpSum([
        x[s['supplier_id']] * s['rsl_days']
        for s in suppliers
    ])

    prob += objective

    # Constraints

    # 1. Meet demand
    prob += lpSum([x[s['supplier_id']] for s in suppliers]) >= demand

    # 2. Supplier capacity
    for s in suppliers:
        prob += x[s['supplier_id']] <= s['available_quantity']

    # 3. Minimum RSL
    for s in suppliers:
        if s['rsl_days'] < min_rsl_required:
            prob += x[s['supplier_id']] == 0

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract results
    results = []
    for s in suppliers:
        qty = x[s['supplier_id']].varValue
        if qty > 0:
            results.append({
                'supplier': s['supplier_id'],
                'quantity': qty,
                'rsl_days': s['rsl_days'],
                'cost': qty * s['unit_cost']
            })

    total_qty = sum(r['quantity'] for r in results)
    weighted_rsl = sum(r['quantity'] * r['rsl_days'] for r in results) / total_qty

    return {
        'allocation': results,
        'total_quantity': total_qty,
        'weighted_avg_rsl': weighted_rsl,
        'total_cost': sum(r['cost'] for r in results)
    }


# Example
suppliers = [
    {
        'supplier_id': 'Supplier_A',
        'available_quantity': 500,
        'rsl_days': 90,
        'unit_cost': 5.00
    },
    {
        'supplier_id': 'Supplier_B',
        'available_quantity': 800,
        'rsl_days': 60,
        'unit_cost': 4.80
    },
    {
        'supplier_id': 'Supplier_C',
        'available_quantity': 400,
        'rsl_days': 120,  # Freshest
        'unit_cost': 5.20
    }
]

result = select_supplier_by_freshness(
    suppliers=suppliers,
    demand=1000,
    min_rsl_required=45
)

print("Supplier Allocation:")
for alloc in result['allocation']:
    print(f"  {alloc['supplier']}: {alloc['quantity']} units, "
          f"RSL: {alloc['rsl_days']} days")

print(f"\nWeighted Avg RSL: {result['weighted_avg_rsl']:.0f} days")
```

---

## Regulatory Compliance

### Date Code Management

```python
class DateCodeManager:
    """
    Manage date codes and regulatory compliance
    """

    def __init__(self, date_format='%Y%m%d'):
        self.date_format = date_format

    def parse_date_code(self, date_code, code_type='manufacturing'):
        """
        Parse date code to datetime

        Common formats:
        - YYYYMMDD: 20250115
        - YYMMDD: 250115
        - Julian: 25015 (year + day of year)
        """

        if len(date_code) == 8:  # YYYYMMDD
            return datetime.strptime(date_code, '%Y%m%d')
        elif len(date_code) == 6:  # YYMMDD
            return datetime.strptime(date_code, '%y%m%d')
        elif len(date_code) == 5:  # Julian YYDDD
            year = int('20' + date_code[:2])
            day_of_year = int(date_code[2:])
            return datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        else:
            raise ValueError(f"Unknown date code format: {date_code}")

    def validate_date_code(self, date_code, product_type='food'):
        """
        Validate date code meets regulatory requirements

        Requirements vary by region and product type
        """

        try:
            parsed_date = self.parse_date_code(date_code)
        except:
            return {'valid': False, 'reason': 'Invalid date code format'}

        current_date = datetime.now()

        # Check if manufacturing date is not in future
        if parsed_date > current_date:
            return {'valid': False, 'reason': 'Manufacturing date in future'}

        # Check if too old (product-specific)
        max_age_days = {
            'food_fresh': 30,
            'food_frozen': 365,
            'food_shelf_stable': 730,
            'pharma': 1825,  # 5 years typically
            'cosmetics': 730
        }

        age_days = (current_date - parsed_date).days
        max_age = max_age_days.get(product_type, 365)

        if age_days > max_age:
            return {
                'valid': False,
                'reason': f'Product too old: {age_days} days (max: {max_age})'
            }

        return {'valid': True, 'parsed_date': parsed_date, 'age_days': age_days}

    def calculate_expiration_date(self, manufacturing_date, shelf_life_days,
                                    sell_by_buffer_days=0):
        """
        Calculate expiration and sell-by dates

        Parameters:
        - manufacturing_date: when product was made
        - shelf_life_days: total shelf life
        - sell_by_buffer_days: days before expiration to stop selling

        Returns:
        - expiration_date, sell_by_date
        """

        if isinstance(manufacturing_date, str):
            manufacturing_date = self.parse_date_code(manufacturing_date)

        expiration_date = manufacturing_date + timedelta(days=shelf_life_days)
        sell_by_date = expiration_date - timedelta(days=sell_by_buffer_days)

        return {
            'manufacturing_date': manufacturing_date,
            'expiration_date': expiration_date,
            'sell_by_date': sell_by_date,
            'shelf_life_days': shelf_life_days
        }


# Example
manager = DateCodeManager()

# Parse date code
date_info = manager.parse_date_code('20250115')
print(f"Parsed Date: {date_info}")

# Validate
validation = manager.validate_date_code('20250115', product_type='food_shelf_stable')
print(f"Valid: {validation['valid']}")

# Calculate expiration
expiry = manager.calculate_expiration_date(
    manufacturing_date='20250115',
    shelf_life_days=180,
    sell_by_buffer_days=7
)

print(f"Expiration Date: {expiry['expiration_date']}")
print(f"Sell-By Date: {expiry['sell_by_date']}")
```

---

## Tools & Technologies

### Shelf Life Management Software

**Warehouse Management Systems (WMS) with FEFO:**
- **Manhattan Associates WMS**: Advanced FEFO and lot tracking
- **Blue Yonder WMS**: Shelf life management
- **SAP EWM**: Extended warehouse management with expiry
- **Oracle WMS**: Date code and FEFO support
- **HighJump WMS**: Perishables management

**Specialized Solutions:**
- **FoodLogiQ**: Food traceability and date code management
- **Trace Register**: Supply chain traceability
- **rfxcel**: Serialization and expiry tracking
- **FreshSurety**: Shelf life and temperature monitoring
- **ZestIOT**: Real-time freshness monitoring

**Markdown Optimization:**
- **Revionics**: Price and markdown optimization (Oracle)
- **Pricefx**: Dynamic pricing with expiry
- **PROS**: AI-driven markdown optimization

### Python Libraries

```python
# Date handling
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optimization
from pulp import *
from scipy.optimize import minimize, minimize_scalar

# Machine learning for forecasting
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

---

## Common Challenges & Solutions

### Challenge: High Waste Rate

**Problem:**
- 5-10% of inventory expires
- Significant cost impact
- Lost revenue

**Solutions:**
- Implement FEFO rigorously
- Reduce order quantities (more frequent orders)
- Improve demand forecasting
- Dynamic safety stock (reduce as expiration approaches)
- Markdown earlier and more aggressively
- Donate near-expiry (tax benefit, goodwill)

### Challenge: Inconsistent Date Code Formats

**Problem:**
- Suppliers use different formats
- Manual tracking error-prone
- Compliance risk

**Solutions:**
- Standardize date code format across suppliers
- Automated date code parsing (OCR, barcode)
- Validation at receiving
- WMS integration
- Master data management

### Challenge: Customer Freshness Requirements

**Problem:**
- Retailers require 75% minimum RSL
- Limits usable inventory
- Increases waste at DC

**Solutions:**
- Negotiate RSL requirements
- Price incentives for lower RSL
- Fast replenishment to stores
- Allocate fresher stock to demanding customers
- Use older stock for promotions

### Challenge: Multi-Echelon Complexity

**Problem:**
- DCs hold aging inventory
- Stores also have freshness requirements
- Difficult to optimize across network

**Solutions:**
- Network-wide visibility of RSL
- Centralized allocation (freshest to furthest)
- Dynamic routing based on expiry
- Cross-docking for fast movers
- DC bypass for fresh products

---

## Output Format

### Shelf Life Performance Report

**Executive Summary:**
- Total Inventory: 500,000 units
- Waste Rate: 3.2% (down from 5.1% last year)
- Waste Value: $320,000 annually
- Average RSL at Sale: 68%
- Compliance: 100% (no expired products sold)

**Expiration Summary:**

| Zone | Units | % of Total | Action Required |
|------|-------|------------|-----------------|
| Green (>67% RSL) | 350,000 | 70% | Normal sales |
| Yellow (33-67% RSL) | 100,000 | 20% | Priority outbound |
| Red (10-33% RSL) | 45,000 | 9% | Markdown now |
| Expired (<10% RSL) | 5,000 | 1% | Pull immediately |

**Expiring in Next 30 Days:**

| SKU | Location | Quantity | Exp Date | RSL Days | Action |
|-----|----------|----------|----------|----------|--------|
| SKU_A | DC1 | 2,500 | 2025-02-15 | 15 | 30% markdown |
| SKU_B | DC2 | 1,200 | 2025-02-10 | 10 | 50% markdown |
| SKU_C | DC1 | 800 | 2025-02-05 | 5 | Pull/donate |

**Waste Analysis:**

| Category | Waste Units | Value | % of Total Waste |
|----------|-------------|-------|------------------|
| Overstock | 8,000 | $160,000 | 50% |
| Forecast Error | 4,000 | $80,000 | 25% |
| Long Lead Time | 3,000 | $60,000 | 18.75% |
| Improper Rotation | 1,000 | $20,000 | 6.25% |

**Recommendations:**
1. Implement automated FEFO allocation (reduce rotation errors)
2. Reduce order quantities for SKU_A, SKU_B (high waste items)
3. Earlier markdown trigger for slow movers (Red zone → markdown at 40% RSL)
4. Partner with food bank for donation program
5. Negotiate extended RSL requirements with retailers

---

## Questions to Ask

If you need more context:
1. What products have shelf life concerns? Shelf life duration?
2. Current waste/spoilage rate and cost?
3. Do you have FEFO capability in WMS?
4. What are customer RSL requirements?
5. Date code tracking and format?
6. Markdown process and timing?
7. Multi-echelon network or single location?
8. Regulatory requirements (FDA, USDA, etc.)?

---

## Related Skills

- **inventory-optimization**: For safety stock with expiration constraints
- **demand-forecasting**: To reduce overstock and waste
- **warehouse-slotting-optimization**: For FEFO-friendly slotting
- **food-beverage-supply-chain**: For perishable product supply chain
- **pharmaceutical-supply-chain**: For drug expiry management
- **markdown-optimization**: For price optimization of expiring products
- **quality-management**: For quality control and compliance
- **replenishment-strategy**: For optimal reorder policies with expiry

---
name: slotting-fees-optimization
description: When the user wants to optimize slotting fees, negotiate trade terms with retailers, plan new product introductions, or manage retail shelf space economics. Also use when the user mentions "slotting allowances," "shelf placement fees," "new item setup fees," "retail listing fees," "pay-to-stay," or "category management fees." For promotional planning, see promotional-planning. For retail replenishment, see retail-replenishment.
---

# Slotting Fees Optimization

You are an expert in retail slotting fees, trade spend optimization, and category management economics. Your goal is to help optimize slotting fee investments, negotiate better terms with retailers, and maximize return on shelf space investments.

## Initial Assessment

Before optimizing slotting fees, understand:

1. **Business Context**
   - What products/categories are you introducing or maintaining?
   - What retailers and channels? (grocery, mass, club, convenience)
   - Current slotting fee spend? ($ and % of sales)
   - New product launch plans?
   - Market position? (leader, challenger, niche)

2. **Retailer Relationships**
   - Existing relationships and history?
   - Volume with each retailer?
   - Category captain status?
   - Current shelf space and facings?
   - Performance metrics (velocity, profit per linear foot)?

3. **Product Performance**
   - Expected sales velocity?
   - Margin structure?
   - Promotional plans?
   - Cannibalization of existing products?
   - Competitive set and differentiation?

4. **Financial Constraints**
   - New product launch budget?
   - Slotting fee budget limits?
   - ROI requirements?
   - Payback period expectations?

---

## Slotting Fee Framework

### Understanding Slotting Fees

**What Are Slotting Fees?**
- One-time payment to retailer for shelf space
- Compensates retailer for:
  - Opportunity cost (displacing existing product)
  - Risk of failure (most new products fail)
  - Administrative costs (setup, systems, labor)
  - Warehouse space allocation

**Typical Slotting Fee Ranges (per SKU per store):**

| Channel | Low | Average | High |
|---------|-----|---------|------|
| Grocery | $500 | $1,500 | $5,000 |
| Mass Merchant | $1,000 | $3,000 | $10,000 |
| Club | $5,000 | $15,000 | $50,000 |
| Convenience | $100 | $500 | $2,000 |
| Drug | $1,000 | $2,500 | $7,500 |
| Natural/Organic | $500 | $2,000 | $5,000 |

**Factors Influencing Slotting Fees:**
- Retailer size and power
- Product category (shelf-stable vs. refrigerated vs. frozen)
- Brand strength (established vs. unknown)
- Shelf space availability
- Expected velocity
- Number of facings requested
- Geographic scope (national vs. regional)

---

## Slotting Fee Economics

### ROI Calculation Model

```python
import pandas as pd
import numpy as np

class SlottingFeeAnalyzer:
    """
    Analyze slotting fee economics and ROI
    """

    def __init__(self, product_data, retailer_data):
        """
        Initialize analyzer

        Parameters:
        - product_data: dict with product financials
        - retailer_data: dict with retailer scope and fees
        """
        self.product = product_data
        self.retailer = retailer_data

    def calculate_roi(self, time_horizon_months=12):
        """
        Calculate ROI on slotting fee investment

        Returns:
        - comprehensive financial analysis
        """

        # Slotting fee investment
        slotting_fee_per_store = self.retailer['slotting_fee_per_store']
        num_stores = self.retailer['num_stores']
        total_slotting = slotting_fee_per_store * num_stores

        # Sales forecast
        weekly_sales_per_store = self.product['expected_weekly_units_per_store']
        weeks_in_period = (time_horizon_months / 12) * 52
        total_units = weekly_sales_per_store * num_stores * weeks_in_period

        # Revenue
        retail_price = self.product['retail_price']
        wholesale_price = retail_price * (1 - self.retailer['retail_margin'])
        total_revenue = total_units * wholesale_price

        # Costs
        cogs_per_unit = self.product['cogs']
        total_cogs = total_units * cogs_per_unit

        # Gross profit
        gross_profit = total_revenue - total_cogs

        # Other trade spend
        promotional_rate = self.product.get('promotional_rate_pct', 0.20)
        promotional_spend = total_revenue * promotional_rate

        # Marketing support
        marketing_spend = self.product.get('marketing_spend', 0)

        # Net profit
        total_trade_investment = total_slotting + promotional_spend + marketing_spend
        net_profit = gross_profit - total_trade_investment

        # ROI metrics
        roi = net_profit / total_trade_investment if total_trade_investment > 0 else 0
        payback_months = (total_slotting / (net_profit / time_horizon_months)
                         if net_profit > 0 else float('inf'))

        # Profit per store
        profit_per_store = net_profit / num_stores

        # Sales per linear foot (assuming 1 facing = 4 inches)
        facings = self.product.get('facings', 1)
        linear_feet = (facings * 4) / 12
        annual_sales_per_lf = (total_revenue / time_horizon_months * 12) / (num_stores * linear_feet)

        return {
            'investment': {
                'slotting_fee': total_slotting,
                'slotting_per_store': slotting_fee_per_store,
                'promotional_spend': promotional_spend,
                'marketing_spend': marketing_spend,
                'total_investment': total_trade_investment
            },
            'sales': {
                'total_units': total_units,
                'total_revenue': total_revenue,
                'units_per_store_per_week': weekly_sales_per_store
            },
            'profitability': {
                'gross_profit': gross_profit,
                'net_profit': net_profit,
                'profit_per_store': profit_per_store,
                'gross_margin_pct': (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
            },
            'roi_metrics': {
                'roi': roi,
                'roi_pct': roi * 100,
                'payback_months': payback_months,
                'sales_per_linear_foot': annual_sales_per_lf
            },
            'recommendation': self._generate_recommendation(roi, payback_months)
        }

    def _generate_recommendation(self, roi, payback_months):
        """Generate go/no-go recommendation"""

        if roi > 0.50 and payback_months < 12:
            return {
                'decision': 'STRONG GO',
                'rationale': 'High ROI and fast payback'
            }
        elif roi > 0.25 and payback_months < 18:
            return {
                'decision': 'GO',
                'rationale': 'Positive ROI with acceptable payback'
            }
        elif roi > 0:
            return {
                'decision': 'MARGINAL',
                'rationale': 'Positive but weak ROI - negotiate better terms'
            }
        else:
            return {
                'decision': 'NO GO',
                'rationale': 'Negative ROI - do not proceed'
            }

    def sensitivity_analysis(self, variable, values):
        """
        Run sensitivity analysis on key variables

        Parameters:
        - variable: which variable to vary ('sales', 'slotting_fee', etc.)
        - values: range of values to test

        Returns:
        - sensitivity results
        """

        base_case = self.calculate_roi()
        results = []

        for value in values:
            # Modify parameter
            if variable == 'weekly_sales_per_store':
                self.product['expected_weekly_units_per_store'] = value
            elif variable == 'slotting_fee_per_store':
                self.retailer['slotting_fee_per_store'] = value
            elif variable == 'retail_price':
                self.product['retail_price'] = value

            # Recalculate
            scenario = self.calculate_roi()

            results.append({
                variable: value,
                'roi_pct': scenario['roi_metrics']['roi_pct'],
                'payback_months': scenario['roi_metrics']['payback_months'],
                'net_profit': scenario['profitability']['net_profit']
            })

        return pd.DataFrame(results)


# Example usage
product = {
    'sku': 'NEW_PRODUCT_A',
    'retail_price': 4.99,
    'cogs': 2.00,
    'expected_weekly_units_per_store': 5,
    'facings': 2,
    'promotional_rate_pct': 0.20,
    'marketing_spend': 50000
}

retailer = {
    'name': 'Major Grocery Chain',
    'num_stores': 500,
    'slotting_fee_per_store': 1500,
    'retail_margin': 0.25
}

analyzer = SlottingFeeAnalyzer(product, retailer)
analysis = analyzer.calculate_roi(time_horizon_months=12)

print(f"Total Investment: ${analysis['investment']['total_investment']:,.0f}")
print(f"Net Profit (Year 1): ${analysis['profitability']['net_profit']:,.0f}")
print(f"ROI: {analysis['roi_metrics']['roi_pct']:.0f}%")
print(f"Payback: {analysis['roi_metrics']['payback_months']:.1f} months")
print(f"Recommendation: {analysis['recommendation']['decision']}")
```

---

## Negotiation Strategies

### Slotting Fee Negotiation Framework

```python
class SlottingNegotiator:
    """
    Framework for negotiating slotting fees
    """

    def __init__(self, manufacturer_profile, product_profile):
        self.manufacturer = manufacturer_profile
        self.product = product_profile

    def assess_negotiating_power(self):
        """
        Assess negotiating power with retailer

        Factors that strengthen position:
        - Established brand with consumer pull
        - Category leadership
        - High expected velocity
        - Innovation/differentiation
        - Strong marketing support
        - Category captain status
        """

        power_score = 0

        # Brand strength (0-25 points)
        brand_strength = self.manufacturer.get('brand_awareness_pct', 0)
        power_score += (brand_strength / 4)

        # Market share in category (0-25 points)
        market_share = self.manufacturer.get('category_market_share_pct', 0)
        power_score += (market_share * 2.5)

        # Innovation (0-20 points)
        if self.product.get('first_to_market', False):
            power_score += 20
        elif self.product.get('highly_differentiated', False):
            power_score += 15
        elif self.product.get('line_extension', False):
            power_score += 5

        # Expected velocity (0-15 points)
        expected_velocity = self.product.get('expected_velocity_rank', 'medium')
        velocity_points = {'high': 15, 'medium': 10, 'low': 5}
        power_score += velocity_points.get(expected_velocity, 5)

        # Marketing support (0-15 points)
        marketing_budget = self.product.get('marketing_budget', 0)
        if marketing_budget > 500000:
            power_score += 15
        elif marketing_budget > 100000:
            power_score += 10
        else:
            power_score += 5

        # Classify negotiating position
        if power_score >= 75:
            position = 'STRONG'
        elif power_score >= 50:
            position = 'MODERATE'
        else:
            position = 'WEAK'

        return {
            'power_score': power_score,
            'position': position,
            'factors': self._identify_strengths_weaknesses(power_score)
        }

    def _identify_strengths_weaknesses(self, score):
        """Identify negotiating strengths and weaknesses"""

        strengths = []
        weaknesses = []

        if self.manufacturer.get('brand_awareness_pct', 0) > 60:
            strengths.append('Strong brand awareness')
        else:
            weaknesses.append('Low brand awareness')

        if self.manufacturer.get('category_market_share_pct', 0) > 15:
            strengths.append('Category leader')
        else:
            weaknesses.append('Small market share')

        if self.product.get('first_to_market', False):
            strengths.append('First-to-market innovation')

        return {'strengths': strengths, 'weaknesses': weaknesses}

    def generate_negotiation_tactics(self):
        """Generate negotiation tactics based on position"""

        power = self.assess_negotiating_power()
        position = power['position']

        tactics = []

        if position == 'STRONG':
            tactics = [
                'Refuse slotting fees entirely (consumer pull)',
                'Offer performance-based fees (pay only if targets met)',
                'Demand prime shelf location',
                'Request multi-year shelf space guarantee',
                'Negotiate category captain role',
                'Offer exclusive innovation window'
            ]

        elif position == 'MODERATE':
            tactics = [
                'Negotiate reduced slotting fee',
                'Offer performance guarantees (sales velocity)',
                'Bundle multiple SKUs for lower per-SKU fee',
                'Propose trial period with reduced fee',
                'Offer incremental promotional support',
                'Share consumer research data'
            ]

        else:  # WEAK
            tactics = [
                'Pay standard slotting fee',
                'Offer higher promotional support',
                'Accept secondary shelf location',
                'Start with limited distribution (test stores)',
                'Propose consignment terms',
                'Offer free fills or extended payment terms'
            ]

        return {
            'position': position,
            'tactics': tactics,
            'priority': self._prioritize_tactics(tactics)
        }

    def _prioritize_tactics(self, tactics):
        """Prioritize tactics to focus negotiation"""
        return tactics[:3]  # Top 3 tactics


# Example
manufacturer = {
    'brand_awareness_pct': 75,
    'category_market_share_pct': 22,
    'category_captain': True
}

product = {
    'first_to_market': True,
    'highly_differentiated': True,
    'expected_velocity_rank': 'high',
    'marketing_budget': 1000000
}

negotiator = SlottingNegotiator(manufacturer, product)
power = negotiator.assess_negotiating_power()
tactics = negotiator.generate_negotiation_tactics()

print(f"Negotiating Position: {power['position']} (Score: {power['power_score']:.0f})")
print("\nStrengths:")
for s in power['factors']['strengths']:
    print(f"  - {s}")

print("\nRecommended Tactics:")
for t in tactics['priority']:
    print(f"  - {t}")
```

---

## Alternative Approaches to Slotting Fees

### Performance-Based Agreements

```python
def design_performance_based_agreement(product, retailer, targets):
    """
    Design performance-based slotting agreement

    Instead of upfront slotting fee, tie payments to performance

    Parameters:
    - product: product information
    - retailer: retailer information
    - targets: performance targets

    Returns:
    - performance-based terms
    """

    # Traditional slotting fee
    traditional_fee = retailer['traditional_slotting_fee']

    # Performance tiers
    agreement = {
        'structure': 'performance_based',
        'upfront_fee': traditional_fee * 0.25,  # 25% upfront
        'performance_tiers': []
    }

    # Tier 1: Meet baseline target
    baseline_sales = targets['baseline_units_year1']
    tier1_payment = traditional_fee * 0.25

    agreement['performance_tiers'].append({
        'tier': 1,
        'target': f"{baseline_sales:,} units in Year 1",
        'payment': tier1_payment,
        'trigger': 'Meet baseline sales'
    })

    # Tier 2: Exceed baseline by 25%
    tier2_target = baseline_sales * 1.25
    tier2_payment = traditional_fee * 0.25

    agreement['performance_tiers'].append({
        'tier': 2,
        'target': f"{tier2_target:,} units in Year 1",
        'payment': tier2_payment,
        'trigger': 'Exceed baseline by 25%'
    })

    # Tier 3: Exceed baseline by 50%
    tier3_target = baseline_sales * 1.50
    tier3_payment = traditional_fee * 0.25

    agreement['performance_tiers'].append({
        'tier': 3,
        'target': f"{tier3_target:,} units in Year 1",
        'payment': tier3_payment,
        'trigger': 'Exceed baseline by 50%'
    })

    # Total potential payment
    agreement['max_total_payment'] = (
        agreement['upfront_fee'] +
        sum(tier['payment'] for tier in agreement['performance_tiers'])
    )

    # Benefits
    agreement['benefits'] = {
        'manufacturer': 'Reduced upfront risk, pay only for performance',
        'retailer': 'Potential for higher total fees, shared success'
    }

    return agreement


# Example
performance_agreement = design_performance_based_agreement(
    product={'sku': 'NEW_PRODUCT'},
    retailer={'traditional_slotting_fee': 750000},
    targets={'baseline_units_year1': 100000}
)

print("Performance-Based Agreement:")
print(f"Upfront: ${performance_agreement['upfront_fee']:,.0f}")
for tier in performance_agreement['performance_tiers']:
    print(f"Tier {tier['tier']}: {tier['target']} → ${tier['payment']:,.0f}")
print(f"Max Total: ${performance_agreement['max_total_payment']:,.0f}")
```

---

## Portfolio Optimization

### Optimizing Slotting Investments Across Products

```python
from pulp import *

def optimize_slotting_portfolio(products, total_budget, constraints):
    """
    Optimize slotting fee allocation across product portfolio

    Parameters:
    - products: list of products with slotting costs and expected returns
    - total_budget: total slotting fee budget
    - constraints: business constraints

    Returns:
    - optimal allocation
    """

    # Create problem
    prob = LpProblem("Slotting_Portfolio", LpMaximize)

    # Decision variables: invest in product i or not
    invest = LpVariable.dicts("Invest",
                               [p['sku'] for p in products],
                               cat='Binary')

    # Objective: Maximize total NPV
    prob += lpSum([
        invest[p['sku']] * p['expected_npv']
        for p in products
    ])

    # Constraints

    # 1. Budget constraint
    prob += lpSum([
        invest[p['sku']] * p['total_slotting_cost']
        for p in products
    ]) <= total_budget

    # 2. Minimum new products per year
    min_new_products = constraints.get('min_new_products', 0)
    prob += lpSum([invest[p['sku']] for p in products]) >= min_new_products

    # 3. Category requirements (at least X products per category)
    categories = set(p['category'] for p in products)
    for category in categories:
        min_per_category = constraints.get(f'min_{category}', 0)
        prob += lpSum([
            invest[p['sku']]
            for p in products
            if p['category'] == category
        ]) >= min_per_category

    # 4. Strategic must-haves
    must_invest = constraints.get('must_invest', [])
    for sku in must_invest:
        if sku in [p['sku'] for p in products]:
            prob += invest[sku] == 1

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract results
    selected_products = []
    total_cost = 0
    total_npv = 0

    for p in products:
        if invest[p['sku']].varValue > 0.5:
            selected_products.append({
                'sku': p['sku'],
                'category': p['category'],
                'slotting_cost': p['total_slotting_cost'],
                'expected_npv': p['expected_npv'],
                'roi': p['expected_npv'] / p['total_slotting_cost']
            })
            total_cost += p['total_slotting_cost']
            total_npv += p['expected_npv']

    results = {
        'status': LpStatus[prob.status],
        'selected_products': pd.DataFrame(selected_products),
        'num_products': len(selected_products),
        'total_slotting_cost': total_cost,
        'total_expected_npv': total_npv,
        'portfolio_roi': total_npv / total_cost if total_cost > 0 else 0
    }

    return results


# Example
products = [
    {'sku': 'Product_A', 'category': 'snacks', 'total_slotting_cost': 500000,
     'expected_npv': 750000},
    {'sku': 'Product_B', 'category': 'snacks', 'total_slotting_cost': 400000,
     'expected_npv': 300000},
    {'sku': 'Product_C', 'category': 'beverages', 'total_slotting_cost': 600000,
     'expected_npv': 900000},
    {'sku': 'Product_D', 'category': 'beverages', 'total_slotting_cost': 350000,
     'expected_npv': 200000},
    {'sku': 'Product_E', 'category': 'snacks', 'total_slotting_cost': 450000,
     'expected_npv': 600000}
]

result = optimize_slotting_portfolio(
    products=products,
    total_budget=1500000,
    constraints={
        'min_new_products': 3,
        'must_invest': ['Product_C']  # Strategic priority
    }
)

print(f"Selected {result['num_products']} products")
print(f"Total Investment: ${result['total_slotting_cost']:,.0f}")
print(f"Expected NPV: ${result['total_expected_npv']:,.0f}")
print(f"Portfolio ROI: {result['portfolio_roi']:.1%}")
print("\nSelected Products:")
print(result['selected_products'][['sku', 'slotting_cost', 'expected_npv', 'roi']])
```

---

## Monitoring and Performance Tracking

### Post-Launch Performance Tracking

```python
class SlottingPerformanceTracker:
    """
    Track actual performance vs. business case
    """

    def __init__(self, business_case):
        self.business_case = business_case
        self.actual_performance = []

    def record_performance(self, period_data):
        """Record actual sales performance for a period"""
        self.actual_performance.append(period_data)

    def compare_to_business_case(self):
        """Compare actual to business case projections"""

        if not self.actual_performance:
            return None

        df = pd.DataFrame(self.actual_performance)

        # Cumulative actuals
        cumulative_units = df['units_sold'].sum()
        cumulative_revenue = df['revenue'].sum()

        # Business case projections (annualized)
        bc_units = self.business_case['projected_annual_units']
        bc_revenue = self.business_case['projected_annual_revenue']

        # Months of data
        months_tracked = len(df)

        # Annualized actuals
        annualized_units = cumulative_units * (12 / months_tracked)
        annualized_revenue = cumulative_revenue * (12 / months_tracked)

        # Variance
        units_variance = annualized_units - bc_units
        revenue_variance = annualized_revenue - bc_revenue

        # ROI recalculation
        actual_gross_profit = cumulative_revenue * self.business_case['gross_margin']
        slotting_investment = self.business_case['slotting_fee']
        annualized_gross_profit = actual_gross_profit * (12 / months_tracked)

        actual_roi = (annualized_gross_profit - slotting_investment) / slotting_investment

        comparison = {
            'months_tracked': months_tracked,
            'business_case': {
                'projected_units': bc_units,
                'projected_revenue': bc_revenue,
                'projected_roi': self.business_case['projected_roi']
            },
            'actual_annualized': {
                'units': annualized_units,
                'revenue': annualized_revenue,
                'roi': actual_roi
            },
            'variance': {
                'units': units_variance,
                'units_pct': units_variance / bc_units * 100,
                'revenue': revenue_variance,
                'revenue_pct': revenue_variance / bc_revenue * 100
            },
            'performance_vs_plan': self._classify_performance(
                units_variance / bc_units * 100
            )
        }

        return comparison

    def _classify_performance(self, variance_pct):
        """Classify performance vs. plan"""

        if variance_pct > 10:
            return 'EXCEEDING'
        elif variance_pct > -10:
            return 'ON_TRACK'
        else:
            return 'UNDERPERFORMING'


# Example
business_case = {
    'projected_annual_units': 500000,
    'projected_annual_revenue': 2500000,
    'projected_roi': 0.45,
    'slotting_fee': 750000,
    'gross_margin': 0.35
}

tracker = SlottingPerformanceTracker(business_case)

# Record actual performance
tracker.record_performance({'month': 1, 'units_sold': 35000, 'revenue': 175000})
tracker.record_performance({'month': 2, 'units_sold': 40000, 'revenue': 200000})
tracker.record_performance({'month': 3, 'units_sold': 45000, 'revenue': 225000})

comparison = tracker.compare_to_business_case()

print(f"Performance: {comparison['performance_vs_plan']}")
print(f"Projected Annual Units: {comparison['business_case']['projected_units']:,}")
print(f"Actual Annualized Units: {comparison['actual_annualized']['units']:,.0f}")
print(f"Variance: {comparison['variance']['units_pct']:+.1f}%")
```

---

## Tools & Technologies

### Slotting and Trade Management Software

**Trade Promotion Management (TPM/TPO):**
- **SAP TPM**: Trade promotion and slotting management
- **Oracle TPM**: Deductions and trade spend management
- **Blacksmith Applications**: TPM/TPO for CPG
- **AFS Trade Promotion**: Analytics and optimization
- **Wipro Promax**: Cloud TPM platform

**Retail Analytics:**
- **Nielsen**: Retail measurement and shelf analytics
- **IRI**: Syndicated data and space planning
- **84.51° (Kroger)**: Retailer data and insights
- **Catalina**: Personalized retail media
- **dunnhumby**: Retail science and analytics

### Python Libraries

```python
# Financial modeling
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Optimization
from pulp import *

# Data analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```

---

## Common Challenges & Solutions

### Challenge: High Slotting Fees Prevent New Product Launch

**Problem:**
- Retailer wants $2M in slotting fees
- Budget only $1M
- Can't launch without this retailer

**Solutions:**
- Negotiate performance-based terms (reduced upfront)
- Start with smaller store count (test stores)
- Bundle multiple SKUs for reduced per-SKU fee
- Offer higher promotional support instead of slotting
- Find alternative retailers with lower fees
- Delay launch until more capital available

### Challenge: Poor ROI on Previous Launches

**Problem:**
- Last 3 products underperformed projections
- Lost money on slotting investments
- Credibility damaged with management

**Solutions:**
- Post-mortem analysis (why did products fail?)
- Improve sales forecasting (conservative projections)
- Better product testing (more research, test markets)
- Tighter launch criteria (raise ROI bar)
- Negotiate performance guarantees with retailers
- Focus on fewer, better products

### Challenge: Retailer Demands Increasing Fees

**Problem:**
- Fees up 20% year-over-year
- Squeezing margins
- Threatening profitability

**Solutions:**
- Negotiate multi-year agreements (lock in rates)
- Demonstrate high velocity (data-driven negotiation)
- Pursue category captain status (eliminate some fees)
- Shift mix to online/direct (bypass retailers)
- Consolidate SKUs (reduce fee burden)
- Build consumer demand (pull vs. push)

---

## Output Format

### Slotting Fee Business Case Template

**Product Information:**
- Product: New Organic Granola Bar
- Category: Snacks - Bars
- SKUs: 3 flavors
- Retail Price: $4.99
- COGS: $2.10
- Gross Margin: 58%

**Retailer Information:**
- Retailer: National Grocery Chain
- Stores: 1,200
- Slotting Fee: $1,800 per SKU per store
- Total Slotting Investment: $6,480,000

**Sales Projections (Year 1):**

| Metric | Per Store Per Week | Total Annual |
|--------|--------------------|--------------|
| Units | 8 | 499,200 |
| Revenue (wholesale) | $30 | $18,637,000 |
| Gross Profit | $17.40 | $10,809,000 |

**Cost Structure:**

| Item | Amount | % of Sales |
|------|--------|------------|
| Slotting Fees | $6,480,000 | 34.8% |
| Promotional Support | $1,863,700 | 10.0% |
| Marketing | $500,000 | 2.7% |
| **Total Investment** | **$8,843,700** | **47.5%** |

**Financial Metrics:**

| Metric | Value |
|--------|-------|
| Year 1 Net Profit | $1,965,300 |
| ROI | 22.2% |
| Payback Period | 10.8 months |
| Sales per Linear Foot | $17,320 |

**Recommendation: GO**
- Positive ROI and sub-12 month payback
- High sales per linear foot justifies space
- Strong promotional plan to drive velocity
- Mitigate risk: Start with 600 stores (test phase)

**Risks & Mitigation:**
1. Risk: Lower than expected velocity
   - Mitigation: Performance-based agreement (25% upfront, balance at 6 months if targets met)

2. Risk: High cannibalization of existing products
   - Mitigation: Test market analysis shows <15% cannibalization

3. Risk: Retailer demands pay-to-stay fees Year 2
   - Mitigation: Negotiate 2-year agreement upfront

---

## Questions to Ask

If you need more context:
1. What products are you planning to introduce or maintain?
2. What retailers and how many stores?
3. What slotting fees are being quoted?
4. What are your sales projections (units per store per week)?
5. What's your product's margin structure?
6. Do you have existing relationships with these retailers?
7. Is this a new category entry or line extension?
8. What's your negotiating leverage (brand strength, innovation, etc.)?

---

## Related Skills

- **promotional-planning**: For trade promotion optimization
- **retail-replenishment**: For retailer inventory management
- **demand-forecasting**: For sales projections
- **markdown-optimization**: For pricing strategies
- **procurement-optimization**: For negotiation frameworks
- **supplier-selection**: For retailer selection criteria
- **strategic-sourcing**: For long-term retailer partnerships
- **spend-analysis**: For trade spend tracking and optimization

---
name: sales-operations-planning
description: When the user wants to implement S&OP process, integrate demand and supply planning, balance business objectives, or facilitate executive planning meetings. Also use when the user mentions "S&OP," "IBP" (Integrated Business Planning), "demand-supply balancing," "executive S&OP," "monthly planning cycle," "consensus demand," "supply review," or "financial reconciliation." For detailed demand forecasts, see demand-forecasting. For capacity analysis, see capacity-planning.
---

# Sales & Operations Planning (S&OP)

You are an expert in Sales & Operations Planning (S&OP) and Integrated Business Planning (IBP). Your goal is to help organizations implement effective S&OP processes that align demand, supply, finance, and strategy to drive better business decisions.

## Initial Assessment

Before implementing or improving S&OP, understand:

1. **Current State**
   - Existing planning process? (informal, Excel-based, system-driven)
   - Planning cycle and frequency? (monthly, weekly)
   - Who participates? (Sales, Ops, Finance, Exec team)
   - Current pain points and issues?

2. **Organization Context**
   - Company size and complexity?
   - Number of product families/SKUs?
   - Geographic scope? (single site, regional, global)
   - Industry characteristics? (seasonal, promotional, project-based)

3. **Planning Maturity**
   - Forecast accuracy levels?
   - Cross-functional collaboration quality?
   - Data systems and integration?
   - Performance measurement?

4. **Business Objectives**
   - Strategic goals? (growth, margin, service, inventory)
   - Key metrics and targets?
   - Decision-making authority and governance?
   - Stakeholder expectations?

---

## S&OP Framework

### S&OP Definition

**Sales & Operations Planning (S&OP)** is a monthly integrated business management process that brings together all plans for the business (customers, sales, marketing, operations, engineering, finance, product management) into one integrated set of plans.

**Key Principles:**
1. **Cross-Functional Integration**: Break down silos
2. **Forward-Looking**: 18-24 month rolling horizon
3. **Executive-Owned**: Leadership accountability
4. **Single Version of Truth**: One consensus plan
5. **Decision-Focused**: Drive actions, not just reports

### S&OP vs. IBP (Integrated Business Planning)

| Aspect | S&OP | IBP |
|--------|------|-----|
| **Scope** | Demand-Supply balance | Strategic + Financial integration |
| **Horizon** | 12-18 months | 24-36+ months |
| **Frequency** | Monthly | Monthly + quarterly reviews |
| **Focus** | Operational alignment | Strategic + operational |
| **Participants** | Cross-functional | Includes Finance, Strategy |
| **Outputs** | Volume plans | Volume + Financial plans |

---

## S&OP Process Steps

### Monthly S&OP Cycle

**Five-Step Process:**

```
Week 1: Data Gathering & Demand Review
Week 2: Supply/Capacity Review
Week 3: Pre-S&OP Meeting
Week 4: Executive S&OP Meeting
Week 5: Implementation & Monitoring
```

### Step 1: Data Gathering & Forecasting

**Objectives:**
- Collect actual performance data
- Generate statistical baseline forecasts
- Identify exceptions and anomalies

**Activities:**
- Update actual sales, production, inventory
- Run forecasting models
- Calculate forecast accuracy
- Flag exceptions (large changes, new products)

**Outputs:**
- Statistical forecast by product family
- Forecast accuracy metrics (MAPE, bias)
- Exception reports
- Data quality issues log

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SopDataGathering:
    """Data gathering and baseline forecasting for S&OP"""

    def __init__(self, historical_data):
        """
        Parameters:
        - historical_data: DataFrame with 'date', 'product_family', 'actual_sales'
        """
        self.data = historical_data

    def calculate_forecast_accuracy(self, forecast_col='forecast',
                                    actual_col='actual_sales'):
        """Calculate forecast accuracy metrics"""

        df = self.data.copy()

        # MAPE (Mean Absolute Percentage Error)
        df['ape'] = np.abs((df[actual_col] - df[forecast_col]) / df[actual_col]) * 100
        mape = df['ape'].mean()

        # Bias (average error)
        df['error'] = df[forecast_col] - df[actual_col]
        bias = df['error'].mean()
        bias_pct = (bias / df[actual_col].mean()) * 100

        # MAD (Mean Absolute Deviation)
        mad = np.abs(df['error']).mean()

        # Tracking signal
        cumulative_error = df['error'].sum()
        tracking_signal = cumulative_error / mad if mad > 0 else 0

        return {
            'mape': mape,
            'bias': bias,
            'bias_pct': bias_pct,
            'mad': mad,
            'tracking_signal': tracking_signal
        }

    def identify_exceptions(self, threshold_pct=20):
        """
        Identify forecast exceptions requiring attention

        Parameters:
        - threshold_pct: % change threshold for flagging
        """

        df = self.data.copy()

        # Calculate month-over-month change
        df = df.sort_values(['product_family', 'date'])
        df['prev_forecast'] = df.groupby('product_family')['forecast'].shift(1)
        df['forecast_change_pct'] = (
            (df['forecast'] - df['prev_forecast']) / df['prev_forecast'] * 100
        )

        # Flag large changes
        exceptions = df[
            np.abs(df['forecast_change_pct']) > threshold_pct
        ].copy()

        return exceptions[['product_family', 'date', 'forecast',
                          'prev_forecast', 'forecast_change_pct']]

    def generate_baseline_forecast(self, periods_ahead=18):
        """
        Generate statistical baseline forecast

        Simple approach using moving average with trend
        """

        forecasts = []

        for product in self.data['product_family'].unique():
            product_data = self.data[
                self.data['product_family'] == product
            ].sort_values('date')

            # Calculate moving average and trend
            recent_avg = product_data['actual_sales'].tail(3).mean()
            older_avg = product_data['actual_sales'].tail(6).head(3).mean()
            trend = (recent_avg - older_avg) / 3  # Per period trend

            # Generate forecast
            last_date = product_data['date'].max()

            for i in range(1, periods_ahead + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                forecast_value = recent_avg + (trend * i)
                forecast_value = max(0, forecast_value)  # No negative forecasts

                forecasts.append({
                    'product_family': product,
                    'date': forecast_date,
                    'baseline_forecast': forecast_value,
                    'method': 'MA_with_trend'
                })

        return pd.DataFrame(forecasts)

# Example usage
historical = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=12, freq='MS'),
    'product_family': ['Electronics'] * 12,
    'actual_sales': [1000, 1100, 1050, 1200, 1250, 1300, 1280, 1350, 1400, 1450, 1500, 1550],
    'forecast': [980, 1120, 1000, 1180, 1300, 1250, 1300, 1320, 1380, 1480, 1520, 1500]
})

sop_data = SopDataGathering(historical)

# Calculate accuracy
accuracy = sop_data.calculate_forecast_accuracy()
print("Forecast Accuracy:")
print(f"  MAPE: {accuracy['mape']:.1f}%")
print(f"  Bias: {accuracy['bias_pct']:.1f}%")

# Identify exceptions
exceptions = sop_data.identify_exceptions(threshold_pct=15)
print(f"\nExceptions (>{15}% change): {len(exceptions)}")

# Generate baseline
baseline = sop_data.generate_baseline_forecast(periods_ahead=6)
print("\nBaseline Forecast (next 6 months):")
print(baseline.head())
```

### Step 2: Demand Review

**Objectives:**
- Review and adjust statistical forecast
- Incorporate market intelligence
- Build consensus demand plan

**Participants:**
- Sales leadership
- Marketing
- Product management
- Demand planning
- Finance (observer)

**Activities:**
- Review statistical forecast
- Discuss upcoming promotions, launches
- Consider market trends, competitive actions
- Adjust forecasts based on business intelligence
- Document assumptions and risks

**Key Questions:**
- What's changed since last month?
- Any new customer wins/losses?
- Promotional plans finalized?
- Pricing changes impact?
- Competitive landscape shifts?

**Outputs:**
- Consensus demand forecast by product family
- Demand assumptions documented
- Upside/downside scenarios
- Risks and opportunities identified

```python
class DemandReview:
    """Demand review and consensus building"""

    def __init__(self, statistical_forecast):
        self.statistical_fcst = statistical_forecast
        self.adjustments = []
        self.assumptions = []

    def add_adjustment(self, product_family, period, adjustment_type,
                      adjustment_value, reason, owner):
        """
        Record demand adjustment

        Parameters:
        - adjustment_type: 'absolute', 'percentage', 'additive'
        - adjustment_value: amount of adjustment
        - reason: business rationale
        - owner: who made the adjustment (Sales, Marketing, etc.)
        """

        self.adjustments.append({
            'product_family': product_family,
            'period': period,
            'adjustment_type': adjustment_type,
            'adjustment_value': adjustment_value,
            'reason': reason,
            'owner': owner,
            'timestamp': datetime.now()
        })

    def add_assumption(self, assumption_text, category, impact_level):
        """
        Document planning assumption

        Parameters:
        - category: 'promotion', 'market', 'competitive', 'economic'
        - impact_level: 'low', 'medium', 'high'
        """

        self.assumptions.append({
            'assumption': assumption_text,
            'category': category,
            'impact_level': impact_level,
            'date_added': datetime.now()
        })

    def calculate_consensus_forecast(self):
        """Apply adjustments to create consensus forecast"""

        consensus = self.statistical_fcst.copy()

        for adj in self.adjustments:
            mask = (
                (consensus['product_family'] == adj['product_family']) &
                (consensus['period'] == adj['period'])
            )

            if adj['adjustment_type'] == 'percentage':
                consensus.loc[mask, 'consensus_forecast'] = (
                    consensus.loc[mask, 'statistical_forecast'] *
                    (1 + adj['adjustment_value'] / 100)
                )
            elif adj['adjustment_type'] == 'additive':
                consensus.loc[mask, 'consensus_forecast'] = (
                    consensus.loc[mask, 'statistical_forecast'] +
                    adj['adjustment_value']
                )
            elif adj['adjustment_type'] == 'absolute':
                consensus.loc[mask, 'consensus_forecast'] = adj['adjustment_value']

        # Calculate forecast value added (FVA)
        consensus['adjustment'] = (
            consensus['consensus_forecast'] - consensus['statistical_forecast']
        )

        return consensus

    def create_demand_scenarios(self, consensus_forecast):
        """
        Create upside/downside scenarios

        Typically: Base, Optimistic (+15%), Pessimistic (-15%)
        """

        scenarios = consensus_forecast.copy()

        scenarios['base_case'] = scenarios['consensus_forecast']
        scenarios['optimistic'] = scenarios['consensus_forecast'] * 1.15
        scenarios['pessimistic'] = scenarios['consensus_forecast'] * 0.85

        return scenarios

    def get_adjustment_summary(self):
        """Summarize adjustments by owner and type"""

        if not self.adjustments:
            return pd.DataFrame()

        df = pd.DataFrame(self.adjustments)

        summary = df.groupby(['owner', 'adjustment_type']).agg({
            'adjustment_value': ['count', 'sum', 'mean']
        }).reset_index()

        return summary

# Example
statistical_fcst = pd.DataFrame({
    'product_family': ['Electronics', 'Electronics', 'Appliances', 'Appliances'],
    'period': ['2025-01', '2025-02', '2025-01', '2025-02'],
    'statistical_forecast': [10000, 10200, 5000, 5100]
})

demand_review = DemandReview(statistical_fcst)

# Add adjustments
demand_review.add_adjustment(
    product_family='Electronics',
    period='2025-01',
    adjustment_type='percentage',
    adjustment_value=10,  # +10%
    reason='New product launch expected to drive 10% uplift',
    owner='Product Management'
)

demand_review.add_adjustment(
    product_family='Appliances',
    period='2025-02',
    adjustment_type='additive',
    adjustment_value=-500,
    reason='Competitor pricing pressure',
    owner='Sales'
)

# Add assumptions
demand_review.add_assumption(
    'Q1 promotional campaign will increase demand 10-15%',
    category='promotion',
    impact_level='high'
)

# Calculate consensus
consensus = demand_review.calculate_consensus_forecast()
print("Consensus Demand Forecast:")
print(consensus)

# Scenarios
scenarios = demand_review.create_demand_scenarios(consensus)
print("\nDemand Scenarios:")
print(scenarios[['product_family', 'period', 'pessimistic', 'base_case', 'optimistic']])
```

### Step 3: Supply Review

**Objectives:**
- Assess supply capacity to meet demand
- Identify constraints and gaps
- Develop supply alternatives

**Participants:**
- Operations leadership
- Manufacturing
- Procurement
- Supply planning
- Engineering

**Activities:**
- Review production capacity vs. demand
- Identify bottlenecks and constraints
- Assess supplier capability
- Evaluate inventory positions
- Propose supply solutions

**Key Questions:**
- Can we meet the demand plan?
- What are the constraints?
- Lead time for capacity additions?
- Supply chain risks?
- Cost implications?

**Outputs:**
- Supply plan by product family
- Capacity gaps identified
- Alternative supply scenarios
- Inventory strategy
- Investment requirements

```python
class SupplyReview:
    """Supply review and capacity analysis"""

    def __init__(self, consensus_demand, capacity_data):
        """
        Parameters:
        - consensus_demand: demand forecast
        - capacity_data: available capacity by period
        """
        self.demand = consensus_demand
        self.capacity = capacity_data

    def analyze_capacity_gaps(self):
        """Identify where demand exceeds capacity"""

        # Merge demand and capacity
        analysis = self.demand.merge(
            self.capacity,
            on=['product_family', 'period'],
            how='left'
        )

        # Calculate gap
        analysis['gap'] = analysis['available_capacity'] - analysis['consensus_forecast']
        analysis['gap_pct'] = (analysis['gap'] / analysis['available_capacity']) * 100
        analysis['status'] = analysis['gap'].apply(
            lambda x: 'OK' if x >= 0 else 'CONSTRAINED'
        )

        # Utilization
        analysis['utilization'] = (
            analysis['consensus_forecast'] / analysis['available_capacity'] * 100
        )

        return analysis

    def propose_supply_scenarios(self, gap_analysis):
        """
        Develop supply alternatives for constrained periods

        Scenarios:
        1. Do Nothing (accept stockouts)
        2. Add Overtime
        3. Outsource
        4. Build Ahead
        """

        constrained = gap_analysis[gap_analysis['status'] == 'CONSTRAINED'].copy()

        if constrained.empty:
            return pd.DataFrame()

        scenarios = []

        for idx, row in constrained.iterrows():
            product = row['product_family']
            period = row['period']
            shortage = abs(row['gap'])

            # Scenario 1: Do Nothing
            scenarios.append({
                'product_family': product,
                'period': period,
                'scenario': 'Do Nothing',
                'additional_supply': 0,
                'cost': shortage * 100,  # Lost sales cost
                'risk': 'High',
                'feasibility': 'Certain'
            })

            # Scenario 2: Overtime
            overtime_capacity = row['available_capacity'] * 0.15  # 15% OT
            overtime_supply = min(shortage, overtime_capacity)
            scenarios.append({
                'product_family': product,
                'period': period,
                'scenario': 'Overtime',
                'additional_supply': overtime_supply,
                'cost': overtime_supply * 15,  # Premium cost
                'risk': 'Medium',
                'feasibility': 'High'
            })

            # Scenario 3: Outsource
            scenarios.append({
                'product_family': product,
                'period': period,
                'scenario': 'Outsource',
                'additional_supply': shortage,
                'cost': shortage * 20,  # Outsource premium
                'risk': 'Medium',
                'feasibility': 'Medium'
            })

            # Scenario 4: Build Ahead
            # Produce in prior period
            scenarios.append({
                'product_family': product,
                'period': period,
                'scenario': 'Build Ahead',
                'additional_supply': shortage,
                'cost': shortage * 5,  # Inventory holding cost
                'risk': 'Low',
                'feasibility': 'High'
            })

        return pd.DataFrame(scenarios)

    def calculate_inventory_plan(self, gap_analysis, safety_stock_days=30):
        """
        Develop inventory strategy

        Target: Safety stock + cycle stock
        """

        inventory_plan = gap_analysis.copy()

        # Daily demand rate
        inventory_plan['daily_demand'] = inventory_plan['consensus_forecast'] / 30

        # Safety stock
        inventory_plan['safety_stock'] = inventory_plan['daily_demand'] * safety_stock_days

        # Cycle stock (assume monthly production)
        inventory_plan['cycle_stock'] = inventory_plan['consensus_forecast'] / 2

        # Target inventory
        inventory_plan['target_inventory'] = (
            inventory_plan['safety_stock'] + inventory_plan['cycle_stock']
        )

        return inventory_plan[['product_family', 'period', 'consensus_forecast',
                              'safety_stock', 'cycle_stock', 'target_inventory']]

# Example
consensus_demand = pd.DataFrame({
    'product_family': ['Electronics', 'Electronics', 'Appliances'],
    'period': ['2025-01', '2025-02', '2025-01'],
    'consensus_forecast': [11000, 10200, 4500]
})

capacity_data = pd.DataFrame({
    'product_family': ['Electronics', 'Electronics', 'Appliances'],
    'period': ['2025-01', '2025-02', '2025-01'],
    'available_capacity': [10000, 12000, 5000]
})

supply_review = SupplyReview(consensus_demand, capacity_data)

# Analyze gaps
gaps = supply_review.analyze_capacity_gaps()
print("Capacity Gap Analysis:")
print(gaps[['product_family', 'period', 'consensus_forecast',
            'available_capacity', 'gap', 'status', 'utilization']])

# Propose scenarios
scenarios = supply_review.propose_supply_scenarios(gaps)
if not scenarios.empty:
    print("\nSupply Scenarios for Constrained Periods:")
    print(scenarios)

# Inventory plan
inventory = supply_review.calculate_inventory_plan(gaps)
print("\nInventory Plan:")
print(inventory)
```

### Step 4: Pre-S&OP Meeting

**Objectives:**
- Reconcile demand and supply
- Develop recommendations for executive team
- Prepare scenarios and trade-offs

**Participants:**
- S&OP process owner/facilitator
- Demand planning leader
- Supply planning leader
- Finance
- Selected business unit leaders

**Activities:**
- Review demand-supply balance
- Identify gaps and conflicts
- Develop scenarios with pros/cons/costs
- Align financial impact
- Prepare executive presentation

**Key Outputs:**
- Balanced demand-supply plan (base case)
- Alternative scenarios with trade-offs
- Financial reconciliation
- Recommendation with rationale
- Open issues requiring executive decision

```python
class PreSopMeeting:
    """Pre-S&OP meeting preparation and analysis"""

    def __init__(self, demand_plan, supply_plan, financial_data):
        self.demand = demand_plan
        self.supply = supply_plan
        self.financial = financial_data

    def create_demand_supply_balance(self):
        """Create integrated view of demand and supply"""

        balance = self.demand.merge(
            self.supply,
            on=['product_family', 'period'],
            how='outer'
        )

        balance['supply_demand_gap'] = (
            balance['supply_plan'] - balance['consensus_forecast']
        )

        balance['inventory_impact'] = balance['supply_demand_gap'].cumsum()

        return balance

    def financial_reconciliation(self, balance_plan):
        """Calculate financial impact of plan"""

        financial = balance_plan.merge(
            self.financial,
            on='product_family',
            how='left'
        )

        # Revenue
        financial['revenue'] = (
            financial['consensus_forecast'] * financial['price_per_unit']
        )

        # COGS
        financial['cogs'] = (
            financial['supply_plan'] * financial['cost_per_unit']
        )

        # Inventory value
        financial['inventory_value'] = (
            financial['inventory_impact'] * financial['cost_per_unit']
        )

        # Gross margin
        financial['gross_margin'] = financial['revenue'] - financial['cogs']

        # Aggregate by period
        summary = financial.groupby('period').agg({
            'revenue': 'sum',
            'cogs': 'sum',
            'gross_margin': 'sum',
            'inventory_value': 'sum'
        }).reset_index()

        summary['gross_margin_pct'] = (
            summary['gross_margin'] / summary['revenue'] * 100
        )

        return summary

    def create_scenario_comparison(self, scenarios_list):
        """
        Compare multiple scenarios

        Parameters:
        - scenarios_list: list of dicts with scenario details
        """

        comparison = pd.DataFrame(scenarios_list)

        # Rank scenarios
        comparison['total_score'] = (
            comparison['service_level'] * 0.4 +
            (100 - comparison['cost_impact_pct']) * 0.3 +
            comparison['feasibility'] * 0.3
        )

        comparison = comparison.sort_values('total_score', ascending=False)

        return comparison

    def generate_executive_summary(self, balance, financials):
        """Create executive summary for S&OP meeting"""

        summary = {
            'planning_period': balance['period'].min(),
            'total_demand': balance['consensus_forecast'].sum(),
            'total_supply': balance['supply_plan'].sum(),
            'net_inventory_change': balance['supply_demand_gap'].sum(),
            'revenue_plan': financials['revenue'].sum(),
            'gross_margin_pct': (
                financials['gross_margin'].sum() /
                financials['revenue'].sum() * 100
            ),
            'constraints': balance[balance['supply_demand_gap'] < 0].shape[0],
            'excess_capacity': balance[balance['supply_demand_gap'] > balance['consensus_forecast'] * 0.2].shape[0]
        }

        return summary

# Example
demand_plan = pd.DataFrame({
    'product_family': ['Electronics'] * 3,
    'period': ['2025-01', '2025-02', '2025-03'],
    'consensus_forecast': [11000, 10200, 10500]
})

supply_plan = pd.DataFrame({
    'product_family': ['Electronics'] * 3,
    'period': ['2025-01', '2025-02', '2025-03'],
    'supply_plan': [10000, 11000, 10500]
})

financial_data = pd.DataFrame({
    'product_family': ['Electronics'],
    'price_per_unit': [100],
    'cost_per_unit': [60]
})

pre_sop = PreSopMeeting(demand_plan, supply_plan, financial_data)

# Create balance
balance = pre_sop.create_demand_supply_balance()
print("Demand-Supply Balance:")
print(balance)

# Financial reconciliation
financials = pre_sop.financial_reconciliation(balance)
print("\nFinancial Summary:")
print(financials)

# Executive summary
exec_summary = pre_sop.generate_executive_summary(balance, financials)
print("\nExecutive Summary:")
for key, value in exec_summary.items():
    print(f"  {key}: {value}")

# Scenario comparison
scenarios = [
    {
        'scenario': 'Base Plan',
        'service_level': 95,
        'cost_impact_pct': 0,
        'feasibility': 90,
        'description': 'Meet demand with overtime'
    },
    {
        'scenario': 'Constrain Demand',
        'service_level': 85,
        'cost_impact_pct': -5,
        'feasibility': 100,
        'description': 'Limit sales to capacity'
    },
    {
        'scenario': 'Outsource',
        'service_level': 98,
        'cost_impact_pct': 15,
        'feasibility': 70,
        'description': 'Use contract manufacturer'
    }
]

scenario_comparison = pre_sop.create_scenario_comparison(scenarios)
print("\nScenario Comparison:")
print(scenario_comparison)
```

### Step 5: Executive S&OP Meeting

**Objectives:**
- Make final decisions on plans
- Resolve gaps and trade-offs
- Approve resource commitments
- Align on strategic priorities

**Participants:**
- CEO or COO (chair)
- VP Sales
- VP Operations
- CFO
- VP Supply Chain
- VP Product/Marketing
- Business unit heads

**Duration:** 2-4 hours

**Agenda:**
1. **Review Performance** (15 min)
   - Last month actual vs. plan
   - Key metrics and KPIs
   - Forecast accuracy

2. **Demand Review** (30 min)
   - Consensus demand by product family
   - Key changes and assumptions
   - Risks and opportunities

3. **Supply Review** (30 min)
   - Capacity and constraints
   - Supply alternatives
   - Inventory strategy

4. **Financial Review** (20 min)
   - Revenue and margin impact
   - Working capital
   - Budget alignment

5. **Scenarios and Trade-Offs** (45 min)
   - Present alternatives
   - Discuss implications
   - Make decisions

6. **Strategic Issues** (30 min)
   - New product launches
   - Capacity investments
   - Market opportunities
   - Risk mitigation

7. **Decisions and Actions** (20 min)
   - Document decisions
   - Assign action items
   - Set follow-ups

**Key Decisions:**
- Approve demand plan
- Approve supply and inventory plan
- Resource allocation decisions
- Investment approvals
- Policy changes

**Outputs:**
- Approved S&OP plan
- Decision log
- Action items with owners
- Risks and contingencies

```python
from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class SopDecision:
    """Record of S&OP decision"""
    decision_id: str
    decision_text: str
    category: str  # 'demand', 'supply', 'financial', 'strategic'
    owner: str
    due_date: datetime
    status: str  # 'approved', 'pending', 'rejected'
    rationale: str
    financial_impact: float

@dataclass
class SopActionItem:
    """Action item from S&OP meeting"""
    action_id: str
    description: str
    owner: str
    due_date: datetime
    status: str  # 'open', 'in_progress', 'completed'
    priority: str  # 'high', 'medium', 'low'

class ExecutiveSopMeeting:
    """Executive S&OP meeting management"""

    def __init__(self, meeting_date):
        self.meeting_date = meeting_date
        self.decisions = []
        self.action_items = []
        self.approved_plan = None

    def add_decision(self, decision: SopDecision):
        """Record a decision from the meeting"""
        self.decisions.append(decision)

    def add_action_item(self, action: SopActionItem):
        """Record an action item"""
        self.action_items.append(action)

    def approve_plan(self, plan_data):
        """Approve the S&OP plan"""
        self.approved_plan = {
            'approval_date': self.meeting_date,
            'plan': plan_data,
            'status': 'approved'
        }

    def generate_meeting_minutes(self):
        """Create meeting minutes document"""

        minutes = {
            'meeting_date': self.meeting_date,
            'decisions_count': len(self.decisions),
            'action_items_count': len(self.action_items),
            'decisions': [
                {
                    'id': d.decision_id,
                    'decision': d.decision_text,
                    'owner': d.owner,
                    'impact': d.financial_impact
                }
                for d in self.decisions
            ],
            'action_items': [
                {
                    'id': a.action_id,
                    'action': a.description,
                    'owner': a.owner,
                    'due': a.due_date,
                    'priority': a.priority
                }
                for a in self.action_items
            ]
        }

        return minutes

    def get_decision_summary(self):
        """Summarize decisions by category"""

        if not self.decisions:
            return pd.DataFrame()

        decisions_df = pd.DataFrame([
            {
                'category': d.category,
                'status': d.status,
                'financial_impact': d.financial_impact
            }
            for d in self.decisions
        ])

        summary = decisions_df.groupby(['category', 'status']).agg({
            'financial_impact': 'sum'
        }).reset_index()

        return summary

# Example
exec_sop = ExecutiveSopMeeting(meeting_date=datetime(2025, 1, 15))

# Record decisions
exec_sop.add_decision(SopDecision(
    decision_id='DEC-001',
    decision_text='Approve demand plan for Q1 with 11K units for Electronics',
    category='demand',
    owner='VP Sales',
    due_date=datetime(2025, 1, 31),
    status='approved',
    rationale='Aligned with new product launch plan',
    financial_impact=0
))

exec_sop.add_decision(SopDecision(
    decision_id='DEC-002',
    decision_text='Authorize overtime to cover Jan capacity gap',
    category='supply',
    owner='VP Operations',
    due_date=datetime(2025, 1, 20),
    status='approved',
    rationale='Most cost-effective option, $150K vs $220K outsourcing',
    financial_impact=-150000
))

exec_sop.add_decision(SopDecision(
    decision_id='DEC-003',
    decision_text='Approve $3.5M investment for new production line',
    category='strategic',
    owner='CFO',
    due_date=datetime(2025, 6, 30),
    status='approved',
    rationale='Required to support Q3 demand growth',
    financial_impact=-3500000
))

# Add action items
exec_sop.add_action_item(SopActionItem(
    action_id='ACT-001',
    description='Finalize contract with overtime labor agency',
    owner='HR Director',
    due_date=datetime(2025, 1, 20),
    status='open',
    priority='high'
))

exec_sop.add_action_item(SopActionItem(
    action_id='ACT-002',
    description='Complete business case for production line investment',
    owner='VP Operations',
    due_date=datetime(2025, 2, 15),
    status='in_progress',
    priority='high'
))

# Generate minutes
minutes = exec_sop.generate_meeting_minutes()
print("S&OP Meeting Minutes:")
print(f"Date: {minutes['meeting_date']}")
print(f"Decisions: {minutes['decisions_count']}")
print(f"Action Items: {minutes['action_items_count']}")

# Decision summary
decision_summary = exec_sop.get_decision_summary()
print("\nDecisions by Category:")
print(decision_summary)
```

---

## S&OP Maturity Model

### Level 1: Reactive (Ad Hoc)

**Characteristics:**
- No formal S&OP process
- Excel-based, manual
- Sales and Ops don't communicate
- Fire-fighting mode

**Symptoms:**
- Frequent expediting
- Poor forecast accuracy (<60%)
- Excess inventory or stockouts
- Missed commitments

### Level 2: Rudimentary

**Characteristics:**
- Monthly S&OP meetings started
- Some cross-functional participation
- Focus on short-term (1-3 months)
- Limited data integration

**Capabilities:**
- Basic demand review
- Capacity check
- Executive awareness

### Level 3: Standard

**Characteristics:**
- Structured 5-step process
- 12-18 month horizon
- Good data integration
- Regular cadence

**Capabilities:**
- Consensus forecasting
- Capacity planning
- Financial reconciliation
- Scenario analysis

### Level 4: Advanced

**Characteristics:**
- Integrated Business Planning (IBP)
- 24+ month horizon
- Strategic alignment
- Real-time updates

**Capabilities:**
- Portfolio management
- Profitability optimization
- What-if simulation
- Automated analytics

### Level 5: Proactive/Best-in-Class

**Characteristics:**
- Predictive and prescriptive
- AI/ML-driven insights
- Dynamic planning
- Value chain collaboration

**Capabilities:**
- Predictive analytics
- Optimization algorithms
- Digital twin modeling
- External collaboration (suppliers, customers)

---

## S&OP Metrics & KPIs

### Forecast Accuracy

**MAPE (Mean Absolute Percentage Error)**
- Target: <20% for aggregate, <30% for detailed
- Measured monthly by product family
- Track trend over time

**Bias**
- Target: ±5%
- Positive bias = over-forecasting
- Negative bias = under-forecasting

**Forecast Value Added (FVA)**
- Does manual override improve accuracy?
- Measure statistical vs. consensus performance

### Supply Performance

**Plan Attainment**
- % of production plan achieved
- Target: >95%

**Capacity Utilization**
- % of capacity used
- Target: 80-90% (allows flexibility)

**Supply Flexibility**
- Lead time to adjust capacity
- % changeover time

### Inventory Metrics

**Inventory Turns**
- COGS / Average Inventory
- Target: Industry-dependent (6-12 turns typical)

**Days of Supply**
- Inventory / Daily Sales
- Target: 30-60 days

**Inventory Accuracy**
- % of SKUs within tolerance
- Target: >95%

### Financial Metrics

**Revenue Plan Attainment**
- Actual revenue / plan
- Target: 95-105%

**Gross Margin**
- % margin vs. plan
- Track variance drivers

**Working Capital**
- Cash tied up in inventory
- Optimize cash-to-cash cycle

### Process Metrics

**Meeting Effectiveness**
- % decisions made on time
- Action item completion rate
- Participant engagement scores

**Cycle Time**
- Time from data gathering to decision
- Target: <4 weeks

**Plan Stability**
- How often plan changes
- Nervousness metric

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SopMetrics:
    """S&OP performance metrics tracking"""

    def __init__(self):
        self.metrics_history = []

    def calculate_monthly_metrics(self, actual_data, plan_data):
        """
        Calculate S&OP metrics for a month

        Parameters:
        - actual_data: actual performance
        - plan_data: planned performance
        """

        # Forecast accuracy (MAPE)
        mape = np.mean(
            np.abs((actual_data['demand'] - plan_data['forecast']) /
                   actual_data['demand'])
        ) * 100

        # Bias
        bias = np.mean(plan_data['forecast'] - actual_data['demand'])
        bias_pct = (bias / np.mean(actual_data['demand'])) * 100

        # Plan attainment
        plan_attainment = (
            actual_data['production'].sum() / plan_data['production_plan'].sum()
        ) * 100

        # Inventory metrics
        inventory_turns = (
            actual_data['cogs'].sum() /
            actual_data['avg_inventory'].mean()
        )

        days_of_supply = (
            actual_data['ending_inventory'].iloc[-1] /
            (actual_data['demand'].sum() / 30)
        )

        # Revenue attainment
        revenue_attainment = (
            actual_data['revenue'].sum() / plan_data['revenue_plan'].sum()
        ) * 100

        metrics = {
            'period': actual_data['period'].iloc[0],
            'mape': mape,
            'bias_pct': bias_pct,
            'plan_attainment': plan_attainment,
            'inventory_turns': inventory_turns,
            'days_of_supply': days_of_supply,
            'revenue_attainment': revenue_attainment
        }

        self.metrics_history.append(metrics)

        return metrics

    def plot_metrics_dashboard(self):
        """Create S&OP metrics dashboard"""

        if not self.metrics_history:
            return None

        df = pd.DataFrame(self.metrics_history)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('S&OP Performance Dashboard', fontsize=16, fontweight='bold')

        # MAPE
        axes[0, 0].plot(df['period'], df['mape'], marker='o', linewidth=2)
        axes[0, 0].axhline(y=20, color='r', linestyle='--', label='Target')
        axes[0, 0].set_title('Forecast Accuracy (MAPE)')
        axes[0, 0].set_ylabel('MAPE (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Bias
        axes[0, 1].plot(df['period'], df['bias_pct'], marker='o', linewidth=2, color='orange')
        axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=-5, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Forecast Bias')
        axes[0, 1].set_ylabel('Bias (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plan Attainment
        axes[0, 2].plot(df['period'], df['plan_attainment'], marker='o', linewidth=2, color='green')
        axes[0, 2].axhline(y=95, color='r', linestyle='--', label='Min Target')
        axes[0, 2].axhline(y=105, color='r', linestyle='--')
        axes[0, 2].set_title('Plan Attainment')
        axes[0, 2].set_ylabel('Attainment (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Inventory Turns
        axes[1, 0].plot(df['period'], df['inventory_turns'], marker='o', linewidth=2, color='purple')
        axes[1, 0].axhline(y=8, color='g', linestyle='--', label='Target')
        axes[1, 0].set_title('Inventory Turns')
        axes[1, 0].set_ylabel('Turns')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Days of Supply
        axes[1, 1].plot(df['period'], df['days_of_supply'], marker='o', linewidth=2, color='brown')
        axes[1, 1].axhline(y=45, color='g', linestyle='--', label='Target')
        axes[1, 1].set_title('Days of Supply')
        axes[1, 1].set_ylabel('Days')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Revenue Attainment
        axes[1, 2].plot(df['period'], df['revenue_attainment'], marker='o', linewidth=2, color='red')
        axes[1, 2].axhline(y=95, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axhline(y=105, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].axhspan(95, 105, alpha=0.2, color='green')
        axes[1, 2].set_title('Revenue Attainment')
        axes[1, 2].set_ylabel('Attainment (%)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

# Example
metrics = SopMetrics()

# Simulate 6 months of data
for month in range(1, 7):
    actual = pd.DataFrame({
        'period': [f'2025-{month:02d}'],
        'demand': [10000 + np.random.randint(-500, 500)],
        'production': [10000 + np.random.randint(-300, 300)],
        'revenue': [1000000 + np.random.randint(-50000, 50000)],
        'cogs': [600000],
        'avg_inventory': [500000],
        'ending_inventory': [400000]
    })

    plan = pd.DataFrame({
        'forecast': [10000],
        'production_plan': [10000],
        'revenue_plan': [1000000]
    })

    monthly_metrics = metrics.calculate_monthly_metrics(actual, plan)
    print(f"Month {month} Metrics: MAPE={monthly_metrics['mape']:.1f}%")

# Plot dashboard
metrics.plot_metrics_dashboard()
```

---

## Tools & Libraries

### Python Libraries

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy`: Statistical analysis

**Optimization:**
- `pulp`: Linear programming
- `pyomo`: Advanced optimization

**Visualization:**
- `matplotlib`, `seaborn`: Charts
- `plotly`: Interactive dashboards
- `dash`: Web applications

### Commercial S&OP Software

**Enterprise Platforms:**
- **Kinaxis RapidResponse**: Leading S&OP/IBP platform
- **o9 Solutions**: AI-powered digital platform
- **SAP IBP**: Integrated Business Planning
- **Oracle Cloud Supply Chain Planning**: S&OP modules
- **Blue Yonder**: S&OP and demand-supply matching
- **Anaplan**: Connected planning platform
- **Logility**: Supply chain planning suite

**Collaboration Tools:**
- **Microsoft Teams**: Meeting and collaboration
- **Slack**: Async communication
- **Miro / Mural**: Virtual whiteboarding
- **Power BI / Tableau**: Visualization

### Excel / Google Sheets

Still widely used for S&OP in mid-size companies:
- Planning templates
- Scenario modeling
- Financial reconciliation
- Executive dashboards

---

## Common Challenges & Solutions

### Challenge: Poor Cross-Functional Collaboration

**Problem:**
- Silos between Sales and Operations
- Finger-pointing and blame
- Low meeting engagement

**Solutions:**
- Executive sponsorship and accountability
- Clear roles and responsibilities (RACI)
- Shared metrics and incentives
- Trust-building activities
- Professional facilitation

### Challenge: Data Quality and Integration

**Problem:**
- Multiple sources of truth
- Manual data gathering
- Errors and inconsistencies
- Time-consuming preparation

**Solutions:**
- Single integrated system
- Automated data feeds
- Data governance process
- Master data management
- Exception-based review

### Challenge: Short-Term Focus

**Problem:**
- Only look 1-3 months ahead
- Reactive vs. proactive
- Miss strategic issues

**Solutions:**
- Extend horizon to 18-24 months
- Monthly rolling forecasts
- Quarterly strategic reviews
- Long-term capacity planning
- New product integration

### Challenge: Meeting Overload

**Problem:**
- Too many meetings
- Repetitive discussions
- Decision fatigue

**Solutions:**
- Streamline to 5-step process
- Pre-work and preparation
- Clear agendas and time limits
- Delegate tactical decisions
- Exception-based reviews

### Challenge: Lack of Executive Engagement

**Problem:**
- Executives skip meetings
- Rubber-stamp decisions
- Don't see value

**Solutions:**
- Right level of detail (strategic, not tactical)
- Clear decision needs
- Show business impact
- Tie to financial results
- Success stories and wins

---

## Output Format

### S&OP Report Structure

**Executive Dashboard (1 page):**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Forecast Accuracy (MAPE) | 18.5% | <20% | ✓ |
| Plan Attainment | 96% | >95% | ✓ |
| Inventory Turns | 7.2 | 8.0 | ⚠ |
| Revenue Attainment | 102% | 95-105% | ✓ |

**Demand Plan by Product Family:**

| Family | Jan | Feb | Mar | Q1 Total | vs. Last Month | vs. Last Year |
|--------|-----|-----|-----|----------|----------------|---------------|
| Electronics | 11,000 | 10,200 | 10,500 | 31,700 | +5% | +12% |
| Appliances | 4,500 | 4,800 | 5,000 | 14,300 | +2% | +8% |

**Supply Plan & Constraints:**

| Family | Demand | Capacity | Gap | Status | Solution |
|--------|--------|----------|-----|--------|----------|
| Electronics | 11,000 | 10,000 | (1,000) | Constrained | Overtime approved |
| Appliances | 4,500 | 5,000 | +500 | OK | Normal production |

**Financial Summary:**

| Metric | This Month | Next Month | Q1 Plan |
|--------|-----------|------------|---------|
| Revenue | $1.5M | $1.4M | $4.3M |
| COGS | $0.9M | $0.85M | $2.6M |
| Gross Margin % | 40% | 39% | 39.5% |
| Inventory Value | $5.2M | $5.0M | $5.0M |

**Decisions Made:**
1. Approve Q1 demand plan with noted assumptions
2. Authorize overtime for Electronics in January
3. Proceed with capacity expansion business case
4. Review appliance pricing strategy by March

**Action Items:**
- Finalize contract with overtime agency (HR, Jan 20)
- Complete expansion business case (Ops, Feb 15)
- Pricing analysis for appliances (Marketing, Mar 1)

---

## Questions to Ask

If you need more context:
1. What's the current S&OP process maturity level?
2. Who participates in S&OP currently?
3. What's the planning horizon? (months ahead)
4. What systems/tools are used?
5. What are the biggest challenges or pain points?
6. Who owns the S&OP process?
7. How are forecasts created today?
8. What decisions typically need to be made?

---

## Related Skills

- **demand-forecasting**: For demand planning input to S&OP
- **capacity-planning**: For supply capacity analysis
- **master-production-scheduling**: For detailed execution
- **scenario-planning**: For risk analysis and alternatives
- **inventory-optimization**: For inventory strategy
- **supply-chain-analytics**: For metrics and KPIs
- **network-design**: For strategic network decisions
- **financial-planning**: For financial reconciliation
