---
name: network-flow-optimization
description: "When the user wants to optimize network flows, solve minimum cost flow problems, or design flow networks. Also use when the user mentions \"min cost flow,\" \"maximum flow,\" \"network flow problem,\" \"transportation problem,\" \"transshipment problem,\" \"multi-commodity flow,\" \"supply chain flow optimization,\" or \"network capacity planning.\" For facility location, see facility-location-problem. For distribution networks, see distribution-center-network."
---

# Network Flow Optimization

You are an expert in network flow optimization and graph-based supply chain problems. Your goal is to help optimize flows through networks to minimize costs, maximize throughput, or balance multiple objectives while respecting capacity constraints and flow conservation.

## Initial Assessment

Before optimizing network flows, understand:

1. **Network Type**
   - Transportation problem? (sources → destinations, single commodity)
   - Transshipment problem? (intermediate nodes allowed)
   - Min-cost flow? (minimize cost given supplies and demands)
   - Max flow? (maximize total flow from source to sink)
   - Multi-commodity flow? (multiple products sharing network)

2. **Network Structure**
   - How many nodes? (sources, intermediate, sinks)
   - How many arcs/edges?
   - Directed or undirected?
   - Node types: supply nodes, demand nodes, transshipment nodes?
   - Network layers or echelons?

3. **Flow Characteristics**
   - Single commodity or multi-commodity?
   - Splittable flows? (can split along multiple paths)
   - Flow units? (tons, pallets, vehicles, data packets)
   - Time dimension? (static or dynamic flows)

4. **Capacities and Costs**
   - Arc capacities? (upper bounds on flows)
   - Node capacities? (throughput limits)
   - Flow costs per unit?
   - Fixed costs for using arcs?
   - Economies of scale?

5. **Supplies and Demands**
   - Supply at source nodes?
   - Demand at destination nodes?
   - Balanced network? (total supply = total demand)
   - Excess supply or unmet demand allowed?

---

## Network Flow Problem Framework

### Problem Classification

**1. Transportation Problem**
```
m sources → n destinations
- Single commodity
- Direct shipments only
- Minimize total transportation cost
```

**2. Transshipment Problem**
```
Sources → Intermediate nodes → Destinations
- Allows intermediate stops
- More flexible routing
- Includes warehouses, hubs, cross-docks
```

**3. Minimum Cost Flow Problem**
```
General network with supplies, demands, costs, capacities
- Most general formulation
- Subsumes transportation and transshipment
- Linear programming problem
```

**4. Maximum Flow Problem**
```
Single source → Single sink
- Maximize total flow
- Subject to arc capacities
- Applications: network capacity, throughput
```

**5. Multi-Commodity Flow**
```
Multiple products/commodities sharing network
- Product-specific demands
- Shared arc capacities
- More complex but realistic
```

---

## Mathematical Formulations

### Minimum Cost Flow Problem

**Network:**
- G = (N, A): Directed graph with nodes N and arcs A

**Parameters:**
- b_i: Net supply at node i
  - b_i > 0: supply node
  - b_i < 0: demand node (demand = -b_i)
  - b_i = 0: transshipment node
- c_{ij}: Unit cost on arc (i,j)
- u_{ij}: Capacity on arc (i,j)
- l_{ij}: Lower bound on arc (i,j) (often 0)

**Decision Variables:**
- x_{ij}: Flow on arc (i,j)

**Objective Function:**
```
Minimize: Σ_{(i,j) ∈ A} c_{ij} × x_{ij}
```

**Constraints:**
```
1. Flow conservation at each node:
   Σ_{j:(i,j)∈A} x_{ij} - Σ_{j:(j,i)∈A} x_{ji} = b_i,  ∀i ∈ N

   (Outflow - Inflow = Net Supply)

2. Arc capacity constraints:
   l_{ij} ≤ x_{ij} ≤ u_{ij},  ∀(i,j) ∈ A

3. Non-negativity (if no lower bounds):
   x_{ij} ≥ 0,  ∀(i,j) ∈ A

4. Balanced network:
   Σ_{i∈N} b_i = 0
   (Total supply = Total demand)
```

**Properties:**
- Linear programming problem
- Polynomial-time solvable
- Network simplex very efficient

### Transportation Problem

**Simplified formulation:**
- m sources with supplies s_i
- n destinations with demands d_j
- Cost c_{ij} to ship from source i to destination j

**Variables:**
- x_{ij}: Amount shipped from source i to destination j

**Objective:**
```
Minimize: Σ_i Σ_j c_{ij} × x_{ij}
```

**Constraints:**
```
1. Supply constraints:
   Σ_j x_{ij} ≤ s_i,  ∀i (sources)

2. Demand constraints:
   Σ_i x_{ij} ≥ d_j,  ∀j (destinations)

3. Non-negativity:
   x_{ij} ≥ 0,  ∀i,j
```

### Multi-Commodity Flow

**Additional notation:**
- K: Set of commodities/products
- b_i^k: Supply/demand of commodity k at node i
- c_{ij}^k: Cost per unit of commodity k on arc (i,j)
- u_{ij}: Total capacity on arc (i,j) (shared)

**Variables:**
- x_{ij}^k: Flow of commodity k on arc (i,j)

**Objective:**
```
Minimize: Σ_k Σ_{(i,j)∈A} c_{ij}^k × x_{ij}^k
```

**Constraints:**
```
1. Flow conservation per commodity:
   Σ_j x_{ij}^k - Σ_j x_{ji}^k = b_i^k,  ∀i ∈ N, ∀k ∈ K

2. Shared arc capacity:
   Σ_k x_{ij}^k ≤ u_{ij},  ∀(i,j) ∈ A

3. Non-negativity:
   x_{ij}^k ≥ 0,  ∀(i,j) ∈ A, ∀k ∈ K
```

---

## Solution Methods

### 1. Minimum Cost Flow with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

def solve_min_cost_flow_nx(nodes, arcs, supplies, costs, capacities):
    """
    Solve minimum cost flow using NetworkX

    Args:
        nodes: list of node IDs
        arcs: list of (source, target) tuples
        supplies: dict {node: supply} (negative for demand)
        costs: dict {(source, target): cost}
        capacities: dict {(source, target): capacity}

    Returns:
        optimal flow solution
    """

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with demands
    for node in nodes:
        demand = -supplies.get(node, 0)  # NetworkX uses demand (negative supply)
        G.add_node(node, demand=demand)

    # Add arcs with costs and capacities
    for (i, j) in arcs:
        G.add_edge(i, j,
                  weight=costs.get((i,j), 0),
                  capacity=capacities.get((i,j), float('inf')))

    # Solve min cost flow
    try:
        flow_dict = nx.min_cost_flow(G)

        # Extract flows
        flows = {}
        for i in flow_dict:
            for j in flow_dict[i]:
                if flow_dict[i][j] > 0:
                    flows[(i,j)] = flow_dict[i][j]

        # Calculate total cost
        total_cost = nx.cost_of_flow(G, flow_dict)

        return {
            'status': 'Optimal',
            'flows': flows,
            'total_cost': total_cost,
            'flow_dict': flow_dict
        }

    except nx.NetworkXUnfeasible:
        return {'status': 'Infeasible'}

    except Exception as e:
        return {'status': f'Error: {str(e)}'}


# Example usage
if __name__ == "__main__":
    # Example: Supply chain network
    # 2 plants → 2 DCs → 3 customers

    nodes = ['Plant1', 'Plant2', 'DC1', 'DC2',
             'Customer1', 'Customer2', 'Customer3']

    # Define arcs (directed edges)
    arcs = [
        # Plant to DC
        ('Plant1', 'DC1'), ('Plant1', 'DC2'),
        ('Plant2', 'DC1'), ('Plant2', 'DC2'),
        # DC to Customer
        ('DC1', 'Customer1'), ('DC1', 'Customer2'), ('DC1', 'Customer3'),
        ('DC2', 'Customer1'), ('DC2', 'Customer2'), ('DC2', 'Customer3')
    ]

    # Supplies (positive) and demands (negative)
    supplies = {
        'Plant1': 100,
        'Plant2': 150,
        'DC1': 0,  # Transshipment
        'DC2': 0,  # Transshipment
        'Customer1': -80,
        'Customer2': -90,
        'Customer3': -80
    }

    # Transportation costs
    costs = {
        ('Plant1', 'DC1'): 10, ('Plant1', 'DC2'): 12,
        ('Plant2', 'DC1'): 8, ('Plant2', 'DC2'): 11,
        ('DC1', 'Customer1'): 5, ('DC1', 'Customer2'): 7, ('DC1', 'Customer3'): 6,
        ('DC2', 'Customer1'): 6, ('DC2', 'Customer2'): 5, ('DC2', 'Customer3'): 8
    }

    # Arc capacities
    capacities = {
        ('Plant1', 'DC1'): 80, ('Plant1', 'DC2'): 70,
        ('Plant2', 'DC1'): 90, ('Plant2', 'DC2'): 100,
        ('DC1', 'Customer1'): 60, ('DC1', 'Customer2'): 70, ('DC1', 'Customer3'): 50,
        ('DC2', 'Customer1'): 70, ('DC2', 'Customer2'): 60, ('DC2', 'Customer3'): 80
    }

    print("="*70)
    print("MINIMUM COST FLOW PROBLEM")
    print("="*70)
    print(f"Nodes: {len(nodes)}")
    print(f"Arcs: {len(arcs)}")
    print(f"Total supply: {sum(v for v in supplies.values() if v > 0)}")
    print(f"Total demand: {-sum(v for v in supplies.values() if v < 0)}")

    result = solve_min_cost_flow_nx(nodes, arcs, supplies, costs, capacities)

    print(f"\n{'='*70}")
    print(f"OPTIMAL SOLUTION")
    print(f"{'='*70}")
    print(f"Status: {result['status']}")
    print(f"Total Cost: ${result['total_cost']:,.2f}")

    print(f"\nOptimal Flows:")
    for (i, j), flow in result['flows'].items():
        cost = costs.get((i,j), 0)
        capacity = capacities.get((i,j), 'inf')
        print(f"  {i:12} → {j:12}: {flow:6.1f} units "
              f"(cost=${cost}, capacity={capacity})")
```

### 2. Transportation Problem with PuLP

```python
from pulp import *
import numpy as np

def solve_transportation_problem(sources, destinations, supplies,
                                demands, costs):
    """
    Solve Transportation Problem

    Args:
        sources: list of source IDs
        destinations: list of destination IDs
        supplies: dict {source: supply}
        demands: dict {destination: demand}
        costs: dict {(source, dest): unit_cost}

    Returns:
        optimal transportation plan
    """

    # Create problem
    prob = LpProblem("Transportation", LpMinimize)

    # Decision variables
    x = {}
    for i in sources:
        for j in destinations:
            x[i,j] = LpVariable(f"ship_{i}_{j}", lowBound=0, cat='Continuous')

    # Objective: Minimize total transportation cost
    prob += (
        lpSum([costs[i,j] * x[i,j] for i in sources for j in destinations]),
        "Total_Cost"
    )

    # Constraints

    # 1. Supply constraints (can't ship more than available)
    for i in sources:
        prob += (
            lpSum([x[i,j] for j in destinations]) <= supplies[i],
            f"Supply_{i}"
        )

    # 2. Demand constraints (must meet all demand)
    for j in destinations:
        prob += (
            lpSum([x[i,j] for i in sources]) >= demands[j],
            f"Demand_{j}"
        )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        shipments = {}
        for i in sources:
            for j in destinations:
                if x[i,j].varValue > 0.01:
                    shipments[i,j] = x[i,j].varValue

        # Calculate utilization
        source_utilization = {}
        for i in sources:
            total_shipped = sum(x[i,j].varValue for j in destinations)
            source_utilization[i] = (total_shipped / supplies[i]) * 100

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'shipments': shipments,
            'source_utilization': source_utilization,
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
sources = ['Factory_A', 'Factory_B', 'Factory_C']
destinations = ['Market_1', 'Market_2', 'Market_3', 'Market_4']

supplies = {
    'Factory_A': 200,
    'Factory_B': 300,
    'Factory_C': 250
}

demands = {
    'Market_1': 150,
    'Market_2': 200,
    'Market_3': 180,
    'Market_4': 170
}

# Randomly generate costs
np.random.seed(42)
costs = {}
for i in sources:
    for j in destinations:
        costs[i,j] = np.random.uniform(10, 50)

print("\n" + "="*70)
print("TRANSPORTATION PROBLEM")
print("="*70)
print(f"Sources: {len(sources)}")
print(f"Destinations: {len(destinations)}")
print(f"Total supply: {sum(supplies.values())}")
print(f"Total demand: {sum(demands.values())}")

result = solve_transportation_problem(sources, destinations, supplies,
                                     demands, costs)

print(f"\n{'='*70}")
print(f"OPTIMAL SOLUTION")
print(f"{'='*70}")
print(f"Status: {result['status']}")
print(f"Total Cost: ${result['total_cost']:,.2f}")

print(f"\nOptimal Shipments:")
for (i, j), quantity in result['shipments'].items():
    cost_per_unit = costs[i,j]
    total_arc_cost = quantity * cost_per_unit
    print(f"  {i} → {j}: {quantity:.1f} units "
          f"(@${cost_per_unit:.2f}/unit = ${total_arc_cost:,.2f})")

print(f"\nSource Utilization:")
for source, util in result['source_utilization'].items():
    print(f"  {source}: {util:.1f}% ({supplies[source]} available)")
```

### 3. Multi-Commodity Flow

```python
def solve_multi_commodity_flow(nodes, arcs, products, supplies, costs,
                               arc_capacities):
    """
    Solve Multi-Commodity Flow Problem

    Args:
        nodes: list of nodes
        arcs: list of (source, target) tuples
        products: list of product IDs
        supplies: dict {(node, product): supply} (negative for demand)
        costs: dict {(source, target, product): cost}
        arc_capacities: dict {(source, target): shared capacity}

    Returns:
        optimal multi-commodity flow
    """

    prob = LpProblem("Multi_Commodity_Flow", LpMinimize)

    # Decision variables: x[i,j,k] = flow of product k on arc (i,j)
    x = {}
    for (i, j) in arcs:
        for k in products:
            x[i,j,k] = LpVariable(f"flow_{i}_{j}_{k}",
                                 lowBound=0, cat='Continuous')

    # Objective: Minimize total cost across all products
    prob += (
        lpSum([costs.get((i,j,k), 0) * x[i,j,k]
               for (i,j) in arcs for k in products]),
        "Total_Cost"
    )

    # Constraints

    # 1. Flow conservation for each product at each node
    for node in nodes:
        for k in products:
            # Outflow - Inflow = Supply
            outflow = lpSum([x[i,j,k] for (i,j) in arcs if i == node])
            inflow = lpSum([x[i,j,k] for (i,j) in arcs if j == node])

            prob += (
                outflow - inflow == supplies.get((node, k), 0),
                f"Flow_Conservation_{node}_{k}"
            )

    # 2. Shared arc capacity constraints
    for (i, j) in arcs:
        prob += (
            lpSum([x[i,j,k] for k in products]) <= arc_capacities.get((i,j), float('inf')),
            f"Capacity_{i}_{j}"
        )

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        flows = {}
        for (i,j) in arcs:
            for k in products:
                if x[i,j,k].varValue > 0.01:
                    flows[i,j,k] = x[i,j,k].varValue

        # Calculate arc utilization
        arc_utilization = {}
        for (i,j) in arcs:
            total_flow = sum(x[i,j,k].varValue for k in products)
            capacity = arc_capacities.get((i,j), float('inf'))
            if capacity != float('inf'):
                arc_utilization[i,j] = (total_flow / capacity) * 100

        return {
            'status': LpStatus[prob.status],
            'total_cost': value(prob.objective),
            'flows': flows,
            'arc_utilization': arc_utilization,
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'solve_time': solve_time
        }


# Example usage
nodes = ['Plant', 'DC1', 'DC2', 'Customer1', 'Customer2']
arcs = [
    ('Plant', 'DC1'), ('Plant', 'DC2'),
    ('DC1', 'Customer1'), ('DC1', 'Customer2'),
    ('DC2', 'Customer1'), ('DC2', 'Customer2')
]
products = ['ProductA', 'ProductB', 'ProductC']

# Product-specific supplies/demands
supplies = {
    ('Plant', 'ProductA'): 100,
    ('Plant', 'ProductB'): 150,
    ('Plant', 'ProductC'): 120,
    ('Customer1', 'ProductA'): -40,
    ('Customer1', 'ProductB'): -60,
    ('Customer1', 'ProductC'): -50,
    ('Customer2', 'ProductA'): -60,
    ('Customer2', 'ProductB'): -90,
    ('Customer2', 'ProductC'): -70
}

# Product-specific costs
costs = {}
for (i,j) in arcs:
    for k in products:
        costs[i,j,k] = np.random.uniform(5, 25)

# Shared arc capacities
arc_capacities = {
    ('Plant', 'DC1'): 200,
    ('Plant', 'DC2'): 180,
    ('DC1', 'Customer1'): 100,
    ('DC1', 'Customer2'): 120,
    ('DC2', 'Customer1'): 110,
    ('DC2', 'Customer2'): 130
}

print("\n" + "="*70)
print("MULTI-COMMODITY FLOW PROBLEM")
print("="*70)
print(f"Nodes: {len(nodes)}")
print(f"Arcs: {len(arcs)}")
print(f"Products: {len(products)}")

result = solve_multi_commodity_flow(nodes, arcs, products, supplies,
                                   costs, arc_capacities)

print(f"\n{'='*70}")
print(f"OPTIMAL SOLUTION")
print(f"{'='*70}")
print(f"Status: {result['status']}")
print(f"Total Cost: ${result['total_cost']:,.2f}")

print(f"\nOptimal Flows (sample - first 15):")
count = 0
for (i,j,k), flow in result['flows'].items():
    if count >= 15:
        break
    cost_per_unit = costs[i,j,k]
    print(f"  {i} → {j} ({k}): {flow:.1f} units @${cost_per_unit:.2f}/unit")
    count += 1

print(f"\nArc Utilization:")
for (i,j), util in result['arc_utilization'].items():
    capacity = arc_capacities[i,j]
    print(f"  {i} → {j}: {util:.1f}% (capacity={capacity})")
```

---

## Advanced Algorithms

### 1. Maximum Flow (Ford-Fulkerson)

```python
def max_flow_ford_fulkerson(graph, source, sink):
    """
    Maximum flow using Ford-Fulkerson algorithm with BFS (Edmonds-Karp)

    Args:
        graph: dict {node: {neighbor: capacity}}
        source: source node
        sink: sink node

    Returns:
        maximum flow value and flow assignment
    """
    from collections import deque, defaultdict

    # Create residual graph
    residual = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for v in graph[u]:
            residual[u][v] = graph[u][v]

    def bfs_find_path():
        """Find augmenting path using BFS"""
        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            node, path = queue.popleft()

            if node == sink:
                return path

            for neighbor in residual[node]:
                if neighbor not in visited and residual[node][neighbor] > 0:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    max_flow_value = 0

    # Find augmenting paths
    while True:
        path = bfs_find_path()

        if path is None:
            break

        # Find bottleneck capacity
        flow = min(residual[path[i]][path[i+1]]
                  for i in range(len(path)-1))

        # Update residual graph
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            residual[u][v] -= flow
            residual[v][u] += flow

        max_flow_value += flow

    # Extract final flow
    flow_assignment = {}
    for u in graph:
        for v in graph[u]:
            flow_on_edge = graph[u][v] - residual[u][v]
            if flow_on_edge > 0:
                flow_assignment[u,v] = flow_on_edge

    return {
        'max_flow': max_flow_value,
        'flow_assignment': flow_assignment
    }


# Example
graph = {
    'S': {'A': 10, 'B': 5},
    'A': {'B': 15, 'C': 10},
    'B': {'D': 10},
    'C': {'D': 10, 'T': 5},
    'D': {'T': 10},
    'T': {}
}

result = max_flow_ford_fulkerson(graph, 'S', 'T')

print("\n" + "="*70)
print("MAXIMUM FLOW PROBLEM")
print("="*70)
print(f"Maximum Flow: {result['max_flow']}")
print(f"\nFlow Assignment:")
for (u,v), flow in result['flow_assignment'].items():
    print(f"  {u} → {v}: {flow}")
```

---

## Complete Network Flow Solver

```python
class NetworkFlowSolver:
    """
    Comprehensive Network Flow Optimization Solver
    """

    def __init__(self):
        self.problem_type = None
        self.loaded = False

    def load_min_cost_flow(self, nodes, arcs, supplies, costs, capacities):
        """Load minimum cost flow problem"""
        self.nodes = nodes
        self.arcs = arcs
        self.supplies = supplies
        self.costs = costs
        self.capacities = capacities
        self.problem_type = 'min_cost_flow'
        self.loaded = True

        print(f"Loaded Minimum Cost Flow Problem:")
        print(f"  Nodes: {len(nodes)}")
        print(f"  Arcs: {len(arcs)}")
        total_supply = sum(v for v in supplies.values() if v > 0)
        total_demand = -sum(v for v in supplies.values() if v < 0)
        print(f"  Total supply: {total_supply}")
        print(f"  Total demand: {total_demand}")
        print(f"  Balanced: {abs(total_supply - total_demand) < 0.01}")

    def load_transportation(self, sources, destinations, supplies,
                          demands, costs):
        """Load transportation problem"""
        self.sources = sources
        self.destinations = destinations
        self.supplies_trans = supplies
        self.demands_trans = demands
        self.costs_trans = costs
        self.problem_type = 'transportation'
        self.loaded = True

        print(f"Loaded Transportation Problem:")
        print(f"  Sources: {len(sources)}")
        print(f"  Destinations: {len(destinations)}")
        print(f"  Total supply: {sum(supplies.values())}")
        print(f"  Total demand: {sum(demands.values())}")

    def solve_exact(self):
        """Solve with exact method"""
        if not self.loaded:
            raise ValueError("Problem not loaded")

        if self.problem_type == 'min_cost_flow':
            return solve_min_cost_flow_nx(
                self.nodes, self.arcs, self.supplies,
                self.costs, self.capacities
            )

        elif self.problem_type == 'transportation':
            return solve_transportation_problem(
                self.sources, self.destinations,
                self.supplies_trans, self.demands_trans,
                self.costs_trans
            )

    def visualize_network(self, solution=None):
        """Visualize network and flows"""
        if self.problem_type != 'min_cost_flow':
            print("Visualization only for min cost flow currently")
            return

        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.DiGraph()

        # Add nodes
        for node in self.nodes:
            supply = self.supplies.get(node, 0)
            if supply > 0:
                G.add_node(node, node_type='supply')
            elif supply < 0:
                G.add_node(node, node_type='demand')
            else:
                G.add_node(node, node_type='transshipment')

        # Add arcs
        for (i, j) in self.arcs:
            G.add_edge(i, j)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        plt.figure(figsize=(14, 10))

        # Draw nodes by type
        supply_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'supply']
        demand_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'demand']
        trans_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'transshipment']

        nx.draw_networkx_nodes(G, pos, nodelist=supply_nodes,
                             node_color='lightgreen', node_size=800,
                             label='Supply Nodes')
        nx.draw_networkx_nodes(G, pos, nodelist=demand_nodes,
                             node_color='lightcoral', node_size=800,
                             label='Demand Nodes')
        nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes,
                             node_color='lightblue', node_size=800,
                             label='Transshipment')

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True,
                             arrowsize=20, width=2)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        # If solution provided, highlight flows
        if solution and 'flows' in solution:
            edge_labels = {}
            for (i, j), flow in solution['flows'].items():
                cost = self.costs.get((i,j), 0)
                edge_labels[(i,j)] = f"{flow:.0f}\n${cost}"

            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                        font_size=8)

        plt.title("Network Flow Visualization")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# Complete example
if __name__ == "__main__":
    print("="*70)
    print("NETWORK FLOW OPTIMIZATION - COMPREHENSIVE EXAMPLE")
    print("="*70)

    # Create sample network
    nodes = ['S1', 'S2', 'DC1', 'DC2', 'DC3', 'D1', 'D2', 'D3']

    arcs = [
        ('S1', 'DC1'), ('S1', 'DC2'),
        ('S2', 'DC2'), ('S2', 'DC3'),
        ('DC1', 'D1'), ('DC1', 'D2'),
        ('DC2', 'D1'), ('DC2', 'D2'), ('DC2', 'D3'),
        ('DC3', 'D2'), ('DC3', 'D3')
    ]

    supplies = {
        'S1': 150, 'S2': 200,
        'DC1': 0, 'DC2': 0, 'DC3': 0,
        'D1': -100, 'D2': -120, 'D3': -130
    }

    costs = {}
    for (i,j) in arcs:
        costs[i,j] = np.random.uniform(8, 30)

    capacities = {}
    for (i,j) in arcs:
        capacities[i,j] = np.random.uniform(60, 150)

    # Create solver
    solver = NetworkFlowSolver()
    solver.load_min_cost_flow(nodes, arcs, supplies, costs, capacities)

    # Solve
    print("\n" + "="*70)
    print("SOLVING...")
    print("="*70)

    solution = solver.solve_exact()

    print(f"\n{'='*70}")
    print(f"OPTIMAL SOLUTION")
    print(f"{'='*70}")
    print(f"Status: {solution['status']}")
    print(f"Total Cost: ${solution['total_cost']:,.2f}")

    print(f"\nOptimal Flows:")
    for (i,j), flow in sorted(solution['flows'].items()):
        cost = costs[i,j]
        capacity = capacities[i,j]
        util = (flow / capacity) * 100
        print(f"  {i:4} → {j:4}: {flow:6.1f}/{capacity:6.1f} "
              f"({util:5.1f}%) @${cost:.2f}/unit")

    # Visualize
    solver.visualize_network(solution)
```

---

## Tools & Libraries

### Python Libraries
- **NetworkX**: Graph algorithms, flow optimization
- **PuLP/Pyomo**: MIP formulation
- **OR-Tools**: Google network optimization
- **SciPy**: Sparse matrix operations
- **igraph**: Fast network analysis

### Commercial Software
- **CPLEX/Gurobi**: High-performance solvers
- **AIMMS**: Optimization modeling
- **AMPL**: Mathematical modeling

---

## Common Challenges & Solutions

**Large Networks:** Use specialized algorithms (network simplex), decomposition

**Integer Flows:** Add integrality constraints, use branch-and-bound

**Time-Varying Demands:** Dynamic network flows, time-expanded networks

**Uncertainty:** Stochastic optimization, robust optimization

**Multiple Objectives:** Multi-objective optimization, weighted objectives

---

## Output Format

**Network Flow Solution:**
- Total Cost: $X
- Maximum Flow: Y units
- Arc Utilization: Z%
- Flow Pattern: [detailed routing]

---

## Questions to Ask

1. Network structure? (nodes, arcs, capacities)
2. Single or multi-commodity?
3. Supplies and demands?
4. Cost structure?
5. Capacity constraints?
6. Time dimension?
7. Optimization objective?

---

## Related Skills

- **facility-location-problem**: Location decisions with flows
- **distribution-center-network**: Multi-echelon networks
- **vehicle-routing-problem**: Routing after flow allocation
- **inventory-routing-problem**: Integrated inventory-flow
- **optimization-modeling**: MIP formulation
- **hub-location-problem**: Hub-based flow networks

---
name: vehicle-routing-problem
description: When the user wants to solve the Vehicle Routing Problem (VRP), optimize multi-vehicle routes, or plan fleet delivery routes. Also use when the user mentions "VRP," "fleet routing," "multi-vehicle routing," "delivery route planning," "vehicle dispatch," "fleet optimization," or "route assignment." For single vehicle, see traveling-salesman-problem. For time windows, see vrp-time-windows.
---

# Vehicle Routing Problem (VRP)

You are an expert in the Vehicle Routing Problem and fleet optimization. Your goal is to help determine optimal routes for a fleet of vehicles to serve a set of customers, minimizing total distance/cost while respecting vehicle capacities and other constraints.

## Initial Assessment

Before solving VRP instances, understand:

1. **Fleet Characteristics**
   - How many vehicles available?
   - Vehicle capacities (weight, volume, pallets)?
   - Homogeneous fleet (all same) or heterogeneous?
   - Fixed costs per vehicle vs. variable costs?
   - Maximum route duration or distance?

2. **Customer Requirements**
   - How many customers to serve?
   - Customer demands (quantities)?
   - Service time at each location?
   - Any delivery time windows? → see **vrp-time-windows**
   - Pickup and delivery? → see **pickup-delivery-problem**

3. **Problem Scale**
   - Small (< 50 customers, < 5 vehicles): Exact methods possible
   - Medium (50-200 customers): Advanced heuristics
   - Large (200+ customers): Metaheuristics, decomposition

4. **Constraints**
   - Capacity constraints?
   - Maximum route length/duration?
   - Driver breaks required?
   - Depot open hours?
   - Multiple depots? → see **multi-depot-vrp**

5. **Objectives**
   - Minimize total distance?
   - Minimize number of vehicles?
   - Minimize total cost?
   - Balance routes?

---

## Mathematical Formulation

### Capacitated VRP (CVRP) - Two-Index Formulation

**Sets:**
- V = {0, 1, ..., n}: Set of nodes (0 = depot, 1..n = customers)
- K = {1, ..., m}: Set of vehicles

**Parameters:**
- c_{ij}: Cost/distance from node i to node j
- d_i: Demand at customer i
- Q_k: Capacity of vehicle k
- M: Number of available vehicles

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j, 0 otherwise

**Objective Function:**
```
Minimize: Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each customer visited exactly once:
   Σ_{k∈K} Σ_{i∈V} x_{ijk} = 1,  ∀j ∈ V\{0}

2. Flow conservation (what goes in must come out):
   Σ_{i∈V} x_{ihk} - Σ_{j∈V} x_{hjk} = 0,  ∀h ∈ V, ∀k ∈ K

3. Vehicle starts from depot:
   Σ_{j∈V\{0}} x_{0jk} = 1,  ∀k ∈ K

4. Vehicle returns to depot:
   Σ_{i∈V\{0}} x_{i0k} = 1,  ∀k ∈ K

5. Capacity constraint:
   Σ_{i∈V\{0}} Σ_{j∈V} d_i * x_{ijk} ≤ Q_k,  ∀k ∈ K

6. Subtour elimination (various formulations):
   - MTZ constraints
   - Flow-based constraints
   - Cutset constraints

7. Binary variables:
   x_{ijk} ∈ {0,1},  ∀i,j ∈ V, ∀k ∈ K
```

### Vehicle Minimization Formulation

When minimizing number of vehicles is primary objective:

```
Minimize: Σ_{k∈K} Σ_{j∈V\{0}} x_{0jk} + α * Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

Where α is a small weight on total distance (secondary objective).

---

## Exact Algorithms

### 1. Branch-and-Cut with Set Partitioning

```python
from pulp import *
import numpy as np

def vrp_branch_and_cut_simple(dist_matrix, demands, vehicle_capacity,
                              num_vehicles, depot=0):
    """
    VRP using Branch-and-Cut (simplified version)

    Suitable for small-medium instances (up to 50 customers)

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands for each customer
        vehicle_capacity: capacity of each vehicle
        num_vehicles: number of available vehicles
        depot: depot node index

    Returns:
        dict with solution details
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Create problem
    prob = LpProblem("CVRP", LpMinimize)

    # Decision variables: x[i,j,k] = 1 if vehicle k goes from i to j
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    x[i,j,k] = LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    # Objective: Minimize total distance
    prob += lpSum([dist_matrix[i][j] * x[i,j,k]
                   for i in range(n) for j in range(n) if i != j
                   for k in range(num_vehicles)]), "Total_Distance"

    # Constraints

    # 1. Each customer visited exactly once
    for j in customers:
        prob += lpSum([x[i,j,k] for i in range(n) if i != j
                      for k in range(num_vehicles)]) == 1, \
                f"Visit_{j}"

    # 2. Flow conservation
    for h in range(n):
        for k in range(num_vehicles):
            prob += (lpSum([x[i,h,k] for i in range(n) if i != h]) ==
                    lpSum([x[h,j,k] for j in range(n) if j != h])), \
                    f"Flow_{h}_{k}"

    # 3. Each vehicle leaves depot at most once
    for k in range(num_vehicles):
        prob += lpSum([x[depot,j,k] for j in customers]) <= 1, \
                f"Leave_Depot_{k}"

    # 4. Each vehicle returns to depot at most once
    for k in range(num_vehicles):
        prob += lpSum([x[i,depot,k] for i in customers]) <= 1, \
                f"Return_Depot_{k}"

    # 5. Capacity constraints (using flow-based formulation)
    for k in range(num_vehicles):
        prob += lpSum([demands[j] * lpSum([x[i,j,k] for i in range(n) if i != j])
                      for j in customers]) <= vehicle_capacity, \
                f"Capacity_{k}"

    # 6. Subtour elimination (MTZ-style)
    u = {}
    for i in customers:
        for k in range(num_vehicles):
            u[i,k] = LpVariable(f"u_{i}_{k}", lowBound=0,
                               upBound=vehicle_capacity, cat='Continuous')

    for k in range(num_vehicles):
        for i in customers:
            for j in customers:
                if i != j:
                    prob += (u[i,k] - u[j,k] + vehicle_capacity * x[i,j,k] <=
                            vehicle_capacity - demands[j]), \
                            f"Subtour_{i}_{j}_{k}"

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] == 'Optimal' or LpStatus[prob.status] == 'Feasible':
        routes = [[] for _ in range(num_vehicles)]

        for k in range(num_vehicles):
            # Build route for vehicle k
            current = depot
            route = [depot]

            while True:
                next_node = None
                for j in range(n):
                    if j != current and (current,j,k) in x:
                        if x[current,j,k].varValue > 0.5:
                            next_node = j
                            break

                if next_node is None or next_node == depot:
                    route.append(depot)
                    break

                route.append(next_node)
                current = next_node

            if len(route) > 2:  # Route has customers
                routes[k] = route

        # Remove empty routes
        routes = [r for r in routes if len(r) > 2]

        return {
            'status': LpStatus[prob.status],
            'total_distance': value(prob.objective),
            'routes': routes,
            'num_vehicles_used': len(routes),
            'solve_time': solve_time
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'total_distance': None,
            'routes': None,
            'solve_time': solve_time
        }


# Example usage
if __name__ == "__main__":
    # Example: 10 customers + 1 depot
    np.random.seed(42)

    # Generate random coordinates
    coords = np.random.rand(11, 2) * 100

    # Calculate distance matrix
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

    # Customer demands (depot has 0 demand)
    demands = [0] + [np.random.randint(5, 20) for _ in range(10)]

    vehicle_capacity = 50
    num_vehicles = 3

    result = vrp_branch_and_cut_simple(dist_matrix, demands,
                                      vehicle_capacity, num_vehicles)

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Number of Vehicles Used: {result['num_vehicles_used']}")
    print("\nRoutes:")
    for i, route in enumerate(result['routes']):
        route_demand = sum(demands[j] for j in route[1:-1])
        print(f"  Vehicle {i+1}: {route}")
        print(f"    Demand: {route_demand}/{vehicle_capacity}")
```

---

## Classical Heuristics

### 1. Clarke-Wright Savings Algorithm

```python
def clarke_wright_savings(dist_matrix, demands, vehicle_capacity, depot=0):
    """
    Clarke-Wright Savings Algorithm for VRP

    One of the most famous VRP heuristics

    Time complexity: O(n^2 log n)
    Quality: Good solutions, fast computation

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot: depot index

    Returns:
        dict with routes and total distance
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Calculate savings s_{ij} = d_{0i} + d_{0j} - d_{ij}
    savings = []
    for i in customers:
        for j in customers:
            if i < j:
                saving = (dist_matrix[depot][i] +
                         dist_matrix[depot][j] -
                         dist_matrix[i][j])
                savings.append((saving, i, j))

    # Sort savings in descending order
    savings.sort(reverse=True)

    # Initialize: each customer in separate route
    routes = [[depot, customer, depot] for customer in customers]
    route_demands = [demands[customer] for customer in customers]

    # Merge routes based on savings
    for saving, i, j in savings:
        # Find routes containing i and j
        route_i = None
        route_j = None

        for idx, route in enumerate(routes):
            if i in route:
                route_i = idx
            if j in route:
                route_j = idx

        if route_i is None or route_j is None:
            continue

        if route_i == route_j:
            continue  # Already in same route

        # Check if i and j are at ends of their routes
        # (can only merge if they're at route ends)
        route_i_data = routes[route_i]
        route_j_data = routes[route_j]

        i_at_end = (route_i_data[1] == i or route_i_data[-2] == i)
        j_at_end = (route_j_data[1] == j or route_j_data[-2] == j)

        if not (i_at_end and j_at_end):
            continue

        # Check capacity constraint
        combined_demand = route_demands[route_i] + route_demands[route_j]
        if combined_demand > vehicle_capacity:
            continue

        # Merge routes
        # Remove depots and merge
        route_i_interior = route_i_data[1:-1]
        route_j_interior = route_j_data[1:-1]

        # Determine merge order
        if route_i_interior[-1] == i and route_j_interior[0] == j:
            new_route = [depot] + route_i_interior + route_j_interior + [depot]
        elif route_i_interior[-1] == i and route_j_interior[-1] == j:
            new_route = [depot] + route_i_interior + route_j_interior[::-1] + [depot]
        elif route_i_interior[0] == i and route_j_interior[0] == j:
            new_route = [depot] + route_i_interior[::-1] + route_j_interior + [depot]
        elif route_i_interior[0] == i and route_j_interior[-1] == j:
            new_route = [depot] + route_j_interior + route_i_interior + [depot]
        else:
            continue

        # Update routes
        routes[route_i] = new_route
        route_demands[route_i] = combined_demand

        # Remove route_j
        del routes[route_j]
        del route_demands[route_j]

    # Calculate total distance
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i+1]]

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes),
        'route_demands': route_demands
    }
```

### 2. Sweep Algorithm

```python
def sweep_algorithm(coordinates, demands, vehicle_capacity, depot_idx=0):
    """
    Sweep Algorithm for VRP

    Works with polar coordinates - sweeps around depot

    Best for: Geographically clustered customers

    Args:
        coordinates: list of (x, y) tuples
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot_idx: depot index

    Returns:
        dict with routes and total distance
    """
    import math

    n = len(coordinates)
    depot = coordinates[depot_idx]

    # Calculate polar angles from depot
    angles = []
    for i, coord in enumerate(coordinates):
        if i == depot_idx:
            continue

        dx = coord[0] - depot[0]
        dy = coord[1] - depot[1]
        angle = math.atan2(dy, dx)
        angles.append((angle, i))

    # Sort customers by angle
    angles.sort()

    # Build routes by sweeping
    routes = []
    current_route = [depot_idx]
    current_load = 0

    for angle, customer in angles:
        if current_load + demands[customer] <= vehicle_capacity:
            current_route.append(customer)
            current_load += demands[customer]
        else:
            # Start new route
            current_route.append(depot_idx)
            routes.append(current_route)

            current_route = [depot_idx, customer]
            current_load = demands[customer]

    # Add last route
    if len(current_route) > 1:
        current_route.append(depot_idx)
        routes.append(current_route)

    # Calculate distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = math.sqrt(
                (coordinates[i][0] - coordinates[j][0])**2 +
                (coordinates[i][1] - coordinates[j][1])**2
            )

    # Calculate total distance
    total_distance = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i]][route[i+1]]

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

### 3. Sequential Insertion Heuristics

```python
def sequential_insertion_vrp(dist_matrix, demands, vehicle_capacity,
                            depot=0, insertion_criterion='cheapest'):
    """
    Sequential insertion heuristic for VRP

    Builds routes one at a time by inserting customers

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands
        vehicle_capacity: vehicle capacity
        depot: depot index
        insertion_criterion: 'cheapest', 'farthest', 'nearest'

    Returns:
        dict with routes and total distance
    """
    n = len(dist_matrix)
    customers = set(range(n)) - {depot}
    routes = []

    while customers:
        # Start new route
        route = [depot]
        route_load = 0

        # Select seed customer based on criterion
        if insertion_criterion == 'farthest':
            seed = max(customers, key=lambda c: dist_matrix[depot][c])
        elif insertion_criterion == 'nearest':
            seed = min(customers, key=lambda c: dist_matrix[depot][c])
        else:  # cheapest
            seed = min(customers)

        route.append(seed)
        route.append(depot)
        route_load += demands[seed]
        customers.remove(seed)

        # Insert remaining customers into this route
        while customers:
            best_customer = None
            best_position = None
            best_cost_increase = float('inf')

            # Try inserting each customer
            for customer in customers:
                if route_load + demands[customer] > vehicle_capacity:
                    continue

                # Try each position in route
                for pos in range(1, len(route)):
                    # Cost of inserting customer at position pos
                    cost_increase = (
                        dist_matrix[route[pos-1]][customer] +
                        dist_matrix[customer][route[pos]] -
                        dist_matrix[route[pos-1]][route[pos]]
                    )

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_customer = customer
                        best_position = pos

            if best_customer is None:
                break  # No more customers fit in this route

            # Insert best customer
            route.insert(best_position, best_customer)
            route_load += demands[best_customer]
            customers.remove(best_customer)

        routes.append(route)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes
    )

    return {
        'routes': routes,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

---

## Improvement Heuristics

### 1. Intra-Route Improvement (2-Opt)

```python
def intra_route_2opt(route, dist_matrix):
    """
    2-opt improvement within a single route

    Args:
        route: list of node indices
        dist_matrix: distance matrix

    Returns:
        improved route
    """
    improved = True
    best_route = route.copy()

    while improved:
        improved = False
        n = len(best_route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Calculate change in distance
                delta = (
                    dist_matrix[best_route[i-1]][best_route[j]] +
                    dist_matrix[best_route[i]][best_route[j+1]] -
                    dist_matrix[best_route[i-1]][best_route[i]] -
                    dist_matrix[best_route[j]][best_route[j+1]]
                )

                if delta < -1e-10:
                    # Improvement found - reverse segment
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    improved = True
                    break

            if improved:
                break

    return best_route


def improve_all_routes_2opt(routes, dist_matrix):
    """
    Apply 2-opt to all routes

    Args:
        routes: list of routes
        dist_matrix: distance matrix

    Returns:
        improved routes and total distance
    """
    improved_routes = []

    for route in routes:
        improved_route = intra_route_2opt(route, dist_matrix)
        improved_routes.append(improved_route)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in improved_routes
    )

    return {
        'routes': improved_routes,
        'total_distance': total_distance
    }
```

### 2. Inter-Route Improvement (Cross-Exchange)

```python
def cross_exchange(routes, dist_matrix, demands, vehicle_capacity):
    """
    Cross-exchange operator between routes

    Tries swapping segments between different routes

    Args:
        routes: list of routes
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(r1 + 1, num_routes):
                route1 = routes[r1]
                route2 = routes[r2]

                # Try swapping single customers
                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Check capacity feasibility
                        customer1 = route1[i]
                        customer2 = route2[j]

                        load1 = sum(demands[c] for c in route1[1:-1])
                        load2 = sum(demands[c] for c in route2[1:-1])

                        new_load1 = load1 - demands[customer1] + demands[customer2]
                        new_load2 = load2 - demands[customer2] + demands[customer1]

                        if (new_load1 > vehicle_capacity or
                            new_load2 > vehicle_capacity):
                            continue

                        # Calculate change in distance
                        # Current edges
                        current_cost = (
                            dist_matrix[route1[i-1]][route1[i]] +
                            dist_matrix[route1[i]][route1[i+1]] +
                            dist_matrix[route2[j-1]][route2[j]] +
                            dist_matrix[route2[j]][route2[j+1]]
                        )

                        # New edges after swap
                        new_cost = (
                            dist_matrix[route1[i-1]][customer2] +
                            dist_matrix[customer2][route1[i+1]] +
                            dist_matrix[route2[j-1]][customer1] +
                            dist_matrix[customer1][route2[j+1]]
                        )

                        if new_cost < current_cost - 1e-10:
                            # Perform swap
                            route1[i] = customer2
                            route2[j] = customer1
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes


def relocate_operator(routes, dist_matrix, demands, vehicle_capacity):
    """
    Relocate operator: move customer from one route to another

    Args:
        routes: list of routes
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(num_routes):
                if r1 == r2:
                    continue

                route1 = routes[r1]
                route2 = routes[r2]

                # Try moving each customer from route1 to route2
                for i in range(1, len(route1) - 1):
                    customer = route1[i]

                    # Check if route2 can accommodate this customer
                    load2 = sum(demands[c] for c in route2[1:-1])
                    if load2 + demands[customer] > vehicle_capacity:
                        continue

                    # Try inserting at each position in route2
                    for j in range(1, len(route2)):
                        # Calculate change in distance
                        # Removal cost
                        removal_cost = (
                            dist_matrix[route1[i-1]][route1[i]] +
                            dist_matrix[route1[i]][route1[i+1]] -
                            dist_matrix[route1[i-1]][route1[i+1]]
                        )

                        # Insertion cost
                        insertion_cost = (
                            dist_matrix[route2[j-1]][customer] +
                            dist_matrix[customer][route2[j]] -
                            dist_matrix[route2[j-1]][route2[j]]
                        )

                        delta = insertion_cost - removal_cost

                        if delta < -1e-10:
                            # Perform move
                            route2.insert(j, customer)
                            route1.pop(i)
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    # Remove empty routes
    routes = [r for r in routes if len(r) > 2]

    return routes
```

---

## Metaheuristics

### 1. Genetic Algorithm for VRP

```python
import random

def genetic_algorithm_vrp(dist_matrix, demands, vehicle_capacity,
                         max_vehicles, population_size=50,
                         generations=200, mutation_rate=0.15):
    """
    Genetic Algorithm for VRP

    Args:
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity
        max_vehicles: maximum number of vehicles
        population_size: GA population size
        generations: number of generations
        mutation_rate: mutation probability

    Returns:
        best solution found
    """
    n = len(dist_matrix)
    depot = 0
    customers = list(range(1, n))

    def decode_chromosome(chromosome):
        """
        Decode chromosome into routes

        Chromosome is a permutation of customers with
        route delimiters
        """
        routes = []
        current_route = [depot]
        current_load = 0

        for gene in chromosome:
            if gene < 0:  # Route delimiter
                if len(current_route) > 1:
                    current_route.append(depot)
                    routes.append(current_route)
                current_route = [depot]
                current_load = 0
            else:  # Customer
                if current_load + demands[gene] <= vehicle_capacity:
                    current_route.append(gene)
                    current_load += demands[gene]
                else:
                    # Start new route
                    current_route.append(depot)
                    routes.append(current_route)
                    current_route = [depot, gene]
                    current_load = demands[gene]

        if len(current_route) > 1:
            current_route.append(depot)
            routes.append(current_route)

        return routes

    def calculate_fitness(chromosome):
        """Calculate fitness (inverse of total distance)"""
        routes = decode_chromosome(chromosome)

        if len(routes) > max_vehicles:
            return 0  # Infeasible

        total_distance = sum(
            sum(dist_matrix[routes[i][j]][routes[i][j+1]]
                for j in range(len(routes[i])-1))
            for i in range(len(routes))
        )

        return 1.0 / (1.0 + total_distance)

    def create_individual():
        """Create random chromosome"""
        chromosome = customers.copy()
        random.shuffle(chromosome)

        # Insert random route delimiters
        num_delimiters = random.randint(0, max_vehicles - 1)
        positions = random.sample(range(len(chromosome)), num_delimiters)

        for pos in sorted(positions, reverse=True):
            chromosome.insert(pos, -1)  # -1 is route delimiter

        return chromosome

    def crossover(parent1, parent2):
        """Order crossover"""
        # Remove delimiters
        p1_customers = [g for g in parent1 if g >= 0]
        p2_customers = [g for g in parent2 if g >= 0]

        # Perform OX crossover
        size = len(p1_customers)
        start, end = sorted(random.sample(range(size), 2))

        child = [-2] * size
        child[start:end] = p1_customers[start:end]

        pos = end
        for gene in p2_customers[end:] + p2_customers[:end]:
            if gene not in child:
                if pos >= size:
                    pos = 0
                child[pos] = gene
                pos += 1

        # Add random delimiters
        num_delimiters = random.randint(0, max_vehicles - 1)
        positions = random.sample(range(len(child)), num_delimiters)

        for pos in sorted(positions, reverse=True):
            child.insert(pos, -1)

        return child

    def mutate(chromosome):
        """Swap mutation"""
        if random.random() < mutation_rate:
            # Get customer positions (not delimiters)
            customer_positions = [i for i, g in enumerate(chromosome) if g >= 0]

            if len(customer_positions) >= 2:
                i, j = random.sample(customer_positions, 2)
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

        return chromosome

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_chromosome = None
    best_fitness = 0

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_fitness(ind) for ind in population]

        # Track best
        max_fitness = max(fitnesses)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[fitnesses.index(max_fitness)].copy()

        # Selection and reproduction
        new_population = []

        # Elitism
        elite_count = int(0.1 * population_size)
        elite_indices = sorted(range(len(fitnesses)),
                             key=lambda i: fitnesses[i],
                             reverse=True)[:elite_count]
        new_population = [population[i].copy() for i in elite_indices]

        # Create offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = max(random.sample(list(zip(population, fitnesses)), 3),
                         key=lambda x: x[1])[0]
            parent2 = max(random.sample(list(zip(population, fitnesses)), 3),
                         key=lambda x: x[1])[0]

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    # Decode best solution
    best_routes = decode_chromosome(best_chromosome)

    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in best_routes
    )

    return {
        'routes': best_routes,
        'total_distance': total_distance,
        'num_vehicles': len(best_routes)
    }
```

### 2. Large Neighborhood Search (LNS)

```python
def large_neighborhood_search_vrp(dist_matrix, demands, vehicle_capacity,
                                 initial_routes, max_iterations=100,
                                 destroy_size=0.3):
    """
    Large Neighborhood Search for VRP

    Destroys and repairs parts of the solution iteratively

    Args:
        dist_matrix: distance matrix
        demands: customer demands
        vehicle_capacity: vehicle capacity
        initial_routes: initial solution
        max_iterations: number of iterations
        destroy_size: fraction of customers to remove (0-1)

    Returns:
        improved solution
    """
    import copy

    def calculate_cost(routes):
        return sum(
            sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
            for route in routes
        )

    def shaw_removal(routes, num_remove):
        """
        Shaw removal: remove similar customers
        """
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])

        if not all_customers:
            return routes, []

        # Select seed customer randomly
        seed = random.choice(all_customers)

        # Calculate relatedness (based on distance)
        relatedness = []
        for customer in all_customers:
            if customer != seed:
                similarity = dist_matrix[seed][customer]
                relatedness.append((similarity, customer))

        relatedness.sort()

        # Remove most related customers
        to_remove = {seed}
        for _, customer in relatedness[:num_remove-1]:
            to_remove.add(customer)

        # Remove from routes
        new_routes = []
        for route in routes:
            new_route = [route[0]]
            for customer in route[1:-1]:
                if customer not in to_remove:
                    new_route.append(customer)
            new_route.append(route[-1])

            if len(new_route) > 2:
                new_routes.append(new_route)

        return new_routes, list(to_remove)

    def greedy_insertion(routes, removed_customers):
        """
        Greedily reinsert removed customers
        """
        depot = routes[0][0]
        uninserted = removed_customers.copy()

        while uninserted:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost_increase = float('inf')

            # Try inserting each customer
            for customer in uninserted:
                # Try each route
                for route_idx, route in enumerate(routes):
                    # Check capacity
                    current_load = sum(demands[c] for c in route[1:-1])
                    if current_load + demands[customer] > vehicle_capacity:
                        continue

                    # Try each position
                    for pos in range(1, len(route)):
                        cost_increase = (
                            dist_matrix[route[pos-1]][customer] +
                            dist_matrix[customer][route[pos]] -
                            dist_matrix[route[pos-1]][route[pos]]
                        )

                        if cost_increase < best_cost_increase:
                            best_cost_increase = cost_increase
                            best_customer = customer
                            best_route_idx = route_idx
                            best_position = pos

            if best_customer is None:
                # Create new route
                routes.append([depot, uninserted[0], depot])
                uninserted.pop(0)
            else:
                # Insert customer
                routes[best_route_idx].insert(best_position, best_customer)
                uninserted.remove(best_customer)

        return routes

    # LNS main loop
    current_routes = copy.deepcopy(initial_routes)
    current_cost = calculate_cost(current_routes)

    best_routes = copy.deepcopy(current_routes)
    best_cost = current_cost

    all_customers = []
    for route in current_routes:
        all_customers.extend(route[1:-1])

    num_remove = max(1, int(len(all_customers) * destroy_size))

    for iteration in range(max_iterations):
        # Destroy
        partial_routes, removed = shaw_removal(current_routes, num_remove)

        # Repair
        new_routes = greedy_insertion(partial_routes, removed)

        # Evaluate
        new_cost = calculate_cost(new_routes)

        # Accept or reject (simulated annealing acceptance)
        temp = 100 * (1 - iteration / max_iterations)
        if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / temp):
            current_routes = new_routes
            current_cost = new_cost

            if new_cost < best_cost:
                best_routes = copy.deepcopy(new_routes)
                best_cost = new_cost

    return {
        'routes': best_routes,
        'total_distance': best_cost,
        'num_vehicles': len(best_routes)
    }
```

---

## Using OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_vrp_ortools(dist_matrix, demands, vehicle_capacities,
                     num_vehicles, depot=0, time_limit=30):
    """
    Solve VRP using Google OR-Tools

    Most practical and efficient approach for real-world problems

    Args:
        dist_matrix: n x n distance matrix
        demands: list of demands for each location
        vehicle_capacities: list of capacities (or single value)
        num_vehicles: number of vehicles
        depot: depot index
        time_limit: time limit in seconds

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)

    # Handle uniform fleet
    if isinstance(vehicle_capacities, (int, float)):
        vehicle_capacities = [vehicle_capacities] * num_vehicles

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 100)  # Scale for integer

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demand callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    # Add capacity constraints
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = True

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_distance = 0
        total_load = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            route_load = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                route_load += demands[node]
                index = solution.Value(routing.NextVar(index))

            route.append(manager.IndexToNode(index))

            if len(route) > 2:  # Route has customers
                routes.append(route)
                total_load += route_load

                # Calculate route distance
                route_distance = 0
                for i in range(len(route) - 1):
                    route_distance += dist_matrix[route[i]][route[i+1]]
                total_distance += route_distance

        return {
            'status': 'Optimal' if solution.ObjectiveValue() > 0 else 'Feasible',
            'routes': routes,
            'total_distance': total_distance,
            'num_vehicles_used': len(routes),
            'total_load': total_load,
            'objective_value': solution.ObjectiveValue() / 100.0
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Example usage with visualization
def visualize_vrp_solution(coordinates, routes, save_path=None):
    """
    Visualize VRP solution

    Args:
        coordinates: list of (x, y) coordinates
        routes: list of routes
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    # Plot routes
    for idx, route in enumerate(routes):
        route_coords = [coordinates[i] for i in route]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        plt.plot(xs, ys, 'o-', color=colors[idx],
                linewidth=2, markersize=8,
                label=f'Vehicle {idx+1}')

    # Plot depot
    depot = coordinates[0]
    plt.plot(depot[0], depot[1], 's', color='red',
            markersize=15, label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('VRP Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Complete example
if __name__ == "__main__":
    # Generate random problem
    np.random.seed(42)

    n = 21  # 1 depot + 20 customers
    coordinates = np.random.rand(n, 2) * 100

    # Distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Demands (depot has 0 demand)
    demands = [0] + [random.randint(5, 25) for _ in range(n-1)]

    vehicle_capacity = 100
    num_vehicles = 5

    print("Solving VRP with OR-Tools...")
    result = solve_vrp_ortools(dist_matrix, demands, vehicle_capacity,
                              num_vehicles, time_limit=30)

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Vehicles Used: {result['num_vehicles_used']}/{num_vehicles}")
    print(f"Total Load: {result['total_load']}")

    print("\nRoutes:")
    for i, route in enumerate(result['routes']):
        route_load = sum(demands[j] for j in route[1:-1])
        route_dist = sum(dist_matrix[route[j]][route[j+1]]
                        for j in range(len(route)-1))
        print(f"  Vehicle {i+1}: {route}")
        print(f"    Load: {route_load}/{vehicle_capacity}")
        print(f"    Distance: {route_dist:.2f}")

    # Visualize
    visualize_vrp_solution(coordinates, result['routes'])
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **OR-Tools (Google)**: Best for practical VRP (highly recommended)
- **PuLP**: MIP modeling
- **Pyomo**: Advanced modeling
- **VRPy**: VRP-specific library
- **python-tsp**: TSP and VRP utilities

**Metaheuristics:**
- **pymhlib**: Metaheuristic library
- **deap**: Evolutionary algorithms
- **pyswarms**: Particle swarm optimization

**Visualization:**
- **matplotlib**: Basic plots
- **plotly**: Interactive visualization
- **folium**: Map visualization with real coordinates

### Commercial Software

- **Gurobi**: State-of-art MIP solver
- **CPLEX**: IBM solver
- **Xpress**: FICO optimization
- **ORTEC**: Commercial routing software
- **Descartes**: Route planning software

### Cloud Services

- **Google Maps Platform**: Distance Matrix API
- **HERE Routing API**: Commercial routing
- **Azure Maps**: Microsoft routing service

---

## Common Challenges & Solutions

### Challenge: Problem Too Large

**Problem:**
- 500+ customers makes exact methods impractical
- Need solutions in minutes, not hours

**Solutions:**
- Use OR-Tools with time limits
- Clarke-Wright + 2-opt
- Decompose: cluster customers, solve subproblems
- Rolling horizon for dynamic problems

### Challenge: Heterogeneous Fleet

**Problem:**
- Different vehicle types (capacity, cost, speed)
- Different drivers, shifts

**Solutions:**
- Extend formulation with vehicle-specific parameters
- OR-Tools handles naturally with different capacity arrays
- Sequential approach: assign to vehicle types, then route

### Challenge: Time Windows

**Problem:**
- Customers have delivery time windows
- See **vrp-time-windows** skill

**Solutions:**
- Add temporal constraints to formulation
- Use specialized time window algorithms
- OR-Tools time window dimension

### Challenge: Real-World Distances

**Problem:**
- Euclidean distances unrealistic
- Need real road network

**Solutions:**
- Use Google Maps Distance Matrix API
- Pre-compute distance/time matrix
- Cache frequently used routes
- Consider traffic patterns

### Challenge: Dynamic Requests

**Problem:**
- New orders arrive during execution
- Need to reoptimize on the fly

**Solutions:**
- Keep routes with slack capacity
- Fast re-optimization heuristics
- Insertion methods for new customers
- Trigger re-optimization periodically

---

## Output Format

### VRP Solution Report

**Problem Instance:**
- Customers: 50
- Vehicles: 5 (capacity: 100 units each)
- Depot: Distribution Center
- Total demand: 487 units

**Solution Quality:**

| Metric | Value |
|--------|-------|
| Total Distance | 1,247.3 km |
| Vehicles Used | 5 / 5 |
| Average Route Distance | 249.5 km |
| Total Load | 487 / 500 units |
| Average Load per Vehicle | 97.4% |
| Solution Time | 8.3 seconds |

**Route Details:**

**Vehicle 1:** DC → C12 → C34 → C7 → C23 → DC
- Distance: 235.4 km
- Load: 98 / 100 units (98%)
- Customers: 4

**Vehicle 2:** DC → C5 → C18 → C42 → C31 → C9 → DC
- Distance: 278.9 km
- Load: 95 / 100 units (95%)
- Customers: 5

[Continue for all vehicles...]

**Statistics:**
- Average customers per route: 10.0
- Longest route: Vehicle 2 (278.9 km)
- Shortest route: Vehicle 5 (198.7 km)
- Total driving time: ~15.8 hours

---

## Questions to Ask

If you need more context:
1. How many customers need service? How many vehicles available?
2. What are the vehicle capacities? Homogeneous or heterogeneous fleet?
3. What units are demands measured in? (weight, volume, pallets)
4. Are there time windows for deliveries?
5. Is there a maximum route duration or distance?
6. Do you have coordinates or a distance matrix?
7. Fixed cost per vehicle or just distance-based?
8. Are there multiple depots?
9. Is this a one-time problem or recurring daily?
10. What's the acceptable solution time?

---

## Related Skills

- **traveling-salesman-problem**: For single vehicle routing
- **vrp-time-windows**: For VRP with time window constraints
- **capacitated-vrp**: For detailed CVRP formulations
- **multi-depot-vrp**: For problems with multiple depots
- **pickup-delivery-problem**: For pickup and delivery VRP
- **vrp-backhauls**: For VRP with pickups after deliveries
- **split-delivery-vrp**: For allowing multiple visits
- **route-optimization**: For practical routing applications
- **last-mile-delivery**: For final delivery mile optimization
- **fleet-management**: For overall fleet operations

---
name: vehicle-loading-optimization
description: When the user wants to optimize truck loading, load delivery vehicles, or plan vehicle capacity utilization. Also use when the user mentions "truck loading," "delivery vehicle optimization," "van loading," "cargo van packing," "multi-drop vehicle loading," "delivery route loading," "axle weight distribution," or "vehicle utilization." For container loading, see container-loading-optimization. For route optimization, see route-optimization.
---

# Vehicle Loading Optimization

You are an expert in vehicle loading optimization and logistics. Your goal is to help efficiently load trucks, vans, and delivery vehicles while maximizing utilization, ensuring safety, meeting weight constraints, and accommodating multi-stop delivery sequences.

## Initial Assessment

Before optimizing vehicle loading, understand:

1. **Vehicle Specifications**
   - Vehicle type? (box truck, van, semi-trailer, flatbed)
   - Cargo area dimensions: length x width x height
   - Weight capacity (GVWR - vehicle weight)
   - Axle weight limits (front/rear distribution)
   - Door configuration (rear, side, roll-up)?

2. **Cargo Characteristics**
   - Palletized, boxed, or loose cargo?
   - Pallet/package dimensions and weights
   - How many items/orders to load?
   - Stackability and fragility
   - Special handling requirements?

3. **Delivery Requirements**
   - Single destination or multi-stop route?
   - Delivery sequence (LIFO/FIFO)?
   - Unloading access (rear only, side door)?
   - Time windows at delivery locations?
   - Customer-specific requirements?

4. **Loading Constraints**
   - Weight distribution requirements
   - Axle weight limits for road compliance
   - Securing requirements (straps, nets, bars)
   - Climate control zones?
   - Hazmat separation rules?

5. **Optimization Goals**
   - Maximize vehicle utilization?
   - Minimize number of vehicles?
   - Ensure easy unloading sequence?
   - Balance weight distribution?
   - Minimize loading/unloading time?

---

## Vehicle Loading Framework

### Common Vehicle Types

**Cargo Van**
- Typical: Ford Transit, Mercedes Sprinter
- Cargo: 10-14ft L x 6ft W x 6ft H
- Capacity: 3,000-4,500 lbs
- Use: Last-mile delivery, service calls

**Box Truck (10-16ft)**
- Typical: 14ft box truck
- Cargo: 14ft L x 7ft W x 7ft H
- Capacity: 3,000-5,000 lbs
- Use: Local delivery, moving

**Box Truck (20-26ft)**
- Typical: 26ft box truck
- Cargo: 26ft L x 8ft W x 8ft H
- Capacity: 10,000-15,000 lbs
- Use: Regional delivery, freight

**Semi-Trailer (Dry Van)**
- Typical: 53ft trailer
- Cargo: 53ft L x 8.5ft W x 9ft H
- Capacity: 40,000-45,000 lbs
- Use: Long-haul, FTL shipments

### Key Loading Principles

**1. Axle Weight Distribution**
- Front axle: 12,000-13,000 lbs max (varies by state)
- Rear axle(s): 34,000 lbs max
- Gross Vehicle Weight Rating (GVWR): varies
- Target: 40% front, 60% rear (general rule)

**2. Load Sequence for Multi-Stop**
- Last stop items loaded first (LIFO)
- Accessible from door for each stop
- Zone-based loading by delivery sequence
- Minimize handling at each stop

**3. Weight Distribution**
- Heavy items at bottom, centered
- Distribute weight evenly side-to-side
- Avoid concentrated loads
- Balance front-to-back

**4. Load Securing**
- Prevent shifting during transport
- Use load bars, straps, or airbags
- Comply with DOT cargo securement rules
- Protect fragile items

---

## Mathematical Formulation

### Vehicle Loading Problem

**Decision Variables:**
- x_i, y_i, z_i = position of item i in vehicle
- v_i = vehicle assigned to item i
- s_i = delivery stop for item i
- used_j = 1 if vehicle j is used

**Objective Functions:**

1. **Minimize vehicles:**
   Minimize Σ used_j

2. **Maximize utilization:**
   Maximize (Σ volume_loaded) / (Σ vehicle_capacity)

3. **Minimize handling:**
   Minimize unloading moves at each stop

**Constraints:**
1. Vehicle capacity (volume and weight)
2. Axle weight limits
3. Loading sequence (multi-stop accessibility)
4. Weight distribution balance
5. Securing and safety requirements
6. Item compatibility

---

## Algorithms and Solution Methods

### Single-Stop Vehicle Loading

```python
class SingleStopVehicleLoader:
    """
    Optimize loading for single-destination delivery

    Uses 3D bin packing with vehicle-specific constraints
    """

    def __init__(self, vehicle_length, vehicle_width, vehicle_height,
                 weight_capacity, front_axle_limit=13000, rear_axle_limit=34000):
        """
        Initialize vehicle loader

        Parameters:
        - vehicle_length, vehicle_width, vehicle_height: cargo area dimensions (inches)
        - weight_capacity: max payload weight (lbs)
        - front_axle_limit, rear_axle_limit: axle weight limits (lbs)
        """

        self.vehicle_dims = (vehicle_length, vehicle_width, vehicle_height)
        self.weight_capacity = weight_capacity
        self.front_axle_limit = front_axle_limit
        self.rear_axle_limit = rear_axle_limit

        self.items = []
        self.solution = None

    def add_item(self, length, width, height, weight, item_id=None):
        """Add item to load"""
        if item_id is None:
            item_id = f"Item_{len(self.items)}"

        self.items.append({
            'id': item_id,
            'dims': (length, width, height),
            'weight': weight
        })

    def optimize_loading(self, algorithm='weight_balanced'):
        """
        Optimize vehicle loading

        Algorithms:
        - 'weight_balanced': Prioritize weight distribution
        - 'space_efficient': Maximize space utilization
        - 'easy_unload': Place items for easy access
        """

        L, W, H = self.vehicle_dims

        if algorithm == 'weight_balanced':
            # Sort by weight (heaviest first)
            sorted_items = sorted(self.items,
                                 key=lambda x: x['weight'],
                                 reverse=True)
        elif algorithm == 'space_efficient':
            # Sort by volume
            sorted_items = sorted(self.items,
                                 key=lambda x: x['dims'][0] * x['dims'][1] * x['dims'][2],
                                 reverse=True)
        else:
            sorted_items = self.items

        loaded_items = []
        current_weight = 0

        # Simple layer-based loading
        current_y = 0
        current_z = 0
        current_x = 0

        for item in sorted_items:
            l, w, h = item['dims']
            weight = item['weight']

            # Check weight
            if current_weight + weight > self.weight_capacity:
                continue  # Skip item if too heavy

            # Try to place item
            if current_x + l <= L:
                # Place in current row
                loaded_items.append({
                    'item': item,
                    'position': (current_x, current_y, current_z),
                    'dims': (l, w, h)
                })
                current_x += l
                current_weight += weight

            elif current_y + w <= W:
                # Start new row
                current_x = 0
                current_y += w
                loaded_items.append({
                    'item': item,
                    'position': (current_x, current_y, current_z),
                    'dims': (l, w, h)
                })
                current_x += l
                current_weight += weight

            elif current_z + h <= H:
                # Start new layer
                current_x = 0
                current_y = 0
                current_z += h
                loaded_items.append({
                    'item': item,
                    'position': (current_x, current_y, current_z),
                    'dims': (l, w, h)
                })
                current_x += l
                current_weight += weight

        # Check axle weights
        axle_check = self.check_axle_weights(loaded_items)

        self.solution = {
            'loaded_items': loaded_items,
            'total_weight': current_weight,
            'items_loaded': len(loaded_items),
            'items_not_loaded': len(self.items) - len(loaded_items),
            'utilization': self.calculate_utilization(loaded_items),
            'axle_weights': axle_check
        }

        return self.solution

    def check_axle_weights(self, loaded_items):
        """
        Calculate axle weight distribution

        Assumes:
        - Front axle at 0 (front of vehicle)
        - Rear axle at 60% of vehicle length
        """

        L = self.vehicle_dims[0]
        rear_axle_position = L * 0.6

        front_axle_weight = 0
        rear_axle_weight = 0

        for item_data in loaded_items:
            pos = item_data['position']
            dims = item_data['dims']
            weight = item_data['item']['weight']

            # Calculate center of mass
            com_x = pos[0] + dims[0] / 2

            # Distance from axles
            distance_from_rear = rear_axle_position - com_x

            if distance_from_rear > 0:
                # Weight forward of rear axle - distributes to both
                front_ratio = distance_from_rear / rear_axle_position
                front_axle_weight += weight * front_ratio
                rear_axle_weight += weight * (1 - front_ratio)
            else:
                # Weight behind rear axle - all on rear
                rear_axle_weight += weight

        return {
            'front_axle': front_axle_weight,
            'rear_axle': rear_axle_weight,
            'front_limit': self.front_axle_limit,
            'rear_limit': self.rear_axle_limit,
            'front_ok': front_axle_weight <= self.front_axle_limit,
            'rear_ok': rear_axle_weight <= self.rear_axle_limit,
            'balanced': abs(front_axle_weight - rear_axle_weight) / (front_axle_weight + rear_axle_weight) < 0.3
        }

    def calculate_utilization(self, loaded_items):
        """Calculate volume utilization"""
        L, W, H = self.vehicle_dims
        vehicle_volume = L * W * H

        loaded_volume = sum(
            item_data['dims'][0] * item_data['dims'][1] * item_data['dims'][2]
            for item_data in loaded_items
        )

        return (loaded_volume / vehicle_volume * 100) if vehicle_volume > 0 else 0

    def print_solution(self):
        """Print loading solution"""
        if not self.solution:
            print("No solution available")
            return

        print("=" * 70)
        print("VEHICLE LOADING SOLUTION")
        print("=" * 70)
        print(f"Vehicle: {self.vehicle_dims[0]}L x {self.vehicle_dims[1]}W x {self.vehicle_dims[2]}H in")
        print(f"Weight Capacity: {self.weight_capacity:,} lbs")
        print()
        print(f"Items Loaded: {self.solution['items_loaded']} / {len(self.items)}")
        print(f"Total Weight: {self.solution['total_weight']:,} lbs")
        print(f"Utilization: {self.solution['utilization']:.1f}%")
        print()
        print("Axle Weight Distribution:")
        axle = self.solution['axle_weights']
        print(f"  Front Axle: {axle['front_axle']:,.0f} lbs "
              f"({'OK' if axle['front_ok'] else 'OVER LIMIT'})")
        print(f"  Rear Axle: {axle['rear_axle']:,.0f} lbs "
              f"({'OK' if axle['rear_ok'] else 'OVER LIMIT'})")
        print(f"  Balanced: {'Yes' if axle['balanced'] else 'No - adjust load'}")


# Example usage
if __name__ == "__main__":
    # 26ft box truck
    loader = SingleStopVehicleLoader(
        vehicle_length=312,  # 26ft in inches
        vehicle_width=96,    # 8ft
        vehicle_height=96,   # 8ft
        weight_capacity=10000  # lbs
    )

    # Add items
    loader.add_item(48, 40, 60, 800, "Pallet_1")
    loader.add_item(48, 40, 50, 700, "Pallet_2")
    loader.add_item(48, 40, 55, 750, "Pallet_3")
    loader.add_item(36, 30, 40, 500, "Box_1")
    loader.add_item(36, 30, 40, 500, "Box_2")

    # Optimize
    solution = loader.optimize_loading(algorithm='weight_balanced')
    loader.print_solution()
```

### Multi-Stop Vehicle Loading

```python
class MultiStopVehicleLoader:
    """
    Optimize vehicle loading for multi-stop delivery routes

    Ensures items are accessible in delivery sequence
    """

    def __init__(self, vehicle_dims, weight_capacity):
        self.vehicle_dims = vehicle_dims
        self.weight_capacity = weight_capacity
        self.stops = []  # List of delivery stops
        self.solution = None

    def add_stop(self, stop_id, items, delivery_sequence):
        """
        Add delivery stop with items

        Parameters:
        - stop_id: stop identifier
        - items: list of item dicts with 'dims' and 'weight'
        - delivery_sequence: order in route (1, 2, 3, ...)
        """

        self.stops.append({
            'id': stop_id,
            'items': items,
            'sequence': delivery_sequence
        })

    def optimize_loading(self):
        """
        Optimize loading for multi-stop delivery

        Strategy:
        - Zone vehicle by delivery sequence
        - Last stop loaded first (at door)
        - Earlier stops loaded deeper in vehicle
        """

        # Sort stops by reverse delivery sequence
        sorted_stops = sorted(self.stops,
                            key=lambda s: s['sequence'],
                            reverse=True)

        L, W, H = self.vehicle_dims

        # Calculate zones
        num_stops = len(sorted_stops)
        zone_length = L / num_stops

        zones = []
        current_weight = 0

        for idx, stop in enumerate(sorted_stops):
            zone_start = idx * zone_length
            zone_end = (idx + 1) * zone_length

            zone_items = []

            # Load items for this stop in this zone
            for item in stop['items']:
                if current_weight + item['weight'] <= self.weight_capacity:
                    # Simple placement (could be more sophisticated)
                    zone_items.append({
                        'item': item,
                        'stop_id': stop['id'],
                        'zone': (zone_start, zone_end)
                    })
                    current_weight += item['weight']

            zones.append({
                'stop': stop,
                'zone': (zone_start, zone_end),
                'items': zone_items
            })

        self.solution = {
            'zones': zones,
            'total_weight': current_weight,
            'stops_loaded': len(zones),
            'total_items': sum(len(z['items']) for z in zones)
        }

        return self.solution

    def print_solution(self):
        """Print multi-stop loading plan"""
        if not self.solution:
            print("No solution available")
            return

        print("=" * 70)
        print("MULTI-STOP VEHICLE LOADING PLAN")
        print("=" * 70)
        print(f"Total Weight: {self.solution['total_weight']:,} lbs")
        print(f"Stops: {self.solution['stops_loaded']}")
        print(f"Total Items: {self.solution['total_items']}")
        print()

        for zone_data in self.solution['zones']:
            stop = zone_data['stop']
            zone = zone_data['zone']
            print(f"Stop {stop['sequence']}: {stop['id']}")
            print(f"  Zone: {zone[0]:.0f}-{zone[1]:.0f} inches from front")
            print(f"  Items: {len(zone_data['items'])}")
            print()
```

### Fleet Loading Optimization

```python
def optimize_fleet_loading(orders, vehicles):
    """
    Optimize loading across multiple vehicles

    Assigns orders to vehicles to minimize:
    - Number of vehicles used
    - Total distance traveled
    - Loading/unloading complexity

    Parameters:
    - orders: list of order dicts with items, destination, priority
    - vehicles: list of available vehicle specs

    Returns: assignment of orders to vehicles
    """

    from pulp import *

    n_orders = len(orders)
    n_vehicles = len(vehicles)

    # Create problem
    prob = LpProblem("Fleet_Loading", LpMinimize)

    # Decision variables
    # x[i,j] = 1 if order i assigned to vehicle j
    x = LpVariable.dicts("assign",
                        [(i, j) for i in range(n_orders) for j in range(n_vehicles)],
                        cat='Binary')

    # y[j] = 1 if vehicle j is used
    y = LpVariable.dicts("use_vehicle", range(n_vehicles), cat='Binary')

    # Objective: Minimize vehicles used + routing cost
    prob += lpSum([y[j] * vehicles[j]['cost']
                   for j in range(n_vehicles)]), "Total_Cost"

    # Constraints

    # 1. Each order assigned to exactly one vehicle
    for i in range(n_orders):
        prob += lpSum([x[i,j] for j in range(n_vehicles)]) == 1, f"Order_{i}"

    # 2. Vehicle capacity (weight)
    for j in range(n_vehicles):
        prob += (lpSum([orders[i]['weight'] * x[i,j] for i in range(n_orders)]) <=
                vehicles[j]['weight_capacity']), f"Weight_{j}"

    # 3. Vehicle capacity (volume)
    for j in range(n_vehicles):
        prob += (lpSum([orders[i]['volume'] * x[i,j] for i in range(n_orders)]) <=
                vehicles[j]['volume_capacity']), f"Volume_{j}"

    # 4. Vehicle used if orders assigned
    for j in range(n_vehicles):
        for i in range(n_orders):
            prob += x[i,j] <= y[j], f"VehicleUsed_{i}_{j}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract solution
    assignments = [[] for _ in range(n_vehicles)]

    for i in range(n_orders):
        for j in range(n_vehicles):
            if x[i,j].varValue and x[i,j].varValue > 0.5:
                assignments[j].append(i)

    return {
        'status': LpStatus[prob.status],
        'vehicles_used': sum(1 for a in assignments if a),
        'assignments': assignments
    }
```

---

## Common Challenges & Solutions

### Challenge: Axle Weight Violations

**Problem:**
- Front or rear axle exceeds legal limit
- Roadside inspection failure
- Safety issues

**Solutions:**
- Use axle weight calculation in optimization
- Place heavy items between axles
- Avoid loading too much at front or rear
- Use load bars to shift weight
- Consider adding/removing items

### Challenge: Multi-Stop Accessibility

**Problem:**
- Items for early stops buried deep
- Need to unload/reload at each stop
- Time wasted, risk of damage

**Solutions:**
- Zone loading by delivery sequence
- Last stop at door, first stop at front
- Use vertical stacking per zone
- Load light/small items on top for easy removal
- Consider side-door access vehicles

### Challenge: Mixed Item Sizes

**Problem:**
- Pallets + loose boxes
- Different heights cause wasted space
- Difficult to secure

**Solutions:**
- Group similar items together
- Use pallets as base for loose items
- Fill gaps with soft goods or dunnage
- Stack smaller items on larger bases
- Use load nets or straps

---

## Output Format

### Vehicle Loading Report

**Vehicle: 26ft Box Truck**
- Dimensions: 26'L x 8'W x 8'H
- Weight Capacity: 12,000 lbs
- Route: 5 stops

**Loading Plan:**

Zone 1 (0-5ft) - Stop 5 (Last):
- 3 pallets: P045, P046, P047
- Weight: 2,100 lbs
- Accessible from rear door

Zone 2 (5-10ft) - Stop 4:
- 2 pallets + 8 boxes
- Weight: 1,850 lbs

Zone 3 (10-15ft) - Stop 3:
- 4 pallets
- Weight: 2,800 lbs

Zone 4 (15-20ft) - Stop 2:
- 3 pallets + 12 boxes
- Weight: 2,400 lbs

Zone 5 (20-26ft) - Stop 1 (First):
- 2 pallets
- Weight: 1,600 lbs

**Summary:**
- Total Weight: 10,750 lbs (90% capacity)
- Volume Utilization: 82%
- Axle Weights: Front 4,200 lbs, Rear 6,550 lbs ✓
- Load Secured: Yes (4 load bars, stretch wrap)

---

## Questions to Ask

1. What type of vehicle? (van, box truck, semi?)
2. What are the cargo dimensions and weight capacity?
3. Single destination or multiple stops?
4. If multi-stop, what's the delivery sequence?
5. Any axle weight concerns or restrictions?
6. Are items palletized or loose cargo?
7. Any special handling requirements?

---

## Related Skills

- **route-optimization**: For delivery route planning
- **3d-bin-packing**: For general 3D packing algorithms
- **pallet-loading**: For pallet-level optimization
- **container-loading-optimization**: For container loading
- **fleet-management**: For vehicle fleet optimization
- **last-mile-delivery**: For final delivery optimization

---
name: value-analysis
description: When the user wants to conduct value analysis, evaluate medical products, establish formularies, or optimize product selection for healthcare. Also use when the user mentions "value analysis committee," "product evaluation," "clinical outcomes analysis," "total cost of ownership," "standardization," "formulary management," "evidence-based sourcing," "product selection," "physician preference items," or "cost-quality optimization." For hospital supply chain operations, see hospital-logistics. For cost analysis in general, see spend-analysis.
---

# Value Analysis

You are an expert in healthcare value analysis and product evaluation. Your goal is to help healthcare organizations make evidence-based product selection decisions that optimize clinical outcomes, cost-effectiveness, and operational efficiency.

## Initial Assessment

Before conducting value analysis, understand:

1. **Organization Context**
   - Organization type? (hospital, health system, IDN)
   - Number of facilities and beds?
   - Service lines and specialties?
   - Current value analysis process maturity?

2. **Product Category Focus**
   - Product category? (medical devices, supplies, equipment, drugs)
   - Physician preference items (PPI) vs. commodities?
   - Current spend in category?
   - Number of SKUs and vendors?

3. **Value Analysis Structure**
   - Value analysis committee (VAC) in place?
   - Committee composition and meeting frequency?
   - Decision-making authority level?
   - Physician engagement level?

4. **Goals & Objectives**
   - Cost savings targets?
   - Quality improvement goals?
   - Standardization objectives?
   - Contract compliance issues?

---

## Value Analysis Framework

### Value Definition in Healthcare

**Value = (Clinical Outcomes + Patient Experience + Staff Experience) / Total Cost**

**Components:**
- **Clinical Outcomes**: Safety, efficacy, patient outcomes
- **Patient Experience**: Satisfaction, comfort, convenience
- **Staff Experience**: Ease of use, workflow, training requirements
- **Total Cost**: Acquisition + utilization + waste + complications

### Value Analysis Committee (VAC) Structure

**Membership:**
- **Executive Sponsor**: VP Supply Chain or CNO
- **Physician Champions**: Representatives from key specialties
- **Nursing Representatives**: Clinical nurse specialists
- **Materials Management**: Supply chain leaders
- **Finance**: Financial analyst
- **Infection Prevention**: When applicable
- **Risk Management**: When applicable
- **Quality/Patient Safety**: Quality director

**Meeting Cadence:**
- Monthly standing meetings
- Ad-hoc for urgent requests
- Quarterly strategic reviews

---

## Product Evaluation Process

### 7-Step Value Analysis Process

**Step 1: Request Submission**
- Standardized request form
- Clinical justification
- Cost impact estimation
- Urgency level

**Step 2: Initial Screening**
- Completeness check
- Preliminary impact assessment
- Committee assignment

**Step 3: Data Gathering**
- Clinical evidence review
- Cost analysis (TCO)
- Utilization data
- Peer feedback
- Vendor information

**Step 4: Product Trial/Evaluation**
- Clinical trial with defined metrics
- User feedback surveys
- Complication tracking
- Utilization monitoring

**Step 5: Financial Analysis**
- Total cost of ownership
- Budget impact
- ROI calculation
- Savings opportunity

**Step 6: Committee Review**
- Present findings
- Clinical discussion
- Financial review
- Vote on recommendation

**Step 7: Implementation & Monitoring**
- Formulary update
- Contract execution
- Training and rollout
- Outcome tracking

### Python Implementation Framework

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

class ProductCategory(Enum):
    IMPLANT = "implant"
    SURGICAL_SUPPLY = "surgical_supply"
    MEDICAL_DEVICE = "medical_device"
    CAPITAL_EQUIPMENT = "capital_equipment"
    PHARMACEUTICAL = "pharmaceutical"
    COMMODITY = "commodity"

class RequestStatus(Enum):
    SUBMITTED = "submitted"
    SCREENING = "screening"
    DATA_GATHERING = "data_gathering"
    CLINICAL_TRIAL = "clinical_trial"
    FINANCIAL_ANALYSIS = "financial_analysis"
    COMMITTEE_REVIEW = "committee_review"
    APPROVED = "approved"
    DENIED = "denied"
    ON_HOLD = "on_hold"

class UrgencyLevel(Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"

@dataclass
class ValueAnalysisRequest:
    """Value analysis request structure"""
    request_id: str
    submission_date: datetime
    submitted_by: str
    product_name: str
    manufacturer: str
    category: ProductCategory
    clinical_justification: str
    estimated_annual_volume: int
    estimated_unit_cost: float
    urgency: UrgencyLevel
    status: RequestStatus
    current_alternative: Optional[str] = None
    physician_champion: Optional[str] = None

class ValueAnalysisCommittee:
    """
    Manage value analysis process and product evaluations
    """

    def __init__(self, organization_name):
        self.organization_name = organization_name
        self.requests = {}
        self.evaluations = {}
        self.formulary = {}
        self.committee_members = []

    def submit_request(self, request_data):
        """
        Submit new value analysis request

        Parameters:
        - request_data: Dict with request information
        """

        request_id = f"VAR-{len(self.requests)+1:05d}"

        request = ValueAnalysisRequest(
            request_id=request_id,
            submission_date=datetime.now(),
            submitted_by=request_data['submitted_by'],
            product_name=request_data['product_name'],
            manufacturer=request_data['manufacturer'],
            category=ProductCategory[request_data['category'].upper()],
            clinical_justification=request_data['clinical_justification'],
            estimated_annual_volume=request_data['estimated_annual_volume'],
            estimated_unit_cost=request_data['estimated_unit_cost'],
            urgency=UrgencyLevel[request_data.get('urgency', 'ROUTINE').upper()],
            status=RequestStatus.SUBMITTED,
            current_alternative=request_data.get('current_alternative'),
            physician_champion=request_data.get('physician_champion')
        )

        self.requests[request_id] = request

        return request

    def initial_screening(self, request_id):
        """
        Conduct initial screening of request
        """

        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")

        request = self.requests[request_id]

        # Screening criteria
        screening_results = {
            'complete_submission': self._check_completeness(request),
            'clinical_justification_adequate': len(request.clinical_justification) > 50,
            'cost_impact_reasonable': request.estimated_unit_cost > 0,
            'duplicate_request': self._check_duplicates(request),
            'formulary_already_exists': self._check_formulary_exists(request.product_name)
        }

        # Determine if passes screening
        passes_screening = (
            screening_results['complete_submission'] and
            screening_results['clinical_justification_adequate'] and
            screening_results['cost_impact_reasonable'] and
            not screening_results['duplicate_request']
        )

        if passes_screening:
            request.status = RequestStatus.DATA_GATHERING
        else:
            request.status = RequestStatus.ON_HOLD

        return {
            'request_id': request_id,
            'passes_screening': passes_screening,
            'screening_results': screening_results,
            'next_steps': 'Proceed to data gathering' if passes_screening else 'Revise and resubmit'
        }

    def _check_completeness(self, request):
        """Check if request is complete"""
        required_fields = [
            request.product_name,
            request.manufacturer,
            request.clinical_justification,
            request.estimated_annual_volume,
            request.estimated_unit_cost
        ]
        return all(required_fields)

    def _check_duplicates(self, request):
        """Check for duplicate requests"""
        for req_id, req in self.requests.items():
            if (req.product_name == request.product_name and
                req.manufacturer == request.manufacturer and
                req.status not in [RequestStatus.DENIED, RequestStatus.APPROVED]):
                return True
        return False

    def _check_formulary_exists(self, product_name):
        """Check if product already on formulary"""
        return product_name in self.formulary

    def conduct_clinical_trial(self, request_id, trial_parameters):
        """
        Set up clinical trial/evaluation

        Parameters:
        - request_id: Value analysis request
        - trial_parameters: Dict with trial setup
        """

        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")

        request = self.requests[request_id]

        trial = {
            'request_id': request_id,
            'product_name': request.product_name,
            'start_date': trial_parameters.get('start_date', datetime.now()),
            'duration_days': trial_parameters.get('duration_days', 30),
            'trial_sites': trial_parameters.get('trial_sites', []),
            'sample_size': trial_parameters.get('sample_size', 20),
            'evaluation_criteria': trial_parameters.get('criteria', []),
            'participating_physicians': trial_parameters.get('physicians', []),
            'status': 'active'
        }

        self.evaluations[request_id] = trial
        request.status = RequestStatus.CLINICAL_TRIAL

        return trial

    def collect_trial_results(self, request_id, results_data):
        """
        Collect and analyze trial results

        Parameters:
        - request_id: Request ID
        - results_data: List of trial result records
        """

        if request_id not in self.evaluations:
            raise ValueError(f"No trial found for request {request_id}")

        trial = self.evaluations[request_id]

        # Aggregate results
        results_df = pd.DataFrame(results_data)

        # Calculate summary statistics
        summary = {
            'total_uses': len(results_df),
            'clinical_success_rate': (results_df['clinical_success'].sum() / len(results_df) * 100) if len(results_df) > 0 else 0,
            'user_satisfaction_avg': results_df['user_satisfaction'].mean() if 'user_satisfaction' in results_df.columns else None,
            'complications': results_df['complication'].sum() if 'complication' in results_df.columns else 0,
            'ease_of_use_avg': results_df['ease_of_use'].mean() if 'ease_of_use' in results_df.columns else None,
            'would_recommend_pct': (results_df['would_recommend'].sum() / len(results_df) * 100) if 'would_recommend' in results_df.columns and len(results_df) > 0 else 0
        }

        trial['results_summary'] = summary
        trial['results_data'] = results_df
        trial['status'] = 'completed'

        # Update request status
        self.requests[request_id].status = RequestStatus.FINANCIAL_ANALYSIS

        return summary

    def financial_analysis(self, request_id, financial_data):
        """
        Conduct total cost of ownership analysis

        Parameters:
        - request_id: Request ID
        - financial_data: Dict with cost components
        """

        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")

        request = self.requests[request_id]

        # Total Cost of Ownership (TCO) calculation
        tco = self._calculate_tco(request, financial_data)

        # Compare to current alternative
        if request.current_alternative:
            current_tco = financial_data.get('current_alternative_tco', {})
            comparison = self._compare_alternatives(tco, current_tco)
        else:
            comparison = None

        analysis = {
            'request_id': request_id,
            'product_name': request.product_name,
            'tco_analysis': tco,
            'comparison': comparison,
            'budget_impact': self._calculate_budget_impact(tco, request),
            'roi_analysis': self._calculate_roi(tco, comparison, financial_data) if comparison else None
        }

        self.evaluations[request_id]['financial_analysis'] = analysis

        return analysis

    def _calculate_tco(self, request, financial_data):
        """
        Calculate total cost of ownership

        TCO Components:
        - Acquisition cost
        - Training costs
        - Storage/handling costs
        - Waste/expiry costs
        - Complication costs
        - Disposal costs
        """

        annual_volume = request.estimated_annual_volume
        unit_cost = request.estimated_unit_cost

        tco = {
            'acquisition_cost': annual_volume * unit_cost,
            'training_cost': financial_data.get('training_cost', 0),
            'storage_cost': financial_data.get('storage_cost_per_unit', 0) * annual_volume,
            'waste_cost': financial_data.get('waste_rate', 0.02) * annual_volume * unit_cost,
            'complication_cost': financial_data.get('complication_rate', 0) * financial_data.get('complication_cost_per_event', 0) * annual_volume,
            'disposal_cost': financial_data.get('disposal_cost_per_unit', 0) * annual_volume
        }

        tco['total_annual_cost'] = sum(tco.values())
        tco['cost_per_use'] = tco['total_annual_cost'] / annual_volume if annual_volume > 0 else 0

        return tco

    def _compare_alternatives(self, new_tco, current_tco):
        """Compare new product TCO to current alternative"""

        new_total = new_tco['total_annual_cost']
        current_total = current_tco.get('total_annual_cost', new_total)

        comparison = {
            'new_product_tco': new_total,
            'current_product_tco': current_total,
            'annual_difference': new_total - current_total,
            'percent_change': ((new_total - current_total) / current_total * 100) if current_total > 0 else 0,
            'recommendation': 'Cost savings' if new_total < current_total else 'Cost increase'
        }

        return comparison

    def _calculate_budget_impact(self, tco, request):
        """Calculate budget impact"""

        # Assume budget is based on current spend
        current_budget = request.estimated_annual_volume * request.estimated_unit_cost

        budget_impact = {
            'current_budget': current_budget,
            'projected_spend': tco['total_annual_cost'],
            'budget_variance': tco['total_annual_cost'] - current_budget,
            'budget_variance_pct': ((tco['total_annual_cost'] - current_budget) / current_budget * 100) if current_budget > 0 else 0
        }

        return budget_impact

    def _calculate_roi(self, new_tco, comparison, financial_data):
        """Calculate return on investment"""

        if not comparison:
            return None

        annual_savings = -comparison['annual_difference']  # Negative if cost increase

        implementation_costs = (
            financial_data.get('implementation_cost', 0) +
            new_tco.get('training_cost', 0)
        )

        if annual_savings <= 0:
            roi = {
                'annual_savings': annual_savings,
                'implementation_cost': implementation_costs,
                'payback_period_years': None,
                'roi_3_year': None,
                'recommendation': 'Negative ROI - cost increase'
            }
        else:
            payback_period = implementation_costs / annual_savings if annual_savings > 0 else None
            roi_3_year = (annual_savings * 3) - implementation_costs

            roi = {
                'annual_savings': annual_savings,
                'implementation_cost': implementation_costs,
                'payback_period_years': round(payback_period, 2) if payback_period else None,
                'roi_3_year': round(roi_3_year, 2),
                'recommendation': 'Positive ROI' if payback_period and payback_period < 2 else 'ROI marginal'
            }

        return roi

    def committee_vote(self, request_id, vote_data):
        """
        Record committee vote on product

        Parameters:
        - request_id: Request ID
        - vote_data: Dict with voting results
        """

        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")

        request = self.requests[request_id]

        vote_results = {
            'request_id': request_id,
            'vote_date': datetime.now(),
            'votes_for': vote_data['votes_for'],
            'votes_against': vote_data['votes_against'],
            'abstentions': vote_data['abstentions'],
            'decision': 'approved' if vote_data['votes_for'] > vote_data['votes_against'] else 'denied',
            'conditions': vote_data.get('conditions', []),
            'implementation_plan': vote_data.get('implementation_plan')
        }

        self.evaluations[request_id]['committee_vote'] = vote_results

        # Update request status
        if vote_results['decision'] == 'approved':
            request.status = RequestStatus.APPROVED
            # Add to formulary
            self._add_to_formulary(request)
        else:
            request.status = RequestStatus.DENIED

        return vote_results

    def _add_to_formulary(self, request):
        """Add approved product to formulary"""

        self.formulary[request.product_name] = {
            'product_name': request.product_name,
            'manufacturer': request.manufacturer,
            'category': request.category,
            'approval_date': datetime.now(),
            'request_id': request.request_id,
            'status': 'active'
        }

    def generate_executive_summary(self, request_id):
        """
        Generate executive summary for committee review
        """

        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")

        request = self.requests[request_id]
        evaluation = self.evaluations.get(request_id, {})

        summary = {
            'request_id': request_id,
            'product_name': request.product_name,
            'manufacturer': request.manufacturer,
            'category': request.category.value,
            'submitted_by': request.submitted_by,
            'submission_date': request.submission_date,
            'clinical_justification': request.clinical_justification,
            'trial_results': evaluation.get('results_summary'),
            'financial_analysis': evaluation.get('financial_analysis'),
            'recommendation': self._generate_recommendation(evaluation),
            'status': request.status.value
        }

        return summary

    def _generate_recommendation(self, evaluation):
        """Generate recommendation based on trial and financial data"""

        if not evaluation:
            return "Insufficient data for recommendation"

        trial_results = evaluation.get('results_summary', {})
        financial = evaluation.get('financial_analysis', {})

        # Clinical criteria
        clinical_success = trial_results.get('clinical_success_rate', 0) >= 90
        user_satisfaction = trial_results.get('would_recommend_pct', 0) >= 70

        # Financial criteria
        if financial and financial.get('comparison'):
            cost_favorable = financial['comparison']['annual_difference'] <= 0
        else:
            cost_favorable = True  # If no comparison, assume neutral

        # Recommendation logic
        if clinical_success and user_satisfaction and cost_favorable:
            return "APPROVE - Meets clinical and financial criteria"
        elif clinical_success and user_satisfaction and not cost_favorable:
            return "CONDITIONAL APPROVAL - Clinical benefits may justify cost increase"
        elif not clinical_success or not user_satisfaction:
            return "DENY - Does not meet clinical criteria"
        else:
            return "DEFER - Additional evaluation needed"

# Example usage
vac = ValueAnalysisCommittee(organization_name="Memorial Hospital System")

# Submit request
request_data = {
    'submitted_by': 'Dr. Sarah Johnson, Orthopedic Surgery',
    'product_name': 'NextGen Hip Implant System',
    'manufacturer': 'Advanced Orthopedics Inc',
    'category': 'IMPLANT',
    'clinical_justification': 'New ceramic-on-ceramic bearing surface shows reduced wear in published studies. Lower dislocation rates reported. Improved patient outcomes expected.',
    'estimated_annual_volume': 150,
    'estimated_unit_cost': 4200,
    'urgency': 'ROUTINE',
    'current_alternative': 'Current Hip System A',
    'physician_champion': 'Dr. Sarah Johnson'
}

request = vac.submit_request(request_data)
print(f"Request submitted: {request.request_id}")

# Initial screening
screening = vac.initial_screening(request.request_id)
print(f"Screening result: {screening['passes_screening']}")

# Conduct trial
trial_params = {
    'duration_days': 90,
    'trial_sites': ['Main Campus OR'],
    'sample_size': 20,
    'criteria': ['Clinical success', 'User satisfaction', 'Complications', 'Ease of use'],
    'physicians': ['Dr. Johnson', 'Dr. Smith', 'Dr. Williams']
}

trial = vac.conduct_clinical_trial(request.request_id, trial_params)
print(f"Clinical trial initiated: {trial['sample_size']} cases")

# Collect trial results (simulated)
trial_results = [
    {
        'case_number': i+1,
        'clinical_success': True,
        'user_satisfaction': np.random.randint(7, 11),  # 7-10 scale
        'complication': False,
        'ease_of_use': np.random.randint(7, 11),
        'would_recommend': True
    }
    for i in range(20)
]

results_summary = vac.collect_trial_results(request.request_id, trial_results)
print(f"\nTrial Results:")
print(f"  Clinical success rate: {results_summary['clinical_success_rate']:.1f}%")
print(f"  Would recommend: {results_summary['would_recommend_pct']:.1f}%")

# Financial analysis
financial_data = {
    'training_cost': 5000,
    'storage_cost_per_unit': 10,
    'waste_rate': 0.01,
    'complication_rate': 0.02,
    'complication_cost_per_event': 15000,
    'disposal_cost_per_unit': 50,
    'current_alternative_tco': {
        'total_annual_cost': 150 * 4000  # Current product costs less upfront
    },
    'implementation_cost': 10000
}

financial_analysis = vac.financial_analysis(request.request_id, financial_data)
print(f"\nFinancial Analysis:")
print(f"  New product TCO: ${financial_analysis['tco_analysis']['total_annual_cost']:,.2f}")
print(f"  Annual difference: ${financial_analysis['comparison']['annual_difference']:,.2f}")
if financial_analysis['roi_analysis']:
    print(f"  Payback period: {financial_analysis['roi_analysis']['payback_period_years']} years")

# Committee vote
vote_data = {
    'votes_for': 8,
    'votes_against': 1,
    'abstentions': 1,
    'conditions': ['Monitor complications for first 50 cases', 'Quarterly utilization review'],
    'implementation_plan': 'Phased rollout - 3 surgeons initially, expand after 30 cases'
}

vote = vac.committee_vote(request.request_id, vote_data)
print(f"\nCommittee Vote: {vote['decision'].upper()}")
print(f"  For: {vote['votes_for']}, Against: {vote['votes_against']}, Abstain: {vote['abstentions']}")

# Executive summary
summary = vac.generate_executive_summary(request.request_id)
print(f"\nRecommendation: {summary['recommendation']}")
```

---

## Standardization Analysis

### Product Standardization Opportunities

```python
def analyze_standardization_opportunity(usage_data_df):
    """
    Identify product standardization opportunities

    Parameters:
    - usage_data_df: DataFrame with columns:
        - category: Product category
        - product_name: Product identifier
        - manufacturer: Manufacturer
        - annual_volume: Units used
        - unit_cost: Cost per unit
        - user: Department or physician using
    """

    results = []

    # Group by category
    for category in usage_data_df['category'].unique():
        category_data = usage_data_df[usage_data_df['category'] == category]

        # Calculate fragmentation
        num_products = category_data['product_name'].nunique()
        num_manufacturers = category_data['manufacturer'].nunique()
        num_users = category_data['user'].nunique()

        # Calculate spend
        total_spend = (category_data['annual_volume'] * category_data['unit_cost']).sum()

        # Identify top products
        product_spend = category_data.groupby('product_name').apply(
            lambda x: (x['annual_volume'] * x['unit_cost']).sum()
        ).sort_values(ascending=False)

        top_2_spend = product_spend.head(2).sum()
        top_2_coverage = (top_2_spend / total_spend * 100) if total_spend > 0 else 0

        # Standardization opportunity score
        # Higher score = more opportunity
        fragmentation_score = min(num_products / 5, 1.0) * 30  # Max 30 points
        manufacturer_diversity = min(num_manufacturers / 3, 1.0) * 20  # Max 20 points
        spend_concentration = (100 - top_2_coverage) / 100 * 30  # Max 30 points if very fragmented
        spend_magnitude = min(total_spend / 100000, 1.0) * 20  # Max 20 points if >$100K

        opportunity_score = fragmentation_score + manufacturer_diversity + spend_concentration + spend_magnitude

        # Recommendation
        if opportunity_score >= 60:
            recommendation = "HIGH PRIORITY - Significant standardization opportunity"
        elif opportunity_score >= 40:
            recommendation = "MEDIUM PRIORITY - Moderate opportunity"
        else:
            recommendation = "LOW PRIORITY - Already standardized or low impact"

        # Potential savings (assume 15% from standardization)
        potential_savings = total_spend * 0.15

        results.append({
            'category': category,
            'num_products': num_products,
            'num_manufacturers': num_manufacturers,
            'num_users': num_users,
            'total_annual_spend': round(total_spend, 2),
            'top_2_coverage_pct': round(top_2_coverage, 1),
            'opportunity_score': round(opportunity_score, 1),
            'potential_savings': round(potential_savings, 2),
            'recommendation': recommendation
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('opportunity_score', ascending=False)

    return results_df

# Example usage
usage_data = pd.DataFrame({
    'category': ['Surgical Gloves'] * 8 + ['Hip Implants'] * 6 + ['IV Catheters'] * 4,
    'product_name': [
        'Glove-A', 'Glove-B', 'Glove-C', 'Glove-D', 'Glove-E', 'Glove-F', 'Glove-G', 'Glove-H',
        'Hip-System-A', 'Hip-System-B', 'Hip-System-C', 'Hip-System-D', 'Hip-System-E', 'Hip-System-F',
        'IV-Cath-A', 'IV-Cath-B', 'IV-Cath-A', 'IV-Cath-B'
    ],
    'manufacturer': [
        'MfgA', 'MfgB', 'MfgC', 'MfgD', 'MfgA', 'MfgE', 'MfgF', 'MfgG',
        'Ortho-A', 'Ortho-B', 'Ortho-C', 'Ortho-A', 'Ortho-D', 'Ortho-E',
        'IV-Mfg-A', 'IV-Mfg-B', 'IV-Mfg-A', 'IV-Mfg-B'
    ],
    'annual_volume': [
        10000, 8000, 5000, 3000, 2000, 1500, 1000, 500,
        80, 60, 30, 20, 15, 10,
        15000, 12000, 10000, 8000
    ],
    'unit_cost': [
        0.35, 0.38, 0.34, 0.40, 0.36, 0.39, 0.37, 0.41,
        4000, 4200, 3800, 4100, 4500, 3900,
        8.50, 8.75, 8.50, 8.75
    ],
    'user': [
        'OR', 'OR', 'ER', 'ICU', 'Peds', 'NICU', 'Cath Lab', 'Clinic',
        'Dr. Smith', 'Dr. Johnson', 'Dr. Williams', 'Dr. Brown', 'Dr. Davis', 'Dr. Wilson',
        'All Units', 'All Units', 'All Units', 'All Units'
    ]
})

standardization_analysis = analyze_standardization_opportunity(usage_data)

print("Standardization Opportunity Analysis:")
print(standardization_analysis[['category', 'num_products', 'total_annual_spend',
                                'opportunity_score', 'potential_savings', 'recommendation']])

print(f"\nTotal potential savings: ${standardization_analysis['potential_savings'].sum():,.2f}")
```

---

## Evidence-Based Sourcing

### Clinical Evidence Evaluation

```python
class ClinicalEvidenceEvaluator:
    """
    Evaluate clinical evidence for product decisions
    """

    def __init__(self):
        self.evidence_levels = {
            'Level 1': 'Systematic review/meta-analysis of RCTs',
            'Level 2': 'Individual randomized controlled trial (RCT)',
            'Level 3': 'Controlled trial without randomization',
            'Level 4': 'Case-control or cohort study',
            'Level 5': 'Systematic review of descriptive/qualitative studies',
            'Level 6': 'Single descriptive or qualitative study',
            'Level 7': 'Expert opinion'
        }

    def evaluate_evidence(self, product_name, evidence_list):
        """
        Evaluate quality of clinical evidence

        Parameters:
        - product_name: Product being evaluated
        - evidence_list: List of studies/evidence with metadata
        """

        if not evidence_list:
            return {
                'product_name': product_name,
                'evidence_strength': 'Insufficient',
                'recommendation': 'Additional evidence required',
                'studies_count': 0
            }

        # Categorize by evidence level
        evidence_by_level = {}
        for evidence in evidence_list:
            level = evidence.get('evidence_level', 'Level 7')
            if level not in evidence_by_level:
                evidence_by_level[level] = []
            evidence_by_level[level].append(evidence)

        # Determine overall evidence strength
        if 'Level 1' in evidence_by_level or (
            'Level 2' in evidence_by_level and len(evidence_by_level['Level 2']) >= 2
        ):
            evidence_strength = 'Strong'
            recommendation = 'Supported by high-quality evidence'
        elif 'Level 2' in evidence_by_level or 'Level 3' in evidence_by_level:
            evidence_strength = 'Moderate'
            recommendation = 'Supported by moderate evidence'
        elif any(level in evidence_by_level for level in ['Level 4', 'Level 5', 'Level 6']):
            evidence_strength = 'Weak'
            recommendation = 'Limited evidence - proceed with caution'
        else:
            evidence_strength = 'Insufficient'
            recommendation = 'Insufficient evidence - require clinical trial'

        evaluation = {
            'product_name': product_name,
            'evidence_strength': evidence_strength,
            'studies_count': len(evidence_list),
            'evidence_by_level': {
                level: len(studies) for level, studies in evidence_by_level.items()
            },
            'recommendation': recommendation,
            'highest_evidence_level': min(evidence_by_level.keys(), key=lambda x: int(x.split()[1])) if evidence_by_level else None
        }

        return evaluation

    def comparative_effectiveness_analysis(self, product_a, product_b,
                                          outcome_data_a, outcome_data_b):
        """
        Compare clinical effectiveness of two products

        Parameters:
        - product_a, product_b: Product names
        - outcome_data_a, outcome_data_b: Clinical outcome data
        """

        # Calculate outcome metrics
        metrics_a = self._calculate_outcomes(outcome_data_a)
        metrics_b = self._calculate_outcomes(outcome_data_b)

        # Compare
        comparison = {}
        for metric in metrics_a.keys():
            if metric in metrics_b:
                comparison[metric] = {
                    'product_a': metrics_a[metric],
                    'product_b': metrics_b[metric],
                    'difference': metrics_a[metric] - metrics_b[metric],
                    'better_product': product_a if metrics_a[metric] > metrics_b[metric] else product_b
                }

        # Determine clinical superiority
        a_better_count = sum(1 for c in comparison.values() if c['better_product'] == product_a)
        b_better_count = sum(1 for c in comparison.values() if c['better_product'] == product_b)

        if a_better_count > b_better_count:
            conclusion = f"{product_a} demonstrates superior clinical outcomes"
        elif b_better_count > a_better_count:
            conclusion = f"{product_b} demonstrates superior clinical outcomes"
        else:
            conclusion = "Products demonstrate equivalent clinical outcomes"

        return {
            'product_a': product_a,
            'product_b': product_b,
            'comparison': comparison,
            'conclusion': conclusion
        }

    def _calculate_outcomes(self, outcome_data):
        """Calculate outcome metrics from raw data"""

        if not outcome_data:
            return {}

        metrics = {
            'success_rate': (outcome_data.get('successes', 0) / outcome_data.get('total_cases', 1)) * 100,
            'complication_rate': (outcome_data.get('complications', 0) / outcome_data.get('total_cases', 1)) * 100,
            'readmission_rate': (outcome_data.get('readmissions', 0) / outcome_data.get('total_cases', 1)) * 100,
            'patient_satisfaction': outcome_data.get('satisfaction_score', 0)
        }

        return metrics

# Example usage
evaluator = ClinicalEvidenceEvaluator()

# Evidence evaluation
evidence = [
    {'title': 'RCT of NextGen vs Standard Hip', 'evidence_level': 'Level 2', 'n': 200, 'conclusion': 'Favorable'},
    {'title': 'Meta-analysis ceramic bearings', 'evidence_level': 'Level 1', 'n': 1500, 'conclusion': 'Reduced wear'},
    {'title': 'Retrospective cohort study', 'evidence_level': 'Level 4', 'n': 500, 'conclusion': 'Lower dislocation'}
]

evidence_eval = evaluator.evaluate_evidence('NextGen Hip Implant', evidence)

print("Clinical Evidence Evaluation:")
print(f"  Evidence Strength: {evidence_eval['evidence_strength']}")
print(f"  Studies: {evidence_eval['studies_count']}")
print(f"  Recommendation: {evidence_eval['recommendation']}")

# Comparative effectiveness
outcomes_new = {
    'total_cases': 150,
    'successes': 145,
    'complications': 3,
    'readmissions': 2,
    'satisfaction_score': 9.2
}

outcomes_current = {
    'total_cases': 150,
    'successes': 140,
    'complications': 8,
    'readmissions': 5,
    'satisfaction_score': 8.5
}

comparison = evaluator.comparative_effectiveness_analysis(
    'NextGen Hip', 'Current Hip System',
    outcomes_new, outcomes_current
)

print(f"\n{comparison['conclusion']}")
```

---

## Physician Engagement Strategies

### Engaging Physicians in Value Analysis

**Key Principles:**
1. **Clinical leadership**: Physician champions lead initiatives
2. **Data-driven**: Show evidence, not just cost
3. **Transparency**: Open about process and criteria
4. **Respect expertise**: Value clinical judgment
5. **Win-win mindset**: Clinical quality + cost efficiency

```python
def physician_preference_analysis(ppi_usage_df):
    """
    Analyze physician preference item (PPI) variation

    Parameters:
    - ppi_usage_df: DataFrame with:
        - physician: Physician name
        - procedure_type: Type of procedure
        - product_used: Product selected
        - unit_cost: Cost of product
        - outcome: Clinical outcome (success/complication)
    """

    analysis = []

    # Group by procedure type
    for procedure in ppi_usage_df['procedure_type'].unique():
        proc_data = ppi_usage_df[ppi_usage_df['procedure_type'] == procedure]

        # Product variation by physician
        product_variation = proc_data.groupby('physician')['product_used'].nunique()
        avg_products_per_physician = product_variation.mean()

        # Cost variation
        cost_by_physician = proc_data.groupby('physician')['unit_cost'].mean()
        cost_std_dev = cost_by_physician.std()
        cost_range = cost_by_physician.max() - cost_by_physician.min()

        # Outcome analysis
        outcome_by_product = proc_data.groupby('product_used').apply(
            lambda x: (x['outcome'] == 'success').sum() / len(x) * 100
        )

        # Cost vs. outcome correlation
        # Group by product, get avg cost and outcome rate
        product_analysis = proc_data.groupby('product_used').agg({
            'unit_cost': 'mean',
            'outcome': lambda x: (x == 'success').sum() / len(x) * 100
        }).reset_index()

        product_analysis.columns = ['product', 'avg_cost', 'success_rate']

        # Identify opportunities
        if cost_range > 1000 and cost_std_dev > 500:
            opportunity = "HIGH - Significant cost variation without clinical justification"
            priority = 1
        elif cost_range > 500:
            opportunity = "MEDIUM - Moderate cost variation"
            priority = 2
        else:
            opportunity = "LOW - Minimal variation"
            priority = 3

        analysis.append({
            'procedure_type': procedure,
            'num_physicians': proc_data['physician'].nunique(),
            'num_products': proc_data['product_used'].nunique(),
            'avg_products_per_physician': round(avg_products_per_physician, 1),
            'cost_range': round(cost_range, 2),
            'cost_std_dev': round(cost_std_dev, 2),
            'opportunity': opportunity,
            'priority': priority,
            'engagement_strategy': 'Physician-led review with outcome data' if priority <= 2 else 'Monitor'
        })

    analysis_df = pd.DataFrame(analysis)
    analysis_df = analysis_df.sort_values('priority')

    return analysis_df

# Example usage
ppi_usage = pd.DataFrame({
    'physician': ['Dr. Smith'] * 30 + ['Dr. Johnson'] * 30 + ['Dr. Williams'] * 30,
    'procedure_type': ['Total Hip Arthroplasty'] * 90,
    'product_used': (
        ['Hip-System-A'] * 30 +
        ['Hip-System-B'] * 20 + ['Hip-System-C'] * 10 +
        ['Hip-System-A'] * 15 + ['Hip-System-D'] * 15
    ),
    'unit_cost': (
        [4000] * 30 +
        [4200] * 20 + [5500] * 10 +
        [4000] * 15 + [6000] * 15
    ),
    'outcome': np.random.choice(['success', 'complication'], 90, p=[0.95, 0.05])
})

ppi_analysis = physician_preference_analysis(ppi_usage)

print("Physician Preference Item Analysis:")
print(ppi_analysis[['procedure_type', 'num_physicians', 'num_products',
                    'cost_range', 'opportunity', 'engagement_strategy']])
```

---

## Value Analysis Metrics & KPIs

### Measuring Value Analysis Impact

```python
def calculate_value_analysis_kpis(va_data, implementation_data):
    """
    Calculate value analysis program KPIs

    Parameters:
    - va_data: Value analysis requests and decisions
    - implementation_data: Implementation tracking data
    """

    kpis = {}

    # Request throughput
    if 'submission_date' in va_data.columns and 'decision_date' in va_data.columns:
        va_data['cycle_time_days'] = (va_data['decision_date'] - va_data['submission_date']).dt.days
        kpis['avg_cycle_time_days'] = va_data['cycle_time_days'].mean()

    # Approval rate
    if 'decision' in va_data.columns:
        kpis['approval_rate'] = (va_data['decision'] == 'approved').sum() / len(va_data) * 100

    # Cost savings realized
    if 'projected_savings' in implementation_data.columns and 'actual_savings' in implementation_data.columns:
        kpis['projected_savings'] = implementation_data['projected_savings'].sum()
        kpis['actual_savings'] = implementation_data['actual_savings'].sum()
        kpis['savings_realization_rate'] = (kpis['actual_savings'] / kpis['projected_savings'] * 100) if kpis['projected_savings'] > 0 else 0

    # Standardization progress
    if 'sku_count_before' in implementation_data.columns and 'sku_count_after' in implementation_data.columns:
        total_sku_reduction = (implementation_data['sku_count_before'] - implementation_data['sku_count_after']).sum()
        kpis['sku_reduction'] = total_sku_reduction

    # Contract compliance
    if 'contract_compliant' in implementation_data.columns:
        kpis['contract_compliance_rate'] = (implementation_data['contract_compliant'].sum() / len(implementation_data) * 100)

    # Physician engagement
    if 'physician_champion_assigned' in va_data.columns:
        kpis['physician_engagement_rate'] = (va_data['physician_champion_assigned'].sum() / len(va_data) * 100)

    # Format KPIs
    for key in kpis:
        if 'rate' in key or key == 'savings_realization_rate':
            kpis[key] = round(kpis[key], 2)
        elif 'savings' in key:
            kpis[key] = round(kpis[key], 2)
        elif 'reduction' in key:
            kpis[key] = int(kpis[key])
        else:
            kpis[key] = round(kpis[key], 1)

    return kpis

# Example data
va_requests = pd.DataFrame({
    'request_id': range(1, 51),
    'submission_date': pd.date_range('2023-01-01', periods=50, freq='7D'),
    'decision_date': pd.date_range('2023-01-01', periods=50, freq='7D') + pd.Timedelta(days=45),
    'decision': np.random.choice(['approved', 'denied'], 50, p=[0.70, 0.30]),
    'physician_champion_assigned': np.random.choice([True, False], 50, p=[0.80, 0.20])
})

implementation = pd.DataFrame({
    'initiative_id': range(1, 36),  # 35 approved items implemented
    'projected_savings': np.random.randint(10000, 100000, 35),
    'actual_savings': np.random.randint(8000, 95000, 35),
    'sku_count_before': np.random.randint(3, 10, 35),
    'sku_count_after': np.random.randint(1, 3, 35),
    'contract_compliant': np.random.choice([True, False], 35, p=[0.90, 0.10])
})

# Align actual savings to be roughly 80-90% of projected
implementation['actual_savings'] = (implementation['projected_savings'] * np.random.uniform(0.75, 0.95, 35)).astype(int)

kpis = calculate_value_analysis_kpis(va_requests, implementation)

print("Value Analysis Program KPIs:")
for metric, value in kpis.items():
    if 'savings' in metric:
        print(f"  {metric}: ${value:,.2f}")
    elif 'rate' in metric:
        print(f"  {metric}: {value}%")
    else:
        print(f"  {metric}: {value}")
```

---

## Tools & Libraries

### Value Analysis Software

**Value Analysis Platforms:**
- **GHX Lumere**: Clinical product evaluation
- **ECRI Guidelines**: Evidence-based clinical guidelines
- **Innovaccer**: Healthcare analytics platform
- **Definitive Healthcare**: Market intelligence
- **Repertoire**: Value analysis and product evaluation

**Decision Support:**
- **ECRI**: Medical device safety and effectiveness
- **Hayes**: Medical technology assessment
- **AHRQ**: Agency for Healthcare Research and Quality resources
- **Cochrane**: Systematic reviews

**Data Analytics:**
- **Tableau/Power BI**: Visualization and dashboards
- **Qlik**: Analytics platform
- **SAP Analytics Cloud**: Enterprise analytics

### Python Libraries

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scipy`: Statistical analysis
- `statsmodels`: Statistical modeling

**Optimization:**
- `pulp`: Linear programming
- `scipy.optimize`: Optimization algorithms

**Visualization:**
- `matplotlib`, `seaborn`: Charts
- `plotly`: Interactive dashboards

---

## Common Challenges & Solutions

### Challenge: Physician Resistance to Standardization

**Problem:**
- "My patients are different"
- Preference for familiar products
- Fear of compromising quality
- Autonomy concerns

**Solutions:**
- Physician-led value analysis teams
- Data on outcomes, not just cost
- Grandfather clauses where clinically justified
- Trial periods with option to revert
- Peer comparison (blinded)
- Focus on high-variation/low-outcome-difference products first

### Challenge: Slow Value Analysis Process

**Problem:**
- Requests languish for months
- Committee meetings infrequent
- Data gathering delays
- Missing information from submitters

**Solutions:**
- Dedicated VA staff/coordinator
- Streamlined request forms
- 30/60/90-day timelines by urgency
- Pre-committee screening
- Standard data packages from vendors
- Monthly standing meetings
- Fast-track process for urgent needs

### Challenge: Savings Not Realized

**Problem:**
- Approved changes not implemented
- Off-contract purchasing continues
- Insufficient adoption/compliance
- Overestimated savings projections

**Solutions:**
- Dedicated implementation plans
- ERP/MMM system updates (hard blocks if needed)
- Physician champions drive adoption
- Compliance monitoring and reporting
- Conservative savings estimates (75% rule)
- Quarterly savings validation
- Link to supply chain/physician scorecards

### Challenge: Limited Clinical Evidence

**Problem:**
- New/emerging technologies
- Limited published studies
- Vendor-sponsored research
- Conflicting evidence

**Solutions:**
- Require independent studies when available
- Clinical trials before formulary addition
- Expert panel review
- Evidence grading framework
- Peer institution consultation
- Conditional approval with monitoring
- Registry participation for outcomes tracking

### Challenge: Total Cost of Ownership Blind Spots

**Problem:**
- Focus only on acquisition cost
- Hidden costs (training, waste, complications)
- Downstream impacts not considered
- Different cost allocations by department

**Solutions:**
- TCO model required for all evaluations
- Include all stakeholders (OR, SPD, Infection Prevention)
- Track actual utilization and waste
- Complication cost analysis
- Multi-year view
- Activity-based costing when possible

### Challenge: Balancing Innovation with Cost Control

**Problem:**
- Risk-averse decision-making
- "Prove it saves money or no"
- Stifling innovation
- Competitive disadvantage for recruitment

**Solutions:**
- Innovation fund/budget allocation
- Separate track for true innovation vs. me-too products
- Early adopter programs with monitoring
- Value framework beyond just cost
- Strategic partnerships with manufacturers
- Centers of excellence can get broader latitude
- Balanced scorecard approach

---

## Output Format

### Value Analysis Decision Report

**Product Evaluation Summary**

**Product:** NextGen Hip Implant System
**Manufacturer:** Advanced Orthopedics Inc
**Request ID:** VAR-00245
**Submitted by:** Dr. Sarah Johnson, Orthopedic Surgery
**Decision Date:** February 15, 2024

---

**Clinical Evidence:**
- Evidence Strength: **Strong**
- Number of Studies: 12 (3 Level 1, 5 Level 2, 4 Level 3-4)
- Key Findings:
  - Reduced wear rates vs. current system (p<0.01)
  - Lower dislocation rates: 0.5% vs. 1.8% current system
  - Improved patient-reported outcomes (HOOS scores)
  - No significant difference in infection rates

**Clinical Trial Results:**
- Sample Size: 20 cases across 3 surgeons
- Clinical Success Rate: 100%
- User Satisfaction: 9.1/10 average
- Complications: 0
- Would Recommend: 95%

**Financial Analysis:**

| Cost Component | New Product | Current | Difference |
|----------------|-------------|---------|------------|
| Acquisition Cost | $630,000 | $600,000 | +$30,000 |
| Training Cost | $5,000 | $0 | +$5,000 |
| Complication Cost | $4,500 | $40,500 | -$36,000 |
| **Total Annual Cost** | **$639,500** | **$640,500** | **-$1,000** |
| **Cost per Use** | **$4,263** | **$4,270** | **-$7** |

**ROI Analysis:**
- Implementation Cost: $10,000
- Annual Savings: $1,000
- 3-Year Savings: -$7,000 (cost neutral accounting for implementation)
- Payback Period: Cost neutral
- **Primary Value: Clinical outcomes improvement, cost neutral**

**Committee Recommendation:**

**APPROVED** (8-1-1 vote)

**Rationale:**
- Strong clinical evidence supporting improved outcomes
- Superior performance in clinical trial
- High physician satisfaction
- Cost neutral when accounting for reduced complications
- Aligns with organization's quality and patient safety goals

**Conditions:**
1. Monitor complications for first 50 cases
2. Quarterly utilization and outcome review
3. Phased implementation - 3 surgeons initially

**Implementation Plan:**
- **Phase 1 (Month 1-2):** Training for initial 3 surgeons
- **Phase 2 (Month 3-4):** Expand to all orthopedic surgeons
- **Phase 3 (Month 5-6):** Full conversion, retire current system
- **Ongoing:** Quarterly outcome tracking for first year

---

**Projected Impact:**
- Improved patient outcomes (reduced dislocations)
- Cost neutral financially
- Enhanced surgeon satisfaction
- Competitive advantage for orthopedic service line

---

## Questions to Ask

If you need more context:

1. What's the organization structure? (single hospital, health system, IDN)
2. Is there a value analysis committee in place?
3. What product categories are priorities?
4. What's the physician engagement level?
5. What's the current spend under evaluation?
6. Are there specific cost savings targets?
7. What's the decision-making authority and approval process?
8. What data systems are available? (ERP, clinical data, outcomes)
9. What are the main challenges with current process?
10. What's the timeline for evaluation and implementation?

---

## Related Skills

- **hospital-logistics**: Hospital supply chain operations
- **spend-analysis**: Spend analysis and cost management
- **strategic-sourcing**: Strategic sourcing and contracting
- **supplier-selection**: Supplier evaluation and selection
- **contract-management**: Contract management and compliance
- **inventory-optimization**: Inventory optimization
- **quality-management**: Quality management and patient safety
- **compliance-management**: Regulatory compliance
- **data-analytics**: Healthcare analytics (if exists)

---
name: trim-loss-minimization
description: When the user wants to minimize material waste, reduce trim loss, or optimize material utilization in cutting operations. Also use when the user mentions "waste minimization," "scrap reduction," "material efficiency," "trim optimization," "yield maximization," "off-cut management," or "residual material utilization." For specific cutting problems, see 1d-cutting-stock, 2d-cutting-stock, or nesting-optimization.
---

# Trim Loss Minimization

You are an expert in trim loss minimization and material waste reduction for cutting operations. Your goal is to help manufacturers minimize material waste, reduce costs, and improve sustainability by optimizing cutting patterns, managing residual materials, and implementing best practices for material utilization.

## Initial Assessment

Before addressing trim loss problems, understand:

1. **Material and Process Characteristics**
   - What materials? (steel, wood, glass, fabric, plastic, paper)
   - Cutting process? (saw, laser, waterjet, shear, die cutting)
   - Material dimensions and formats?
   - Material cost per unit ($/kg, $/m², $/piece)?
   - Are there different material grades or qualities?

2. **Current Waste Situation**
   - Current trim loss percentage?
   - Where is waste generated? (ends, edges, between parts, defects)
   - What happens to scrap? (recycled, sold, discarded)
   - Scrap recovery value?
   - Cost of waste disposal?

3. **Production Requirements**
   - Production volume (units per day/week/month)?
   - Item mix (how many different parts/sizes)?
   - Demand variability (stable or fluctuating)?
   - Quality tolerances?
   - Customer-specific requirements?

4. **Existing Constraints**
   - Minimum usable piece size?
   - Standard stock sizes available?
   - Can you change stock sizes or suppliers?
   - Equipment limitations?
   - Setup time/cost considerations?

5. **Business Objectives**
   - Primary goal: minimize waste %, minimize cost, or maximize throughput?
   - Acceptable trade-offs (cost vs. waste vs. complexity)?
   - Sustainability/environmental goals?
   - Target waste reduction?

---

## Trim Loss Framework

### Understanding Trim Loss

**Trim Loss Definition:**
Trim loss is the percentage of raw material that becomes waste after cutting operations.

**Formula:**
```
Trim Loss % = (Total Material - Usable Material) / Total Material × 100
```

Or:
```
Trim Loss % = (1 - Utilization %) × 100
```

**Components of Trim Loss:**

1. **Edge Trim**
   - Material trimmed from sheet edges
   - Often due to material irregularities
   - Standard practice in many industries

2. **Inter-Part Waste**
   - Material between cut parts
   - Saw kerf (material removed by cutting tool)
   - Minimum spacing requirements

3. **End Trim**
   - Material at ends of stocks/sheets
   - Too small for useful parts
   - Accumulates with each stock used

4. **Pattern Inefficiency**
   - Poor nesting or pattern design
   - Suboptimal item arrangement
   - Irregular part shapes

5. **Quality Defects**
   - Material defects requiring cutting around
   - Quality failures requiring rework
   - Damaged material

6. **Residuals**
   - Leftover pieces too small for current orders
   - May be usable for future orders
   - Storage and tracking overhead

---

## Trim Loss Measurement and Analysis

### Comprehensive Measurement System

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TrimLossAnalyzer:
    """
    Comprehensive Trim Loss Analysis Tool

    Tracks, measures, and analyzes trim loss across operations
    """

    def __init__(self):
        self.cutting_records = []
        self.material_specs = {}

    def add_material_spec(self, material_id, cost_per_unit, unit='m2', scrap_value=0):
        """
        Add material specification

        Parameters:
        - material_id: material identifier
        - cost_per_unit: cost per unit area/length/piece
        - unit: measurement unit
        - scrap_value: recovery value of scrap
        """
        self.material_specs[material_id] = {
            'cost_per_unit': cost_per_unit,
            'unit': unit,
            'scrap_value': scrap_value
        }

    def record_cutting_operation(self, material_id, total_material,
                                 usable_material, waste_material,
                                 waste_breakdown=None, date=None):
        """
        Record a cutting operation

        Parameters:
        - material_id: material type
        - total_material: total material used
        - usable_material: material in final parts
        - waste_material: total waste generated
        - waste_breakdown: dict with waste categories
        - date: operation date
        """

        trim_loss_pct = (waste_material / total_material * 100) if total_material > 0 else 0
        utilization_pct = (usable_material / total_material * 100) if total_material > 0 else 0

        record = {
            'material_id': material_id,
            'date': date or pd.Timestamp.now(),
            'total_material': total_material,
            'usable_material': usable_material,
            'waste_material': waste_material,
            'trim_loss_pct': trim_loss_pct,
            'utilization_pct': utilization_pct,
            'waste_breakdown': waste_breakdown or {}
        }

        self.cutting_records.append(record)

    def calculate_financial_impact(self, material_id=None, time_period=None):
        """
        Calculate financial impact of trim loss

        Returns cost analysis and potential savings
        """

        # Filter records
        records = self.cutting_records

        if material_id:
            records = [r for r in records if r['material_id'] == material_id]

        if time_period:
            # time_period should be tuple (start_date, end_date)
            start, end = time_period
            records = [r for r in records if start <= r['date'] <= end]

        if not records:
            return None

        # Aggregate data
        total_material_used = sum(r['total_material'] for r in records)
        total_waste = sum(r['waste_material'] for r in records)
        total_usable = sum(r['usable_material'] for r in records)

        avg_trim_loss = (total_waste / total_material_used * 100) if total_material_used > 0 else 0

        # Calculate costs
        material_costs = {}
        waste_costs = {}
        scrap_value = {}

        for material_id in set(r['material_id'] for r in records):
            if material_id not in self.material_specs:
                continue

            spec = self.material_specs[material_id]

            material_records = [r for r in records if r['material_id'] == material_id]
            mat_total = sum(r['total_material'] for r in material_records)
            mat_waste = sum(r['waste_material'] for r in material_records)

            material_costs[material_id] = mat_total * spec['cost_per_unit']
            waste_costs[material_id] = mat_waste * spec['cost_per_unit']
            scrap_value[material_id] = mat_waste * spec['scrap_value']

        total_material_cost = sum(material_costs.values())
        total_waste_cost = sum(waste_costs.values())
        total_scrap_value = sum(scrap_value.values())
        net_waste_cost = total_waste_cost - total_scrap_value

        # Calculate potential savings scenarios
        scenarios = {}

        for reduction_pct in [5, 10, 15, 20, 25]:
            reduced_waste = total_waste * (1 - reduction_pct/100)
            reduced_waste_cost = (reduced_waste / total_material_used) * total_material_cost
            savings = total_waste_cost - reduced_waste_cost

            scenarios[f'{reduction_pct}% reduction'] = {
                'new_waste': reduced_waste,
                'new_trim_loss_pct': (reduced_waste / total_material_used * 100),
                'annual_savings': savings,
                'payback_potential': savings
            }

        return {
            'total_material_used': total_material_used,
            'total_waste': total_waste,
            'total_usable': total_usable,
            'avg_trim_loss_pct': avg_trim_loss,
            'total_material_cost': total_material_cost,
            'total_waste_cost': total_waste_cost,
            'total_scrap_value': total_scrap_value,
            'net_waste_cost': net_waste_cost,
            'waste_cost_percentage': (net_waste_cost / total_material_cost * 100) if total_material_cost > 0 else 0,
            'improvement_scenarios': scenarios
        }

    def analyze_waste_breakdown(self, material_id=None):
        """
        Analyze waste by category

        Returns breakdown of waste sources
        """

        records = self.cutting_records
        if material_id:
            records = [r for r in records if r['material_id'] == material_id]

        # Aggregate waste by category
        waste_categories = {}

        for record in records:
            if 'waste_breakdown' in record and record['waste_breakdown']:
                for category, amount in record['waste_breakdown'].items():
                    waste_categories[category] = waste_categories.get(category, 0) + amount

        total_waste = sum(waste_categories.values())

        if total_waste > 0:
            waste_breakdown = {
                cat: {
                    'amount': amt,
                    'percentage': (amt / total_waste * 100)
                }
                for cat, amt in waste_categories.items()
            }
        else:
            waste_breakdown = {}

        return {
            'total_waste': total_waste,
            'waste_by_category': waste_breakdown
        }

    def generate_pareto_analysis(self, material_id=None):
        """
        Generate Pareto analysis of waste sources

        Returns top waste contributors (80/20 analysis)
        """

        breakdown = self.analyze_waste_breakdown(material_id)

        if not breakdown['waste_by_category']:
            return None

        # Sort by amount
        sorted_categories = sorted(
            breakdown['waste_by_category'].items(),
            key=lambda x: x[1]['amount'],
            reverse=True
        )

        # Calculate cumulative percentages
        cumulative_pct = 0
        pareto_data = []

        for category, data in sorted_categories:
            cumulative_pct += data['percentage']
            pareto_data.append({
                'category': category,
                'amount': data['amount'],
                'percentage': data['percentage'],
                'cumulative_pct': cumulative_pct
            })

        return {
            'pareto_data': pareto_data,
            'top_80_pct_contributors': [
                item for item in pareto_data
                if item['cumulative_pct'] <= 80
            ]
        }

    def plot_trim_loss_trends(self, material_id=None, save_path=None):
        """
        Plot trim loss trends over time
        """

        records = self.cutting_records
        if material_id:
            records = [r for r in records if r['material_id'] == material_id]

        if not records:
            print("No data to plot")
            return

        df = pd.DataFrame(records)
        df = df.sort_values('date')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Trim loss over time
        ax1.plot(df['date'], df['trim_loss_pct'], marker='o', linewidth=2)
        ax1.axhline(y=df['trim_loss_pct'].mean(), color='r',
                   linestyle='--', label=f'Average: {df["trim_loss_pct"].mean():.2f}%')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Trim Loss %', fontsize=12)
        ax1.set_title('Trim Loss Trend Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative waste
        df['cumulative_waste'] = df['waste_material'].cumsum()
        df['cumulative_material'] = df['total_material'].cumsum()

        ax2.fill_between(df['date'], 0, df['cumulative_waste'],
                        alpha=0.3, color='red', label='Cumulative Waste')
        ax2.plot(df['date'], df['cumulative_material'],
                color='blue', linewidth=2, label='Cumulative Material Used')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Material (units)', fontsize=12)
        ax2.set_title('Cumulative Material Usage and Waste', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_pareto_chart(self, material_id=None, save_path=None):
        """
        Plot Pareto chart of waste categories
        """

        pareto = self.generate_pareto_analysis(material_id)

        if not pareto:
            print("No data for Pareto analysis")
            return

        data = pareto['pareto_data']
        categories = [d['category'] for d in data]
        amounts = [d['amount'] for d in data]
        cumulative = [d['cumulative_pct'] for d in data]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar chart
        x_pos = np.arange(len(categories))
        ax1.bar(x_pos, amounts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Waste Category', fontsize=12)
        ax1.set_ylabel('Waste Amount', fontsize=12, color='steelblue')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # Line chart for cumulative
        ax2 = ax1.twinx()
        ax2.plot(x_pos, cumulative, color='red', marker='o',
                linewidth=2, markersize=8)
        ax2.axhline(y=80, color='red', linestyle='--',
                   alpha=0.5, label='80% Line')
        ax2.set_ylabel('Cumulative %', fontsize=12, color='red')
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend()

        plt.title('Pareto Analysis of Waste Sources',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


# Example usage
def example_trim_loss_analysis():
    """Example: Analyze trim loss data"""

    analyzer = TrimLossAnalyzer()

    # Add material specifications
    analyzer.add_material_spec(
        'Steel_Sheet',
        cost_per_unit=15.50,  # $/m²
        unit='m2',
        scrap_value=2.00  # $/m² scrap value
    )

    # Simulate cutting records
    import random
    from datetime import datetime, timedelta

    base_date = datetime(2024, 1, 1)

    for i in range(50):
        date = base_date + timedelta(days=i)
        total = 100 + random.uniform(-10, 10)
        trim_loss = 15 + random.uniform(-5, 5)  # Average 15% trim loss
        waste = total * (trim_loss / 100)
        usable = total - waste

        analyzer.record_cutting_operation(
            material_id='Steel_Sheet',
            total_material=total,
            usable_material=usable,
            waste_material=waste,
            waste_breakdown={
                'edge_trim': waste * 0.3,
                'inter_part': waste * 0.4,
                'end_trim': waste * 0.2,
                'defects': waste * 0.1
            },
            date=date
        )

    # Financial analysis
    print("FINANCIAL IMPACT ANALYSIS")
    print("=" * 70)

    impact = analyzer.calculate_financial_impact('Steel_Sheet')

    print(f"Total Material Used: {impact['total_material_used']:.2f} m²")
    print(f"Total Waste: {impact['total_waste']:.2f} m² ({impact['avg_trim_loss_pct']:.2f}%)")
    print(f"Total Material Cost: ${impact['total_material_cost']:.2f}")
    print(f"Total Waste Cost: ${impact['total_waste_cost']:.2f}")
    print(f"Scrap Recovery Value: ${impact['total_scrap_value']:.2f}")
    print(f"Net Waste Cost: ${impact['net_waste_cost']:.2f}")
    print(f"Waste as % of Material Cost: {impact['waste_cost_percentage']:.2f}%")
    print()

    print("IMPROVEMENT SCENARIOS:")
    print("-" * 70)
    for scenario, data in impact['improvement_scenarios'].items():
        print(f"{scenario}:")
        print(f"  New Trim Loss: {data['new_trim_loss_pct']:.2f}%")
        print(f"  Annual Savings: ${data['annual_savings']:.2f}")
        print()

    # Pareto analysis
    print("\nPARETO ANALYSIS")
    print("=" * 70)

    pareto = analyzer.generate_pareto_analysis('Steel_Sheet')

    print("Top 80% Contributors:")
    for item in pareto['top_80_pct_contributors']:
        print(f"  {item['category']}: {item['amount']:.2f} ({item['percentage']:.1f}%)")

    # Plots
    analyzer.plot_trim_loss_trends('Steel_Sheet')
    analyzer.plot_pareto_chart('Steel_Sheet')

    return analyzer
```

---

## Trim Loss Minimization Strategies

### Strategy 1: Cutting Pattern Optimization

```python
def optimize_cutting_patterns(items, stock_length, current_trim_loss_pct,
                              target_trim_loss_pct):
    """
    Optimize cutting patterns to reduce trim loss

    Compares current performance to optimized solution
    """

    from skills.one_d_cutting_stock import ColumnGenerationCuttingStock

    # Current situation (using simple heuristic)
    current_stocks = estimate_stocks_needed(items, stock_length,
                                            trim_loss_pct=current_trim_loss_pct)

    # Optimized solution (using column generation)
    solver = ColumnGenerationCuttingStock(stock_length, items)
    optimal_solution = solver.solve()

    # Compare
    comparison = {
        'current': {
            'stocks': current_stocks,
            'trim_loss_pct': current_trim_loss_pct,
            'waste': current_stocks * stock_length * (current_trim_loss_pct / 100)
        },
        'optimized': {
            'stocks': optimal_solution['num_stocks'],
            'trim_loss_pct': 100 - optimal_solution['utilization'],
            'waste': optimal_solution['total_waste']
        }
    }

    # Improvement
    stocks_saved = current_stocks - optimal_solution['num_stocks']
    trim_loss_reduction = current_trim_loss_pct - (100 - optimal_solution['utilization'])

    comparison['improvement'] = {
        'stocks_saved': stocks_saved,
        'stocks_saved_pct': (stocks_saved / current_stocks * 100) if current_stocks > 0 else 0,
        'trim_loss_reduction': trim_loss_reduction,
        'achieved_target': (100 - optimal_solution['utilization']) <= target_trim_loss_pct
    }

    return comparison

def estimate_stocks_needed(items, stock_length, trim_loss_pct):
    """Estimate stocks needed given current trim loss"""
    total_length_needed = sum(length * qty for length, qty, _ in items)
    effective_length = stock_length * (1 - trim_loss_pct / 100)
    return int(np.ceil(total_length_needed / effective_length))
```

### Strategy 2: Residual Material Management

```python
class ResidualMaterialManager:
    """
    Manage and utilize residual/leftover materials

    Tracks inventory of residuals and matches them to new orders
    """

    def __init__(self):
        self.residuals = []  # List of available residual pieces

    def add_residual(self, length, width, material_id, location=None):
        """Add residual piece to inventory"""
        self.residuals.append({
            'length': length,
            'width': width,
            'material_id': material_id,
            'area': length * width,
            'location': location,
            'date_added': pd.Timestamp.now()
        })

    def find_matching_residuals(self, required_length, required_width,
                                material_id, tolerance=0):
        """
        Find residuals that can satisfy requirement

        Parameters:
        - required_length, required_width: minimum dimensions needed
        - material_id: material type
        - tolerance: acceptable size tolerance

        Returns: list of matching residuals
        """

        matches = []

        for idx, residual in enumerate(self.residuals):
            if residual['material_id'] != material_id:
                continue

            # Check if residual is large enough
            if (residual['length'] >= required_length - tolerance and
                residual['width'] >= required_width - tolerance):
                matches.append({
                    'index': idx,
                    'residual': residual,
                    'excess_length': residual['length'] - required_length,
                    'excess_width': residual['width'] - required_width,
                    'excess_area': (residual['length'] - required_length) * \
                                   (residual['width'] - required_width)
                })

        # Sort by least excess (best fit)
        matches.sort(key=lambda x: x['excess_area'])

        return matches

    def allocate_residual(self, residual_index, amount_used):
        """
        Allocate residual to an order

        If fully used, remove from inventory
        If partially used, update dimensions
        """

        if residual_index >= len(self.residuals):
            return False

        residual = self.residuals[residual_index]

        # For simplicity, assume full usage here
        # In practice, you'd update dimensions based on how it was cut

        del self.residuals[residual_index]

        return True

    def calculate_residual_value(self, material_specs):
        """
        Calculate total value of residual inventory

        Parameters:
        - material_specs: dict with material costs

        Returns: total value
        """

        total_value = 0

        for residual in self.residuals:
            material_id = residual['material_id']
            if material_id in material_specs:
                cost_per_unit = material_specs[material_id]['cost_per_unit']
                total_value += residual['area'] * cost_per_unit

        return total_value

    def identify_slow_moving_residuals(self, age_threshold_days=90):
        """
        Identify residuals that have been in inventory too long

        These may need special action (discount, scrap, etc.)
        """

        now = pd.Timestamp.now()
        slow_moving = []

        for residual in self.residuals:
            age_days = (now - residual['date_added']).days

            if age_days > age_threshold_days:
                slow_moving.append({
                    'residual': residual,
                    'age_days': age_days
                })

        return slow_moving


# Example usage
def example_residual_management():
    """Example: Managing residual materials"""

    manager = ResidualMaterialManager()

    # Add some residuals
    manager.add_residual(1200, 800, 'Steel_Sheet', 'Rack_A1')
    manager.add_residual(900, 600, 'Steel_Sheet', 'Rack_A2')
    manager.add_residual(1500, 1000, 'Steel_Sheet', 'Rack_A3')

    # Need to cut a part 1000x700
    matches = manager.find_matching_residuals(1000, 700, 'Steel_Sheet')

    print("Matching residuals for 1000x700 part:")
    for match in matches:
        res = match['residual']
        print(f"  {res['length']}x{res['width']} at {res['location']} "
              f"(excess: {match['excess_area']} mm²)")

    if matches:
        # Use best match
        print(f"\nUsing residual from {matches[0]['residual']['location']}")
        manager.allocate_residual(matches[0]['index'], (1000, 700))

    return manager
```

### Strategy 3: Multi-Objective Optimization

```python
def multi_objective_trim_loss_optimization(items, stock_specs, weights):
    """
    Multi-objective optimization balancing:
    - Material cost minimization
    - Trim loss minimization
    - Cutting complexity minimization

    Parameters:
    - items: list of items to cut
    - stock_specs: available stock specifications
    - weights: dict with objective weights

    Returns: Pareto optimal solutions
    """

    from pulp import *

    # Define objectives
    objectives = {
        'material_cost': 0,
        'trim_loss': 0,
        'complexity': 0
    }

    # Weighted sum approach
    # In practice, use NSGA-II or other multi-objective algorithms

    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}

    # Solve for different weight combinations
    solutions = []

    # This is simplified - full implementation would explore
    # multiple weight combinations and return Pareto front

    return solutions
```

---

## Best Practices for Trim Loss Minimization

### 1. Material Selection

- **Standardize Stock Sizes:** Use fewer standard sizes
- **Match Stock to Demand:** Choose stock sizes that align with typical orders
- **Negotiate Custom Sizes:** Work with suppliers for optimal stock dimensions

### 2. Order Consolidation

- **Batch Similar Orders:** Combine orders for better nesting
- **Optimize Order Quantities:** Consider material efficiency when quoting
- **Plan Ahead:** Look ahead at upcoming orders for better planning

### 3. Process Improvements

- **Precision Cutting:** Reduce kerf width with better equipment
- **Quality Control:** Minimize defects that cause scrap
- **Operator Training:** Ensure operators understand waste impact
- **Maintenance:** Keep equipment calibrated and maintained

### 4. Technology Investment

- **Optimization Software:** Implement cutting optimization software
- **Automated Nesting:** Use automatic nesting systems
- **Real-time Tracking:** Monitor trim loss in real-time
- **Data Analytics:** Analyze patterns to identify improvements

### 5. Organizational Changes

- **Incentive Programs:** Reward waste reduction
- **Continuous Improvement:** Regular review and improvement cycles
- **Cross-functional Teams:** Involve purchasing, production, sales
- **Supplier Partnerships:** Work with suppliers on waste reduction

---

## Industry Benchmarks

### Typical Trim Loss by Industry

| Industry | Material | Typical Trim Loss | Best-in-Class |
|----------|----------|-------------------|---------------|
| Steel Fabrication | Sheet Metal | 10-20% | 5-8% |
| Wood Products | Lumber | 15-25% | 8-12% |
| Glass Cutting | Flat Glass | 12-18% | 6-10% |
| Textile/Apparel | Fabric | 10-15% | 5-8% |
| Paper Converting | Paper Rolls | 3-8% | 1-3% |
| Plastic Extrusion | Plastic Sheet | 8-15% | 4-7% |

---

## ROI Calculation for Trim Loss Reduction

```python
def calculate_trim_loss_reduction_roi(current_annual_material_cost,
                                     current_trim_loss_pct,
                                     target_trim_loss_pct,
                                     implementation_cost,
                                     scrap_recovery_rate=0):
    """
    Calculate ROI for trim loss reduction initiative

    Returns payback period and annual savings
    """

    # Current waste cost
    current_waste_cost = current_annual_material_cost * (current_trim_loss_pct / 100)

    # Target waste cost
    target_waste_cost = current_annual_material_cost * (target_trim_loss_pct / 100)

    # Annual savings
    gross_savings = current_waste_cost - target_waste_cost
    scrap_value_loss = gross_savings * scrap_recovery_rate  # Lost scrap sales
    net_annual_savings = gross_savings - scrap_value_loss

    # ROI metrics
    payback_period = implementation_cost / net_annual_savings if net_annual_savings > 0 else float('inf')
    roi_year1 = ((net_annual_savings - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0
    roi_year3 = ((net_annual_savings * 3 - implementation_cost) / implementation_cost * 100) if implementation_cost > 0 else 0

    return {
        'current_waste_cost': current_waste_cost,
        'target_waste_cost': target_waste_cost,
        'annual_savings': net_annual_savings,
        'implementation_cost': implementation_cost,
        'payback_period_years': payback_period,
        'roi_year_1_pct': roi_year1,
        'roi_year_3_pct': roi_year3,
        'npv_3_year': net_annual_savings * 3 - implementation_cost  # Simplified NPV
    }


# Example
def example_roi_calculation():
    """Example: Calculate ROI for trim loss reduction project"""

    roi = calculate_trim_loss_reduction_roi(
        current_annual_material_cost=1_000_000,  # $1M per year
        current_trim_loss_pct=15,  # Current: 15% waste
        target_trim_loss_pct=8,   # Target: 8% waste
        implementation_cost=50_000,  # $50K for optimization software + training
        scrap_recovery_rate=0.15  # Recover 15% of waste cost through scrap sales
    )

    print("TRIM LOSS REDUCTION ROI ANALYSIS")
    print("=" * 70)
    print(f"Current Annual Waste Cost: ${roi['current_waste_cost']:,.2f}")
    print(f"Target Annual Waste Cost: ${roi['target_waste_cost']:,.2f}")
    print(f"Annual Savings: ${roi['annual_savings']:,.2f}")
    print(f"Implementation Cost: ${roi['implementation_cost']:,.2f}")
    print(f"Payback Period: {roi['payback_period_years']:.1f} years")
    print(f"ROI Year 1: {roi['roi_year_1_pct']:.1f}%")
    print(f"ROI Year 3: {roi['roi_year_3_pct']:.1f}%")
    print(f"3-Year NPV: ${roi['npv_3_year']:,.2f}")

    return roi
```

---

## Output Format

### Trim Loss Analysis Report

**Executive Summary:**
- Current Trim Loss: 15.2%
- Industry Benchmark: 8-12%
- Gap: 3.2-7.2 percentage points
- Annual Material Cost: $1,250,000
- Annual Waste Cost: $190,000
- Improvement Opportunity: $40,000-$90,000/year

**Waste Breakdown (Pareto Analysis):**
1. Inter-part waste: 45% ($85,500)
2. Edge trim: 28% ($53,200)
3. End trim: 18% ($34,200)
4. Quality defects: 9% ($17,100)

**Top 3 Improvement Opportunities:**
1. Implement cutting optimization software → 5% reduction → $62,500/year
2. Residual material management system → 2% reduction → $25,000/year
3. Operator training program → 1% reduction → $12,500/year

**Recommended Action Plan:**
- Phase 1 (0-3 months): Implement software, train operators
- Phase 2 (3-6 months): Launch residual management
- Phase 3 (6-12 months): Continuous improvement program
- Total Investment: $75,000
- Expected Annual Savings: $100,000
- Payback: 9 months

---

## Questions to Ask

1. What is your current trim loss percentage?
2. What materials do you cut and what are their costs?
3. How do you currently track waste?
4. What happens to scrap material?
5. What cutting processes do you use?
6. How many different part types do you produce?
7. What is your annual material spend?
8. What waste reduction targets do you have?
9. Do you use cutting optimization software?
10. How do you manage leftover materials?

---

## Related Skills

- **1d-cutting-stock**: For linear cutting optimization
- **2d-cutting-stock**: For sheet cutting optimization
- **nesting-optimization**: For irregular shape nesting
- **guillotine-cutting**: For guillotine cutting problems
- **lean-manufacturing**: For waste reduction methodology
- **process-optimization**: For overall process improvement
- **supply-chain-analytics**: For data analysis and tracking

