---
name: column-generation
description: "When the user wants to solve large-scale optimization problems using column generation, decomposition methods, or Dantzig-Wolfe decomposition. Also use when the user mentions \"column generation,\" \"master problem,\" \"pricing problem,\" \"cutting stock,\" \"crew scheduling,\" \"vehicle routing with column gen,\" \"branch-and-price,\" or when the problem has exponentially many variables. For general optimization, see optimization-modeling. For metaheuristics, see metaheuristic-optimization."
---

# Column Generation

You are an expert in column generation and decomposition methods for large-scale supply chain optimization. Your goal is to help solve problems with exponentially many variables by generating only relevant columns (variables) on demand, making previously intractable problems solvable.

## Initial Assessment

Before applying column generation, understand:

1. **Problem Structure**
   - Does problem have exponentially many variables?
   - Can it be decomposed into master and subproblems?
   - Is there a natural decomposition structure?
   - Are constraints decomposable?

2. **Problem Characteristics**
   - Problem type? (cutting stock, routing, crew scheduling, packing)
   - Size? (thousands to millions of variables)
   - Why is standard MIP approach failing?
   - Required solution quality?

3. **Computational Environment**
   - Solver access? (need LP/MIP solver)
   - Subproblem complexity? (easy or hard to solve)
   - Time constraints?
   - Parallel computing available?

4. **Technical Expertise**
   - Team familiarity with column generation?
   - Ability to formulate subproblem?
   - Need for exact vs. heuristic solutions?

---

## Column Generation Framework

### Core Concept

**Problem Decomposition:**
- **Master Problem**: Restricted problem with subset of variables
- **Pricing Problem**: Find new variables (columns) to improve solution
- **Iteration**: Solve master → solve pricing → add columns → repeat
- **Termination**: When no improving columns found

### Mathematical Foundation

**Original Problem (Set Partitioning):**
```
min  Σ c_j x_j
s.t. Σ a_ij x_j = 1  ∀i  (cover each element exactly once)
     x_j ∈ {0,1}
```

**Relaxed Master Problem:**
```
min  Σ c_j x_j      (j ∈ J_current)
s.t. Σ a_ij x_j = 1  ∀i
     x_j ≥ 0

Dual variables: π_i
```

**Pricing Problem:**
```
Find column j with reduced cost: c_j - Σ a_ij π_i < 0

This becomes an optimization problem specific to the application
```

---

## Cutting Stock Problem with Column Generation

### Problem Description

**1D Cutting Stock:**
- Cut large rolls of width W into smaller pieces
- Meet demand for different piece widths
- Minimize number of rolls used (minimize waste)

### Implementation

```python
import numpy as np
from pulp import *
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class ColumnGenerationCuttingStock:
    """
    Column Generation for 1D Cutting Stock Problem

    Problem: Cut standard-width rolls into smaller pieces to meet demand
    Objective: Minimize number of rolls used
    """

    def __init__(self,
                 roll_width: float,
                 piece_widths: List[float],
                 piece_demands: List[int],
                 max_iterations: int = 100):
        """
        Initialize Column Generation for Cutting Stock

        roll_width: width of standard roll
        piece_widths: list of required piece widths
        piece_demands: demand for each piece width
        """

        self.roll_width = roll_width
        self.piece_widths = piece_widths
        self.piece_demands = piece_demands
        self.n_pieces = len(piece_widths)

        self.max_iterations = max_iterations

        # Pattern storage: each pattern is dict {piece_idx: quantity}
        self.patterns = []

        # Results
        self.optimal_patterns = None
        self.num_rolls = None
        self.iteration_history = []

    def _generate_initial_patterns(self) -> List[Dict[int, int]]:
        """
        Generate initial patterns (one piece type per pattern)

        Each pattern: maximum pieces of one width that fit in roll
        """

        patterns = []

        for i, width in enumerate(self.piece_widths):
            max_pieces = int(self.roll_width / width)
            pattern = {i: max_pieces}
            patterns.append(pattern)

        return patterns

    def _solve_master_problem(self, patterns: List[Dict[int, int]]) -> Tuple[float, List[float]]:
        """
        Solve Restricted Master Problem (RMP)

        Minimize number of rolls used
        Subject to: meet demand for each piece type

        Returns: objective value, dual values (shadow prices)
        """

        n_patterns = len(patterns)

        # Create LP model
        master = LpProblem("Master_Cutting_Stock", LpMinimize)

        # Decision variables: number of times to use each pattern
        pattern_vars = [LpVariable(f"Pattern_{j}", lowBound=0, cat='Continuous')
                       for j in range(n_patterns)]

        # Objective: minimize total number of rolls
        master += lpSum(pattern_vars), "Total_Rolls"

        # Constraints: meet demand for each piece type
        constraints = []
        for i in range(self.n_pieces):
            constraint = lpSum([
                patterns[j].get(i, 0) * pattern_vars[j]
                for j in range(n_patterns)
            ]) >= self.piece_demands[i]

            master += constraint, f"Demand_Piece_{i}"
            constraints.append(constraint)

        # Solve
        master.solve(PULP_CBC_CMD(msg=0))

        # Extract results
        obj_value = value(master.objective)

        # Extract dual values (shadow prices)
        dual_values = []
        for constraint in constraints:
            # Get dual value from constraint
            dual = constraint.pi if hasattr(constraint, 'pi') else 0
            dual_values.append(dual)

        # Fallback if dual extraction fails (use simplified approach)
        if all(d == 0 for d in dual_values):
            # Estimate duals from solution
            for i in range(self.n_pieces):
                total_produced = sum(patterns[j].get(i, 0) * pattern_vars[j].varValue
                                   for j in range(n_patterns))
                if total_produced > 0:
                    dual_values[i] = 1.0 / self.piece_widths[i]
                else:
                    dual_values[i] = 1.0

        return obj_value, dual_values

    def _solve_pricing_problem(self, dual_values: List[float]) -> Tuple[Dict[int, int], float]:
        """
        Solve Pricing Subproblem (knapsack problem)

        Find cutting pattern with negative reduced cost

        Reduced cost = 1 - Σ (pieces_i * dual_i)

        This is a knapsack problem:
        max  Σ dual_i * pieces_i
        s.t. Σ width_i * pieces_i ≤ roll_width
             pieces_i ≥ 0, integer

        Returns: new pattern, reduced cost
        """

        # Create knapsack model
        pricing = LpProblem("Pricing_Knapsack", LpMaximize)

        # Decision variables: number of pieces of each type in pattern
        pieces = [LpVariable(f"Piece_{i}", lowBound=0, cat='Integer')
                 for i in range(self.n_pieces)]

        # Objective: maximize value (dual values)
        pricing += lpSum([dual_values[i] * pieces[i]
                         for i in range(self.n_pieces)]), "Pattern_Value"

        # Constraint: don't exceed roll width
        pricing += lpSum([self.piece_widths[i] * pieces[i]
                         for i in range(self.n_pieces)]) <= self.roll_width, \
                   "Roll_Width"

        # Solve
        pricing.solve(PULP_CBC_CMD(msg=0))

        # Extract pattern
        new_pattern = {}
        for i in range(self.n_pieces):
            quantity = int(pieces[i].varValue)
            if quantity > 0:
                new_pattern[i] = quantity

        # Calculate reduced cost
        pattern_value = value(pricing.objective)
        reduced_cost = 1.0 - pattern_value

        return new_pattern, reduced_cost

    def optimize(self) -> Dict:
        """
        Run Column Generation algorithm

        Returns: optimization results
        """

        print(f"Starting Column Generation for Cutting Stock...")
        print(f"Roll Width: {self.roll_width}")
        print(f"Piece Types: {self.n_pieces}")
        print(f"Total Demand: {sum(self.piece_demands)}")

        # Initialize with simple patterns
        self.patterns = self._generate_initial_patterns()

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Solve master problem
            obj_value, dual_values = self._solve_master_problem(self.patterns)

            self.iteration_history.append({
                'iteration': iteration,
                'objective': obj_value,
                'num_patterns': len(self.patterns)
            })

            print(f"\nIteration {iteration}:")
            print(f"  Current Objective: {obj_value:.2f} rolls")
            print(f"  Number of Patterns: {len(self.patterns)}")
            print(f"  Dual Values: {[f'{d:.3f}' for d in dual_values]}")

            # Solve pricing problem
            new_pattern, reduced_cost = self._solve_pricing_problem(dual_values)

            print(f"  Reduced Cost: {reduced_cost:.6f}")

            # Check termination
            if reduced_cost >= -1e-6:  # No improving pattern found
                print(f"\nOptimality reached! No more improving patterns.")
                break

            # Add new pattern
            print(f"  Adding new pattern: {new_pattern}")
            self.patterns.append(new_pattern)

        # Final solve with integer variables
        print(f"\nSolving final MIP with {len(self.patterns)} patterns...")
        self.num_rolls, self.optimal_patterns = self._solve_final_mip(self.patterns)

        return {
            'num_rolls': self.num_rolls,
            'optimal_patterns': self.optimal_patterns,
            'all_patterns': self.patterns,
            'iterations': iteration,
            'iteration_history': self.iteration_history
        }

    def _solve_final_mip(self, patterns: List[Dict[int, int]]) -> Tuple[float, Dict]:
        """
        Solve final MIP with integer pattern variables

        Returns: optimal number of rolls, pattern usage
        """

        n_patterns = len(patterns)

        # Create MIP model
        final = LpProblem("Final_Cutting_Stock_MIP", LpMinimize)

        # Decision variables: integer number of times to use each pattern
        pattern_vars = [LpVariable(f"Pattern_{j}", lowBound=0, cat='Integer')
                       for j in range(n_patterns)]

        # Objective: minimize total number of rolls
        final += lpSum(pattern_vars), "Total_Rolls"

        # Constraints: meet demand for each piece type
        for i in range(self.n_pieces):
            final += lpSum([
                patterns[j].get(i, 0) * pattern_vars[j]
                for j in range(n_patterns)
            ]) >= self.piece_demands[i], f"Demand_Piece_{i}"

        # Solve
        final.solve(PULP_CBC_CMD(msg=0))

        # Extract solution
        num_rolls = value(final.objective)

        pattern_usage = {}
        for j in range(n_patterns):
            usage = pattern_vars[j].varValue
            if usage > 0.5:  # Used
                pattern_usage[j] = int(usage)

        return num_rolls, pattern_usage

    def print_solution(self):
        """Print detailed solution"""

        print("\n" + "="*70)
        print("OPTIMAL CUTTING STOCK SOLUTION")
        print("="*70)

        print(f"\nTotal Rolls Used: {self.num_rolls}")
        print(f"\nPatterns Used:")

        total_pieces_produced = {i: 0 for i in range(self.n_pieces)}

        for pattern_idx, usage in sorted(self.optimal_patterns.items()):
            pattern = self.patterns[pattern_idx]

            print(f"\n  Pattern {pattern_idx} (use {usage} times):")

            for piece_idx, quantity in sorted(pattern.items()):
                width = self.piece_widths[piece_idx]
                print(f"    - {quantity} pieces of width {width}")
                total_pieces_produced[piece_idx] += quantity * usage

            # Calculate waste
            used_width = sum(self.piece_widths[i] * quantity
                           for i, quantity in pattern.items())
            waste = self.roll_width - used_width
            waste_pct = (waste / self.roll_width) * 100
            print(f"    Waste: {waste:.2f} ({waste_pct:.1f}%)")

        # Verify demand met
        print(f"\nDemand Satisfaction:")
        for i in range(self.n_pieces):
            demand = self.piece_demands[i]
            produced = total_pieces_produced[i]
            status = "✓" if produced >= demand else "✗"
            print(f"  Width {self.piece_widths[i]}: "
                  f"Demand = {demand}, Produced = {produced} {status}")

        # Calculate total waste
        total_waste = 0
        for pattern_idx, usage in self.optimal_patterns.items():
            pattern = self.patterns[pattern_idx]
            used_width = sum(self.piece_widths[i] * quantity
                           for i, quantity in pattern.items())
            total_waste += (self.roll_width - used_width) * usage

        total_material = self.num_rolls * self.roll_width
        waste_pct = (total_waste / total_material) * 100

        print(f"\nTotal Waste: {total_waste:.2f} ({waste_pct:.1f}%)")
        print("="*70)

    def plot_convergence(self):
        """Plot convergence of objective value"""

        iterations = [h['iteration'] for h in self.iteration_history]
        objectives = [h['objective'] for h in self.iteration_history]

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, objectives, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Number of Rolls (LP Relaxation)', fontsize=12)
        plt.title('Column Generation Convergence', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_patterns(self):
        """Visualize cutting patterns"""

        if not self.optimal_patterns:
            print("No solution to plot!")
            return

        fig, axes = plt.subplots(len(self.optimal_patterns), 1,
                                figsize=(14, 2*len(self.optimal_patterns)))

        if len(self.optimal_patterns) == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, self.n_pieces))

        for ax_idx, (pattern_idx, usage) in enumerate(sorted(self.optimal_patterns.items())):
            pattern = self.patterns[pattern_idx]

            ax = axes[ax_idx]

            # Draw roll
            ax.add_patch(plt.Rectangle((0, 0), self.roll_width, 1,
                                      fill=False, edgecolor='black', linewidth=2))

            # Draw pieces
            current_pos = 0
            for piece_idx in sorted(pattern.keys()):
                quantity = pattern[piece_idx]
                width = self.piece_widths[piece_idx]

                for _ in range(quantity):
                    ax.add_patch(plt.Rectangle((current_pos, 0), width, 1,
                                              facecolor=colors[piece_idx],
                                              edgecolor='black', linewidth=1))
                    # Add label
                    ax.text(current_pos + width/2, 0.5, f'{width}',
                           ha='center', va='center', fontsize=10, fontweight='bold')

                    current_pos += width

            # Waste
            waste = self.roll_width - current_pos
            if waste > 0:
                ax.add_patch(plt.Rectangle((current_pos, 0), waste, 1,
                                          facecolor='lightgray',
                                          edgecolor='black', linewidth=1,
                                          hatch='//'))
                ax.text(current_pos + waste/2, 0.5, 'Waste',
                       ha='center', va='center', fontsize=9, style='italic')

            ax.set_xlim(0, self.roll_width)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f'Pattern {pattern_idx} (use {usage} times)',
                        fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example problem
    roll_width = 100  # Standard roll width

    # Required piece widths and demands
    piece_widths = [45, 36, 31, 14]
    piece_demands = [97, 610, 395, 211]

    print("Cutting Stock Problem:")
    print(f"  Standard Roll Width: {roll_width}")
    for i, (width, demand) in enumerate(zip(piece_widths, piece_demands)):
        print(f"  Piece {i}: Width = {width}, Demand = {demand}")

    # Create and solve
    cg_solver = ColumnGenerationCuttingStock(
        roll_width=roll_width,
        piece_widths=piece_widths,
        piece_demands=piece_demands,
        max_iterations=50
    )

    result = cg_solver.optimize()

    # Print solution
    cg_solver.print_solution()

    # Visualize
    cg_solver.plot_convergence()
    cg_solver.plot_patterns()
```

---

## Vehicle Routing with Column Generation

### Route-Based Formulation

**Instead of**: arc variables x_{ij} (exponentially many)
**Use**: route variables r_k (generate on demand)

### Implementation Outline

```python
class ColumnGenerationVRP:
    """
    Column Generation for Vehicle Routing Problem

    Master Problem: Select routes to cover all customers
    Pricing Problem: Find new profitable route (ESPP - shortest path with resources)
    """

    def __init__(self, customers, demands, capacity, distances):
        self.customers = customers
        self.demands = demands
        self.capacity = capacity
        self.distances = distances

        # Route storage
        self.routes = []  # List of routes (each route is list of customer IDs)

    def _generate_initial_routes(self):
        """Generate initial routes (one customer per route)"""

        routes = []
        for customer in self.customers:
            if self.demands[customer] <= self.capacity:
                routes.append([customer])

        return routes

    def _solve_master_problem(self, routes):
        """
        Set Partitioning Master Problem

        min  Σ cost(route_k) * x_k
        s.t. Σ a_ik * x_k = 1  ∀i  (each customer covered exactly once)
             x_k ∈ {0,1}  (route used or not)

        LP Relaxation for column generation
        """

        master = LpProblem("VRP_Master", LpMinimize)

        # Decision variables
        route_vars = [LpVariable(f"Route_{k}", lowBound=0, cat='Continuous')
                     for k in range(len(routes))]

        # Objective: minimize total route cost
        master += lpSum([self._route_cost(routes[k]) * route_vars[k]
                        for k in range(len(routes))]), "Total_Cost"

        # Constraints: each customer covered exactly once
        for customer in self.customers:
            master += lpSum([
                (1 if customer in routes[k] else 0) * route_vars[k]
                for k in range(len(routes))
            ]) == 1, f"Cover_Customer_{customer}"

        master.solve(PULP_CBC_CMD(msg=0))

        # Extract dual values
        dual_values = self._extract_duals(master)

        return value(master.objective), dual_values

    def _solve_pricing_problem(self, dual_values):
        """
        Elementary Shortest Path Problem with Resource Constraints (ESPPRC)

        Find route with negative reduced cost:
        cost(route) - Σ dual_i (for customers in route) < 0

        This is NP-hard, often solved with:
        - Dynamic programming (labeling algorithm)
        - Heuristics for large instances
        """

        # Simplified: use heuristic (nearest neighbor with dual values)
        new_route, reduced_cost = self._heuristic_pricing(dual_values)

        return new_route, reduced_cost

    def _heuristic_pricing(self, dual_values):
        """
        Heuristic pricing using nearest neighbor with dual values

        Build route greedily to maximize: Σ dual_i - distance_cost
        """

        best_route = None
        best_reduced_cost = 0

        # Try starting from each customer
        for start_customer in self.customers:
            route = [start_customer]
            remaining_capacity = self.capacity - self.demands[start_customer]
            unvisited = set(self.customers) - {start_customer}
            current = start_customer

            # Greedy construction
            while unvisited:
                # Find best next customer
                best_next = None
                best_value = -float('inf')

                for next_customer in unvisited:
                    if self.demands[next_customer] <= remaining_capacity:
                        # Value = dual - distance cost
                        value = (dual_values.get(next_customer, 0) -
                               self.distances[current][next_customer])

                        if value > best_value:
                            best_value = value
                            best_next = next_customer

                if best_next is None:
                    break

                route.append(best_next)
                current = best_next
                remaining_capacity -= self.demands[best_next]
                unvisited.remove(best_next)

            # Calculate reduced cost for this route
            route_cost = self._route_cost(route)
            dual_sum = sum(dual_values.get(c, 0) for c in route)
            reduced_cost = route_cost - dual_sum

            if reduced_cost < best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_route = route

        return best_route, best_reduced_cost

    def optimize(self):
        """Run column generation for VRP"""

        # Initialize routes
        self.routes = self._generate_initial_routes()

        iteration = 0
        max_iterations = 100

        while iteration < max_iterations:
            iteration += 1

            # Solve master
            obj_value, dual_values = self._solve_master_problem(self.routes)

            print(f"Iteration {iteration}: Objective = {obj_value:.2f}")

            # Solve pricing
            new_route, reduced_cost = self._solve_pricing_problem(dual_values)

            if reduced_cost >= -1e-6:  # No improving route
                print("Optimal solution found!")
                break

            # Add new route
            self.routes.append(new_route)
            print(f"  Added route: {new_route}")

        return self.routes
```

---

## Branch-and-Price

### Combining Column Generation with Branch-and-Bound

**Branch-and-Price:**
- Column generation at each node of branch-and-bound tree
- Needed when integer solution required
- Branching on original variables difficult (many not in RMP)

**Branching Strategies:**
1. **Branch on original variables**: Force variable to 0 or 1
2. **Branch on aggregate information**: e.g., "customer i served by vehicle k"
3. **Ryan-Foster branching**: For set partitioning

### Implementation Sketch

```python
class BranchAndPrice:
    """
    Branch-and-Price framework

    Combines column generation (pricing) with branch-and-bound (branching)
    """

    def __init__(self, problem):
        self.problem = problem
        self.incumbent = None
        self.incumbent_value = float('inf')

    def solve(self):
        """Branch-and-price algorithm"""

        # Initialize with root node
        root_node = BPNode(
            problem=self.problem,
            bounds=[],  # No branching constraints yet
            depth=0
        )

        node_queue = [root_node]

        while node_queue:
            # Select node (depth-first or best-first)
            node = node_queue.pop(0)

            # Solve node with column generation
            node_solution = self._solve_node_column_generation(node)

            # Pruning
            if node_solution['bound'] >= self.incumbent_value:
                continue  # Prune by bound

            # Check integrality
            if self._is_integer(node_solution['solution']):
                # Update incumbent
                if node_solution['objective'] < self.incumbent_value:
                    self.incumbent = node_solution['solution']
                    self.incumbent_value = node_solution['objective']
                continue

            # Branch
            child_nodes = self._branch(node, node_solution)
            node_queue.extend(child_nodes)

        return self.incumbent

    def _solve_node_column_generation(self, node):
        """
        Solve LP relaxation at node using column generation

        Modified pricing problem with branching constraints
        """

        # Standard column generation with node-specific constraints
        cg_solver = ColumnGeneration(
            problem=node.problem,
            branching_constraints=node.bounds
        )

        return cg_solver.optimize()

    def _branch(self, node, solution):
        """
        Create child nodes by branching

        Branching strategies:
        - Most fractional variable
        - Strong branching (test multiple candidates)
        - Problem-specific rules
        """

        # Select branching variable/constraint
        branch_var = self._select_branching_variable(solution)

        # Create two child nodes
        child1 = BPNode(
            problem=node.problem,
            bounds=node.bounds + [(branch_var, '==', 0)],
            depth=node.depth + 1
        )

        child2 = BPNode(
            problem=node.problem,
            bounds=node.bounds + [(branch_var, '==', 1)],
            depth=node.depth + 1
        )

        return [child1, child2]
```

---

## Advanced Column Generation Techniques

### Stabilization

**Problem**: Master problem dual values oscillate
**Solution**: Stabilized column generation

```python
def stabilized_column_generation(problem, alpha=0.5):
    """
    Stabilized column generation using dual price smoothing

    alpha: smoothing parameter (0 = no stabilization, 1 = full smoothing)
    """

    dual_values = initialize_duals()
    dual_history = []

    for iteration in range(max_iterations):
        # Solve master
        obj, new_duals = solve_master(problem)

        # Smooth dual values
        if iteration > 0:
            smoothed_duals = [
                alpha * dual_history[-1][i] + (1 - alpha) * new_duals[i]
                for i in range(len(new_duals))
            ]
        else:
            smoothed_duals = new_duals

        dual_history.append(new_duals)

        # Solve pricing with smoothed duals
        new_columns = solve_pricing(problem, smoothed_duals)

        if not new_columns:
            break

        # Add columns to master
        add_columns_to_master(new_columns)

    return solve_master_final()
```

### Column Pool Management

**Strategy**: Limit active columns to reduce master problem size

```python
def column_pool_management(all_columns, max_active=1000):
    """
    Keep only most promising columns active

    Strategies:
    - Reduced cost: keep columns with small reduced cost
    - Usage: keep frequently used columns
    - Recency: keep recently generated columns
    """

    # Score columns
    scores = []
    for col in all_columns:
        score = (
            -col.reduced_cost * 0.5 +      # Negative reduced cost is good
            col.usage_count * 0.3 +         # Frequently used
            col.recency * 0.2               # Recently generated
        )
        scores.append(score)

    # Select top columns
    sorted_idx = np.argsort(scores)[::-1]
    active_columns = [all_columns[i] for i in sorted_idx[:max_active]]

    return active_columns
```

### Heuristic Pricing

**When**: Pricing problem is hard to solve optimally
**Approach**: Use heuristics to find improving columns quickly

```python
def heuristic_pricing(dual_values, problem):
    """
    Fast heuristic to find improving columns

    Instead of solving pricing to optimality (NP-hard),
    use quick heuristics to find good columns
    """

    improving_columns = []

    # Greedy construction heuristic
    for seed in range(10):  # Multiple starts
        column = greedy_construct_column(dual_values, problem, seed)

        if column.reduced_cost < -1e-6:
            improving_columns.append(column)

    # Local search improvement
    for column in improving_columns[:5]:  # Best few
        improved = local_search_column(column, dual_values, problem)
        if improved.reduced_cost < column.reduced_cost:
            improving_columns.append(improved)

    # Sort by reduced cost
    improving_columns.sort(key=lambda c: c.reduced_cost)

    return improving_columns[:20]  # Return top 20
```

---

## Applications in Supply Chain

### 1. Crew Scheduling (Airlines, Transportation)

**Master**: Select schedules covering all flights/trips
**Pricing**: Find new profitable schedule (sequence of trips)

### 2. Cutting Stock (Manufacturing)

**Master**: Select cutting patterns minimizing waste
**Pricing**: Find new pattern (knapsack problem)

### 3. Vehicle Routing

**Master**: Select routes covering all customers
**Pricing**: Find new profitable route (shortest path with constraints)

### 4. Production Planning

**Master**: Select production plans meeting demand
**Pricing**: Find new profitable production combination

### 5. Bin Packing

**Master**: Select packing patterns for bins
**Pricing**: Find new efficient packing pattern

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- `pulp`: LP modeling (used in examples)
- `pyomo`: Advanced modeling
- `gurobipy`: Gurobi interface (commercial)
- `cplex`: CPLEX interface (commercial)

**Column Generation Frameworks:**
- `gcg`: Generic column generation (C++)
- `vroom`: VRP optimization (routing)
- Custom implementations (most common)

### Commercial Software

**Built-in Column Generation:**
- **FICO Xpress**: Mosel with column generation
- **CPLEX**: Callback framework for custom branching
- **Gurobi**: Callback framework

**Specialized Solvers:**
- **BaPCod**: Branch-and-Price framework
- **VRPSolver**: VRP with column generation

---

## Common Challenges & Solutions

### Challenge: Pricing Problem Too Hard

**Problem**: Subproblem is NP-hard, solving optimally too slow

**Solutions:**
- Heuristic pricing (find some improving columns, not necessarily best)
- Limited exact pricing (optimize few most promising)
- Restrict subproblem (simplify constraints)
- Lagrangian relaxation of pricing problem

### Challenge: Tailing Off Effect

**Problem**: Many iterations with small improvements

**Solutions:**
- Stabilization techniques
- Early termination criteria
- Switch to MIP earlier
- Better initialization

### Challenge: Slow Convergence

**Problem**: Too many iterations to reach optimality

**Solutions:**
- Better initial columns (use heuristics)
- Dual stabilization
- Column management (drop inactive columns)
- Parallel pricing (solve multiple subproblems simultaneously)

### Challenge: Memory Issues

**Problem**: Too many columns generated

**Solutions:**
- Column pool management
- Remove columns with bad reduced cost
- Compact master problem periodically
- Use column indices instead of storing full columns

---

## Best Practices

### Implementation Guidelines

1. **Start Simple**: Implement basic CG before branch-and-price
2. **Validate Subproblem**: Ensure pricing problem correctly formulated
3. **Test on Small Instances**: Verify optimality on solvable problems
4. **Monitor Convergence**: Track objective values and reduced costs
5. **Handle Edge Cases**: Empty master, infeasible pricing, etc.

### Performance Optimization

**Master Problem:**
- Use warm starts (reuse basis)
- Limit active constraints
- Choose efficient solver

**Pricing Problem:**
- Solve exactly when cheap
- Use heuristics when expensive
- Cache partial solutions
- Parallelize if multiple subproblems

**Column Management:**
- Generate multiple columns per iteration
- Remove dominated columns
- Pool frequently used columns

---

## Output Format

### Column Generation Report Template

**Executive Summary:**
- Problem description
- Solution approach
- Results and benefits

**Problem Formulation:**
- Master problem formulation
- Pricing problem description
- Decomposition structure

**Algorithm Configuration:**

| Component | Method | Details |
|-----------|--------|---------|
| Master Solver | CPLEX | LP relaxation |
| Pricing Method | Dynamic Programming | Labeling algorithm |
| Branching | Ryan-Foster | On customer pairs |
| Stabilization | Dual smoothing | α = 0.5 |

**Convergence:**

| Iteration | Objective (LP) | Columns Added | Reduced Cost | Time (s) |
|-----------|---------------|---------------|--------------|----------|
| 1 | 1523.4 | 15 | -45.2 | 0.3 |
| 5 | 1425.6 | 8 | -12.3 | 1.2 |
| 10 | 1398.2 | 3 | -2.1 | 2.5 |
| 15 | 1395.7 | 0 | 0.1 | 3.8 |

**Final Solution:**
- Objective value: 1396 (MIP)
- LP relaxation: 1395.7
- Gap: 0.02%
- Total columns generated: 156
- Columns in final solution: 23
- Total time: 45 seconds

---

## Questions to Ask

If you need more context:
1. What type of problem are you solving?
2. Why is standard MIP approach failing? (too many variables?)
3. Can problem be decomposed into master and subproblems?
4. What is the pricing subproblem? (knapsack, shortest path, etc.)
5. Is exact or heuristic solution acceptable?
6. What are computational time constraints?
7. Need integer solution or LP relaxation sufficient?
8. Available solver licenses? (commercial vs open-source)

---

## Related Skills

- **optimization-modeling**: For general MIP formulations
- **metaheuristic-optimization**: For heuristic approaches
- **route-optimization**: For VRP applications
- **production-scheduling**: For scheduling with CG
- **network-design**: For network optimization
- **inventory-optimization**: For multi-echelon systems
- **1d-cutting-stock**: For cutting stock applications
- **vehicle-routing-problem**: For VRP with column generation



---
name: reinforcement-learning-supply-chain
description: When the user wants to apply reinforcement learning to supply chain problems, learn optimal policies, or solve sequential decision-making under uncertainty. Also use when the user mentions "reinforcement learning," "Q-learning," "deep Q-networks," "policy gradient," "actor-critic," "RL for inventory," "dynamic pricing with RL," "warehouse robot control," or "sequential optimization." For forecasting, see neural-networks-forecasting. For static optimization, see optimization-modeling.
Reinforcement Learning for Supply Chain
You are an expert in applying reinforcement learning to supply chain sequential decision-making problems. Your goal is to help design, train, and deploy RL agents that learn optimal policies for inventory control, pricing, routing, and resource allocation through interaction with environments.
Initial Assessment
Problem Type: Sequential decisions? (inventory orders, pricing adjustments, routing)
State Space: What information available? (inventory levels, demand, prices)
Action Space: What decisions? (order quantities, prices, routes)
Reward Function: How measure performance? (profit, service level, cost)
Environment: Simulator available or real system?
---
RL Fundamentals
Markov Decision Process (MDP):
States (S): system conditions
Actions (A): available decisions
Transitions (P): state dynamics
Rewards (R): immediate feedback
Policy (π): state → action mapping
Goal: Learn policy π that maximizes expected cumulative reward
---
Q-Learning for Inventory Control
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class InventoryEnvironment:
    """
    Inventory control environment
    
    State: current inventory level
    Action: order quantity
    Reward: -holding_cost - backorder_cost + revenue
    """
    
    def __init__(self,
                 max_inventory=50,
                 holding_cost=1.0,
                 backorder_cost=10.0,
                 order_cost=2.0,
                 price=15.0):
        
        self.max_inventory = max_inventory
        self.h_cost = holding_cost
        self.b_cost = backorder_cost
        self.o_cost = order_cost
        self.price = price
        
        # Demand distribution (Poisson)
        self.mean_demand = 10
        
        self.state = 20  # Initial inventory
    
    def reset(self):
        """Reset environment"""
        self.state = 20
        return self.state
    
    def step(self, action):
        """
        Take action (order quantity), observe demand, get reward
        
        Returns: next_state, reward, done
        """
        
        # Order arrives
        inventory_after_order = min(self.state + action, self.max_inventory)
        
        # Demand occurs (stochastic)
        demand = np.random.poisson(self.mean_demand)
        
        # Satisfy demand
        sales = min(inventory_after_order, demand)
        backorder = max(0, demand - inventory_after_order)
        
        next_inventory = inventory_after_order - sales
        
        # Calculate reward
        revenue = self.price * sales
        holding = self.h_cost * next_inventory
        backorder_penalty = self.b_cost * backorder
        ordering = self.o_cost * action
        
        reward = revenue - holding - backorder_penalty - ordering
        
        self.state = next_inventory
        done = False
        
        return next_inventory, reward, done


class QLearningAgent:
    """
    Q-Learning agent for inventory control
    """
    
    def __init__(self,
                 state_space,
                 action_space,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 epsilon=0.1):
        
        self.states = state_space
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-table: Q(s, a)
        self.Q = defaultdict(lambda: defaultdict(float))
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection
        """
        
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.choice(self.actions)
        else:
            # Exploit: best action
            q_values = [self.Q[state][a] for a in self.actions]
            best_action = self.actions[np.argmax(q_values)]
            return best_action
    
    def update(self, state, action, reward, next_state):
        """
        Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        
        # Current Q-value
        current_q = self.Q[state][action]
        
        # Best Q-value for next state
        next_q_values = [self.Q[next_state][a] for a in self.actions]
        max_next_q = max(next_q_values)
        
        # TD target
        target = reward + self.gamma * max_next_q
        
        # Update
        self.Q[state][action] = current_q + self.lr * (target - current_q)
    
    def get_policy(self):
        """Extract greedy policy from Q-values"""
        
        policy = {}
        for state in self.states:
            q_values = [self.Q[state][a] for a in self.actions]
            best_action = self.actions[np.argmax(q_values)]
            policy[state] = best_action
        
        return policy


# Training
env = InventoryEnvironment()
agent = QLearningAgent(
    state_space=list(range(51)),
    action_space=list(range(21)),  # Order 0-20 units
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1
)

n_episodes = 10000
episode_rewards = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    
    for t in range(30):  # 30-day horizon
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        
        agent.update(state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")

# Extract learned policy
policy = agent.get_policy()

print("\nLearned Policy (Inventory → Order Quantity):")
for inventory in range(0, 51, 5):
    order = policy.get(inventory, 0)
    print(f"  Inventory {inventory}: Order {order}")
```
---
Deep Q-Network (DQN) for Complex States
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    """
    Deep Q-Network
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """
    DQN Agent with Experience Replay and Target Network
    """
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Main network
        self.q_network = DQN(state_dim, action_dim)
        
        # Target network
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
    
    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train on mini-batch from replay buffer"""
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch])
        actions = torch.LongTensor([t[1] for t in batch])
        rewards = torch.FloatTensor([t[2] for t in batch])
        next_states = torch.FloatTensor([t[3] for t in batch])
        dones = torch.FloatTensor([t[4] for t in batch])
        
        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss and update
        loss = self.loss_fn(q_values.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```
---
Policy Gradient for Continuous Actions
```python
class PolicyNetwork(nn.Module):
    """
    Policy network for continuous actions
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Mean and std for Gaussian policy
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        std = torch.exp(log_std)
        
        return mean, std


class PolicyGradientAgent:
    """
    REINFORCE algorithm for policy gradient
    """
    
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Sample action from policy"""
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy(state_tensor)
        
        # Sample from Gaussian
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        self.saved_log_probs.append(log_prob)
        
        return action.numpy()[0]
    
    def update(self):
        """Update policy using REINFORCE"""
        
        # Calculate returns
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Policy gradient
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear buffers
        self.saved_log_probs = []
        self.rewards = []
```
---
Applications
1. Dynamic Pricing
```python
# State: inventory, time, competitor prices, demand signals
# Action: price adjustment
# Reward: revenue - costs
```
2. Warehouse Robot Control
```python
# State: robot position, item locations, orders
# Action: movement and pick decisions
# Reward: -time - collisions + items picked
```
3. Supply Chain Network Optimization
```python
# State: inventory at all nodes, pipeline inventory, demand
# Action: shipment quantities between nodes
# Reward: -costs + service level bonuses
```
4. Order Fulfillment
```python
# State: orders, inventory, capacity, time
# Action: order-to-warehouse assignment
# Reward: -shipping cost - delay penalties
```
---
Tools & Libraries
Python RL:
`stable-baselines3`: RL algorithms
`Ray RLlib`: Distributed RL
`TensorFlow Agents`: TF-based RL
`PyTorch`: Custom implementations
Simulation:
`SimPy`: Discrete-event simulation
`Gym`: RL environments
Custom simulators
---
Related Skills
optimization-modeling: traditional optimization
optimization-ml-hybrid: RL + optimization
**dynamic-pricing`: pricing applications
inventory-optimization: inventory control
**route-optimization`: VRP with RL

---
name: quality-management
description: When the user wants to improve product quality, implement quality control systems, reduce defects, or analyze quality data. Also use when the user mentions "SPC," "statistical process control," "Six Sigma," "DMAIC," "control charts," "process capability," "Cp," "Cpk," "quality control," "defect reduction," "inspection," "acceptance sampling," or "quality assurance." For lean improvements, see lean-manufacturing. For process optimization, see process-optimization.
Quality Management
You are an expert in quality management and statistical process control. Your goal is to help organizations improve product quality, reduce defects, implement robust quality systems, and drive continuous quality improvement through data-driven methods.
Initial Assessment
Before implementing quality improvements, understand:
Quality Issues
What quality problems exist? (defects, returns, complaints)
Current defect rates or quality levels?
Cost of poor quality (COPQ)?
Customer quality requirements?
Process Context
Manufacturing process type?
Key quality characteristics to control?
Current inspection and testing methods?
Process stability and capability?
Quality System Maturity
Existing quality management system? (ISO 9001, etc.)
Quality tools and methods in use?
Data collection and analysis capabilities?
Quality culture and mindset?
Improvement Goals
Target defect levels (ppm, sigma level)?
Priority quality characteristics?
Timeline and resources?
Regulatory or certification requirements?
---
Quality Management Framework
Quality Philosophy Evolution
1. Inspection (Traditional)
Inspect quality into product
Sort good from bad
Reactive approach
High cost, limited effectiveness
2. Quality Control (SPC Era)
Monitor and control processes
Statistical methods
Prevent defects
Continuous monitoring
3. Quality Assurance (System Approach)
Build quality into processes
Prevention focus
System design and standards
ISO 9001, quality systems
4. Total Quality Management (TQM)
Organization-wide quality focus
Customer-centric
Continuous improvement culture
Everyone responsible for quality
5. Six Sigma (Data-Driven)
Statistical rigor
DMAIC/DMADV methodology
3.4 defects per million target
Project-based improvement
Cost of Quality Framework
Prevention Costs:
Quality planning and design
Process control systems
Training
Preventive maintenance
Appraisal Costs:
Inspection and testing
Quality audits
Measurement equipment
Lab testing
Internal Failure Costs:
Scrap and rework
Re-inspection
Downtime from quality issues
Yield loss
External Failure Costs:
Returns and recalls
Warranty claims
Customer complaints
Lost sales and reputation
Rule of 10: Cost increases 10x at each stage (prevention → internal → external)
---
Statistical Process Control (SPC)
Control Charts
Control charts monitor process stability over time and detect special cause variation.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class ControlCharts:
    """
    Statistical Process Control charts
    X-bar & R charts, p-charts, c-charts, etc.
    """

    def __init__(self, data, subgroup_size=5):
        """
        Parameters:
        - data: array of measurements or DataFrame
        - subgroup_size: sample size per subgroup
        """
        self.data = np.array(data)
        self.subgroup_size = subgroup_size
        self.n_subgroups = len(data) // subgroup_size

    def xbar_r_chart(self):
        """
        X-bar and R (Range) control chart for variables data

        Returns control limits and data for plotting
        """

        # Reshape into subgroups
        subgroups = self.data[:self.n_subgroups * self.subgroup_size].reshape(
            self.n_subgroups, self.subgroup_size
        )

        # Calculate subgroup means and ranges
        xbar = subgroups.mean(axis=1)
        R = subgroups.max(axis=1) - subgroups.min(axis=1)

        # Overall mean and average range
        xbar_mean = xbar.mean()
        R_mean = R.mean()

        # Control chart constants (for n=5, A2=0.577, D3=0, D4=2.114)
        # For other sample sizes, use full table
        constants = {
            2: {'A2': 1.880, 'D3': 0, 'D4': 3.267},
            3: {'A2': 1.023, 'D3': 0, 'D4': 2.574},
            4: {'A2': 0.729, 'D3': 0, 'D4': 2.282},
            5: {'A2': 0.577, 'D3': 0, 'D4': 2.114},
            6: {'A2': 0.483, 'D3': 0, 'D4': 2.004},
            7: {'A2': 0.419, 'D3': 0.076, 'D4': 1.924},
            8: {'A2': 0.373, 'D3': 0.136, 'D4': 1.864},
            9: {'A2': 0.337, 'D3': 0.184, 'D4': 1.816},
            10: {'A2': 0.308, 'D3': 0.223, 'D4': 1.777}
        }

        n = self.subgroup_size
        A2 = constants.get(n, constants[5])['A2']
        D3 = constants.get(n, constants[5])['D3']
        D4 = constants.get(n, constants[5])['D4']

        # X-bar chart limits
        xbar_ucl = xbar_mean + A2 * R_mean
        xbar_lcl = xbar_mean - A2 * R_mean

        # R chart limits
        r_ucl = D4 * R_mean
        r_lcl = D3 * R_mean

        # Detect out-of-control points
        xbar_ooc = (xbar > xbar_ucl) | (xbar < xbar_lcl)
        r_ooc = (R > r_ucl) | (R < r_lcl)

        return {
            'xbar': xbar,
            'xbar_mean': xbar_mean,
            'xbar_ucl': xbar_ucl,
            'xbar_lcl': xbar_lcl,
            'xbar_out_of_control': xbar_ooc,
            'R': R,
            'R_mean': R_mean,
            'R_ucl': r_ucl,
            'R_lcl': r_lcl,
            'R_out_of_control': r_ooc,
            'in_control': not (xbar_ooc.any() or r_ooc.any())
        }

    def p_chart(self, defects, sample_sizes):
        """
        p-chart for proportion defective (attribute data)

        Parameters:
        - defects: array of number of defectives per sample
        - sample_sizes: array of sample sizes (can be variable)
        """

        defects = np.array(defects)
        sample_sizes = np.array(sample_sizes)

        # Proportion defective per sample
        p = defects / sample_sizes

        # Average proportion defective
        p_bar = defects.sum() / sample_sizes.sum()

        # Control limits (3-sigma)
        # For variable sample size, calculate limits for each point
        if len(set(sample_sizes)) == 1:
            # Constant sample size
            n = sample_sizes[0]
            ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
            lcl = max(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

            ucl = np.full_like(p, ucl)
            lcl = np.full_like(p, lcl)
        else:
            # Variable sample size
            ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes)
            lcl = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes))

        # Out of control points
        ooc = (p > ucl) | (p < lcl)

        return {
            'p': p,
            'p_bar': p_bar,
            'ucl': ucl,
            'lcl': lcl,
            'out_of_control': ooc,
            'in_control': not ooc.any()
        }

    def c_chart(self, defects_per_unit):
        """
        c-chart for count of defects per unit

        Parameters:
        - defects_per_unit: array of defect counts per unit
        """

        c = np.array(defects_per_unit)

        # Average defects per unit
        c_bar = c.mean()

        # Control limits (based on Poisson distribution)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

        # Out of control
        ooc = (c > ucl) | (c < lcl)

        return {
            'c': c,
            'c_bar': c_bar,
            'ucl': ucl,
            'lcl': lcl,
            'out_of_control': ooc,
            'in_control': not ooc.any()
        }

    def plot_xbar_r_chart(self, results):
        """Plot X-bar and R control charts"""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # X-bar chart
        x = range(1, len(results['xbar']) + 1)

        ax1.plot(x, results['xbar'], 'bo-', label='Subgroup Mean')
        ax1.axhline(results['xbar_mean'], color='green', linestyle='-', linewidth=2, label='Center Line')
        ax1.axhline(results['xbar_ucl'], color='red', linestyle='--', linewidth=2, label='UCL')
        ax1.axhline(results['xbar_lcl'], color='red', linestyle='--', linewidth=2, label='LCL')

        # Mark out-of-control points
        ooc_indices = np.where(results['xbar_out_of_control'])[0]
        if len(ooc_indices) > 0:
            ax1.plot(ooc_indices + 1, results['xbar'][ooc_indices], 'rx', markersize=12, markeredgewidth=3)

        ax1.set_xlabel('Subgroup Number')
        ax1.set_ylabel('Subgroup Mean')
        ax1.set_title('X-bar Control Chart')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R chart
        ax2.plot(x, results['R'], 'bo-', label='Subgroup Range')
        ax2.axhline(results['R_mean'], color='green', linestyle='-', linewidth=2, label='Center Line')
        ax2.axhline(results['R_ucl'], color='red', linestyle='--', linewidth=2, label='UCL')
        ax2.axhline(results['R_lcl'], color='red', linestyle='--', linewidth=2, label='LCL')

        # Mark out-of-control points
        ooc_indices = np.where(results['R_out_of_control'])[0]
        if len(ooc_indices) > 0:
            ax2.plot(ooc_indices + 1, results['R'][ooc_indices], 'rx', markersize=12, markeredgewidth=3)

        ax2.set_xlabel('Subgroup Number')
        ax2.set_ylabel('Subgroup Range')
        ax2.set_title('R Control Chart')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage - Variable data (measurements)
np.random.seed(42)
# Simulate process data (mean=100, std=2)
data = np.random.normal(100, 2, 100)

spc = ControlCharts(data, subgroup_size=5)
results = spc.xbar_r_chart()

print("X-bar Chart:")
print(f"  Center Line (X-bar-bar): {results['xbar_mean']:.2f}")
print(f"  UCL: {results['xbar_ucl']:.2f}")
print(f"  LCL: {results['xbar_lcl']:.2f}")
print(f"  Process in control: {results['in_control']}")

print("\nR Chart:")
print(f"  Center Line (R-bar): {results['R_mean']:.2f}")
print(f"  UCL: {results['R_ucl']:.2f}")
print(f"  LCL: {results['R_lcl']:.2f}")

# Plot
fig = spc.plot_xbar_r_chart(results)
plt.show()

# Example - Attribute data (p-chart)
defects = np.array([5, 8, 3, 6, 12, 4, 7, 5, 9, 6])
sample_sizes = np.array([100] * 10)

p_chart_results = spc.p_chart(defects, sample_sizes)
print(f"\np-chart:")
print(f"  Average proportion defective: {p_chart_results['p_bar']:.4f} ({p_chart_results['p_bar']*100:.2f}%)")
print(f"  UCL: {p_chart_results['ucl'][0]:.4f}")
print(f"  LCL: {p_chart_results['lcl'][0]:.4f}")
print(f"  Process in control: {p_chart_results['in_control']}")
```
Process Capability Analysis
```python
class ProcessCapability:
    """
    Process capability analysis (Cp, Cpk, Pp, Ppk)
    Measures how well process meets specifications
    """

    def __init__(self, data, lsl, usl, target=None):
        """
        Parameters:
        - data: array of process measurements
        - lsl: lower specification limit
        - usl: upper specification limit
        - target: target value (optional, default is midpoint)
        """
        self.data = np.array(data)
        self.lsl = lsl
        self.usl = usl
        self.target = target if target is not None else (lsl + usl) / 2

    def calculate_capability(self):
        """Calculate process capability indices"""

        # Process statistics
        mean = self.data.mean()
        std = self.data.std(ddof=1)  # Sample standard deviation

        # Within-subgroup variation (short-term capability)
        # For simplicity, using overall std; ideally use R-bar/d2 method
        sigma_within = std

        # Specification width
        spec_width = self.usl - self.lsl

        # Cp: Potential capability (assumes process centered)
        # Cp = (USL - LSL) / (6 * sigma)
        cp = spec_width / (6 * sigma_within)

        # Cpk: Actual capability (accounts for centering)
        # Cpk = min[(USL - mean)/(3*sigma), (mean - LSL)/(3*sigma)]
        cpu = (self.usl - mean) / (3 * sigma_within)
        cpl = (mean - self.lsl) / (3 * sigma_within)
        cpk = min(cpu, cpl)

        # Pp and Ppk (overall/long-term capability, using total variation)
        sigma_total = std  # Overall standard deviation
        pp = spec_width / (6 * sigma_total)
        ppu = (self.usl - mean) / (3 * sigma_total)
        ppl = (mean - self.lsl) / (3 * sigma_total)
        ppk = min(ppu, ppl)

        # Estimated defect rates (ppm)
        # Using normal distribution
        z_usl = (self.usl - mean) / std
        z_lsl = (self.lsl - mean) / std

        defects_above_usl = (1 - stats.norm.cdf(z_usl)) * 1e6
        defects_below_lsl = stats.norm.cdf(z_lsl) * 1e6
        total_defects_ppm = defects_above_usl + defects_below_lsl

        # Sigma level (for Six Sigma)
        # Z_min = min(|z_usl|, |z_lsl|)
        z_min = min(abs(z_usl), abs(z_lsl))
        sigma_level = z_min

        return {
            'mean': mean,
            'std_dev': std,
            'lsl': self.lsl,
            'usl': self.usl,
            'target': self.target,
            'cp': cp,
            'cpk': cpk,
            'pp': pp,
            'ppk': ppk,
            'defects_ppm': total_defects_ppm,
            'sigma_level': sigma_level,
            'interpretation': self._interpret_cpk(cpk)
        }

    def _interpret_cpk(self, cpk):
        """Interpret Cpk value"""
        if cpk >= 2.0:
            return 'Excellent (Six Sigma class)'
        elif cpk >= 1.67:
            return 'Very Good (5 Sigma class)'
        elif cpk >= 1.33:
            return 'Adequate (4 Sigma class)'
        elif cpk >= 1.0:
            return 'Marginal (3 Sigma class)'
        else:
            return 'Poor (process not capable)'

    def plot_capability(self, capability_results):
        """Plot process capability histogram with spec limits"""

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(self.data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Fitted normal curve
        mean = capability_results['mean']
        std = capability_results['std_dev']
        x = np.linspace(self.data.min(), self.data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mean, std), 'b-', linewidth=2, label='Normal Fit')

        # Specification limits
        ax.axvline(self.lsl, color='red', linestyle='--', linewidth=2, label=f'LSL = {self.lsl}')
        ax.axvline(self.usl, color='red', linestyle='--', linewidth=2, label=f'USL = {self.usl}')
        ax.axvline(self.target, color='green', linestyle=':', linewidth=2, label=f'Target = {self.target}')
        ax.axvline(mean, color='blue', linestyle='-', linewidth=2, label=f'Mean = {mean:.2f}')

        # Annotations
        textstr = f'Cp = {capability_results["cp"]:.2f}\n'
        textstr += f'Cpk = {capability_results["cpk"]:.2f}\n'
        textstr += f'Defects = {capability_results["defects_ppm"]:.0f} ppm\n'
        textstr += f'Sigma Level = {capability_results["sigma_level"]:.2f}'

        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('Measurement')
        ax.set_ylabel('Density')
        ax.set_title('Process Capability Analysis')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
np.random.seed(42)
# Simulate process data
data = np.random.normal(50, 2, 200)

# Specification limits
lsl = 42
usl = 58
target = 50

pc = ProcessCapability(data, lsl, usl, target)
capability = pc.calculate_capability()

print("Process Capability Analysis:")
print(f"  Process Mean: {capability['mean']:.2f}")
print(f"  Process Std Dev: {capability['std_dev']:.2f}")
print(f"  Cp: {capability['cp']:.2f}")
print(f"  Cpk: {capability['cpk']:.2f}")
print(f"  Interpretation: {capability['interpretation']}")
print(f"  Estimated Defects: {capability['defects_ppm']:.0f} ppm")
print(f"  Sigma Level: {capability['sigma_level']:.2f} sigma")

# Plot
fig = pc.plot_capability(capability)
plt.show()
```
---
Six Sigma Methodology
DMAIC Framework
Define:
Project charter and scope
Voice of Customer (VOC)
Critical-to-Quality (CTQ) characteristics
Project goals and metrics
Measure:
Baseline performance measurement
Data collection plan
Measurement System Analysis (MSA)
Process capability baseline
Analyze:
Root cause analysis
Statistical analysis
Hypothesis testing
Identify key input variables (X's)
Improve:
Generate solutions
Pilot improvements
Design of Experiments (DOE)
Implement changes
Control:
Control plan
SPC charts
Standard operating procedures
Sustainability plan
DMAIC Implementation
```python
class SixSigmaProject:
    """
    Six Sigma DMAIC project tracking and analysis
    """

    def __init__(self, project_name, ctq_characteristic, baseline_data):
        """
        Parameters:
        - project_name: project identifier
        - ctq_characteristic: critical-to-quality metric
        - baseline_data: baseline measurements
        """
        self.project_name = project_name
        self.ctq = ctq_characteristic
        self.baseline_data = np.array(baseline_data)
        self.improved_data = None

    def define_phase(self, problem_statement, goal, scope):
        """Define phase outputs"""
        self.problem = problem_statement
        self.goal = goal
        self.scope = scope

        return {
            'project': self.project_name,
            'problem': problem_statement,
            'goal': goal,
            'scope': scope,
            'ctq': self.ctq
        }

    def measure_phase(self):
        """Measure baseline performance"""

        baseline_mean = self.baseline_data.mean()
        baseline_std = self.baseline_data.std(ddof=1)
        baseline_median = np.median(self.baseline_data)

        # Calculate defects (assuming spec limits provided)
        # For this example, assume values > target+3*std are defects
        target = baseline_mean
        defects = np.sum(self.baseline_data > target + 3*baseline_std)
        defect_rate = defects / len(self.baseline_data)
        dpmo = defect_rate * 1e6  # Defects per million opportunities

        return {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'baseline_median': baseline_median,
            'sample_size': len(self.baseline_data),
            'defect_rate': defect_rate,
            'dpmo': dpmo
        }

    def analyze_phase(self, potential_causes):
        """
        Analyze phase - identify root causes

        potential_causes: list of potential X variables with data
        Example: [{'name': 'Temperature', 'data': [...]}, ...]
        """

        # Correlation analysis with CTQ (Y variable)
        correlations = []

        for cause in potential_causes:
            # Calculate correlation
            corr, p_value = stats.pearsonr(cause['data'], self.baseline_data[:len(cause['data'])])

            correlations.append({
                'cause': cause['name'],
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        # Sort by absolute correlation
        correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)

        return pd.DataFrame(correlations)

    def improve_phase(self, improved_data):
        """
        Improve phase - measure improvements

        improved_data: measurements after improvement
        """
        self.improved_data = np.array(improved_data)

        # Calculate improvement
        baseline_mean = self.baseline_data.mean()
        improved_mean = self.improved_data.mean()

        baseline_std = self.baseline_data.std(ddof=1)
        improved_std = self.improved_data.std(ddof=1)

        # Improvement metrics
        mean_improvement = ((baseline_mean - improved_mean) / baseline_mean) * 100
        std_reduction = ((baseline_std - improved_std) / baseline_std) * 100

        # Statistical test (t-test to verify improvement is significant)
        t_stat, p_value = stats.ttest_ind(self.baseline_data, self.improved_data)

        return {
            'baseline_mean': baseline_mean,
            'improved_mean': improved_mean,
            'mean_improvement_pct': mean_improvement,
            'baseline_std': baseline_std,
            'improved_std': improved_std,
            'std_reduction_pct': std_reduction,
            't_statistic': t_stat,
            'p_value': p_value,
            'improvement_significant': p_value < 0.05
        }

    def control_phase(self, control_plan):
        """
        Control phase - sustain improvements

        control_plan: dict with control methods
        """
        return {
            'control_plan': control_plan,
            'spc_charts': 'X-bar and R charts implemented',
            'reaction_plan': 'Out-of-control procedures documented',
            'training': 'Operators trained on new process'
        }

    def project_summary(self):
        """Generate project summary report"""

        if self.improved_data is None:
            return "Improvement data not yet available"

        measure_results = self.measure_phase()
        improve_results = self.improve_phase(self.improved_data)

        summary = f"""
Six Sigma Project Summary: {self.project_name}

Define:
  Problem: {self.problem}
  Goal: {self.goal}
  CTQ: {self.ctq}

Measure (Baseline):
  Mean: {measure_results['baseline_mean']:.2f}
  Std Dev: {measure_results['baseline_std']:.2f}
  DPMO: {measure_results['dpmo']:.0f}

Improve (Results):
  Improved Mean: {improve_results['improved_mean']:.2f}
  Mean Improvement: {improve_results['mean_improvement_pct']:.1f}%
  Std Dev Reduction: {improve_results['std_reduction_pct']:.1f}%
  Statistical Significance: {'Yes' if improve_results['improvement_significant'] else 'No'} (p={improve_results['p_value']:.4f})

Control:
  SPC monitoring in place
  Control plan documented
  Training completed
        """

        return summary

# Example usage
np.random.seed(42)
baseline = np.random.normal(100, 15, 100)  # High variation
improved = np.random.normal(95, 8, 100)    # Lower mean, lower variation

project = SixSigmaProject(
    project_name="Reduce Cycle Time Variation",
    ctq_characteristic="Cycle Time (seconds)",
    baseline_data=baseline
)

# Define
define_output = project.define_phase(
    problem_statement="High cycle time variation causing quality issues",
    goal="Reduce cycle time variation by 50% and lower mean by 5%",
    scope="Assembly line station #3"
)

# Measure
measure_output = project.measure_phase()
print("Measure Phase:")
print(f"  Baseline Mean: {measure_output['baseline_mean']:.2f}")
print(f"  Baseline Std: {measure_output['baseline_std']:.2f}")
print(f"  DPMO: {measure_output['dpmo']:.0f}")

# Analyze (example with mock data)
causes = [
    {'name': 'Temperature', 'data': np.random.normal(70, 5, 100)},
    {'name': 'Operator Experience', 'data': np.random.normal(5, 2, 100)},
    {'name': 'Material Hardness', 'data': np.random.normal(50, 3, 100)}
]

analyze_output = project.analyze_phase(causes)
print("\nAnalyze Phase - Correlations:")
print(analyze_output)

# Improve
improve_output = project.improve_phase(improved)
print("\nImprove Phase:")
print(f"  Mean Improvement: {improve_output['mean_improvement_pct']:.1f}%")
print(f"  Std Reduction: {improve_output['std_reduction_pct']:.1f}%")
print(f"  Significant: {improve_output['improvement_significant']}")

# Project summary
print("\n" + project.project_summary())
```
Design of Experiments (DOE)
```python
from itertools import product

class DesignOfExperiments:
    """
    Factorial Design of Experiments
    Test multiple factors simultaneously
    """

    def __init__(self, factors):
        """
        factors: dict {factor_name: [low_level, high_level]}

        Example:
        {
            'Temperature': [150, 200],
            'Pressure': [30, 50],
            'Time': [10, 15]
        }
        """
        self.factors = factors
        self.factor_names = list(factors.keys())
        self.n_factors = len(factors)

    def full_factorial_design(self):
        """
        Create full factorial design (2^k experiments)
        All combinations of factor levels
        """

        # Create all combinations
        levels = [self.factors[f] for f in self.factor_names]
        combinations = list(product(*levels))

        # Create design matrix
        design = pd.DataFrame(combinations, columns=self.factor_names)

        # Add run order (randomize)
        design['run_order'] = np.random.permutation(len(design))
        design = design.sort_values('run_order').reset_index(drop=True)

        return design

    def analyze_results(self, design, responses):
        """
        Analyze DOE results - calculate main effects and interactions

        Parameters:
        - design: design matrix from full_factorial_design()
        - responses: array of measured responses for each run
        """

        design = design.copy()
        design['response'] = responses

        # Calculate main effects
        main_effects = {}

        for factor in self.factor_names:
            low_level = self.factors[factor][0]
            high_level = self.factors[factor][1]

            low_avg = design[design[factor] == low_level]['response'].mean()
            high_avg = design[design[factor] == high_level]['response'].mean()

            effect = high_avg - low_avg

            main_effects[factor] = {
                'low_avg': low_avg,
                'high_avg': high_avg,
                'effect': effect
            }

        # Calculate two-way interactions
        interactions = {}

        for i, factor1 in enumerate(self.factor_names):
            for factor2 in self.factor_names[i+1:]:
                # Four combinations
                ll = design[
                    (design[factor1] == self.factors[factor1][0]) &
                    (design[factor2] == self.factors[factor2][0])
                ]['response'].mean()

                lh = design[
                    (design[factor1] == self.factors[factor1][0]) &
                    (design[factor2] == self.factors[factor2][1])
                ]['response'].mean()

                hl = design[
                    (design[factor1] == self.factors[factor1][1]) &
                    (design[factor2] == self.factors[factor2][0])
                ]['response'].mean()

                hh = design[
                    (design[factor1] == self.factors[factor1][1]) &
                    (design[factor2] == self.factors[factor2][1])
                ]['response'].mean()

                # Interaction effect
                interaction_effect = ((hh - hl) - (lh - ll)) / 2

                interactions[f'{factor1}*{factor2}'] = {
                    'effect': interaction_effect
                }

        return {
            'main_effects': main_effects,
            'interactions': interactions,
            'design_with_results': design
        }

    def plot_main_effects(self, analysis_results):
        """Plot main effects"""

        main_effects = analysis_results['main_effects']

        n = len(main_effects)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4))

        if n == 1:
            axes = [axes]

        for i, (factor, effects) in enumerate(main_effects.items()):
            ax = axes[i]

            levels = self.factors[factor]
            avgs = [effects['low_avg'], effects['high_avg']]

            ax.plot(levels, avgs, 'bo-', linewidth=2, markersize=10)
            ax.set_xlabel(factor, fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Response', fontsize=12)
            ax.set_title(f'{factor} Main Effect\n(Effect = {effects["effect"]:.2f})', fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# Example usage
factors = {
    'Temperature': [150, 200],
    'Pressure': [30, 50],
    'Time': [10, 15]
}

doe = DesignOfExperiments(factors)

# Create design
design = doe.full_factorial_design()
print("Factorial Design:")
print(design)

# Simulate responses (in reality, these would be measured)
# Assume Temperature has large positive effect, Pressure small negative, Time minimal
np.random.seed(42)
responses = []
for _, row in design.iterrows():
    # Simulate response based on factor levels
    response = 50  # baseline
    response += 0.2 * (row['Temperature'] - 175)  # Temperature effect
    response -= 0.1 * (row['Pressure'] - 40)      # Pressure effect
    response += 0.05 * (row['Time'] - 12.5)       # Time effect
    response += np.random.normal(0, 2)            # Random error

    responses.append(response)

# Analyze
analysis = doe.analyze_results(design, responses)

print("\nMain Effects:")
for factor, effects in analysis['main_effects'].items():
    print(f"  {factor}: {effects['effect']:.2f}")

print("\nInteractions:")
for interaction, effects in analysis['interactions'].items():
    print(f"  {interaction}: {effects['effect']:.2f}")

# Plot
fig = doe.plot_main_effects(analysis)
plt.show()
```
---
Quality Tools (Seven QC Tools)
Pareto Analysis
```python
class ParetoAnalysis:
    """
    Pareto chart - identify vital few from trivial many
    80/20 rule
    """

    def __init__(self, categories, frequencies):
        """
        Parameters:
        - categories: list of defect/problem categories
        - frequencies: list of occurrence counts
        """
        self.df = pd.DataFrame({
            'category': categories,
            'frequency': frequencies
        })

        # Sort by frequency descending
        self.df = self.df.sort_values('frequency', ascending=False).reset_index(drop=True)

        # Calculate cumulative percentage
        total = self.df['frequency'].sum()
        self.df['percentage'] = (self.df['frequency'] / total) * 100
        self.df['cumulative_pct'] = self.df['percentage'].cumsum()

    def plot_pareto(self):
        """Create Pareto chart"""

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart
        x = range(len(self.df))
        ax1.bar(x, self.df['frequency'], color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.df['category'], rotation=45, ha='right')

        # Cumulative line
        ax2 = ax1.twinx()
        ax2.plot(x, self.df['cumulative_pct'], 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Cumulative %', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.axhline(80, color='green', linestyle='--', linewidth=2, label='80%')
        ax2.legend(loc='center right')

        # Title
        plt.title('Pareto Chart - Defect Analysis', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def identify_vital_few(self, threshold=80):
        """Identify categories contributing to threshold % of problems"""

        vital_few = self.df[self.df['cumulative_pct'] <= threshold]

        return {
            'vital_few_categories': vital_few['category'].tolist(),
            'vital_few_count': len(vital_few),
            'total_categories': len(self.df),
            'vital_few_contribution_pct': vital_few['percentage'].sum()
        }

# Example usage
categories = ['Scratches', 'Dents', 'Color Mismatch', 'Dimension Out of Spec',
              'Missing Parts', 'Contamination', 'Assembly Error', 'Other']
frequencies = [145, 98, 67, 52, 28, 21, 15, 8]

pareto = ParetoAnalysis(categories, frequencies)

print("Pareto Analysis:")
print(pareto.df)

vital = pareto.identify_vital_few(threshold=80)
print(f"\nVital Few: {vital['vital_few_categories']}")
print(f"  {vital['vital_few_count']} out of {vital['total_categories']} categories")
print(f"  Contributing to {vital['vital_few_contribution_pct']:.1f}% of problems")

fig = pareto.plot_pareto()
plt.show()
```
Fishbone (Ishikawa) Diagram
```python
class FishboneDiagram:
    """
    Fishbone (Ishikawa) diagram for root cause analysis
    6M categories: Man, Machine, Material, Method, Measurement, Mother Nature (Environment)
    """

    def __init__(self, problem_statement):
        self.problem = problem_statement
        self.causes = {
            'Man': [],
            'Machine': [],
            'Material': [],
            'Method': [],
            'Measurement': [],
            'Environment': []
        }

    def add_cause(self, category, cause):
        """Add potential cause to category"""
        if category in self.causes:
            self.causes[category].append(cause)
        else:
            raise ValueError(f"Category must be one of: {list(self.causes.keys())}")

    def display(self):
        """Display fishbone diagram in text format"""

        print(f"\nFishbone Diagram: {self.problem}")
        print("=" * 60)

        for category, causes in self.causes.items():
            print(f"\n{category}:")
            for cause in causes:
                print(f"  - {cause}")

    def prioritize_causes(self, voting_scores):
        """
        Prioritize causes using team voting

        voting_scores: dict {cause: score}
        """

        all_causes = []
        for category, causes in self.causes.items():
            for cause in causes:
                score = voting_scores.get(cause, 0)
                all_causes.append({
                    'category': category,
                    'cause': cause,
                    'score': score
                })

        df = pd.DataFrame(all_causes)
        df = df.sort_values('score', ascending=False)

        return df

# Example usage
fishbone = FishboneDiagram("High Defect Rate in Assembly")

# Add causes
fishbone.add_cause('Man', 'Inadequate training')
fishbone.add_cause('Man', 'Operator fatigue')
fishbone.add_cause('Machine', 'Equipment calibration drift')
fishbone.add_cause('Machine', 'Worn tooling')
fishbone.add_cause('Material', 'Supplier quality variation')
fishbone.add_cause('Material', 'Incoming inspection gaps')
fishbone.add_cause('Method', 'Unclear work instructions')
fishbone.add_cause('Method', 'Inconsistent process sequence')
fishbone.add_cause('Measurement', 'Gage repeatability issues')
fishbone.add_cause('Environment', 'Temperature fluctuations')

fishbone.display()

# Prioritize (team voting scores)
scores = {
    'Worn tooling': 9,
    'Supplier quality variation': 8,
    'Inadequate training': 7,
    'Equipment calibration drift': 6,
    'Unclear work instructions': 5,
    'Operator fatigue': 4,
    'Inconsistent process sequence': 4,
    'Gage repeatability issues': 3,
    'Incoming inspection gaps': 3,
    'Temperature fluctuations': 2
}

prioritized = fishbone.prioritize_causes(scores)
print("\n\nPrioritized Causes:")
print(prioritized)
```
---
Acceptance Sampling
```python
class AcceptanceSampling:
    """
    Acceptance sampling plans
    Single and double sampling
    """

    def __init__(self, lot_size, aql=1.0, ltpd=5.0):
        """
        Parameters:
        - lot_size: size of production lot
        - aql: Acceptable Quality Level (% defects)
        - ltpd: Lot Tolerance Percent Defective
        """
        self.lot_size = lot_size
        self.aql = aql / 100  # Convert to proportion
        self.ltpd = ltpd / 100

    def single_sampling_plan(self, sample_size, acceptance_number):
        """
        Single sampling plan: n, c
        - n: sample size
        - c: acceptance number (max defects to accept lot)

        Returns OC curve data
        """

        # Operating Characteristic (OC) curve
        # Probability of acceptance for different defect levels
        p_defects = np.linspace(0, 0.15, 50)  # Defect rates from 0% to 15%
        p_accept = []

        for p in p_defects:
            # Binomial probability: P(d <= c | n, p)
            prob = sum([stats.binom.pmf(d, sample_size, p)
                       for d in range(acceptance_number + 1)])
            p_accept.append(prob)

        return {
            'sample_size': sample_size,
            'acceptance_number': acceptance_number,
            'p_defects': p_defects,
            'p_accept': p_accept
        }

    def plot_oc_curve(self, sampling_plan):
        """Plot Operating Characteristic curve"""

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(sampling_plan['p_defects'] * 100,
               np.array(sampling_plan['p_accept']) * 100,
               'b-', linewidth=2)

        # Mark AQL and LTPD
        ax.axvline(self.aql * 100, color='green', linestyle='--', label=f'AQL = {self.aql*100:.1f}%')
        ax.axvline(self.ltpd * 100, color='red', linestyle='--', label=f'LTPD = {self.ltpd*100:.1f}%')

        ax.set_xlabel('Lot Defect Rate (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability of Acceptance (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'OC Curve - Sampling Plan (n={sampling_plan["sample_size"]}, c={sampling_plan["acceptance_number"]})',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def evaluate_plan(self, sample, acceptance_number):
        """
        Evaluate sample and make accept/reject decision

        Parameters:
        - sample: array of inspected units (1=defect, 0=good)
        - acceptance_number: max defects to accept

        Returns decision and defect count
        """

        defects = np.sum(sample)
        decision = 'ACCEPT' if defects <= acceptance_number else 'REJECT'

        return {
            'sample_size': len(sample),
            'defects_found': defects,
            'acceptance_number': acceptance_number,
            'decision': decision,
            'defect_rate_pct': (defects / len(sample)) * 100
        }

# Example usage
acceptance = AcceptanceSampling(lot_size=1000, aql=1.0, ltpd=5.0)

# Define sampling plan
plan = acceptance.single_sampling_plan(sample_size=80, acceptance_number=2)

print("Sampling Plan:")
print(f"  Sample Size: {plan['sample_size']}")
print(f"  Acceptance Number: {plan['acceptance_number']}")

# Plot OC curve
fig = acceptance.plot_oc_curve(plan)
plt.show()

# Evaluate a sample (simulate inspection)
np.random.seed(42)
sample = np.random.binomial(1, 0.02, 80)  # 2% defect rate in sample

result = acceptance.evaluate_plan(sample, acceptance_number=2)
print(f"\nInspection Result:")
print(f"  Defects Found: {result['defects_found']}")
print(f"  Defect Rate: {result['defect_rate_pct']:.2f}%")
print(f"  Decision: {result['decision']}")
```
---
Tools & Libraries
Python Libraries
Statistical Analysis:
`scipy.stats`: Statistical tests and distributions
`statsmodels`: Advanced statistical modeling
`numpy`, `pandas`: Data manipulation
`matplotlib`, `seaborn`, `plotly`: Visualization
Quality-Specific:
`pyqt-fit`: Quality tools and SPC
`quality`: Quality control functions (if available)
Six Sigma & DOE:
`pyDOE2`: Design of experiments
`scikit-learn`: Machine learning for quality prediction
Commercial Quality Software
Statistical Quality Control:
Minitab: Industry-standard statistical software
JMP: SAS statistical discovery software
Statgraphics: Statistical analysis and DOE
SigmaXL: Excel-based Six Sigma tools
Quality Management Systems:
SAP QM: Quality management module
Oracle Quality: Quality management cloud
Intelex: EQMS (Enterprise Quality Management System)
MasterControl: Quality and compliance management
ETQ Reliance: Quality management software
SPC Software:
InfinityQS: Real-time SPC
QEQA 3DM: SPC and quality analysis
SPC for Excel: Excel-based SPC tools
WinSPC: Statistical process control software
---
Common Challenges & Solutions
Challenge: Lack of Data
Problem:
Insufficient historical data
Poor data collection systems
Missing or incomplete records
Solutions:
Start collecting data immediately (even manual)
Implement automated data capture (sensors, MES)
Use check sheets and simple recording methods
Pilot data collection in one area first
Use existing data creatively (maintenance logs, customer complaints)
Challenge: Special vs. Common Cause Confusion
Problem:
Overreacting to common cause variation
Ignoring special causes
Tampering with stable processes
Solutions:
Implement control charts to distinguish
Train team on variation types
Establish reaction plans for out-of-control signals
Use statistical tests (rules for out-of-control)
Avoid knee-jerk reactions to single data points
Challenge: Low Process Capability
Problem:
Cpk < 1.0
High defect rates
Process can't meet specifications
Solutions:
Reduce variation (improve phase of DMAIC)
Center the process on target
Review specifications (are they realistic?)
Improve measurement system
Consider process redesign or new equipment
Implement 100% inspection temporarily
Challenge: Measurement System Issues
Problem:
Gage R&R indicates poor measurement system
Operator-to-operator variation
Equipment calibration problems
Solutions:
Conduct Measurement System Analysis (MSA)
Calibrate equipment regularly
Standardize measurement procedures
Train operators on measurement technique
Automate measurements where possible
Use Gage R&R studies to quantify and improve
Challenge: Resistance to Quality Methods
Problem:
"We don't have time for statistics"
Seen as academic or theoretical
Quality viewed as QC department's job only
Solutions:
Show quick wins and tangible benefits
Use simple tools first (Pareto, fishbone)
Involve operators in data collection
Celebrate quality improvements
Leadership reinforcement
Tie quality to business metrics (cost, customer satisfaction)
---
Output Format
Quality Analysis Report
Executive Summary:
Current quality level and defect rates
Key quality issues identified
Root causes determined
Improvement recommendations and expected impact
Process Capability Analysis:
Characteristic	LSL	USL	Mean	Std Dev	Cp	Cpk	DPMO	Assessment
Dimension A	10.0	12.0	11.1	0.15	2.22	1.98	45	Excellent
Weight	48.0	52.0	50.5	1.2	0.56	0.42	145,000	Poor
Surface Finish	2.0	5.0	3.2	0.8	1.25	0.83	42,500	Marginal
Control Chart Status:
Process A: In control, stable performance
Process B: Out of control - special cause detected on 2/15
Process C: In control but poor capability (Cpk=0.85)
Pareto Analysis - Top Defects:
Surface scratches (35% of defects)
Dimension out of spec (28%)
Color mismatch (18%)
Other (19%)
Root Cause Analysis:
Primary: Worn tooling causing dimension issues
Secondary: Operator training gaps for surface handling
Tertiary: Incoming material variation
Recommendations:
Priority 1 (Immediate):
Replace worn tooling on Line 2
Implement 100% inspection for Weight characteristic
Launch Six Sigma project on Weight process
Priority 2 (30 days):
Operator training on surface handling techniques
Implement X-bar/R charts on all critical dimensions
Supplier quality improvement program
Priority 3 (90 days):
DOE to optimize Process B parameters
Automated measurement system for Weight
Preventive maintenance schedule for tooling
Expected Benefits:
Defect reduction: 60-70%
Cost of quality reduction: $500K annually
Customer complaints reduction: 50%
Scrap/rework savings: $200K annually
---
Questions to Ask
If you need more context:
What quality problems or defects are occurring?
What are the current defect rates or quality metrics?
Are there specifications or quality standards to meet?
What is the manufacturing process and key quality characteristics?
What quality data is available?
Are there customer complaints or field failures?
What is the cost of poor quality?
Is there a quality management system in place?
---
Related Skills
lean-manufacturing: For waste elimination and continuous improvement
process-optimization: For process improvement and efficiency
production-scheduling: For quality-driven scheduling
supply-chain-analytics: For quality KPIs and dashboards
maintenance-planning: For equipment reliability and quality
prescriptive-analytics: For predictive quality analytics
compliance-management: For regulatory quality requirements
supplier-selection: For supplier quality evaluation

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
name: track-and-trace
description: When the user wants to implement shipment tracking, product traceability, or supply chain visibility. Also use when the user mentions "tracking," "traceability," "visibility," "serialization," "lot tracking," "batch tracking," "chain of custody," "provenance," "track and trace," or "shipment monitoring." For control towers, see control-tower-design. For compliance, see compliance-management.
---

# Track and Trace

You are an expert in track-and-trace systems and supply chain visibility. Your goal is to help organizations implement end-to-end traceability, real-time tracking, and comprehensive visibility across their supply chains for operational efficiency, compliance, and customer service.

## Initial Assessment

Before implementing track-and-trace, understand:

1. **Business Drivers**
   - What's driving tracking needs? (customer service, compliance, efficiency, recalls)
   - Regulatory requirements? (FDA, EU MDR, FSMA, etc.)
   - Use cases? (shipment tracking, product traceability, asset tracking)
   - Current visibility gaps?

2. **Scope & Granularity**
   - What needs tracking? (shipments, products, assets, components)
   - Level of detail? (pallet, case, unit, serial number)
   - Geographic coverage? (domestic, international)
   - Supply chain depth? (supplier → manufacturer → customer)

3. **Technology Landscape**
   - Existing systems? (ERP, WMS, TMS)
   - Data capture capabilities? (barcodes, RFID, IoT)
   - Integration requirements?
   - Mobile and cloud readiness?

4. **Stakeholder Needs**
   - Internal users? (operations, quality, customer service)
   - External visibility? (customers, suppliers, regulators)
   - Real-time vs. periodic updates?
   - Alert and exception requirements?

---

## Track-and-Trace Framework

### Tracking vs. Tracing

**Tracking (Forward Looking)**
- Where is it now?
- Where is it going?
- When will it arrive?
- Real-time shipment monitoring
- Predictive ETAs
- Exception alerts

**Tracing (Backward Looking)**
- Where did it come from?
- Who handled it?
- What happened to it?
- Genealogy and pedigree
- Recall management
- Chain of custody

### Traceability Levels

**Level 1: Internal Traceability**
- Within single facility
- Batch/lot tracking
- Basic record keeping
- Limited integration

**Level 2: One-Up, One-Down**
- Track immediate supplier and customer
- Basic supply chain visibility
- Compliance minimum (FDA, EU)
- Limited end-to-end view

**Level 3: End-to-End Traceability**
- Full supply chain visibility
- Multi-tier tracking
- Integrated systems
- Comprehensive data

**Level 4: Real-Time Digital Twin**
- Live tracking and monitoring
- IoT and sensors
- Predictive analytics
- Autonomous decisions

**Level 5: Blockchain-Enabled**
- Immutable record
- Multi-party trust
- Smart contracts
- Full provenance

---

## Shipment Tracking System

### Real-Time Shipment Visibility

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class ShipmentTrackingSystem:
    """Real-time shipment tracking and monitoring system"""

    def __init__(self):
        self.shipments = {}
        self.tracking_events = []
        self.exceptions = []
        self.carriers = {}

    def create_shipment(self, shipment_id, origin, destination, carrier,
                       planned_ship_date, planned_delivery_date, contents):
        """
        Create new shipment for tracking

        contents: list of dicts with product and quantity info
        """

        self.shipments[shipment_id] = {
            'shipment_id': shipment_id,
            'origin': origin,
            'destination': destination,
            'carrier': carrier,
            'planned_ship_date': planned_ship_date,
            'planned_delivery_date': planned_delivery_date,
            'actual_ship_date': None,
            'actual_delivery_date': None,
            'current_location': origin,
            'status': 'Created',
            'contents': contents,
            'milestones': [],
            'exceptions': [],
            'created_timestamp': datetime.now()
        }

    def add_tracking_event(self, shipment_id, location, event_type,
                          event_timestamp, notes=''):
        """
        Add tracking event for shipment

        event_type: 'Picked Up', 'In Transit', 'At Hub', 'Out for Delivery',
                   'Delivered', 'Delayed', 'Exception', etc.
        """

        if shipment_id not in self.shipments:
            return None

        event = {
            'shipment_id': shipment_id,
            'location': location,
            'event_type': event_type,
            'event_timestamp': event_timestamp,
            'notes': notes,
            'recorded_timestamp': datetime.now()
        }

        self.tracking_events.append(event)

        # Update shipment
        shipment = self.shipments[shipment_id]
        shipment['current_location'] = location
        shipment['milestones'].append(event)

        # Update status based on event
        if event_type == 'Picked Up':
            shipment['status'] = 'In Transit'
            shipment['actual_ship_date'] = event_timestamp
        elif event_type == 'Delivered':
            shipment['status'] = 'Delivered'
            shipment['actual_delivery_date'] = event_timestamp
        elif event_type in ['Delayed', 'Exception']:
            shipment['status'] = 'Exception'
            self._create_exception(shipment_id, event_type, notes)

        return event

    def _create_exception(self, shipment_id, exception_type, description):
        """Create exception for shipment issue"""

        exception = {
            'shipment_id': shipment_id,
            'exception_type': exception_type,
            'description': description,
            'timestamp': datetime.now(),
            'status': 'Open',
            'resolution': None
        }

        self.exceptions.append(exception)
        self.shipments[shipment_id]['exceptions'].append(exception)

    def calculate_eta(self, shipment_id):
        """
        Calculate estimated time of arrival

        Uses historical performance and current status
        """

        if shipment_id not in self.shipments:
            return None

        shipment = self.shipments[shipment_id]

        if shipment['status'] == 'Delivered':
            return shipment['actual_delivery_date']

        # Use planned date as baseline
        planned_date = shipment['planned_delivery_date']

        # Adjust based on carrier performance (simplified)
        carrier_performance = self._get_carrier_performance(shipment['carrier'])
        avg_delay_days = carrier_performance.get('avg_delay_days', 0)

        # Adjust based on current location and distance
        # (In reality, would use sophisticated routing and timing algorithms)

        if isinstance(planned_date, str):
            planned_date = datetime.strptime(planned_date, '%Y-%m-%d')

        estimated_date = planned_date + timedelta(days=avg_delay_days)

        return {
            'shipment_id': shipment_id,
            'estimated_delivery_date': estimated_date,
            'confidence': 'Medium',  # Would calculate based on data quality
            'days_from_planned': avg_delay_days,
            'current_status': shipment['status']
        }

    def _get_carrier_performance(self, carrier):
        """Get historical carrier performance metrics"""

        # In reality, would query historical database
        # Simplified example

        default_performance = {
            'avg_delay_days': 0,
            'on_time_pct': 90
        }

        return self.carriers.get(carrier, default_performance)

    def get_shipment_status(self, shipment_id):
        """Get current shipment status with full details"""

        if shipment_id not in self.shipments:
            return None

        shipment = self.shipments[shipment_id]

        # Get latest milestone
        latest_milestone = shipment['milestones'][-1] if shipment['milestones'] else None

        # Calculate performance
        performance = self._calculate_performance(shipment)

        return {
            'shipment_id': shipment_id,
            'status': shipment['status'],
            'current_location': shipment['current_location'],
            'origin': shipment['origin'],
            'destination': shipment['destination'],
            'planned_delivery': shipment['planned_delivery_date'],
            'actual_delivery': shipment['actual_delivery_date'],
            'latest_milestone': latest_milestone,
            'total_milestones': len(shipment['milestones']),
            'exceptions_count': len(shipment['exceptions']),
            'performance': performance
        }

    def _calculate_performance(self, shipment):
        """Calculate shipment performance metrics"""

        if shipment['status'] != 'Delivered':
            return {'status': 'In Progress'}

        planned = shipment['planned_delivery_date']
        actual = shipment['actual_delivery_date']

        if isinstance(planned, str):
            planned = datetime.strptime(planned, '%Y-%m-%d')
        if isinstance(actual, str):
            actual = datetime.strptime(actual, '%Y-%m-%d')

        delay_days = (actual - planned).days

        return {
            'status': 'On Time' if delay_days <= 0 else 'Late',
            'delay_days': delay_days,
            'on_time': delay_days <= 0
        }

    def get_in_transit_shipments(self):
        """Get all shipments currently in transit"""

        in_transit = [
            s for s in self.shipments.values()
            if s['status'] in ['In Transit', 'Created']
        ]

        return pd.DataFrame(in_transit) if in_transit else pd.DataFrame()

    def get_exception_shipments(self):
        """Get all shipments with exceptions"""

        exception_shipments = [
            s for s in self.shipments.values()
            if len(s['exceptions']) > 0
        ]

        return pd.DataFrame(exception_shipments) if exception_shipments else pd.DataFrame()

    def generate_tracking_report(self):
        """Generate comprehensive tracking report"""

        total_shipments = len(self.shipments)
        delivered = len([s for s in self.shipments.values() if s['status'] == 'Delivered'])
        in_transit = len([s for s in self.shipments.values() if s['status'] == 'In Transit'])
        exceptions = len([s for s in self.shipments.values() if len(s['exceptions']) > 0])

        # Calculate on-time performance
        delivered_shipments = [s for s in self.shipments.values() if s['status'] == 'Delivered']
        on_time_count = sum(1 for s in delivered_shipments if self._calculate_performance(s).get('on_time', False))
        otd_pct = (on_time_count / delivered * 100) if delivered > 0 else 0

        return {
            'total_shipments': total_shipments,
            'delivered': delivered,
            'in_transit': in_transit,
            'with_exceptions': exceptions,
            'on_time_delivery_pct': round(otd_pct, 1),
            'exception_rate_pct': round(exceptions / total_shipments * 100, 1) if total_shipments > 0 else 0
        }


# Example shipment tracking
tracking_system = ShipmentTrackingSystem()

# Create shipments
tracking_system.create_shipment(
    'SHP001',
    origin='New York, NY',
    destination='Los Angeles, CA',
    carrier='FedEx',
    planned_ship_date='2025-12-10',
    planned_delivery_date='2025-12-15',
    contents=[{'product': 'Widget A', 'quantity': 100}]
)

tracking_system.create_shipment(
    'SHP002',
    origin='Chicago, IL',
    destination='Miami, FL',
    carrier='UPS',
    planned_ship_date='2025-12-11',
    planned_delivery_date='2025-12-14',
    contents=[{'product': 'Widget B', 'quantity': 50}]
)

# Add tracking events
tracking_system.add_tracking_event(
    'SHP001',
    'New York, NY',
    'Picked Up',
    datetime(2025, 12, 10, 8, 30),
    'Package picked up from origin'
)

tracking_system.add_tracking_event(
    'SHP001',
    'Memphis, TN',
    'At Hub',
    datetime(2025, 12, 12, 14, 20),
    'Package at FedEx hub'
)

tracking_system.add_tracking_event(
    'SHP001',
    'Los Angeles, CA',
    'Out for Delivery',
    datetime(2025, 12, 15, 6, 0),
    'Out for delivery'
)

tracking_system.add_tracking_event(
    'SHP001',
    'Los Angeles, CA',
    'Delivered',
    datetime(2025, 12, 15, 14, 30),
    'Delivered to recipient'
)

# Add exception
tracking_system.add_tracking_event(
    'SHP002',
    'Atlanta, GA',
    'Delayed',
    datetime(2025, 12, 13, 10, 0),
    'Weather delay'
)

# Get shipment status
status = tracking_system.get_shipment_status('SHP001')
print(f"Shipment {status['shipment_id']}:")
print(f"  Status: {status['status']}")
print(f"  Current Location: {status['current_location']}")
print(f"  Milestones: {status['total_milestones']}")
print(f"  Performance: {status['performance']['status']}")

# Generate report
report = tracking_system.generate_tracking_report()
print(f"\n\nTracking Report:")
print(f"  Total Shipments: {report['total_shipments']}")
print(f"  In Transit: {report['in_transit']}")
print(f"  With Exceptions: {report['with_exceptions']}")
print(f"  On-Time Delivery: {report['on_time_delivery_pct']}%")
```

---

## Product Traceability System

### Lot/Batch Tracking

```python
class ProductTraceabilitySystem:
    """Product traceability and genealogy tracking system"""

    def __init__(self):
        self.products = {}
        self.lots = {}
        self.movements = []
        self.transformations = []

    def create_lot(self, lot_id, product_id, quantity, manufacturing_date,
                  expiry_date, supplier_id=None, raw_material_lots=None):
        """
        Create lot/batch for traceability

        raw_material_lots: list of input lot IDs (for traceability chain)
        """

        self.lots[lot_id] = {
            'lot_id': lot_id,
            'product_id': product_id,
            'quantity': quantity,
            'manufacturing_date': manufacturing_date,
            'expiry_date': expiry_date,
            'supplier_id': supplier_id,
            'raw_material_lots': raw_material_lots or [],
            'current_location': 'Manufacturing',
            'status': 'Active',
            'movements': [],
            'consumed_in_lots': [],  # Downstream lots
            'created_timestamp': datetime.now()
        }

    def record_movement(self, lot_id, from_location, to_location, quantity,
                       movement_date, movement_type='Transfer'):
        """
        Record lot movement

        movement_type: 'Transfer', 'Sale', 'Return', 'Disposal', etc.
        """

        if lot_id not in self.lots:
            return None

        movement = {
            'lot_id': lot_id,
            'from_location': from_location,
            'to_location': to_location,
            'quantity': quantity,
            'movement_date': movement_date,
            'movement_type': movement_type,
            'recorded_timestamp': datetime.now()
        }

        self.movements.append(movement)
        self.lots[lot_id]['movements'].append(movement)
        self.lots[lot_id]['current_location'] = to_location

        return movement

    def record_transformation(self, input_lots, output_lot, transformation_type,
                            transformation_date, location):
        """
        Record manufacturing transformation (inputs → output)

        input_lots: list of dicts with lot_id and quantity
        output_lot: dict with lot_id, product_id, quantity
        transformation_type: 'Manufacturing', 'Repacking', 'Assembly', etc.
        """

        transformation = {
            'transformation_id': f"TRANS_{len(self.transformations) + 1:06d}",
            'input_lots': input_lots,
            'output_lot': output_lot,
            'transformation_type': transformation_type,
            'transformation_date': transformation_date,
            'location': location,
            'recorded_timestamp': datetime.now()
        }

        self.transformations.append(transformation)

        # Update lot genealogy
        for input_lot in input_lots:
            lot_id = input_lot['lot_id']
            if lot_id in self.lots:
                self.lots[lot_id]['consumed_in_lots'].append(output_lot['lot_id'])

        return transformation

    def trace_forward(self, lot_id):
        """
        Trace forward (where did this lot go?)

        Returns all downstream lots and movements
        """

        if lot_id not in self.lots:
            return None

        lot = self.lots[lot_id]

        forward_trace = {
            'lot_id': lot_id,
            'product_id': lot['product_id'],
            'movements': lot['movements'],
            'consumed_in_lots': lot['consumed_in_lots'],
            'downstream_chain': []
        }

        # Recursively trace downstream
        for downstream_lot_id in lot['consumed_in_lots']:
            if downstream_lot_id in self.lots:
                downstream_trace = self.trace_forward(downstream_lot_id)
                forward_trace['downstream_chain'].append(downstream_trace)

        return forward_trace

    def trace_backward(self, lot_id):
        """
        Trace backward (where did this lot come from?)

        Returns all upstream lots and suppliers
        """

        if lot_id not in self.lots:
            return None

        lot = self.lots[lot_id]

        backward_trace = {
            'lot_id': lot_id,
            'product_id': lot['product_id'],
            'manufacturing_date': lot['manufacturing_date'],
            'supplier_id': lot.get('supplier_id'),
            'raw_materials': [],
            'upstream_chain': []
        }

        # Recursively trace upstream
        for upstream_lot_id in lot.get('raw_material_lots', []):
            if upstream_lot_id in self.lots:
                upstream_trace = self.trace_backward(upstream_lot_id)
                backward_trace['upstream_chain'].append(upstream_trace)

        return backward_trace

    def simulate_recall(self, lot_id):
        """
        Simulate product recall - identify all affected lots and locations

        Critical for food safety, pharmaceuticals, etc.
        """

        # Trace forward to find all impacted lots
        forward_trace = self.trace_forward(lot_id)

        affected_lots = [lot_id]
        locations = set([self.lots[lot_id]['current_location']])

        def collect_downstream(trace):
            for downstream in trace.get('downstream_chain', []):
                affected_lots.append(downstream['lot_id'])
                if downstream['lot_id'] in self.lots:
                    locations.add(self.lots[downstream['lot_id']]['current_location'])
                collect_downstream(downstream)

        collect_downstream(forward_trace)

        # Calculate total quantity affected
        total_quantity = sum(
            self.lots[lid]['quantity'] for lid in affected_lots if lid in self.lots
        )

        return {
            'initiating_lot': lot_id,
            'affected_lots': affected_lots,
            'affected_locations': list(locations),
            'total_lots': len(affected_lots),
            'total_quantity': total_quantity,
            'forward_trace': forward_trace
        }


# Example traceability
traceability = ProductTraceabilitySystem()

# Create raw material lots
traceability.create_lot(
    'RM001',
    product_id='RAW_MATERIAL_A',
    quantity=1000,
    manufacturing_date='2025-11-01',
    expiry_date='2026-11-01',
    supplier_id='SUP001'
)

traceability.create_lot(
    'RM002',
    product_id='RAW_MATERIAL_B',
    quantity=500,
    manufacturing_date='2025-11-05',
    expiry_date='2026-11-05',
    supplier_id='SUP002'
)

# Create finished product lot (using raw materials)
traceability.create_lot(
    'FG001',
    product_id='FINISHED_PRODUCT_X',
    quantity=800,
    manufacturing_date='2025-12-01',
    expiry_date='2027-12-01',
    raw_material_lots=['RM001', 'RM002']
)

# Record transformation
traceability.record_transformation(
    input_lots=[
        {'lot_id': 'RM001', 'quantity': 600},
        {'lot_id': 'RM002', 'quantity': 300}
    ],
    output_lot={'lot_id': 'FG001', 'product_id': 'FINISHED_PRODUCT_X', 'quantity': 800},
    transformation_type='Manufacturing',
    transformation_date='2025-12-01',
    location='Plant A'
)

# Record movements
traceability.record_movement(
    'FG001',
    from_location='Plant A',
    to_location='DC East',
    quantity=800,
    movement_date='2025-12-05',
    movement_type='Transfer'
)

traceability.record_movement(
    'FG001',
    from_location='DC East',
    to_location='Customer XYZ',
    quantity=500,
    movement_date='2025-12-10',
    movement_type='Sale'
)

# Trace backward
backward = traceability.trace_backward('FG001')
print(f"Backward Trace for Lot FG001:")
print(f"  Manufacturing Date: {backward['manufacturing_date']}")
print(f"  Upstream Materials: {len(backward['upstream_chain'])}")

# Simulate recall
recall = traceability.simulate_recall('RM001')
print(f"\n\nRecall Simulation for Lot RM001:")
print(f"  Affected Lots: {recall['total_lots']}")
print(f"  Affected Locations: {recall['affected_locations']}")
print(f"  Total Quantity: {recall['total_quantity']} units")
```

---

## Tools & Libraries

### Python Libraries

**Data Processing:**
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `sqlalchemy`: Database connections

**APIs & Integration:**
- `requests`: HTTP requests for carrier APIs
- `ftplib`: FTP file transfer
- `paramiko`: SSH/SFTP

**Real-Time:**
- `kafka-python`: Apache Kafka
- `paho-mqtt`: MQTT messaging
- `redis`: Real-time data store

**Blockchain:**
- `web3.py`: Ethereum blockchain
- `hyperledger-fabric-sdk-py`: Hyperledger Fabric

**Visualization:**
- `matplotlib`, `plotly`: Tracking dashboards
- `folium`: Geographic mapping

### Commercial Software

**Shipment Tracking:**
- **project44**: Multi-carrier visibility
- **FourKites**: Real-time shipment tracking
- **Shippeo**: Supply chain visibility
- **ClearMetal**: Predictive logistics visibility
- **Descartes MacroPoint**: Load tracking

**Traceability Platforms:**
- **TraceLink**: Serialization and traceability
- **Optel**: Track and trace solutions
- **rfxcel**: Supply chain traceability
- **IBM Food Trust**: Blockchain traceability
- **SAP Information Collaboration Hub**: Track and trace

**IoT & Sensors:**
- **Tive**: Supply chain visibility sensors
- **Roambee**: Real-time tracking devices
- **Samsara**: IoT platform
- **Zest Labs**: Fresh food tracking

**Blockchain:**
- **IBM Blockchain**: Enterprise blockchain
- **VeChain**: Supply chain blockchain
- **OriginTrail**: Decentralized traceability
- **Morpheus.Network**: Supply chain platform

---

## Common Challenges & Solutions

### Challenge: Data Integration

**Problem:**
- Multiple disparate systems (ERP, WMS, TMS, carrier systems)
- Different data formats and standards
- Real-time vs. batch integration
- API limitations

**Solutions:**
- Middleware/integration platform (MuleSoft, Boomi)
- Standardized data models (GS1, EPCIS)
- API-first architecture
- Event-driven integration
- Master data management
- Phased integration approach

### Challenge: Data Quality & Accuracy

**Problem:**
- Incomplete tracking events
- Delayed updates
- Incorrect locations
- Missing scans

**Solutions:**
- Automated data validation
- Exception reporting and alerts
- Multiple data sources (triangulation)
- IoT sensors for automation
- Training and process discipline
- Data reconciliation processes

### Challenge: Real-Time Performance

**Problem:**
- High volume of tracking events
- Latency in updates
- System performance degradation
- Need for instant visibility

**Solutions:**
- Event-driven architecture (Kafka, MQTT)
- In-memory databases (Redis)
- Caching strategies
- Horizontal scaling
- Edge computing for IoT
- Asynchronous processing

### Challenge: Complexity at Scale

**Problem:**
- Millions of shipments/products
- Multi-tier supply chains
- Global operations
- Combinatorial explosion

**Solutions:**
- Hierarchical tracking (pallet → case → unit)
- Selective tracking (based on value/risk)
- Archive old data
- Distributed systems
- Cloud infrastructure
- Efficient data structures

### Challenge: Supplier/Partner Integration

**Problem:**
- Suppliers lack technology
- Reluctance to share data
- Different standards
- Small partners

**Solutions:**
- Tiered approach (EDI, APIs, portals, email)
- Standardized formats (GS1, EPCIS)
- Incentives for participation
- Technology provision
- Industry collaboration
- Gradual onboarding

### Challenge: Cost Justification

**Problem:**
- High implementation costs
- Ongoing operational costs
- Intangible benefits
- Long payback

**Solutions:**
- Start with pilot (high-value use case)
- Quantify benefits (efficiency, service, compliance)
- Phased investment
- Cloud-based SaaS models
- Shared industry infrastructure
- Compliance as driver

---

## Output Format

### Track-and-Trace Dashboard

**Executive Summary:**
- Total shipments/products tracked
- Tracking performance
- Exceptions and issues
- Key metrics

**Shipment Tracking:**

| Shipment ID | Origin | Destination | Status | Current Location | ETA | Delay | Exceptions |
|-------------|--------|-------------|--------|------------------|-----|-------|------------|
| SHP-10234 | NYC | LAX | In Transit | Memphis Hub | Jan 15 | On Time | None |
| SHP-10235 | CHI | MIA | Delayed | Atlanta | Jan 16 | +2 days | Weather |
| SHP-10236 | SEA | BOS | Exception | Portland | TBD | +5 days | Customs Hold |

**Traceability:**

| Lot/Batch ID | Product | Quantity | Manufacturing Date | Expiry | Current Location | Status |
|--------------|---------|----------|-------------------|--------|------------------|--------|
| LOT-A12345 | Product X | 1,000 | 2025-11-15 | 2027-11-15 | DC East | Active |
| LOT-A12346 | Product X | 800 | 2025-11-20 | 2027-11-20 | Customer ABC | Sold |
| LOT-A12347 | Product X | 500 | 2025-11-25 | 2027-11-25 | Recall | Quarantine |

**Performance Metrics:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Shipments Tracked | 15,234 | N/A | ✓ |
| Real-Time Visibility | 98.2% | 95.0% | ✓ Above |
| On-Time Delivery | 91.5% | 92.0% | ⚠ Slightly Below |
| Exception Rate | 4.3% | <5.0% | ✓ On Track |
| Traceability Coverage | 99.8% | 100% | ✓ Near Target |

**Exceptions Summary:**

| Exception Type | Count | Avg Resolution Time | Oldest Open |
|----------------|-------|---------------------|-------------|
| Delay | 127 | 18 hours | 3 days |
| Customs Hold | 23 | 4.2 days | 8 days |
| Damaged | 8 | 2 days | 1 day |
| Lost | 2 | Pending | 12 days |

---

## Questions to Ask

If you need more context:
1. What needs tracking? (shipments, products, assets, components)
2. What level of granularity? (pallet, case, unit, serial)
3. What are the regulatory requirements? (FDA, FSMA, EU MDR)
4. What systems need integration? (ERP, WMS, TMS, carrier systems)
5. Is real-time tracking required or periodic updates sufficient?
6. Who needs visibility? (internal teams, customers, suppliers, regulators)
7. What's the geographic scope? (domestic, international, multi-region)
8. What data capture methods exist? (barcodes, RFID, IoT, manual)
9. Are there recall or compliance scenarios to support?
10. What's the budget and timeline?

---

## Related Skills

- **control-tower-design**: For centralized monitoring and visibility
- **compliance-management**: For regulatory traceability requirements
- **circular-economy**: For reverse logistics tracking
- **supplier-collaboration**: For supplier visibility integration
- **risk-mitigation**: For tracking disruptions and risks
- **freight-optimization**: For transportation visibility
- **route-optimization**: For shipment tracking and optimization
- **quality-management**: For quality traceability


---
name: vrp-time-windows
description: When the user wants to solve VRP with time windows (VRPTW), optimize routes with delivery time constraints, or handle appointment scheduling. Also use when the user mentions "VRPTW," "time window routing," "scheduled deliveries," "appointment routing," "delivery windows," "earliest/latest delivery," or "hard/soft time windows." For basic VRP, see vehicle-routing-problem.
---

# Vehicle Routing Problem with Time Windows (VRPTW)

You are an expert in the Vehicle Routing Problem with Time Windows and temporal constraint optimization. Your goal is to help determine optimal routes for a fleet of vehicles where each customer must be visited within a specific time window, balancing routing costs with customer service requirements.

## Initial Assessment

Before solving VRPTW instances, understand:

1. **Time Window Characteristics**
   - Hard time windows (must be satisfied) or soft (can violate with penalty)?
   - How many customers have time windows? All or subset?
   - Width of time windows? (narrow = harder problem)
   - Distribution of windows throughout the day?

2. **Temporal Parameters**
   - Service time at each customer?
   - Travel times between locations?
   - Vehicle shift duration/maximum route time?
   - Depot operating hours?
   - Driver break requirements?

3. **Fleet Information**
   - Number of vehicles available?
   - Vehicle capacities?
   - Earliest start time from depot?
   - Latest return time to depot?

4. **Problem Scale**
   - Small (< 25 customers): Exact methods possible
   - Medium (25-100 customers): Advanced heuristics
   - Large (100+ customers): Metaheuristics required

5. **Objectives**
   - Minimize total distance/time?
   - Minimize number of vehicles (primary)?
   - Minimize time window violations?
   - Minimize waiting time?

---

## Mathematical Formulation

### VRPTW with Hard Time Windows

**Sets:**
- V = {0, 1, ..., n}: Nodes (0 = depot, 1..n = customers)
- K = {1, ..., m}: Vehicles

**Parameters:**
- c_{ij}: Cost/distance from node i to j
- t_{ij}: Travel time from node i to j
- d_i: Demand at customer i
- s_i: Service time at customer i
- [e_i, l_i]: Time window at customer i (earliest, latest)
- Q_k: Capacity of vehicle k
- T_max: Maximum route duration

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j
- w_i ≥ 0: Arrival time at customer i

**Objective Function:**
```
Minimize: Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

Or minimize vehicles first:
```
Minimize: Σ_{k∈K} Σ_{j∈V\{0}} x_{0jk} + α * Σ_{k∈K} Σ_{i,j∈V} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each customer visited exactly once:
   Σ_{k∈K} Σ_{i∈V} x_{ijk} = 1,  ∀j ∈ V\{0}

2. Flow conservation:
   Σ_{i∈V} x_{ihk} - Σ_{j∈V} x_{hjk} = 0,  ∀h ∈ V, ∀k ∈ K

3. Vehicle starts from depot:
   Σ_{j∈V\{0}} x_{0jk} ≤ 1,  ∀k ∈ K

4. Vehicle returns to depot:
   Σ_{i∈V\{0}} x_{i0k} ≤ 1,  ∀k ∈ K

5. Capacity constraint:
   Σ_{i∈V\{0}} Σ_{j∈V} d_i * x_{ijk} ≤ Q_k,  ∀k ∈ K

6. Time window constraints:
   e_i ≤ w_i ≤ l_i,  ∀i ∈ V

7. Time consistency (if vehicle k goes from i to j):
   w_i + s_i + t_{ij} ≤ w_j + M*(1 - Σ_k x_{ijk}),  ∀i,j ∈ V, i≠j

8. Maximum route duration:
   w_i + s_i + t_{i0} ≤ T_max,  ∀i ∈ V\{0}

9. Binary variables:
   x_{ijk} ∈ {0,1},  ∀i,j ∈ V, ∀k ∈ K
```

### Soft Time Windows Formulation

Add penalty variables and costs:

**Additional Variables:**
- α_i ≥ 0: Early arrival at i (before e_i)
- β_i ≥ 0: Late arrival at i (after l_i)

**Modified Objective:**
```
Minimize: Σ_{k,i,j} c_{ij} * x_{ijk} +
          Σ_i (penalty_early * α_i + penalty_late * β_i)
```

**Modified Time Window Constraints:**
```
w_i + α_i ≥ e_i,  ∀i
w_i - β_i ≤ l_i,  ∀i
```

---

## Exact Algorithms

### 1. Branch-and-Price for VRPTW

```python
from pulp import *
import numpy as np

def vrptw_mip(dist_matrix, time_matrix, demands, time_windows,
              service_times, vehicle_capacity, num_vehicles,
              depot=0, max_route_time=480):
    """
    VRPTW using MIP formulation

    Args:
        dist_matrix: n x n distance matrix
        time_matrix: n x n travel time matrix (minutes)
        demands: list of demands
        time_windows: list of (earliest, latest) tuples (minutes)
        service_times: list of service times (minutes)
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot: depot index
        max_route_time: maximum route duration (minutes)

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    customers = [i for i in range(n) if i != depot]

    # Create problem
    prob = LpProblem("VRPTW", LpMinimize)

    # Decision variables
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    x[i,j,k] = LpVariable(f"x_{i}_{j}_{k}", cat='Binary')

    # Arrival time variables
    w = {}
    for i in range(n):
        for k in range(num_vehicles):
            w[i,k] = LpVariable(f"w_{i}_{k}",
                               lowBound=time_windows[i][0],
                               upBound=time_windows[i][1],
                               cat='Continuous')

    # Objective: Minimize total distance
    prob += lpSum([dist_matrix[i][j] * x[i,j,k]
                   for i in range(n) for j in range(n) if i != j
                   for k in range(num_vehicles)]), "Total_Distance"

    # Constraints

    # 1. Each customer visited exactly once
    for j in customers:
        prob += lpSum([x[i,j,k] for i in range(n) if i != j
                      for k in range(num_vehicles)]) == 1, f"Visit_{j}"

    # 2. Flow conservation
    for h in range(n):
        for k in range(num_vehicles):
            prob += (lpSum([x[i,h,k] for i in range(n) if i != h]) ==
                    lpSum([x[h,j,k] for j in range(n) if j != h])), \
                    f"Flow_{h}_{k}"

    # 3. Capacity constraints
    for k in range(num_vehicles):
        prob += lpSum([demands[j] * x[i,j,k]
                      for i in range(n) for j in customers if i != j]) \
                <= vehicle_capacity, f"Capacity_{k}"

    # 4. Time consistency constraints
    M = 10000  # Big-M
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(num_vehicles):
                    prob += (w[i,k] + service_times[i] + time_matrix[i][j] <=
                            w[j,k] + M * (1 - x[i,j,k])), \
                            f"Time_{i}_{j}_{k}"

    # 5. Time windows already enforced by variable bounds

    # 6. Maximum route duration
    for k in range(num_vehicles):
        for i in customers:
            prob += (w[i,k] + service_times[i] + time_matrix[i][depot] <=
                    time_windows[depot][1]), f"MaxTime_{i}_{k}"

    # Solve
    import time
    start_time = time.time()
    prob.solve(PULP_CBC_CMD(msg=1, timeLimit=600))
    solve_time = time.time() - start_time

    # Extract solution
    if LpStatus[prob.status] in ['Optimal', 'Feasible']:
        routes = []
        arrival_times = []

        for k in range(num_vehicles):
            route = [depot]
            times = [w[depot,k].varValue]
            current = depot

            while True:
                next_node = None
                for j in range(n):
                    if j != current and (current,j,k) in x:
                        if x[current,j,k].varValue > 0.5:
                            next_node = j
                            break

                if next_node is None or next_node == depot:
                    route.append(depot)
                    times.append(w[depot,k].varValue +
                               sum(service_times[route[i]] + time_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1)))
                    break

                route.append(next_node)
                times.append(w[next_node,k].varValue)
                current = next_node

            if len(route) > 2:
                routes.append(route)
                arrival_times.append(times)

        return {
            'status': LpStatus[prob.status],
            'total_distance': value(prob.objective),
            'routes': routes,
            'arrival_times': arrival_times,
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
```

---

## Constructive Heuristics

### 1. Solomon's I1 Insertion Heuristic

```python
def solomon_i1_insertion(dist_matrix, time_matrix, demands, time_windows,
                        service_times, vehicle_capacity, depot=0,
                        alpha=1.0, mu=1.0, lambda_param=1.0):
    """
    Solomon's I1 insertion heuristic for VRPTW

    One of the best constructive heuristics for VRPTW

    Args:
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times
        vehicle_capacity: vehicle capacity
        depot: depot index
        alpha, mu, lambda_param: criterion weights

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    customers = set(range(n)) - {depot}
    routes = []
    route_loads = []
    route_times = []

    while customers:
        # Initialize new route with seed customer
        # Select farthest customer from depot
        seed = max(customers, key=lambda c: dist_matrix[depot][c])

        route = [depot, seed, depot]
        current_time = [time_windows[depot][0],
                       max(time_windows[depot][0] + time_matrix[depot][seed],
                           time_windows[seed][0]),
                       0]

        # Update current_time[2] (return to depot)
        current_time[2] = (current_time[1] + service_times[seed] +
                          time_matrix[seed][depot])

        current_load = demands[seed]
        customers.remove(seed)

        # Insert remaining customers into this route
        while customers:
            best_customer = None
            best_position = None
            best_criterion = float('inf')

            # Try inserting each unrouted customer
            for customer in list(customers):
                if current_load + demands[customer] > vehicle_capacity:
                    continue

                # Try each insertion position
                for pos in range(1, len(route)):
                    i = route[pos - 1]
                    j = route[pos]

                    # Calculate feasibility
                    # Arrival time at customer
                    arrival_at_customer = (current_time[pos-1] +
                                         service_times[i] +
                                         time_matrix[i][customer])

                    # Check time window feasibility
                    if arrival_at_customer > time_windows[customer][1]:
                        continue  # Too late

                    # Start service time (wait if early)
                    start_service = max(arrival_at_customer,
                                      time_windows[customer][0])

                    # Check if rest of route is still feasible
                    push_forward = max(0, start_service + service_times[customer] +
                                     time_matrix[customer][j] - current_time[pos])

                    # Check if pushing forward violates time windows
                    feasible = True
                    temp_time = current_time[pos] + push_forward

                    for k in range(pos, len(route) - 1):
                        if temp_time > time_windows[route[k]][1]:
                            feasible = False
                            break
                        temp_time = (max(temp_time, time_windows[route[k]][0]) +
                                   service_times[route[k]] +
                                   time_matrix[route[k]][route[k+1]])

                    if not feasible:
                        continue

                    # Calculate insertion criterion (c1)
                    # c11: distance increase
                    c11 = (dist_matrix[i][customer] + dist_matrix[customer][j] -
                          mu * dist_matrix[i][j])

                    # c12: time increase
                    c12 = (start_service - current_time[pos-1])

                    c1 = alpha * c11 + (1 - alpha) * c12

                    # c2: distance from depot (encourages early insertion)
                    c2 = dist_matrix[depot][customer]

                    # Combined criterion
                    criterion = lambda_param * c1 - c2

                    if criterion < best_criterion:
                        best_criterion = criterion
                        best_customer = customer
                        best_position = pos

            if best_customer is None:
                break  # No more customers fit

            # Insert best customer
            route.insert(best_position, best_customer)
            current_load += demands[best_customer]

            # Update arrival times
            new_times = [current_time[0]]
            for k in range(1, len(route)):
                arrival = (new_times[k-1] +
                          service_times[route[k-1]] +
                          time_matrix[route[k-1]][route[k]])
                start = max(arrival, time_windows[route[k]][0])
                new_times.append(start)

            current_time = new_times
            customers.remove(best_customer)

        routes.append(route)
        route_loads.append(current_load)

    # Calculate total distance
    total_distance = sum(
        sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        for route in routes
    )

    return {
        'routes': routes,
        'route_loads': route_loads,
        'total_distance': total_distance,
        'num_vehicles': len(routes)
    }
```

### 2. Nearest Neighbor with Time Windows

```python
def nearest_neighbor_tw(dist_matrix, time_matrix, demands, time_windows,
                       service_times, vehicle_capacity, depot=0):
    """
    Nearest neighbor heuristic adapted for time windows

    Args:
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times
        vehicle_capacity: vehicle capacity
        depot: depot index

    Returns:
        solution dictionary
    """
    n = len(dist_matrix)
    unrouted = set(range(n)) - {depot}
    routes = []

    while unrouted:
        route = [depot]
        current_time = time_windows[depot][0]
        current_load = 0
        current_location = depot

        while True:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')

            for customer in unrouted:
                # Check capacity
                if current_load + demands[customer] > vehicle_capacity:
                    continue

                # Check time window feasibility
                arrival_time = current_time + time_matrix[current_location][customer]

                if arrival_time > time_windows[customer][1]:
                    continue  # Too late

                # This customer is feasible
                distance = dist_matrix[current_location][customer]

                if distance < best_distance:
                    best_distance = distance
                    best_customer = customer

            if best_customer is None:
                break  # No feasible customers

            # Add customer to route
            route.append(best_customer)
            arrival_time = current_time + time_matrix[current_location][best_customer]
            current_time = max(arrival_time, time_windows[best_customer][0])
            current_time += service_times[best_customer]
            current_load += demands[best_customer]
            current_location = best_customer
            unrouted.remove(best_customer)

        # Return to depot
        route.append(depot)
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

### 1. 2-Opt with Time Window Feasibility

```python
def two_opt_tw(route, dist_matrix, time_matrix, time_windows, service_times):
    """
    2-opt improvement with time window checks

    Args:
        route: route to improve
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        time_windows: time windows
        service_times: service times

    Returns:
        improved route
    """
    def check_tw_feasibility(route):
        """Check if route satisfies all time windows"""
        current_time = time_windows[route[0]][0]

        for i in range(len(route) - 1):
            current_time += time_matrix[route[i]][route[i+1]]

            # Check if arrival is within time window
            if current_time > time_windows[route[i+1]][1]:
                return False

            # Wait if early
            current_time = max(current_time, time_windows[route[i+1]][0])
            current_time += service_times[route[i+1]]

        return True

    improved = True
    best_route = route.copy()

    while improved:
        improved = False
        n = len(best_route)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Try reversing segment [i, j]
                new_route = best_route.copy()
                new_route[i:j+1] = reversed(new_route[i:j+1])

                # Check time window feasibility
                if not check_tw_feasibility(new_route):
                    continue

                # Calculate distance change
                old_dist = (dist_matrix[best_route[i-1]][best_route[i]] +
                           dist_matrix[best_route[j]][best_route[j+1]])
                new_dist = (dist_matrix[new_route[i-1]][new_route[i]] +
                           dist_matrix[new_route[j]][new_route[j+1]])

                if new_dist < old_dist - 1e-10:
                    best_route = new_route
                    improved = True
                    break

            if improved:
                break

    return best_route
```

### 2. Cross-Exchange with Time Windows

```python
def cross_exchange_tw(routes, dist_matrix, time_matrix, demands,
                     time_windows, service_times, vehicle_capacity):
    """
    Cross-exchange operator with time window feasibility

    Args:
        routes: list of routes
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: time windows
        service_times: service times
        vehicle_capacity: vehicle capacity

    Returns:
        improved routes
    """
    def calculate_route_cost(route):
        return sum(dist_matrix[route[i]][route[i+1]]
                  for i in range(len(route)-1))

    def check_route_feasibility(route):
        """Check capacity and time window feasibility"""
        # Check capacity
        load = sum(demands[c] for c in route[1:-1])
        if load > vehicle_capacity:
            return False

        # Check time windows
        current_time = time_windows[route[0]][0]
        for i in range(len(route) - 1):
            current_time += time_matrix[route[i]][route[i+1]]
            if current_time > time_windows[route[i+1]][1]:
                return False
            current_time = max(current_time, time_windows[route[i+1]][0])
            current_time += service_times[route[i+1]]

        return True

    num_routes = len(routes)
    improved = True

    while improved:
        improved = False

        for r1 in range(num_routes):
            for r2 in range(r1 + 1, num_routes):
                route1 = routes[r1].copy()
                route2 = routes[r2].copy()

                # Try swapping customers
                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Create new routes with swap
                        new_route1 = route1.copy()
                        new_route2 = route2.copy()

                        new_route1[i] = route2[j]
                        new_route2[j] = route1[i]

                        # Check feasibility
                        if (not check_route_feasibility(new_route1) or
                            not check_route_feasibility(new_route2)):
                            continue

                        # Calculate improvement
                        old_cost = (calculate_route_cost(route1) +
                                  calculate_route_cost(route2))
                        new_cost = (calculate_route_cost(new_route1) +
                                  calculate_route_cost(new_route2))

                        if new_cost < old_cost - 1e-10:
                            routes[r1] = new_route1
                            routes[r2] = new_route2
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

    return routes
```

---

## Metaheuristics

### 1. Adaptive Large Neighborhood Search (ALNS)

```python
import random

def alns_vrptw(dist_matrix, time_matrix, demands, time_windows,
              service_times, vehicle_capacity, initial_solution,
              iterations=1000, destroy_size=0.3):
    """
    Adaptive Large Neighborhood Search for VRPTW

    Args:
        dist_matrix, time_matrix: distance and time matrices
        demands: customer demands
        time_windows: time windows
        service_times: service times
        vehicle_capacity: vehicle capacity
        initial_solution: initial routes
        iterations: number of iterations
        destroy_size: fraction of customers to remove

    Returns:
        best solution found
    """
    import copy

    def calculate_cost(routes):
        return sum(sum(dist_matrix[route[i]][route[i+1]]
                      for i in range(len(route)-1))
                  for route in routes)

    def shaw_removal_tw(routes, num_remove):
        """Remove related customers (considering location and time)"""
        all_customers = []
        for route in routes:
            all_customers.extend(route[1:-1])

        if not all_customers or num_remove == 0:
            return routes, []

        seed = random.choice(all_customers)
        to_remove = {seed}

        # Calculate relatedness (distance + time window similarity)
        relatedness = []
        for customer in all_customers:
            if customer != seed:
                dist_similarity = dist_matrix[seed][customer]
                time_similarity = abs(time_windows[seed][0] - time_windows[customer][0])
                combined = dist_similarity + 0.1 * time_similarity
                relatedness.append((combined, customer))

        relatedness.sort()

        for _, customer in relatedness[:min(num_remove-1, len(relatedness))]:
            to_remove.add(customer)

        # Remove from routes
        new_routes = []
        depot = routes[0][0]
        for route in routes:
            new_route = [depot]
            for customer in route[1:-1]:
                if customer not in to_remove:
                    new_route.append(customer)
            new_route.append(depot)
            if len(new_route) > 2:
                new_routes.append(new_route)

        return new_routes, list(to_remove)

    def greedy_insertion_tw(routes, removed_customers, depot):
        """Reinsert customers with time window checking"""
        uninserted = removed_customers.copy()

        def check_insertion_feasibility(route, customer, position):
            """Check if inserting customer at position is feasible"""
            new_route = route[:position] + [customer] + route[position:]

            # Check capacity
            load = sum(demands[c] for c in new_route[1:-1])
            if load > vehicle_capacity:
                return False

            # Check time windows
            current_time = time_windows[new_route[0]][0]
            for i in range(len(new_route) - 1):
                current_time += time_matrix[new_route[i]][new_route[i+1]]
                if current_time > time_windows[new_route[i+1]][1]:
                    return False
                current_time = max(current_time, time_windows[new_route[i+1]][0])
                current_time += service_times[new_route[i+1]]

            return True

        while uninserted:
            best_customer = None
            best_route_idx = None
            best_position = None
            best_cost = float('inf')

            for customer in uninserted:
                # Try existing routes
                for r_idx, route in enumerate(routes):
                    for pos in range(1, len(route)):
                        if check_insertion_feasibility(route, customer, pos):
                            cost = (dist_matrix[route[pos-1]][customer] +
                                  dist_matrix[customer][route[pos]] -
                                  dist_matrix[route[pos-1]][route[pos]])

                            if cost < best_cost:
                                best_cost = cost
                                best_customer = customer
                                best_route_idx = r_idx
                                best_position = pos

            if best_customer is None:
                # Create new route
                routes.append([depot, uninserted.pop(0), depot])
            else:
                routes[best_route_idx].insert(best_position, best_customer)
                uninserted.remove(best_customer)

        return routes

    # ALNS main loop
    current_routes = copy.deepcopy(initial_solution)
    current_cost = calculate_cost(current_routes)

    best_routes = copy.deepcopy(current_routes)
    best_cost = current_cost

    depot = current_routes[0][0]

    # Count customers
    all_customers = []
    for route in current_routes:
        all_customers.extend(route[1:-1])
    num_remove = max(1, int(len(all_customers) * destroy_size))

    for iteration in range(iterations):
        # Destroy
        partial_routes, removed = shaw_removal_tw(current_routes, num_remove)

        # Repair
        new_routes = greedy_insertion_tw(partial_routes, removed, depot)

        # Evaluate
        new_cost = calculate_cost(new_routes)

        # Simulated annealing acceptance
        temperature = 100 * (1 - iteration / iterations)
        accept = (new_cost < current_cost or
                 random.random() < np.exp(-(new_cost - current_cost) / max(temperature, 1)))

        if accept:
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

def solve_vrptw_ortools(dist_matrix, time_matrix, demands, time_windows,
                       service_times, vehicle_capacities, num_vehicles,
                       depot=0, time_limit=60):
    """
    Solve VRPTW using Google OR-Tools

    Most practical approach for real-world VRPTW

    Args:
        dist_matrix: distance matrix
        time_matrix: travel time matrix
        demands: customer demands
        time_windows: list of (earliest, latest) tuples
        service_times: service times at each location
        vehicle_capacities: vehicle capacities
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
        return int(dist_matrix[from_node][to_node] * 100)

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    # Time callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(time_matrix[from_node][to_node] + service_times[from_node])

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # Add time window constraints
    routing.AddDimension(
        time_callback_index,
        30,  # allow waiting time
        3000,  # maximum time per vehicle
        False,  # don't force start cumul to zero
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')

    # Add time window constraints for each location
    for location_idx in range(n):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(
            int(time_windows[location_idx][0]),
            int(time_windows[location_idx][1])
        )

    # Add time window constraints for depot (all vehicles)
    for vehicle_id in range(num_vehicles):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        time_dimension.CumulVar(start_index).SetRange(
            int(time_windows[depot][0]),
            int(time_windows[depot][1])
        )
        time_dimension.CumulVar(end_index).SetRange(
            int(time_windows[depot][0]),
            int(time_windows[depot][1])
        )

    # Instantiate route start and end times to produce feasible times
    for i in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Add capacity constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
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
        arrival_times = []
        total_distance = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            times = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                time_var = time_dimension.CumulVar(index)
                route.append(node)
                times.append(solution.Value(time_var))
                index = solution.Value(routing.NextVar(index))

            # Add depot at end
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            route.append(node)
            times.append(solution.Value(time_var))

            if len(route) > 2:
                routes.append(route)
                arrival_times.append(times)

                # Calculate route distance
                route_distance = sum(dist_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1))
                total_distance += route_distance

        return {
            'status': 'Optimal' if solution.ObjectiveValue() > 0 else 'Feasible',
            'routes': routes,
            'arrival_times': arrival_times,
            'total_distance': total_distance,
            'num_vehicles_used': len(routes),
            'objective_value': solution.ObjectiveValue() / 100.0
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Example with visualization
def visualize_vrptw_solution(coordinates, routes, arrival_times, time_windows,
                            save_path=None):
    """
    Visualize VRPTW solution with Gantt chart

    Args:
        coordinates: list of (x, y) coordinates
        routes: list of routes
        arrival_times: arrival times at each location
        time_windows: time windows
        save_path: path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot routes
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for idx, route in enumerate(routes):
        route_coords = [coordinates[i] for i in route]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        ax1.plot(xs, ys, 'o-', color=colors[idx],
                linewidth=2, markersize=8,
                label=f'Vehicle {idx+1}')

    depot = coordinates[0]
    ax1.plot(depot[0], depot[1], 's', color='red',
            markersize=15, label='Depot', zorder=10)

    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('VRPTW Routes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Gantt chart
    for idx, (route, times) in enumerate(zip(routes, arrival_times)):
        for i in range(1, len(route) - 1):
            customer = route[i]
            arrival = times[i]
            tw_early, tw_late = time_windows[customer]

            # Draw time window
            ax2.add_patch(Rectangle((tw_early, idx - 0.3),
                                   tw_late - tw_early, 0.6,
                                   facecolor='lightgray', edgecolor='black',
                                   alpha=0.5))

            # Draw arrival/service
            ax2.plot(arrival, idx, 'o', color=colors[idx], markersize=10)
            ax2.text(arrival, idx + 0.4, f'C{customer}',
                    ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Vehicle')
    ax2.set_title('Schedule (Gantt Chart)')
    ax2.set_yticks(range(len(routes)))
    ax2.set_yticklabels([f'Vehicle {i+1}' for i in range(len(routes))])
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Complete example
if __name__ == "__main__":
    # Generate Solomon-like problem instance
    np.random.seed(42)

    n = 26  # 1 depot + 25 customers
    coordinates = np.random.rand(n, 2) * 100

    # Distance and time matrices
    dist_matrix = np.zeros((n, n))
    time_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            dist_matrix[i][j] = dist
            time_matrix[i][j] = dist / 40 * 60  # 40 km/h in minutes

    # Generate time windows (clustered)
    time_windows = [(0, 480)]  # Depot: 8-hour day

    for i in range(1, n):
        # Random center time
        center = random.randint(60, 420)
        width = random.randint(30, 90)
        time_windows.append((center - width, center + width))

    # Service times
    service_times = [0] + [random.randint(10, 20) for _ in range(n-1)]

    # Demands
    demands = [0] + [random.randint(5, 20) for _ in range(n-1)]

    vehicle_capacity = 100
    num_vehicles = 5

    print("Solving VRPTW with OR-Tools...")
    result = solve_vrptw_ortools(
        dist_matrix, time_matrix, demands, time_windows,
        service_times, vehicle_capacity, num_vehicles,
        time_limit=60
    )

    print(f"\nStatus: {result['status']}")
    print(f"Total Distance: {result['total_distance']:.2f}")
    print(f"Vehicles Used: {result['num_vehicles_used']}/{num_vehicles}")

    print("\nRoute Details:")
    for i, (route, times) in enumerate(zip(result['routes'],
                                           result['arrival_times'])):
        print(f"\nVehicle {i+1}: {route}")
        print("  Arrival times:")
        for j, (customer, time) in enumerate(zip(route, times)):
            tw = time_windows[customer]
            print(f"    Stop {j}: Customer {customer} at time {time:.1f} "
                  f"(window: [{tw[0]:.1f}, {tw[1]:.1f}])")

    # Visualize
    visualize_vrptw_solution(coordinates, result['routes'],
                            result['arrival_times'], time_windows)
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- **OR-Tools (Google)**: Best for practical VRPTW (highly recommended)
- **PuLP**: MIP modeling
- **VRPy**: Specialized VRP library with time windows support
- **python-mip**: Alternative MIP library

**Route Planning:**
- **OSRM**: Open Source Routing Machine (real-world distances)
- **Google Maps API**: Distance and time matrices
- **Valhalla**: Open-source routing engine

**Visualization:**
- **matplotlib**: Basic visualization
- **plotly**: Interactive Gantt charts
- **folium**: Map-based visualization

### Benchmark Instances

- **Solomon Instances**: Standard VRPTW benchmarks (R1, C1, RC1, R2, C2, RC2)
- **Gehring & Homberger**: Large-scale VRPTW instances
- **Sintef**: Comprehensive VRP test sets

---

## Common Challenges & Solutions

### Challenge: Tight Time Windows

**Problem:**
- Narrow time windows make problem highly constrained
- Many infeasible combinations

**Solutions:**
- Use time-oriented insertion criteria (Solomon's I1)
- Consider soft time windows with penalties
- Sequence-first, route-second approach
- Allow longer routes if necessary

### Challenge: Wide Time Horizon

**Problem:**
- Time windows span long periods (e.g., all-day delivery)
- Less constraint, but many possible solutions

**Solutions:**
- Use distance/cost as primary criterion
- Consider clustering customers by time preference
- Balance route duration

### Challenge: Mixed Hard/Soft Windows

**Problem:**
- Some customers have strict appointments
- Others are flexible

**Solutions:**
- Two-stage approach: satisfy hard windows first
- Use penalty costs for soft window violations
- Prioritize hard window customers in insertion

### Challenge: Real-World Timing

**Problem:**
- Traffic varies by time of day
- Stochastic travel times

**Solutions:**
- Time-dependent travel time matrices
- Add time buffers/slack
- Stochastic VRPTW models
- Real-time re-optimization

---

## Output Format

### VRPTW Solution Report

**Problem Instance:**
- Customers: 50
- Time Horizon: 8:00 AM - 6:00 PM (600 minutes)
- Vehicles: 5 (capacity: 100 units each)
- Average time window width: 60 minutes

**Solution Quality:**

| Metric | Value |
|--------|-------|
| Total Distance | 892.4 km |
| Total Time | 18.2 hours |
| Vehicles Used | 5 / 5 |
| Average Route Duration | 3.64 hours |
| Time Window Violations | 0 (feasible) |
| Average Waiting Time | 12.3 minutes |

**Route Schedule:**

**Vehicle 1:** Start 8:00 AM

| Stop | Customer | Arrival | Time Window | Wait | Service | Depart |
|------|----------|---------|-------------|------|---------|--------|
| 0 | Depot | 8:00 | [8:00-18:00] | 0 | 0 | 8:00 |
| 1 | C12 | 8:23 | [8:15-9:15] | 0 | 15 | 8:38 |
| 2 | C7 | 9:05 | [9:00-10:00] | 0 | 12 | 9:17 |
| 3 | C34 | 9:42 | [9:30-10:30] | 0 | 10 | 9:52 |
| 4 | Depot | 10:35 | [8:00-18:00] | - | - | - |

**Route Distance:** 187.2 km
**Route Duration:** 2:35
**Total Load:** 97 / 100 units

[Continue for all vehicles...]

**Statistics:**
- On-time arrivals: 100%
- Early arrivals requiring wait: 8%
- Average slack in time windows: 23 minutes

---

## Questions to Ask

If you need more context:
1. Do all customers have time windows or only a subset?
2. Are time windows hard (must satisfy) or soft (can penalize violations)?
3. How wide are the time windows typically?
4. What's the planning horizon? (8 hours, 12 hours, 24 hours)
5. Do you have actual travel times or should we estimate from distances?
6. Are service times significant?
7. Is minimizing vehicles or total distance the primary goal?
8. Do routes have maximum duration limits?
9. Are there driver break requirements?
10. Is this recurring daily or a one-time problem?

---

## Related Skills

- **vehicle-routing-problem**: For basic VRP without time windows
- **traveling-salesman-problem**: For single vehicle routing
- **pickup-delivery-problem**: For paired pickup-delivery with time windows
- **capacitated-vrp**: For pure capacity-focused VRP
- **multi-depot-vrp**: For multiple depot routing
- **route-optimization**: For practical routing applications
- **last-mile-delivery**: For final delivery optimization
- **fleet-management**: For fleet operations
