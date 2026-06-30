---
name: order-batching-optimization
description: "When the user wants to optimize order batching, group orders for efficient picking, or reduce picker travel distance. Also use when the user mentions \"batch picking,\" \"order grouping,\" \"cluster picking,\" \"multi-order picking,\" \"batch-and-sort,\" or \"zone batching.\" For wave planning, see wave-planning-optimization. For picker routing, see picker-routing-optimization."
---

# Order Batching Optimization

You are an expert in warehouse order batching and picking optimization. Your goal is to help group orders into optimal batches to minimize picker travel distance, maximize picking efficiency, reduce order cycle time, and improve overall warehouse productivity.

## Initial Assessment

Before optimizing order batching, understand:

1. **Picking Method**
   - Discrete picking (one order at a time)?
   - Batch picking (multiple orders)?
   - Zone picking (pass to next zone)?
   - Cluster picking (pick-to-cart)?
   - Pick-to-light or voice picking?

2. **Order Characteristics**
   - Average lines per order?
   - Order size distribution (single-line vs. multi-line)?
   - SKU overlap between orders?
   - Order priority levels?
   - Daily order volume?

3. **Warehouse Configuration**
   - Warehouse layout (grid, diagonal, irregular)?
   - Number of aisles and pick faces?
   - Pick cart capacity (orders and units)?
   - Sorting method after batch (manual, automated)?
   - Forward pick vs. reserve locations?

4. **Current Performance**
   - Current picks per hour?
   - Picker travel distance per order?
   - Batch sizes used?
   - Sort time per batch?
   - Mispick or sort error rates?

---

## Order Batching Framework

### Batching Strategies

**1. Discrete Picking (No Batching)**
- One order per trip
- **Pros**: Simple, no sorting, low error rate
- **Cons**: High travel distance, low efficiency
- **Use**: High-value orders, complex orders, each-pick only

**2. Batch Picking**
- Pick multiple orders simultaneously
- **Pros**: Reduced travel (60-80% reduction), higher productivity
- **Cons**: Requires sorting, more complex
- **Use**: Standard warehouse operations

**3. Zone Batch Picking**
- Batch orders, but each picker handles one zone
- Pass totes/carts to next zone
- **Pros**: Smaller batches, balanced workload
- **Cons**: Handoff points, coordination needed

**4. Cluster Picking (Pick-to-Cart)**
- Multi-compartment cart (e.g., 4-8 orders)
- Pick directly into order containers
- **Pros**: No sorting, medium efficiency
- **Cons**: Limited by cart capacity, order size variability

**5. Wave-Less Batching**
- Continuous batching as orders arrive
- No fixed wave times
- **Pros**: Lower cycle time, responsive
- **Cons**: Requires sophisticated WMS

### Batching Objectives

```
Primary Goals:
1. Minimize total picker travel distance
2. Maximize picks per hour
3. Balance batch sizes (avoid very small/large)
4. Minimize sort time and errors
5. Meet order cutoff times

Constraints:
- Cart capacity (units and orders)
- Sorting capacity
- Time windows (priority orders)
- Zone limitations
```

---

## Mathematical Formulation

### Batching as Clustering Problem

**Decision Variables:**
- x[o,b] = 1 if order o assigned to batch b, 0 otherwise
- y[b] = 1 if batch b is used, 0 otherwise

**Parameters:**
- S[o,o'] = similarity score between orders o and o' (SKU overlap, proximity)
- D[o] = total travel distance for order o alone
- L[o] = number of lines in order o
- C_max = maximum capacity per batch (orders or lines)

**Objective Function:**

```
Minimize:
  Total Travel Distance + Penalty for unbalanced batches

Formally:
  Σ Σ (travel_distance[b] × y[b])
  + α × Σ (deviation from ideal batch size)
  + β × (number of batches)

where travel_distance[b] is calculated from batched order locations
```

**Constraints:**

```python
# 1. Each order in exactly one batch
for o in orders:
    Σ x[o,b] = 1  for all b

# 2. Batch capacity (orders)
for b in batches:
    Σ x[o,b] ≤ C_max_orders  for all o

# 3. Batch capacity (lines)
for b in batches:
    Σ (L[o] × x[o,b]) ≤ C_max_lines  for all o

# 4. Link batch usage to order assignment
for b in batches:
    for o in orders:
        x[o,b] ≤ y[b]

# 5. Time window constraints (priority orders)
for o in priority_orders:
    for b in batches:
        if x[o,b] = 1:
            completion_time[b] ≤ deadline[o]
```

### Similarity-Based Clustering

Orders that should be batched together have high similarity:

```
Similarity(order_i, order_j) =
  α × (SKU overlap / total unique SKUs)
  + β × (1 - distance between order centroids / max_distance)
  + γ × (time window compatibility)

Higher similarity → Better batch candidates
```

---

## Batching Algorithms

### Greedy Seed-Based Batching

```python
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def seed_based_batching(orders, locations, max_batch_size=6):
    """
    Greedy batching using seed orders

    Algorithm:
    1. Select "seed" order (e.g., most lines, earliest deadline)
    2. Add most similar orders until capacity reached
    3. Repeat for remaining orders

    Parameters:
    -----------
    orders : DataFrame
        Columns: order_id, sku_list, priority, deadline
    locations : dict
        {sku: (x, y) location}
    max_batch_size : int
        Maximum orders per batch

    Returns:
    --------
    Batch assignments
    """

    # Calculate order similarity matrix
    similarity_matrix = calculate_order_similarity(orders, locations)

    remaining_orders = set(orders['order_id'])
    batches = []
    batch_id = 1

    while remaining_orders:
        # Select seed order (highest priority remaining)
        seed_candidates = orders[orders['order_id'].isin(remaining_orders)]
        seed_order = seed_candidates.sort_values(
            ['priority', 'deadline'],
            ascending=[False, True]
        ).iloc[0]['order_id']

        # Initialize batch with seed
        batch = [seed_order]
        remaining_orders.remove(seed_order)

        # Add similar orders to batch
        while len(batch) < max_batch_size and remaining_orders:
            # Find most similar remaining order
            best_order = None
            best_similarity = -1

            for candidate in remaining_orders:
                # Average similarity to all orders in batch
                avg_similarity = np.mean([
                    similarity_matrix.loc[candidate, b_order]
                    for b_order in batch
                ])

                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_order = candidate

            if best_order and best_similarity > 0.3:  # Threshold
                batch.append(best_order)
                remaining_orders.remove(best_order)
            else:
                break  # No good candidates

        batches.append({
            'batch_id': batch_id,
            'orders': batch,
            'num_orders': len(batch),
            'seed_order': seed_order
        })
        batch_id += 1

    return pd.DataFrame(batches)


def calculate_order_similarity(orders, locations):
    """
    Calculate pairwise similarity between orders

    Similarity based on:
    - SKU overlap (Jaccard similarity)
    - Spatial proximity of pick locations
    """

    order_ids = orders['order_id'].tolist()
    n = len(order_ids)
    similarity = np.zeros((n, n))

    for i, order_i_id in enumerate(order_ids):
        for j, order_j_id in enumerate(order_ids):
            if i == j:
                similarity[i, j] = 1.0
                continue

            order_i = orders[orders['order_id'] == order_i_id].iloc[0]
            order_j = orders[orders['order_id'] == order_j_id].iloc[0]

            # SKU overlap (Jaccard)
            skus_i = set(order_i['sku_list'])
            skus_j = set(order_j['sku_list'])

            intersection = len(skus_i & skus_j)
            union = len(skus_i | skus_j)
            jaccard = intersection / union if union > 0 else 0

            # Spatial proximity (simplified: centroid distance)
            centroid_i = calculate_order_centroid(order_i['sku_list'], locations)
            centroid_j = calculate_order_centroid(order_j['sku_list'], locations)

            distance = np.linalg.norm(np.array(centroid_i) - np.array(centroid_j))
            max_distance = 500  # warehouse size
            proximity = 1 - min(distance / max_distance, 1)

            # Combined similarity
            similarity[i, j] = 0.6 * jaccard + 0.4 * proximity

    return pd.DataFrame(similarity, index=order_ids, columns=order_ids)


def calculate_order_centroid(sku_list, locations):
    """Calculate centroid of pick locations for an order"""
    coords = [locations.get(sku, (0, 0)) for sku in sku_list if sku in locations]
    if not coords:
        return (0, 0)
    return (np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords]))


# Example usage
orders = pd.DataFrame({
    'order_id': [f'ORD{i:03d}' for i in range(1, 21)],
    'sku_list': [
        np.random.choice(['SKU_A', 'SKU_B', 'SKU_C', 'SKU_D', 'SKU_E',
                         'SKU_F', 'SKU_G', 'SKU_H', 'SKU_I', 'SKU_J'],
                        size=np.random.randint(1, 8), replace=False).tolist()
        for _ in range(20)
    ],
    'priority': np.random.choice([1, 2, 3], 20),
    'deadline': pd.date_range('2024-01-01 16:00', periods=20, freq='30T')
})

locations = {
    'SKU_A': (10, 20), 'SKU_B': (15, 25), 'SKU_C': (50, 30),
    'SKU_D': (55, 35), 'SKU_E': (80, 40), 'SKU_F': (85, 45),
    'SKU_G': (20, 60), 'SKU_H': (25, 65), 'SKU_I': (60, 70),
    'SKU_J': (65, 75)
}

batches = seed_based_batching(orders, locations, max_batch_size=6)

print("Order Batching Results:")
print(f"Total Batches: {len(batches)}")
for _, batch in batches.iterrows():
    print(f"Batch {batch['batch_id']}: {batch['num_orders']} orders")
    print(f"  Orders: {', '.join(batch['orders'])}")
```

### K-Means Clustering for Batching

```python
from sklearn.cluster import KMeans

def kmeans_batching(orders, locations, num_batches):
    """
    Use K-Means clustering to batch orders

    Cluster based on spatial and temporal features

    Parameters:
    -----------
    orders : DataFrame
        Order data with sku_list, priority, deadline
    locations : dict
        SKU locations
    num_batches : int
        Target number of batches

    Returns:
    --------
    Batch assignments
    """

    # Feature engineering
    features = []

    for idx, order in orders.iterrows():
        # Spatial feature: order centroid
        centroid = calculate_order_centroid(order['sku_list'], locations)

        # Temporal feature: deadline urgency (hours until deadline)
        hours_until_deadline = (
            (order['deadline'] - pd.Timestamp.now()).total_seconds() / 3600
        )

        # Size feature: number of lines
        num_lines = len(order['sku_list'])

        # Combine features (normalized)
        features.append([
            centroid[0] / 100,  # Normalize by warehouse size
            centroid[1] / 100,
            hours_until_deadline / 24,  # Normalize to days
            num_lines / 10  # Normalize by typical order size
        ])

    features = np.array(features)

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_batches, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    orders['batch_id'] = cluster_labels + 1

    # Group into batches
    batches = []
    for batch_id in range(1, num_batches + 1):
        batch_orders = orders[orders['batch_id'] == batch_id]['order_id'].tolist()

        if batch_orders:
            batches.append({
                'batch_id': batch_id,
                'orders': batch_orders,
                'num_orders': len(batch_orders),
                'centroid': kmeans.cluster_centers_[batch_id - 1]
            })

    return pd.DataFrame(batches)


# Example
batches_kmeans = kmeans_batching(orders, locations, num_batches=4)

print("\nK-Means Batching Results:")
for _, batch in batches_kmeans.iterrows():
    print(f"Batch {batch['batch_id']}: {batch['num_orders']} orders")
```

### Optimization Model: MIP-Based Batching

```python
from pulp import *

def optimize_order_batching(orders, travel_distances, max_orders_per_batch=8,
                           max_lines_per_batch=100):
    """
    Optimal order batching using Mixed-Integer Programming

    Parameters:
    -----------
    orders : list
        Order identifiers
    travel_distances : dict
        {batch_composition: total_distance}
        Precomputed for all possible batch combinations
    max_orders_per_batch : int
        Cart capacity (orders)
    max_lines_per_batch : int
        Cart capacity (lines)

    Returns:
    --------
    Optimal batching
    """

    # For tractability, use a simplified model
    # In practice, would use heuristics to generate candidate batches
    # then solve assignment problem

    prob = LpProblem("Order_Batching", LpMinimize)

    # Generate candidate batches (simplified: all pairs and triples)
    candidate_batches = []
    batch_id = 0

    # Single-order batches
    for o in orders:
        candidate_batches.append({
            'batch_id': batch_id,
            'orders': [o],
            'distance': travel_distances.get((o,), 100)
        })
        batch_id += 1

    # Pair batches
    for i, o1 in enumerate(orders):
        for o2 in orders[i+1:]:
            candidate_batches.append({
                'batch_id': batch_id,
                'orders': [o1, o2],
                'distance': travel_distances.get((o1, o2), 150)
            })
            batch_id += 1

    batches = range(len(candidate_batches))

    # Decision variables
    # y[b] = 1 if batch b is used
    y = LpVariable.dicts("use_batch", batches, cat='Binary')

    # Objective: minimize total travel distance
    prob += lpSum([
        candidate_batches[b]['distance'] * y[b]
        for b in batches
    ]), "Total_Distance"

    # Constraints

    # Each order in exactly one batch
    for o in orders:
        prob += lpSum([
            y[b] for b in batches
            if o in candidate_batches[b]['orders']
        ]) == 1, f"Order_{o}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract solution
    selected_batches = []
    for b in batches:
        if y[b].varValue > 0.5:
            selected_batches.append(candidate_batches[b])

    return {
        'status': LpStatus[prob.status],
        'total_distance': value(prob.objective),
        'batches': selected_batches,
        'num_batches': len(selected_batches)
    }


# Example (simplified with mock distances)
orders_list = [f'ORD{i:03d}' for i in range(1, 11)]

# Mock travel distances
travel_distances = {
    (o,): np.random.randint(80, 120) for o in orders_list
}
for i, o1 in enumerate(orders_list):
    for o2 in orders_list[i+1:]:
        # Batched distance is less than sum of individual
        travel_distances[(o1, o2)] = int(
            travel_distances[(o1,)] + travel_distances[(o2,)] * 0.6
        )

result = optimize_order_batching(orders_list, travel_distances)

print(f"\nOptimization Status: {result['status']}")
print(f"Total Distance: {result['total_distance']:.0f}")
print(f"Number of Batches: {result['num_batches']}")
print("\nBatches:")
for batch in result['batches']:
    print(f"  Batch {batch['batch_id']}: {batch['orders']} "
          f"(distance: {batch['distance']})")
```

---

## Advanced Batching Techniques

### Dynamic Batching with Real-Time Updates

```python
class DynamicBatcher:
    """
    Dynamic order batching system with real-time updates
    """

    def __init__(self, max_batch_size=6, max_wait_time=15):
        """
        Parameters:
        -----------
        max_batch_size : int
            Maximum orders per batch
        max_wait_time : int
            Maximum minutes to wait for batch to fill
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_orders = []
        self.completed_batches = []
        self.current_batch_id = 1

    def add_order(self, order):
        """Add new order to pending queue"""
        order['received_time'] = datetime.now()
        self.pending_orders.append(order)

    def should_release_batch(self):
        """
        Determine if a batch should be released

        Release if:
        1. Batch size reached
        2. Oldest order exceeds max wait time
        3. High-priority order needs immediate processing
        """

        if len(self.pending_orders) == 0:
            return False

        # Check size threshold
        if len(self.pending_orders) >= self.max_batch_size:
            return True

        # Check wait time
        oldest_order = min(self.pending_orders,
                          key=lambda x: x['received_time'])
        wait_time = (datetime.now() - oldest_order['received_time']).total_seconds() / 60

        if wait_time >= self.max_wait_time:
            return True

        # Check for high-priority urgent orders
        priority_orders = [o for o in self.pending_orders if o.get('priority', 3) == 1]
        if priority_orders:
            # If priority order waiting > 5 min, release
            for po in priority_orders:
                wait = (datetime.now() - po['received_time']).total_seconds() / 60
                if wait >= 5:
                    return True

        return False

    def create_batch(self):
        """
        Create batch from pending orders using similarity-based grouping
        """

        if not self.pending_orders:
            return None

        # Sort by priority and received time
        sorted_orders = sorted(
            self.pending_orders,
            key=lambda x: (x.get('priority', 3), x['received_time'])
        )

        # Take up to max_batch_size orders
        batch_orders = sorted_orders[:self.max_batch_size]

        # Remove from pending
        for order in batch_orders:
            self.pending_orders.remove(order)

        batch = {
            'batch_id': self.current_batch_id,
            'orders': batch_orders,
            'created_time': datetime.now(),
            'num_orders': len(batch_orders)
        }

        self.completed_batches.append(batch)
        self.current_batch_id += 1

        return batch

    def optimize_pending_batches(self, locations):
        """
        Re-optimize pending orders into best batches

        Called periodically or when significant orders accumulated
        """

        if len(self.pending_orders) < 2:
            return

        # Create DataFrame from pending
        pending_df = pd.DataFrame(self.pending_orders)

        # Use seed-based batching
        batches_df = seed_based_batching(
            pending_df,
            locations,
            max_batch_size=self.max_batch_size
        )

        # Update pending with batch assignments
        for idx, row in pending_df.iterrows():
            order_id = row['order_id']
            # Find batch assignment
            for _, batch in batches_df.iterrows():
                if order_id in batch['orders']:
                    # Update order with batch hint
                    for pending_order in self.pending_orders:
                        if pending_order['order_id'] == order_id:
                            pending_order['suggested_batch'] = batch['batch_id']
                    break


# Example usage
batcher = DynamicBatcher(max_batch_size=6, max_wait_time=15)

# Simulate order arrivals
for i in range(15):
    order = {
        'order_id': f'ORD{i:03d}',
        'sku_list': np.random.choice(['SKU_A', 'SKU_B', 'SKU_C'],
                                    size=np.random.randint(1, 5),
                                    replace=False).tolist(),
        'priority': np.random.choice([1, 2, 3]),
        'deadline': datetime.now() + timedelta(hours=4)
    }
    batcher.add_order(order)

# Check if should release
if batcher.should_release_batch():
    batch = batcher.create_batch()
    print(f"Released Batch {batch['batch_id']}:")
    print(f"  Orders: {len(batch['orders'])}")
    for order in batch['orders']:
        print(f"    {order['order_id']}: {order['sku_list']}")

print(f"\nPending Orders: {len(batcher.pending_orders)}")
```

### Batch-and-Sort Optimization

```python
def optimize_batch_and_sort(batch_orders, sort_stations=4):
    """
    Optimize batch picking with downstream sorting

    Minimize: Pick time + Sort time

    Parameters:
    -----------
    batch_orders : list
        Orders in the batch
    sort_stations : int
        Number of parallel sort stations

    Returns:
    --------
    Optimized pick sequence and sort assignments
    """

    # Calculate pick route (TSP-style)
    # Simplified: assume pre-calculated

    # Calculate sort time based on item distribution
    total_items = sum(len(o['sku_list']) for o in batch_orders)

    # Sort time depends on:
    # 1. Number of unique SKUs (more touchpoints)
    # 2. Number of orders (more destinations)
    # 3. Sort method (manual, automated)

    # Manual sort time estimation (seconds per item)
    items_per_order = total_items / len(batch_orders)

    if items_per_order <= 2:
        sort_time_per_item = 3  # Simple, few items per order
    elif items_per_order <= 5:
        sort_time_per_item = 5  # Moderate complexity
    else:
        sort_time_per_item = 8  # Complex, many items per order

    total_sort_time = total_items * sort_time_per_item / sort_stations

    # Pick time estimation (assuming 100 picks/hour = 36 sec/pick)
    pick_time_per_line = 36
    total_pick_time = total_items * pick_time_per_line

    # Total time
    total_time = total_pick_time + total_sort_time

    return {
        'pick_time': total_pick_time,
        'sort_time': total_sort_time,
        'total_time': total_time,
        'efficiency': total_items / (total_time / 3600)  # items per hour
    }


# Example
batch = [
    {'order_id': 'ORD001', 'sku_list': ['A', 'B', 'C']},
    {'order_id': 'ORD002', 'sku_list': ['A', 'D']},
    {'order_id': 'ORD003', 'sku_list': ['B', 'C', 'D', 'E']},
]

result = optimize_batch_and_sort(batch, sort_stations=2)
print("\nBatch-and-Sort Analysis:")
print(f"Pick Time: {result['pick_time']:.0f} seconds")
print(f"Sort Time: {result['sort_time']:.0f} seconds")
print(f"Total Time: {result['total_time']:.0f} seconds")
print(f"Efficiency: {result['efficiency']:.1f} items/hour")
```

---

## Tools & Libraries

### Order Batching Software

**Warehouse Management Systems with Batching:**
- **Manhattan WMS**: Advanced batching algorithms
- **Blue Yonder (JDA) WMS**: AI-optimized batching
- **SAP EWM**: Batch determination rules
- **HighJump WMS**: Multi-order picking strategies
- **Körber WMS**: Dynamic batch creation

**Specialized Optimization:**
- **Optislot**: Slotting and batching optimization
- **Wise Systems**: Dynamic routing and batching for last-mile
- **Lucas Systems**: Voice-directed batch picking optimization

### Python Libraries

```python
# Optimization
from pulp import *
from scipy.optimize import linear_sum_assignment
from ortools.constraint_solver import pywrapcp

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Distance Calculations
from scipy.spatial.distance import pdist, squareform, jaccard
import networkx as nx  # For TSP routing within batch

# Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## Common Challenges & Solutions

### Challenge: High Sort Time

**Problem:**
- Sorting takes longer than picking
- Sort errors and mispicks
- Bottleneck at sort stations

**Solutions:**
- Reduce batch size (less sorting complexity)
- Use cluster picking instead (pick-to-cart, no sorting)
- Automate sorting (put-to-light, automated sorters)
- Pre-sort during picking (numbered totes per order)
- Sequence picks by order to minimize sort mixing
- Add more sort stations or lanes

### Challenge: Uneven Order Sizes

**Problem:**
- Mix of 1-line and 20-line orders
- Small orders waste batch capacity
- Large orders can't batch with others

**Solutions:**
- Separate batching strategies by order size
  - Small orders (1-3 lines): Large batches (10-12 orders)
  - Medium (4-10 lines): Standard batches (4-6 orders)
  - Large (>10 lines): Discrete picking or small batches
- Line-based batching instead of order-based
- Adjust batch size dynamically based on order profile

### Challenge: Low SKU Overlap

**Problem:**
- Orders have unique SKUs (low overlap)
- Batching doesn't reduce travel much
- Near-discrete picking efficiency

**Solutions:**
- Focus on spatial clustering (zone proximity) over SKU overlap
- Use smaller batches (2-3 orders) with better spatial fit
- Implement zone picking instead (each zone does portion)
- Consider goods-to-person systems (no travel benefits anyway)
- Improve slotting to increase density of fast-movers

### Challenge: Priority Order Conflicts

**Problem:**
- High-priority orders interrupt batches
- Can't wait for batch to fill (urgency)
- Mix of rush and standard orders

**Solutions:**
- Separate priority lanes (express batching with size=1 or 2)
- Reserve pickers for priority orders
- Dynamic batch release (don't wait for full batch if priority)
- Use different picking methods by priority (zone for standard, discrete for priority)
- Pre-allocate inventory for known rush orders

### Challenge: Cart Capacity Limitations

**Problem:**
- Physical cart only holds 6 totes
- Can't batch more orders due to equipment
- Weight/volume limits

**Solutions:**
- Multiple cart types (4-tote, 8-tote, 12-tote)
- Batch based on available cart types
- Two-pass picking (large batches split across trips)
- Use powered carts or AGVs (higher capacity)
- Zone picking with conveyors (unlimited capacity)

---

## Output Format

### Order Batching Report

**Batch Optimization Summary - January 15, 2024**

**Performance Comparison:**

| Metric | Discrete Picking | Current Batching | Optimized Batching |
|--------|------------------|------------------|-------------------|
| Avg Orders per Batch | 1.0 | 4.2 | 5.8 |
| Avg Travel per Order (ft) | 425 | 185 | 142 |
| Travel Reduction | 0% | 56% | 67% |
| Picks per Hour | 65 | 145 | 175 |
| Sort Time per Order (min) | 0 | 2.5 | 2.1 |
| Total Cycle Time (min) | 24 | 14 | 12 |

**Batch Details:**

```
Batch 001:
  Orders: 6
  Total Lines: 42
  SKU Overlap: 68%
  Estimated Travel: 620 ft
  Estimated Pick Time: 24 min
  Estimated Sort Time: 12 min
  Priority Orders: 1 (ORD0045)

  Order Breakdown:
    ORD0042: 8 lines, Zone A (4), Zone B (3), Zone C (1)
    ORD0043: 6 lines, Zone A (3), Zone B (2), Zone C (1)
    ORD0045: 9 lines [PRIORITY], Zone A (5), Zone B (3), Zone C (1)
    ORD0047: 5 lines, Zone A (2), Zone B (2), Zone C (1)
    ORD0048: 7 lines, Zone A (4), Zone B (2), Zone C (1)
    ORD0050: 7 lines, Zone A (3), Zone B (3), Zone C (1)

  Shared SKUs: SKU_A (6 orders), SKU_B (5 orders), SKU_D (4 orders)
```

**Batching Statistics:**

- Total Orders: 120
- Total Batches: 21
- Avg Batch Size: 5.7 orders
- Batch Size Distribution:
  - 3 orders: 2 batches
  - 4 orders: 4 batches
  - 5 orders: 6 batches
  - 6 orders: 7 batches
  - 7 orders: 2 batches

**Travel Savings:**

- Total Travel (Discrete): 51,000 ft
- Total Travel (Batched): 16,850 ft
- **Savings: 34,150 ft (67% reduction)**

**Productivity Impact:**

- Picker Hours (Discrete): 30.8 hours
- Picker Hours (Batched): 17.6 hours
- **Labor Savings: 13.2 hours (43% reduction)**

**Recommendations:**
1. Implement optimized batching strategy
2. Add 2 sort stations for peak capacity
3. Separate priority orders into express batches
4. Consider 8-tote carts for larger batches

---

## Questions to Ask

If you need more context:
1. What picking method do you currently use?
2. What's your average order size (lines)?
3. Do you have sorting capability after picking?
4. What's your cart capacity (orders and units)?
5. What's your daily order volume?
6. Do you have priority or rush orders?
7. What's your warehouse layout (zones, aisles)?
8. What's your current picks per hour?

---

## Related Skills

- **picker-routing-optimization**: For optimizing pick path within batches
- **wave-planning-optimization**: For wave design and release strategy
- **warehouse-slotting-optimization**: For SKU placement affecting batching
- **traveling-salesman-problem**: For TSP-based routing within batches
- **clustering-algorithms**: For order similarity and grouping
- **task-assignment-problem**: For assigning pickers to batches
- **order-fulfillment**: For overall fulfillment process design


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


---
name: scenario-planning
description: When the user wants to analyze supply chain scenarios, perform risk analysis, evaluate what-if scenarios, or build contingency plans. Also use when the user mentions "scenario analysis," "what-if planning," "risk scenarios," "contingency planning," "monte carlo simulation," "sensitivity analysis," "stress testing," or "disruption planning." For demand uncertainty, see demand-forecasting. For S&OP scenario integration, see sales-operations-planning.
---

# Scenario Planning

You are an expert in supply chain scenario planning and risk analysis. Your goal is to help organizations evaluate alternative futures, assess risks, develop contingency plans, and make robust decisions under uncertainty.

## Initial Assessment

Before building scenario plans, understand:

1. **Planning Context**
   - What decisions need to be made?
   - What uncertainties or risks are most concerning?
   - Planning horizon? (short-term tactical vs. long-term strategic)
   - Current planning process and tools?

2. **Business Environment**
   - Key market uncertainties? (demand, supply, pricing)
   - External risks? (geopolitical, natural disasters, economic)
   - Internal vulnerabilities? (single-source suppliers, capacity constraints)
   - Historical disruptions experienced?

3. **Scope & Objectives**
   - What part of supply chain? (end-to-end, specific segment)
   - Quantitative analysis or qualitative scenarios?
   - One-time analysis or ongoing process?
   - Decision context? (investment, strategy, operations)

4. **Data & Resources**
   - Historical variability data available?
   - Cost and performance data?
   - Risk databases or incident logs?
   - Modeling tools and capabilities?

---

## Scenario Planning Framework

### Types of Scenario Analysis

**1. Sensitivity Analysis**
- Test impact of single variable changes
- "What if demand increases by 10%?"
- Simple, quick insights
- Foundation for deeper analysis

**2. What-If Scenarios**
- Specific event-based scenarios
- "What if supplier X fails?"
- Defined discrete events
- Test preparedness

**3. Monte Carlo Simulation**
- Probabilistic analysis with random sampling
- Thousands of scenarios
- Generate probability distributions
- Risk quantification

**4. Strategic Scenarios**
- Long-term alternative futures
- Multiple variables changing together
- Qualitative + quantitative
- 3-5 year horizons

**5. Stress Testing**
- Extreme event scenarios
- Test system resilience
- Identify breaking points
- Regulatory or internal requirements

---

## Scenario Development Process

### Phase 1: Define Scope & Objectives

**Key Questions:**
- What decision is being supported?
- What are we trying to learn?
- What time horizon matters?
- Who needs to use the analysis?

**Output:** Clear problem statement and success criteria

### Phase 2: Identify Key Uncertainties

**External Uncertainties:**
- **Demand volatility**: Market changes, customer behavior
- **Supply disruptions**: Supplier failures, natural disasters
- **Cost volatility**: Raw materials, transportation, labor
- **Regulatory changes**: Trade policies, environmental rules
- **Technology disruption**: New competitors, obsolescence
- **Economic conditions**: Recession, inflation, currency

**Internal Uncertainties:**
- Production yields and quality
- Equipment reliability
- Workforce availability
- IT system performance
- New product launch success

**Prioritization Matrix:**

```python
import pandas as pd
import matplotlib.pyplot as plt

def prioritize_uncertainties(uncertainties_data):
    """
    Prioritize uncertainties by impact and likelihood

    Parameters:
    - uncertainties_data: list of dicts with 'name', 'impact', 'likelihood'
    """
    df = pd.DataFrame(uncertainties_data)

    # Calculate priority score
    df['priority_score'] = df['impact'] * df['likelihood']
    df = df.sort_values('priority_score', ascending=False)

    # Plot impact-likelihood matrix
    plt.figure(figsize=(10, 8))
    plt.scatter(df['likelihood'], df['impact'], s=200, alpha=0.6)

    for idx, row in df.iterrows():
        plt.annotate(row['name'],
                    (row['likelihood'], row['impact']),
                    fontsize=8)

    plt.xlabel('Likelihood (1-10)')
    plt.ylabel('Business Impact (1-10)')
    plt.title('Uncertainty Prioritization Matrix')
    plt.grid(True, alpha=0.3)

    # Add quadrant lines
    plt.axhline(y=5, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=5, color='r', linestyle='--', alpha=0.3)

    return df

# Example usage
uncertainties = [
    {'name': 'Supplier Bankruptcy', 'impact': 9, 'likelihood': 3},
    {'name': 'Demand Spike', 'impact': 7, 'likelihood': 6},
    {'name': 'Port Strike', 'impact': 8, 'likelihood': 4},
    {'name': 'Fuel Price Increase', 'impact': 6, 'likelihood': 7},
    {'name': 'Quality Issue', 'impact': 7, 'likelihood': 5}
]

priority_df = prioritize_uncertainties(uncertainties)
```

### Phase 3: Define Scenarios

**Approaches:**

**A. Driver-Based Scenarios**
- Select 2-3 key uncertain drivers
- Define high/low states for each
- Creates 2x2 or 2x2x2 scenario matrix

**Example: Demand Growth vs. Supply Stability**

|                    | Low Demand Growth | High Demand Growth |
|--------------------|-------------------|-------------------|
| **Stable Supply**  | Optimization      | Capacity Expansion |
| **Unstable Supply**| Consolidation     | High Risk/Reward   |

**B. Event-Based Scenarios**
- Specific disruptive events
- Examples: Hurricane, cyber attack, supplier bankruptcy
- Detailed impact modeling

**C. Monte Carlo Scenarios**
- Statistical distributions for uncertain variables
- Random sampling
- Large number of scenarios (1000+)

### Phase 4: Model Scenarios

**Build Base Model:**
```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SupplyChainState:
    """Represents supply chain configuration"""
    facilities: List[str]
    suppliers: Dict[str, float]  # supplier: reliability
    inventory_levels: Dict[str, float]  # location: units
    capacity: Dict[str, float]  # facility: units/month
    costs: Dict[str, float]

class ScenarioModel:
    """Supply chain scenario modeling"""

    def __init__(self, base_state: SupplyChainState):
        self.base_state = base_state
        self.scenarios = {}

    def add_scenario(self, name: str, changes: Dict):
        """Define a scenario with changes from base"""
        self.scenarios[name] = changes

    def evaluate_scenario(self, scenario_name: str,
                         demand: np.ndarray,
                         time_periods: int = 12) -> Dict:
        """
        Simulate supply chain performance under scenario

        Returns metrics: cost, service level, inventory
        """

        scenario_changes = self.scenarios[scenario_name]

        # Apply scenario changes to base state
        state = self._apply_changes(self.base_state, scenario_changes)

        # Simulate over time periods
        results = {
            'total_cost': 0,
            'service_level': [],
            'inventory': [],
            'stockouts': 0,
            'periods': []
        }

        inventory = state.inventory_levels.copy()

        for t in range(time_periods):
            period_demand = demand[t]

            # Check if can meet demand
            total_inventory = sum(inventory.values())

            if total_inventory >= period_demand:
                # Meet demand
                service = 1.0
                stockout = 0
                units_sold = period_demand
            else:
                # Stockout
                service = total_inventory / period_demand
                stockout = period_demand - total_inventory
                units_sold = total_inventory

            # Calculate costs
            holding_cost = sum(inventory.values()) * state.costs.get('holding', 1.0)
            stockout_cost = stockout * state.costs.get('stockout', 100.0)
            period_cost = holding_cost + stockout_cost

            # Production/replenishment
            production = min(
                period_demand * 1.2,  # Target 120% of demand
                sum(state.capacity.values())
            )

            # Apply supply reliability
            actual_production = production * scenario_changes.get('supply_reliability', 1.0)

            # Update inventory
            for loc in inventory:
                inventory[loc] = max(0, inventory[loc] - units_sold / len(inventory))
                inventory[loc] += actual_production / len(inventory)

            # Record results
            results['total_cost'] += period_cost
            results['service_level'].append(service)
            results['inventory'].append(sum(inventory.values()))
            results['stockouts'] += stockout
            results['periods'].append(t)

        # Calculate summary metrics
        results['avg_service_level'] = np.mean(results['service_level'])
        results['avg_inventory'] = np.mean(results['inventory'])
        results['total_stockouts'] = results['stockouts']

        return results

    def _apply_changes(self, base_state, changes):
        """Apply scenario changes to base state"""
        import copy
        state = copy.deepcopy(base_state)

        # Apply modifications based on scenario
        if 'capacity_reduction' in changes:
            for facility in state.capacity:
                state.capacity[facility] *= (1 - changes['capacity_reduction'])

        if 'cost_increase' in changes:
            for cost_type in state.costs:
                state.costs[cost_type] *= (1 + changes['cost_increase'])

        return state

    def compare_scenarios(self, demand: np.ndarray) -> pd.DataFrame:
        """Compare all scenarios"""

        results_list = []

        for scenario_name in self.scenarios:
            result = self.evaluate_scenario(scenario_name, demand)
            results_list.append({
                'Scenario': scenario_name,
                'Total_Cost': result['total_cost'],
                'Avg_Service_Level': result['avg_service_level'] * 100,
                'Avg_Inventory': result['avg_inventory'],
                'Total_Stockouts': result['total_stockouts']
            })

        return pd.DataFrame(results_list)

# Example usage
base = SupplyChainState(
    facilities=['DC_East', 'DC_West'],
    suppliers={'Supplier_A': 0.95, 'Supplier_B': 0.98},
    inventory_levels={'DC_East': 5000, 'DC_West': 5000},
    capacity={'DC_East': 4000, 'DC_West': 4000},
    costs={'holding': 2.0, 'stockout': 150.0}
)

model = ScenarioModel(base)

# Define scenarios
model.add_scenario('Base_Case', {
    'supply_reliability': 1.0,
    'capacity_reduction': 0.0,
    'cost_increase': 0.0
})

model.add_scenario('Supplier_Disruption', {
    'supply_reliability': 0.6,
    'capacity_reduction': 0.0,
    'cost_increase': 0.0
})

model.add_scenario('Capacity_Constraint', {
    'supply_reliability': 1.0,
    'capacity_reduction': 0.3,
    'cost_increase': 0.0
})

model.add_scenario('Cost_Inflation', {
    'supply_reliability': 1.0,
    'capacity_reduction': 0.0,
    'cost_increase': 0.25
})

model.add_scenario('Perfect_Storm', {
    'supply_reliability': 0.7,
    'capacity_reduction': 0.2,
    'cost_increase': 0.3
})

# Generate demand scenarios
np.random.seed(42)
demand = np.random.normal(8000, 1000, 12)  # 12 months

# Compare scenarios
comparison = model.compare_scenarios(demand)
print(comparison)
```

---

## Monte Carlo Simulation

### Probabilistic Scenario Analysis

**When to Use:**
- Multiple uncertain variables
- Need probability distributions, not point estimates
- Risk quantification required
- Portfolio or investment decisions

**Process:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class MonteCarloScenarioAnalysis:
    """Monte Carlo simulation for supply chain scenarios"""

    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations
        self.results = None

    def simulate_supply_chain(self,
                             demand_params: Dict,
                             cost_params: Dict,
                             supply_params: Dict,
                             periods: int = 12):
        """
        Run Monte Carlo simulation

        Parameters:
        - demand_params: {'mean': x, 'std': y, 'distribution': 'normal'}
        - cost_params: {'transport': {...}, 'holding': {...}}
        - supply_params: {'reliability': {...}}
        """

        results = []

        for sim in range(self.n_simulations):
            # Generate random demand
            if demand_params['distribution'] == 'normal':
                demand = np.random.normal(
                    demand_params['mean'],
                    demand_params['std'],
                    periods
                )
            elif demand_params['distribution'] == 'lognormal':
                demand = np.random.lognormal(
                    np.log(demand_params['mean']),
                    demand_params['std'],
                    periods
                )

            # Generate random costs
            transport_cost = np.random.uniform(
                cost_params['transport']['min'],
                cost_params['transport']['max']
            )

            holding_cost = np.random.normal(
                cost_params['holding']['mean'],
                cost_params['holding']['std']
            )

            # Generate supply reliability
            supply_reliability = np.random.beta(
                supply_params['reliability']['alpha'],
                supply_params['reliability']['beta']
            )

            # Simulate supply chain performance
            total_demand = demand.sum()
            actual_supply = total_demand * supply_reliability
            shortage = max(0, total_demand - actual_supply)

            # Calculate total cost
            transport_cost_total = actual_supply * transport_cost
            holding_cost_total = actual_supply * holding_cost * 0.1
            shortage_cost_total = shortage * 200  # Penalty

            total_cost = (transport_cost_total +
                         holding_cost_total +
                         shortage_cost_total)

            service_level = (actual_supply / total_demand) * 100

            results.append({
                'simulation': sim,
                'total_demand': total_demand,
                'actual_supply': actual_supply,
                'shortage': shortage,
                'transport_cost': transport_cost_total,
                'holding_cost': holding_cost_total,
                'shortage_cost': shortage_cost_total,
                'total_cost': total_cost,
                'service_level': service_level
            })

        self.results = pd.DataFrame(results)
        return self.results

    def analyze_results(self):
        """Statistical analysis of simulation results"""

        if self.results is None:
            raise ValueError("Run simulation first")

        analysis = {
            'total_cost': {
                'mean': self.results['total_cost'].mean(),
                'std': self.results['total_cost'].std(),
                'p10': self.results['total_cost'].quantile(0.10),
                'p50': self.results['total_cost'].quantile(0.50),
                'p90': self.results['total_cost'].quantile(0.90),
                'p95': self.results['total_cost'].quantile(0.95),
                'min': self.results['total_cost'].min(),
                'max': self.results['total_cost'].max()
            },
            'service_level': {
                'mean': self.results['service_level'].mean(),
                'std': self.results['service_level'].std(),
                'p10': self.results['service_level'].quantile(0.10),
                'p50': self.results['service_level'].quantile(0.50),
                'p90': self.results['service_level'].quantile(0.90)
            }
        }

        return analysis

    def value_at_risk(self, confidence_level=0.95):
        """Calculate Value at Risk (VaR) for costs"""

        if self.results is None:
            raise ValueError("Run simulation first")

        var = self.results['total_cost'].quantile(confidence_level)

        # Conditional VaR (CVaR) - expected loss beyond VaR
        cvar = self.results[
            self.results['total_cost'] >= var
        ]['total_cost'].mean()

        return {'VaR': var, 'CVaR': cvar}

    def plot_distributions(self):
        """Visualize simulation results"""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Total cost distribution
        axes[0, 0].hist(self.results['total_cost'], bins=50,
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Total Cost Distribution')
        axes[0, 0].set_xlabel('Total Cost ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.results['total_cost'].mean(),
                          color='r', linestyle='--', label='Mean')
        axes[0, 0].legend()

        # Service level distribution
        axes[0, 1].hist(self.results['service_level'], bins=50,
                       edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_title('Service Level Distribution')
        axes[0, 1].set_xlabel('Service Level (%)')
        axes[0, 1].set_ylabel('Frequency')

        # Shortage distribution
        axes[1, 0].hist(self.results['shortage'], bins=50,
                       edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_title('Shortage Distribution')
        axes[1, 0].set_xlabel('Shortage Units')
        axes[1, 0].set_ylabel('Frequency')

        # Cost components
        cost_components = self.results[[
            'transport_cost', 'holding_cost', 'shortage_cost'
        ]].mean()
        axes[1, 1].bar(cost_components.index, cost_components.values)
        axes[1, 1].set_title('Average Cost Components')
        axes[1, 1].set_ylabel('Cost ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

# Example usage
mc = MonteCarloScenarioAnalysis(n_simulations=10000)

demand_params = {
    'mean': 10000,
    'std': 2000,
    'distribution': 'normal'
}

cost_params = {
    'transport': {'min': 2.0, 'max': 4.0},
    'holding': {'mean': 1.5, 'std': 0.3}
}

supply_params = {
    'reliability': {'alpha': 9, 'beta': 1}  # Beta distribution
}

results = mc.simulate_supply_chain(
    demand_params, cost_params, supply_params, periods=12
)

# Analyze
analysis = mc.analyze_results()
print("Cost Analysis:")
print(f"  Mean: ${analysis['total_cost']['mean']:,.0f}")
print(f"  Std Dev: ${analysis['total_cost']['std']:,.0f}")
print(f"  P95: ${analysis['total_cost']['p95']:,.0f}")

var_metrics = mc.value_at_risk(0.95)
print(f"\nValue at Risk (95%): ${var_metrics['VaR']:,.0f}")
print(f"Conditional VaR: ${var_metrics['CVaR']:,.0f}")

# Plot
mc.plot_distributions()
```

---

## Sensitivity Analysis

### One-Way Sensitivity

**Test single variable impact:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sensitivity_analysis(base_model, variable_name,
                        test_range, other_params):
    """
    Perform one-way sensitivity analysis

    Parameters:
    - base_model: function that returns metric given parameters
    - variable_name: name of variable to test
    - test_range: array of values to test
    - other_params: dict of fixed parameters
    """

    results = []

    for value in test_range:
        # Update parameter
        params = other_params.copy()
        params[variable_name] = value

        # Evaluate model
        metric = base_model(**params)

        results.append({
            variable_name: value,
            'metric': metric
        })

    return pd.DataFrame(results)

# Example: Test demand sensitivity
def calculate_cost(demand, unit_cost, holding_rate, capacity):
    """Simple cost model"""
    production = min(demand, capacity)
    shortage = max(0, demand - capacity)

    production_cost = production * unit_cost
    holding_cost = production * holding_rate * 0.5
    shortage_cost = shortage * unit_cost * 3  # 3x penalty

    return production_cost + holding_cost + shortage_cost

# Test demand scenarios
demand_range = np.linspace(5000, 15000, 20)

other_params = {
    'unit_cost': 10,
    'holding_rate': 0.2,
    'capacity': 10000
}

demand_sensitivity = sensitivity_analysis(
    calculate_cost,
    'demand',
    demand_range,
    other_params
)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(demand_sensitivity['demand'],
         demand_sensitivity['metric'],
         marker='o', linewidth=2)
plt.xlabel('Demand (units)')
plt.ylabel('Total Cost ($)')
plt.title('Sensitivity to Demand Changes')
plt.grid(True, alpha=0.3)
plt.axvline(x=10000, color='r', linestyle='--', label='Capacity Limit')
plt.legend()
plt.show()
```

### Multi-Way Sensitivity (Tornado Diagram)

**Compare impact of multiple variables:**

```python
def tornado_analysis(base_model, variables, ranges, base_params):
    """
    Create tornado diagram showing sensitivity to multiple variables

    Parameters:
    - base_model: function to evaluate
    - variables: list of variable names
    - ranges: dict {variable: (low, high)}
    - base_params: base case parameters
    """

    # Calculate base case
    base_result = base_model(**base_params)

    results = []

    for var in variables:
        # Test low value
        params_low = base_params.copy()
        params_low[var] = ranges[var][0]
        result_low = base_model(**params_low)

        # Test high value
        params_high = base_params.copy()
        params_high[var] = ranges[var][1]
        result_high = base_model(**params_high)

        # Calculate swing
        swing = abs(result_high - result_low)

        results.append({
            'variable': var,
            'low_value': ranges[var][0],
            'high_value': ranges[var][1],
            'result_low': result_low,
            'result_high': result_high,
            'swing': swing,
            'impact_pct': (swing / base_result) * 100
        })

    df = pd.DataFrame(results)
    df = df.sort_values('swing', ascending=True)

    # Plot tornado diagram
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(df))

    for i, row in df.iterrows():
        low = row['result_low']
        high = row['result_high']

        ax.barh(row['variable'], high - base_result,
                left=base_result, color='red', alpha=0.6)
        ax.barh(row['variable'], base_result - low,
                left=low, color='blue', alpha=0.6)

    ax.axvline(x=base_result, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Total Cost ($)')
    ax.set_title('Tornado Diagram - Sensitivity Analysis')
    ax.grid(True, alpha=0.3, axis='x')

    return df, fig

# Example usage
variables = ['demand', 'unit_cost', 'holding_rate']

ranges = {
    'demand': (8000, 12000),
    'unit_cost': (8, 12),
    'holding_rate': (0.15, 0.25)
}

base_params = {
    'demand': 10000,
    'unit_cost': 10,
    'holding_rate': 0.2,
    'capacity': 10000
}

tornado_df, fig = tornado_analysis(
    calculate_cost, variables, ranges, base_params
)
```

---

## Risk Assessment & Quantification

### Risk Probability-Impact Matrix

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RiskAssessment:
    """Supply chain risk assessment framework"""

    def __init__(self):
        self.risks = []

    def add_risk(self, name, category, probability, impact,
                detection_difficulty, mitigation_cost):
        """
        Add a risk to the register

        Parameters:
        - probability: 1-5 scale (1=very low, 5=very high)
        - impact: 1-5 scale (1=minimal, 5=catastrophic)
        - detection_difficulty: 1-5 (1=easy to detect, 5=hard)
        - mitigation_cost: 1-5 (1=low cost, 5=high cost)
        """

        risk = {
            'name': name,
            'category': category,
            'probability': probability,
            'impact': impact,
            'detection': detection_difficulty,
            'mitigation_cost': mitigation_cost,
            'risk_score': probability * impact,
            'risk_priority_number': probability * impact * detection_difficulty
        }

        self.risks.append(risk)

    def get_risk_register(self):
        """Return risk register as DataFrame"""
        df = pd.DataFrame(self.risks)
        df = df.sort_values('risk_score', ascending=False)
        return df

    def plot_risk_matrix(self):
        """Plot probability-impact matrix"""

        df = pd.DataFrame(self.risks)

        fig, ax = plt.subplots(figsize=(12, 10))

        # Color by risk score
        scatter = ax.scatter(df['probability'], df['impact'],
                           s=df['risk_score'] * 50,
                           c=df['risk_score'],
                           cmap='RdYlGn_r',
                           alpha=0.6,
                           edgecolors='black',
                           linewidth=1.5)

        # Annotate risks
        for idx, row in df.iterrows():
            ax.annotate(row['name'],
                       (row['probability'], row['impact']),
                       fontsize=8,
                       xytext=(5, 5),
                       textcoords='offset points')

        ax.set_xlabel('Probability (1-5)', fontsize=12)
        ax.set_ylabel('Impact (1-5)', fontsize=12)
        ax.set_title('Supply Chain Risk Matrix', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add risk zones
        ax.axhline(y=3, color='orange', linestyle='--', alpha=0.3, linewidth=2)
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.3, linewidth=2)

        # Add text labels for zones
        ax.text(1.2, 4.5, 'Low Prob\nHigh Impact', fontsize=9, alpha=0.5)
        ax.text(4.2, 4.5, 'HIGH RISK', fontsize=11, fontweight='bold',
               color='red', alpha=0.7)
        ax.text(4.2, 1.5, 'High Prob\nLow Impact', fontsize=9, alpha=0.5)
        ax.text(1.2, 1.5, 'LOW RISK', fontsize=10, alpha=0.5)

        plt.colorbar(scatter, label='Risk Score')

        return fig

    def prioritize_risks(self, min_score=10):
        """Return high-priority risks"""
        df = self.get_risk_register()
        return df[df['risk_score'] >= min_score]

# Example usage
risk_assessment = RiskAssessment()

# Add supply chain risks
risk_assessment.add_risk(
    name='Single Source Supplier',
    category='Supply',
    probability=3,
    impact=5,
    detection_difficulty=2,
    mitigation_cost=4
)

risk_assessment.add_risk(
    name='Port Congestion',
    category='Logistics',
    probability=4,
    impact=3,
    detection_difficulty=1,
    mitigation_cost=2
)

risk_assessment.add_risk(
    name='Demand Forecast Error',
    category='Demand',
    probability=5,
    impact=3,
    detection_difficulty=2,
    mitigation_cost=3
)

risk_assessment.add_risk(
    name='Natural Disaster',
    category='External',
    probability=2,
    impact=5,
    detection_difficulty=1,
    mitigation_cost=5
)

risk_assessment.add_risk(
    name='Cyber Attack',
    category='Technology',
    probability=3,
    impact=4,
    detection_difficulty=4,
    mitigation_cost=4
)

risk_assessment.add_risk(
    name='Quality Defect',
    category='Operations',
    probability=4,
    impact=4,
    detection_difficulty=3,
    mitigation_cost=3
)

# Get risk register
risk_register = risk_assessment.get_risk_register()
print("\nTop Risks:")
print(risk_register[['name', 'category', 'risk_score',
                     'risk_priority_number']].head(10))

# Plot
risk_assessment.plot_risk_matrix()

# High priority risks
high_risks = risk_assessment.prioritize_risks(min_score=12)
print(f"\nHigh Priority Risks (score >= 12): {len(high_risks)}")
```

---

## Strategic Scenario Development

### Shell Scenario Planning Method

**For Long-Term Strategic Planning:**

**Step 1: Define Focal Question**
- "What supply chain strategy will be most resilient over next 5 years?"

**Step 2: Identify Key Forces**
- Predetermined elements (trends, demographics)
- Critical uncertainties (regulations, technology adoption)

**Step 3: Select Scenario Dimensions**
- Choose 2 most important/uncertain drivers
- Create 2x2 matrix

**Step 4: Develop Scenario Narratives**
- Create compelling stories for each quadrant
- Name scenarios descriptively
- Detail implications

**Step 5: Identify Implications & Strategies**
- What strategies work in all scenarios? (robust)
- What strategies are scenario-dependent? (contingent)
- Early warning indicators?

**Example: Global Trade Scenarios**

**Dimensions:**
- Economic Integration (High/Low)
- Environmental Regulation (Strict/Relaxed)

**Four Scenarios:**

1. **"Global Efficiency"** (High Integration, Relaxed Regulations)
   - Optimized global networks
   - Lowest cost production
   - Long supply chains
   - Limited regionalization

2. **"Green Global"** (High Integration, Strict Regulations)
   - Carbon-neutral logistics
   - Sustainable sourcing priority
   - Green technology investments
   - Higher costs, strong compliance

3. **"Regional Resilience"** (Low Integration, Strict Regulations)
   - Near-shoring/friend-shoring
   - Regional supply chains
   - Emphasis on reliability
   - Moderate costs

4. **"Fragmented World"** (Low Integration, Relaxed Regulations)
   - Trade barriers
   - Country-specific strategies
   - Duplicated infrastructure
   - High inefficiency

---

## Contingency Planning

### Business Continuity Plans

**Key Components:**

1. **Risk Identification**
   - What can go wrong?
   - Likelihood and impact

2. **Impact Analysis**
   - Revenue impact
   - Customer impact
   - Recovery time objectives (RTO)
   - Recovery point objectives (RPO)

3. **Response Strategies**
   - Immediate actions (0-24 hours)
   - Short-term (1-7 days)
   - Medium-term (1-4 weeks)
   - Long-term recovery

4. **Resource Requirements**
   - People, systems, facilities
   - Backup suppliers
   - Inventory buffers
   - Financial reserves

**Contingency Plan Template:**

```python
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class ContingencyAction:
    """Single action in contingency plan"""
    action_id: str
    description: str
    trigger: str
    responsible_party: str
    timeline: str  # "Immediate", "24 hours", "Week 1", etc.
    resources_needed: List[str]
    cost_estimate: float
    dependencies: List[str]

@dataclass
class ContingencyPlan:
    """Complete contingency plan for a risk scenario"""
    risk_name: str
    risk_category: str
    trigger_conditions: List[str]
    severity_level: str  # "Low", "Medium", "High", "Critical"
    actions: List[ContingencyAction]
    communication_plan: Dict
    escalation_procedure: List[str]

    def get_immediate_actions(self):
        """Return actions needed immediately"""
        return [a for a in self.actions if a.timeline == "Immediate"]

    def get_actions_by_timeline(self, timeline: str):
        """Get actions for specific timeline"""
        return [a for a in self.actions if a.timeline == timeline]

    def estimate_total_cost(self):
        """Calculate total cost of plan execution"""
        return sum(a.cost_estimate for a in self.actions)

# Example: Supplier Disruption Contingency Plan
supplier_disruption_plan = ContingencyPlan(
    risk_name="Primary Supplier Disruption",
    risk_category="Supply Risk",
    trigger_conditions=[
        "Supplier bankruptcy filing",
        "Natural disaster at supplier location",
        "Quality issues requiring supplier shutdown",
        "Supply < 50% of normal for > 3 days"
    ],
    severity_level="High",
    actions=[
        ContingencyAction(
            action_id="SD-001",
            description="Activate backup supplier contracts",
            trigger="Confirmed disruption > 24 hours",
            responsible_party="Procurement Director",
            timeline="Immediate",
            resources_needed=["Backup supplier contact list", "Expedited PO"],
            cost_estimate=50000,
            dependencies=[]
        ),
        ContingencyAction(
            action_id="SD-002",
            description="Increase safety stock from alternative sources",
            trigger="Expected disruption > 1 week",
            responsible_party="Supply Chain Manager",
            timeline="24 hours",
            resources_needed=["Emergency budget authorization", "Warehouse space"],
            cost_estimate=200000,
            dependencies=["SD-001"]
        ),
        ContingencyAction(
            action_id="SD-003",
            description="Implement product rationing to priority customers",
            trigger="Inventory below critical level",
            responsible_party="Customer Service VP",
            timeline="48 hours",
            resources_needed=["Customer priority list", "Communication templates"],
            cost_estimate=0,
            dependencies=["SD-002"]
        ),
        ContingencyAction(
            action_id="SD-004",
            description="Qualify and onboard new supplier",
            trigger="Expected disruption > 30 days",
            responsible_party="Procurement Director",
            timeline="Week 2-4",
            resources_needed=["Supplier qualification team", "Quality testing"],
            cost_estimate=150000,
            dependencies=["SD-001", "SD-002"]
        )
    ],
    communication_plan={
        "internal": ["Executive team notification within 2 hours",
                    "Daily status updates to stakeholders"],
        "external": ["Customer notification within 24 hours if impact expected",
                    "Supplier engagement for resolution timeline"],
        "escalation": ["CFO if cost > $500K", "CEO if duration > 4 weeks"]
    },
    escalation_procedure=[
        "Level 1: Supply Chain Manager (0-24 hours)",
        "Level 2: VP Operations (24-72 hours)",
        "Level 3: COO (> 72 hours or impact > $1M)",
        "Level 4: CEO (Critical customer impact or > $5M)"
    ]
)

# Display immediate actions
print("IMMEDIATE ACTIONS:")
for action in supplier_disruption_plan.get_immediate_actions():
    print(f"  [{action.action_id}] {action.description}")
    print(f"      Owner: {action.responsible_party}")
    print(f"      Cost: ${action.cost_estimate:,}")

print(f"\nTotal Plan Cost: ${supplier_disruption_plan.estimate_total_cost():,}")
```

---

## Tools & Libraries

### Python Libraries

**Simulation & Modeling:**
- `simpy`: Discrete-event simulation
- `scipy.stats`: Statistical distributions
- `numpy`: Random number generation, arrays
- `pandas`: Data manipulation and analysis

**Optimization Under Uncertainty:**
- `pyomo`: Stochastic programming
- `pulp`: Linear programming with scenarios
- `mip`: Mixed-integer programming

**Visualization:**
- `matplotlib`, `seaborn`: Statistical plots
- `plotly`: Interactive dashboards
- `networkx`: Network diagrams

**Risk Analysis:**
- `risktools`: Risk metrics and VaR
- `copulas`: Dependency modeling

### Commercial Software

**Scenario Planning Platforms:**
- **Kinaxis RapidResponse**: Scenario analysis and collaboration
- **o9 Solutions**: AI-driven scenario planning
- **LLamasoft**: Supply chain scenario modeling
- **Blue Yonder**: What-if analysis

**Risk Management:**
- **Resilinc**: Supply chain risk intelligence
- **Everstream Analytics**: Predictive risk analytics
- **Riskmethods**: Risk monitoring and assessment

**Simulation:**
- **AnyLogic**: Multi-method simulation
- **Arena**: Discrete-event simulation
- **Simio**: 3D simulation modeling
- **FlexSim**: 3D simulation

---

## Common Challenges & Solutions

### Challenge: Too Many Scenarios

**Problem:**
- Analysis paralysis
- Overwhelming number of combinations
- Can't make decisions

**Solutions:**
- Focus on 3-5 key scenarios
- Use scenario reduction techniques
- Group similar scenarios
- Prioritize by probability × impact
- Start with base/best/worst cases

### Challenge: Lack of Data

**Problem:**
- No historical disruption data
- Difficult to estimate probabilities
- Uncertainty about impacts

**Solutions:**
- Use expert judgment (Delphi method)
- Industry benchmarks and case studies
- Start with ranges instead of points
- Sensitivity analysis to understand drivers
- Learn and update as events occur

### Challenge: Scenario Bias

**Problem:**
- Anchoring to current state
- Ignoring low-probability high-impact events
- Confirmation bias in scenario selection

**Solutions:**
- Include diverse perspectives
- Use structured methods (Shell approach)
- Challenge assumptions explicitly
- Include "black swan" scenarios
- Independent facilitation

### Challenge: Actionability

**Problem:**
- Scenarios don't lead to decisions
- Analysis doesn't translate to strategy
- Lack of clear next steps

**Solutions:**
- Link scenarios to specific decisions
- Identify "no regret" moves (work in all scenarios)
- Define trigger points and early warnings
- Create contingency plans for each scenario
- Regular review and update process

### Challenge: Complexity vs. Usability

**Problem:**
- Models too complex for stakeholders
- Can't explain results
- Black box simulations

**Solutions:**
- Start simple, add complexity as needed
- Visual dashboards and summaries
- Clear documentation of assumptions
- Sensitivity analysis to show key drivers
- Scenario narratives, not just numbers

---

## Output Format

### Scenario Analysis Report

**Executive Summary:**
- Purpose and scope of analysis
- Key scenarios evaluated
- Critical findings and recommendations
- Decision implications

**Scenario Definitions:**

| Scenario Name | Description | Key Assumptions | Probability |
|--------------|-------------|-----------------|-------------|
| Base Case | Most likely future | Current trends continue | 50% |
| High Growth | Demand surge | Economic boom, new markets | 20% |
| Supply Disruption | Major supplier failure | Single-source risk realizes | 15% |
| Perfect Storm | Multiple issues | Demand spike + supply shortage | 10% |
| Cost Inflation | Rising costs | Fuel, labor, materials up 30% | 25% |

**Scenario Results:**

| Scenario | Total Cost | Service Level | Inventory | Key Metrics |
|----------|-----------|---------------|-----------|-------------|
| Base Case | $25M | 95% | 8,000 units | Baseline |
| High Growth | $32M | 87% | 12,000 units | Capacity constrained |
| Supply Disruption | $38M | 78% | 5,000 units | High stockout costs |
| Perfect Storm | $45M | 65% | 7,000 units | Crisis mode |
| Cost Inflation | $31M | 93% | 8,000 units | Margin pressure |

**Risk Assessment:**
- High-priority risks identified
- Risk mitigation strategies
- Contingency plans for top scenarios
- Early warning indicators

**Recommendations:**

1. **Immediate Actions** (0-3 months)
   - Qualify backup suppliers for top 10 components
   - Increase safety stock for critical items
   - Implement demand sensing for early warning

2. **Short-Term Initiatives** (3-12 months)
   - Diversify supplier base
   - Add flexible capacity options
   - Enhance scenario planning capability

3. **Strategic Investments** (1-3 years)
   - Build regional distribution center
   - Implement control tower technology
   - Develop dual-sourcing strategy

**Monitoring & Review:**
- Key performance indicators to track
- Scenario review frequency (quarterly)
- Trigger points for plan activation
- Governance and escalation process

---

## Questions to Ask

If you need more context:
1. What decision is this scenario analysis supporting?
2. What uncertainties or risks are most concerning?
3. What's the planning horizon? (tactical vs. strategic)
4. What data is available on historical variability and disruptions?
5. Are there specific events or scenarios to model?
6. Quantitative analysis or qualitative scenarios needed?
7. Who will use the analysis and how?
8. Existing risk management or continuity plans?

---

## Related Skills

- **demand-forecasting**: For demand uncertainty analysis
- **sales-operations-planning**: For S&OP scenario integration
- **capacity-planning**: For capacity scenarios and stress testing
- **network-design**: For network strategy scenarios
- **inventory-optimization**: For inventory buffer strategies
- **risk-mitigation**: For risk response planning
- **supply-chain-analytics**: For performance monitoring
- **digital-twin-modeling**: For real-time scenario simulation

---
name: seasonal-planning
description: When the user wants to optimize seasonal planning, manage seasonal buy decisions, or plan for seasonal demand. Also use when the user mentions "seasonal planning," "seasonal buy," "holiday planning," "back-to-school," "spring/fall collection," "seasonal inventory," "peak season," or "seasonal assortment." For demand forecasting, see demand-forecasting. For retail allocation, see retail-allocation.
---

# Seasonal Planning

You are an expert in retail seasonal planning and merchandise buying. Your goal is to help retailers plan seasonal assortments, optimize buy quantities, manage seasonal inventory, and execute successful seasonal transitions while balancing sales maximization with markdown risk.

## Initial Assessment

Before planning seasonal buys, understand:

1. **Business Context**
   - What retail category? (apparel, home, toys, etc.)
   - What season? (spring, summer, fall, holiday, back-to-school)
   - Season length? (weeks of selling season)
   - Historical seasonal performance? (sales, sell-through, markdowns)

2. **Financial Targets**
   - Season sales target? (revenue goal)
   - Target gross margin? (initial markup, markdown budget)
   - Inventory turn goals?
   - Open-to-buy budget?
   - Cash flow constraints?

3. **Product Mix**
   - Carry-over vs. new products? (% of each)
   - Core basics vs. fashion/trend items?
   - Price point distribution? (good/better/best)
   - SKU count target?
   - Vendor/supplier lead times?

4. **Historical Data Available**
   - Past season sales by week?
   - Sell-through rates by category/style?
   - Markdown rates and timing?
   - Stockout frequency?
   - Weather impacts?

---

## Seasonal Planning Framework

### Season Phases

**Pre-Season (Weeks -12 to 0)**
- Trend forecasting and market research
- Assortment planning (styles, colors, sizes)
- Buy planning and vendor negotiations
- Allocation planning
- Marketing campaign planning

**Early Season (Weeks 1-4)**
- Initial receipts and allocation
- Monitor early sell-through
- Identify fast/slow movers
- Adjust future orders (if possible)
- Replenishment decisions

**Peak Season (Weeks 5-8)**
- Peak sales volume
- Maintain in-stock on winners
- Begin markdown planning for slow movers
- Chase orders for hot items
- Maximize full-price selling

**Late Season (Weeks 9-12)**
- Aggressive markdowns to clear
- Minimize leftover inventory
- Transition space to next season
- Pack-away vs. liquidation decisions
- Post-season analysis

---

## Buy Planning & Optimization

### Seasonal Buy Quantity Optimization

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

class SeasonalBuyOptimizer:
    """
    Optimize seasonal buy quantities

    Balance:
    - Under-buying: Lost sales (stockouts)
    - Over-buying: Markdowns and excess inventory
    """

    def __init__(self, season_config):
        """
        Parameters:
        - season_config: Season parameters (length, targets, costs)
        """
        self.season = season_config

    def calculate_optimal_buy(self, sku_forecast, unit_cost, retail_price,
                             markdown_rate=0.50, stockout_cost_multiplier=1.5):
        """
        Calculate optimal buy quantity using newsvendor model

        Classic single-period inventory problem
        """

        # Expected demand
        mean_demand = sku_forecast['mean']
        std_demand = sku_forecast['std']

        # Profit margins
        full_price_margin = retail_price - unit_cost
        markdown_price = retail_price * (1 - markdown_rate)
        markdown_margin = markdown_price - unit_cost

        # Cost of under-stocking (lost profit + goodwill)
        cost_understocking = full_price_margin * stockout_cost_multiplier

        # Cost of over-stocking (forced markdown or liquidation)
        cost_overstocking = unit_cost - markdown_price

        # Critical ratio (newsvendor)
        critical_ratio = cost_understocking / (cost_understocking + cost_overstocking)

        # Optimal order quantity (quantile of demand distribution)
        optimal_quantity = stats.norm.ppf(critical_ratio, mean_demand, std_demand)

        # Calculate expected profit at optimal quantity
        expected_sales = self._expected_sales(optimal_quantity, mean_demand, std_demand)
        expected_markdowns = max(0, optimal_quantity - expected_sales)

        expected_revenue = (expected_sales * retail_price +
                           expected_markdowns * markdown_price)
        expected_cost = optimal_quantity * unit_cost
        expected_profit = expected_revenue - expected_cost

        # Service level (fill rate)
        service_level = stats.norm.cdf(optimal_quantity, mean_demand, std_demand)

        return {
            'optimal_buy_quantity': round(optimal_quantity, 0),
            'expected_demand': mean_demand,
            'demand_std': std_demand,
            'expected_sales': round(expected_sales, 0),
            'expected_markdowns': round(expected_markdowns, 0),
            'expected_profit': round(expected_profit, 2),
            'service_level': round(service_level * 100, 1),
            'markdown_rate': markdown_rate * 100,
            'critical_ratio': round(critical_ratio, 3)
        }

    def _expected_sales(self, quantity, mean, std):
        """
        Calculate expected sales given quantity

        Accounts for potential stockouts
        """

        # E[Sales] = E[min(Demand, Quantity)]
        # Using normal distribution approximation

        if std == 0:
            return min(quantity, mean)

        z = (quantity - mean) / std
        expected_sales = mean * stats.norm.cdf(z) + std * stats.norm.pdf(z)

        return min(expected_sales, quantity)

    def optimize_assortment_mix(self, product_options, total_budget,
                                category_constraints=None):
        """
        Optimize product mix within budget

        Select which products to buy and in what quantities
        """

        n_products = len(product_options)

        # Objective: Maximize total expected profit
        def objective(quantities):
            total_profit = 0

            for i, qty in enumerate(quantities):
                product = product_options.iloc[i]

                # Calculate profit for this quantity
                result = self.calculate_optimal_buy(
                    sku_forecast={'mean': product['forecast_mean'],
                                 'std': product['forecast_std']},
                    unit_cost=product['unit_cost'],
                    retail_price=product['retail_price']
                )

                # Adjust for actual quantity vs. optimal
                if qty > 0:
                    # Approximate profit at this quantity
                    profit_at_qty = result['expected_profit'] * (qty / result['optimal_buy_quantity'])
                    total_profit += profit_at_qty

            return -total_profit  # Negative for minimization

        # Constraints
        def budget_constraint(quantities):
            total_cost = sum(
                quantities[i] * product_options.iloc[i]['unit_cost']
                for i in range(n_products)
            )
            return total_budget - total_cost

        constraints = [{'type': 'ineq', 'fun': budget_constraint}]

        # Bounds (non-negative quantities)
        bounds = [(0, product_options.iloc[i]['max_quantity']) for i in range(n_products)]

        # Initial guess (proportional to forecast)
        x0 = np.array([
            min(product_options.iloc[i]['forecast_mean'],
                product_options.iloc[i]['max_quantity'])
            for i in range(n_products)
        ]) * 0.8  # Start conservative

        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        optimal_quantities = result.x

        # Build result dataframe
        results = []
        for i, qty in enumerate(optimal_quantities):
            product = product_options.iloc[i]

            if qty > 5:  # Only include products with meaningful quantity
                buy_analysis = self.calculate_optimal_buy(
                    sku_forecast={'mean': product['forecast_mean'],
                                 'std': product['forecast_std']},
                    unit_cost=product['unit_cost'],
                    retail_price=product['retail_price']
                )

                results.append({
                    'sku': product['sku'],
                    'category': product['category'],
                    'buy_quantity': round(qty, 0),
                    'unit_cost': product['unit_cost'],
                    'retail_price': product['retail_price'],
                    'total_cost': round(qty * product['unit_cost'], 2),
                    'expected_profit': buy_analysis['expected_profit'],
                    'expected_markdown_rate': buy_analysis['markdown_rate']
                })

        results_df = pd.DataFrame(results)

        return results_df, result

    def calculate_open_to_buy(self, sales_plan, beginning_inventory,
                              on_order, target_end_inventory,
                              markdown_receipts=0):
        """
        Calculate open-to-buy (OTB) budget

        OTB = Sales Plan + Target End Inv - Beginning Inv - On Order + Markdowns
        """

        otb = (
            sales_plan +
            target_end_inventory -
            beginning_inventory -
            on_order +
            markdown_receipts
        )

        return {
            'sales_plan': sales_plan,
            'beginning_inventory': beginning_inventory,
            'on_order': on_order,
            'target_end_inventory': target_end_inventory,
            'markdown_receipts': markdown_receipts,
            'open_to_buy': otb,
            'otb_pct_of_sales': (otb / sales_plan * 100) if sales_plan > 0 else 0
        }

# Example usage
season_config = {
    'season_name': 'Fall 2024',
    'start_date': '2024-08-01',
    'end_date': '2024-11-30',
    'weeks': 16
}

optimizer = SeasonalBuyOptimizer(season_config)

# Single SKU optimization
sku_forecast = {'mean': 500, 'std': 150}
buy_decision = optimizer.calculate_optimal_buy(
    sku_forecast=sku_forecast,
    unit_cost=25,
    retail_price=60,
    markdown_rate=0.50
)

print("Optimal Buy Analysis:")
print(f"  Optimal quantity: {buy_decision['optimal_buy_quantity']}")
print(f"  Expected sales: {buy_decision['expected_sales']}")
print(f"  Expected markdowns: {buy_decision['expected_markdowns']}")
print(f"  Expected profit: ${buy_decision['expected_profit']:,.2f}")
print(f"  Service level: {buy_decision['service_level']:.1f}%")

# Assortment optimization
product_options = pd.DataFrame({
    'sku': [f'SKU{i:03d}' for i in range(1, 21)],
    'category': np.random.choice(['Tops', 'Bottoms', 'Dresses'], 20),
    'forecast_mean': np.random.uniform(200, 800, 20),
    'forecast_std': np.random.uniform(50, 200, 20),
    'unit_cost': np.random.uniform(15, 40, 20),
    'retail_price': np.random.uniform(40, 100, 20),
    'max_quantity': 1000
})

assortment, optimization_result = optimizer.optimize_assortment_mix(
    product_options,
    total_budget=150000
)

print(f"\nOptimized Assortment (Budget: $150K):")
print(f"Products selected: {len(assortment)}")
print(f"Total cost: ${assortment['total_cost'].sum():,.0f}")
print(f"Expected total profit: ${assortment['expected_profit'].sum():,.0f}")
```

---

## Seasonal Forecasting

### Seasonal Demand Modeling

```python
class SeasonalDemandForecaster:
    """
    Forecast seasonal demand patterns

    Accounts for:
    - Historical seasonal trends
    - Year-over-year growth
    - Fashion trends and newness
    - Weather impacts
    """

    def __init__(self, historical_data):
        """
        Parameters:
        - historical_data: Historical sales by week/season
          columns: ['season', 'year', 'week', 'sales', 'category']
        """
        self.history = historical_data

    def forecast_seasonal_curve(self, season, category):
        """
        Create seasonal sales curve

        Shows expected % of season sales by week
        """

        # Get historical data for this season
        season_history = self.history[
            (self.history['season'] == season) &
            (self.history['category'] == category)
        ]

        if len(season_history) == 0:
            # Use generic curve
            return self._generic_seasonal_curve()

        # Average sales by week across years
        weekly_avg = season_history.groupby('week')['sales'].mean()
        total_season_sales = weekly_avg.sum()

        # Calculate % distribution
        weekly_pct = (weekly_avg / total_season_sales * 100).to_dict()

        # Smooth the curve
        weeks = sorted(weekly_pct.keys())
        smoothed_pct = {}

        for week in weeks:
            # 3-week moving average
            nearby_weeks = [w for w in weeks if abs(w - week) <= 1]
            smoothed_pct[week] = np.mean([weekly_pct[w] for w in nearby_weeks])

        return smoothed_pct

    def _generic_seasonal_curve(self):
        """Generic seasonal curve (normal distribution)"""

        weeks = range(1, 17)  # 16-week season
        peak_week = 6  # Peak in week 6

        curve = {}
        total = 0

        for week in weeks:
            # Normal distribution centered at peak
            sales = np.exp(-((week - peak_week) ** 2) / 20)
            curve[week] = sales
            total += sales

        # Convert to percentages
        for week in weeks:
            curve[week] = curve[week] / total * 100

        return curve

    def forecast_total_season_sales(self, season, category, last_year_sales,
                                    growth_rate=0.05, trend_factor=1.0):
        """
        Forecast total season sales

        Based on:
        - Last year performance
        - Expected growth rate
        - Category trends
        """

        # Base forecast: last year + growth
        base_forecast = last_year_sales * (1 + growth_rate)

        # Adjust for trends
        adjusted_forecast = base_forecast * trend_factor

        return {
            'season': season,
            'category': category,
            'last_year_sales': last_year_sales,
            'growth_rate': growth_rate * 100,
            'trend_factor': trend_factor,
            'forecasted_sales': adjusted_forecast
        }

    def allocate_forecast_to_skus(self, total_forecast, sku_mix):
        """
        Allocate total forecast to individual SKUs

        Based on:
        - Historical performance (for carry-overs)
        - Analogous products (for new items)
        - Price point distribution
        """

        sku_forecasts = []

        for idx, sku in sku_mix.iterrows():
            if sku['is_new']:
                # New item: use analog performance
                forecast_pct = sku['analog_sales_pct']
            else:
                # Carry-over: use historical
                forecast_pct = sku['historical_sales_pct']

            # Adjust for price point appeal
            price_adjustment = sku.get('price_elasticity', 1.0)

            sku_forecast = total_forecast * (forecast_pct / 100) * price_adjustment

            # Add uncertainty (standard deviation)
            sku_std = sku_forecast * 0.30  # 30% coefficient of variation

            sku_forecasts.append({
                'sku': sku['sku'],
                'forecast_mean': sku_forecast,
                'forecast_std': sku_std,
                'is_new': sku['is_new'],
                'confidence': 'Low' if sku['is_new'] else 'High'
            })

        return pd.DataFrame(sku_forecasts)

    def simulate_season(self, initial_inventory, seasonal_curve,
                       total_forecast, n_simulations=1000):
        """
        Monte Carlo simulation of season performance

        Accounts for demand uncertainty
        """

        results = []

        for sim in range(n_simulations):
            # Simulate demand with uncertainty
            weekly_demand_pct = seasonal_curve.copy()

            # Add random variation
            for week in weekly_demand_pct.keys():
                noise = np.random.normal(1.0, 0.15)  # 15% noise
                weekly_demand_pct[week] *= noise

            # Normalize back to 100%
            total_pct = sum(weekly_demand_pct.values())
            weekly_demand_pct = {k: v/total_pct*100 for k, v in weekly_demand_pct.items()}

            # Simulate season
            inventory = initial_inventory
            total_sales = 0
            total_stockouts = 0

            for week, pct in sorted(weekly_demand_pct.items()):
                weekly_demand = total_forecast * (pct / 100)

                # Sales limited by inventory
                weekly_sales = min(weekly_demand, inventory)
                stockout = max(0, weekly_demand - inventory)

                inventory -= weekly_sales
                total_sales += weekly_sales
                total_stockouts += stockout

            # Calculate metrics
            sell_through_rate = (total_sales / initial_inventory * 100) if initial_inventory > 0 else 0
            stockout_rate = (total_stockouts / total_forecast * 100) if total_forecast > 0 else 0
            leftover_inventory = inventory

            results.append({
                'simulation': sim,
                'total_sales': total_sales,
                'sell_through_rate': sell_through_rate,
                'stockout_rate': stockout_rate,
                'leftover_inventory': leftover_inventory
            })

        results_df = pd.DataFrame(results)

        # Summary statistics
        summary = {
            'mean_sell_through': results_df['sell_through_rate'].mean(),
            'p10_sell_through': results_df['sell_through_rate'].quantile(0.10),
            'p50_sell_through': results_df['sell_through_rate'].quantile(0.50),
            'p90_sell_through': results_df['sell_through_rate'].quantile(0.90),
            'mean_stockout_rate': results_df['stockout_rate'].mean(),
            'mean_leftover': results_df['leftover_inventory'].mean()
        }

        return results_df, summary

# Example
historical_data = pd.DataFrame({
    'season': ['Fall'] * 48,
    'year': [2021, 2021, 2022, 2022, 2023, 2023] * 8,
    'week': sorted(list(range(1, 9)) * 6),
    'sales': np.random.uniform(5000, 15000, 48),
    'category': 'Sweaters'
})

forecaster = SeasonalDemandForecaster(historical_data)

# Get seasonal curve
curve = forecaster.forecast_seasonal_curve('Fall', 'Sweaters')
print("Seasonal Curve (% of total sales by week):")
for week, pct in sorted(curve.items())[:8]:
    print(f"  Week {week}: {pct:.1f}%")

# Forecast total season
season_forecast = forecaster.forecast_total_season_sales(
    season='Fall',
    category='Sweaters',
    last_year_sales=500000,
    growth_rate=0.08,
    trend_factor=1.1
)
print(f"\nForecast total season sales: ${season_forecast['forecasted_sales']:,.0f}")

# Simulate season
simulation_results, summary = forecaster.simulate_season(
    initial_inventory=5000,
    seasonal_curve=curve,
    total_forecast=season_forecast['forecasted_sales'] / 50,  # Per SKU
    n_simulations=1000
)

print(f"\nSeason Simulation Results:")
print(f"  Mean sell-through: {summary['mean_sell_through']:.1f}%")
print(f"  P10/P50/P90 sell-through: {summary['p10_sell_through']:.1f}% / {summary['p50_sell_through']:.1f}% / {summary['p90_sell_through']:.1f}%")
print(f"  Mean stockout rate: {summary['mean_stockout_rate']:.1f}%")
```

---

## In-Season Management

### Chase & Markdown Strategy

```python
class InSeasonManager:
    """
    Manage in-season performance

    React to actual performance vs. plan
    """

    def __init__(self, season_plan):
        self.plan = season_plan

    def identify_chase_opportunities(self, actual_sales, weeks_elapsed,
                                    current_inventory):
        """
        Identify products to chase (reorder)

        Chase when:
        - Selling faster than planned
        - Current inventory insufficient for season
        - Vendor lead time allows
        """

        opportunities = []

        for sku, sales in actual_sales.items():
            plan_sales = self.plan.get(sku, {}).get('total_plan', 0)
            weeks_remaining = self.plan['season_weeks'] - weeks_elapsed

            if weeks_elapsed == 0:
                continue

            # Calculate sell-through rate
            weekly_rate = sales / weeks_elapsed
            projected_total_sales = weekly_rate * self.plan['season_weeks']

            # Compare to plan
            vs_plan_pct = (projected_total_sales / plan_sales - 1) * 100 if plan_sales > 0 else 0

            # Check inventory sufficiency
            inventory_remaining = current_inventory.get(sku, 0)
            projected_remaining_sales = weekly_rate * weeks_remaining

            if vs_plan_pct > 20 and inventory_remaining < projected_remaining_sales:
                # Chase opportunity
                chase_qty = projected_remaining_sales - inventory_remaining

                # Check if lead time allows
                vendor_lead_time = self.plan.get(sku, {}).get('lead_time_weeks', 8)

                if weeks_remaining > vendor_lead_time + 2:  # Buffer
                    opportunities.append({
                        'sku': sku,
                        'vs_plan': vs_plan_pct,
                        'projected_total_sales': projected_total_sales,
                        'inventory_remaining': inventory_remaining,
                        'recommended_chase_qty': round(chase_qty, 0),
                        'urgency': 'High' if weeks_remaining < vendor_lead_time + 4 else 'Medium'
                    })

        return pd.DataFrame(opportunities)

    def identify_markdown_candidates(self, actual_sales, weeks_elapsed,
                                    current_inventory, target_str=0.75):
        """
        Identify products needing markdown

        Markdown when:
        - Selling slower than planned
        - Risk of excess inventory at season end
        """

        candidates = []

        weeks_remaining = self.plan['season_weeks'] - weeks_elapsed

        for sku, sales in actual_sales.items():
            initial_buy = self.plan.get(sku, {}).get('buy_quantity', 0)
            inventory_remaining = current_inventory.get(sku, 0)

            if initial_buy == 0:
                continue

            # Current sell-through rate
            current_str = (initial_buy - inventory_remaining) / initial_buy

            # Projected final sell-through
            if weeks_elapsed > 0:
                weekly_rate = sales / weeks_elapsed
                projected_additional_sales = weekly_rate * weeks_remaining
                projected_final_str = (sales + projected_additional_sales) / initial_buy
            else:
                projected_final_str = 0

            # If projected STR < target, need markdown
            if projected_final_str < target_str and inventory_remaining > 0:
                # Recommend markdown depth based on urgency
                if projected_final_str < 0.50:
                    recommended_markdown = 40
                elif projected_final_str < 0.60:
                    recommended_markdown = 30
                else:
                    recommended_markdown = 20

                candidates.append({
                    'sku': sku,
                    'current_str': round(current_str * 100, 1),
                    'projected_str': round(projected_final_str * 100, 1),
                    'inventory_remaining': inventory_remaining,
                    'recommended_markdown': recommended_markdown,
                    'urgency': 'High' if projected_final_str < 0.50 else 'Medium'
                })

        return pd.DataFrame(candidates)

    def calculate_season_health_score(self, actual_sales, weeks_elapsed,
                                     current_inventory):
        """
        Calculate overall season health score (0-100)

        Factors:
        - Sales vs. plan
        - Sell-through rate
        - Markdown rate
        - Inventory balance
        """

        total_plan_sales = sum(sku.get('total_plan', 0) for sku in self.plan.values() if isinstance(sku, dict))
        total_actual_sales = sum(actual_sales.values())

        # Sales attainment
        if total_plan_sales > 0:
            sales_attainment = total_actual_sales / (total_plan_sales * weeks_elapsed / self.plan['season_weeks'])
        else:
            sales_attainment = 0

        sales_score = min(sales_attainment * 50, 50)  # Max 50 points

        # Sell-through rate
        total_initial_buy = sum(sku.get('buy_quantity', 0) for sku in self.plan.values() if isinstance(sku, dict))
        total_current_inv = sum(current_inventory.values())

        if total_initial_buy > 0:
            current_str = (total_initial_buy - total_current_inv) / total_initial_buy
        else:
            current_str = 0

        # Target STR at this point in season
        target_str_now = weeks_elapsed / self.plan['season_weeks'] * 0.80  # 80% by end

        str_score = min(current_str / target_str_now * 30, 30) if target_str_now > 0 else 0

        # Inventory balance (not too much, not too little)
        weeks_remaining = self.plan['season_weeks'] - weeks_elapsed
        weekly_run_rate = total_actual_sales / weeks_elapsed if weeks_elapsed > 0 else 0
        weeks_of_supply = total_current_inv / weekly_run_rate if weekly_run_rate > 0 else 999

        if 0.8 * weeks_remaining <= weeks_of_supply <= 1.2 * weeks_remaining:
            balance_score = 20  # Perfect balance
        else:
            balance_score = max(0, 20 - abs(weeks_of_supply - weeks_remaining) * 2)

        total_score = sales_score + str_score + balance_score

        return {
            'health_score': round(total_score, 1),
            'sales_attainment': round(sales_attainment * 100, 1),
            'sell_through_rate': round(current_str * 100, 1),
            'weeks_of_supply': round(weeks_of_supply, 1),
            'interpretation': self._interpret_health_score(total_score)
        }

    def _interpret_health_score(self, score):
        """Interpret health score"""
        if score >= 80:
            return 'Excellent - on track for strong season'
        elif score >= 65:
            return 'Good - minor adjustments needed'
        elif score >= 50:
            return 'Fair - action required'
        else:
            return 'Poor - aggressive intervention needed'

# Example
season_plan = {
    'season_weeks': 16,
    'SKU001': {'total_plan': 10000, 'buy_quantity': 8000, 'lead_time_weeks': 6},
    'SKU002': {'total_plan': 15000, 'buy_quantity': 12000, 'lead_time_weeks': 8},
    'SKU003': {'total_plan': 5000, 'buy_quantity': 5000, 'lead_time_weeks': 6}
}

manager = InSeasonManager(season_plan)

# Week 6 performance
actual_sales = {'SKU001': 4500, 'SKU002': 4000, 'SKU003': 1200}
current_inventory = {'SKU001': 2500, 'SKU002': 7000, 'SKU003': 3500}
weeks_elapsed = 6

# Identify chase opportunities
chase_opps = manager.identify_chase_opportunities(
    actual_sales, weeks_elapsed, current_inventory
)
print("Chase Opportunities:")
print(chase_opps)

# Identify markdown candidates
markdown_candidates = manager.identify_markdown_candidates(
    actual_sales, weeks_elapsed, current_inventory, target_str=0.75
)
print("\nMarkdown Candidates:")
print(markdown_candidates)

# Season health score
health = manager.calculate_season_health_score(
    actual_sales, weeks_elapsed, current_inventory
)
print(f"\nSeason Health Score: {health['health_score']:.1f}/100")
print(f"Interpretation: {health['interpretation']}")
print(f"Sales attainment: {health['sales_attainment']:.1f}%")
print(f"Sell-through rate: {health['sell_through_rate']:.1f}%")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- `scipy.optimize`: Newsvendor optimization
- `pulp`, `pyomo`: Linear programming for assortment
- `numpy`: Numerical computations

**Forecasting:**
- `statsmodels`: Time series analysis
- `prophet`: Seasonal forecasting
- `pandas`: Data manipulation

**Simulation:**
- `numpy.random`: Monte Carlo simulation
- `scipy.stats`: Statistical distributions

### Commercial Software

**Planning Systems:**
- **Blue Yonder (JDA) Assortment**: Seasonal planning and optimization
- **o9 Solutions**: Digital planning platform
- **Oracle Retail Merchandise Planning**: Seasonal merchandise planning
- **SAP IBP**: Integrated business planning
- **RELEX Solutions**: Seasonal demand planning

**Specialized Tools:**
- **Armonia**: Retail planning suite
- **TXT Retail**: Fashion planning
- **APTOS Merchandise Lifecycle Management**: Seasonal planning

---

## Common Challenges & Solutions

### Challenge: Forecasting New Products

**Problem:**
- No historical data
- High uncertainty
- Risk of over/under buying

**Solutions:**
- Analog product approach
- Test markets / pilot stores
- Start conservative, chase winners
- Use product attributes (price, color, style)
- Market research and trend analysis
- Multiple scenarios (optimistic/realistic/conservative)

### Challenge: Weather Dependency

**Problem:**
- Unseasonable weather impacts sales
- Hard to predict
- Risk management

**Solutions:**
- Weather-based contingency plans
- Flexible vendor agreements
- Geographic diversification
- Pack-away programs (hold for next year)
- Transfer between climates
- Quick markdown response

### Challenge: Late Vendor Deliveries

**Problem:**
- Receipts arrive late
- Miss selling window
- Forced markdowns

**Solutions:**
- Air freight contingencies
- Vendor scorecards and penalties
- Multiple sourcing
- Buffer lead times in planning
- Early production starts
- Substitute product strategies

### Challenge: Balancing Newness vs. Basics

**Problem:**
- Fashion/trend items riskier
- Basics boring but reliable
- Need both for assortment

**Solutions:**
- 70/30 or 60/40 ratio (basics/fashion)
- Test fashion in limited quantities
- Fast fashion model (short lead times)
- Core basics with fashion colors
- Clear newness every season
- Price segmentation (basics lower, fashion higher)

### Challenge: End-of-Season Clearance

**Problem:**
- Leftover inventory
- Deep markdowns hurt margins
- Storage costs

**Solutions:**
- Aggressive early markdowns
- Pack-away for next year (if feasible)
- Outlet store distribution
- Liquidation companies
- Donation (tax benefit)
- Improved planning to reduce leftovers

---

## Output Format

### Seasonal Planning Report

**Executive Summary:**
- Season: Fall 2024 (August - November)
- Total buy plan: $4.2M at cost ($10.5M retail)
- Target sales: $9.2M (88% sell-through at full price)
- Target margin: 62% IMU, 58% maintained margin
- SKU count: 425 SKUs across 8 categories

**Financial Plan:**

| Metric | Target |
|--------|--------|
| Total buy at cost | $4.2M |
| Total buy at retail | $10.5M |
| Initial markup (IMU) | 62% |
| Sales plan | $9.2M |
| Sell-through target | 88% |
| Markdown budget | 4% of sales ($368K) |
| Maintained margin | 58% |
| Gross profit | $5.3M |

**Category Mix:**

| Category | Buy $ | Buy % | SKU Count | Avg Price | Strategy |
|----------|-------|-------|-----------|-----------|----------|
| Outerwear | $1.2M | 29% | 65 | $125 | Core + fashion, focus on trend colors |
| Sweaters | $980K | 23% | 95 | $68 | Basics with fashion accents |
| Dresses | $750K | 18% | 80 | $95 | Fashion-forward, limited quantities |
| Tops | $620K | 15% | 110 | $45 | High volume, core basics |
| Bottoms | $480K | 11% | 55 | $78 | Denim focus, seasonal colors |
| Accessories | $170K | 4% | 20 | $35 | Impulse items, high margin |

**New vs. Carry-Over:**

| Type | Buy $ | Buy % | Risk Level | Strategy |
|------|-------|-------|------------|----------|
| Carry-over (proven) | $2.5M | 60% | Low | Core basics, repeat winners |
| New items | $1.7M | 40% | High | Fashion, test quantities |

**Weekly Receipt Flow:**

| Week | Receipt $ | Cum % | Focus |
|------|-----------|-------|-------|
| Week -2 | $420K | 10% | Core basics early |
| Week 0 | $840K | 30% | Launch assortment |
| Week 2 | $630K | 45% | Fill-in and fashion |
| Week 4 | $420K | 55% | Fresh arrivals |
| Week 6-8 | $890K | 76% | Peak season support |
| Week 10+ | $1M | 100% | Late season, limited items |

**Risk Assessment:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Warm weather (delayed season start) | Medium | High | Conservative initial buy, chase plans ready |
| New product performance | High | Medium | Test quantities, monitor week 1 closely |
| Vendor delays | Low | High | Air freight budget, alternate suppliers |
| Competitive pricing | Medium | Medium | Markdown budget, price match capability |

**Success Metrics:**

| Metric | Target | Week 4 Check | Week 8 Check | Season End |
|--------|--------|--------------|--------------|------------|
| Sales vs. plan | 100% | ≥90% | ≥95% | ≥97% |
| Sell-through rate | 88% | ≥30% | ≥60% | ≥85% |
| Markdown rate | <4% | 0% | <2% | <5% |
| Gross margin | 58% | 62% | 60% | ≥57% |

**Action Plan:**

| Week | Action | Owner |
|------|--------|-------|
| Week -4 | Final assortment review, POs placed | Buyer |
| Week -2 | First receipts, allocation to stores | Allocator |
| Week 0 | Season launch, marketing campaign | Marketing |
| Week 1 | Monitor early reads, identify trends | Planner |
| Week 4 | Chase order decisions for winners | Buyer |
| Week 8 | First markdown evaluation | Planner |
| Week 12 | Aggressive clearance markdowns | Buyer |

---

## Questions to Ask

If you need more context:
1. What season are you planning? (spring, fall, holiday, etc.)
2. What's the season length? (weeks of selling)
3. What was last year's performance? (sales, sell-through, markdowns)
4. What's your sales target for this season?
5. What's your open-to-buy budget?
6. What % is new vs. carry-over merchandise?
7. What are your vendor lead times?
8. What's your target markdown rate?
9. What categories/product types are included?

---

## Related Skills

- **demand-forecasting**: Demand forecasting methodologies
- **retail-allocation**: Store allocation optimization
- **markdown-optimization**: Markdown strategy and timing
- **inventory-optimization**: Safety stock and inventory management
- **retail-replenishment**: In-season replenishment
- **planogram-optimization**: Space planning for seasonal sets
- **supply-chain-analytics**: Performance metrics and tracking

---
name: process-optimization
description: When the user wants to optimize manufacturing processes, improve throughput, reduce cycle times, or simulate process performance. Also use when the user mentions "process improvement," "bottleneck analysis," "simulation," "discrete-event simulation," "throughput optimization," "cycle time reduction," "process efficiency," "queuing theory," "process mapping," or "capacity analysis." For lean methods, see lean-manufacturing. For scheduling, see production-scheduling.
---

# Process Optimization

You are an expert in process optimization and industrial engineering. Your goal is to help organizations analyze, simulate, and optimize manufacturing and operational processes to improve throughput, reduce cycle times, eliminate bottlenecks, and maximize efficiency.

## Initial Assessment

Before optimizing processes, understand:

1. **Process Context**
   - What process needs optimization?
   - Current process flow and steps?
   - Known bottlenecks or constraints?
   - Current performance metrics?

2. **Process Characteristics**
   - Process type? (serial, parallel, job shop, assembly line)
   - Cycle times and processing rates?
   - Resource constraints (machines, labor, materials)?
   - Variability and randomness in process?

3. **Optimization Goals**
   - Increase throughput?
   - Reduce cycle time or lead time?
   - Improve resource utilization?
   - Reduce WIP inventory?

4. **Data Availability**
   - Historical process data available?
   - Time studies conducted?
   - Current state documented?
   - Access to observe process?

---

## Process Optimization Framework

### Process Analysis Methodology

**1. Define & Document**
- Process mapping (flowcharts, VSM)
- Identify inputs, outputs, resources
- Document current state

**2. Measure & Collect Data**
- Time studies
- Cycle time measurements
- Resource utilization tracking
- Quality data collection

**3. Analyze**
- Bottleneck identification
- Statistical analysis
- Root cause analysis
- Capacity calculations

**4. Simulate**
- Discrete-event simulation
- What-if scenarios
- Capacity planning
- Validate improvements

**5. Optimize**
- Implement improvements
- Balance resources
- Optimize scheduling
- Reduce variability

**6. Control & Monitor**
- Performance tracking
- Continuous improvement
- SPC monitoring

---

## Process Analysis & Bottleneck Identification

### Throughput Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ProcessAnalyzer:
    """
    Analyze process flow and identify bottlenecks
    Calculate throughput, cycle times, and utilization
    """

    def __init__(self, process_steps):
        """
        process_steps: list of dicts with process information

        Example:
        {
            'step': 'Cutting',
            'capacity_per_hour': 100,
            'processing_time_min': 0.6,
            'setup_time_min': 30,
            'reliability': 0.90
        }
        """
        self.steps = pd.DataFrame(process_steps)

    def identify_bottleneck(self):
        """
        Identify bottleneck process step
        Bottleneck = step with lowest capacity
        """

        # Adjust capacity for reliability
        self.steps['effective_capacity'] = (
            self.steps['capacity_per_hour'] * self.steps['reliability']
        )

        # Find bottleneck
        bottleneck_idx = self.steps['effective_capacity'].idxmin()
        bottleneck = self.steps.loc[bottleneck_idx]

        # System throughput limited by bottleneck
        system_throughput = bottleneck['effective_capacity']

        # Calculate utilization of each step based on bottleneck
        self.steps['utilization'] = (system_throughput / self.steps['effective_capacity']) * 100

        return {
            'bottleneck_step': bottleneck['step'],
            'bottleneck_capacity': bottleneck['effective_capacity'],
            'system_throughput': system_throughput,
            'process_analysis': self.steps
        }

    def calculate_cycle_time(self):
        """
        Calculate total cycle time (processing time through all steps)
        Assumes serial process
        """

        total_processing_time = self.steps['processing_time_min'].sum()
        total_setup_time = self.steps['setup_time_min'].sum()

        # Critical path (longest path)
        critical_path_time = total_processing_time

        return {
            'total_processing_time_min': total_processing_time,
            'total_processing_time_hours': total_processing_time / 60,
            'total_setup_time_min': total_setup_time,
            'critical_path_time': critical_path_time
        }

    def calculate_little_law(self, wip, throughput_per_hour):
        """
        Little's Law: WIP = Throughput × Lead Time
        or: Lead Time = WIP / Throughput

        Parameters:
        - wip: Work-in-Process inventory (units)
        - throughput_per_hour: throughput rate (units/hour)

        Returns lead time
        """

        lead_time_hours = wip / throughput_per_hour
        lead_time_days = lead_time_hours / 24

        return {
            'wip': wip,
            'throughput_per_hour': throughput_per_hour,
            'lead_time_hours': lead_time_hours,
            'lead_time_days': lead_time_days
        }

    def what_if_analysis(self, step_name, new_capacity):
        """
        What-if analysis: impact of changing capacity at one step

        Parameters:
        - step_name: name of step to modify
        - new_capacity: new capacity value

        Returns new system performance
        """

        modified_steps = self.steps.copy()
        modified_steps.loc[modified_steps['step'] == step_name, 'capacity_per_hour'] = new_capacity

        # Recalculate effective capacity
        modified_steps['effective_capacity'] = (
            modified_steps['capacity_per_hour'] * modified_steps['reliability']
        )

        # New bottleneck
        new_bottleneck_idx = modified_steps['effective_capacity'].idxmin()
        new_bottleneck = modified_steps.loc[new_bottleneck_idx]
        new_throughput = new_bottleneck['effective_capacity']

        # Improvement
        current_throughput = self.identify_bottleneck()['system_throughput']
        improvement = ((new_throughput - current_throughput) / current_throughput) * 100

        return {
            'modified_step': step_name,
            'original_capacity': self.steps.loc[self.steps['step'] == step_name, 'capacity_per_hour'].values[0],
            'new_capacity': new_capacity,
            'new_throughput': new_throughput,
            'new_bottleneck': new_bottleneck['step'],
            'improvement_pct': improvement
        }

    def plot_capacity_analysis(self):
        """Plot capacity analysis showing bottleneck"""

        bottleneck_analysis = self.identify_bottleneck()
        df = bottleneck_analysis['process_analysis']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Capacity bar chart
        colors = ['red' if step == bottleneck_analysis['bottleneck_step'] else 'skyblue'
                 for step in df['step']]

        ax1.bar(df['step'], df['effective_capacity'], color=colors, edgecolor='black', linewidth=1.5)
        ax1.axhline(bottleneck_analysis['system_throughput'], color='red', linestyle='--',
                   linewidth=2, label='System Throughput')
        ax1.set_xlabel('Process Step', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Capacity (units/hour)', fontsize=12, fontweight='bold')
        ax1.set_title('Process Capacity Analysis\n(Red = Bottleneck)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Utilization chart
        ax2.bar(df['step'], df['utilization'], color='lightgreen', edgecolor='black', linewidth=1.5)
        ax2.axhline(100, color='red', linestyle='--', linewidth=2, label='100% Utilization')
        ax2.set_xlabel('Process Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Utilization (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Resource Utilization', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 110])
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

# Example usage
process_steps = [
    {'step': 'Receiving', 'capacity_per_hour': 120, 'processing_time_min': 0.5, 'setup_time_min': 0, 'reliability': 1.0},
    {'step': 'Cutting', 'capacity_per_hour': 100, 'processing_time_min': 0.6, 'setup_time_min': 30, 'reliability': 0.90},
    {'step': 'Welding', 'capacity_per_hour': 80, 'processing_time_min': 0.75, 'setup_time_min': 45, 'reliability': 0.85},
    {'step': 'Assembly', 'capacity_per_hour': 90, 'processing_time_min': 0.67, 'setup_time_min': 20, 'reliability': 0.95},
    {'step': 'Testing', 'capacity_per_hour': 110, 'processing_time_min': 0.55, 'setup_time_min': 10, 'reliability': 0.98},
    {'step': 'Packaging', 'capacity_per_hour': 130, 'processing_time_min': 0.46, 'setup_time_min': 5, 'reliability': 0.99}
]

analyzer = ProcessAnalyzer(process_steps)

# Identify bottleneck
bottleneck = analyzer.identify_bottleneck()
print("Bottleneck Analysis:")
print(f"  Bottleneck: {bottleneck['bottleneck_step']}")
print(f"  Bottleneck Capacity: {bottleneck['bottleneck_capacity']:.1f} units/hour")
print(f"  System Throughput: {bottleneck['system_throughput']:.1f} units/hour")

print("\nProcess Utilization:")
print(bottleneck['process_analysis'][['step', 'effective_capacity', 'utilization']])

# Cycle time
cycle_time = analyzer.calculate_cycle_time()
print(f"\nCycle Time Analysis:")
print(f"  Total Processing Time: {cycle_time['total_processing_time_min']:.1f} minutes")

# Little's Law
littles = analyzer.calculate_little_law(wip=200, throughput_per_hour=bottleneck['system_throughput'])
print(f"\nLittle's Law (Lead Time Calculation):")
print(f"  WIP: {littles['wip']} units")
print(f"  Throughput: {littles['throughput_per_hour']:.1f} units/hour")
print(f"  Lead Time: {littles['lead_time_hours']:.1f} hours ({littles['lead_time_days']:.2f} days)")

# What-if analysis
what_if = analyzer.what_if_analysis('Welding', new_capacity=120)
print(f"\nWhat-If Analysis: Increase Welding capacity to 120 units/hour")
print(f"  New System Throughput: {what_if['new_throughput']:.1f} units/hour")
print(f"  New Bottleneck: {what_if['new_bottleneck']}")
print(f"  Improvement: {what_if['improvement_pct']:.1f}%")

# Plot
fig = analyzer.plot_capacity_analysis()
plt.show()
```

---

## Discrete-Event Simulation

### SimPy Manufacturing Simulation

```python
import simpy
import numpy as np
import pandas as pd

class ManufacturingProcess:
    """
    Discrete-event simulation of manufacturing process using SimPy
    """

    def __init__(self, env, process_config):
        """
        env: SimPy environment
        process_config: dict with process parameters
        """
        self.env = env
        self.config = process_config

        # Create resources (machines)
        self.machines = {
            name: simpy.Resource(env, capacity=config['capacity'])
            for name, config in process_config.items()
        }

        # Statistics tracking
        self.stats = {
            'completed_jobs': 0,
            'total_cycle_time': 0,
            'cycle_times': [],
            'wait_times': {name: [] for name in process_config.keys()},
            'queue_lengths': {name: [] for name in process_config.keys()},
            'utilization': {name: 0 for name in process_config.keys()}
        }

    def job_generator(self, interarrival_time=5.0, num_jobs=100):
        """
        Generate jobs arriving at process

        Parameters:
        - interarrival_time: mean time between job arrivals (minutes)
        - num_jobs: total jobs to generate
        """

        for i in range(num_jobs):
            # Random interarrival time (exponential distribution)
            yield self.env.timeout(np.random.exponential(interarrival_time))

            # Create job
            self.env.process(self.job_process(f'Job_{i}'))

    def job_process(self, job_id):
        """
        Process a single job through all steps
        """

        arrival_time = self.env.now

        for step_name, step_config in self.config.items():
            # Request resource
            with self.machines[step_name].request() as request:
                # Wait for resource
                wait_start = self.env.now
                yield request
                wait_time = self.env.now - wait_start

                # Track wait time
                self.stats['wait_times'][step_name].append(wait_time)

                # Processing time (can be deterministic or stochastic)
                if 'processing_time_std' in step_config:
                    process_time = np.random.normal(
                        step_config['processing_time'],
                        step_config['processing_time_std']
                    )
                    process_time = max(0.1, process_time)  # Ensure positive
                else:
                    process_time = step_config['processing_time']

                # Process
                yield self.env.timeout(process_time)

        # Job completed
        completion_time = self.env.now
        cycle_time = completion_time - arrival_time

        self.stats['completed_jobs'] += 1
        self.stats['total_cycle_time'] += cycle_time
        self.stats['cycle_times'].append(cycle_time)

    def monitor_queues(self, interval=10):
        """
        Monitor queue lengths at regular intervals

        Parameters:
        - interval: monitoring frequency (minutes)
        """

        while True:
            for step_name, machine in self.machines.items():
                queue_length = len(machine.queue)
                self.stats['queue_lengths'][step_name].append({
                    'time': self.env.now,
                    'queue_length': queue_length
                })

            yield self.env.timeout(interval)

    def calculate_results(self):
        """Calculate simulation results and statistics"""

        results = {
            'completed_jobs': self.stats['completed_jobs'],
            'avg_cycle_time': np.mean(self.stats['cycle_times']) if self.stats['cycle_times'] else 0,
            'std_cycle_time': np.std(self.stats['cycle_times']) if self.stats['cycle_times'] else 0,
            'min_cycle_time': np.min(self.stats['cycle_times']) if self.stats['cycle_times'] else 0,
            'max_cycle_time': np.max(self.stats['cycle_times']) if self.stats['cycle_times'] else 0,
            'throughput_per_hour': (self.stats['completed_jobs'] / self.env.now) * 60 if self.env.now > 0 else 0
        }

        # Average wait times by step
        results['avg_wait_times'] = {
            step: np.mean(waits) if waits else 0
            for step, waits in self.stats['wait_times'].items()
        }

        # Average queue lengths
        results['avg_queue_lengths'] = {
            step: np.mean([q['queue_length'] for q in queues]) if queues else 0
            for step, queues in self.stats['queue_lengths'].items()
        }

        return results


def run_simulation(process_config, interarrival_time=5.0, num_jobs=100, sim_time=None):
    """
    Run manufacturing simulation

    Parameters:
    - process_config: dict defining process steps and parameters
    - interarrival_time: mean time between arrivals
    - num_jobs: number of jobs to simulate
    - sim_time: simulation time limit (optional)

    Returns simulation results
    """

    # Create simulation environment
    env = simpy.Environment()

    # Create manufacturing process
    process = ManufacturingProcess(env, process_config)

    # Start job generator
    env.process(process.job_generator(interarrival_time, num_jobs))

    # Start queue monitoring
    env.process(process.monitor_queues(interval=10))

    # Run simulation
    if sim_time:
        env.run(until=sim_time)
    else:
        env.run()

    # Calculate results
    results = process.calculate_results()

    return results, process


# Example usage
process_config = {
    'Cutting': {
        'capacity': 2,  # 2 machines
        'processing_time': 6.0,  # 6 minutes average
        'processing_time_std': 1.0  # variability
    },
    'Welding': {
        'capacity': 1,  # 1 machine (potential bottleneck)
        'processing_time': 8.0,
        'processing_time_std': 1.5
    },
    'Assembly': {
        'capacity': 2,
        'processing_time': 5.0,
        'processing_time_std': 0.8
    },
    'Inspection': {
        'capacity': 1,
        'processing_time': 3.0,
        'processing_time_std': 0.5
    }
}

print("Running simulation...")
results, process = run_simulation(
    process_config,
    interarrival_time=4.0,  # Jobs arrive every 4 minutes on average
    num_jobs=200
)

print("\nSimulation Results:")
print(f"  Completed Jobs: {results['completed_jobs']}")
print(f"  Average Cycle Time: {results['avg_cycle_time']:.2f} minutes")
print(f"  Std Dev Cycle Time: {results['std_cycle_time']:.2f} minutes")
print(f"  Throughput: {results['throughput_per_hour']:.2f} jobs/hour")

print("\nAverage Wait Times by Step:")
for step, wait_time in results['avg_wait_times'].items():
    print(f"  {step}: {wait_time:.2f} minutes")

print("\nAverage Queue Lengths:")
for step, queue_length in results['avg_queue_lengths'].items():
    print(f"  {step}: {queue_length:.2f} jobs")

# What-if scenario: Add capacity at bottleneck
print("\n" + "="*50)
print("What-If Scenario: Add 1 machine to Welding")
print("="*50)

process_config_improved = process_config.copy()
process_config_improved['Welding'] = {
    'capacity': 2,  # Increase from 1 to 2
    'processing_time': 8.0,
    'processing_time_std': 1.5
}

results_improved, _ = run_simulation(
    process_config_improved,
    interarrival_time=4.0,
    num_jobs=200
)

print("\nImproved Results:")
print(f"  Completed Jobs: {results_improved['completed_jobs']}")
print(f"  Average Cycle Time: {results_improved['avg_cycle_time']:.2f} minutes (was {results['avg_cycle_time']:.2f})")
print(f"  Throughput: {results_improved['throughput_per_hour']:.2f} jobs/hour (was {results['throughput_per_hour']:.2f})")

improvement = ((results_improved['throughput_per_hour'] - results['throughput_per_hour']) /
              results['throughput_per_hour']) * 100
print(f"  Improvement: {improvement:.1f}%")
```

### Queuing Theory Analysis

```python
class QueuingAnalysis:
    """
    Queuing theory (M/M/c) analysis for process performance
    """

    def __init__(self, arrival_rate, service_rate, num_servers):
        """
        Parameters:
        - arrival_rate: λ (lambda) - jobs per hour
        - service_rate: μ (mu) - jobs per hour per server
        - num_servers: c - number of servers/machines
        """
        self.lambda_rate = arrival_rate
        self.mu_rate = service_rate
        self.c = num_servers

        # Traffic intensity
        self.rho = arrival_rate / (service_rate * num_servers)

    def calculate_performance(self):
        """
        Calculate M/M/c queue performance metrics

        Returns:
        - L: Average number in system
        - Lq: Average number in queue
        - W: Average time in system
        - Wq: Average time in queue
        - utilization: Server utilization
        """

        lambda_rate = self.lambda_rate
        mu = self.mu_rate
        c = self.c
        rho = self.rho

        # Check stability
        if rho >= 1:
            return {
                'status': 'UNSTABLE - Arrival rate exceeds service capacity',
                'utilization': rho * 100
            }

        # Calculate P0 (probability of 0 in system)
        # Simplified for c servers
        sum_term = sum([(lambda_rate / mu)**n / np.math.factorial(n) for n in range(c)])
        last_term = (lambda_rate / mu)**c / (np.math.factorial(c) * (1 - rho))
        P0 = 1 / (sum_term + last_term)

        # Average number in queue (Lq)
        Lq = (P0 * (lambda_rate / mu)**c * rho) / (np.math.factorial(c) * (1 - rho)**2)

        # Average number in system (L)
        L = Lq + (lambda_rate / mu)

        # Average time in queue (Wq)
        Wq = Lq / lambda_rate

        # Average time in system (W)
        W = Wq + (1 / mu)

        # Utilization
        utilization = rho

        return {
            'status': 'STABLE',
            'L_avg_in_system': L,
            'Lq_avg_in_queue': Lq,
            'W_avg_time_in_system_hours': W,
            'Wq_avg_time_in_queue_hours': Wq,
            'utilization_pct': utilization * 100,
            'P0_prob_empty': P0,
            'throughput': lambda_rate
        }

    def calculate_optimal_servers(self, max_wait_time_hours):
        """
        Find minimum number of servers to meet wait time target

        Parameters:
        - max_wait_time_hours: maximum acceptable wait time

        Returns optimal number of servers
        """

        for c in range(1, 50):
            self.c = c
            self.rho = self.lambda_rate / (self.mu_rate * c)

            if self.rho < 1:
                perf = self.calculate_performance()

                if perf['status'] == 'STABLE' and perf['Wq_avg_time_in_queue_hours'] <= max_wait_time_hours:
                    return {
                        'optimal_servers': c,
                        'wait_time_hours': perf['Wq_avg_time_in_queue_hours'],
                        'utilization_pct': perf['utilization_pct'],
                        'performance': perf
                    }

        return {
            'optimal_servers': None,
            'message': 'Could not find solution within server range'
        }

# Example usage
# Process with arrival rate of 15 jobs/hour, service rate of 6 jobs/hour per server
queuing = QueuingAnalysis(
    arrival_rate=15,    # 15 jobs/hour arrive
    service_rate=6,     # Each server can process 6 jobs/hour
    num_servers=3       # 3 servers available
)

performance = queuing.calculate_performance()

print("Queuing Theory Analysis (M/M/c):")
print(f"  Status: {performance['status']}")
print(f"  Average # in System (L): {performance['L_avg_in_system']:.2f} jobs")
print(f"  Average # in Queue (Lq): {performance['Lq_avg_in_queue']:.2f} jobs")
print(f"  Average Time in System (W): {performance['W_avg_time_in_system_hours']:.3f} hours ({performance['W_avg_time_in_system_hours']*60:.1f} min)")
print(f"  Average Wait Time (Wq): {performance['Wq_avg_time_in_queue_hours']:.3f} hours ({performance['Wq_avg_time_in_queue_hours']*60:.1f} min)")
print(f"  Server Utilization: {performance['utilization_pct']:.1f}%")

# Find optimal servers for max 5 minute wait
optimal = queuing.calculate_optimal_servers(max_wait_time_hours=5/60)
print(f"\nOptimal Server Count (for max 5 min wait):")
print(f"  Optimal Servers: {optimal['optimal_servers']}")
print(f"  Expected Wait Time: {optimal['wait_time_hours']*60:.2f} minutes")
print(f"  Utilization: {optimal['utilization_pct']:.1f}%")
```

---

## Process Improvement Techniques

### Process Balancing

```python
class ProcessBalancing:
    """
    Balance process to eliminate bottlenecks and improve flow
    """

    def __init__(self, workstations, target_output_per_hour):
        """
        workstations: list of dicts with workstation info
        target_output_per_hour: desired production rate
        """
        self.workstations = pd.DataFrame(workstations)
        self.target_output = target_output_per_hour

    def calculate_balance(self):
        """
        Calculate line balance metrics
        """

        # Required cycle time (takt time)
        takt_time = 60 / self.target_output  # minutes per unit

        # Current cycle time (bottleneck determines)
        self.workstations['cycle_time'] = 60 / self.workstations['capacity_per_hour']

        bottleneck_time = self.workstations['cycle_time'].max()
        actual_output = 60 / bottleneck_time

        # Calculate idle time
        self.workstations['idle_time'] = bottleneck_time - self.workstations['cycle_time']

        # Balance efficiency
        total_work_time = self.workstations['cycle_time'].sum()
        balance_efficiency = (total_work_time / (len(self.workstations) * bottleneck_time)) * 100

        # Balance delay
        balance_delay = 100 - balance_efficiency

        return {
            'takt_time': takt_time,
            'bottleneck_cycle_time': bottleneck_time,
            'actual_output_per_hour': actual_output,
            'target_output_per_hour': self.target_output,
            'balance_efficiency_pct': balance_efficiency,
            'balance_delay_pct': balance_delay,
            'workstation_analysis': self.workstations
        }

    def recommend_improvements(self, balance_results):
        """Generate improvement recommendations"""

        recommendations = []

        df = balance_results['workstation_analysis']

        # Identify bottleneck
        bottleneck = df.loc[df['cycle_time'].idxmax()]

        recommendations.append({
            'priority': 'High',
            'workstation': bottleneck['workstation'],
            'issue': 'Bottleneck',
            'action': f"Reduce cycle time from {bottleneck['cycle_time']:.2f} to {balance_results['takt_time']:.2f} minutes",
            'methods': [
                'Add parallel workstation',
                'Improve methods/tools',
                'Redistribute tasks to other stations',
                'Eliminate non-value-added activities'
            ]
        })

        # Identify highly imbalanced stations
        for _, ws in df.iterrows():
            if ws['idle_time'] > balance_results['takt_time'] * 0.3:  # >30% idle
                recommendations.append({
                    'priority': 'Medium',
                    'workstation': ws['workstation'],
                    'issue': 'Underutilized',
                    'action': f"Add tasks to utilize {ws['idle_time']:.2f} min of idle time",
                    'methods': [
                        'Redistribute tasks from bottleneck',
                        'Combine with adjacent workstation',
                        'Reduce number of operators'
                    ]
                })

        return pd.DataFrame(recommendations)

# Example usage
workstations = [
    {'workstation': 'WS1', 'capacity_per_hour': 65, 'operators': 1},
    {'workstation': 'WS2', 'capacity_per_hour': 50, 'operators': 1},  # Bottleneck
    {'workstation': 'WS3', 'capacity_per_hour': 70, 'operators': 1},
    {'workstation': 'WS4', 'capacity_per_hour': 60, 'operators': 1},
]

balancing = ProcessBalancing(workstations, target_output_per_hour=55)

balance = balancing.calculate_balance()

print("Process Balance Analysis:")
print(f"  Target Output: {balance['target_output_per_hour']:.1f} units/hour")
print(f"  Actual Output: {balance['actual_output_per_hour']:.1f} units/hour")
print(f"  Takt Time: {balance['takt_time']:.2f} minutes")
print(f"  Bottleneck Cycle Time: {balance['bottleneck_cycle_time']:.2f} minutes")
print(f"  Balance Efficiency: {balance['balance_efficiency_pct']:.1f}%")
print(f"  Balance Delay: {balance['balance_delay_pct']:.1f}%")

print("\nWorkstation Analysis:")
print(balance['workstation_analysis'][['workstation', 'capacity_per_hour', 'cycle_time', 'idle_time']])

recommendations = balancing.recommend_improvements(balance)
print("\nImprovement Recommendations:")
print(recommendations[['priority', 'workstation', 'issue', 'action']])
```

---

## Tools & Libraries

### Python Libraries

**Simulation:**
- `simpy`: Discrete-event simulation framework
- `salabim`: Simulation with animation
- `mesa`: Agent-based modeling

**Optimization:**
- `scipy.optimize`: Optimization algorithms
- `pulp`: Linear programming
- `pyomo`: Optimization modeling

**Analysis:**
- `numpy`, `pandas`: Data analysis
- `matplotlib`, `seaborn`, `plotly`: Visualization
- `networkx`: Process flow diagrams

### Commercial Process Optimization Software

**Simulation:**
- **Arena**: Discrete-event simulation (Rockwell)
- **AnyLogic**: Multi-method simulation
- **Simio**: Process simulation and optimization
- **FlexSim**: 3D simulation
- **Plant Simulation**: Siemens process simulation

**Process Mining:**
- **Celonis**: Process mining and optimization
- **UiPath Process Mining**: Process discovery
- **ProcessGold**: Process intelligence
- **Disco**: Fluxicon process mining

**Industrial Engineering:**
- **ProModel**: Process simulation
- **WITNESS**: Simulation modeling

---

## Common Challenges & Solutions

### Challenge: Data Not Available

**Problem:**
- No historical process data
- Difficult to measure cycle times
- Variability unknown

**Solutions:**
- Time studies (observe and measure)
- Pilot data collection period
- Use simulation with estimated parameters
- Sensitivity analysis on assumptions
- Start with deterministic model, add variability later

### Challenge: High Process Variability

**Problem:**
- Unpredictable cycle times
- Random failures and disruptions
- Difficult to optimize

**Solutions:**
- Identify and reduce sources of variation
- Add buffers strategically
- Use simulation to understand impact
- Queue theory to size buffers
- Focus on most variable steps first

### Challenge: Complex Interdependencies

**Problem:**
- Steps depend on each other
- Rework loops and quality checks
- Shared resources

**Solutions:**
- Use simulation (handles complexity well)
- Map dependencies explicitly
- Simplify model first, add complexity incrementally
- Focus on critical path

### Challenge: Multiple Objectives

**Problem:**
- Minimize cycle time vs. minimize WIP
- Maximize throughput vs. minimize cost
- Trade-offs not clear

**Solutions:**
- Define priority of objectives
- Multi-objective optimization
- Use simulation to evaluate trade-offs
- Pareto analysis
- Involve stakeholders in prioritization

---

## Output Format

### Process Optimization Report

**Executive Summary:**
- Current process performance
- Bottlenecks identified
- Improvement opportunities
- Expected benefits

**Process Analysis:**

| Step | Capacity | Cycle Time | Utilization | Queue | Bottleneck |
|------|----------|------------|-------------|-------|------------|
| Cutting | 100/hr | 0.6 min | 68% | 2.3 jobs | No |
| Welding | 80/hr | 0.75 min | 85% | 5.7 jobs | **YES** |
| Assembly | 90/hr | 0.67 min | 76% | 1.8 jobs | No |
| Testing | 110/hr | 0.55 min | 62% | 0.5 jobs | No |

**Current Performance:**
- System Throughput: 68 units/hour (limited by Welding)
- Average Cycle Time: 45 minutes
- Average WIP: 25 units
- Balance Efficiency: 72%

**Simulation Results:**
- Baseline: 68 units/hour, 45 min cycle time
- Scenario 1 (Add Welding capacity): 85 units/hour (+25%), 35 min cycle time
- Scenario 2 (Balance line): 75 units/hour (+10%), 38 min cycle time
- Scenario 3 (Combined): 95 units/hour (+40%), 30 min cycle time

**Recommendations:**

**Priority 1: Address Welding Bottleneck**
- Add 1 welding station (increase from 1 to 2 machines)
- Expected improvement: +25% throughput
- Investment: $150K
- ROI: 8 months

**Priority 2: Balance Workstations**
- Redistribute tasks to balance cycle times
- Target takt time: 0.63 minutes
- Expected improvement: +10% throughput
- Investment: Training only

**Priority 3: Reduce Variability**
- Implement standard work at Welding
- Preventive maintenance to reduce breakdowns
- Expected: Reduce cycle time variation by 30%

**Expected Benefits:**
- Throughput increase: 35-40%
- Cycle time reduction: 30-35%
- WIP reduction: 40%
- Annual savings: $500K

---

## Questions to Ask

If you need more context:
1. What process needs optimization?
2. What are current cycle times and throughput?
3. What are the process steps and resource constraints?
4. What data is available (time studies, historical data)?
5. What is the primary optimization goal?
6. Are there quality or reliability issues?
7. What is the budget for improvements?
8. Timeline for implementation?

---

## Related Skills

- **production-scheduling**: For scheduling optimization
- **lean-manufacturing**: For waste elimination and flow
- **capacity-planning**: For capacity analysis
- **assembly-line-balancing**: For line balancing specifics
- **quality-management**: For process quality improvement
- **maintenance-planning**: For equipment reliability
- **optimization-modeling**: For mathematical optimization
- **supply-chain-analytics**: For performance metrics

---
name: order-fulfillment
description: When the user wants to design or optimize order fulfillment operations, improve pick-pack-ship processes, or reduce fulfillment costs. Also use when the user mentions "order processing," "pick-pack-ship," "picking strategy," "packing operations," "shipping optimization," "wave planning," "batch picking," or "fulfillment center operations." For warehouse layout, see warehouse-design. For routing pickers, see picker-routing-optimization.
---

# Order Fulfillment

You are an expert in order fulfillment operations and optimization. Your goal is to help design efficient, accurate, and cost-effective fulfillment processes that meet customer service expectations while minimizing labor and operational costs.

## Initial Assessment

Before optimizing fulfillment, understand:

1. **Order Profile**
   - Order volume? (orders/day, peak vs. average)
   - Order characteristics? (lines/order, units/order)
   - Order types? (B2B pallets, B2C eaches, mixed)
   - SKU count and velocity distribution?

2. **Service Requirements**
   - Delivery speed? (same-day, next-day, 2-day, standard)
   - Cutoff times for shipping?
   - Order accuracy targets? (99.5%+)
   - Special services? (gift wrap, kitting, customization)

3. **Current Operations**
   - Current fulfillment process and flow?
   - Pick accuracy and productivity rates?
   - Technology in use? (WMS, automation, RF scanners)
   - Pain points and bottlenecks?

4. **Constraints**
   - Labor availability and cost?
   - Facility layout and space?
   - Capital budget for improvements?
   - IT systems and integration capabilities?

---

## Order Fulfillment Framework

### Core Fulfillment Processes

**1. Order Receipt & Validation**
- Order intake from channels (web, EDI, phone)
- Inventory availability check (ATP)
- Credit/payment verification
- Order prioritization and batching

**2. Picking**
- Retrieve items from storage locations
- Verify SKU and quantity accuracy
- Multiple strategies (discrete, batch, zone, wave)

**3. Packing**
- Select appropriate packaging
- Pack items securely
- Insert documents (packing slip, returns)
- Generate shipping label

**4. Shipping**
- Carrier selection and manifesting
- Trailer loading and departure
- Track and trace updates

**5. Returns Processing**
- Receive and inspect returns
- Disposition (restock, liquidate, destroy)
- Customer refund/exchange processing

---

## Picking Strategies

### Strategy Comparison

| Strategy | Orders/Hour | Labor Efficiency | Accuracy | Complexity | Best For |
|----------|-------------|------------------|----------|------------|----------|
| Discrete (Single Order) | 20-40 | Low | High | Low | Low volume, high value |
| Batch Picking | 60-100 | Medium | Medium | Medium | Medium volume |
| Zone Picking | 80-150 | High | Medium | High | High SKU count |
| Wave Picking | 100-200+ | Very High | Medium | Very High | High volume |
| Cluster Picking | 100-150 | High | High | Medium | Multi-order picking |

### Discrete Order Picking

**Description:**
- Pick one order at a time
- Complete each order before starting next
- Simple, accurate, inefficient

**When to Use:**
- Low order volume (<500 orders/day)
- High-value orders requiring accuracy
- Complex orders with customization

```python
import numpy as np
import pandas as pd

def discrete_picking_capacity(orders_per_day, lines_per_order,
                              pick_rate_per_hour=100,
                              working_hours=8):
    """
    Calculate labor requirements for discrete picking

    Parameters:
    - orders_per_day: Daily order volume
    - lines_per_order: Average lines per order
    - pick_rate_per_hour: Picks per person per hour (includes travel)
    - working_hours: Working hours per shift
    """

    picks_per_day = orders_per_day * lines_per_order

    # Labor hours needed
    labor_hours_needed = picks_per_day / pick_rate_per_hour

    # Pickers needed
    pickers_needed = labor_hours_needed / working_hours

    # Orders per picker per day
    orders_per_picker = orders_per_day / pickers_needed

    return {
        'picks_per_day': picks_per_day,
        'labor_hours_needed': round(labor_hours_needed, 1),
        'pickers_needed': round(pickers_needed, 1),
        'orders_per_picker_per_day': round(orders_per_picker, 0),
        'picks_per_picker_per_hour': pick_rate_per_hour
    }

# Example
discrete = discrete_picking_capacity(
    orders_per_day=500,
    lines_per_order=5,
    pick_rate_per_hour=100,
    working_hours=8
)

print(f"Pickers needed: {discrete['pickers_needed']}")
print(f"Orders per picker: {discrete['orders_per_picker_per_day']}")
```

### Batch Picking

**Description:**
- Pick multiple orders simultaneously
- Pick each SKU once for all orders in batch
- Sort items to orders after picking

**Benefits:**
- Reduce travel time (visit each location once)
- 40-60% productivity improvement vs. discrete

**Implementation:**

```python
def batch_picking_optimization(orders_df, batch_size=10, sort_time_per_unit=3):
    """
    Optimize batch picking

    Parameters:
    - orders_df: DataFrame with columns ['order_id', 'sku', 'quantity', 'location']
    - batch_size: Orders per batch
    - sort_time_per_unit: Seconds to sort each unit to order

    Returns:
    - Batch assignments and performance metrics
    """

    # Create batches
    orders_df = orders_df.copy()
    orders_df['batch'] = (orders_df.index // batch_size) + 1

    # Calculate picks per batch
    batch_summary = orders_df.groupby('batch').agg({
        'order_id': 'nunique',
        'sku': 'count',
        'quantity': 'sum',
        'location': 'nunique'
    }).rename(columns={
        'order_id': 'orders',
        'sku': 'pick_lines',
        'location': 'unique_locations'
    })

    # Estimate time savings
    # Discrete: visit each location for each order
    # Batch: visit each location once per batch

    discrete_picks = len(orders_df)
    batch_picks = batch_summary['unique_locations'].sum()

    pick_reduction_pct = (discrete_picks - batch_picks) / discrete_picks

    # Sort time required
    total_units = batch_summary['quantity'].sum()
    sort_time_hours = (total_units * sort_time_per_unit) / 3600

    return {
        'total_batches': len(batch_summary),
        'avg_orders_per_batch': batch_summary['orders'].mean(),
        'discrete_picks': discrete_picks,
        'batch_picks': batch_picks,
        'pick_reduction_%': round(pick_reduction_pct * 100, 1),
        'sort_time_hours': round(sort_time_hours, 2),
        'batch_summary': batch_summary
    }

# Example
orders_df = pd.DataFrame({
    'order_id': [f'ORD_{i}' for i in range(1, 101)],
    'sku': np.random.choice(['SKU_A', 'SKU_B', 'SKU_C', 'SKU_D'], 100),
    'quantity': np.random.randint(1, 5, 100),
    'location': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], 100)
})

batch_result = batch_picking_optimization(orders_df, batch_size=10)
print(f"Pick reduction: {batch_result['pick_reduction_%']}%")
print(f"Sort time required: {batch_result['sort_time_hours']} hours")
```

### Zone Picking

**Description:**
- Divide warehouse into zones
- Each picker assigned to a zone
- Orders pass through zones sequentially or consolidate at end

**Types:**
- **Sequential zone picking**: Order travels zone to zone
- **Batch zone picking**: Pick to totes, consolidate later

**Benefits:**
- Pickers become expert in their zone
- Parallel processing (multiple orders picked simultaneously)
- Reduces congestion

```python
def zone_picking_design(warehouse_sq_ft, num_pickers, orders_per_day,
                       lines_per_order, sku_distribution):
    """
    Design zone picking system

    Parameters:
    - warehouse_sq_ft: Total picking area
    - num_pickers: Available pickers
    - orders_per_day: Daily order volume
    - lines_per_order: Average lines per order
    - sku_distribution: Dict with zone: % of picks
    """

    picks_per_day = orders_per_day * lines_per_order

    # Allocate zones based on pick volume
    zone_allocation = {}

    for zone, pct in sku_distribution.items():
        picks_in_zone = picks_per_day * pct
        pickers_needed = picks_in_zone / (100 * 8)  # 100 picks/hr, 8 hrs

        zone_allocation[zone] = {
            'pick_volume': round(picks_in_zone, 0),
            'pickers_needed': round(pickers_needed, 1),
            'sq_ft': round(warehouse_sq_ft * pct, 0)
        }

    return zone_allocation

# Example
sku_dist = {
    'Zone_A_Fast': 0.40,  # 40% of picks
    'Zone_B_Medium': 0.35,
    'Zone_C_Slow': 0.25
}

zones = zone_picking_design(
    warehouse_sq_ft=50000,
    num_pickers=20,
    orders_per_day=2000,
    lines_per_order=5,
    sku_distribution=sku_dist
)

print("Zone Allocation:")
for zone, data in zones.items():
    print(f"\n  {zone}:")
    print(f"    Pick volume: {data['pick_volume']}")
    print(f"    Pickers needed: {data['pickers_needed']}")
    print(f"    Square footage: {data['sq_ft']}")
```

### Wave Picking

**Description:**
- Release groups of orders together as "waves"
- Optimize wave composition for efficiency
- Coordinate picking, packing, shipping

**Wave Design Factors:**
- Carrier schedule (UPS pickup at 5pm)
- Order priority (SLA requirements)
- Resource availability (pickers, packers)
- Warehouse capacity (staging space)

```python
def wave_planning(orders_df, waves_per_day=4, target_wave_size=500):
    """
    Plan picking waves

    Parameters:
    - orders_df: DataFrame with ['order_id', 'priority', 'lines', 'carrier', 'cutoff_time']
    - waves_per_day: Number of waves per day
    - target_wave_size: Target orders per wave

    Returns:
    - Wave assignments
    """

    orders_df = orders_df.copy()

    # Sort by priority and cutoff time
    orders_df = orders_df.sort_values(['priority', 'cutoff_time'], ascending=[False, True])

    # Assign to waves
    orders_df['wave'] = (orders_df.index // target_wave_size) % waves_per_day + 1

    # Wave summary
    wave_summary = orders_df.groupby('wave').agg({
        'order_id': 'count',
        'lines': 'sum',
        'cutoff_time': 'max'
    }).rename(columns={
        'order_id': 'orders',
        'lines': 'total_picks'
    })

    # Estimate time required per wave
    wave_summary['pick_hours'] = wave_summary['total_picks'] / 100  # 100 picks/hr
    wave_summary['pickers_needed'] = np.ceil(wave_summary['pick_hours'] / 2)  # 2 hr wave

    return orders_df[['order_id', 'wave', 'priority']], wave_summary

# Example
orders_data = pd.DataFrame({
    'order_id': [f'ORD_{i}' for i in range(1, 2001)],
    'priority': np.random.choice([1, 2, 3], 2000, p=[0.1, 0.3, 0.6]),
    'lines': np.random.poisson(5, 2000),
    'carrier': np.random.choice(['UPS', 'FedEx', 'USPS'], 2000),
    'cutoff_time': np.random.choice(['12:00', '15:00', '17:00'], 2000)
})

wave_assignments, wave_summary = wave_planning(orders_data, waves_per_day=4)
print("Wave Summary:")
print(wave_summary)
```

### Cluster Picking

**Description:**
- Pick multiple orders to a multi-compartment cart
- Each compartment represents an order
- Pick all orders simultaneously

**Benefits:**
- High productivity (combine benefits of batch and discrete)
- Maintain order integrity
- Ideal for e-commerce (small orders, many SKUs)

**Equipment:**
- Pick carts with 4-12 totes/compartments
- Voice or RF-directed picking

---

## Pick Path Optimization

### Routing Strategies

**1. S-Shape Routing**
- Enter aisles with picks, skip empty aisles
- Most common, simple to implement

**2. Return Routing**
- Enter and exit same end of aisle
- Good for selective aisles with few picks

**3. Midpoint Routing**
- Enter nearest end of aisle
- Most efficient for random pick locations

**4. Largest Gap**
- Skip largest gap between picks in aisle
- Optimal for most scenarios

```python
def calculate_pick_path_distance(pick_locations, aisle_length_ft=100,
                                aisle_width_ft=10, routing='s-shape'):
    """
    Calculate travel distance for pick path

    Parameters:
    - pick_locations: List of tuples [(aisle, position_pct), ...]
    - aisle_length_ft: Length of aisle
    - aisle_width_ft: Width between aisles
    - routing: 's-shape', 'return', 'largest-gap'

    Returns:
    - Total travel distance
    """

    # Group picks by aisle
    aisles_with_picks = {}
    for aisle, position in pick_locations:
        if aisle not in aisles_with_picks:
            aisles_with_picks[aisle] = []
        aisles_with_picks[aisle].append(position)

    total_distance = 0

    if routing == 's-shape':
        # Traverse aisles with picks, skip empty
        for aisle, positions in sorted(aisles_with_picks.items()):
            # Full aisle length + cross-aisle
            total_distance += aisle_length_ft + aisle_width_ft

    elif routing == 'return':
        # Enter and exit same end
        for aisle, positions in aisles_with_picks.items():
            max_position = max(positions)
            # Go to furthest pick and return
            total_distance += (max_position * aisle_length_ft * 2) + aisle_width_ft

    elif routing == 'largest-gap':
        # Optimal routing (simplified)
        for aisle, positions in aisles_with_picks.items():
            positions_sorted = sorted(positions)

            # Find largest gap
            gaps = [positions_sorted[i+1] - positions_sorted[i]
                   for i in range(len(positions_sorted)-1)]

            if gaps:
                largest_gap = max(gaps)
                # Distance = full aisle - largest gap
                distance_in_aisle = aisle_length_ft * (1 - largest_gap)
            else:
                distance_in_aisle = positions_sorted[0] * aisle_length_ft

            total_distance += distance_in_aisle + aisle_width_ft

    return round(total_distance, 1)

# Example pick path
picks = [
    (1, 0.2),   # Aisle 1, 20% down
    (1, 0.8),   # Aisle 1, 80% down
    (3, 0.5),   # Aisle 3, 50% down
    (5, 0.3),   # Aisle 5, 30% down
]

for routing in ['s-shape', 'return', 'largest-gap']:
    distance = calculate_pick_path_distance(picks, routing=routing)
    print(f"{routing}: {distance} ft")
```

---

## Packing Operations

### Packing Strategies

**1. Single-Pass Packing**
- Pick directly into shipping box
- Fastest, but requires known box size
- Good for single-item orders

**2. Pack Station (Traditional)**
- Central packing area
- Pickers bring items to packers
- Flexibility in box selection

**3. In-Line Packing**
- Pack as you pick
- Requires pick-to-belt or cart system

**4. Automated Packing**
- Auto-box selection and sealing
- High throughput (500+ boxes/hour)
- Capital intensive ($500K+)

### Box Selection Optimization

```python
def optimize_box_selection(order_items, box_inventory):
    """
    Select optimal box size for order

    Parameters:
    - order_items: List of dicts [{'sku': 'A', 'qty': 2, 'dims': (10,8,4)}, ...]
    - box_inventory: List of available boxes [{'box_id': 'Small', 'dims': (12,10,8), 'cost': 0.50}, ...]

    Returns:
    - Optimal box selection
    """

    # Calculate total volume needed
    total_volume = sum(
        item['qty'] * item['dims'][0] * item['dims'][1] * item['dims'][2]
        for item in order_items
    )

    # Find boxes that fit (with utilization target)
    target_utilization = 0.80  # 80% full
    required_volume = total_volume / target_utilization

    suitable_boxes = [
        box for box in box_inventory
        if box['dims'][0] * box['dims'][1] * box['dims'][2] >= required_volume
    ]

    if not suitable_boxes:
        return None

    # Select smallest suitable box (minimize cost)
    optimal_box = min(suitable_boxes,
                     key=lambda b: b['dims'][0] * b['dims'][1] * b['dims'][2])

    actual_volume = optimal_box['dims'][0] * optimal_box['dims'][1] * optimal_box['dims'][2]
    utilization = total_volume / actual_volume

    return {
        'box_id': optimal_box['box_id'],
        'box_dims': optimal_box['dims'],
        'box_cost': optimal_box['cost'],
        'utilization_%': round(utilization * 100, 1)
    }

# Example
order = [
    {'sku': 'A', 'qty': 1, 'dims': (10, 8, 4)},
    {'sku': 'B', 'qty': 2, 'dims': (6, 6, 3)}
]

boxes = [
    {'box_id': 'Small', 'dims': (12, 10, 8), 'cost': 0.50},
    {'box_id': 'Medium', 'dims': (18, 14, 12), 'cost': 0.75},
    {'box_id': 'Large', 'dims': (24, 18, 16), 'cost': 1.00}
]

result = optimize_box_selection(order, boxes)
if result:
    print(f"Optimal box: {result['box_id']}")
    print(f"Utilization: {result['utilization_%']}%")
    print(f"Cost: ${result['box_cost']}")
```

### Packing Labor Requirements

```python
def packing_labor_requirements(orders_per_day, packing_rate_per_hour=40,
                               working_hours=8, multi_item_pct=0.60):
    """
    Calculate packing labor needs

    Parameters:
    - orders_per_day: Daily order volume
    - packing_rate_per_hour: Orders packed per person per hour
    - working_hours: Working hours per shift
    - multi_item_pct: % of orders with multiple items (slower to pack)

    Returns:
    - Packing labor requirements
    """

    # Adjust rate for multi-item orders
    multi_item_orders = orders_per_day * multi_item_pct
    single_item_orders = orders_per_day * (1 - multi_item_pct)

    # Multi-item takes ~1.5x longer
    equivalent_orders = single_item_orders + (multi_item_orders * 1.5)

    # Labor hours needed
    labor_hours = equivalent_orders / packing_rate_per_hour

    # Packers needed
    packers_needed = labor_hours / working_hours

    # Packing stations needed (assume 80% utilization)
    stations_needed = packers_needed / 0.80

    return {
        'orders_per_day': orders_per_day,
        'equivalent_orders': round(equivalent_orders, 0),
        'labor_hours_needed': round(labor_hours, 1),
        'packers_needed': round(packers_needed, 1),
        'packing_stations_needed': int(np.ceil(stations_needed))
    }

# Example
packing = packing_labor_requirements(
    orders_per_day=2000,
    packing_rate_per_hour=40,
    working_hours=8,
    multi_item_pct=0.60
)

print(f"Packers needed: {packing['packers_needed']}")
print(f"Packing stations: {packing['packing_stations_needed']}")
```

---

## Shipping Operations

### Carrier Selection & Rate Shopping

```python
def carrier_rate_shopping(order_weight_lbs, order_dims, destination_zip,
                         origin_zip, delivery_speed='ground'):
    """
    Compare carrier rates (simplified example)

    In practice, integrate with carrier APIs:
    - UPS API
    - FedEx API
    - USPS API
    """

    # Simplified rate tables (actual rates vary by contract, zone, etc.)
    rates = {
        'UPS': {
            'ground': 8.50 + (order_weight_lbs * 0.50),
            '2day': 15.00 + (order_weight_lbs * 0.75),
            'overnight': 35.00 + (order_weight_lbs * 1.50)
        },
        'FedEx': {
            'ground': 8.75 + (order_weight_lbs * 0.48),
            '2day': 14.50 + (order_weight_lbs * 0.70),
            'overnight': 32.00 + (order_weight_lbs * 1.40)
        },
        'USPS': {
            'ground': 7.50 + (order_weight_lbs * 0.45),
            '2day': 13.00 + (order_weight_lbs * 0.65),
            'overnight': 30.00 + (order_weight_lbs * 1.30)
        }
    }

    # Calculate dimensional weight
    dim_weight = (order_dims[0] * order_dims[1] * order_dims[2]) / 139
    billable_weight = max(order_weight_lbs, dim_weight)

    carrier_quotes = {}
    for carrier, speeds in rates.items():
        if delivery_speed in speeds:
            cost = speeds[delivery_speed]
            # Adjust for dimensional weight
            if billable_weight > order_weight_lbs:
                cost += (billable_weight - order_weight_lbs) * 0.50

            carrier_quotes[carrier] = round(cost, 2)

    # Find cheapest
    optimal_carrier = min(carrier_quotes, key=carrier_quotes.get)

    return {
        'carrier_quotes': carrier_quotes,
        'optimal_carrier': optimal_carrier,
        'optimal_cost': carrier_quotes[optimal_carrier],
        'billable_weight': round(billable_weight, 1)
    }

# Example
shipping = carrier_rate_shopping(
    order_weight_lbs=5.0,
    order_dims=(16, 12, 8),  # inches
    destination_zip='90210',
    origin_zip='10001',
    delivery_speed='ground'
)

print("Carrier Quotes:")
for carrier, cost in shipping['carrier_quotes'].items():
    print(f"  {carrier}: ${cost}")
print(f"\nOptimal: {shipping['optimal_carrier']} - ${shipping['optimal_cost']}")
```

### Manifest & Load Planning

```python
def manifest_optimization(orders_df, trailer_capacity=50000, carrier='UPS'):
    """
    Optimize order manifesting and trailer loading

    Parameters:
    - orders_df: DataFrame with ['order_id', 'weight', 'cube', 'carrier']
    - trailer_capacity: Trailer capacity (lbs or cube)
    - carrier: Filter orders by carrier
    """

    # Filter orders for carrier
    carrier_orders = orders_df[orders_df['carrier'] == carrier].copy()

    # Sort by weight (LIFO loading - heavy first)
    carrier_orders = carrier_orders.sort_values('weight', ascending=False)

    # Assign to trailers
    trailers = []
    current_trailer = {'orders': [], 'weight': 0, 'cube': 0}

    for _, order in carrier_orders.iterrows():
        # Check if fits in current trailer
        if current_trailer['weight'] + order['weight'] <= trailer_capacity:
            current_trailer['orders'].append(order['order_id'])
            current_trailer['weight'] += order['weight']
            current_trailer['cube'] += order['cube']
        else:
            # Start new trailer
            trailers.append(current_trailer)
            current_trailer = {
                'orders': [order['order_id']],
                'weight': order['weight'],
                'cube': order['cube']
            }

    # Add last trailer
    if current_trailer['orders']:
        trailers.append(current_trailer)

    # Summary
    manifest_summary = {
        'carrier': carrier,
        'total_orders': len(carrier_orders),
        'trailers_needed': len(trailers),
        'avg_weight_per_trailer': round(np.mean([t['weight'] for t in trailers]), 0),
        'avg_utilization_%': round(np.mean([t['weight']/trailer_capacity for t in trailers]) * 100, 1)
    }

    return trailers, manifest_summary

# Example
orders = pd.DataFrame({
    'order_id': [f'ORD_{i}' for i in range(1, 501)],
    'weight': np.random.uniform(10, 200, 500),
    'cube': np.random.uniform(1, 20, 500),
    'carrier': np.random.choice(['UPS', 'FedEx', 'USPS'], 500)
})

trailers, summary = manifest_optimization(orders, trailer_capacity=10000, carrier='UPS')
print(f"Carrier: {summary['carrier']}")
print(f"Trailers needed: {summary['trailers_needed']}")
print(f"Average utilization: {summary['avg_utilization_%']}%")
```

---

## Fulfillment Performance Metrics

### Key Performance Indicators (KPIs)

```python
def calculate_fulfillment_kpis(total_orders, orders_on_time, orders_accurate,
                              total_units, labor_hours, total_cost):
    """
    Calculate fulfillment KPIs

    Parameters:
    - total_orders: Orders processed
    - orders_on_time: Orders shipped on time
    - orders_accurate: Orders shipped accurately
    - total_units: Total units shipped
    - labor_hours: Total labor hours
    - total_cost: Total fulfillment cost
    """

    # On-time delivery rate
    on_time_rate = orders_on_time / total_orders

    # Order accuracy
    accuracy_rate = orders_accurate / total_orders

    # Units per labor hour
    units_per_hour = total_units / labor_hours

    # Orders per labor hour
    orders_per_hour = total_orders / labor_hours

    # Cost per order
    cost_per_order = total_cost / total_orders

    # Cost per unit
    cost_per_unit = total_cost / total_units

    kpis = {
        'On_Time_Delivery_%': round(on_time_rate * 100, 2),
        'Order_Accuracy_%': round(accuracy_rate * 100, 2),
        'Units_per_Labor_Hour': round(units_per_hour, 1),
        'Orders_per_Labor_Hour': round(orders_per_hour, 1),
        'Cost_per_Order': round(cost_per_order, 2),
        'Cost_per_Unit': round(cost_per_unit, 2)
    }

    return kpis

# Example
kpis = calculate_fulfillment_kpis(
    total_orders=10000,
    orders_on_time=9700,
    orders_accurate=9950,
    total_units=50000,
    labor_hours=1200,
    total_cost=120000
)

print("Fulfillment KPIs:")
for metric, value in kpis.items():
    print(f"  {metric}: {value}")
```

### Benchmark Targets

| Metric | Target | World-Class |
|--------|--------|-------------|
| Order Accuracy | 99%+ | 99.8%+ |
| On-Time Shipment | 95%+ | 99%+ |
| Units per Labor Hour | 100-150 | 200+ |
| Pick Accuracy | 99.5%+ | 99.9%+ |
| Cost per Order | $3-$8 | <$3 |
| Orders per Labor Hour | 15-25 | 30+ |
| Dock-to-Stock Time | <24 hrs | <4 hrs |

---

## Advanced Fulfillment Strategies

### Multi-Channel Fulfillment

**Strategies:**
- **Dedicated inventory**: Separate stock for each channel
- **Shared inventory**: Single pool, allocate dynamically
- **Hybrid**: Fast movers shared, slow movers dedicated

```python
def multi_channel_inventory_allocation(total_inventory, channels):
    """
    Allocate inventory across channels

    Parameters:
    - total_inventory: Total available inventory
    - channels: Dict with channel: {'demand_rate': X, 'priority': Y}

    Returns:
    - Inventory allocation by channel
    """

    # Calculate total demand
    total_demand = sum(ch['demand_rate'] for ch in channels.values())

    allocations = {}

    for channel, data in channels.items():
        # Proportional allocation based on demand
        base_allocation = (data['demand_rate'] / total_demand) * total_inventory

        # Adjust for priority (higher priority gets +10% buffer)
        priority_factor = 1 + ((data['priority'] - 2) * 0.10)  # Priority 1-3
        allocation = base_allocation * priority_factor

        allocations[channel] = round(allocation, 0)

    # Normalize to total inventory
    adjustment_factor = total_inventory / sum(allocations.values())
    allocations = {ch: round(qty * adjustment_factor, 0)
                  for ch, qty in allocations.items()}

    return allocations

# Example
channels = {
    'Retail_Stores': {'demand_rate': 1000, 'priority': 1},
    'Ecommerce': {'demand_rate': 800, 'priority': 2},
    'Wholesale': {'demand_rate': 500, 'priority': 3}
}

allocation = multi_channel_inventory_allocation(10000, channels)
print("Inventory Allocation:")
for channel, qty in allocation.items():
    print(f"  {channel}: {qty} units")
```

### Returns Processing

**Reverse Logistics Process:**
1. Customer initiates return
2. Generate return label
3. Receive at facility
4. Inspect and grade condition
5. Disposition (restock, liquidate, destroy)
6. Process refund/exchange

```python
def returns_processing_analysis(returns_per_day, inspection_rate_per_hour=50,
                               restock_pct=0.70, liquidate_pct=0.25,
                               destroy_pct=0.05):
    """
    Analyze returns processing requirements

    Parameters:
    - returns_per_day: Daily return volume
    - inspection_rate_per_hour: Returns inspected per person per hour
    - restock_pct, liquidate_pct, destroy_pct: Disposition percentages
    """

    # Labor for inspection
    labor_hours = returns_per_day / inspection_rate_per_hour

    # Disposition volumes
    restock_units = returns_per_day * restock_pct
    liquidate_units = returns_per_day * liquidate_pct
    destroy_units = returns_per_day * destroy_pct

    # Financial impact (example)
    # Assume average unit value $50
    avg_unit_value = 50

    # Restock: 90% recovery
    # Liquidate: 20% recovery
    # Destroy: 0% recovery

    value_recovered = (restock_units * avg_unit_value * 0.90 +
                      liquidate_units * avg_unit_value * 0.20)

    value_lost = (restock_units * avg_unit_value * 0.10 +
                 liquidate_units * avg_unit_value * 0.80 +
                 destroy_units * avg_unit_value)

    return {
        'returns_per_day': returns_per_day,
        'inspection_hours': round(labor_hours, 1),
        'restock_units': round(restock_units, 0),
        'liquidate_units': round(liquidate_units, 0),
        'destroy_units': round(destroy_units, 0),
        'value_recovered_daily': round(value_recovered, 0),
        'value_lost_daily': round(value_lost, 0),
        'recovery_rate_%': round((value_recovered / (value_recovered + value_lost)) * 100, 1)
    }

# Example
returns = returns_processing_analysis(
    returns_per_day=100,
    inspection_rate_per_hour=50
)

print(f"Returns per day: {returns['returns_per_day']}")
print(f"Inspection hours: {returns['inspection_hours']}")
print(f"Restock: {returns['restock_units']} units")
print(f"Value recovery rate: {returns['recovery_rate_%']}%")
```

---

## Tools & Libraries

### Warehouse Management Systems (WMS)

**Enterprise WMS:**
- **Manhattan Associates**: Tier 1 WMS, highly configurable
- **Blue Yonder (JDA)**: Warehouse management suite
- **SAP EWM**: Extended warehouse management
- **Oracle WMS**: Cloud-based warehouse management
- **Infor WMS**: Industry-specific solutions

**Mid-Market WMS:**
- **HighJump (Korber)**: Flexible WMS
- **NetSuite WMS**: Cloud ERP with WMS
- **Fishbowl**: Small to mid-market
- **3PL Central**: 3PL-focused WMS

**Open Source:**
- **Odoo**: Open-source ERP with WMS modules
- **iDempiere**: Open-source ERP/WMS

### Technology Stack

**Hardware:**
- RF scanners (Zebra, Honeywell)
- Mobile computers
- Voice picking systems (Vocollect, Honeywell)
- Label printers (Zebra ZT series)
- Automated sortation
- Pick-to-light / put-to-light

**Software Integrations:**
- OMS (Order Management System)
- TMS (Transportation Management)
- Carrier integrations (APIs)
- E-commerce platforms (Shopify, Magento)

---

## Common Challenges & Solutions

### Challenge: Order Accuracy Issues

**Problem:**
- Picking wrong items or quantities
- Shipping incorrect orders
- Customer complaints and returns

**Solutions:**
- Implement barcode scanning verification
- Use pick-to-light or voice picking
- Require dual verification for high-value items
- QA checkweigh at packing
- Root cause analysis on errors
- Picker training and accountability

### Challenge: Labor Productivity Variability

**Problem:**
- Inconsistent picker rates
- Some workers much slower than others
- Difficult to staff appropriately

**Solutions:**
- Implement labor management system (LMS)
- Track individual productivity
- Gamification and incentives
- Standard work procedures
- Ongoing training
- Ergonomic improvements

### Challenge: Peak Season Capacity

**Problem:**
- 2-3x normal volume during holidays
- Can't hire/train enough temporary workers
- Space constraints

**Solutions:**
- Start hiring/training 2+ months early
- Extended hours (add shifts)
- Simplify processes for temps
- Overflow to 3PL partners
- Automation (scales better than labor)
- Wave planning to spread work

### Challenge: Shipping Cost Escalation

**Problem:**
- Carrier rates increasing
- Dimensional weight charges
- Residential surcharges

**Solutions:**
- Rate shop across carriers
- Negotiate better contracts (volume commitments)
- Right-size packaging (avoid dim weight)
- Zone skipping (bulk to local carrier facility)
- Regional fulfillment centers (reduce zones)
- Customer incentives for slower shipping

### Challenge: Returns Volume

**Problem:**
- High return rates (especially apparel)
- Processing costs add up
- Inventory loss from damaged returns

**Solutions:**
- Better product descriptions (reduce wrong item returns)
- Free returns policy vs. cost trade-off
- Efficient returns processing
- Improve disposition logic (maximize restock %)
- Returns analytics to identify root causes
- Consider restocking fees for policy returns

---

## Output Format

### Fulfillment Operations Design Document

**Executive Summary:**
- Order volume and profile
- Recommended fulfillment strategy
- Labor requirements and costs
- Expected performance metrics

**Order Profile:**

| Metric | Value |
|--------|-------|
| Orders per Day (Avg) | 2,000 |
| Orders per Day (Peak) | 5,000 |
| Lines per Order | 5.2 |
| Units per Order | 6.8 |
| Order Types | 60% each, 30% case, 10% pallet |

**Recommended Picking Strategy:**
- **Fast movers (A items)**: Zone picking, dedicated forward pick area
- **Medium movers (B items)**: Batch picking, 10 orders per batch
- **Slow movers (C items)**: Discrete picking from reserve

**Labor Requirements:**

| Function | Staff Needed | Hours per Day | Shifts |
|----------|--------------|---------------|--------|
| Picking | 15 | 120 | 2 shifts |
| Packing | 10 | 80 | 2 shifts |
| Shipping | 5 | 40 | 2 shifts |
| Returns | 2 | 16 | 1 shift |
| **Total** | **32** | **256** | - |

**Technology Requirements:**
- WMS with wave planning and task management
- RF scanners for all pickers (20 units)
- Pack stations with scales and label printers (12 stations)
- Shipping manifesting software with carrier integrations
- Pick-to-light for fast movers (optional, $200K)

**Performance Targets:**

| KPI | Target | Current (if optimizing) |
|-----|--------|-------------------------|
| Order Accuracy | 99.5% | 97.2% |
| On-Time Shipment | 98% | 92% |
| Cost per Order | $5.50 | $7.20 |
| Units per Labor Hour | 150 | 110 |

**Implementation Plan:**
1. Months 1-2: WMS implementation and configuration
2. Month 3: Slotting optimization and layout changes
3. Month 4: Process rollout and training
4. Month 5: Ramp-up and optimization
5. Month 6: Full production, continuous improvement

---

## Questions to Ask

If you need more context:
1. What's the order volume? (daily average and peak)
2. What's the order profile? (lines/order, units/order, types)
3. What's the SKU count and velocity distribution?
4. What are the service requirements? (delivery speed, accuracy)
5. What's the current fulfillment process and pain points?
6. What technology is in place? (WMS, automation, RF scanning)
7. What's the labor availability and cost in your market?
8. What's the budget for improvements?

---

## Related Skills

- **warehouse-design**: Design facility layout for fulfillment
- **warehouse-slotting-optimization**: Optimize product placement
- **picker-routing-optimization**: Optimize pick paths
- **order-batching-optimization**: Batch order optimization
- **wave-planning-optimization**: Wave release optimization
- **ecommerce-fulfillment**: E-commerce specific strategies
- **omnichannel-fulfillment**: Multi-channel fulfillment
- **last-mile-delivery**: Final delivery optimization

---
name: pharmacy-supply-chain
description: When the user wants to optimize pharmacy supply chain operations, manage medication distribution, ensure pharmaceutical compliance, or handle controlled substances. Also use when the user mentions "pharmacy logistics," "drug distribution," "controlled substances," "340B program," "formulary management," "medication safety," "specialty pharmacy," "drug shortages," "DEA compliance," "pharmaceutical traceability," or "DSCSA compliance." For hospital materials management, see hospital-logistics. For clinical trial drugs, see clinical-trial-logistics.
---

# Pharmacy Supply Chain

You are an expert in pharmacy supply chain management and pharmaceutical distribution. Your goal is to ensure safe, compliant, cost-effective distribution of medications while maintaining regulatory compliance, managing controlled substances, and preventing drug shortages.

## Initial Assessment

Before optimizing pharmacy supply chain, understand:

1. **Pharmacy Type & Scope**
   - Pharmacy type? (hospital, retail, specialty, mail-order, 340B)
   - Number of locations? (single site vs. health system)
   - Patient volume and prescription volume?
   - Service lines? (inpatient, outpatient, infusion, specialty)

2. **Regulatory Environment**
   - DEA registration status and schedules handled?
   - State board of pharmacy requirements?
   - 340B program participation?
   - DSCSA (Drug Supply Chain Security Act) compliance?
   - Accreditations? (Joint Commission, ACHC, URAC)

3. **Inventory & Formulary**
   - Number of formulary drugs?
   - Inventory investment and turns?
   - High-cost specialty medications?
   - Controlled substance volume?
   - Generic vs. brand mix?

4. **Current Challenges**
   - Drug shortages impact?
   - Expiry and waste levels?
   - Controlled substance diversion risk?
   - 340B compliance gaps?
   - Distribution inefficiencies?

---

## Pharmacy Supply Chain Framework

### Pharmaceutical Distribution Channels

**1. Wholesaler/Distributor Model**
- Primary distribution channel (80-90% of drugs)
- Major wholesalers: McKesson, Cardinal Health, AmerisourceBergen
- Benefits: Broad selection, daily delivery, credit terms
- Considerations: Wholesaler fees, contract compliance

**2. Direct from Manufacturer**
- Specialty medications
- Limited distribution drugs
- High-volume generics (cost savings)
- Vaccines and biologics
- Benefits: Lower cost, better supply assurance
- Considerations: Minimum order quantities, less frequent delivery

**3. 340B Contract Pharmacy**
- Covered entities purchase at 340B ceiling price
- Dispense to eligible patients
- Significant cost savings
- Complex compliance requirements

**4. Specialty Pharmacy Distribution**
- High-cost, complex medications
- Limited distribution networks
- Patient support services
- Prior authorization and reimbursement support

---

## DSCSA Compliance & Traceability

### Drug Supply Chain Security Act (DSCSA)

**Requirements:**
- Product tracing at package level (serialization)
- Verification of product legitimacy
- Detection and response to suspect/illegitimate products
- Systems and processes for tracing

**Timeline:**
- 2023: Enhanced drug distribution security
- 2024: Full electronic, interoperable tracing (November 2024)

**Implementation:**

```python
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class ProductIdentifier:
    """
    DSCSA-compliant product identifier
    """
    gtin: str  # Global Trade Item Number (NDC in GTIN-14 format)
    serial_number: str
    lot_number: str
    expiration_date: datetime

    def to_serialized_string(self):
        """Convert to DSCSA format"""
        exp_date = self.expiration_date.strftime("%y%m%d")
        return f"(01){self.gtin}(21){self.serial_number}(10){self.lot_number}(17){exp_date}"

    @staticmethod
    def from_2d_barcode(barcode_string):
        """Parse 2D Data Matrix barcode"""
        import re

        patterns = {
            'gtin': r'\(01\)(\d{14})',
            'serial_number': r'\(21\)([A-Za-z0-9]+)',
            'lot_number': r'\(10\)([A-Za-z0-9]+)',
            'expiration_date': r'\(17\)(\d{6})'
        }

        data = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, barcode_string)
            if match:
                value = match.group(1)
                if field == 'expiration_date':
                    data[field] = datetime.strptime(value, "%y%m%d")
                else:
                    data[field] = value

        return ProductIdentifier(**data) if data else None

@dataclass
class TransactionInformation:
    """
    DSCSA Transaction Information (TI)
    """
    product_identifier: ProductIdentifier
    transaction_date: datetime
    ship_from: str  # Business name and address
    ship_to: str
    quantity: int

@dataclass
class TransactionHistory:
    """
    DSCSA Transaction History (TH) - full chain of ownership
    """
    transactions: List[TransactionInformation]

    def add_transaction(self, transaction: TransactionInformation):
        """Add transaction to history"""
        self.transactions.append(transaction)

    def verify_chain_of_custody(self):
        """Verify unbroken chain of custody"""
        if len(self.transactions) < 2:
            return True

        for i in range(len(self.transactions) - 1):
            current = self.transactions[i]
            next_trans = self.transactions[i + 1]

            # Verify ship_to of current matches ship_from of next
            if current.ship_to != next_trans.ship_from:
                return False

        return True

@dataclass
class TransactionStatement:
    """
    DSCSA Transaction Statement (TS) - attestation of legitimacy
    """
    product_identifier: ProductIdentifier
    statement_date: datetime
    authorized_entity: str
    attestation: str = "Product is legitimate and not counterfeit"

class DSCSATraceabilitySystem:
    """
    Manage DSCSA traceability and compliance
    """

    def __init__(self, business_name, dea_number, license_number):
        self.business_name = business_name
        self.dea_number = dea_number
        self.license_number = license_number
        self.inventory = {}
        self.transactions = []

    def receive_product(self, product_id: ProductIdentifier, quantity: int,
                        transaction_info: TransactionInformation,
                        transaction_history: TransactionHistory,
                        transaction_statement: TransactionStatement):
        """
        Receive product with DSCSA documentation
        """

        # Verify transaction history
        if not transaction_history.verify_chain_of_custody():
            raise ValueError("Chain of custody verification failed")

        # Verify product identifier matches
        if product_id.serial_number != transaction_info.product_identifier.serial_number:
            raise ValueError("Product identifier mismatch")

        # Store product in inventory with TI/TH/TS
        inventory_key = f"{product_id.gtin}-{product_id.serial_number}"

        self.inventory[inventory_key] = {
            'product_identifier': product_id,
            'quantity': quantity,
            'received_date': datetime.now(),
            'transaction_information': transaction_info,
            'transaction_history': transaction_history,
            'transaction_statement': transaction_statement,
            'status': 'in_stock'
        }

        return inventory_key

    def dispense_product(self, inventory_key: str, quantity: int,
                         patient_or_customer: str):
        """
        Dispense product to patient or transfer to another entity
        """

        if inventory_key not in self.inventory:
            raise ValueError(f"Product {inventory_key} not found in inventory")

        product_record = self.inventory[inventory_key]

        if product_record['quantity'] < quantity:
            raise ValueError(f"Insufficient quantity. Available: {product_record['quantity']}")

        # Create transaction record
        transaction = {
            'type': 'dispensed',
            'product_identifier': product_record['product_identifier'],
            'quantity': quantity,
            'date': datetime.now(),
            'recipient': patient_or_customer
        }

        self.transactions.append(transaction)

        # Update inventory
        product_record['quantity'] -= quantity

        if product_record['quantity'] == 0:
            product_record['status'] = 'dispensed'

        return transaction

    def verify_product(self, serialized_string: str):
        """
        Verify product legitimacy using DSCSA data
        """

        product_id = ProductIdentifier.from_2d_barcode(serialized_string)

        if not product_id:
            return {'verified': False, 'reason': 'Invalid product identifier'}

        inventory_key = f"{product_id.gtin}-{product_id.serial_number}"

        if inventory_key in self.inventory:
            product = self.inventory[inventory_key]
            return {
                'verified': True,
                'product': product_id,
                'status': product['status'],
                'received_date': product['received_date']
            }
        else:
            return {'verified': False, 'reason': 'Product not found in inventory'}

    def suspect_product_investigation(self, inventory_key: str, reason: str):
        """
        Quarantine and investigate suspect product
        """

        if inventory_key not in self.inventory:
            raise ValueError(f"Product {inventory_key} not found")

        product = self.inventory[inventory_key]
        product['status'] = 'quarantined'
        product['quarantine_reason'] = reason
        product['quarantine_date'] = datetime.now()

        # Notify FDA and trading partners as required
        investigation = {
            'product': product['product_identifier'],
            'reason': reason,
            'date': datetime.now(),
            'actions_taken': 'Product quarantined, investigation initiated'
        }

        return investigation

# Example usage
dscsa_system = DSCSATraceabilitySystem(
    business_name="Memorial Hospital Pharmacy",
    dea_number="FM1234563",
    license_number="PHY-12345"
)

# Create product identifier
product = ProductIdentifier(
    gtin="00300123456789",  # NDC in GTIN-14 format
    serial_number="ABC123XYZ789",
    lot_number="LOT2024-A",
    expiration_date=datetime(2026, 12, 31)
)

# Transaction information
trans_info = TransactionInformation(
    product_identifier=product,
    transaction_date=datetime.now(),
    ship_from="McKesson Corporation, 1234 Distributor Way",
    ship_to="Memorial Hospital Pharmacy, 5678 Hospital Blvd",
    quantity=100
)

# Transaction history
trans_history = TransactionHistory(transactions=[trans_info])

# Transaction statement
trans_statement = TransactionStatement(
    product_identifier=product,
    statement_date=datetime.now(),
    authorized_entity="McKesson Corporation"
)

# Receive product
inventory_key = dscsa_system.receive_product(
    product_id=product,
    quantity=100,
    transaction_info=trans_info,
    transaction_history=trans_history,
    transaction_statement=trans_statement
)

print(f"Product received: {inventory_key}")

# Verify product
barcode = product.to_serialized_string()
verification = dscsa_system.verify_product(barcode)
print(f"Product verified: {verification['verified']}")
```

---

## Controlled Substance Management

### DEA Controlled Substance Schedules

**Schedule I:** No accepted medical use, high abuse potential
- Examples: Heroin, LSD, marijuana (federally)
- Not typically in pharmacy

**Schedule II:** High abuse potential, accepted medical use
- Examples: Oxycodone, morphine, fentanyl, amphetamine, cocaine
- Requirements: Written prescription (with exceptions), no refills, secure storage

**Schedule III:** Moderate abuse potential
- Examples: Codeine/acetaminophen, ketamine, testosterone
- Requirements: Written or electronic Rx, up to 5 refills in 6 months

**Schedule IV:** Low abuse potential
- Examples: Alprazolam, diazepam, tramadol, zolpidem
- Requirements: Written or electronic Rx, up to 5 refills in 6 months

**Schedule V:** Lowest abuse potential
- Examples: Cough preparations with <200mg codeine
- Requirements: May have OTC availability in some states

### Perpetual Inventory System

```python
import pandas as pd
from datetime import datetime
from enum import Enum

class ControlledSubstanceSchedule(Enum):
    SCHEDULE_II = "C-II"
    SCHEDULE_III = "C-III"
    SCHEDULE_IV = "C-IV"
    SCHEDULE_V = "C-V"

class TransactionType(Enum):
    RECEIVED = "received"
    DISPENSED = "dispensed"
    WASTED = "wasted"
    RETURNED = "returned"
    TRANSFERRED = "transferred"
    DESTROYED = "destroyed"

class ControlledSubstanceManager:
    """
    Perpetual inventory system for controlled substances
    """

    def __init__(self, pharmacy_name, dea_number):
        self.pharmacy_name = pharmacy_name
        self.dea_number = dea_number
        self.inventory = {}
        self.transactions = []

    def add_drug(self, drug_id, drug_name, ndc, schedule, strength, form):
        """
        Add controlled substance to formulary
        """

        self.inventory[drug_id] = {
            'drug_id': drug_id,
            'drug_name': drug_name,
            'ndc': ndc,
            'schedule': schedule,
            'strength': strength,
            'form': form,
            'quantity_on_hand': 0,
            'unit_of_measure': 'each'
        }

    def record_transaction(self, drug_id, transaction_type, quantity,
                          performer, witness=None, metadata=None):
        """
        Record controlled substance transaction

        Parameters:
        - drug_id: Drug identifier
        - transaction_type: Type of transaction (TransactionType enum)
        - quantity: Quantity (positive for additions, negative for removals)
        - performer: Person performing transaction
        - witness: Witness required for Schedule II (optional for others)
        - metadata: Additional data (Rx number, patient, waste reason, etc.)
        """

        if drug_id not in self.inventory:
            raise ValueError(f"Drug {drug_id} not in inventory")

        drug = self.inventory[drug_id]

        # Schedule II requires witness for dispensing and waste
        if drug['schedule'] == ControlledSubstanceSchedule.SCHEDULE_II:
            if transaction_type in [TransactionType.DISPENSED, TransactionType.WASTED]:
                if not witness:
                    raise ValueError("Witness required for Schedule II dispensing/waste")

        # Create transaction record
        transaction = {
            'transaction_id': len(self.transactions) + 1,
            'timestamp': datetime.now(),
            'drug_id': drug_id,
            'drug_name': drug['drug_name'],
            'ndc': drug['ndc'],
            'schedule': drug['schedule'].value,
            'transaction_type': transaction_type.value,
            'quantity': quantity,
            'balance_before': drug['quantity_on_hand'],
            'balance_after': drug['quantity_on_hand'] + quantity,
            'performer': performer,
            'witness': witness,
            'metadata': metadata or {}
        }

        # Update inventory
        drug['quantity_on_hand'] += quantity

        if drug['quantity_on_hand'] < 0:
            raise ValueError(f"Negative inventory not allowed. Current: {drug['quantity_on_hand']}")

        # Store transaction
        self.transactions.append(transaction)

        return transaction

    def receive_order(self, drug_id, quantity, invoice_number, supplier, received_by):
        """
        Receive controlled substance order
        """

        return self.record_transaction(
            drug_id=drug_id,
            transaction_type=TransactionType.RECEIVED,
            quantity=quantity,
            performer=received_by,
            metadata={
                'invoice_number': invoice_number,
                'supplier': supplier
            }
        )

    def dispense_prescription(self, drug_id, quantity, rx_number, patient_id,
                             pharmacist, technician_witness=None):
        """
        Dispense controlled substance prescription
        """

        return self.record_transaction(
            drug_id=drug_id,
            transaction_type=TransactionType.DISPENSED,
            quantity=-quantity,  # Negative for removal
            performer=pharmacist,
            witness=technician_witness,
            metadata={
                'rx_number': rx_number,
                'patient_id': patient_id
            }
        )

    def waste_medication(self, drug_id, quantity, reason, pharmacist, witness):
        """
        Waste controlled substance (expired, damaged, etc.)
        """

        return self.record_transaction(
            drug_id=drug_id,
            transaction_type=TransactionType.WASTED,
            quantity=-quantity,
            performer=pharmacist,
            witness=witness,
            metadata={'waste_reason': reason}
        )

    def physical_count(self, drug_id, counted_quantity, counted_by, witness=None):
        """
        Perform physical count and reconcile with perpetual inventory
        """

        if drug_id not in self.inventory:
            raise ValueError(f"Drug {drug_id} not in inventory")

        drug = self.inventory[drug_id]
        system_quantity = drug['quantity_on_hand']
        discrepancy = counted_quantity - system_quantity

        count_record = {
            'drug_id': drug_id,
            'drug_name': drug['drug_name'],
            'ndc': drug['ndc'],
            'schedule': drug['schedule'].value,
            'count_date': datetime.now(),
            'system_quantity': system_quantity,
            'physical_count': counted_quantity,
            'discrepancy': discrepancy,
            'counted_by': counted_by,
            'witness': witness
        }

        return count_record

    def perpetual_inventory_report(self, drug_id=None, schedule=None):
        """
        Generate perpetual inventory report
        """

        transactions_df = pd.DataFrame(self.transactions)

        if drug_id:
            transactions_df = transactions_df[transactions_df['drug_id'] == drug_id]

        if schedule:
            transactions_df = transactions_df[transactions_df['schedule'] == schedule.value]

        return transactions_df

    def biennial_inventory_report(self):
        """
        DEA biennial (every 2 years) inventory report
        """

        inventory_list = []

        for drug_id, drug in self.inventory.items():
            inventory_list.append({
                'drug_name': drug['drug_name'],
                'ndc': drug['ndc'],
                'schedule': drug['schedule'].value,
                'strength': drug['strength'],
                'form': drug['form'],
                'quantity_on_hand': drug['quantity_on_hand']
            })

        inventory_df = pd.DataFrame(inventory_list)
        inventory_df = inventory_df.sort_values(['schedule', 'drug_name'])

        report = {
            'pharmacy_name': self.pharmacy_name,
            'dea_number': self.dea_number,
            'report_date': datetime.now(),
            'inventory': inventory_df
        }

        return report

# Example usage
cs_manager = ControlledSubstanceManager(
    pharmacy_name="Memorial Hospital Pharmacy",
    dea_number="FM1234563"
)

# Add controlled substances to formulary
cs_manager.add_drug(
    drug_id='OXY-5MG',
    drug_name='Oxycodone',
    ndc='00406-0505-62',
    schedule=ControlledSubstanceSchedule.SCHEDULE_II,
    strength='5mg',
    form='Tablet'
)

cs_manager.add_drug(
    drug_id='DIAZ-5MG',
    drug_name='Diazepam',
    ndc='00591-3445-01',
    schedule=ControlledSubstanceSchedule.SCHEDULE_IV,
    strength='5mg',
    form='Tablet'
)

# Receive order
cs_manager.receive_order(
    drug_id='OXY-5MG',
    quantity=500,
    invoice_number='INV-123456',
    supplier='McKesson',
    received_by='RPh John Smith'
)

# Dispense prescriptions
cs_manager.dispense_prescription(
    drug_id='OXY-5MG',
    quantity=30,
    rx_number='RX-001234',
    patient_id='PT-567890',
    pharmacist='RPh John Smith',
    technician_witness='Tech Jane Doe'  # Schedule II requires witness
)

# Waste expired medication
cs_manager.waste_medication(
    drug_id='OXY-5MG',
    quantity=10,
    reason='Expired',
    pharmacist='RPh John Smith',
    witness='RPh Mary Johnson'
)

# Physical count
count = cs_manager.physical_count(
    drug_id='OXY-5MG',
    counted_quantity=460,
    counted_by='RPh John Smith',
    witness='RPh Mary Johnson'
)

print(f"Physical Count - Expected: {count['system_quantity']}, Counted: {count['physical_count']}, Discrepancy: {count['discrepancy']}")

# Generate perpetual inventory report
report = cs_manager.perpetual_inventory_report(drug_id='OXY-5MG')
print("\nPerpetual Inventory Report:")
print(report[['timestamp', 'transaction_type', 'quantity', 'balance_after', 'performer']])
```

---

## 340B Program Compliance

### 340B Drug Pricing Program

**Overview:**
- Federal program requiring manufacturers to provide outpatient drugs at reduced prices
- Eligible entities: Safety-net providers (hospitals, FQHCs, Ryan White clinics)
- Savings can be 20-50% off wholesale acquisition cost
- Complex compliance requirements

**Key Compliance Requirements:**
1. Patient eligibility determination
2. Drug diversion prevention
3. Duplicate discount prevention
4. Accurate record-keeping
5. Contract pharmacy compliance

```python
class Patient340BEligibility:
    """
    Determine 340B patient eligibility
    """

    def __init__(self, covered_entity_id, entity_type):
        self.covered_entity_id = covered_entity_id
        self.entity_type = entity_type  # DSH, PED, CAH, FQHC, etc.

    def check_eligibility(self, patient_id, encounter_data):
        """
        Determine if patient is eligible for 340B

        Patient must meet all criteria:
        1. Established relationship with covered entity
        2. Received healthcare service from covered entity
        3. Covered entity has responsibility for care
        4. Drug prescribed by provider of covered entity
        5. Drug dispensed by eligible pharmacy
        """

        criteria = {
            'established_patient': self._check_established_patient(patient_id, encounter_data),
            'received_service': self._check_service_received(encounter_data),
            'provider_responsibility': self._check_provider_responsibility(encounter_data),
            'eligible_prescriber': self._check_eligible_prescriber(encounter_data),
            'eligible_pharmacy': self._check_eligible_pharmacy(encounter_data)
        }

        # All criteria must be met
        eligible = all(criteria.values())

        return {
            'eligible': eligible,
            'criteria_results': criteria,
            'reason': self._eligibility_reason(criteria) if not eligible else 'Meets all criteria'
        }

    def _check_established_patient(self, patient_id, encounter_data):
        """Check if patient has established relationship"""
        # Implementation would check patient registration, prior visits, etc.
        return encounter_data.get('established_patient', False)

    def _check_service_received(self, encounter_data):
        """Check if patient received service from covered entity"""
        return encounter_data.get('service_location') == self.covered_entity_id

    def _check_provider_responsibility(self, encounter_data):
        """Check if covered entity has responsibility for patient care"""
        return encounter_data.get('responsible_entity') == self.covered_entity_id

    def _check_eligible_prescriber(self, encounter_data):
        """Check if prescriber is employed/contracted by covered entity"""
        return encounter_data.get('prescriber_entity') == self.covered_entity_id

    def _check_eligible_pharmacy(self, encounter_data):
        """Check if pharmacy is covered entity or registered contract pharmacy"""
        pharmacy = encounter_data.get('dispensing_pharmacy')
        return pharmacy in [self.covered_entity_id] or \
               pharmacy in self._get_contract_pharmacies()

    def _get_contract_pharmacies(self):
        """Get list of registered contract pharmacies"""
        # Would query HRSA database
        return ['CONTRACT-PHARM-001', 'CONTRACT-PHARM-002']

    def _eligibility_reason(self, criteria):
        """Generate reason for ineligibility"""
        failed = [k for k, v in criteria.items() if not v]
        return f"Failed criteria: {', '.join(failed)}"

class Program340BManager:
    """
    Manage 340B program operations and compliance
    """

    def __init__(self, covered_entity_id):
        self.covered_entity_id = covered_entity_id
        self.eligibility_checker = Patient340BEligibility(covered_entity_id, 'DSH')
        self.purchases = []
        self.dispenses = []

    def process_prescription(self, rx_data, patient_id, encounter_data):
        """
        Process prescription and determine 340B eligibility
        """

        # Check patient eligibility
        eligibility = self.eligibility_checker.check_eligibility(patient_id, encounter_data)

        # Check drug eligibility (some drugs excluded from 340B)
        drug_eligible = self._check_drug_eligibility(rx_data['ndc'])

        # Determine if 340B pricing applies
        use_340b = eligibility['eligible'] and drug_eligible

        dispense_record = {
            'rx_number': rx_data['rx_number'],
            'patient_id': patient_id,
            'ndc': rx_data['ndc'],
            'quantity': rx_data['quantity'],
            'dispense_date': datetime.now(),
            '340b_eligible': use_340b,
            'eligibility_reason': eligibility['reason'],
            'acquisition_cost': self._get_acquisition_cost(rx_data['ndc'], use_340b)
        }

        self.dispenses.append(dispense_record)

        return dispense_record

    def _check_drug_eligibility(self, ndc):
        """
        Check if drug is eligible for 340B pricing

        Exclusions:
        - Orphan drugs for rare diseases (when used for that disease)
        - Drugs for cosmetic purposes
        - Drugs for fertility
        """
        # Simplified - would check against exclusion list
        return True

    def _get_acquisition_cost(self, ndc, use_340b):
        """Get drug acquisition cost (340B or WAC)"""
        # Simplified - would query pricing database
        wac_price = 100.00
        price_340b = wac_price * 0.60  # Typical 40% discount

        return price_340b if use_340b else wac_price

    def duplicate_discount_check(self, rx_data, patient_insurance):
        """
        Prevent duplicate discounts (340B + Medicaid rebate)

        Covered entities cannot receive 340B discount AND Medicaid rebate
        """

        is_medicaid = patient_insurance.get('payer_type') == 'Medicaid'
        is_340b = rx_data.get('340b_eligible', False)

        if is_medicaid and is_340b:
            # Options:
            # 1. Carve-out: Don't bill Medicaid, use 340B
            # 2. Carve-in: Bill Medicaid, don't use 340B
            return {
                'duplicate_risk': True,
                'recommendation': 'Use 340B pricing, do not bill Medicaid for rebate'
            }

        return {'duplicate_risk': False}

    def diversion_audit(self, start_date, end_date):
        """
        Audit for drug diversion (using 340B drugs for ineligible patients)
        """

        dispenses_df = pd.DataFrame(self.dispenses)

        # Filter date range
        dispenses_df = dispenses_df[
            (dispenses_df['dispense_date'] >= start_date) &
            (dispenses_df['dispense_date'] <= end_date)
        ]

        # Identify potential diversion
        ineligible_340b = dispenses_df[
            (dispenses_df['340b_eligible'] == False) &
            (dispenses_df['acquisition_cost'] < 100)  # Using 340B-priced inventory
        ]

        audit_results = {
            'total_dispenses': len(dispenses_df),
            '340b_dispenses': len(dispenses_df[dispenses_df['340b_eligible'] == True]),
            'potential_diversion_events': len(ineligible_340b),
            'diversion_details': ineligible_340b
        }

        return audit_results

# Example usage
program_340b = Program340BManager(covered_entity_id='CE-001')

# Process prescription
rx = {
    'rx_number': 'RX-123456',
    'ndc': '00406-0505-62',
    'quantity': 30
}

encounter = {
    'established_patient': True,
    'service_location': 'CE-001',
    'responsible_entity': 'CE-001',
    'prescriber_entity': 'CE-001',
    'dispensing_pharmacy': 'CE-001'
}

dispense = program_340b.process_prescription(rx, patient_id='PT-789', encounter_data=encounter)
print(f"340B Eligible: {dispense['340b_eligible']}")
print(f"Acquisition Cost: ${dispense['acquisition_cost']:.2f}")

# Duplicate discount check
insurance = {'payer_type': 'Medicaid', 'member_id': 'MCD123456'}
duplicate_check = program_340b.duplicate_discount_check(dispense, insurance)
print(f"Duplicate discount risk: {duplicate_check['duplicate_risk']}")
```

---

## Drug Shortage Management

### Shortage Response Framework

```python
import pandas as pd
from datetime import datetime, timedelta

class DrugShortageManager:
    """
    Manage drug shortages and implement mitigation strategies
    """

    def __init__(self, pharmacy_name):
        self.pharmacy_name = pharmacy_name
        self.shortages = []
        self.inventory = {}

    def declare_shortage(self, drug_id, drug_name, ndc, shortage_reason,
                        expected_duration_days, therapeutic_alternatives=None):
        """
        Declare drug shortage
        """

        shortage = {
            'shortage_id': f"SHORT-{len(self.shortages)+1}",
            'drug_id': drug_id,
            'drug_name': drug_name,
            'ndc': ndc,
            'declared_date': datetime.now(),
            'shortage_reason': shortage_reason,
            'expected_resolution': datetime.now() + timedelta(days=expected_duration_days),
            'status': 'active',
            'therapeutic_alternatives': therapeutic_alternatives or [],
            'mitigation_actions': []
        }

        self.shortages.append(shortage)

        return shortage

    def mitigation_strategy(self, shortage_id):
        """
        Develop mitigation strategy for shortage

        Strategies:
        1. Therapeutic substitution
        2. Dosage form modification
        3. Conservation (restrict to critical patients)
        4. Alternative suppliers
        5. Compounding
        """

        shortage = next((s for s in self.shortages if s['shortage_id'] == shortage_id), None)

        if not shortage:
            raise ValueError(f"Shortage {shortage_id} not found")

        strategies = []

        # Check for therapeutic alternatives
        if shortage['therapeutic_alternatives']:
            strategies.append({
                'strategy': 'therapeutic_substitution',
                'description': f"Substitute with: {', '.join(shortage['therapeutic_alternatives'])}",
                'priority': 1
            })

        # Check current inventory
        current_qty = self.inventory.get(shortage['drug_id'], {}).get('quantity', 0)
        days_supply = self._calculate_days_supply(shortage['drug_id'], current_qty)

        if days_supply < 7:
            strategies.append({
                'strategy': 'conservation',
                'description': 'Restrict to critical patients only (ICU, life-saving)',
                'priority': 1
            })

        # Alternative sourcing
        strategies.append({
            'strategy': 'alternative_sourcing',
            'description': 'Contact alternative distributors and direct manufacturers',
            'priority': 2
        })

        # Compounding option (if appropriate)
        if self._can_compound(shortage['drug_id']):
            strategies.append({
                'strategy': 'compounding',
                'description': 'Compound in-house or outsource to 503B',
                'priority': 3
            })

        shortage['mitigation_actions'] = strategies

        return strategies

    def _calculate_days_supply(self, drug_id, quantity):
        """Calculate days of supply remaining based on usage"""
        # Simplified - would use historical usage data
        avg_daily_usage = 10
        return quantity / avg_daily_usage if avg_daily_usage > 0 else 0

    def _can_compound(self, drug_id):
        """Check if drug can be compounded"""
        # Simplified - would check formulary
        return True

    def prioritize_patients(self, drug_id, patient_list):
        """
        Prioritize patients for shortage allocation

        Priority levels:
        1. Life-saving/critical care
        2. Prevent serious morbidity
        3. Symptomatic relief
        """

        prioritized = []

        for patient in patient_list:
            # Determine priority based on indication
            indication = patient.get('indication', '')

            if any(critical in indication.lower() for critical in ['sepsis', 'shock', 'arrest', 'seizure']):
                priority = 1
                priority_desc = 'Critical - Life-saving'
            elif any(serious in indication.lower() for serious in ['infection', 'pain-severe', 'surgery']):
                priority = 2
                priority_desc = 'High - Prevent serious morbidity'
            else:
                priority = 3
                priority_desc = 'Medium - Symptomatic relief'

            prioritized.append({
                'patient_id': patient['patient_id'],
                'indication': indication,
                'priority': priority,
                'priority_description': priority_desc,
                'prescriber': patient.get('prescriber'),
                'requested_quantity': patient.get('quantity')
            })

        # Sort by priority
        prioritized.sort(key=lambda x: x['priority'])

        return pd.DataFrame(prioritized)

    def shortage_report(self):
        """
        Generate active shortages report
        """

        active_shortages = [s for s in self.shortages if s['status'] == 'active']

        if not active_shortages:
            return None

        report = []

        for shortage in active_shortages:
            current_qty = self.inventory.get(shortage['drug_id'], {}).get('quantity', 0)
            days_supply = self._calculate_days_supply(shortage['drug_id'], current_qty)

            report.append({
                'shortage_id': shortage['shortage_id'],
                'drug_name': shortage['drug_name'],
                'ndc': shortage['ndc'],
                'declared_date': shortage['declared_date'],
                'expected_resolution': shortage['expected_resolution'],
                'reason': shortage['shortage_reason'],
                'current_inventory': current_qty,
                'days_supply': round(days_supply, 1),
                'alternatives': ', '.join(shortage['therapeutic_alternatives']),
                'mitigation_actions': len(shortage['mitigation_actions'])
            })

        return pd.DataFrame(report)

# Example usage
shortage_mgr = DrugShortageManager("Memorial Hospital Pharmacy")

# Add inventory
shortage_mgr.inventory['PROP-100MG'] = {'quantity': 50}

# Declare shortage
shortage = shortage_mgr.declare_shortage(
    drug_id='PROP-100MG',
    drug_name='Propofol 100mg/10mL',
    ndc='63323-0269-10',
    shortage_reason='Manufacturing delay at primary supplier',
    expected_duration_days=45,
    therapeutic_alternatives=['Etomidate', 'Ketamine']
)

print(f"Shortage declared: {shortage['shortage_id']}")

# Develop mitigation strategy
strategies = shortage_mgr.mitigation_strategy(shortage['shortage_id'])
print("\nMitigation Strategies:")
for strategy in strategies:
    print(f"  {strategy['priority']}. {strategy['strategy']}: {strategy['description']}")

# Prioritize patients
patients = [
    {'patient_id': 'PT-001', 'indication': 'Septic shock - ICU', 'prescriber': 'Dr. Smith', 'quantity': 20},
    {'patient_id': 'PT-002', 'indication': 'Elective surgery - General anesthesia', 'prescriber': 'Dr. Jones', 'quantity': 20},
    {'patient_id': 'PT-003', 'indication': 'Status epilepticus', 'prescriber': 'Dr. Brown', 'quantity': 10}
]

prioritized = shortage_mgr.prioritize_patients('PROP-100MG', patients)
print("\nPrioritized Patient Allocation:")
print(prioritized[['patient_id', 'indication', 'priority_description', 'requested_quantity']])
```

---

## Specialty Pharmacy Operations

### High-Cost Medication Management

```python
class SpecialtyPharmacyManager:
    """
    Manage specialty pharmacy operations for high-cost medications
    """

    def __init__(self, pharmacy_name):
        self.pharmacy_name = pharmacy_name
        self.specialty_drugs = {}
        self.prior_authorizations = {}

    def add_specialty_drug(self, drug_id, drug_name, ndc, indication,
                          unit_cost, limited_distribution=False,
                          rems_required=False, storage_requirements=None):
        """
        Add specialty drug to formulary
        """

        self.specialty_drugs[drug_id] = {
            'drug_id': drug_id,
            'drug_name': drug_name,
            'ndc': ndc,
            'indication': indication,
            'unit_cost': unit_cost,
            'limited_distribution': limited_distribution,
            'rems_required': rems_required,  # Risk Evaluation and Mitigation Strategy
            'storage_requirements': storage_requirements or 'Room temperature'
        }

    def prior_authorization(self, drug_id, patient_id, prescriber,
                           diagnosis_code, clinical_justification,
                           insurance_info):
        """
        Submit prior authorization request
        """

        pa_id = f"PA-{len(self.prior_authorizations)+1}"

        pa_request = {
            'pa_id': pa_id,
            'drug_id': drug_id,
            'patient_id': patient_id,
            'prescriber': prescriber,
            'diagnosis_code': diagnosis_code,
            'clinical_justification': clinical_justification,
            'insurance_info': insurance_info,
            'submission_date': datetime.now(),
            'status': 'pending',
            'determination': None,
            'determination_date': None
        }

        self.prior_authorizations[pa_id] = pa_request

        return pa_request

    def financial_assistance_screening(self, patient_id, drug_id, annual_income,
                                      insurance_coverage):
        """
        Screen for patient financial assistance programs
        """

        drug = self.specialty_drugs.get(drug_id)

        if not drug:
            raise ValueError(f"Drug {drug_id} not found")

        # Calculate estimated patient cost
        if insurance_coverage:
            copay = insurance_coverage.get('copay', drug['unit_cost'] * 0.30)
            coinsurance_pct = insurance_coverage.get('coinsurance', 0)
        else:
            copay = drug['unit_cost']
            coinsurance_pct = 100

        estimated_annual_cost = copay * 12  # Assuming monthly therapy

        # Assistance program eligibility (simplified)
        assistance_programs = []

        # Manufacturer copay assistance
        if insurance_coverage and copay > 50:
            assistance_programs.append({
                'program': 'Manufacturer Copay Card',
                'estimated_savings': min(copay * 0.80, 12000),  # Up to $12K/year typical
                'eligibility': 'Commercial insurance required'
            })

        # Patient assistance program (PAP)
        federal_poverty_level = 30000  # Simplified
        if annual_income < federal_poverty_level * 5:
            assistance_programs.append({
                'program': 'Manufacturer Patient Assistance Program (PAP)',
                'estimated_savings': drug['unit_cost'],
                'eligibility': f'Income <500% FPL (${federal_poverty_level * 5:,.0f})'
            })

        # Foundation assistance
        if estimated_annual_cost > annual_income * 0.10:  # >10% of income
            assistance_programs.append({
                'program': 'Independent Charitable Foundation',
                'estimated_savings': estimated_annual_cost * 0.50,
                'eligibility': 'Disease-specific, income-based'
            })

        return {
            'patient_id': patient_id,
            'drug_name': drug['drug_name'],
            'estimated_annual_cost': round(estimated_annual_cost, 2),
            'pct_of_income': round(estimated_annual_cost / annual_income * 100, 1) if annual_income > 0 else 0,
            'assistance_programs': assistance_programs
        }

# Example usage
specialty_pharm = SpecialtyPharmacyManager("Memorial Specialty Pharmacy")

# Add specialty drug
specialty_pharm.add_specialty_drug(
    drug_id='HUMIRA-40MG',
    drug_name='Adalimumab (Humira)',
    ndc='00074-4339-02',
    indication='Rheumatoid Arthritis, Crohn\'s Disease',
    unit_cost=6000,
    limited_distribution=False,
    rems_required=False,
    storage_requirements='Refrigerated 2-8°C'
)

# Prior authorization
pa = specialty_pharm.prior_authorization(
    drug_id='HUMIRA-40MG',
    patient_id='PT-123456',
    prescriber='Dr. Williams',
    diagnosis_code='M05.79',  # Rheumatoid arthritis
    clinical_justification='Failed methotrexate and sulfasalazine. Significant disease activity.',
    insurance_info={'payer': 'Blue Cross', 'plan': 'PPO', 'member_id': 'BC123456'}
)

print(f"PA submitted: {pa['pa_id']}, Status: {pa['status']}")

# Financial assistance screening
insurance = {'copay': 500, 'coinsurance': 20}
assistance = specialty_pharm.financial_assistance_screening(
    patient_id='PT-123456',
    drug_id='HUMIRA-40MG',
    annual_income=45000,
    insurance_coverage=insurance
)

print(f"\nEstimated annual cost: ${assistance['estimated_annual_cost']:,.2f}")
print(f"Percent of income: {assistance['pct_of_income']}%")
print(f"\nAssistance programs available: {len(assistance['assistance_programs'])}")
for program in assistance['assistance_programs']:
    print(f"  - {program['program']}: Up to ${program['estimated_savings']:,.0f}")
```

---

## Pharmacy Performance Metrics

### Key Performance Indicators

```python
def calculate_pharmacy_kpis(rx_data_df, inventory_data_df, cs_data_df=None):
    """
    Calculate pharmacy supply chain KPIs

    Parameters:
    - rx_data_df: Prescription dispensing data
    - inventory_data_df: Inventory positions
    - cs_data_df: Controlled substance data (optional)
    """

    kpis = {}

    # Fill rate / Service level
    if 'filled' in rx_data_df.columns:
        kpis['fill_rate'] = (rx_data_df['filled'].sum() / len(rx_data_df) * 100)

    # Inventory turns
    if all(col in inventory_data_df.columns for col in ['annual_usage', 'avg_inventory_value']):
        total_usage_value = inventory_data_df['annual_usage'].sum()
        total_avg_inventory = inventory_data_df['avg_inventory_value'].sum()
        kpis['inventory_turns'] = total_usage_value / total_avg_inventory if total_avg_inventory > 0 else 0

    # Expiry waste rate
    if 'expired' in inventory_data_df.columns:
        expired_value = inventory_data_df[inventory_data_df['expired'] == True]['inventory_value'].sum()
        total_value = inventory_data_df['inventory_value'].sum()
        kpis['expiry_waste_pct'] = (expired_value / total_value * 100) if total_value > 0 else 0

    # Generic dispensing rate (cost savings)
    if 'generic' in rx_data_df.columns:
        kpis['generic_dispensing_rate'] = (rx_data_df['generic'].sum() / len(rx_data_df) * 100)

    # Prescription turnaround time
    if 'turnaround_minutes' in rx_data_df.columns:
        kpis['avg_turnaround_minutes'] = rx_data_df['turnaround_minutes'].mean()

    # 340B capture rate (if applicable)
    if '340b_eligible' in rx_data_df.columns and '340b_used' in rx_data_df.columns:
        eligible = rx_data_df['340b_eligible'].sum()
        used = rx_data_df['340b_used'].sum()
        kpis['340b_capture_rate'] = (used / eligible * 100) if eligible > 0 else 0

    # Controlled substance accuracy (if applicable)
    if cs_data_df is not None and not cs_data_df.empty:
        if 'discrepancy' in cs_data_df.columns:
            kpis['cs_inventory_accuracy'] = (
                (len(cs_data_df[cs_data_df['discrepancy'] == 0]) / len(cs_data_df) * 100)
            )

    # DSCSA compliance rate
    if 'dscsa_compliant' in rx_data_df.columns:
        kpis['dscsa_compliance_rate'] = (rx_data_df['dscsa_compliant'].sum() / len(rx_data_df) * 100)

    # Format KPIs
    for key in kpis:
        if 'rate' in key or 'pct' in key or 'accuracy' in key:
            kpis[key] = round(kpis[key], 2)
        elif 'turns' in key:
            kpis[key] = round(kpis[key], 2)
        else:
            kpis[key] = round(kpis[key], 1)

    return kpis

# Example data
rx_data = pd.DataFrame({
    'rx_number': range(1, 1001),
    'filled': [True] * 980 + [False] * 20,
    'generic': np.random.choice([True, False], 1000, p=[0.85, 0.15]),
    'turnaround_minutes': np.random.normal(20, 5, 1000),
    '340b_eligible': np.random.choice([True, False], 1000, p=[0.40, 0.60]),
    '340b_used': np.random.choice([True, False], 1000, p=[0.35, 0.65]),
    'dscsa_compliant': np.random.choice([True, False], 1000, p=[0.98, 0.02])
})

# Fix 340B logic (can only use if eligible)
rx_data.loc[~rx_data['340b_eligible'], '340b_used'] = False

inventory_data = pd.DataFrame({
    'drug_id': range(1, 501),
    'annual_usage': np.random.randint(1000, 50000, 500),
    'avg_inventory_value': np.random.randint(500, 10000, 500),
    'inventory_value': np.random.randint(500, 10000, 500),
    'expired': np.random.choice([True, False], 500, p=[0.02, 0.98])
})

kpis = calculate_pharmacy_kpis(rx_data, inventory_data)

print("Pharmacy Supply Chain KPIs:")
for metric, value in kpis.items():
    suffix = '%' if any(x in metric for x in ['rate', 'pct', 'accuracy']) else ''
    print(f"  {metric}: {value}{suffix}")
```

---

## Tools & Libraries

### Pharmacy Management Systems

**ERP/Pharmacy Systems:**
- **EPIC Willow Pharmacy**: Integrated with Epic EHR
- **Cerner PharmNet**: Pharmacy management module
- **Omnicell**: Automated dispensing cabinets
- **BD Pyxis**: Medication management system
- **McKesson Pharmacy Systems**: STAR, EnterpriseRx
- **QS/1**: Independent pharmacy management

**Controlled Substance Tracking:**
- **Omnicell ControlledRx**: CS management
- **BD Pyxis CII Safe**: Controlled substance vault
- **Kit Check**: Medication tracking and diversion monitoring
- **Protenus**: AI-powered drug diversion detection

**340B Management:**
- **Kalderos**: 340B claim identification
- **Macro Helix**: 340B program management
- **Verity Solutions**: 340B split-billing
- **RxStrategies**: 340B optimization

**DSCSA Compliance:**
- **TraceLink**: DSCSA compliance platform
- **SAP Information Collaboration Hub for Life Sciences**
- **rfxcel**: Serialization and traceability
- **Systech UniSecure**: Product authentication

### Python Libraries

**Data Analysis:**
- `pandas`: Data manipulation
- `numpy`: Numerical analysis
- `scipy`: Statistical functions

**Regulatory Compliance:**
- `python-barcode`: Barcode generation (NDC)
- `qrcode`: 2D barcode generation
- `hashlib`: Data integrity verification

**Optimization:**
- `pulp`: Linear programming (inventory optimization)
- `scipy.optimize`: Optimization algorithms

**Visualization:**
- `matplotlib`, `seaborn`: Charts and graphs
- `plotly`: Interactive dashboards

---

## Common Challenges & Solutions

### Challenge: DSCSA Compliance by November 2024

**Problem:**
- Electronic, interoperable tracing required
- Legacy systems not compliant
- Trading partner integration
- Serialization at package level

**Solutions:**
- Implement DSCSA-compliant software platform
- Test trading partner connections early
- Staff training on scanning requirements
- Develop suspect product investigation protocols
- Regular compliance audits
- Join industry pilot programs

### Challenge: Controlled Substance Diversion

**Problem:**
- Employee diversion risk
- Documentation gaps
- Inventory discrepancies
- Regulatory penalties

**Solutions:**
- Automated perpetual inventory system
- Two-person verification for Schedule II
- Regular random audits and cycle counts
- Analytics to detect unusual patterns
- Background checks and monitoring
- Clear policies and disciplinary actions
- Confidential reporting mechanisms

### Challenge: 340B Compliance and Audits

**Problem:**
- Complex patient eligibility rules
- Contract pharmacy compliance
- Duplicate discount prevention
- HRSA audits

**Solutions:**
- Automated patient eligibility system
- Electronic health record integration
- Split-billing for Medicaid
- Regular internal compliance audits
- Staff training on eligibility criteria
- Detailed documentation and audit trails
- Work with 340B consultant/expert

### Challenge: Drug Shortages

**Problem:**
- Critical medications unavailable
- Patient care impact
- Therapeutic substitution complexity
- Cost increases from alternatives

**Solutions:**
- Multi-source purchasing strategy
- Early warning monitoring (FDA shortage list)
- Therapeutic substitution protocols
- Conservation strategies for critical patients
- Compounding alternatives (where appropriate)
- Communication with prescribers
- Group purchasing organization leverage

### Challenge: Specialty Medication Costs

**Problem:**
- Extremely high acquisition costs
- Prior authorization delays
- Patient affordability
- Waste from failed PA or non-compliance

**Solutions:**
- Proactive prior authorization
- Financial assistance program navigation
- Buy-and-bill vs. white bagging analysis
- Limited dispensing quantities initially
- Patient adherence support services
- Manufacturer assistance program enrollment
- Specialty pharmacy accreditation

### Challenge: Expiry and Waste

**Problem:**
- Short-dated products
- Slow-moving items
- Storage errors
- Compounded medications

**Solutions:**
- FEFO (first-expired, first-out) enforcement
- Automated expiry alerts
- Right-sizing inventory (data-driven PAR levels)
- Vendor return programs
- Transfer slow-movers between locations
- Just-in-time ordering for slow items
- Beyond-use date (BUD) optimization for compounding

---

## Output Format

### Pharmacy Supply Chain Report

**Executive Summary:**
- Pharmacy overview and scope
- Key performance metrics vs. benchmarks
- Major compliance initiatives
- Financial impact and opportunities

**Inventory Optimization:**

| Drug Category | Items | Inventory Value | Annual Turns | Days on Hand | Expiry Rate | Optimization Opportunity |
|---------------|-------|----------------|--------------|--------------|-------------|--------------------------|
| Antibiotics | 145 | $125,000 | 15.2 | 24 | 0.5% | Reduce safety stock |
| Specialty | 42 | $850,000 | 8.1 | 45 | 1.2% | Buy-and-bill vs. white bag |
| Controlled Substances | 38 | $45,000 | 24.3 | 15 | 0.1% | Appropriate |
| Generic Oral | 1,250 | $180,000 | 18.5 | 20 | 1.8% | Standardization |

**340B Program Performance:**

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Eligible Prescriptions | 2,450/month | - | - |
| 340B Capture Rate | 92.3% | 95% | ⚠ |
| Annual 340B Savings | $1,850,000 | $2,000,000 | ⚠ |
| Contract Pharmacy Compliance | 98.5% | 100% | ⚠ |
| Duplicate Discount Events | 0 | 0 | ✓ |

**Controlled Substance Compliance:**

| Schedule | Items | Perpetual Inv Accuracy | Physical Count Frequency | Discrepancies YTD | Status |
|----------|-------|------------------------|-------------------------|-------------------|--------|
| Schedule II | 15 | 99.8% | Daily | 2 | ✓ |
| Schedule III | 12 | 99.5% | Weekly | 3 | ✓ |
| Schedule IV | 23 | 99.2% | Weekly | 5 | ✓ |

**Active Drug Shortages:**

| Drug Name | NDC | Declared Date | Current Stock | Days Supply | Mitigation Strategy | Status |
|-----------|-----|---------------|---------------|-------------|---------------------|--------|
| Propofol 100mg | 63323-0269-10 | 2024-01-15 | 50 vials | 5 days | Alternative sourcing, therapeutic sub | Critical |
| Cefazolin 1g | 00409-1964-50 | 2024-02-01 | 200 vials | 20 days | Conservation protocol | Monitoring |

**Key Performance Indicators:**

| Metric | Current | Target | Trend |
|--------|---------|--------|-------|
| Fill Rate | 98.2% | 98% | ✓ |
| Inventory Turns | 12.5 | 12.0 | ✓ |
| Generic Dispensing Rate | 89.3% | 85% | ✓ |
| Expiry Waste % | 1.2% | <1.5% | ✓ |
| 340B Capture Rate | 92.3% | 95% | ⚠ |
| DSCSA Compliance | 98.1% | 100% | ⚠ |
| Controlled Substance Accuracy | 99.5% | 99% | ✓ |

**Action Items:**
1. Improve 340B capture rate through enhanced eligibility screening
2. Complete DSCSA compliance for remaining 1.9% of transactions
3. Implement automated shortage monitoring system
4. Reduce specialty medication waste through limited initial fills
5. Optimize inventory for slow-moving antibiotics

---

## Questions to Ask

If you need more context:

1. What type of pharmacy? (hospital, retail, specialty, mail-order)
2. Are you a 340B covered entity?
3. What's your DSCSA compliance status?
4. What controlled substance schedules do you handle?
5. What's the current inventory investment and turnover?
6. Are you experiencing specific drug shortages?
7. What pharmacy management system is in use?
8. What are the biggest compliance concerns?
9. Do you dispense specialty medications?
10. What wholesaler/distributor relationships do you have?

---

## Related Skills

- **hospital-logistics**: Hospital materials management
- **medical-device-distribution**: Medical device logistics
- **clinical-trial-logistics**: Clinical trial supply chain
- **inventory-optimization**: Inventory optimization techniques
- **demand-forecasting**: Forecasting for inventory planning
- **compliance-management**: Regulatory compliance systems
- **track-and-trace**: Product traceability and serialization
- **value-analysis**: Value analysis and cost reduction
- **quality-management**: Quality management systems
