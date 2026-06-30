---
name: omnichannel-fulfillment
description: "When the user wants to optimize omnichannel fulfillment, manage buy-online-pickup-in-store (BOPIS), ship-from-store, or multi-channel inventory. Also use when the user mentions \"omnichannel,\" \"BOPIS,\" \"click and collect,\" \"ship from store,\" \"endless aisle,\" \"unified commerce,\" \"store fulfillment,\" or \"buy online return in store.\" For pure e-commerce, see ecommerce-fulfillment. For last-mile delivery, see last-mile-delivery."
---

# Omnichannel Fulfillment

You are an expert in omnichannel fulfillment strategy and execution. Your goal is to help retailers seamlessly fulfill orders across all channels (stores, online, mobile) using all available inventory nodes (stores, DCs, suppliers) to maximize sales, minimize costs, and deliver exceptional customer experiences.

## Initial Assessment

Before designing omnichannel fulfillment, understand:

1. **Channel Mix**
   - What sales channels exist? (stores, website, mobile app, marketplace)
   - Current split of sales by channel?
   - Which channels are growing vs. declining?
   - Cross-channel customer behavior? (research online, buy in store)

2. **Fulfillment Capabilities**
   - Can stores fulfill online orders? (ship-from-store capability)
   - BOPIS/curbside pickup available?
   - Store inventory visibility to online?
   - DC network size and locations?
   - Order management system (OMS) in place?

3. **Current Performance**
   - Online delivery time (2-day, 3-day, week+)?
   - BOPIS adoption rate?
   - Ship-from-store percentage of online orders?
   - Split shipment rate?
   - Fulfillment cost per order by channel?

4. **Business Goals**
   - Priority: Speed, cost, experience, or balance?
   - Target ship-from-store percentage?
   - Inventory turn goals?
   - Customer experience priorities?
   - Profitability by channel?

---

## Omnichannel Fulfillment Framework

### Core Fulfillment Models

**1. Buy Online, Pickup In Store (BOPIS)**
- Customer orders online, picks up at store
- Benefits: Fast (same-day), low cost, drives store traffic
- Challenges: Inventory accuracy, picking efficiency, customer wait time

**2. Ship from Store (SFS)**
- Stores act as mini-fulfillment centers
- Benefits: Faster delivery, reduces DC load, utilizes store inventory
- Challenges: Store labor, packaging supplies, competing with retail operations

**3. Ship from DC**
- Traditional centralized fulfillment
- Benefits: Efficient picking, lower unit costs, inventory concentration
- Challenges: Longer delivery times, higher transportation costs

**4. Marketplace/Dropship**
- Supplier ships directly to customer
- Benefits: No inventory investment, extended assortment
- Challenges: Quality control, delivery time variability, customer experience

**5. Endless Aisle**
- Store orders out-of-stock items for customer
- Benefits: Reduces lost sales, enhances customer experience
- Challenges: Margin pressure, complexity

**6. Reserve Online, Try In Store**
- Customer reserves items online, tries in store before buying
- Benefits: Reduces returns, drives traffic
- Challenges: Inventory holding, operational complexity

---

## Omnichannel Fulfillment Optimization

### Order Routing & Sourcing Logic

**Intelligent Order Routing:**

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class OmnichannelOrderRouter:
    """
    Intelligent order routing for omnichannel fulfillment

    Routes orders to optimal fulfillment location based on:
    - Inventory availability
    - Customer proximity
    - Delivery speed
    - Fulfillment cost
    - Store/DC capacity
    """

    def __init__(self, fulfillment_nodes, shipping_matrix, cost_matrix):
        """
        Parameters:
        - fulfillment_nodes: DataFrame with node info (stores, DCs)
          columns: ['node_id', 'type', 'lat', 'lon', 'capacity', 'inventory']
        - shipping_matrix: DataFrame with shipping times/costs
        - cost_matrix: Dict with cost per unit by fulfillment type
        """
        self.nodes = fulfillment_nodes
        self.shipping = shipping_matrix
        self.costs = cost_matrix

    def calculate_fulfillment_score(self, order, node):
        """
        Score a fulfillment node for an order

        Lower score = better option
        Balances speed, cost, and inventory health
        """

        # Check inventory availability
        if node['inventory'] < order['quantity']:
            return float('inf')  # Cannot fulfill

        # Calculate distance/delivery time
        distance = self._calculate_distance(
            order['customer_lat'], order['customer_lon'],
            node['lat'], node['lon']
        )

        # Delivery speed score (0-100)
        if node['type'] == 'store' and order['method'] == 'BOPIS':
            delivery_days = 0  # Same day pickup
        elif node['type'] == 'store':
            delivery_days = 1 if distance < 50 else 2
        else:  # DC
            delivery_days = 2 if distance < 500 else 3

        speed_score = delivery_days * 10

        # Cost score (0-100)
        if node['type'] == 'store' and order['method'] == 'BOPIS':
            fulfillment_cost = self.costs['bopis']
        elif node['type'] == 'store':
            fulfillment_cost = self.costs['ship_from_store'] + (distance * 0.5)
        else:
            fulfillment_cost = self.costs['ship_from_dc'] + (distance * 0.3)

        cost_score = fulfillment_cost

        # Inventory health score (0-100)
        # Prefer fulfilling from locations with excess inventory
        weeks_of_supply = node['inventory'] / (node['sales_per_week'] + 0.1)

        if weeks_of_supply > 8:
            inventory_score = -20  # Incentivize using excess inventory
        elif weeks_of_supply < 2:
            inventory_score = 50  # Penalize low inventory
        else:
            inventory_score = 0

        # Capacity score
        capacity_utilization = node['orders_today'] / node['capacity']
        if capacity_utilization > 0.9:
            capacity_score = 30  # Penalize overloaded nodes
        else:
            capacity_score = 0

        # Weighted total score
        total_score = (
            speed_score * 0.4 +
            cost_score * 0.3 +
            inventory_score * 0.2 +
            capacity_score * 0.1
        )

        return total_score

    def route_order(self, order):
        """
        Route order to optimal fulfillment location

        Returns: Best fulfillment node and score
        """

        # Score all eligible nodes
        scores = []
        for idx, node in self.nodes.iterrows():
            score = self.calculate_fulfillment_score(order, node)
            scores.append({
                'node_id': node['node_id'],
                'node_type': node['type'],
                'score': score,
                'delivery_cost': self._estimate_cost(order, node),
                'delivery_days': self._estimate_days(order, node)
            })

        # Select best option
        scores_df = pd.DataFrame(scores)
        best_option = scores_df.loc[scores_df['score'].idxmin()]

        return best_option

    def route_multi_item_order(self, order_items, allow_split=True):
        """
        Route multi-item order

        Decide whether to split shipment or fulfill from single location
        """

        # Try single location fulfillment first
        single_location_options = []

        for idx, node in self.nodes.iterrows():
            can_fulfill_all = all(
                node['inventory_by_sku'].get(item['sku'], 0) >= item['quantity']
                for item in order_items
            )

            if can_fulfill_all:
                total_score = sum(
                    self.calculate_fulfillment_score(
                        {'sku': item['sku'], 'quantity': item['quantity'],
                         'customer_lat': order_items[0]['customer_lat'],
                         'customer_lon': order_items[0]['customer_lon'],
                         'method': order_items[0]['method']},
                        node
                    )
                    for item in order_items
                )

                single_location_options.append({
                    'node_id': node['node_id'],
                    'total_score': total_score,
                    'split': False
                })

        # Try split shipment if allowed
        split_options = []
        if allow_split:
            item_routings = []
            for item in order_items:
                best_node = self.route_order(item)
                item_routings.append(best_node)

            # Calculate split shipment penalty
            unique_nodes = len(set(r['node_id'] for r in item_routings))
            split_penalty = (unique_nodes - 1) * 15  # Penalize splits

            total_split_score = sum(r['score'] for r in item_routings) + split_penalty

            split_options.append({
                'routings': item_routings,
                'total_score': total_split_score,
                'split': True,
                'num_shipments': unique_nodes
            })

        # Compare single vs. split
        all_options = single_location_options + split_options
        if not all_options:
            return {'error': 'Cannot fulfill order'}

        best_option = min(all_options, key=lambda x: x['total_score'])

        return best_option

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points (miles)"""
        from math import radians, sin, cos, sqrt, atan2

        R = 3959  # Earth radius in miles

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return R * c

    def _estimate_cost(self, order, node):
        """Estimate fulfillment cost"""
        if node['type'] == 'store' and order.get('method') == 'BOPIS':
            return self.costs['bopis']
        elif node['type'] == 'store':
            return self.costs['ship_from_store']
        else:
            return self.costs['ship_from_dc']

    def _estimate_days(self, order, node):
        """Estimate delivery days"""
        distance = self._calculate_distance(
            order['customer_lat'], order['customer_lon'],
            node['lat'], node['lon']
        )

        if node['type'] == 'store' and order.get('method') == 'BOPIS':
            return 0
        elif distance < 50:
            return 1
        elif distance < 200:
            return 2
        else:
            return 3

# Example usage
fulfillment_nodes = pd.DataFrame({
    'node_id': ['DC1', 'Store_101', 'Store_102', 'Store_103'],
    'type': ['dc', 'store', 'store', 'store'],
    'lat': [40.7128, 40.7580, 40.6782, 40.7489],
    'lon': [-74.0060, -73.9855, -73.9442, -73.9680],
    'capacity': [5000, 50, 50, 50],
    'inventory': [10000, 500, 300, 450],
    'sales_per_week': [2000, 100, 80, 90],
    'orders_today': [2500, 25, 30, 20]
})

cost_matrix = {
    'bopis': 2.50,
    'ship_from_store': 8.50,
    'ship_from_dc': 6.00
}

router = OmnichannelOrderRouter(fulfillment_nodes, None, cost_matrix)

# Route single order
order = {
    'sku': 'SKU123',
    'quantity': 2,
    'customer_lat': 40.7589,
    'customer_lon': -73.9851,
    'method': 'ship'
}

best_fulfillment = router.route_order(order)
print(f"Route order to: {best_fulfillment['node_id']}")
print(f"Estimated delivery: {best_fulfillment['delivery_days']} days")
print(f"Estimated cost: ${best_fulfillment['delivery_cost']:.2f}")
```

### BOPIS Optimization

**Store Pickup Operations:**

```python
class BOPISOptimizer:
    """
    Optimize Buy Online Pickup In Store (BOPIS) operations

    Focus on:
    - Inventory allocation (reserve for online vs. in-store)
    - Picking efficiency
    - Customer wait time reduction
    - Store capacity management
    """

    def __init__(self, store_config):
        self.store = store_config

    def allocate_inventory(self, sku, available_qty, online_demand_forecast,
                          instore_demand_forecast):
        """
        Allocate inventory between online and in-store channels

        Prevent online channel from selling out store inventory
        """

        total_demand = online_demand_forecast + instore_demand_forecast

        if available_qty >= total_demand:
            # Plenty of inventory - allocate all
            return {
                'online_allocation': online_demand_forecast,
                'instore_allocation': instore_demand_forecast,
                'allocation_strategy': 'Full allocation'
            }

        # Scarce inventory - prioritize based on margin and strategy
        online_margin_per_unit = self.store['online_margin']
        instore_margin_per_unit = self.store['instore_margin']

        # Also consider strategic value (online drives future loyalty)
        online_strategic_value = 1.2  # 20% premium for online
        instore_strategic_value = 1.0

        online_value = online_margin_per_unit * online_strategic_value
        instore_value = instore_margin_per_unit * instore_strategic_value

        # Allocate proportionally to value
        online_pct = online_value / (online_value + instore_value)

        online_allocation = min(
            int(available_qty * online_pct),
            online_demand_forecast
        )
        instore_allocation = available_qty - online_allocation

        return {
            'online_allocation': online_allocation,
            'instore_allocation': instore_allocation,
            'allocation_strategy': 'Value-based allocation',
            'online_pct': online_pct
        }

    def optimize_pickup_scheduling(self, orders, picker_capacity_per_hour=10):
        """
        Schedule BOPIS orders for picking

        Balance speed (customer satisfaction) with efficiency
        """

        orders_df = pd.DataFrame(orders)
        orders_df['order_time'] = pd.to_datetime(orders_df['order_time'])

        # Categorize by urgency
        orders_df['hours_until_pickup'] = (
            pd.to_datetime(orders_df['requested_pickup_time']) -
            orders_df['order_time']
        ).dt.total_seconds() / 3600

        # Priority scoring
        def calculate_priority(row):
            # High priority: short time until pickup, VIP customer, large order
            urgency_score = 100 / max(row['hours_until_pickup'], 0.5)
            vip_score = 20 if row.get('is_vip', False) else 0
            size_score = min(row['num_items'] * 2, 20)

            return urgency_score + vip_score + size_score

        orders_df['priority_score'] = orders_df.apply(calculate_priority, axis=1)

        # Sort by priority
        orders_df = orders_df.sort_values('priority_score', ascending=False)

        # Assign to time slots
        orders_df['assigned_pick_time'] = None
        current_time = datetime.now()
        orders_picked = 0
        slot_start = current_time

        for idx, order in orders_df.iterrows():
            # Estimate pick time for this order
            pick_time_minutes = order['num_items'] * 2  # 2 min per item

            # Assign to current slot
            orders_df.at[idx, 'assigned_pick_time'] = slot_start
            orders_df.at[idx, 'estimated_ready_time'] = (
                slot_start + timedelta(minutes=pick_time_minutes)
            )

            # Update slot
            orders_picked += 1
            if orders_picked >= picker_capacity_per_hour:
                slot_start += timedelta(hours=1)
                orders_picked = 0

        return orders_df[['order_id', 'priority_score', 'assigned_pick_time',
                         'estimated_ready_time', 'requested_pickup_time']]

    def calculate_bopis_roi(self, bopis_orders_per_month, avg_basket_size,
                           bopis_operating_cost_per_order=3.50):
        """
        Calculate ROI of BOPIS program

        Benefits:
        - Incremental sales from convenience
        - Additional impulse purchases in store
        - Reduced delivery costs vs. ship to home
        - Customer lifetime value increase
        """

        # Direct costs
        monthly_operating_cost = bopis_orders_per_month * bopis_operating_cost_per_order

        # Benefits
        # 1. Incremental purchases (customers buy more when picking up)
        impulse_purchase_rate = 0.35  # 35% buy additional items
        avg_impulse_purchase = 25
        impulse_revenue = (
            bopis_orders_per_month *
            impulse_purchase_rate *
            avg_impulse_purchase
        )

        # 2. Cost savings vs. home delivery
        home_delivery_cost = 8.50
        cost_savings = bopis_orders_per_month * (home_delivery_cost - bopis_operating_cost_per_order)

        # 3. Customer lifetime value increase (satisfaction)
        ltv_increase_per_customer = 50
        new_customers_per_month = bopis_orders_per_month * 0.3
        ltv_benefit = new_customers_per_month * ltv_increase_per_customer

        # 4. Inventory turns improvement (use store inventory)
        inventory_turn_benefit = bopis_orders_per_month * avg_basket_size * 0.05  # 5% carrying cost saved

        total_monthly_benefit = (
            impulse_revenue +
            cost_savings +
            ltv_benefit +
            inventory_turn_benefit
        )

        roi = (total_monthly_benefit - monthly_operating_cost) / monthly_operating_cost

        return {
            'monthly_cost': monthly_operating_cost,
            'monthly_benefit': total_monthly_benefit,
            'net_monthly_benefit': total_monthly_benefit - monthly_operating_cost,
            'roi': roi,
            'impulse_revenue': impulse_revenue,
            'cost_savings': cost_savings,
            'ltv_benefit': ltv_benefit
        }

# Example
bopis_optimizer = BOPISOptimizer({
    'online_margin': 15,
    'instore_margin': 18
})

# Inventory allocation
allocation = bopis_optimizer.allocate_inventory(
    sku='SKU456',
    available_qty=50,
    online_demand_forecast=30,
    instore_demand_forecast=35
)
print(f"Online allocation: {allocation['online_allocation']} units")
print(f"In-store allocation: {allocation['instore_allocation']} units")

# ROI calculation
roi_analysis = bopis_optimizer.calculate_bopis_roi(
    bopis_orders_per_month=2500,
    avg_basket_size=75
)
print(f"\nBOPIS ROI: {roi_analysis['roi']:.1%}")
print(f"Net monthly benefit: ${roi_analysis['net_monthly_benefit']:,.0f}")
```

### Ship-from-Store Optimization

**Store Fulfillment Capacity:**

```python
class ShipFromStoreOptimizer:
    """
    Optimize ship-from-store operations

    Balance store operations with fulfillment duties
    """

    def __init__(self, stores_config):
        self.stores = pd.DataFrame(stores_config)

    def calculate_store_fulfillment_capacity(self, store_id):
        """
        Determine optimal ship-from-store capacity for a store

        Based on:
        - Store traffic patterns
        - Staff availability
        - Physical space
        - Historical performance
        """

        store = self.stores[self.stores['store_id'] == store_id].iloc[0]

        # Available hours for fulfillment (avoid peak retail hours)
        if store['format'] == 'mall':
            fulfillment_hours_per_day = 4  # Morning before crowds
        elif store['format'] == 'strip':
            fulfillment_hours_per_day = 6
        else:
            fulfillment_hours_per_day = 8

        # Picking rate
        items_per_hour = 15

        # Staff allocation (% of staff that can be dedicated)
        staff_available = store['total_staff'] * 0.3  # 30% for online

        # Daily capacity
        daily_capacity = (
            fulfillment_hours_per_day *
            items_per_hour *
            staff_available
        )

        # Adjust for store size and layout
        if store['square_feet'] < 5000:
            space_factor = 0.7  # Cramped
        elif store['square_feet'] > 20000:
            space_factor = 1.2  # Ample space
        else:
            space_factor = 1.0

        daily_capacity *= space_factor

        return {
            'store_id': store_id,
            'daily_capacity_orders': int(daily_capacity / 3),  # 3 items per order avg
            'daily_capacity_units': int(daily_capacity),
            'fulfillment_hours': fulfillment_hours_per_day,
            'recommended_staff': staff_available
        }

    def allocate_online_orders_to_stores(self, orders, max_distance_miles=50):
        """
        Allocate online orders to stores for fulfillment

        Balances proximity, capacity, and inventory
        """

        allocations = []

        for order in orders:
            # Find eligible stores (within range, have inventory, have capacity)
            eligible_stores = []

            for idx, store in self.stores.iterrows():
                # Check distance
                distance = self._calculate_distance(
                    order['customer_lat'], order['customer_lon'],
                    store['lat'], store['lon']
                )

                if distance > max_distance_miles:
                    continue

                # Check inventory
                has_inventory = all(
                    store['inventory'].get(item['sku'], 0) >= item['quantity']
                    for item in order['items']
                )

                if not has_inventory:
                    continue

                # Check capacity
                capacity = self.calculate_store_fulfillment_capacity(store['store_id'])
                if store['orders_today'] >= capacity['daily_capacity_orders']:
                    continue

                # Calculate score
                score = distance  # Lower is better

                # Adjust for inventory health
                total_inventory_days = sum(
                    store['inventory'].get(item['sku'], 0) / store['sales_velocity'].get(item['sku'], 1)
                    for item in order['items']
                )

                if total_inventory_days > 60:
                    score *= 0.8  # Prefer stores with excess inventory

                eligible_stores.append({
                    'store_id': store['store_id'],
                    'distance': distance,
                    'score': score
                })

            if eligible_stores:
                # Select best store
                best_store = min(eligible_stores, key=lambda x: x['score'])
                allocations.append({
                    'order_id': order['order_id'],
                    'assigned_store': best_store['store_id'],
                    'distance': best_store['distance'],
                    'estimated_delivery_days': 1 if best_store['distance'] < 25 else 2
                })
            else:
                # No eligible store - route to DC
                allocations.append({
                    'order_id': order['order_id'],
                    'assigned_store': 'DC',
                    'distance': None,
                    'estimated_delivery_days': 3
                })

        return pd.DataFrame(allocations)

    def calculate_sfs_economics(self, annual_sfs_orders, avg_order_value=65):
        """
        Calculate economics of ship-from-store program

        Compare to DC fulfillment
        """

        # Ship-from-store costs
        sfs_labor_per_order = 4.50
        sfs_packaging_per_order = 1.20
        sfs_shipping_per_order = 7.00  # Lower than DC due to proximity
        sfs_overhead_per_order = 1.00
        sfs_total_cost_per_order = (
            sfs_labor_per_order +
            sfs_packaging_per_order +
            sfs_shipping_per_order +
            sfs_overhead_per_order
        )

        # DC fulfillment costs
        dc_labor_per_order = 3.00  # More efficient
        dc_packaging_per_order = 1.00
        dc_shipping_per_order = 9.50  # Farther from customer
        dc_overhead_per_order = 0.80
        dc_total_cost_per_order = (
            dc_labor_per_order +
            dc_packaging_per_order +
            dc_shipping_per_order +
            dc_overhead_per_order
        )

        # Annual comparison
        sfs_annual_cost = annual_sfs_orders * sfs_total_cost_per_order
        dc_annual_cost = annual_sfs_orders * dc_total_cost_per_order

        # Additional benefits of SFS
        # 1. Reduced inventory holding (use store stock)
        inventory_benefit = annual_sfs_orders * avg_order_value * 0.02  # 2% carrying cost

        # 2. Faster delivery (premium pricing potential)
        speed_benefit = annual_sfs_orders * 1.50  # $1.50 premium per order

        # 3. Reduced stockouts (more inventory nodes)
        stockout_reduction_benefit = annual_sfs_orders * avg_order_value * 0.01

        total_sfs_benefit = inventory_benefit + speed_benefit + stockout_reduction_benefit

        net_sfs_advantage = (dc_annual_cost - sfs_annual_cost) + total_sfs_benefit

        return {
            'annual_sfs_orders': annual_sfs_orders,
            'sfs_cost_per_order': sfs_total_cost_per_order,
            'dc_cost_per_order': dc_total_cost_per_order,
            'cost_difference_per_order': dc_total_cost_per_order - sfs_total_cost_per_order,
            'annual_cost_savings': dc_annual_cost - sfs_annual_cost,
            'additional_benefits': total_sfs_benefit,
            'net_annual_advantage': net_sfs_advantage,
            'roi_vs_dc': net_sfs_advantage / sfs_annual_cost if sfs_annual_cost > 0 else 0
        }

    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance in miles"""
        from math import radians, sin, cos, sqrt, atan2
        R = 3959
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

# Example
stores_config = [
    {'store_id': 'S001', 'format': 'mall', 'total_staff': 20, 'square_feet': 8000,
     'lat': 40.7589, 'lon': -73.9851, 'orders_today': 5},
    {'store_id': 'S002', 'format': 'strip', 'total_staff': 15, 'square_feet': 12000,
     'lat': 40.7128, 'lon': -74.0060, 'orders_today': 8},
]

sfs_optimizer = ShipFromStoreOptimizer(stores_config)

# Calculate capacity
capacity = sfs_optimizer.calculate_store_fulfillment_capacity('S001')
print(f"Store S001 daily capacity: {capacity['daily_capacity_orders']} orders")

# Economics
economics = sfs_optimizer.calculate_sfs_economics(annual_sfs_orders=50000)
print(f"Ship-from-store saves: ${economics['cost_difference_per_order']:.2f} per order")
print(f"Annual advantage: ${economics['net_annual_advantage']:,.0f}")
```

---

## Unified Inventory Visibility

**Real-Time Inventory Sync:**

```python
class UnifiedInventoryManager:
    """
    Manage unified inventory across all channels and nodes

    Challenges:
    - Real-time sync across systems
    - Inventory reservations
    - In-transit inventory
    - Accuracy issues
    """

    def __init__(self):
        self.inventory = {}  # SKU -> {node -> quantity}
        self.reservations = {}  # SKU -> {node -> reserved_qty}

    def get_available_to_promise(self, sku, node_id=None):
        """
        Calculate Available-to-Promise (ATP) inventory

        ATP = On-hand - Reserved - Safety stock
        """

        if node_id:
            nodes = [node_id]
        else:
            nodes = self.inventory.get(sku, {}).keys()

        atp_by_node = {}

        for node in nodes:
            on_hand = self.inventory.get(sku, {}).get(node, 0)
            reserved = self.reservations.get(sku, {}).get(node, 0)
            safety_stock = self._get_safety_stock(sku, node)

            atp = max(0, on_hand - reserved - safety_stock)
            atp_by_node[node] = atp

        return atp_by_node

    def reserve_inventory(self, order_id, sku, quantity, node_id,
                         reservation_duration_minutes=15):
        """
        Reserve inventory for an order

        Prevents overselling while customer completes purchase
        """

        atp = self.get_available_to_promise(sku, node_id)

        if atp.get(node_id, 0) < quantity:
            return {
                'success': False,
                'reason': 'Insufficient ATP',
                'available': atp.get(node_id, 0)
            }

        # Create reservation
        if sku not in self.reservations:
            self.reservations[sku] = {}
        if node_id not in self.reservations[sku]:
            self.reservations[sku][node_id] = 0

        self.reservations[sku][node_id] += quantity

        # Schedule expiration (would use job scheduler in production)
        expiration_time = datetime.now() + timedelta(minutes=reservation_duration_minutes)

        return {
            'success': True,
            'order_id': order_id,
            'sku': sku,
            'quantity': quantity,
            'node_id': node_id,
            'expires_at': expiration_time
        }

    def allocate_inventory_across_channels(self, sku, total_available,
                                          channel_forecasts):
        """
        Allocate inventory across channels (online, store, wholesale, etc.)

        Balance channel priorities and demand
        """

        # Calculate total demand
        total_demand = sum(channel_forecasts.values())

        if total_available >= total_demand:
            # Sufficient inventory - allocate all
            return channel_forecasts

        # Scarce inventory - prioritize by channel value
        channel_priority = {
            'online': 1.2,  # 20% premium (strategic)
            'store': 1.0,
            'wholesale': 0.8,
            'marketplace': 0.9
        }

        # Weight demands by priority
        weighted_demands = {
            channel: demand * channel_priority.get(channel, 1.0)
            for channel, demand in channel_forecasts.items()
        }

        total_weighted = sum(weighted_demands.values())

        # Allocate proportionally
        allocations = {
            channel: int(total_available * (weighted / total_weighted))
            for channel, weighted in weighted_demands.items()
        }

        # Handle rounding (allocate remainder to highest priority)
        allocated_sum = sum(allocations.values())
        if allocated_sum < total_available:
            # Give remainder to highest priority channel
            highest_priority_channel = max(
                channel_priority.items(),
                key=lambda x: x[1]
            )[0]
            if highest_priority_channel in allocations:
                allocations[highest_priority_channel] += (total_available - allocated_sum)

        return allocations

    def _get_safety_stock(self, sku, node):
        """Get safety stock for SKU at node"""
        # Simplified - would calculate based on demand variability
        return 10

# Example
inventory_manager = UnifiedInventoryManager()
inventory_manager.inventory = {
    'SKU789': {
        'DC1': 1000,
        'Store_101': 50,
        'Store_102': 30
    }
}

# Check ATP
atp = inventory_manager.get_available_to_promise('SKU789')
print(f"Available-to-Promise: {atp}")

# Reserve inventory
reservation = inventory_manager.reserve_inventory(
    order_id='ORD123',
    sku='SKU789',
    quantity=2,
    node_id='Store_101'
)
print(f"Reservation: {reservation['success']}")

# Allocate across channels
channel_forecasts = {
    'online': 500,
    'store': 400,
    'wholesale': 200
}

allocations = inventory_manager.allocate_inventory_across_channels(
    sku='SKU789',
    total_available=800,
    channel_forecasts=channel_forecasts
)
print(f"Channel allocations: {allocations}")
```

---

## Tools & Libraries

### Python Libraries

**Optimization:**
- `pulp`, `pyomo`: Mathematical optimization for order routing
- `scipy`: Optimization algorithms
- `ortools`: Google OR-Tools for vehicle routing

**Geospatial:**
- `geopy`: Distance calculations
- `folium`: Map visualization
- `geopandas`: Geospatial data analysis

**Data Processing:**
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Clustering for zone optimization

### Commercial Software

**Order Management Systems (OMS):**
- **Manhattan Active Omni**: Enterprise OMS
- **Fluent Commerce**: Cloud-native OMS
- **IBM Sterling**: Order management
- **Blue Yonder Luminate**: Omnichannel fulfillment
- **Salesforce Order Management**: Cloud OMS

**Inventory Management:**
- **Radial**: Omnichannel inventory optimization
- **CommerceHub**: Distributed order management
- **Deposco**: Inventory & fulfillment platform
- **Brightpearl**: Retail operations platform

**Store Systems:**
- **Shopify POS**: Point of sale with online integration
- **Square**: POS and omnichannel
- **NCR**: Retail POS systems
- **Oracle Retail**: Enterprise retail management

---

## Common Challenges & Solutions

### Challenge: Inventory Accuracy

**Problem:**
- Store inventory often inaccurate (shrink, mis-scans)
- Online shows available but store doesn't have it
- Customer frustration, failed BOPIS orders

**Solutions:**
- RFID for real-time inventory tracking
- Perpetual inventory systems
- Regular cycle counts
- Safety stock buffers for online channel
- Probabilistic ATP (account for accuracy)
- Customer notification if item unavailable during pick

### Challenge: Split Shipment Decisions

**Problem:**
- Customer wants fast delivery
- Items available at different locations
- Split shipments increase cost but improve speed

**Solutions:**
- Cost-benefit analysis per order
- Customer preference (allow choice)
- Minimum split threshold (only split if saves 2+ days)
- Free shipping threshold incentives
- Show estimated delivery dates in cart
- Intelligent bundling algorithms

### Challenge: Store Capacity Constraints

**Problem:**
- Stores busy during peak hours
- Online fulfillment competes with retail operations
- Staff overwhelmed

**Solutions:**
- Dynamic capacity management
- Route orders to less busy stores
- Dedicated online fulfillment staff during peak
- Micro-fulfillment centers near stores
- After-hours picking programs
- BOPIS time slot management

### Challenge: Returns Complexity

**Problem:**
- Buy online, return in store (complexity)
- Buy in store, return by mail
- Inventory reallocation after returns
- Refund timing issues

**Solutions:**
- Unified return processing system
- Instant credit upon store return
- Return to any location policy
- Automated restocking workflows
- Return fraud detection
- Cross-channel return analytics

### Challenge: Profitability by Channel

**Problem:**
- Some fulfillment methods unprofitable
- Hard to measure true channel profitability
- Free shipping expectations

**Solutions:**
- Fully-loaded cost accounting by channel
- Minimum order values for free shipping
- BOPIS/pickup incentives (discounts)
- Shipping pass subscriptions
- Zone-based shipping fees
- Product-level profitability analysis

---

## Output Format

### Omnichannel Fulfillment Analysis Report

**Executive Summary:**
- Total orders by channel: Online 45%, Store 40%, BOPIS 15%
- Current fulfillment costs: $8.20 per order
- Ship-from-store percentage: 22%
- Target: Increase SFS to 40%, reduce cost to $7.00

**Channel Performance:**

| Channel | Orders/Month | Avg Basket | Fulfillment Cost | Net Margin | Growth YoY |
|---------|-------------|------------|------------------|------------|------------|
| E-commerce | 45,000 | $72 | $9.50 | 18% | +35% |
| BOPIS | 15,000 | $68 | $3.50 | 24% | +125% |
| Ship from Store | 22,000 | $65 | $8.00 | 19% | +80% |
| Store Retail | 40,000 | $58 | $2.00 | 22% | -5% |

**Fulfillment Network Utilization:**

| Node Type | # Locations | Capacity Util | Orders/Day | Avg Distance | Delivery Speed |
|-----------|-------------|---------------|------------|--------------|----------------|
| Distribution Centers | 3 | 68% | 1,200 | 450 mi | 3.2 days |
| Ship-from-Store | 45 | 42% | 800 | 35 mi | 1.8 days |
| BOPIS Pickup | 120 | 35% | 500 | 8 mi | Same day |

**Order Routing Analysis:**

- **Single-location fulfillment**: 78%
- **Split shipments**: 22%
- **Average splits per order**: 1.3 locations
- **Routing efficiency**: 82% (% routed to optimal location)

**Optimization Opportunities:**

1. **Expand ship-from-store (22% → 40%)**
   - Enable 50 additional stores for SFS
   - Investment: $250K (picking stations, training)
   - Expected impact: -$1.20 per order, -0.8 days delivery time
   - Annual savings: $650K

2. **Improve inventory accuracy (85% → 95%)**
   - RFID implementation in top 50 stores
   - Investment: $400K
   - Expected impact: -5% failed BOPIS, +8% BOPIS adoption
   - Annual savings: $320K + revenue lift

3. **Intelligent order routing optimization**
   - Implement ML-based routing engine
   - Investment: $150K
   - Expected impact: +15% routing efficiency, -$0.40 per order
   - Annual savings: $480K

4. **BOPIS expansion**
   - Add 30 BOPIS-capable stores
   - Investment: $180K
   - Expected impact: +10K orders/month, $25 impulse purchase lift
   - Annual benefit: $850K

**Implementation Roadmap:**

| Quarter | Initiative | Investment | Annual Benefit | Payback |
|---------|-----------|------------|----------------|---------|
| Q1 | Order routing optimization | $150K | $480K | 3.8 mo |
| Q2 | BOPIS expansion (30 stores) | $180K | $850K | 2.5 mo |
| Q3 | SFS expansion (50 stores) | $250K | $650K | 4.6 mo |
| Q4 | RFID inventory accuracy | $400K | $320K | 15 mo |

**Expected Results (Year 1):**

| Metric | Current | Year 1 Target | Improvement |
|--------|---------|---------------|-------------|
| Fulfillment cost per order | $8.20 | $7.00 | -15% |
| Ship-from-store % | 22% | 40% | +18 pts |
| Avg delivery speed | 2.8 days | 2.1 days | -0.7 days |
| BOPIS orders/month | 15,000 | 25,000 | +67% |
| Split shipment rate | 22% | 18% | -4 pts |
| Inventory accuracy | 85% | 95% | +10 pts |

---

## Questions to Ask

If you need more context:
1. What channels do you sell through? (stores, online, marketplace, mobile)
2. Do stores currently fulfill online orders? (BOPIS, ship-from-store)
3. How many stores and DCs in your network?
4. What's your current online order volume?
5. What percentage of online orders are BOPIS?
6. What's your average delivery time for online orders?
7. Do you have real-time inventory visibility across channels?
8. What systems do you use? (OMS, WMS, POS, e-commerce platform)
9. What are your main pain points? (cost, speed, experience, complexity)

---

## Related Skills

- **ecommerce-fulfillment**: Pure e-commerce fulfillment strategy
- **retail-allocation**: Initial allocation to stores
- **retail-replenishment**: Store replenishment optimization
- **last-mile-delivery**: Final delivery optimization
- **route-optimization**: Delivery route planning
- **warehouse-design**: DC and micro-fulfillment center design
- **network-design**: Fulfillment network strategy
- **inventory-optimization**: Safety stock and reorder points
- **demand-forecasting**: Channel demand forecasting



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

---
name: vrp-backhauls
description: When the user wants to solve VRP with Backhauls (VRPB), optimize routes with both deliveries and pickups, or handle reverse logistics. Also use when the user mentions "VRPB," "backhaul optimization," "linehaul and backhaul," "delivery and pickup routes," "reverse logistics," or "return pickups." Backhauls are pickups that occur AFTER all deliveries on a route. For paired pickup-delivery, see pickup-delivery-problem.
---

# Vehicle Routing Problem with Backhauls (VRPB)

You are an expert in the Vehicle Routing Problem with Backhauls and reverse logistics optimization. Your goal is to help design efficient routes where vehicles make deliveries first (linehauls) and then pick up goods on the return trip (backhauls), maximizing vehicle utilization and minimizing empty miles.

## Initial Assessment

Before solving VRPB instances, understand:

1. **Backhaul Characteristics**
   - Strict sequence (all deliveries before all pickups)?
   - Mixed linehauls and backhauls allowed?
   - Can a customer have both delivery AND pickup?

2. **Capacity Considerations**
   - Same vehicle capacity for deliveries and pickups?
   - How does capacity work? (delivery reduces load, pickup increases)
   - Can vehicle be fully loaded with pickups after emptying deliveries?

3. **Customer Types**
   - Linehaul customers (delivery only)
   - Backhaul customers (pickup only)
   - Mixed customers (both delivery and pickup)
   - Number of each type?

4. **Business Context**
   - Return of empty containers/pallets?
   - Reverse logistics (returns, recycling)?
   - Supply redistribution between locations?
   - Waste collection after deliveries?

5. **Problem Scale**
   - Small (< 50 customers): Exact methods possible
   - Medium (50-200): Advanced heuristics
   - Large (200+): Metaheuristics required

---

## Mathematical Formulation

### VRPB with Sequential Constraint

**Sets:**
- V = {0} ∪ L ∪ B: Nodes (0 = depot, L = linehaul, B = backhaul customers)
- K: Vehicles

**Parameters:**
- c_{ij}: Cost/distance from i to j
- d_i: Delivery quantity at linehaul customer i ∈ L
- p_j: Pickup quantity at backhaul customer j ∈ B
- Q: Vehicle capacity

**Decision Variables:**
- x_{ijk} ∈ {0,1}: 1 if vehicle k travels from i to j
- u_{ik} ∈ ℝ: Load on vehicle k after visiting node i

**Objective:**
```
Minimize: Σ_{k∈K} Σ_{i∈V} Σ_{j∈V} c_{ij} * x_{ijk}
```

**Constraints:**
```
1. Each customer visited exactly once:
   Σ_{k∈K} Σ_{i∈V, i≠j} x_{ijk} = 1,  ∀j ∈ L ∪ B

2. Flow conservation:
   Σ_{i∈V} x_{ihk} = Σ_{j∈V} x_{hjk},  ∀h ∈ V, ∀k ∈ K

3. Capacity constraint:
   Σ_{i∈L} d_i * Σ_{j∈V} x_{ijk} ≤ Q,  ∀k ∈ K (deliveries)
   Σ_{j∈B} p_j * Σ_{i∈V} x_{ijjk} ≤ Q,  ∀k ∈ K (pickups)

4. Load tracking:
   Delivery phase: u_{jk} = u_{ik} - d_j (load decreases)
   Pickup phase: u_{jk} = u_{ik} + p_j (load increases)

5. Precedence (all linehauls before backhauls):
   If x_{ijk} = 1 and i ∈ L, j ∈ B, then
   all linehaul customers must be visited before j

6. Subtour elimination

7. Binary variables:
   x_{ijk} ∈ {0,1}
```

---

## Classical Heuristics

### 1. Sequential Cluster-Route for VRPB

```python
import numpy as np

def vrpb_cluster_route(coordinates, linehaul_demands, backhaul_demands,
                      vehicle_capacity, num_vehicles, depot_idx=0):
    """
    Cluster-then-route heuristic for VRPB

    Phase 1: Cluster customers geographically
    Phase 2: Within each cluster, sequence linehauls then backhauls
    Phase 3: Optimize sequences

    Args:
        coordinates: all location coordinates
        linehaul_demands: delivery demands (0 for backhaul-only customers)
        backhaul_demands: pickup demands (0 for linehaul-only customers)
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot_idx: depot index

    Returns:
        solution dictionary
    """
    n = len(coordinates)
    depot = coordinates[depot_idx]

    # Identify customer types
    linehaul_customers = [i for i in range(n)
                         if i != depot_idx and linehaul_demands[i] > 0]
    backhaul_customers = [i for i in range(n)
                         if i != depot_idx and backhaul_demands[i] > 0]

    print(f"Linehaul customers: {len(linehaul_customers)}")
    print(f"Backhaul customers: {len(backhaul_customers)}")

    # Distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Sweep algorithm to create clusters
    import math

    def polar_angle(point):
        dx = point[0] - depot[0]
        dy = point[1] - depot[1]
        return math.atan2(dy, dx)

    # Sort all customers by angle
    all_customers = linehaul_customers + backhaul_customers
    customer_angles = [(polar_angle(coordinates[c]), c) for c in all_customers]
    customer_angles.sort()

    # Build routes by sweeping
    routes = []
    current_route_linehauls = []
    current_route_backhauls = []
    current_linehaul_load = 0
    current_backhaul_load = 0

    for angle, customer in customer_angles:
        is_linehaul = customer in linehaul_customers

        if is_linehaul:
            demand = linehaul_demands[customer]

            # Check if adding this linehaul is feasible
            if current_linehaul_load + demand <= vehicle_capacity:
                current_route_linehauls.append(customer)
                current_linehaul_load += demand
            else:
                # Start new route
                if current_route_linehauls or current_route_backhauls:
                    routes.append({
                        'linehauls': current_route_linehauls,
                        'backhauls': current_route_backhauls
                    })

                current_route_linehauls = [customer]
                current_route_backhauls = []
                current_linehaul_load = demand
                current_backhaul_load = 0
        else:
            demand = backhaul_demands[customer]

            # Check if adding this backhaul is feasible
            if current_backhaul_load + demand <= vehicle_capacity:
                current_route_backhauls.append(customer)
                current_backhaul_load += demand
            else:
                # Check if we can start new route
                if current_route_linehauls or current_route_backhauls:
                    routes.append({
                        'linehauls': current_route_linehauls,
                        'backhauls': current_route_backhauls
                    })

                current_route_linehauls = []
                current_route_backhauls = [customer]
                current_linehaul_load = 0
                current_backhaul_load = demand

    # Add last route
    if current_route_linehauls or current_route_backhauls:
        routes.append({
            'linehauls': current_route_linehauls,
            'backhauls': current_route_backhauls
        })

    # Optimize each route with 2-opt (separately for linehauls and backhauls)
    optimized_routes = []

    for route in routes:
        # Optimize linehaul sequence
        if len(route['linehauls']) > 2:
            linehaul_seq = [depot_idx] + route['linehauls']
            linehaul_seq = two_opt_segment(linehaul_seq, dist_matrix)
            route['linehauls'] = linehaul_seq[1:]  # Remove depot

        # Optimize backhaul sequence
        if len(route['backhauls']) > 2:
            backhaul_seq = route['backhauls'] + [depot_idx]
            backhaul_seq = two_opt_segment(backhaul_seq, dist_matrix)
            route['backhauls'] = backhaul_seq[:-1]  # Remove depot

        optimized_routes.append(route)

    # Convert to full routes and calculate distance
    full_routes = []
    total_distance = 0

    for route in optimized_routes:
        full_route = [depot_idx]
        full_route.extend(route['linehauls'])
        full_route.extend(route['backhauls'])
        full_route.append(depot_idx)

        full_routes.append(full_route)

        # Calculate distance
        route_distance = sum(dist_matrix[full_route[i]][full_route[i+1]]
                           for i in range(len(full_route)-1))
        total_distance += route_distance

    return {
        'routes': full_routes,
        'route_details': optimized_routes,
        'total_distance': total_distance,
        'num_vehicles': len(full_routes)
    }


def two_opt_segment(sequence, dist_matrix):
    """2-opt optimization for a sequence"""
    improved = True
    best = sequence.copy()

    while improved:
        improved = False
        for i in range(len(best) - 2):
            for j in range(i + 2, len(best)):
                if j - i == 1:
                    continue

                current_cost = (dist_matrix[best[i]][best[i+1]] +
                              dist_matrix[best[j-1]][best[j]])
                new_cost = (dist_matrix[best[i]][best[j-1]] +
                           dist_matrix[best[i+1]][best[j]])

                if new_cost < current_cost - 1e-10:
                    best[i+1:j] = reversed(best[i+1:j])
                    improved = True
                    break

            if improved:
                break

    return best
```

### 2. VRPB with OR-Tools

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_vrpb_ortools(coordinates, linehaul_demands, backhaul_demands,
                      vehicle_capacity, num_vehicles, depot=0,
                      time_limit=60):
    """
    Solve VRPB using OR-Tools

    Enforces that all linehauls are served before backhauls on each route

    Args:
        coordinates: location coordinates
        linehaul_demands: delivery demands (0 if backhaul-only)
        backhaul_demands: pickup demands (0 if linehaul-only)
        vehicle_capacity: vehicle capacity
        num_vehicles: number of vehicles
        depot: depot index
        time_limit: time limit

    Returns:
        solution dictionary
    """
    import math

    n = len(coordinates)

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = math.sqrt(
                (coordinates[i][0] - coordinates[j][0])**2 +
                (coordinates[i][1] - coordinates[j][1])**2
            )

    # Identify customer types
    linehaul_only = [i for i in range(n) if linehaul_demands[i] > 0
                    and backhaul_demands[i] == 0]
    backhaul_only = [i for i in range(n) if backhaul_demands[i] > 0
                    and linehaul_demands[i] == 0]
    mixed = [i for i in range(n) if linehaul_demands[i] > 0
            and backhaul_demands[i] > 0]

    # Create routing manager
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)

    # Create routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 100)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add two capacity dimensions: one for deliveries, one for pickups

    # Delivery capacity (starts full, decreases)
    def delivery_demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(linehaul_demands[from_node])

    delivery_callback_index = routing.RegisterUnaryTransitCallback(
        delivery_demand_callback)

    routing.AddDimension(
        delivery_callback_index,
        0,  # no slack
        int(vehicle_capacity),  # maximum capacity
        True,  # start cumul to zero
        'Delivery_Capacity')

    # Pickup capacity (starts at 0, increases)
    def pickup_demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(backhaul_demands[from_node])

    pickup_callback_index = routing.RegisterUnaryTransitCallback(
        pickup_demand_callback)

    routing.AddDimension(
        pickup_callback_index,
        0,  # no slack
        int(vehicle_capacity),  # maximum capacity
        True,  # start cumul to zero
        'Pickup_Capacity')

    # Add precedence constraints: linehauls before backhauls
    # Use a counter dimension to enforce sequence
    def counter_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        # Linehaul customers get value 0, backhauls get value 1000
        if from_node in backhaul_only:
            return 1000
        return 0

    counter_callback_index = routing.RegisterUnaryTransitCallback(counter_callback)

    routing.AddDimension(
        counter_callback_index,
        0,  # no slack
        10000,  # large upper bound
        True,  # start cumul to zero
        'Counter')

    counter_dimension = routing.GetDimensionOrDie('Counter')

    # Enforce that linehauls (counter ~0) come before backhauls (counter ~1000)
    for backhaul_customer in backhaul_only:
        index = manager.NodeToIndex(backhaul_customer)
        counter_dimension.CumulVar(index).SetMin(500)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = time_limit

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_distance = 0

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            linehaul_customers = []
            backhaul_customers = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)

                if node in linehaul_only:
                    linehaul_customers.append(node)
                elif node in backhaul_only:
                    backhaul_customers.append(node)

                index = solution.Value(routing.NextVar(index))

            route.append(manager.IndexToNode(index))

            if len(route) > 2:
                routes.append({
                    'route': route,
                    'linehauls': linehaul_customers,
                    'backhauls': backhaul_customers
                })

                # Calculate distance
                route_distance = sum(dist_matrix[route[i]][route[i+1]]
                                   for i in range(len(route)-1))
                total_distance += route_distance

        return {
            'status': 'Optimal',
            'routes': routes,
            'total_distance': total_distance,
            'num_vehicles': len(routes)
        }
    else:
        return {
            'status': 'No solution found',
            'routes': None
        }


# Visualization
def visualize_vrpb_solution(coordinates, routes, linehaul_demands,
                           backhaul_demands, save_path=None):
    """Visualize VRPB solution with linehauls and backhauls"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))

    for idx, route_info in enumerate(routes):
        route = route_info['route']
        linehauls = route_info.get('linehauls', [])
        backhauls = route_info.get('backhauls', [])

        # Plot full route
        route_coords = [coordinates[i] for i in route]
        xs = [c[0] for c in route_coords]
        ys = [c[1] for c in route_coords]

        ax.plot(xs, ys, '-', color=colors[idx], linewidth=2, alpha=0.5)

        # Mark linehaul customers (circles)
        for customer in linehauls:
            coord = coordinates[customer]
            ax.plot(coord[0], coord[1], 'o', color=colors[idx],
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5)

        # Mark backhaul customers (triangles)
        for customer in backhauls:
            coord = coordinates[customer]
            ax.plot(coord[0], coord[1], '^', color=colors[idx],
                   markersize=12, markeredgecolor='black', markeredgewidth=1.5)

    # Plot depot
    depot = coordinates[0]
    ax.plot(depot[0], depot[1], 's', color='red', markersize=20,
           label='Depot', markeredgecolor='black', markeredgewidth=2, zorder=10)

    # Legend
    ax.plot([], [], 'o', color='gray', markersize=10,
           markeredgecolor='black', label='Linehaul (Delivery)')
    ax.plot([], [], '^', color='gray', markersize=10,
           markeredgecolor='black', label='Backhaul (Pickup)')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('VRP with Backhauls Solution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Example
if __name__ == "__main__":
    import random
    np.random.seed(42)
    random.seed(42)

    # Generate problem
    n_linehauls = 15
    n_backhauls = 10
    n_total = n_linehauls + n_backhauls + 1  # +1 for depot

    coordinates = [(50, 50)]  # Depot
    coordinates.extend(np.random.rand(n_linehauls + n_backhauls, 2).tolist() * 100)

    # Demands
    linehaul_demands = [0]  # Depot
    linehaul_demands.extend([random.randint(5, 20) for _ in range(n_linehauls)])
    linehaul_demands.extend([0] * n_backhauls)  # Backhaul customers have no deliveries

    backhaul_demands = [0]  # Depot
    backhaul_demands.extend([0] * n_linehauls)  # Linehaul customers have no pickups
    backhaul_demands.extend([random.randint(5, 15) for _ in range(n_backhauls)])

    vehicle_capacity = 100
    num_vehicles = 4

    print(f"Problem: {n_linehauls} linehauls, {n_backhauls} backhauls")
    print(f"Total delivery demand: {sum(linehaul_demands)}")
    print(f"Total pickup demand: {sum(backhaul_demands)}")

    print("\nSolving VRPB with OR-Tools...")
    result = solve_vrpb_ortools(coordinates, linehaul_demands, backhaul_demands,
                               vehicle_capacity, num_vehicles, time_limit=60)

    if result['status'] == 'Optimal':
        print(f"\nStatus: {result['status']}")
        print(f"Total Distance: {result['total_distance']:.2f}")
        print(f"Vehicles Used: {result['num_vehicles']}")

        print("\nRoute Details:")
        for i, route_info in enumerate(result['routes']):
            route = route_info['route']
            linehauls = route_info['linehauls']
            backhauls = route_info['backhauls']

            total_delivery = sum(linehaul_demands[c] for c in linehauls)
            total_pickup = sum(backhaul_demands[c] for c in backhauls)

            print(f"\n  Vehicle {i+1}:")
            print(f"    Route: {route}")
            print(f"    Linehauls: {len(linehauls)} customers, {total_delivery} units")
            print(f"    Backhauls: {len(backhauls)} customers, {total_pickup} units")

        # Visualize
        visualize_vrpb_solution(coordinates, result['routes'],
                              linehaul_demands, backhaul_demands)
    else:
        print(f"Status: {result['status']}")
```

---

## Tools & Libraries

- **OR-Tools (Google)**: Best for VRPB (recommended)
- **PuLP/Pyomo**: MIP modeling
- Custom heuristics work well for this variant

---

## Common Challenges & Solutions

### Challenge: Imbalanced Linehauls and Backhauls

**Problem:**
- Many deliveries, few pickups (or vice versa)
- Vehicles return nearly empty

**Solutions:**
- Allow mixed linehaul-backhaul at same customer
- Consider dedicated backhaul-only routes
- Relax sequential constraint if possible

### Challenge: Strict Sequential Constraint Too Restrictive

**Problem:**
- Forcing all deliveries before all pickups reduces efficiency
- Could save distance by mixing

**Solutions:**
- Consider mixed VRPB (allows interleaving)
- See **pickup-delivery-problem** for more flexible variant
- Use soft penalties instead of hard constraint

### Challenge: Capacity Management

**Problem:**
- Vehicle might be full with pickups before finishing backhauls
- Complex capacity tracking

**Solutions:**
- Use two-dimensional capacity in OR-Tools
- Carefully check feasibility in heuristics
- Consider vehicle with compartments

---

## Output Format

### VRPB Solution Report

**Problem:**
- Linehaul customers: 20 (deliveries)
- Backhaul customers: 15 (pickups)
- Vehicles: 5 (capacity: 100 units)

**Solution:**

| Metric | Value |
|--------|-------|
| Total Distance | 1,124 km |
| Vehicles Used | 5 |
| Total Deliveries | 387 units |
| Total Pickups | 276 units |

**Route Details:**

| Vehicle | Linehauls | Deliveries | Backhauls | Pickups | Distance |
|---------|-----------|------------|-----------|---------|----------|
| 1 | 4 | 78 units | 3 | 54 units | 235 km |
| 2 | 5 | 95 units | 2 | 38 units | 198 km |
[...]

---

## Questions to Ask

1. Must all deliveries occur before all pickups?
2. Can customers have both delivery and pickup?
3. What's the ratio of linehauls to backhauls?
4. Is this for reverse logistics or redistribution?
5. Are there time constraints?
6. Same vehicle capacity for deliveries and pickups?

---

## Related Skills

- **vehicle-routing-problem**: For general VRP
- **pickup-delivery-problem**: For paired pickup-delivery
- **capacitated-vrp**: For capacity-focused routing

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

