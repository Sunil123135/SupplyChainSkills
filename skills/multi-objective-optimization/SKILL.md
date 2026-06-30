---
name: multi-objective-optimization
description: "When the user wants to optimize multiple conflicting objectives, find Pareto-optimal solutions, or balance trade-offs between cost, service, quality, and sustainability. Also use when the user mentions \"multi-objective,\" \"Pareto optimization,\" \"NSGA-II,\" \"trade-off analysis,\" \"scalarization,\" \"weighted objectives,\" \"goal programming,\" or \"multiple criteria optimization.\" For single objective, see optimization-modeling."
---

# Multi-Objective Optimization

You are an expert in multi-objective optimization for supply chain. Your goal is to help find and analyze Pareto-optimal solutions that balance conflicting objectives like cost vs service, profit vs sustainability, or efficiency vs resilience.

## Initial Assessment

1. **Objectives**: What are competing goals? (minimize cost, maximize service, minimize carbon)
2. **Preferences**: Known trade-offs or discover Pareto frontier?
3. **Decision Maker**: Interactive or automated selection?
4. **Problem Size**: Solvable with exact methods or need heuristics?

---

## Core Concepts

**Pareto Dominance:** Solution x dominates y if x is better in all objectives

**Pareto Front:** Set of non-dominated solutions

**Trade-off:** Improving one objective worsens another

---

## Methods

### 1. Weighted Sum (Scalarization)

```python
# Combine objectives with weights
objective = w1 * cost + w2 * (-service_level) + w3 * carbon

# Vary weights to get different Pareto points
for w1 in [0.2, 0.5, 0.8]:
    w2, w3 = (1-w1)/2, (1-w1)/2
    solve_with_weights(w1, w2, w3)
```

### 2. ε-Constraint Method

```python
# Optimize one objective, constrain others
minimize cost
subject to:
    service_level ≥ 0.95
    carbon ≤ 1000
```

### 3. NSGA-II (Genetic Algorithm)

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.problems import get_problem

# Multi-objective problem
problem = SupplyChainMO()

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               verbose=True)

# Get Pareto front
pareto_front = res.F
```

### 4. Goal Programming

```python
# Set target for each objective, minimize deviations
targets = {'cost': 100000, 'service': 0.98, 'carbon': 500}

minimize sum(d_minus[obj] + d_plus[obj] for obj in objectives)
subject to:
    actual[obj] + d_plus[obj] - d_minus[obj] = targets[obj]
```

---

## Supply Chain Network Design: Cost vs Service

```python
from pulp import *
import numpy as np
import matplotlib.pyplot as plt

def multi_objective_network_design(customers, facilities, weights):
    """
    Network design with cost and service objectives
    
    Objective 1: Minimize total cost
    Objective 2: Minimize average distance (maximize service)
    """
    
    model = LpProblem("MultiObj_Network", LpMinimize)
    
    # Variables
    open_facility = LpVariable.dicts("Open", facilities, cat='Binary')
    flow = LpVariable.dicts("Flow",
                           [(i,j) for i in customers for j in facilities],
                           lowBound=0)
    
    # Objective: weighted combination
    w_cost, w_service = weights
    
    cost_obj = lpSum([fixed_cost[j] * open_facility[j] for j in facilities]) + \
               lpSum([transport_cost[i,j] * flow[i,j]
                     for i in customers for j in facilities])
    
    service_obj = lpSum([distance[i,j] * flow[i,j]
                        for i in customers for j in facilities])
    
    # Normalize objectives
    max_cost = estimate_max_cost()
    max_distance = estimate_max_distance()
    
    model += w_cost * (cost_obj / max_cost) + \
             w_service * (service_obj / max_distance), "Weighted_Objective"
    
    # Constraints
    for i in customers:
        model += lpSum([flow[i,j] for j in facilities]) >= demand[i]
    
    for j in facilities:
        model += lpSum([flow[i,j] for i in customers]) <= \
                 capacity[j] * open_facility[j]
    
    model.solve()
    
    return {
        'cost': value(cost_obj),
        'service': value(service_obj),
        'open_facilities': [j for j in facilities if open_facility[j].varValue > 0.5]
    }

# Generate Pareto frontier
pareto_solutions = []
for w_cost in np.linspace(0, 1, 20):
    w_service = 1 - w_cost
    sol = multi_objective_network_design(customers, facilities, (w_cost, w_service))
    pareto_solutions.append(sol)

# Plot Pareto front
costs = [s['cost'] for s in pareto_solutions]
services = [s['service'] for s in pareto_solutions]

plt.figure(figsize=(10, 6))
plt.plot(costs, services, 'o-', linewidth=2, markersize=8)
plt.xlabel('Total Cost ($)')
plt.ylabel('Average Distance (Service)')
plt.title('Pareto Frontier: Cost vs Service Trade-off')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Sustainable Supply Chain: Economic-Environmental-Social

```python
class TripleBottomLineOptimization:
    """
    Optimize Economic, Environmental, and Social objectives
    """
    
    def __init__(self, network_data):
        self.data = network_data
    
    def optimize_pareto(self, method='weighted_sum'):
        """
        Find Pareto-optimal solutions for triple bottom line
        
        Objectives:
        1. Economic: Minimize cost
        2. Environmental: Minimize carbon emissions
        3. Social: Maximize local employment
        """
        
        if method == 'weighted_sum':
            solutions = []
            
            # Systematically vary weights
            for w1 in [0.2, 0.4, 0.6, 0.8]:
                for w2 in [0.2, 0.4, 0.6, 0.8]:
                    w3 = max(0, 1 - w1 - w2)
                    if w1 + w2 + w3 > 0.99:  # Valid weight combination
                        sol = self.solve_weighted(w1, w2, w3)
                        solutions.append(sol)
            
            # Filter non-dominated solutions
            pareto_front = self.extract_pareto_front(solutions)
            return pareto_front
        
        elif method == 'epsilon_constraint':
            # Fix two objectives, optimize third
            pareto_front = []
            
            for carbon_limit in np.linspace(min_carbon, max_carbon, 10):
                for employment_target in np.linspace(min_emp, max_emp, 10):
                    sol = self.solve_epsilon_constraint(
                        carbon_limit=carbon_limit,
                        employment_target=employment_target
                    )
                    if sol['feasible']:
                        pareto_front.append(sol)
            
            return pareto_front
    
    def solve_weighted(self, w_economic, w_environmental, w_social):
        """Solve with weighted objectives"""
        
        model = LpProblem("Triple_Bottom_Line", LpMinimize)
        
        # Variables and constraints
        # ...
        
        # Weighted objective
        model += (
            w_economic * economic_cost +
            w_environmental * carbon_emissions +
            w_social * (-local_employment)  # Maximize employment
        )
        
        model.solve()
        
        return {
            'economic': value(economic_cost),
            'environmental': value(carbon_emissions),
            'social': value(local_employment),
            'weights': (w_economic, w_environmental, w_social)
        }
    
    def extract_pareto_front(self, solutions):
        """Filter non-dominated solutions"""
        
        pareto = []
        
        for sol in solutions:
            dominated = False
            
            for other in solutions:
                if self.dominates(other, sol):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(sol)
        
        return pareto
    
    def dominates(self, sol1, sol2):
        """Check if sol1 Pareto-dominates sol2"""
        
        # sol1 dominates if better in all objectives
        better_economic = sol1['economic'] <= sol2['economic']
        better_environmental = sol1['environmental'] <= sol2['environmental']
        better_social = sol1['social'] >= sol2['social']  # Maximize
        
        at_least_one_strictly_better = (
            sol1['economic'] < sol2['economic'] or
            sol1['environmental'] < sol2['environmental'] or
            sol1['social'] > sol2['social']
        )
        
        return (better_economic and better_environmental and better_social and
                at_least_one_strictly_better)
```

---

## Interactive Decision-Making

```python
def interactive_pareto_exploration(problem, decision_maker):
    """
    Interactive method: present solutions, get feedback, refine
    """
    
    # Generate initial Pareto front
    pareto_front = problem.generate_initial_pareto_front()
    
    iteration = 0
    max_iterations = 10
    
    while iteration < max_iterations:
        # Present solutions to decision maker
        print(f"\nIteration {iteration + 1}")
        print("Current Pareto Solutions:")
        for i, sol in enumerate(pareto_front):
            print(f"  {i}: Cost=${sol['cost']}, Service={sol['service']:.2%}, Carbon={sol['carbon']}")
        
        # Get feedback
        preferred_region = decision_maker.get_preference(pareto_front)
        
        if decision_maker.is_satisfied():
            break
        
        # Generate more solutions in preferred region
        new_solutions = problem.explore_region(preferred_region, n_solutions=10)
        pareto_front.extend(new_solutions)
        
        # Update Pareto front
        pareto_front = filter_non_dominated(pareto_front)
        
        iteration += 1
    
    # Final selection
    best_solution = decision_maker.select_final_solution(pareto_front)
    return best_solution
```

---

## Visualization

```python
def visualize_3d_pareto_front(solutions):
    """
    Visualize 3-objective Pareto front
    """
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    costs = [s['cost'] for s in solutions]
    services = [s['service'] for s in solutions]
    carbons = [s['carbon'] for s in solutions]
    
    scatter = ax.scatter(costs, services, carbons,
                        c=carbons, cmap='RdYlGn_r',
                        s=100, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Cost ($)', fontsize=12)
    ax.set_ylabel('Service Level', fontsize=12)
    ax.set_zlabel('Carbon Emissions (tons)', fontsize=12)
    ax.set_title('3D Pareto Frontier', fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, label='Carbon Emissions')
    plt.show()

def visualize_parallel_coordinates(pareto_front):
    """
    Parallel coordinates plot for many objectives
    """
    
    from pandas.plotting import parallel_coordinates
    import pandas as pd
    
    df = pd.DataFrame(pareto_front)
    df['Solution'] = range(len(df))
    
    plt.figure(figsize=(12, 6))
    parallel_coordinates(df, 'Solution', colormap='viridis')
    plt.title('Pareto Solutions - Parallel Coordinates')
    plt.ylabel('Normalized Objective Value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Tools & Libraries

**Python:**
- `pymoo`: Multi-objective optimization
- `platypus`: Evolutionary multi-objective
- `jmetal`: Multi-objective metaheuristics

**Commercial:**
- `modeFRONTIER`: Multi-objective design
- `CPLEX Multi-Objective`

---

## Related Skills

- **optimization-modeling**: single-objective optimization
- **metaheuristic-optimization**: NSGA-II, MOEA
- **sustainable-sourcing**: environmental objectives
- **network-design**: multi-objective network design


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
name: yard-management
description: When the user wants to optimize yard operations, manage trailer parking, or improve dock door utilization. Also use when the user mentions "yard management," "trailer tracking," "yard jockey," "drop trailer program," "trailer pool," "dock scheduling," or "gate management." For cross-dock operations, see cross-docking. For warehouse design, see warehouse-design.
---

# Yard Management

You are an expert in yard management and trailer logistics. Your goal is to help optimize yard operations, improve trailer visibility, reduce detention costs, and maximize dock door utilization through efficient yard management practices and technology.

## Initial Assessment

Before optimizing yard operations, understand:

1. **Facility Characteristics**
   - Yard size and capacity? (trailer spots)
   - Number of dock doors?
   - Layout constraints? (space, access, turning radius)
   - Gate security and check-in process?

2. **Operational Volume**
   - Daily inbound/outbound trailers?
   - Average dwell time per trailer?
   - Peak times and patterns?
   - Types of trailers? (dry van, reefer, flatbed)

3. **Current Challenges**
   - Trailer visibility issues?
   - Long wait times at gate or dock?
   - High detention/demurrage costs?
   - Difficulty finding trailers in yard?
   - Congestion at doors?

4. **Resources**
   - Number of yard jockeys?
   - Yard tractors available?
   - Technology in place? (YMS, GPS, RFID)
   - Staffing and shifts?

---

## Yard Management Framework

### Core Functions of Yard Management

**1. Gate Management**
- Check-in/check-out process
- Carrier credential verification
- BOL and documentation
- Safety inspections
- Appointment verification

**2. Yard Planning & Layout**
- Trailer parking locations
- Staging zones by priority
- Dock door assignments
- Traffic flow optimization

**3. Trailer Movement**
- Yard jockey dispatch
- Spotting trailers at doors
- Repositioning for loading/unloading
- Trailer pool management

**4. Tracking & Visibility**
- Real-time trailer location
- Load status (empty, loaded, in-process)
- Dwell time monitoring
- Exception management

**5. Dock Scheduling**
- Appointment booking
- Door assignment
- Load/unload coordination
- Carrier communication

---

## Yard Layout Optimization

### Yard Design Principles

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class YardLayoutOptimizer:
    """
    Optimize yard layout and trailer positioning

    Minimize jockey moves and door spotting time
    """

    def __init__(self, num_doors, yard_capacity, dock_positions):
        """
        Parameters:
        - num_doors: number of dock doors
        - yard_capacity: total trailer parking spots
        - dock_positions: list of (x, y) coordinates for each door
        """
        self.num_doors = num_doors
        self.yard_capacity = yard_capacity
        self.dock_positions = np.array(dock_positions)

    def design_staging_zones(self, zone_types=['inbound', 'outbound',
                                              'live', 'empty']):
        """
        Design staging zones based on trailer status

        Returns optimal zone assignments
        """

        # Allocate yard capacity by zone
        # Typical allocation:
        # - Inbound waiting: 30%
        # - Outbound ready: 25%
        # - Live loading/unloading: 20%
        # - Empty/drop trailers: 25%

        allocations = {
            'inbound': int(self.yard_capacity * 0.30),
            'outbound': int(self.yard_capacity * 0.25),
            'live': int(self.yard_capacity * 0.20),
            'empty': int(self.yard_capacity * 0.25)
        }

        # Position zones near relevant doors
        zones = {}

        # Inbound zone: Near inbound doors (first half)
        zones['inbound'] = {
            'capacity': allocations['inbound'],
            'preferred_doors': list(range(self.num_doors // 2)),
            'avg_distance_to_door': 50  # feet
        }

        # Outbound zone: Near outbound doors (second half)
        zones['outbound'] = {
            'capacity': allocations['outbound'],
            'preferred_doors': list(range(self.num_doors // 2, self.num_doors)),
            'avg_distance_to_door': 50
        }

        # Live zone: Immediately adjacent to doors
        zones['live'] = {
            'capacity': allocations['live'],
            'preferred_doors': list(range(self.num_doors)),
            'avg_distance_to_door': 20  # Closest
        }

        # Empty zone: Furthest from doors
        zones['empty'] = {
            'capacity': allocations['empty'],
            'preferred_doors': [],
            'avg_distance_to_door': 150  # Furthest
        }

        return zones

    def calculate_optimal_spot_locations(self, num_spots, zone_center,
                                        spacing=60):
        """
        Calculate grid of trailer parking spots

        Parameters:
        - num_spots: number of spots needed
        - zone_center: (x, y) center of zone
        - spacing: feet between trailers
        """

        # Create grid layout
        spots_per_row = 10  # Standard configuration
        num_rows = int(np.ceil(num_spots / spots_per_row))

        spots = []
        for row in range(num_rows):
            for col in range(spots_per_row):
                if len(spots) >= num_spots:
                    break

                x = zone_center[0] + (col * spacing)
                y = zone_center[1] + (row * spacing)

                spots.append({
                    'spot_id': f'S{len(spots)+1:03d}',
                    'position': (x, y),
                    'row': row,
                    'col': col
                })

        return spots

    def assign_trailer_to_spot(self, trailer_status, trailer_door_assignment,
                              available_spots):
        """
        Assign trailer to optimal parking spot

        Minimize distance to assigned door

        Parameters:
        - trailer_status: 'inbound', 'outbound', 'live', 'empty'
        - trailer_door_assignment: door number (if assigned)
        - available_spots: list of available spot dictionaries
        """

        # Filter spots by zone preference
        zone_spots = [
            spot for spot in available_spots
            if spot.get('zone') == trailer_status
        ]

        if not zone_spots:
            zone_spots = available_spots  # Use any available

        if not zone_spots:
            return None  # Yard full

        # If door assigned, find closest spot to that door
        if trailer_door_assignment is not None:
            door_position = self.dock_positions[trailer_door_assignment]

            # Calculate distances
            distances = [
                distance.euclidean(spot['position'], door_position)
                for spot in zone_spots
            ]

            # Select closest spot
            best_spot_idx = np.argmin(distances)
            assigned_spot = zone_spots[best_spot_idx]

        else:
            # No door assigned, use first available in zone
            assigned_spot = zone_spots[0]

        return assigned_spot

    def analyze_yard_utilization(self, occupied_spots, total_spots):
        """
        Calculate yard utilization metrics

        Returns utilization by zone and overall
        """

        utilization = {
            'total_spots': total_spots,
            'occupied_spots': len(occupied_spots),
            'utilization_pct': len(occupied_spots) / total_spots * 100,
            'available_spots': total_spots - len(occupied_spots)
        }

        # By zone
        zones = {}
        for spot in occupied_spots:
            zone = spot.get('zone', 'unknown')
            if zone not in zones:
                zones[zone] = 0
            zones[zone] += 1

        utilization['by_zone'] = zones

        return utilization

# Example usage
optimizer = YardLayoutOptimizer(
    num_doors=40,
    yard_capacity=200,
    dock_positions=[(i*20, 0) for i in range(40)]  # Doors in a line
)

zones = optimizer.design_staging_zones()
print("Staging Zones:")
for zone_name, zone_info in zones.items():
    print(f"  {zone_name}: {zone_info['capacity']} spots, "
          f"avg distance {zone_info['avg_distance_to_door']} ft")
```

---

## Dock Door Scheduling

### Appointment Scheduling System

```python
import pandas as pd
from datetime import datetime, timedelta

class DockSchedulingSystem:
    """
    Manage dock door appointments and scheduling

    Optimize door utilization and minimize wait times
    """

    def __init__(self, num_doors, hours_of_operation=(6, 22)):
        """
        Parameters:
        - num_doors: number of dock doors
        - hours_of_operation: (start_hour, end_hour) tuple
        """
        self.num_doors = num_doors
        self.start_hour = hours_of_operation[0]
        self.end_hour = hours_of_operation[1]
        self.schedule = {}

    def create_time_slots(self, date, slot_duration_hours=2):
        """
        Create available time slots for a date

        Returns list of time slots
        """

        slots = []
        current_time = datetime.combine(date, datetime.min.time()).replace(
            hour=self.start_hour
        )
        end_time = datetime.combine(date, datetime.min.time()).replace(
            hour=self.end_hour
        )

        while current_time < end_time:
            slot_end = current_time + timedelta(hours=slot_duration_hours)

            slots.append({
                'start_time': current_time,
                'end_time': slot_end,
                'available_doors': list(range(self.num_doors))
            })

            current_time = slot_end

        return slots

    def book_appointment(self, carrier, appointment_type, requested_time,
                        duration_hours=2, door_preference=None):
        """
        Book dock appointment

        Parameters:
        - carrier: carrier name
        - appointment_type: 'inbound' or 'outbound'
        - requested_time: datetime
        - duration_hours: expected duration
        - door_preference: specific door number (optional)
        """

        date = requested_time.date()

        # Get or create slots for date
        if date not in self.schedule:
            self.schedule[date] = self.create_time_slots(date)

        # Find matching time slot
        for slot in self.schedule[date]:
            if (slot['start_time'] <= requested_time <
                slot['end_time'] and
                len(slot['available_doors']) > 0):

                # Assign door
                if door_preference and door_preference in slot['available_doors']:
                    assigned_door = door_preference
                else:
                    assigned_door = slot['available_doors'][0]

                # Remove door from available
                slot['available_doors'].remove(assigned_door)

                appointment = {
                    'appointment_id': f"APT{len(self.schedule)*100 + 1}",
                    'carrier': carrier,
                    'type': appointment_type,
                    'scheduled_time': slot['start_time'],
                    'door': assigned_door,
                    'duration': duration_hours,
                    'status': 'scheduled'
                }

                return appointment

        # No available slot found
        return {
            'error': 'No available slot',
            'requested_time': requested_time,
            'suggestion': 'Try different time or date'
        }

    def check_availability(self, date, appointment_type=None):
        """
        Check door availability for a date

        Returns available slots
        """

        if date not in self.schedule:
            self.schedule[date] = self.create_time_slots(date)

        availability = []

        for slot in self.schedule[date]:
            if len(slot['available_doors']) > 0:
                availability.append({
                    'time_slot': f"{slot['start_time'].strftime('%H:%M')} - "
                                f"{slot['end_time'].strftime('%H:%M')}",
                    'available_doors': len(slot['available_doors']),
                    'door_numbers': slot['available_doors'][:5]  # Show first 5
                })

        return availability

    def calculate_utilization(self, date):
        """
        Calculate door utilization for a date

        Returns utilization percentage
        """

        if date not in self.schedule:
            return {'utilization': 0, 'message': 'No appointments scheduled'}

        total_door_slots = 0
        used_door_slots = 0

        for slot in self.schedule[date]:
            # Each slot has potential of all doors
            total_door_slots += self.num_doors

            # Count used doors (initially available - currently available)
            used_doors = self.num_doors - len(slot['available_doors'])
            used_door_slots += used_doors

        utilization = used_door_slots / total_door_slots * 100 if total_door_slots > 0 else 0

        return {
            'date': date,
            'utilization_pct': utilization,
            'total_door_slots': total_door_slots,
            'used_door_slots': used_door_slots,
            'target_utilization': 75  # Best practice target
        }

    def optimize_door_assignments(self, appointments):
        """
        Re-optimize door assignments to minimize moves

        Group similar appointment types on adjacent doors
        """

        # Separate by type
        inbound = [a for a in appointments if a['type'] == 'inbound']
        outbound = [a for a in appointments if a['type'] == 'outbound']

        # Assign inbound to first half of doors
        inbound_doors = list(range(self.num_doors // 2))
        outbound_doors = list(range(self.num_doors // 2, self.num_doors))

        # Reassign
        for idx, appt in enumerate(inbound):
            if idx < len(inbound_doors):
                appt['door'] = inbound_doors[idx]

        for idx, appt in enumerate(outbound):
            if idx < len(outbound_doors):
                appt['door'] = outbound_doors[idx]

        return appointments

# Example usage
scheduler = DockSchedulingSystem(num_doors=40)

# Book appointments
appt1 = scheduler.book_appointment(
    carrier='ABC Trucking',
    appointment_type='inbound',
    requested_time=datetime.now().replace(hour=8, minute=0)
)

print(f"Appointment booked: Door {appt1.get('door')} at "
      f"{appt1.get('scheduled_time')}")

# Check availability
tomorrow = datetime.now().date() + timedelta(days=1)
availability = scheduler.check_availability(tomorrow)
print(f"\nAvailability for {tomorrow}:")
for slot in availability[:3]:
    print(f"  {slot['time_slot']}: {slot['available_doors']} doors available")
```

---

## Trailer Tracking & Visibility

### Yard Management System (YMS) Core Functions

```python
class YardManagementSystem:
    """
    Core yard management system functionality

    Track trailers, manage moves, monitor dwell time
    """

    def __init__(self):
        self.trailers = {}  # trailer_id -> trailer info
        self.yard_spots = {}  # spot_id -> trailer_id
        self.move_history = []
        self.alerts = []

    def check_in_trailer(self, trailer_id, carrier, seal_number,
                        trailer_type='dry_van', is_loaded=True):
        """
        Check in trailer at gate

        Creates trailer record in system
        """

        check_in_time = datetime.now()

        trailer_info = {
            'trailer_id': trailer_id,
            'carrier': carrier,
            'seal_number': seal_number,
            'trailer_type': trailer_type,
            'is_loaded': is_loaded,
            'status': 'in_yard',
            'check_in_time': check_in_time,
            'current_location': 'gate',
            'dock_door': None,
            'moves': 0
        }

        self.trailers[trailer_id] = trailer_info

        # Log move
        self.move_history.append({
            'trailer_id': trailer_id,
            'timestamp': check_in_time,
            'action': 'check_in',
            'location': 'gate'
        })

        return trailer_info

    def assign_yard_spot(self, trailer_id, spot_id):
        """
        Assign trailer to yard parking spot

        Parameters:
        - trailer_id: unique trailer identifier
        - spot_id: yard spot identifier
        """

        if trailer_id not in self.trailers:
            return {'error': f'Trailer {trailer_id} not found'}

        if spot_id in self.yard_spots and self.yard_spots[spot_id] is not None:
            return {'error': f'Spot {spot_id} already occupied'}

        # Update trailer location
        trailer = self.trailers[trailer_id]
        old_location = trailer['current_location']
        trailer['current_location'] = spot_id
        trailer['moves'] += 1

        # Update spot
        if old_location in self.yard_spots:
            self.yard_spots[old_location] = None  # Free old spot

        self.yard_spots[spot_id] = trailer_id

        # Log move
        self.move_history.append({
            'trailer_id': trailer_id,
            'timestamp': datetime.now(),
            'action': 'move_to_spot',
            'from': old_location,
            'to': spot_id
        })

        return {
            'trailer_id': trailer_id,
            'assigned_spot': spot_id,
            'moves': trailer['moves']
        }

    def spot_trailer_at_door(self, trailer_id, door_number):
        """
        Spot trailer at dock door for loading/unloading

        Parameters:
        - trailer_id: trailer to spot
        - door_number: dock door number
        """

        if trailer_id not in self.trailers:
            return {'error': f'Trailer {trailer_id} not found'}

        trailer = self.trailers[trailer_id]
        old_location = trailer['current_location']

        # Update trailer
        trailer['current_location'] = f'door_{door_number}'
        trailer['dock_door'] = door_number
        trailer['status'] = 'at_door'
        trailer['door_arrival_time'] = datetime.now()
        trailer['moves'] += 1

        # Free old spot if in yard
        if old_location in self.yard_spots:
            self.yard_spots[old_location] = None

        # Log move
        self.move_history.append({
            'trailer_id': trailer_id,
            'timestamp': datetime.now(),
            'action': 'spot_at_door',
            'door': door_number,
            'from': old_location
        })

        return {
            'trailer_id': trailer_id,
            'door': door_number,
            'spotted_time': trailer['door_arrival_time']
        }

    def complete_door_activity(self, trailer_id):
        """
        Complete loading/unloading at door

        Move trailer back to yard or check out
        """

        if trailer_id not in self.trailers:
            return {'error': f'Trailer {trailer_id} not found'}

        trailer = self.trailers[trailer_id]

        if trailer['status'] != 'at_door':
            return {'error': 'Trailer not at door'}

        # Calculate door dwell time
        door_dwell = (datetime.now() - trailer['door_arrival_time']).total_seconds() / 3600

        trailer['status'] = 'completed'
        trailer['door_departure_time'] = datetime.now()
        trailer['door_dwell_hours'] = door_dwell

        # Log
        self.move_history.append({
            'trailer_id': trailer_id,
            'timestamp': datetime.now(),
            'action': 'complete_door_activity',
            'door_dwell_hours': door_dwell
        })

        # Check for excessive door time (>2 hours)
        if door_dwell > 2:
            self.alerts.append({
                'alert_type': 'excessive_door_dwell',
                'trailer_id': trailer_id,
                'door_dwell_hours': door_dwell,
                'timestamp': datetime.now()
            })

        return {
            'trailer_id': trailer_id,
            'door_dwell_hours': door_dwell,
            'status': 'completed'
        }

    def check_out_trailer(self, trailer_id):
        """
        Check out trailer from facility

        Final step before trailer leaves
        """

        if trailer_id not in self.trailers:
            return {'error': f'Trailer {trailer_id} not found'}

        trailer = self.trailers[trailer_id]

        # Calculate total yard dwell
        total_dwell = (datetime.now() - trailer['check_in_time']).total_seconds() / 3600

        trailer['status'] = 'checked_out'
        trailer['check_out_time'] = datetime.now()
        trailer['total_yard_dwell_hours'] = total_dwell

        # Log
        self.move_history.append({
            'trailer_id': trailer_id,
            'timestamp': datetime.now(),
            'action': 'check_out',
            'total_dwell_hours': total_dwell
        })

        # Alert if excessive yard dwell (>24 hours)
        if total_dwell > 24:
            self.alerts.append({
                'alert_type': 'excessive_yard_dwell',
                'trailer_id': trailer_id,
                'total_dwell_hours': total_dwell,
                'timestamp': datetime.now()
            })

        return {
            'trailer_id': trailer_id,
            'total_yard_dwell_hours': total_dwell,
            'total_moves': trailer['moves']
        }

    def get_yard_status(self):
        """
        Get current yard status summary

        Returns counts by status
        """

        status_counts = {}
        for trailer in self.trailers.values():
            status = trailer['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        total_trailers = len(self.trailers)
        occupied_spots = sum(1 for spot in self.yard_spots.values()
                           if spot is not None)

        return {
            'total_trailers_in_yard': total_trailers,
            'occupied_spots': occupied_spots,
            'by_status': status_counts,
            'active_alerts': len(self.alerts)
        }

    def find_trailer(self, trailer_id):
        """
        Locate trailer in yard

        Returns current location
        """

        if trailer_id not in self.trailers:
            return {'error': 'Trailer not found'}

        trailer = self.trailers[trailer_id]

        return {
            'trailer_id': trailer_id,
            'current_location': trailer['current_location'],
            'status': trailer['status'],
            'carrier': trailer['carrier'],
            'dwell_time_hours': (datetime.now() - trailer['check_in_time']).total_seconds() / 3600
        }

    def calculate_performance_metrics(self):
        """
        Calculate yard performance metrics

        Returns KPIs
        """

        if not self.trailers:
            return {'message': 'No data available'}

        # Average dwell time
        dwell_times = []
        for trailer in self.trailers.values():
            if 'total_yard_dwell_hours' in trailer:
                dwell_times.append(trailer['total_yard_dwell_hours'])

        avg_dwell = np.mean(dwell_times) if dwell_times else 0

        # Average moves per trailer
        moves = [t['moves'] for t in self.trailers.values()]
        avg_moves = np.mean(moves) if moves else 0

        # Door dwell times
        door_dwells = []
        for trailer in self.trailers.values():
            if 'door_dwell_hours' in trailer:
                door_dwells.append(trailer['door_dwell_hours'])

        avg_door_dwell = np.mean(door_dwells) if door_dwells else 0

        return {
            'avg_yard_dwell_hours': avg_dwell,
            'avg_moves_per_trailer': avg_moves,
            'avg_door_dwell_hours': avg_door_dwell,
            'total_trailers': len(self.trailers),
            'total_alerts': len(self.alerts),
            'target_yard_dwell_hours': 24,
            'target_door_dwell_hours': 2
        }

# Example usage
yms = YardManagementSystem()

# Check in trailer
trailer = yms.check_in_trailer(
    trailer_id='TRL12345',
    carrier='ABC Trucking',
    seal_number='SEAL987',
    is_loaded=True
)
print(f"Trailer {trailer['trailer_id']} checked in at {trailer['check_in_time']}")

# Assign to yard spot
yms.assign_yard_spot('TRL12345', 'S045')
print("Trailer assigned to spot S045")

# Spot at door
yms.spot_trailer_at_door('TRL12345', door_number=12)
print("Trailer spotted at door 12")

# Get yard status
status = yms.get_yard_status()
print(f"\nYard Status: {status['total_trailers_in_yard']} trailers in yard")
```

---

## Yard Jockey Optimization

### Jockey Dispatch & Task Management

```python
class YardJockeyDispatcher:
    """
    Optimize yard jockey task assignment

    Minimize moves and maximize productivity
    """

    def __init__(self, num_jockeys, yard_layout):
        self.num_jockeys = num_jockeys
        self.yard_layout = yard_layout
        self.jockeys = {
            f'Jockey_{i+1}': {
                'current_location': 'office',
                'status': 'available',
                'tasks_completed': 0,
                'total_distance': 0
            }
            for i in range(num_jockeys)
        }
        self.task_queue = []

    def add_move_task(self, trailer_id, from_location, to_location, priority='normal'):
        """
        Add trailer move task to queue

        Parameters:
        - priority: 'urgent', 'normal', 'low'
        """

        task = {
            'task_id': f'TASK{len(self.task_queue)+1:04d}',
            'trailer_id': trailer_id,
            'from': from_location,
            'to': to_location,
            'priority': priority,
            'status': 'queued',
            'created_time': datetime.now()
        }

        self.task_queue.append(task)

        # Sort by priority
        priority_order = {'urgent': 0, 'normal': 1, 'low': 2}
        self.task_queue.sort(
            key=lambda x: priority_order.get(x['priority'], 1)
        )

        return task

    def assign_next_task(self):
        """
        Assign next task to available jockey

        Uses nearest jockey to minimize deadhead
        """

        # Find available jockey
        available_jockeys = [
            (jid, jinfo) for jid, jinfo in self.jockeys.items()
            if jinfo['status'] == 'available'
        ]

        if not available_jockeys or not self.task_queue:
            return None

        # Get next task
        task = self.task_queue[0]

        # Find nearest jockey
        nearest_jockey = None
        min_distance = float('inf')

        for jockey_id, jockey_info in available_jockeys:
            # Calculate distance from jockey to task start location
            distance = self._calculate_distance(
                jockey_info['current_location'],
                task['from']
            )

            if distance < min_distance:
                min_distance = distance
                nearest_jockey = jockey_id

        if nearest_jockey:
            # Assign task
            self.jockeys[nearest_jockey]['status'] = 'busy'
            self.jockeys[nearest_jockey]['current_task'] = task['task_id']

            task['status'] = 'in_progress'
            task['assigned_jockey'] = nearest_jockey
            task['start_time'] = datetime.now()

            self.task_queue.pop(0)

            return {
                'task_id': task['task_id'],
                'jockey': nearest_jockey,
                'trailer': task['trailer_id'],
                'move': f"{task['from']} -> {task['to']}"
            }

        return None

    def complete_task(self, task_id):
        """
        Mark task as completed

        Update jockey status and location
        """

        # Find task
        for task in self.task_queue:
            if task['task_id'] == task_id:
                task['status'] = 'completed'
                task['completion_time'] = datetime.now()

                # Update jockey
                jockey_id = task.get('assigned_jockey')
                if jockey_id:
                    self.jockeys[jockey_id]['status'] = 'available'
                    self.jockeys[jockey_id]['current_location'] = task['to']
                    self.jockeys[jockey_id]['tasks_completed'] += 1

                    # Calculate distance
                    distance = self._calculate_distance(task['from'], task['to'])
                    self.jockeys[jockey_id]['total_distance'] += distance

                return {
                    'task_id': task_id,
                    'jockey': jockey_id,
                    'status': 'completed'
                }

        return {'error': 'Task not found'}

    def _calculate_distance(self, location1, location2):
        """Calculate distance between two locations (simplified)"""

        # In practice, use actual yard coordinates
        # Simplified: random distance 50-500 feet
        return np.random.randint(50, 500)

    def get_jockey_productivity(self):
        """
        Calculate jockey productivity metrics

        Returns moves per hour, utilization
        """

        productivity = []

        for jockey_id, jockey_info in self.jockeys.items():
            productivity.append({
                'jockey_id': jockey_id,
                'tasks_completed': jockey_info['tasks_completed'],
                'total_distance': jockey_info['total_distance'],
                'current_status': jockey_info['status']
            })

        return pd.DataFrame(productivity)

    def optimize_task_sequence(self, tasks):
        """
        Optimize sequence of tasks to minimize total distance

        Uses greedy nearest-neighbor approach
        """

        if not tasks:
            return []

        optimized_sequence = []
        remaining_tasks = tasks.copy()
        current_location = 'office'

        while remaining_tasks:
            # Find nearest task
            nearest_task = None
            min_distance = float('inf')

            for task in remaining_tasks:
                distance = self._calculate_distance(
                    current_location,
                    task['from']
                )

                if distance < min_distance:
                    min_distance = distance
                    nearest_task = task

            if nearest_task:
                optimized_sequence.append(nearest_task)
                remaining_tasks.remove(nearest_task)
                current_location = nearest_task['to']

        return optimized_sequence
```

---

## Common Challenges & Solutions

### Challenge: Trailer Visibility

**Problem:**
- Can't find trailers in yard
- Drivers search for 15-30 minutes
- Wasted time and frustration

**Solutions:**
- Implement YMS with GPS/RFID tracking
- Zone-based yard layout with clear signage
- Mobile app for drivers (trailer locator)
- Digital yard map with real-time updates
- Dedicated staging zones by status
- Color-coded yard spots
- Regular yard audits to verify locations

### Challenge: High Detention Costs

**Problem:**
- Paying detention fees ($50-100/hour)
- Trailers sitting at doors too long
- Slow loading/unloading

**Solutions:**
- Set hard time limits for door dwell (<2 hours)
- Monitor and alert on approaching detention
- Pre-stage loads (ready before truck arrives)
- Live loading/unloading where possible
- Negotiate detention grace periods
- Optimize dock scheduling (avoid overbooking)
- Cross-training to flex labor to doors
- Automated alerts at 75% of free time

### Challenge: Yard Congestion

**Problem:**
- Too many trailers, not enough space
- Difficulty maneuvering jockeys
- Blocked access to trailers

**Solutions:**
- Implement drop trailer program (pre-loaded outbound)
- Dedicated empty trailer pool off-site
- Just-in-time arrival scheduling
- Turn away non-appointment arrivals
- Expand yard capacity or use overflow lot
- Improve trailer turn time (reduce dwell)
- Better appointment scheduling (smooth arrivals)

### Challenge: Long Wait Times at Gate

**Problem:**
- Trucks waiting 30-60 minutes at gate
- Manual check-in process slow
- Paperwork errors and delays

**Solutions:**
- Implement online pre-check-in portal
- Use kiosks for self-check-in
- Dedicated lanes for pre-registered drivers
- Automate BOL scanning and validation
- Pre-approve appointments (pre-verify credentials)
- Add gate capacity (more lanes)
- Mobile check-in before arrival

### Challenge: Inefficient Jockey Utilization

**Problem:**
- Jockeys idle or making unnecessary moves
- Long deadhead distances
- Poor task prioritization

**Solutions:**
- Implement jockey dispatch system
- Zone-based jockey assignments
- Real-time task queue with priorities
- Optimize task sequencing (minimize distance)
- Right-size jockey staffing
- Cross-train warehouse staff as backup
- Performance metrics and incentives

### Challenge: Lack of Appointment Compliance

**Problem:**
- Carriers show up without appointments
- Early or late arrivals disrupt schedule
- Overbooking of doors

**Solutions:**
- Require appointments (enforce policy)
- Charge premium for non-appointment arrivals
- Communicate appointment importance
- Partner with carriers on compliance
- Send appointment reminders (day before, morning of)
- Track and report carrier compliance
- Refuse service to repeat offenders

---

## Yard Management Technology

### Yard Management System (YMS) Selection

**Enterprise YMS Platforms:**
- **C3 Solutions**: Industry leader
- **Zebra (formerly Yard Management Solutions)**: RFID-based
- **Manhattan Associates YMS**: WMS-integrated
- **Oracle Yard Management**: Cloud-based
- **Blue Yonder YMS**: AI-powered
- **4Sight Yard Management**: Mid-market

**Key YMS Features:**
- Real-time trailer tracking (GPS, RFID, manual)
- Gate check-in/check-out automation
- Dock appointment scheduling
- Jockey task management and dispatch
- Dwell time monitoring and alerts
- Reporting and analytics
- Integration with WMS and TMS

### Tracking Technologies

**RFID Tags:**
- Passive tags on trailers
- Readers at gates and key points
- Automatic location updates
- Cost: $5-10 per tag, $1K-5K per reader

**GPS Tracking:**
- Active GPS devices on trailers
- Real-time location accuracy
- Higher cost, requires power
- Cost: $50-150 per device + monthly fees

**Geofencing:**
- Virtual boundaries in yard
- Trigger alerts when crossed
- Works with GPS or RFID

**Barcode/QR Scanning:**
- Low-tech, manual scanning
- Mobile app for jockeys
- Lower accuracy, requires compliance

---

## Output Format

### Yard Management Analysis Report

**Executive Summary:**
- Average yard dwell time: 18.5 hours (target: <12 hours)
- Detention costs: $18,500/month (target: <$10,000)
- Door utilization: 62% (target: 75%)
- Yard capacity utilization: 85% (near capacity)
- Recommendation: Implement YMS, improve dock scheduling

**Current State Metrics:**

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Avg yard dwell time | 18.5 hrs | <12 hrs | ⚠️ 54% over |
| Avg door dwell time | 2.8 hrs | <2 hrs | ⚠️ 40% over |
| Detention costs/month | $18.5K | <$10K | ⚠️ 85% over |
| Gate wait time | 22 min | <10 min | ⚠️ 120% over |
| Door utilization | 62% | 75% | ⚠️ Below target |
| Yard occupancy | 85% | 80% | ⚠️ Near capacity |

**Detention Cost Analysis:**

| Carrier | Monthly Charges | Incidents | Avg Duration | Root Cause |
|---------|----------------|-----------|--------------|------------|
| Carrier A | $6,200 | 42 | 3.2 hrs | Slow unloading |
| Carrier B | $4,800 | 35 | 2.9 hrs | Dock congestion |
| Carrier C | $3,500 | 28 | 2.6 hrs | Missing appointments |
| Others | $4,000 | 32 | 2.5 hrs | Various |

**Yard Dwell Time by Status:**

| Trailer Status | Count | Avg Dwell | Max Dwell | % Over 24 hrs |
|---------------|-------|-----------|-----------|---------------|
| Inbound staged | 45 | 12.5 hrs | 38 hrs | 15% |
| At door (loading) | 18 | 2.8 hrs | 4.5 hrs | 0% |
| Outbound ready | 32 | 28.0 hrs | 72 hrs | 45% ⚠️ |
| Empty/Drop | 25 | 36.0 hrs | 120 hrs | 60% ⚠️ |

**Root Cause: Outbound Delays**
- Waiting for consolidation (3+ days)
- No carrier pickup scheduled
- Recommendation: Daily pickup schedule or use 3PL

**Door Utilization by Day/Time:**

| Time Slot | Mon | Tue | Wed | Thu | Fri | Avg |
|-----------|-----|-----|-----|-----|-----|-----|
| 6-9 AM | 85% | 82% | 88% | 90% | 85% | 86% |
| 9-12 PM | 75% | 70% 72% | 78% | 75% | 74% |
| 12-3 PM | 55% | 52% | 58% | 60% | 55% | 56% ⚠️ |
| 3-6 PM | 48% | 45% | 50% | 52% | 48% | 49% ⚠️ |

**Recommendation: Evening shift to utilize afternoons**

**Improvement Initiatives:**

1. **Implement YMS** - Impact: -30% dwell time, -40% detention
   - Real-time trailer tracking
   - Automated door scheduling
   - Jockey dispatch optimization
   - Investment: $180K, ROI: 14 months

2. **Optimize Dock Scheduling** - Impact: +13% door utilization
   - Implement appointment system
   - Enforce appointment compliance
   - Balance arrivals throughout day
   - Investment: $25K (software)

3. **Reduce Outbound Dwell** - Impact: -50% yard congestion
   - Daily carrier pickup schedule
   - Pre-loaded outbound staging
   - Drop trailer program
   - Savings: $120K annually

4. **Expand Gate Capacity** - Impact: -60% wait time
   - Add second gate lane
   - Self-service kiosk check-in
   - Pre-registration portal
   - Investment: $75K

**Expected Results (12 months):**

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Avg yard dwell | 18.5 hrs | 12 hrs | -35% |
| Detention costs | $18.5K/mo | $9K/mo | -51% |
| Door utilization | 62% | 75% | +13 pts |
| Gate wait time | 22 min | 8 min | -64% |
| Yard occupancy | 85% | 70% | -15 pts |

---

## Questions to Ask

If you need more context:
1. How many dock doors and yard spots?
2. What's your daily trailer volume (in/out)?
3. Do you have a YMS or tracking system?
4. What's your average yard dwell time?
5. Are you paying detention costs? How much?
6. Any appointment scheduling system?
7. How many yard jockeys/tractors?
8. Main pain points? (visibility, congestion, detention, wait times)

---

## Related Skills

- **dock-door-assignment**: Optimize dock door scheduling and assignment
- **cross-docking**: Cross-dock operations and flow-through
- **warehouse-design**: Facility layout and design
- **route-optimization**: Outbound routing and delivery
- **freight-optimization**: Carrier management and transportation
- **supply-chain-automation**: Automation and technology selection
- **process-optimization**: Operational process improvement
- **maintenance-planning**: Equipment and yard tractor maintenance

---
name: workforce-scheduling
description: When the user wants to optimize workforce scheduling, create shift plans, or balance labor demand. Also use when the user mentions "staff scheduling," "labor planning," "shift optimization," "crew scheduling," "roster optimization," or "employee scheduling." For task assignment, see task-assignment-problem. For wave planning labor, see wave-planning-optimization.
---

# Workforce Scheduling

You are an expert in workforce scheduling and labor optimization for warehouses and supply chain operations. Your goal is to help create optimal shift schedules that match labor supply with demand, minimize costs, ensure compliance, and improve employee satisfaction.

## Initial Assessment

Before optimizing workforce scheduling, understand:

1. **Labor Demand**
   - Daily/weekly order volume patterns?
   - Peak periods and seasonality?
   - Tasks to be performed (picking, packing, receiving)?
   - Required skills and certifications?
   - Service level targets (on-time shipping)?

2. **Labor Supply**
   - Total workforce size (full-time, part-time, temp)?
   - Employee availability and preferences?
   - Skill levels and cross-training?
   - Shift length preferences (8hr, 10hr, 12hr)?
   - Union rules and labor agreements?

3. **Business Constraints**
   - Operating hours (24/5, 24/7, day shift only)?
   - Minimum staffing levels?
   - Maximum consecutive days worked?
   - Overtime rules and costs?
   - Break and meal period requirements?
   - Weekend and holiday staffing needs?

4. **Current State**
   - Current scheduling method (manual, software)?
   - Current labor costs (regular + OT)?
   - Labor utilization rates?
   - Employee satisfaction with schedules?
   - Schedule change frequency?

---

## Workforce Scheduling Framework

### Scheduling Objectives

**Primary Goals:**
1. **Match Demand**: Ensure sufficient labor for forecasted workload
2. **Minimize Cost**: Optimize mix of regular hours, overtime, and temps
3. **Maximize Utilization**: Reduce idle time and overstaffing
4. **Employee Satisfaction**: Consider preferences, fairness, work-life balance
5. **Compliance**: Meet labor laws, union rules, company policies

**Key Metrics:**
- Labor cost per unit ($/order, $/line picked)
- Labor utilization % (productive time / scheduled time)
- Schedule efficiency (actual vs. planned labor hours)
- Employee turnover and absenteeism
- Overtime % (OT hours / total hours)

### Shift Design Strategies

**1. Fixed Shifts**
- Same schedule every week
- **Pros**: Predictable, easy to plan life around
- **Cons**: Inflexible, may not match demand
- **Use**: Stable demand, union environments

**2. Rotating Shifts**
- Employees rotate through different shifts
- **Pros**: Fair distribution of undesirable shifts
- **Cons**: Disrupts circadian rhythms, harder on employees
- **Use**: 24/7 operations, fairness priority

**3. Flexible/Variable Shifts**
- Shift start times and lengths vary
- **Pros**: Matches demand, reduces costs
- **Cons**: Unpredictable for employees, harder to schedule
- **Use**: Variable demand, high labor cost sensitivity

**4. Compressed Workweeks**
- 4×10hr or 3×12hr instead of 5×8hr
- **Pros**: Fewer workdays, employee preference, coverage
- **Cons**: Fatigue, may require premium pay
- **Use**: Continuous operations, employee retention

**5. On-Call/Flex Pool**
- Variable hours based on need
- **Pros**: Maximum flexibility, cost-effective
- **Cons**: Unpredictable for workers, may increase turnover
- **Use**: Peak periods, backup capacity

---

## Mathematical Formulation

### Workforce Scheduling as Optimization Problem

**Decision Variables:**
- x[i,s,d] = 1 if employee i works shift s on day d, 0 otherwise
- y[s,d] = number of employees on shift s on day d
- o[i,d] = overtime hours for employee i on day d

**Parameters:**
- D[s,d] = labor demand (hours) for shift s on day d
- C_reg = regular hourly wage
- C_ot = overtime hourly wage (typically 1.5× regular)
- C_temp = temporary worker hourly wage
- A[i,s,d] = availability of employee i for shift s on day d (0 or 1)
- H[s] = length of shift s (hours)
- Max_hours[i] = maximum hours per week for employee i
- Min_rest = minimum hours between shifts

**Objective Function:**

```
Minimize:
  Total Labor Cost = Regular + Overtime + Temp + Penalties

Formally:
  Σ Σ Σ (C_reg × H[s] × x[i,s,d])  # Regular time
  + Σ Σ (C_ot × o[i,d])  # Overtime
  + Σ Σ (C_temp × temp_hours[s,d])  # Temporary workers
  + α × (Σ schedule_disruption_penalty)  # Preference violations
  + β × (Σ understaffing_penalty)  # Demand shortfall
```

**Constraints:**

```python
# 1. Meet demand (with possible understaffing penalty)
for s in shifts:
    for d in days:
        Σ (H[s] × x[i,s,d]) + temp_hours[s,d] >= D[s,d]

# 2. Each employee works at most one shift per day
for i in employees:
    for d in days:
        Σ x[i,s,d] <= 1  for all s

# 3. Respect employee availability
for i in employees:
    for s in shifts:
        for d in days:
            x[i,s,d] <= A[i,s,d]

# 4. Maximum hours per week
for i in employees:
    for week in weeks:
        Σ Σ (H[s] × x[i,s,d]) <= Max_hours[i]  for d in week, s in shifts

# 5. Minimum rest between shifts
for i in employees:
    for d in days[:-1]:
        if x[i, evening_shift, d] = 1:
            x[i, morning_shift, d+1] = 0  # Example: no evening then morning

# 6. Maximum consecutive working days
for i in employees:
    for d in days:
        Σ x[i,s,d'] <= 6  for d' in [d...d+6], s in shifts

# 7. Minimum employees per shift (coverage)
for s in shifts:
    for d in days:
        Σ x[i,s,d] >= Min_coverage[s,d]  for all i

# 8. Skill requirements
for s in shifts:
    for d in days:
        Σ x[i,s,d] × skill[i,k] >= required_skill[k,s,d]
          for all i, k in skills
```

---

## Scheduling Algorithms

### Greedy Demand-Driven Scheduling

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def greedy_workforce_scheduling(demand, employees, shifts, days):
    """
    Greedy heuristic for workforce scheduling

    Algorithm:
    1. Sort days by demand (highest first)
    2. For each day, assign employees to meet demand
    3. Prioritize employees with availability and low weekly hours

    Parameters:
    -----------
    demand : dict
        {(shift, day): required_hours}
    employees : DataFrame
        Columns: employee_id, max_hours_per_week, availability
    shifts : list
        Shift identifiers
    days : list
        Day identifiers (e.g., dates)

    Returns:
    --------
    Schedule assignments
    """

    # Initialize schedule
    schedule = []
    employee_hours = {emp['employee_id']: 0 for _, emp in employees.iterrows()}

    # Sort (shift, day) pairs by demand
    demand_sorted = sorted(demand.items(), key=lambda x: x[1], reverse=True)

    for (shift, day), required_hours in demand_sorted:
        assigned_hours = 0

        # Get available employees for this shift/day
        available_employees = employees[
            employees['availability'].apply(lambda x: (shift, day) in x)
        ].copy()

        # Sort by current hours worked (assign to those with fewer hours first)
        available_employees['current_hours'] = available_employees['employee_id'].map(employee_hours)
        available_employees = available_employees.sort_values('current_hours')

        # Assign employees until demand met
        for idx, emp in available_employees.iterrows():
            emp_id = emp['employee_id']
            shift_length = 8  # Assume 8-hour shifts

            # Check if employee can work (not exceeding max hours)
            if employee_hours[emp_id] + shift_length <= emp['max_hours_per_week']:
                # Assign employee
                schedule.append({
                    'employee_id': emp_id,
                    'shift': shift,
                    'day': day,
                    'hours': shift_length
                })

                employee_hours[emp_id] += shift_length
                assigned_hours += shift_length

                if assigned_hours >= required_hours:
                    break

        # Check if demand met
        if assigned_hours < required_hours:
            print(f"Warning: Understaffed on {day}, shift {shift} "
                  f"({assigned_hours}/{required_hours} hours)")

    return pd.DataFrame(schedule)


# Example usage
employees = pd.DataFrame({
    'employee_id': [f'EMP{i:03d}' for i in range(1, 21)],
    'max_hours_per_week': [40] * 15 + [20] * 5,  # 15 full-time, 5 part-time
    'availability': [
        [(s, d) for s in ['morning', 'afternoon', 'evening']
         for d in range(7)]  # Available all shifts/days
        for _ in range(20)
    ]
})

shifts = ['morning', 'afternoon', 'evening']
days = list(range(7))  # Monday=0, Sunday=6

# Demand varies by day and shift
demand = {
    (shift, day): np.random.randint(40, 120)
    for shift in shifts for day in days
}

# Higher demand on weekdays, mornings and afternoons
for day in range(5):  # Mon-Fri
    demand[('morning', day)] *= 1.5
    demand[('afternoon', day)] *= 1.3

schedule = greedy_workforce_scheduling(demand, employees, shifts, days)

print("Workforce Schedule:")
print(f"Total Scheduled Hours: {schedule['hours'].sum()}")
print(f"Employees Scheduled: {schedule['employee_id'].nunique()}")
print(f"\nSchedule by Shift:")
print(schedule.groupby('shift')['hours'].sum())
```

### Integer Programming Model

```python
from pulp import *

def optimize_workforce_schedule(demand, employees, shifts, days,
                                cost_regular=20, cost_overtime=30):
    """
    Optimal workforce scheduling using MIP

    Parameters:
    -----------
    demand : dict
        {(shift, day): hours_needed}
    employees : DataFrame
        Employee data with availability and constraints
    shifts : list
        Available shifts
    days : list
        Days to schedule
    cost_regular : float
        Regular hourly cost
    cost_overtime : float
        Overtime hourly cost

    Returns:
    --------
    Optimal schedule
    """

    prob = LpProblem("Workforce_Scheduling", LpMinimize)

    # Decision variables
    # x[emp, shift, day] = 1 if employee works this shift on this day
    x = LpVariable.dicts("assign",
                        [(emp['employee_id'], shift, day)
                         for _, emp in employees.iterrows()
                         for shift in shifts
                         for day in days],
                        cat='Binary')

    # Overtime hours variables
    overtime = LpVariable.dicts("overtime",
                               [(emp['employee_id'], day)
                                for _, emp in employees.iterrows()
                                for day in days],
                               lowBound=0,
                               cat='Continuous')

    # Understaffing variables (soft constraint)
    understaffed = LpVariable.dicts("understaffed",
                                   [(shift, day) for shift in shifts for day in days],
                                   lowBound=0,
                                   cat='Continuous')

    # Objective: minimize cost
    shift_hours = 8  # Assume 8-hour shifts

    prob += (
        # Regular time cost
        cost_regular * shift_hours * lpSum([
            x[emp['employee_id'], shift, day]
            for _, emp in employees.iterrows()
            for shift in shifts
            for day in days
        ]) +

        # Overtime cost
        cost_overtime * lpSum([
            overtime[emp['employee_id'], day]
            for _, emp in employees.iterrows()
            for day in days
        ]) +

        # Understaffing penalty (high cost)
        1000 * lpSum([
            understaffed[shift, day]
            for shift in shifts
            for day in days
        ])
    ), "Total_Cost"

    # Constraints

    # 1. Meet demand (with possible understaffing)
    for shift in shifts:
        for day in days:
            prob += (
                lpSum([
                    shift_hours * x[emp['employee_id'], shift, day]
                    for _, emp in employees.iterrows()
                ]) + understaffed[shift, day] >= demand.get((shift, day), 0)
            ), f"Demand_{shift}_{day}"

    # 2. Each employee works at most one shift per day
    for _, emp in employees.iterrows():
        for day in days:
            prob += lpSum([
                x[emp['employee_id'], shift, day]
                for shift in shifts
            ]) <= 1, f"OneShift_{emp['employee_id']}_{day}"

    # 3. Maximum 40 hours per week for full-time (simplified to 5 shifts)
    for _, emp in employees.iterrows():
        max_shifts = emp['max_hours_per_week'] // shift_hours

        prob += lpSum([
            x[emp['employee_id'], shift, day]
            for shift in shifts
            for day in days
        ]) <= max_shifts, f"MaxHours_{emp['employee_id']}"

    # 4. Calculate overtime (hours beyond 40)
    for _, emp in employees.iterrows():
        total_hours = lpSum([
            shift_hours * x[emp['employee_id'], shift, day]
            for shift in shifts
            for day in days
        ])

        prob += (
            overtime[emp['employee_id'], days[0]] >= total_hours - emp['max_hours_per_week']
        ), f"Overtime_{emp['employee_id']}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract solution
    schedule = []
    for _, emp in employees.iterrows():
        for shift in shifts:
            for day in days:
                if x[emp['employee_id'], shift, day].varValue > 0.5:
                    schedule.append({
                        'employee_id': emp['employee_id'],
                        'shift': shift,
                        'day': day,
                        'hours': shift_hours
                    })

    return {
        'status': LpStatus[prob.status],
        'total_cost': value(prob.objective),
        'schedule': pd.DataFrame(schedule)
    }


result = optimize_workforce_schedule(demand, employees, shifts, days)

print(f"\nOptimization Status: {result['status']}")
print(f"Total Cost: ${result['total_cost']:,.2f}")
print(f"\nSchedule Summary:")
print(result['schedule'].groupby('shift').size())
```

---

## Advanced Scheduling Techniques

### Shift Bidding and Preference-Based Scheduling

```python
class PreferenceBasedScheduler:
    """
    Schedule based on employee preferences and seniority
    """

    def __init__(self, employees, shifts, days):
        self.employees = employees
        self.shifts = shifts
        self.days = days
        self.schedule = []

    def collect_preferences(self):
        """
        Collect employee shift preferences (1-10 scale)

        In practice, would come from employee input system
        """

        preferences = {}

        for _, emp in self.employees.iterrows():
            emp_id = emp['employee_id']
            preferences[emp_id] = {}

            for shift in self.shifts:
                for day in self.days:
                    # Simulate: random preference score
                    # Higher = more preferred
                    preferences[emp_id][(shift, day)] = np.random.randint(1, 11)

        return preferences

    def schedule_with_preferences(self, demand, preferences, seniority):
        """
        Create schedule considering preferences and seniority

        Algorithm:
        1. Sort employees by seniority
        2. Senior employees pick preferred shifts first
        3. Fill remaining shifts with junior employees
        4. Balance to meet demand

        Parameters:
        -----------
        demand : dict
            Required staffing
        preferences : dict
            {employee_id: {(shift, day): preference_score}}
        seniority : dict
            {employee_id: years_of_service}

        Returns:
        --------
        Schedule with preference satisfaction
        """

        # Sort employees by seniority
        employees_sorted = sorted(
            self.employees['employee_id'],
            key=lambda emp_id: seniority.get(emp_id, 0),
            reverse=True
        )

        remaining_demand = demand.copy()
        employee_assignments = {emp: 0 for emp in employees_sorted}

        schedule = []

        # Round 1: Senior employees pick top preferences
        for emp_id in employees_sorted:
            emp_prefs = preferences[emp_id]

            # Get top 5 preferred shifts
            top_prefs = sorted(emp_prefs.items(),
                             key=lambda x: x[1],
                             reverse=True)[:5]

            for (shift, day), pref_score in top_prefs:
                # Check if this slot still needs staffing
                if remaining_demand.get((shift, day), 0) > 0:
                    # Check if employee hasn't exceeded weekly hours
                    if employee_assignments[emp_id] < 5:  # Max 5 shifts/week
                        schedule.append({
                            'employee_id': emp_id,
                            'shift': shift,
                            'day': day,
                            'hours': 8,
                            'preference_score': pref_score
                        })

                        employee_assignments[emp_id] += 1
                        remaining_demand[(shift, day)] -= 8

                        break  # Move to next employee

        # Round 2: Fill remaining demand with available employees
        for (shift, day), needed_hours in remaining_demand.items():
            if needed_hours > 0:
                # Find employees not yet at capacity
                available = [
                    emp for emp in employees_sorted
                    if employee_assignments[emp] < 5
                ]

                for emp_id in available:
                    if needed_hours <= 0:
                        break

                    schedule.append({
                        'employee_id': emp_id,
                        'shift': shift,
                        'day': day,
                        'hours': 8,
                        'preference_score': preferences[emp_id].get((shift, day), 0)
                    })

                    employee_assignments[emp_id] += 1
                    needed_hours -= 8

        schedule_df = pd.DataFrame(schedule)

        # Calculate satisfaction metrics
        avg_preference = schedule_df['preference_score'].mean()
        pref_above_7 = (schedule_df['preference_score'] >= 7).sum() / len(schedule_df) * 100

        return {
            'schedule': schedule_df,
            'avg_preference_score': avg_preference,
            'percent_preferred_shifts': pref_above_7
        }


# Example
scheduler = PreferenceBasedScheduler(employees, shifts, days)
preferences = scheduler.collect_preferences()
seniority = {f'EMP{i:03d}': np.random.randint(1, 15) for i in range(1, 21)}

result_pref = scheduler.schedule_with_preferences(demand, preferences, seniority)

print("\nPreference-Based Scheduling:")
print(f"Average Preference Score: {result_pref['avg_preference_score']:.2f}/10")
print(f"Preferred Shifts (7+): {result_pref['percent_preferred_shifts']:.1f}%")
```

### Dynamic Scheduling with Real-Time Adjustments

```python
class DynamicScheduleManager:
    """
    Manage real-time schedule adjustments

    Handle call-outs, demand surges, schedule swaps
    """

    def __init__(self, base_schedule, employee_pool):
        self.base_schedule = base_schedule
        self.employee_pool = employee_pool
        self.adjustments = []

    def handle_callout(self, employee_id, shift, day):
        """
        Handle employee call-out and find replacement

        Priority:
        1. Overtime for already-scheduled employees
        2. On-call employees
        3. Temporary workers
        """

        print(f"Call-out: {employee_id} for {shift} on day {day}")

        # Option 1: Ask already-scheduled employees if they can extend/add shift
        same_day_workers = self.base_schedule[
            (self.base_schedule['day'] == day) &
            (self.base_schedule['employee_id'] != employee_id)
        ]

        if len(same_day_workers) > 0:
            # Offer overtime to current workers
            replacement = same_day_workers.iloc[0]['employee_id']
            print(f"  Replacement: {replacement} (overtime)")

            self.adjustments.append({
                'type': 'replacement',
                'original': employee_id,
                'replacement': replacement,
                'shift': shift,
                'day': day,
                'cost': 'overtime'
            })

            return replacement

        # Option 2: Call on-call employee
        on_call = self.employee_pool[self.employee_pool['type'] == 'on_call']

        if len(on_call) > 0:
            replacement = on_call.iloc[0]['employee_id']
            print(f"  Replacement: {replacement} (on-call)")

            self.adjustments.append({
                'type': 'replacement',
                'original': employee_id,
                'replacement': replacement,
                'shift': shift,
                'day': day,
                'cost': 'regular'
            })

            return replacement

        # Option 3: Hire temp worker
        print(f"  Replacement: TEMP (temporary agency)")

        self.adjustments.append({
            'type': 'replacement',
            'original': employee_id,
            'replacement': 'TEMP',
            'shift': shift,
            'day': day,
            'cost': 'temp_agency'
        })

        return 'TEMP'

    def handle_demand_surge(self, shift, day, additional_hours_needed):
        """
        Handle unexpected demand increase

        Options:
        1. Extend shifts (overtime)
        2. Call in off-duty employees
        3. Hire temps
        """

        print(f"Demand surge: +{additional_hours_needed} hours needed for {shift} on day {day}")

        # Option 1: Extend current shift
        current_workers = self.base_schedule[
            (self.base_schedule['shift'] == shift) &
            (self.base_schedule['day'] == day)
        ]

        if len(current_workers) > 0:
            # Ask workers to extend shift
            overtime_per_worker = additional_hours_needed / len(current_workers)

            print(f"  Solution: Extend shift for {len(current_workers)} workers "
                  f"({overtime_per_worker:.1f} OT hours each)")

            self.adjustments.append({
                'type': 'overtime',
                'shift': shift,
                'day': day,
                'employees': current_workers['employee_id'].tolist(),
                'ot_hours': overtime_per_worker
            })

            return 'overtime'

        # Option 2: Call in off-duty
        # (implementation similar to call-out)

        return 'temp'


# Example usage
manager = DynamicScheduleManager(schedule, employees)

# Simulate call-out
manager.handle_callout('EMP005', 'morning', 2)

# Simulate demand surge
manager.handle_demand_surge('afternoon', 3, 24)

print(f"\nTotal Adjustments: {len(manager.adjustments)}")
```

---

## Tools & Libraries

### Workforce Scheduling Software

**Specialized Scheduling:**
- **Workforce Software (Kronos)**: Enterprise scheduling and time tracking
- **ADP Workforce Now**: Integrated HR and scheduling
- **Shiftboard**: Shift scheduling and labor management
- **When I Work**: Employee scheduling app
- **Deputy**: Workforce management platform
- **7shifts**: Restaurant/retail scheduling
- **Humanity (TCP)**: Employee scheduling and tracking

**WMS with Labor Management:**
- **Manhattan WMS**: Labor management system (LMS)
- **Blue Yonder (JDA) WMS**: Labor planning and scheduling
- **SAP EWM**: Integrated labor management
- **HighJump WMS**: LMS module with engineered standards

### Python Libraries

```python
# Optimization
from pulp import *
from ortools.sat.python import cp_model

# Scheduling
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constraint Programming
from constraint import Problem, AllDifferentConstraint
```

---

## Common Challenges & Solutions

### Challenge: Unpredictable Demand

**Problem:**
- Daily order volume varies ±30%
- Can't predict staffing needs accurately
- Either overstaffed (wasted cost) or understaffed (missed shipments)

**Solutions:**
- Flexible workforce (core + flex pool)
- On-call employees (4-hour notice)
- Cross-train for multiple tasks
- Dynamic scheduling (adjust intraday)
- Overtime as buffer (expensive but effective)
- Partner with temp agencies (fast ramp-up)

### Challenge: Employee Availability Constraints

**Problem:**
- Students only available evenings/weekends
- Parents need specific hours (drop-off/pick-up)
- Second jobs limit availability
- Many unavailability requests

**Solutions:**
- Self-service shift bidding system
- Build larger workforce with part-time
- Premium pay for less desirable shifts
- Advance notice for schedule (2+ weeks)
- Allow shift swaps (peer-to-peer)
- Honor availability constraints in optimization

### Challenge: Fairness and Morale

**Problem:**
- Some employees always get weekends off
- Senior employees get best shifts
- Perceived favoritism
- Low morale affects productivity

**Solutions:**
- Transparent scheduling rules
- Rotate undesirable shifts fairly
- Seniority-based shift bidding (fair process)
- Equal distribution of weekend shifts
- Track and publish fairness metrics
- Anonymous feedback on scheduling

### Challenge: Skill Mix Requirements

**Problem:**
- Not all employees can do all tasks
- Need certified forklift operators
- Quality control requires experienced workers
- Can't schedule purely on availability

**Solutions:**
- Track skills in employee database
- Include skill constraints in optimization
- Minimum skilled workers per shift
- Cross-training programs (expand skill base)
- Pay premiums for critical skills
- Certification tracking and renewal

### Challenge: Last-Minute Changes

**Problem:**
- Call-outs (sick, emergency)
- Demand spikes (unexpected large order)
- Equipment breakdown (need more labor)
- Schedule becomes obsolete

**Solutions:**
- On-call staff pool (10-15% of workforce)
- Automated call-out notification
- Temp agency on retainer
- Overtime authorization rules
- Real-time schedule adjustment app
- Plan for 5-10% call-out rate

---

## Output Format

### Workforce Schedule Report

**Weekly Schedule - Week of January 15-21, 2024**

**Schedule Summary:**

| Day | Shift | Employees | Total Hours | Demand (hrs) | Utilization |
|-----|-------|-----------|-------------|--------------|-------------|
| Mon | Morning | 12 | 96 | 92 | 96% |
| Mon | Afternoon | 10 | 80 | 78 | 98% |
| Mon | Evening | 6 | 48 | 45 | 94% |
| Tue | Morning | 11 | 88 | 85 | 97% |
| ... | ... | ... | ... | ... | ... |

**Employee Assignments:**

```
Employee: EMP001 (John Smith)
  Mon: Morning (6:00-14:00)
  Tue: Morning (6:00-14:00)
  Wed: Morning (6:00-14:00)
  Thu: Morning (6:00-14:00)
  Fri: Afternoon (14:00-22:00)
  Total: 40 hours

Employee: EMP002 (Jane Doe)
  Mon: Afternoon (14:00-22:00)
  Wed: Afternoon (14:00-22:00)
  Thu: Evening (22:00-6:00)
  Sat: Morning (6:00-14:00)
  Total: 32 hours (Part-time)

...
```

**Cost Analysis:**

| Category | Hours | Rate | Cost |
|----------|-------|------|------|
| Regular Time | 1,520 | $20/hr | $30,400 |
| Overtime | 85 | $30/hr | $2,550 |
| Temporary | 40 | $25/hr | $1,000 |
| **Total** | **1,645** | - | **$33,950** |

**Performance Metrics:**

- Average Utilization: 95%
- Overtime %: 5.2%
- Employee Satisfaction (Preferences): 8.2/10
- Coverage: 100% (no understaffed shifts)
- Cost per Hour: $20.64

**Schedule Compliance:**

- ✓ All shifts meet minimum staffing
- ✓ No employees exceed 40 regular hours
- ✓ Minimum 11-hour rest between shifts
- ✓ No more than 6 consecutive days worked
- ✓ Skill requirements met (forklift, QC)

---

## Questions to Ask

If you need more context:
1. What are your operating hours and shift structure?
2. What's your workforce size (full-time, part-time)?
3. How does demand vary (daily, weekly, seasonally)?
4. What scheduling constraints exist (union, overtime rules)?
5. What's your current scheduling method?
6. What's your labor cost structure (regular, OT, temp)?
7. Do employees have availability constraints or preferences?
8. Are there skill or certification requirements?

---

## Related Skills

- **task-assignment-problem**: For assigning workers to specific tasks
- **capacity-planning**: For long-term workforce planning
- **wave-planning-optimization**: For planning pick waves with labor
- **demand-forecasting**: For predicting labor demand
- **constraint-programming**: For complex scheduling constraints
- **optimization-modeling**: For mathematical scheduling models
- **production-scheduling**: For manufacturing workforce scheduling

---
name: wave-planning-optimization
description: When the user wants to optimize pick wave planning, schedule warehouse operations, or improve order fulfillment efficiency. Also use when the user mentions "wave management," "batch picking," "pick wave scheduling," "order release optimization," "wave design," or "pick wave strategy." For order batching, see order-batching-optimization. For workforce scheduling, see workforce-scheduling.
---

# Wave Planning Optimization

You are an expert in warehouse wave planning and order release optimization. Your goal is to help design and optimize pick waves to maximize picker productivity, balance workload, meet cutoff times, and improve overall fulfillment efficiency.

## Initial Assessment

Before optimizing wave planning, understand:

1. **Operational Characteristics**
   - Daily order volume (lines and units)?
   - Order types (each-pick, case-pick, full-pallet)?
   - Warehouse zones and pick methods?
   - Shift structure and available labor?
   - Current wave frequency and size?

2. **Business Requirements**
   - Shipping cutoff times?
   - Priority order types (same-day, next-day)?
   - Customer SLAs and promises?
   - Carrier pickup schedules?
   - Order profile (single-line vs. multi-line)?

3. **Constraints**
   - Equipment capacity (conveyors, sorters)?
   - Packing station capacity?
   - Shipping dock doors available?
   - Labor availability by shift?
   - WMS capabilities and limitations?

4. **Performance Metrics**
   - Current picks per hour?
   - Order cycle time (order → ship)?
   - Labor utilization?
   - On-time shipping performance?
   - Wave completion rates?

---

## Wave Planning Framework

### Wave Design Principles

**1. Wave Sizing**
- **Small Waves (50-200 orders)**
  - Pros: Flexible, quick completion, easy re-wave
  - Cons: More frequent releases, higher admin overhead
  - Use: High variability, frequent cutoffs

- **Medium Waves (200-500 orders)**
  - Pros: Balanced workload, good equipment utilization
  - Cons: Some idle time between waves
  - Use: Standard operations, moderate volume

- **Large Waves (500-1000+ orders)**
  - Pros: Maximum efficiency, fewer releases
  - Cons: Inflexible, longer cycle time
  - Use: High volume, stable demand

**2. Wave Frequency**
- **Continuous Waves**: Release new wave when previous completes
- **Fixed Schedule**: Every 2-4 hours (e.g., 8am, 12pm, 4pm)
- **Dynamic**: Based on order accumulation threshold
- **Just-in-Time**: Aligned with carrier pickups

**3. Wave Composition**
- **Zone-Based**: All orders for a warehouse zone
- **Order-Type Based**: Priority, standard, bulk separately
- **Customer-Based**: Group by customer or ship-to region
- **Carrier-Based**: Group by shipping carrier
- **Hybrid**: Combination of above

### Wave Optimization Objectives

```
Primary Goals:
1. Maximize picker productivity (picks/hour)
2. Balance workload across zones/pickers
3. Meet shipping cutoff times
4. Minimize labor cost
5. Maximize equipment utilization

Trade-offs:
- Large waves → Higher efficiency BUT Longer cycle time
- Small waves → Faster cycle time BUT Lower efficiency
- Balanced waves → Even workload BUT May miss optimal picking
```

---

## Mathematical Formulation

### Wave Planning Optimization Model

**Decision Variables:**
- x[o,w] = 1 if order o assigned to wave w, 0 otherwise
- y[w] = 1 if wave w is used, 0 otherwise
- t[w] = start time of wave w
- z[w,z] = workload (lines) in wave w for zone z

**Parameters:**
- L[o] = number of pick lines in order o
- Z[o,z] = number of lines in order o for zone z
- D[o] = deadline for order o
- P = picker productivity (lines/hour)
- W_min, W_max = min/max lines per wave
- N_pickers[z] = number of pickers in zone z

**Objective Function:**

```
Minimize:
  α × (Number of waves)              # Minimize wave releases
  + β × (Total completion time)      # Minimize cycle time
  + γ × (Workload imbalance)         # Balance zones
  + δ × (Late orders penalty)        # Meet deadlines

Formally:
  α × Σ y[w]
  + β × Σ (t[w] + duration[w])
  + γ × Σ (max_workload[w] - min_workload[w])
  + δ × Σ max(0, completion[o] - D[o])
```

**Constraints:**

```python
# 1. Each order in exactly one wave
for o in orders:
    Σ x[o,w] = 1  for all w

# 2. Wave size limits
for w in waves:
    W_min × y[w] ≤ Σ (L[o] × x[o,w]) ≤ W_max × y[w]  for all o

# 3. Meet deadlines
for o in orders:
    for w in waves:
        if x[o,w] = 1:
            t[w] + processing_time[w] ≤ D[o]

# 4. Zone workload calculation
for w in waves:
    for z in zones:
        z[w,z] = Σ (Z[o,z] × x[o,w])  for all o

# 5. Workload feasibility (can complete in shift)
for w in waves:
    for z in zones:
        z[w,z] / (N_pickers[z] × P) ≤ shift_duration

# 6. Wave sequencing
for w in 1..W-1:
    t[w] + duration[w] ≤ t[w+1]
```

---

## Wave Planning Algorithms

### Greedy Wave Building

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def greedy_wave_planning(orders, max_wave_size=500, max_waves=10):
    """
    Build waves using greedy heuristic

    Sort orders by priority/deadline, then fill waves

    Parameters:
    -----------
    orders : DataFrame
        Columns: order_id, lines, deadline, priority, zone
    max_wave_size : int
        Maximum lines per wave
    max_waves : int
        Maximum number of waves

    Returns:
    --------
    Wave assignments
    """

    # Sort orders: priority first, then deadline
    orders_sorted = orders.sort_values(
        ['priority', 'deadline', 'lines'],
        ascending=[False, True, False]
    )

    waves = []
    current_wave = {
        'wave_id': 1,
        'orders': [],
        'total_lines': 0,
        'zones': {}
    }

    for idx, order in orders_sorted.iterrows():
        order_lines = order['lines']
        order_zone = order.get('zone', 'default')

        # Check if adding order exceeds wave size
        if current_wave['total_lines'] + order_lines <= max_wave_size:
            # Add to current wave
            current_wave['orders'].append(order['order_id'])
            current_wave['total_lines'] += order_lines

            # Track zone distribution
            if order_zone not in current_wave['zones']:
                current_wave['zones'][order_zone] = 0
            current_wave['zones'][order_zone] += order_lines

        else:
            # Start new wave
            waves.append(current_wave)

            if len(waves) >= max_waves:
                break

            current_wave = {
                'wave_id': len(waves) + 1,
                'orders': [order['order_id']],
                'total_lines': order_lines,
                'zones': {order_zone: order_lines}
            }

    # Add final wave
    if current_wave['orders'] and len(waves) < max_waves:
        waves.append(current_wave)

    return pd.DataFrame(waves)


# Example usage
orders = pd.DataFrame({
    'order_id': [f'ORD{i:04d}' for i in range(1, 101)],
    'lines': np.random.randint(1, 50, 100),
    'deadline': pd.date_range('2024-01-01 16:00', periods=100, freq='H'),
    'priority': np.random.choice([1, 2, 3], 100),
    'zone': np.random.choice(['A', 'B', 'C'], 100)
})

waves = greedy_wave_planning(orders, max_wave_size=400, max_waves=8)

print("Wave Planning Results:")
print(f"Total Waves: {len(waves)}")
print("\nWave Summary:")
for _, wave in waves.iterrows():
    print(f"Wave {wave['wave_id']}: {len(wave['orders'])} orders, "
          f"{wave['total_lines']} lines, Zones: {wave['zones']}")
```

### Balanced Wave Planning

```python
def balanced_wave_planning(orders, num_waves, zones):
    """
    Create balanced waves across zones to even workload

    Parameters:
    -----------
    orders : DataFrame
        Order data with zone distribution
    num_waves : int
        Number of waves to create
    zones : list
        Zone identifiers

    Returns:
    --------
    Balanced wave assignments
    """

    # Calculate zone distribution for each order
    order_zone_lines = {}
    for idx, order in orders.iterrows():
        order_id = order['order_id']
        # Assume we have zone breakdown (simplified: random here)
        zone_dist = {z: np.random.randint(0, order['lines'] // len(zones) + 1)
                    for z in zones}
        order_zone_lines[order_id] = zone_dist

    # Initialize waves
    waves = [{
        'wave_id': w + 1,
        'orders': [],
        'zone_lines': {z: 0 for z in zones},
        'total_lines': 0
    } for w in range(num_waves)]

    # Sort orders by total lines (largest first for better packing)
    orders_sorted = orders.sort_values('lines', ascending=False)

    # Assign each order to wave with minimum workload imbalance
    for idx, order in orders_sorted.iterrows():
        order_id = order['order_id']
        zone_dist = order_zone_lines[order_id]

        # Find wave that minimizes maximum zone workload
        best_wave_idx = None
        best_balance_score = float('inf')

        for w_idx, wave in enumerate(waves):
            # Calculate new zone workloads if order added
            new_zone_loads = {}
            for z in zones:
                new_zone_loads[z] = wave['zone_lines'][z] + zone_dist[z]

            # Balance score: range of zone workloads
            max_load = max(new_zone_loads.values())
            min_load = min(new_zone_loads.values())
            balance_score = max_load - min_load

            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_wave_idx = w_idx

        # Assign order to best wave
        waves[best_wave_idx]['orders'].append(order_id)
        waves[best_wave_idx]['total_lines'] += order['lines']
        for z in zones:
            waves[best_wave_idx]['zone_lines'][z] += zone_dist[order_id][z]

    return pd.DataFrame(waves)


# Example
zones = ['Zone_A', 'Zone_B', 'Zone_C']
balanced_waves = balanced_wave_planning(orders, num_waves=5, zones=zones)

print("\nBalanced Wave Planning:")
for _, wave in balanced_waves.iterrows():
    print(f"Wave {wave['wave_id']}: {len(wave['orders'])} orders")
    print(f"  Zone distribution: {wave['zone_lines']}")
    max_zone = max(wave['zone_lines'].values())
    min_zone = min(wave['zone_lines'].values())
    print(f"  Balance: {max_zone - min_zone} lines difference\n")
```

### MIP-Based Wave Optimization

```python
from pulp import *

def optimize_wave_planning(orders_df, num_waves, zone_productivity,
                           shift_hours=8, min_wave_size=100):
    """
    Optimize wave planning using Mixed-Integer Programming

    Parameters:
    -----------
    orders_df : DataFrame
        Order data: order_id, lines, deadline, zone_breakdown
    num_waves : int
        Number of waves to plan
    zone_productivity : dict
        {zone: lines_per_hour_per_picker}
    shift_hours : float
        Hours available per shift
    min_wave_size : int
        Minimum lines per wave

    Returns:
    --------
    Optimal wave assignments
    """

    orders = orders_df['order_id'].tolist()
    waves = list(range(num_waves))
    zones = list(zone_productivity.keys())

    # Create problem
    prob = LpProblem("Wave_Planning", LpMinimize)

    # Decision variables
    # x[o,w] = 1 if order o in wave w
    x = LpVariable.dicts("assign",
                        [(o, w) for o in orders for w in waves],
                        cat='Binary')

    # y[w] = 1 if wave w is used
    y = LpVariable.dicts("use_wave",
                        waves,
                        cat='Binary')

    # Workload variables
    workload = LpVariable.dicts("workload",
                               [(w, z) for w in waves for z in zones],
                               lowBound=0,
                               cat='Continuous')

    # Max workload per wave (for balancing)
    max_workload = LpVariable.dicts("max_load",
                                   waves,
                                   lowBound=0,
                                   cat='Continuous')

    # Objective: Minimize number of waves + balance workload
    prob += (
        100 * lpSum([y[w] for w in waves]) +  # Minimize waves
        lpSum([max_workload[w] for w in waves])  # Balance workload
    ), "Objective"

    # Constraints

    # 1. Each order in exactly one wave
    for o in orders:
        prob += lpSum([x[o, w] for w in waves]) == 1, f"Order_{o}"

    # 2. Calculate workload per wave per zone
    for w in waves:
        for z in zones:
            # Simplified: assume equal zone distribution
            # In practice, would have actual zone breakdown per order
            prob += workload[w, z] == lpSum([
                orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0] / len(zones) * x[o, w]
                for o in orders
            ]), f"Workload_{w}_{z}"

    # 3. Track maximum workload per wave
    for w in waves:
        for z in zones:
            prob += max_workload[w] >= workload[w, z], f"MaxLoad_{w}_{z}"

    # 4. Wave size constraints
    for w in waves:
        total_lines = lpSum([
            orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0] * x[o, w]
            for o in orders
        ])

        # Minimum wave size
        prob += total_lines >= min_wave_size * y[w], f"MinSize_{w}"

        # Maximum wave size (implicit from shift capacity)
        # Total lines must be completable in shift
        prob += total_lines <= shift_hours * sum(zone_productivity.values()) * y[w], \
                f"MaxSize_{w}"

    # 5. Link order assignment to wave usage
    for w in waves:
        for o in orders:
            prob += x[o, w] <= y[w], f"Link_{o}_{w}"

    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))

    # Extract solution
    wave_assignments = []
    for w in waves:
        if y[w].varValue > 0.5:
            wave_orders = [o for o in orders if x[o, w].varValue > 0.5]
            total_lines = sum(
                orders_df.loc[orders_df['order_id'] == o, 'lines'].values[0]
                for o in wave_orders
            )

            wave_workloads = {
                z: workload[w, z].varValue for z in zones
            }

            wave_assignments.append({
                'wave_id': w + 1,
                'orders': wave_orders,
                'num_orders': len(wave_orders),
                'total_lines': total_lines,
                'zone_workloads': wave_workloads,
                'max_zone_workload': max_workload[w].varValue
            })

    return {
        'status': LpStatus[prob.status],
        'objective': value(prob.objective),
        'waves': pd.DataFrame(wave_assignments)
    }


# Example
zone_productivity = {'Zone_A': 100, 'Zone_B': 120, 'Zone_C': 110}  # lines/hour

result = optimize_wave_planning(
    orders.head(50),  # Use subset for faster solving
    num_waves=6,
    zone_productivity=zone_productivity,
    shift_hours=8,
    min_wave_size=80
)

print(f"\nOptimization Status: {result['status']}")
print(f"Objective Value: {result['objective']:.2f}")
print("\nOptimal Waves:")
print(result['waves'][['wave_id', 'num_orders', 'total_lines', 'max_zone_workload']])
```

---

## Advanced Wave Planning Techniques

### Dynamic Wave Release Strategy

```python
class DynamicWaveManager:
    """
    Manage dynamic wave releases based on order accumulation
    """

    def __init__(self, target_wave_size=400, min_wave_size=200,
                 release_threshold=0.9):
        self.target_wave_size = target_wave_size
        self.min_wave_size = min_wave_size
        self.release_threshold = release_threshold
        self.pending_orders = []
        self.released_waves = []

    def add_order(self, order):
        """Add new order to pending queue"""
        self.pending_orders.append(order)

    def should_release_wave(self):
        """
        Determine if wave should be released

        Release criteria:
        1. Accumulated lines >= target size
        2. Oldest order waiting > max_wait_time
        3. Cutoff time approaching
        """

        total_lines = sum(o['lines'] for o in self.pending_orders)

        # Criterion 1: Size threshold
        if total_lines >= self.target_wave_size * self.release_threshold:
            return True, "Size threshold reached"

        # Criterion 2: Max wait time (simplified)
        if len(self.pending_orders) > 0:
            oldest_order_time = min(o['received_time'] for o in self.pending_orders)
            wait_minutes = (datetime.now() - oldest_order_time).total_seconds() / 60

            if wait_minutes > 120:  # 2 hours max wait
                return True, "Max wait time exceeded"

        # Criterion 3: Minimum size and approaching cutoff
        if total_lines >= self.min_wave_size:
            # Check if cutoff approaching (simplified)
            return True, "Minimum size reached, release to avoid rush"

        return False, "Not ready"

    def release_wave(self):
        """Release wave from pending orders"""

        if len(self.pending_orders) == 0:
            return None

        # Take up to target_wave_size
        wave_orders = []
        total_lines = 0

        # Sort by priority and deadline
        sorted_orders = sorted(
            self.pending_orders,
            key=lambda x: (x.get('priority', 0), x.get('deadline', datetime.max)),
            reverse=True
        )

        for order in sorted_orders:
            if total_lines + order['lines'] <= self.target_wave_size:
                wave_orders.append(order)
                total_lines += order['lines']

        # Remove released orders from pending
        for order in wave_orders:
            self.pending_orders.remove(order)

        wave = {
            'wave_id': len(self.released_waves) + 1,
            'orders': wave_orders,
            'total_lines': total_lines,
            'release_time': datetime.now()
        }

        self.released_waves.append(wave)

        return wave

    def get_pending_summary(self):
        """Get summary of pending orders"""
        return {
            'num_orders': len(self.pending_orders),
            'total_lines': sum(o['lines'] for o in self.pending_orders),
            'oldest_order': min((o['received_time'] for o in self.pending_orders),
                              default=None)
        }


# Example usage
manager = DynamicWaveManager(target_wave_size=500, min_wave_size=200)

# Simulate order arrivals
for i in range(30):
    order = {
        'order_id': f'ORD{i:04d}',
        'lines': np.random.randint(10, 50),
        'priority': np.random.choice([1, 2, 3]),
        'deadline': datetime.now() + timedelta(hours=4),
        'received_time': datetime.now() - timedelta(minutes=np.random.randint(0, 180))
    }
    manager.add_order(order)

# Check if should release
should_release, reason = manager.should_release_wave()
print(f"Should release wave: {should_release} ({reason})")

if should_release:
    wave = manager.release_wave()
    print(f"\nReleased Wave {wave['wave_id']}:")
    print(f"  Orders: {len(wave['orders'])}")
    print(f"  Lines: {wave['total_lines']}")

pending = manager.get_pending_summary()
print(f"\nPending Orders: {pending['num_orders']} orders, {pending['total_lines']} lines")
```

### Multi-Shift Wave Planning

```python
def plan_multi_shift_waves(orders, shifts, pickers_per_shift,
                           productivity=100):
    """
    Plan waves across multiple shifts

    Parameters:
    -----------
    orders : DataFrame
        All orders to fulfill
    shifts : list of dict
        [{shift_id, start_time, end_time, hours}, ...]
    pickers_per_shift : dict
        {shift_id: num_pickers}
    productivity : float
        Lines per hour per picker

    Returns:
    --------
    Wave plan with shift assignments
    """

    # Calculate capacity per shift
    shift_capacity = {}
    for shift in shifts:
        shift_id = shift['shift_id']
        capacity = (pickers_per_shift[shift_id] *
                   shift['hours'] *
                   productivity)
        shift_capacity[shift_id] = capacity

    # Sort orders by deadline
    orders_sorted = orders.sort_values('deadline')

    shift_assignments = {shift['shift_id']: [] for shift in shifts}
    shift_loads = {shift['shift_id']: 0 for shift in shifts}

    # Assign orders to shifts
    for idx, order in orders_sorted.iterrows():
        order_lines = order['lines']
        deadline = order['deadline']

        # Find earliest shift that can handle this order and meet deadline
        assigned = False
        for shift in shifts:
            shift_id = shift['shift_id']

            # Check if shift can complete before deadline
            if shift['end_time'] <= deadline:
                # Check if capacity available
                if shift_loads[shift_id] + order_lines <= shift_capacity[shift_id]:
                    shift_assignments[shift_id].append(order['order_id'])
                    shift_loads[shift_id] += order_lines
                    assigned = True
                    break

        if not assigned:
            print(f"Warning: Could not assign order {order['order_id']}")

    # Create waves within each shift
    all_waves = []
    wave_counter = 1

    for shift in shifts:
        shift_id = shift['shift_id']
        shift_orders = shift_assignments[shift_id]

        if not shift_orders:
            continue

        # Split shift into waves (e.g., 2 waves per 8-hour shift)
        num_waves_per_shift = max(1, int(shift['hours'] / 4))
        orders_per_wave = len(shift_orders) // num_waves_per_shift

        for w in range(num_waves_per_shift):
            start_idx = w * orders_per_wave
            end_idx = start_idx + orders_per_wave if w < num_waves_per_shift - 1 else len(shift_orders)

            wave_orders = shift_orders[start_idx:end_idx]
            wave_lines = sum(
                orders.loc[orders['order_id'] == o, 'lines'].values[0]
                for o in wave_orders
            )

            all_waves.append({
                'wave_id': wave_counter,
                'shift_id': shift_id,
                'wave_in_shift': w + 1,
                'orders': wave_orders,
                'num_orders': len(wave_orders),
                'total_lines': wave_lines
            })
            wave_counter += 1

    return pd.DataFrame(all_waves)


# Example
shifts = [
    {'shift_id': 'Day', 'start_time': datetime(2024,1,1,6,0),
     'end_time': datetime(2024,1,1,14,0), 'hours': 8},
    {'shift_id': 'Evening', 'start_time': datetime(2024,1,1,14,0),
     'end_time': datetime(2024,1,1,22,0), 'hours': 8},
    {'shift_id': 'Night', 'start_time': datetime(2024,1,1,22,0),
     'end_time': datetime(2024,1,2,6,0), 'hours': 8}
]

pickers = {'Day': 12, 'Evening': 8, 'Night': 4}

multi_shift_waves = plan_multi_shift_waves(orders.head(100), shifts, pickers)

print("\nMulti-Shift Wave Plan:")
print(multi_shift_waves.groupby('shift_id')[['num_orders', 'total_lines']].sum())
```

---

## Tools & Libraries

### Wave Management Software

**Warehouse Management Systems:**
- **Manhattan WMS**: Advanced wave planning and optimization
- **Blue Yonder (JDA) WMS**: AI-driven wave management
- **SAP EWM**: Wave templates and dynamic release
- **HighJump WMS**: Configurable wave strategies
- **Körber WMS**: Multi-wave parallel processing

**Order Management Systems:**
- **IBM Sterling OMS**: Wave release and order orchestration
- **Fluent Commerce**: Real-time order promising and waving
- **Radial OMS**: Distributed order management with waving

### Python Libraries

```python
# Optimization
from pulp import *
from ortools.sat.python import cp_model

# Scheduling
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Analysis
from sklearn.cluster import KMeans  # Order clustering
import matplotlib.pyplot as plt
```

---

## Common Challenges & Solutions

### Challenge: Cutoff Time Pressure

**Problem:**
- Orders arrive late, need same-day ship
- Wave released too early misses late orders
- Wave released too late misses carrier pickup

**Solutions:**
- Multiple cutoff waves (e.g., 12pm, 2pm, 4pm)
- Express wave for urgent orders (small, frequent)
- Dynamic wave release triggered at 80% threshold
- Pre-stage high-probability orders
- Negotiate later carrier pickups
- Use last-mile carriers with flexible schedules

### Challenge: Workload Imbalance

**Problem:**
- Some zones finish early, others late
- Pickers idle while others overwhelmed
- Bottlenecks at packing stations

**Solutions:**
- Balance waves across zones mathematically
- Cross-train pickers to work multiple zones
- Dynamic picker redeployment mid-wave
- Adjust wave size by zone capacity
- Use pick-to-light or goods-to-person (balanced automatically)
- Monitor real-time progress, rebalance next wave

### Challenge: Order Profile Variability

**Problem:**
- Mix of single-line and 50-line orders
- Some days 500 orders, some days 2000
- Seasonal peaks disrupt standard waves

**Solutions:**
- Separate waves by order complexity (each vs. case)
- Variable wave size (200-600 lines, not fixed)
- Reserve capacity for large orders
- Pre-pick high-velocity items during off-peak
- Hire temp labor for peaks
- Adjust wave frequency dynamically (not fixed schedule)

### Challenge: Equipment Constraints

**Problem:**
- Conveyor at capacity (max 500 units/hour)
- Sorter can't handle wave volume
- Packing stations become bottleneck

**Solutions:**
- Wave size limited by downstream capacity
- Stagger wave releases (15-min offset between zones)
- Use wave pools (release to picking, but meter to packing)
- Add surge capacity (temporary packing stations)
- Batch packing for same customer/carrier
- Upgrade equipment or add parallel lines

### Challenge: WMS Limitations

**Problem:**
- WMS only supports fixed wave sizes
- Can't auto-release based on thresholds
- Limited wave templates
- No cross-zone wave support

**Solutions:**
- Use middleware for advanced wave logic
- Manual monitoring with alert thresholds
- Pre-build wave templates for common scenarios
- Upgrade WMS or add bolt-on optimization
- Work with WMS vendor on custom logic
- Implement external optimization, feed results to WMS

---

## Output Format

### Wave Planning Report

**Daily Wave Schedule - January 15, 2024**

| Wave | Start Time | Lines | Orders | Zones | Est. Duration | Cutoff Met | Status |
|------|------------|-------|--------|-------|---------------|----------|--------|
| W01 | 06:00 | 420 | 87 | A,B,C | 3.2 hrs | 12pm | Complete |
| W02 | 09:00 | 385 | 96 | A,B,C | 2.9 hrs | 12pm | In Progress |
| W03 | 12:00 | 510 | 102 | A,B,C | 3.8 hrs | 4pm | Scheduled |
| W04 | 16:00 | 295 | 74 | A,B,C | 2.2 hrs | 6pm | Scheduled |

**Wave W02 Details:**

```
Orders: 96
Total Lines: 385
Average Lines/Order: 4.0

Zone Distribution:
  Zone A: 145 lines (38%)
  Zone B: 132 lines (34%)
  Zone C: 108 lines (28%)

Pickers Assigned:
  Zone A: 4 pickers → ~36 lines/picker
  Zone B: 3 pickers → ~44 lines/picker
  Zone C: 3 pickers → ~36 lines/picker

Expected Completion: 11:54 AM
Cutoff: 12:00 PM (6 min buffer)

Priority Orders: 12 (flagged for early pick)
```

**Performance Summary:**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Waves per Day | 8-10 | 9 | ✓ On Target |
| Avg Wave Size | 350-450 | 403 | ✓ On Target |
| Balance (max-min zone) | <50 lines | 37 lines | ✓ On Target |
| Cutoff Adherence | >95% | 98% | ✓ On Target |
| Picker Utilization | 80-90% | 86% | ✓ On Target |

**Recommendations:**
- Wave 3 is largest - consider split if issues arise
- Zone B slightly higher workload in W02 - add 1 picker if available
- Suggest moving cutoff to 12:30pm for 30min buffer

---

## Questions to Ask

If you need more context:
1. What's your daily order volume (orders and lines)?
2. How many pick waves do you currently run per day?
3. What are your shipping cutoff times?
4. How many warehouse zones and pickers per zone?
5. What WMS do you use? Wave planning capabilities?
6. What's your average picks per hour per picker?
7. Any equipment bottlenecks (conveyor, sorter, packing)?
8. Do you have priority or express orders?

---

## Related Skills

- **order-batching-optimization**: For grouping orders within waves
- **picker-routing-optimization**: For optimizing pick paths within waves
- **workforce-scheduling**: For shift and labor planning
- **task-assignment-problem**: For assigning pickers to zones/waves
- **warehouse-slotting-optimization**: For SKU placement affecting pick efficiency
- **order-fulfillment**: For overall fulfillment process design
- **capacity-planning**: For long-term wave capacity planning
