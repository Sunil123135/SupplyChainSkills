---
name: idea-generator-frameworks
description: Comprehensive multi-framework idea generation tool using TRIZ, Six Thinking Hats, Design of Experiments, SCAMPER, Morphological Analysis, Lateral Thinking, First Principles, Jobs to be Done, and Blue Ocean Strategy. Use this skill whenever the user needs to ideate solutions, improve a product, solve a problem, explore opportunities, run brainstorms, innovate on existing concepts, optimize processes, or needs creative problem-solving across supply chain, product development, engineering, business strategy, or entrepreneurship. Works with vague challenges ("we need to innovate") or specific constraints ("reduce cost by 30%"). Generates 50+ exhaustive ideas with framework comparison, interactive decision matrices, ranked priority lists, and visual mind maps. Let user choose output format and ideation depth.
---

# Idea Generator: Frameworks

A comprehensive idea generation system combining 9 proven frameworks for systematic innovation. Works across all domains and problem types.

## How It Works

This skill applies **all 9 frameworks** to your challenge in parallel, generating 50+ exhaustive ideas. Each framework brings different creative angles:

| Framework | What It Does | Best For |
|-----------|-------------|----------|
| **TRIZ** (Theory of Inventive Problem Solving) | Applies 40 inventive principles to resolve contradictions | Technical conflicts; engineering optimization |
| **Six Thinking Hats** (De Bono) | Separates thinking into 6 perspectives (facts, emotion, critical, creative, control, vision) | Team alignment; multi-perspective analysis |
| **Design of Experiments (DoE)** | Systematic factorial testing to find optimal combinations | Product testing; process optimization; A/B ideas |
| **SCAMPER** | Prompts substitution, combination, adaptation, modification, put-to-other-use, elimination, reversal | Feature improvement; product evolution |
| **Morphological Analysis** | Breaks problem into dimensions, explores all combinations | Market expansion; new product creation |
| **Lateral Thinking** | Random word association, forced connections, constraint removal | Breakthrough ideas; unconventional approaches |
| **First Principles** | Deconstructs problem to fundamentals, rebuilds from scratch | Deep innovation; business model disruption |
| **Jobs to be Done (JTBD)** | Reframes around underlying customer jobs and outcomes | Customer-centric solutions; market fit |
| **Blue Ocean Strategy** | Value innovation through eliminating, reducing, raising, creating | Market differentiation; competitive advantage |

## Usage

### Basic Syntax
```
Run idea generation on: [your challenge/problem/opportunity]
```

### Input Types Supported

1. **Problem Statement**: "Reduce order fulfillment time by 50% in our supply chain"
2. **Improvement Challenge**: "How can we make our onboarding less painful?"
3. **Opportunity Exploration**: "What new revenue streams could we create?"
4. **Vague Brief**: "We should innovate around customer retention"
5. **Constraint-Based**: "Generate ideas with a budget of <$10K and 2-week timeline"
6. **Domain-Specific**: "Supply chain visibility improvements" or "AI agent architecture for procurement"

### Customization Options

**Depth Control:**
- `depth:quick` → 20-30 ideas total, ~2-3 per framework
- `depth:standard` → 40-50 ideas, ~5-6 per framework  
- `depth:exhaustive` → 60+ ideas, 7+ per framework (default)

**Output Format:**
- `format:markdown` → Structured report with categorized ideas
- `format:matrix` → Interactive HTML decision matrix with scoring
- `format:ranked` → Priority-ranked list with implementation guidance
- `format:all` → All three formats (default)

**Focus Areas:**
- `focus:technical` → Emphasize engineering/technical solutions
- `focus:business` → Emphasize business model/revenue angles
- `focus:process` → Emphasize operational/workflow improvements
- `focus:user` → Emphasize customer experience angles

**Example:**
```
Run idea generation on: Reduce our inventory holding costs by 40%
depth:exhaustive
focus:process,business
format:matrix,ranked
```

## The Process

For each challenge, the skill:

1. **Understands the Context**
   - Extracts the core problem/opportunity
   - Identifies constraints and success criteria
   - Maps the current state

2. **Applies Each Framework**
   - TRIZ: Maps to inventive principles, identifies contradictions
   - Six Hats: Generates ideas from 6 distinct perspectives
   - DoE: Creates factorial combinations to test
   - SCAMPER: Prompts modifications across 7 dimensions
   - Morphological: Breaks into attributes, explores combinations
   - Lateral: Forces random connections, removes assumptions
   - First Principles: Rebuilds from fundamentals
   - JTBD: Reframes around underlying jobs and emotions
   - Blue Ocean: Explores value innovation angles

3. **Generates 50+ Ideas**
   - Deduplicates across frameworks
   - Attributes each idea to its framework(s)
   - Enriches with implementation hints

4. **Produces Output**
   - **Markdown Report**: Categorized ideas with framework attribution
   - **Interactive Matrix**: Score ideas on impact vs effort, novelty vs feasibility, etc.
   - **Ranked List**: Priority-ordered with reasoning, quick wins, and moonshots

## Output Formats Explained

### 1. Markdown Report
Structured list of 50+ ideas organized by:
- Framework of origin
- Category (quick win, medium effort, transformational)
- Implementation difficulty
- Estimated impact
- Related ideas and synergies

**Use when:** You want to scan all ideas systematically, share with a team, or use as a reference

### 2. Interactive Decision Matrix
HTML widget where you can:
- Plot ideas on custom axes (Impact vs Effort, Novelty vs Feasibility, etc.)
- Filter by framework or category
- Add custom scoring criteria
- Download as CSV for further analysis

**Use when:** You want to evaluate and prioritize ideas interactively, compare trade-offs, or involve a team in selection

### 3. Ranked Priority List
Top 20-30 ideas ordered by:
- Expected impact (high/medium/low)
- Implementation effort (quick win / medium / complex)
- Feasibility in your context
- Strategic alignment to your goals

Each includes:
- Quick description
- Framework(s) it comes from
- Why it matters
- Potential first steps
- Risks/blockers to consider

**Use when:** You want actionable next steps, want to decide what to experiment with first, or need exec-ready recommendations

## Tips for Best Results

### Be Specific (But Not Prescriptive)
❌ "We need to be more innovative"  
✅ "Reduce order-to-delivery time from 7 days to 2 days while keeping quality >99%"

### Include Constraints
- Budget, timeline, regulatory, technical, resource limits
- Available tools/platforms
- Team size or skills

### Share Context
- Current state and what's already been tried
- Why the status quo isn't working
- Who the idea should benefit (users, business, organization)

### Leverage the Frameworks
- If you need "breakthrough" ideas → focus on **First Principles** + **Lateral Thinking**
- If you need "customer-centric" ideas → focus on **JTBD** + **Six Hats**
- If you need "competitive advantage" → focus on **Blue Ocean** + **SCAMPER**
- If you need "quick wins" → focus on **SCAMPER** + **Morphological Analysis**

## Examples

### Example 1: Supply Chain Optimization
```
Run idea generation on: Reduce inventory holding costs by 40% while maintaining 95% fulfillment rate
focus:process,business
format:ranked
depth:exhaustive
```

Would generate ideas like:
- Just-in-time supplier partnerships (JTBD + DoE)
- Dynamic pricing based on inventory age (Blue Ocean + SCAMPER)
- Predictive demand models with AI (First Principles + DoE)
- Vendor-managed inventory programs (SCAMPER + Six Hats)
- Circular supply loops (Blue Ocean + First Principles)

### Example 2: Product Feature Innovation
```
Run idea generation on: Make onboarding 10x faster without losing personalization
focus:user,technical
format:matrix
```

Would generate ideas mapped to frameworks with scoring on effort vs impact.

### Example 3: Business Model Disruption
```
Run idea generation on: Create new revenue stream using our existing supply chain infrastructure
depth:exhaustive
focus:business
format:all
```

Would generate ideas across all frameworks with full analysis, interactive matrix, and ranked priorities.

## Framework Details

### TRIZ (Inventive Problem Solving)
- **Principle**: Contradictions can be resolved through 40 inventive principles
- **Process**: Identify what you want to improve and what gets worse, find principle matches
- **Best for**: Technical problems, optimization, innovation in engineering/manufacturing
- **Output**: Ideas structured around principles like "segmentation," "local quality," "feedback," "dynamic systems"

### Six Thinking Hats (De Bono)
- **White Hat**: Facts, data, information
- **Red Hat**: Emotion, intuition, gut feeling
- **Black Hat**: Critical judgment, risks, what could go wrong
- **Yellow Hat**: Optimism, best-case scenarios, benefits
- **Green Hat**: Creativity, alternatives, new ideas
- **Blue Hat**: Process control, what should we do next
- **Output**: Ideas from each perspective, integrated holistic view

### Design of Experiments (DoE)
- **Principle**: Systematically vary factors to find optimal combinations
- **Process**: Identify key variables, design factorial matrix, generate hypothesis combinations
- **Best for**: Product optimization, A/B ideas, process tuning
- **Output**: Testable idea combinations with expected impact factors

### SCAMPER
- **Substitute**: Replace components, materials, processes
- **Combine**: Merge with other products/services/processes
- **Adapt**: Adjust to different context or audience
- **Modify**: Change attributes, scale, shape, qualities
- **Put to Other Use**: New applications or repurposing
- **Eliminate**: Remove elements, simplify
- **Reverse**: Opposite approach, flip the flow
- **Output**: 7 categories of improvement ideas

### Morphological Analysis
- **Principle**: Break problem into dimensions, explore combinations
- **Process**: Define attributes, generate values for each, explore matrix
- **Best for**: New product concepts, market expansion, comprehensive exploration
- **Output**: Systematic combinations revealing unexpected opportunities

### Lateral Thinking
- **Principle**: Break linear thinking patterns, force new connections
- **Techniques**: Random word association, constraint reversal, assumption questioning
- **Best for**: Breakthrough ideas, unconventional solutions, thinking outside current paradigms
- **Output**: Surprising connections and non-obvious approaches

### First Principles
- **Principle**: Deconstruct to fundamentals, rebuild from scratch
- **Process**: Ask "why" repeatedly, identify irreducible assumptions, reconstruct from ground zero
- **Best for**: Deep innovation, business model disruption, long-term competitive advantage
- **Output**: Fundamental rethinks of how things could be

### Jobs to be Done (JTBD)
- **Principle**: Focus on the underlying job customer wants done, not the product
- **Process**: Identify functional, emotional, and social jobs; reframe solutions around job success
- **Best for**: Customer-centric innovation, understanding value creation, market fit
- **Output**: Ideas focused on job completion, emotional outcomes, and success metrics

### Blue Ocean Strategy
- **Principle**: Create new, uncontested market space instead of competing in red ocean
- **Process**: Eliminate/reduce/raise/create factors to reshape industry value curve
- **Best for**: Competitive differentiation, market disruption, value innovation
- **Output**: Ideas that shift the competitive landscape

---

## Next Steps After Ideation

Once you have your 50+ ideas:

1. **Cluster**: Group similar ideas into themes
2. **Evaluate**: Score on impact, effort, alignment, novelty
3. **Experiment**: Pick 3-5 quick wins to test this week
4. **Deep Dive**: For promising ideas, apply [Design Brief], [Experiment Designer], or [Technical Spec Template]
5. **Iterate**: Share results, gather feedback, refine top ideas

---

**Related Skills**: 
- `feature-prioritisation` — Rank ideas by strategic value
- `experiment-designer` — Test ideas rigorously
- `product-health-analysis` — Validate impact after launch
- `stakeholder-influence-mapper` — Build consensus around ideas
