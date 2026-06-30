# Nexova Skills

A library of **200 agent skills** for supply chain, operations, optimization, regulated-industry domain knowledge, content/marketing, and build tooling — packaged for **Cursor** and **Claude Code**.

Each skill is a folder under `skills/` containing a `SKILL.md`. That's the whole format.

---

## 1. Push this to YOUR GitHub (one time)

Download and unzip this repo, then from inside the `nexova-skills/` folder:

```bash
cd nexova-skills
git init
git add .
git commit -m "Nexova skills library v1.0.0"

# Easiest if you have the GitHub CLI installed:
gh repo create nexova-skills --public --source=. --push

# --- OR manually ---
# 1) create an empty repo named "nexova-skills" on github.com (no README)
# 2) then:
git remote add origin https://github.com/YOUR_USERNAME/nexova-skills.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME`. Use `--private` instead of `--public` if you want it private (npx still works for you when authenticated).

---

## 2. After it's pushed — pull skills anywhere with npx

Once the repo lives on GitHub, the skills CLI can fetch from it. **This is the npx-style install you asked for** — it exists the moment the repo is public:

```bash
# list everything in the repo
npx skills add YOUR_USERNAME/nexova-skills --list

# install one skill into Cursor
npx skills add YOUR_USERNAME/nexova-skills@demand-forecasting -a cursor

# install several
npx skills add YOUR_USERNAME/nexova-skills@multi-echelon-inventory -a cursor
npx skills add YOUR_USERNAME/nexova-skills@capacitated-vrp -a cursor
```

Run from your project root; skills land in `.cursor/skills/`. Restart Cursor after installing.

> If a particular CLI build expects skills at the repo root instead of `skills/`, point it at the path or use Method 3 below — the files are identical either way.

---

## 3. Option 3 — the self-install script (no GitHub needed)

Works straight from the unzipped folder, offline. Copies skills into a project's `.cursor/skills/`.

```bash
# ALL skills into the current project
bash install.sh

# into a specific project
bash install.sh /path/to/your/project

# only specific skills (into current project)
bash install.sh . demand-forecasting inventory-optimization capacitated-vrp
```

Restart Cursor, then invoke any skill with `/skill-name` in chat.

---

## 4. Clone-and-install (team workflow)

Anyone on your team can:

```bash
git clone https://github.com/YOUR_USERNAME/nexova-skills.git
cd nexova-skills
bash install.sh /path/to/their/project
```

Or commit `.cursor/skills/` directly into your app repo so every clone has the skills automatically — Cursor is project-scoped, so that's the most reliable team distribution.

---

## What's inside (200 skills)

Highlights for the supply-chain platform build:

- **Planning** — demand-forecasting, inventory-optimization, multi-echelon-inventory, demand-supply-matching, dynamic-lot-sizing, lot-sizing-problems
- **Logistics** — capacitated-vrp, multi-depot-vrp, container-loading-optimization, load-building-optimization, freight-optimization, distribution-center-network, inventory-routing-problem, network-flow-optimization, facility-location-problem
- **Production** — master-production-scheduling, capacity-planning, flow-shop-scheduling, assembly-line-balancing
- **Optimization core** — optimization (metaheuristic / multi-objective / ml-hybrid), constraint-programming, column-generation, knapsack-problems
- **Cross-cutting** — control-tower-design, digital-twin-modeling
- **Regulated-domain moat** — medical-device-distribution, pharmacy-supply-chain*, clinical-trial-logistics, hospital-logistics, compliance-management, carbon-footprint-tracking
- **Build tooling** — mcp-builder, frontend-design, technical-spec-template, sql-query-explainer
- Plus ~150 more across content, marketing, PM, research, and ops.

Browse `skills/` for the full list.

---

## Notes

- Cursor reads skills from `.cursor/skills/` (project-scoped; no global dir). Claude Code reads from `.claude/skills/` and also supports the optional `.claude-plugin/marketplace.json` included here.
- A skill = a folder + `SKILL.md`. To add your own, drop a new folder under `skills/` and re-push.
- License/usage: your private library — keep the repo private if these shouldn't be public.
