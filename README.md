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

## 2. Add to Cursor

Once pushed, install with:

```bash
npx degit YOUR_USERNAME/nexova-skills nexova-skills
```

Then copy the desired skill folders into your Cursor project.

---

## 3. Add to Claude Code

This repo includes Claude plugin metadata in `.claude-plugin/marketplace.json`.

Install locally:

```bash
claude plugin install .
```

---

## 4. Skill format

```text
skills/<skill-name>/SKILL.md
```

Each `SKILL.md` contains:

- when to use the skill
- operating procedure
- inputs / outputs
- examples
- quality checks

---

## 5. Categories

This library includes skills for:

- supply chain planning
- demand forecasting
- inventory optimization
- procurement
- warehousing
- logistics
- cold chain
- pharma / medtech / FMCG / automotive / retail
- optimization models
- executive communication
- product and marketing
- software development workflows

---

## 6. Suggested repo name

Recommended GitHub repo name:

```text
nexova-skills
```

Alternative:

```text
SupplyChainSkills
```

---

## 7. Install script

Run:

```bash
chmod +x install.sh
./install.sh
```

---

## 8. License

Add your preferred license before publishing commercially.

Recommended:

- MIT for open adoption
- Apache-2.0 for enterprise-friendly use
- Proprietary if you plan to monetize as paid IP
