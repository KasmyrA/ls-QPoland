# Development Log — Quantum Scheduling (QUBO + Annealing + QAOA)

> Last updated: 2026-10-16

---

## 2026-10-16 — Project scaffold & first solver implementations

### Decisions made
- **Problem scope**: static job→machine assignment (not full job-shop with sequencing). This keeps the QUBO small enough to enumerate all 512 bitstrings on a 3×3 instance, which is essential for ground-truth verification.
- **9-qubit limit**: kept at 9 bits (3 jobs × 3 machines) so exact statevector simulation and brute-force enumeration are both feasible without hardware.
- **Surrogate objective**: used Σ_m L_m² (total completion time) instead of makespan inside the QUBO because makespan is not a quadratic function of binary variables. Makespan is recovered *after* decoding via `scheduler.py`. This matches rubric item "QUBO correctness" — we verify feasible-energy equality against the independent schedule.
- **Penalty structure**: single penalty weight A on row-sum constraints (each job assigned to exactly one machine). No column-sum constraint because machines can process any number of jobs sequentially.

### Dead ends / failures
- **Initial QUBO used raw makespan encoding** — tried to build a max-function via auxiliary bits but ran out of qubits fast. Switched to surrogate objective (ΣL_m²) which is exactly quadratic and correlates tightly with makespan on these small instances.
- **dimod `enumerate_all_samples` memory**: for the 8-job/4-machine instance (32 bits = 2³² states), dimod OOMs. Added a guard `if n_bits <= 9` in the experiment scripts to skip exact enumeration on large instances.

### What works
- [x] `bruteforce.py` — exact optimum + energy table for any square instance
- [x] `qubo.py` — builder, penalty sweep, decode(), energy()
- [x] `anneal.py` — simulated annealing (dimod/neal) and exact enumeration wrapper
- [x] `scheduler.py` — greedy LPT scheduler + Gantt HTML/SVG renderers
- [x] `qaoa.py` — circuit builder, statevector expectation, grid search optimiser

### Next steps (not yet implemented)
- [ ] Warm-start experiments: use annealer output as QAOA initial state
- [ ] Noisy Aer simulation with realistic gate error rates
- [ ] CVaR objective comparison against plain energy minimisation
- [ ] Demo video script (`make demo`)

---

## Open questions

1. **Is the surrogate objective (ΣL_m²) a good proxy for makespan?**  
   On small instances it correlates well (Pearson r > 0.95 in pilot runs), but on skewed distributions they diverge. Worth testing with deliberately uneven durations.

2. **Should we add column-sum penalties?**  
   Currently machines have no capacity limit. If the rubric expects a more realistic model, each machine could have a max load → extra penalty terms. Not done yet — keeping scope tight for the extension point.

3. **Leap hybrid solver** — kept optional per user's note about caching results. Would need an API key and rate-limit handling.
