"""Command line entry point.

  # path 1 - instance JSON only
  python run_demo.py basic    instance_3x3.json --backend sa     --out out/basic.json
  python run_demo.py basic    instance_3x3.json --backend sbm    --out out/basic_velox.json     # needs VELOX_TOKEN

  # path 2 - instance + machines (+ events [+ previous output])
  python run_demo.py extended instance_3x3.json machines.json --out out/ext.json
  python run_demo.py extended instance_3x3.json machines.json --events events.json \
                     --prev out/ext.json --out out/resched.json --report out/report.json [--window 6]

  # rolling horizon (small QUBO per slice)
  python run_demo.py rolling  instance_3x3.json machines.json --step 3 --lookahead 2 --out out/rolling.json

backends: sa (local), exact (<=24 vars), sbm / veloxq (cloud), cpsat (classical reference)
"""
import argparse

from pipeline import solve_basic, solve_extended, solve_rolling


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["basic", "extended", "rolling"])
    p.add_argument("instance")
    p.add_argument("machines", nargs="?")
    p.add_argument("--events"); p.add_argument("--prev")
    p.add_argument("--backend", default="sa", choices=["sa", "exact", "sbm", "veloxq", "cpsat"])
    p.add_argument("--objective", default=None, choices=["makespan", "sum_completion"])
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--step", type=int, default=3); p.add_argument("--lookahead", type=int, default=2)
    p.add_argument("--out"); p.add_argument("--report")
    a = p.parse_args()

    if a.mode == "basic":
        st = solve_basic(a.instance, a.backend, a.objective or "makespan", a.out)
    elif a.mode == "extended":
        if not a.machines:
            p.error("extended needs machines.json")
        st, rep = solve_extended(a.instance, a.machines, a.events, a.prev, a.backend, a.objective or "makespan",
                                 a.window, a.out, a.report)
        if rep:
            print({k: v for k, v in rep.items() if k.startswith(("makespan", "ops_", "violations"))})
    else:
        st = solve_rolling(a.instance, a.machines, a.events, a.backend, a.objective or "sum_completion",
                           a.step, a.lookahead, a.out)
    print(f"{st.method}: makespan={st.makespan}  violations={st.metadata.get('violations')}")
    for t in st.schedule:
        print(f"  {t.job}  {t.unit or t.machine:<3} {t.start:>3} -> {t.end:<3}")


if __name__ == "__main__":
    main()