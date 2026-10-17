"""Verify brute-force accepts custom n_machines argument."""
import numpy as np
from src.bruteforce import solve_bruteforce


def test_solve_bruteforce_with_custom_n_machines():
    """Test that n_machines parameter works correctly for rectangular instances."""
    durations = np.array([6.0, 2.0, 8.0, 1.0, 5.0, 3.0, 7.0, 4.0])

    # Large instance: 8 jobs × 4 machines (rectangular)
    result = solve_bruteforce(durations, n_machines=4)

    assert result["optimal_assignment"] is not None
    assert result["optimal_makespan"] > 0
    assert result["n_total"] == 65536  # 4^8 combinations
    assert result["n_feasible"] > 0


def test_solve_bruteforce_default_square():
    """Test backward compatibility: default n_machines = n_jobs."""
    durations = np.array([4.0, 3.0, 2.0])

    # Should use n_machines = n_jobs = 3 by default
    result = solve_bruteforce(durations)

    assert result["n_total"] == 27  # 3^3 combinations
    assert result["optimal_assignment"].shape == (3, 3)


if __name__ == "__main__":
    test_solve_bruteforce_with_custom_n_machines()
    print("✓ test_solve_bruteforce_with_custom_n_machines passed")

    test_solve_bruteforce_default_square()
    print("✓ test_solve_bruteforce_default_square passed")
