"""Tests for the classical greedy scheduler."""
import numpy as np


class TestScheduleFromAssignment:
    """Verify schedule construction from a flat assignment list."""

    def test_simple_assignment(self):
        from src.scheduler import schedule_from_assignment

        durations = np.array([4, 3, 2])
        assignment = [0, 1, 2]  # each job to its own machine
        result = schedule_from_assignment(durations, assignment)

        assert "makespan" in result
        assert "gantt" in result
        assert "machine_loads" in result
        assert result["makespan"] == 4.0  # max(4,3,2)

    def test_all_jobs_one_machine(self):
        from src.scheduler import schedule_from_assignment

        durations = np.array([4, 3, 2])
        assignment = [0, 0, 0]  # all to machine 0
        result = schedule_from_assignment(durations, assignment)

        assert result["makespan"] == 9.0  # sum of all durations

    def test_gantt_has_entries(self):
        from src.scheduler import schedule_from_assignment

        durations = np.array([4, 3, 2])
        assignment = [0, 1, 2]
        result = schedule_from_assignment(durations, assignment)

        # Each machine should have one job entry
        for m, entries in result["gantt"].items():
            assert len(entries) > 0
            for j_idx, start, end in entries:
                assert start < end
                assert end - start == durations[j_idx]


class TestGreedyLPT:
    """Verify LPT greedy scheduler behavior."""

    def test_lpt_assigns_all_jobs(self):
        from src.scheduler import greedy_lpt

        durations = np.array([4, 3, 2])
        result = greedy_lpt(durations)

        assert len(result["assignment"]) == 3
        # All jobs should be assigned (no -1 in assignment)
        assert all(a >= 0 for a in result["assignment"])


class TestGanttHTML:
    """Verify HTML rendering."""

    def test_generate_gantt_html_returns_string(self):
        from src.scheduler import greedy_lpt, generate_gantt_html

        durations = np.array([4, 3, 2])
        schedule = greedy_lpt(durations)
        html = generate_gantt_html(schedule)

        assert isinstance(html, str)
        assert "<table" in html
        assert "</table>" in html


class TestGanttSVG:
    """Verify SVG rendering."""

    def test_generate_gantt_svg_returns_string(self):
        from src.scheduler import greedy_lpt, generate_gantt_svg

        durations = np.array([4, 3, 2])
        schedule = greedy_lpt(durations)
        svg = generate_gantt_svg(schedule)

        assert isinstance(svg, str)
        assert "<svg" in svg
        assert "</svg>" in svg
