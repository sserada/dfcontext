"""Tests for TokenBudgetAllocator."""

import pandas as pd

from dfcontext.budget import BudgetPlan, TokenBudgetAllocator


class TestBudgetPlan:
    def test_dataclass_fields(self) -> None:
        plan = BudgetPlan(
            schema_budget=100,
            column_budgets={"a": 50, "b": 50},
            sample_budget=50,
            total_budget=200,
        )
        assert plan.schema_budget == 100
        assert plan.total_budget == 200


class TestTokenBudgetAllocator:
    def test_default_allocation(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        allocator = TokenBudgetAllocator()
        plan = allocator.allocate(df, total_budget=1000)

        assert plan.schema_budget > 0
        assert plan.sample_budget > 0
        assert len(plan.column_budgets) == 2
        assert all(v > 0 for v in plan.column_budgets.values())

    def test_budget_sums_reasonable(self) -> None:
        df = pd.DataFrame({"a": range(10)})
        allocator = TokenBudgetAllocator()
        plan = allocator.allocate(df, total_budget=1000)

        total_allocated = (
            plan.schema_budget
            + sum(plan.column_budgets.values())
            + plan.sample_budget
        )
        # Should not exceed total budget
        assert total_allocated <= plan.total_budget * 1.1

    def test_disable_sections(self) -> None:
        df = pd.DataFrame({"a": [1]})
        allocator = TokenBudgetAllocator()

        plan = allocator.allocate(
            df,
            total_budget=1000,
            include_schema=False,
            include_samples=False,
        )
        assert plan.schema_budget == 0
        assert plan.sample_budget == 0
        assert sum(plan.column_budgets.values()) > 0

    def test_hint_boosts_relevant_column(self) -> None:
        df = pd.DataFrame({
            "sales": [100, 200, 300],
            "region": ["E", "W", "N"],
            "id": [1, 2, 3],
        })
        allocator = TokenBudgetAllocator()

        plan_no_hint = allocator.allocate(df, total_budget=1000)
        plan_hint = allocator.allocate(
            df, total_budget=1000, hint="sales trends"
        )

        # Sales column should get more budget with hint
        ratio_no_hint = (
            plan_no_hint.column_budgets["sales"]
            / sum(plan_no_hint.column_budgets.values())
        )
        ratio_hint = (
            plan_hint.column_budgets["sales"]
            / sum(plan_hint.column_budgets.values())
        )
        assert ratio_hint > ratio_no_hint

    def test_high_null_column_gets_less_budget(self) -> None:
        df = pd.DataFrame({
            "full": range(100),
            "sparse": [None] * 90 + list(range(10)),
        })
        allocator = TokenBudgetAllocator()
        plan = allocator.allocate(df, total_budget=1000)

        assert plan.column_budgets["full"] > plan.column_budgets[
            "sparse"
        ]

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        allocator = TokenBudgetAllocator()
        plan = allocator.allocate(df, total_budget=1000)

        assert plan.column_budgets == {}

    def test_custom_ratio(self) -> None:
        df = pd.DataFrame({"a": [1]})
        allocator = TokenBudgetAllocator(
            budget_ratio={"schema": 0.5, "stats": 0.3, "samples": 0.2}
        )
        plan = allocator.allocate(df, total_budget=1000)

        assert plan.schema_budget == 500
