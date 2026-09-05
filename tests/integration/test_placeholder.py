"""Integration tests (Postgres/network) land with S1+. Marked so CI's unit lane skips them."""

import pytest


@pytest.mark.integration
def test_pipeline_end_to_end_placeholder() -> None:
    pytest.skip("End-to-end pipeline integration test lands in S1+.")
