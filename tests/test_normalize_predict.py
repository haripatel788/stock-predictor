import pytest

from app.ticker import normalize_symbol


@pytest.mark.parametrize(
    "bad",
    ["", "123", "!!!", "^", "A" * 11, ".."],
)
def test_normalize_symbol_rejects_invalid(bad):
    with pytest.raises(ValueError):
        normalize_symbol(bad)


def test_normalize_symbol_accepts_common_tickers():
    assert normalize_symbol("  aapl ") == "AAPL"
    assert normalize_symbol("BRK.B") == "BRK.B"
    assert normalize_symbol("^GSPC") == "^GSPC"
