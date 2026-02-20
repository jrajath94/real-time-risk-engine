"""Tests for the real-time risk engine.

Covers position tracking, covariance estimation, all three VaR methods
(historical, parametric, Monte Carlo), CVaR, stress testing, and edge cases.
"""
import numpy as np
import pytest

from real_time_risk_engine.risk import (
    DEFAULT_CONFIDENCE_LEVEL,
    MIN_HISTORY_FOR_COVARIANCE,
    Portfolio,
    Position,
    RiskReport,
    StressScenario,
    StressTestResult,
    VaRMethod,
    estimate_covariance,
    historical_var,
    monte_carlo_var,
    parametric_var,
    stress_test,
)


# --- Fixtures ---

@pytest.fixture
def single_position() -> Position:
    """A single equity position."""
    return Position(symbol="AAPL", quantity=100, current_price=150.0)


@pytest.fixture
def multi_positions() -> list[Position]:
    """Multiple positions for a diversified portfolio."""
    return [
        Position(symbol="AAPL", quantity=100, current_price=150.0),
        Position(symbol="GOOGL", quantity=50, current_price=2800.0),
        Position(symbol="MSFT", quantity=200, current_price=300.0),
    ]


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Synthetic return history for 3 assets, 100 days."""
    rng = np.random.default_rng(42)
    # Slightly correlated returns
    mean = [0.0005, 0.0003, 0.0004]
    cov = [
        [0.0004, 0.0001, 0.00015],
        [0.0001, 0.0003, 0.0001],
        [0.00015, 0.0001, 0.00035],
    ]
    return rng.multivariate_normal(mean, cov, size=100)


@pytest.fixture
def portfolio_with_history(
    multi_positions: list[Position],
    sample_returns: np.ndarray,
) -> Portfolio:
    """A fully populated portfolio with returns history."""
    portfolio = Portfolio(positions=multi_positions)
    portfolio.returns_history = sample_returns
    return portfolio


@pytest.fixture
def single_asset_portfolio(single_position: Position) -> Portfolio:
    """Portfolio with one asset and 1-D returns."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0005, 0.02, size=100)
    portfolio = Portfolio(positions=[single_position])
    portfolio.returns_history = returns
    return portfolio


# --- Position Tests ---

class TestPosition:
    """Tests for Position dataclass."""

    def test_market_value_long(self, single_position: Position) -> None:
        """Market value of a long position is quantity * price."""
        assert single_position.market_value == 15000.0

    def test_market_value_short(self) -> None:
        """Market value of a short position is negative."""
        pos = Position(symbol="TSLA", quantity=-50, current_price=200.0)
        assert pos.market_value == -10000.0

    @pytest.mark.parametrize("qty,price,expected", [
        (0, 100.0, 0.0),
        (1, 0.0, 0.0),
        (10, 50.0, 500.0),
        (-10, 50.0, -500.0),
    ])
    def test_market_value_parametrized(
        self, qty: float, price: float, expected: float
    ) -> None:
        """Market value calculation across various inputs."""
        pos = Position(symbol="TEST", quantity=qty, current_price=price)
        assert pos.market_value == pytest.approx(expected)


# --- Portfolio Tests ---

class TestPortfolio:
    """Tests for Portfolio management."""

    def test_add_position(self, single_position: Position) -> None:
        """Adding a position increases the position list."""
        portfolio = Portfolio()
        portfolio.add_position(single_position)
        assert len(portfolio.positions) == 1
        assert portfolio.symbols == ["AAPL"]

    def test_update_existing_position(self) -> None:
        """Adding a position with same symbol updates in place."""
        portfolio = Portfolio()
        portfolio.add_position(Position("AAPL", 100, 150.0))
        portfolio.add_position(Position("AAPL", 200, 155.0))
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].quantity == 200

    def test_remove_position(self) -> None:
        """Removing a position by symbol works."""
        portfolio = Portfolio()
        portfolio.add_position(Position("AAPL", 100, 150.0))
        portfolio.remove_position("AAPL")
        assert len(portfolio.positions) == 0

    def test_remove_nonexistent_raises(self) -> None:
        """Removing a non-existent position raises KeyError."""
        portfolio = Portfolio()
        with pytest.raises(KeyError, match="Position not found"):
            portfolio.remove_position("FAKE")

    def test_weights_sum_to_one(
        self, multi_positions: list[Position]
    ) -> None:
        """Portfolio weights sum to 1.0 for long-only portfolios."""
        portfolio = Portfolio(positions=multi_positions)
        weights = portfolio.weights
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)

    def test_total_value(
        self, multi_positions: list[Position]
    ) -> None:
        """Total value sums absolute market values."""
        portfolio = Portfolio(positions=multi_positions)
        expected = 100 * 150 + 50 * 2800 + 200 * 300  # 215000
        assert portfolio.total_value == pytest.approx(expected)


# --- Covariance Tests ---

class TestCovariance:
    """Tests for covariance estimation."""

    def test_covariance_shape(self, sample_returns: np.ndarray) -> None:
        """Covariance matrix has shape (N, N)."""
        cov = estimate_covariance(sample_returns)
        assert cov.shape == (3, 3)

    def test_covariance_symmetric(self, sample_returns: np.ndarray) -> None:
        """Covariance matrix is symmetric."""
        cov = estimate_covariance(sample_returns)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_covariance_too_few_observations(self) -> None:
        """Raises ValueError with insufficient observations."""
        tiny = np.random.default_rng(0).normal(size=(3, 2))
        with pytest.raises(ValueError, match="at least"):
            estimate_covariance(tiny)


# --- Historical VaR Tests ---

class TestHistoricalVaR:
    """Tests for historical VaR calculation."""

    def test_returns_risk_report(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Historical VaR returns a valid RiskReport."""
        report = historical_var(portfolio_with_history)
        assert isinstance(report, RiskReport)
        assert report.method == VaRMethod.HISTORICAL

    def test_var_is_negative(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """VaR value should be negative (represents a loss)."""
        report = historical_var(portfolio_with_history)
        assert report.var_value < 0

    def test_cvar_worse_than_var(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """CVaR (expected shortfall) should be <= VaR."""
        report = historical_var(portfolio_with_history)
        assert report.cvar_value <= report.var_value

    def test_higher_confidence_larger_var(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Higher confidence means a more extreme VaR."""
        var_95 = historical_var(portfolio_with_history, confidence=0.95)
        var_99 = historical_var(portfolio_with_history, confidence=0.99)
        # 99% VaR should be a larger loss (more negative)
        assert var_99.var_value < var_95.var_value

    def test_single_asset(
        self, single_asset_portfolio: Portfolio
    ) -> None:
        """Historical VaR works with a single-asset portfolio."""
        report = historical_var(single_asset_portfolio)
        assert report.var_value < 0


# --- Parametric VaR Tests ---

class TestParametricVaR:
    """Tests for parametric (variance-covariance) VaR."""

    def test_returns_parametric_method(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Parametric VaR report has correct method tag."""
        report = parametric_var(portfolio_with_history)
        assert report.method == VaRMethod.PARAMETRIC

    def test_var_is_negative(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Parametric VaR should be negative."""
        report = parametric_var(portfolio_with_history)
        assert report.var_value < 0

    def test_cvar_worse_than_var_parametric(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Parametric CVaR should be worse than VaR."""
        report = parametric_var(portfolio_with_history)
        assert report.cvar_value <= report.var_value


# --- Monte Carlo VaR Tests ---

class TestMonteCarloVaR:
    """Tests for Monte Carlo VaR."""

    def test_returns_monte_carlo_method(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """MC VaR report has correct method tag."""
        report = monte_carlo_var(portfolio_with_history, seed=42)
        assert report.method == VaRMethod.MONTE_CARLO

    def test_reproducible_with_seed(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Same seed produces identical VaR."""
        r1 = monte_carlo_var(portfolio_with_history, seed=123)
        r2 = monte_carlo_var(portfolio_with_history, seed=123)
        assert r1.var_value == pytest.approx(r2.var_value)

    def test_different_seeds_differ(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Different seeds produce different VaR (in general)."""
        r1 = monte_carlo_var(portfolio_with_history, seed=1)
        r2 = monte_carlo_var(portfolio_with_history, seed=999)
        # Extremely unlikely to be exactly equal
        assert r1.var_value != r2.var_value

    def test_var_is_negative(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """MC VaR should be negative."""
        report = monte_carlo_var(portfolio_with_history, seed=42)
        assert report.var_value < 0


# --- Stress Test ---

class TestStressTest:
    """Tests for stress testing."""

    def test_stress_test_single_scenario(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Stress test with a market crash scenario."""
        scenarios = [
            StressScenario(
                name="Market Crash",
                shocks={"AAPL": -0.20, "GOOGL": -0.15, "MSFT": -0.25},
            )
        ]
        results = stress_test(portfolio_with_history, scenarios)
        assert len(results) == 1
        assert results[0].scenario_name == "Market Crash"
        assert results[0].portfolio_pnl < 0

    def test_stress_test_positive_scenario(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Stress test with a bull market scenario."""
        scenarios = [
            StressScenario(
                name="Bull Run",
                shocks={"AAPL": 0.10, "GOOGL": 0.15, "MSFT": 0.12},
            )
        ]
        results = stress_test(portfolio_with_history, scenarios)
        assert results[0].portfolio_pnl > 0

    def test_stress_test_no_positions_raises(self) -> None:
        """Stress test on empty portfolio raises ValueError."""
        empty = Portfolio()
        with pytest.raises(ValueError, match="no positions"):
            stress_test(empty, [StressScenario("X", {"A": -0.1})])

    def test_stress_test_partial_shocks(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Symbols without shocks get 0 P&L."""
        scenarios = [
            StressScenario(
                name="AAPL Only",
                shocks={"AAPL": -0.50},
            )
        ]
        results = stress_test(portfolio_with_history, scenarios)
        pnls = results[0].asset_pnls
        assert pnls["GOOGL"] == 0.0
        assert pnls["MSFT"] == 0.0
        assert pnls["AAPL"] < 0


# --- Error Cases ---

class TestErrorCases:
    """Tests for error handling and edge cases."""

    def test_var_no_positions(self) -> None:
        """VaR on empty portfolio raises ValueError."""
        empty = Portfolio()
        with pytest.raises(ValueError, match="no positions"):
            historical_var(empty)

    def test_var_invalid_confidence(
        self, portfolio_with_history: Portfolio
    ) -> None:
        """Confidence outside (0,1) raises ValueError."""
        with pytest.raises(ValueError, match="Confidence"):
            historical_var(portfolio_with_history, confidence=1.5)

    def test_var_no_returns_history(self) -> None:
        """VaR without returns history raises ValueError."""
        portfolio = Portfolio(
            positions=[Position("X", 100, 50.0)]
        )
        with pytest.raises(ValueError, match="no returns history"):
            historical_var(portfolio)

    def test_mismatched_returns_columns(self) -> None:
        """Returns with wrong column count raises ValueError."""
        portfolio = Portfolio(
            positions=[Position("A", 100, 50.0), Position("B", 50, 100.0)]
        )
        portfolio.returns_history = np.random.default_rng(0).normal(
            size=(50, 3)
        )
        with pytest.raises(ValueError, match="columns"):
            historical_var(portfolio)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5, 2.0])
    def test_invalid_confidence_values(
        self,
        portfolio_with_history: Portfolio,
        confidence: float,
    ) -> None:
        """Various invalid confidence values all raise errors."""
        with pytest.raises(ValueError):
            parametric_var(portfolio_with_history, confidence=confidence)
