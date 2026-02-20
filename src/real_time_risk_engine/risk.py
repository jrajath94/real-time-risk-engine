"""Portfolio Value-at-Risk engine with Monte Carlo, historical, and parametric methods.

Provides position tracking, covariance estimation, multiple VaR methodologies,
and stress testing for multi-asset portfolios.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_MONTE_CARLO_SAMPLES: int = 10_000
DEFAULT_TRADING_DAYS_PER_YEAR: int = 252
MIN_HISTORY_FOR_COVARIANCE: int = 5
NUMERICAL_FLOOR: float = 1e-12


class VaRMethod(Enum):
    """Supported Value-at-Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class Position:
    """Represents a single portfolio position.

    Args:
        symbol: Ticker or asset identifier.
        quantity: Number of shares/units held (can be negative for short).
        current_price: Latest mark-to-market price per unit.
    """
    symbol: str
    quantity: float
    current_price: float

    @property
    def market_value(self) -> float:
        """Compute notional market value of the position."""
        return self.quantity * self.current_price


@dataclass
class StressScenario:
    """Defines a stress-test scenario applied to asset returns.

    Args:
        name: Human-readable scenario label.
        shocks: Mapping of symbol to absolute return shock (e.g. -0.10 for -10%).
    """
    name: str
    shocks: dict[str, float]


@dataclass
class RiskReport:
    """Container for a complete risk computation result.

    Args:
        portfolio_value: Total portfolio notional.
        var_value: Value-at-Risk dollar amount (loss, expressed as negative).
        cvar_value: Conditional VaR (expected shortfall).
        method: Which VaR methodology was used.
        confidence: Confidence level used.
        holding_period_days: Holding period in trading days.
    """
    portfolio_value: float
    var_value: float
    cvar_value: float
    method: VaRMethod
    confidence: float
    holding_period_days: int


@dataclass
class StressTestResult:
    """Result of applying a stress scenario.

    Args:
        scenario_name: Name of the stress scenario.
        portfolio_pnl: Dollar P&L under the scenario.
        asset_pnls: Per-asset P&L breakdown.
    """
    scenario_name: str
    portfolio_pnl: float
    asset_pnls: dict[str, float]


@dataclass
class Portfolio:
    """Tracks a collection of positions and their historical returns.

    Args:
        positions: List of current positions.
        returns_history: Array of shape (T, N) — T observations, N assets.
            Column order must match the order of ``positions``.
    """
    positions: list[Position] = field(default_factory=list)
    returns_history: Optional[np.ndarray] = None

    @property
    def symbols(self) -> list[str]:
        """Ordered list of asset symbols."""
        return [p.symbol for p in self.positions]

    @property
    def weights(self) -> np.ndarray:
        """Compute portfolio weight vector from market values."""
        values = np.array([p.market_value for p in self.positions])
        total = np.sum(np.abs(values))
        if total < NUMERICAL_FLOOR:
            return np.zeros(len(self.positions))
        return values / total

    @property
    def total_value(self) -> float:
        """Sum of absolute market values."""
        return float(np.sum(np.abs(
            [p.market_value for p in self.positions]
        )))

    def add_position(self, position: Position) -> None:
        """Add or update a position in the portfolio.

        Args:
            position: The position to add. If a position with the same
                symbol exists, its quantity and price are updated.
        """
        for i, existing in enumerate(self.positions):
            if existing.symbol == position.symbol:
                self.positions[i] = position
                logger.info("Updated position: %s", position.symbol)
                return
        self.positions.append(position)
        logger.info("Added position: %s", position.symbol)

    def remove_position(self, symbol: str) -> None:
        """Remove a position by symbol.

        Args:
            symbol: Asset identifier to remove.

        Raises:
            KeyError: If the symbol is not found in the portfolio.
        """
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                self.positions.pop(i)
                logger.info("Removed position: %s", symbol)
                return
        raise KeyError(f"Position not found: {symbol}")


def estimate_covariance(returns: np.ndarray) -> np.ndarray:
    """Estimate the covariance matrix from a returns history array.

    Args:
        returns: Array of shape (T, N) with T time observations and N assets.

    Returns:
        Covariance matrix of shape (N, N).

    Raises:
        ValueError: If fewer than MIN_HISTORY_FOR_COVARIANCE observations.
    """
    if returns.shape[0] < MIN_HISTORY_FOR_COVARIANCE:
        raise ValueError(
            f"Need at least {MIN_HISTORY_FOR_COVARIANCE} observations, "
            f"got {returns.shape[0]}"
        )
    return np.cov(returns, rowvar=False)


def _validate_inputs(
    portfolio: Portfolio,
    confidence: float,
) -> None:
    """Common input validation for VaR calculations.

    Args:
        portfolio: The portfolio to validate.
        confidence: Confidence level to validate.

    Raises:
        ValueError: On invalid inputs.
    """
    if not portfolio.positions:
        raise ValueError("Portfolio has no positions")
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"Confidence must be in (0, 1), got {confidence}"
        )


def _validate_returns_history(portfolio: Portfolio) -> np.ndarray:
    """Validate and return the portfolio's returns history.

    Args:
        portfolio: Portfolio whose returns_history to validate.

    Returns:
        The validated returns history array.

    Raises:
        ValueError: If returns_history is missing or mismatched.
    """
    if portfolio.returns_history is None:
        raise ValueError("Portfolio has no returns history")
    num_assets = len(portfolio.positions)
    if portfolio.returns_history.ndim == 1:
        if num_assets != 1:
            raise ValueError(
                "1-D returns history only valid for single-asset portfolio"
            )
        return portfolio.returns_history.reshape(-1, 1)
    if portfolio.returns_history.shape[1] != num_assets:
        raise ValueError(
            f"Returns history has {portfolio.returns_history.shape[1]} "
            f"columns but portfolio has {num_assets} positions"
        )
    return portfolio.returns_history


def historical_var(
    portfolio: Portfolio,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    holding_period_days: int = 1,
) -> RiskReport:
    """Compute Value-at-Risk using the historical simulation method.

    Applies actual historical returns to current portfolio weights
    and reads the loss quantile directly from the P&L distribution.

    Args:
        portfolio: Portfolio with positions and returns_history.
        confidence: Confidence level (e.g. 0.95 for 95% VaR).
        holding_period_days: Holding period in trading days.

    Returns:
        A RiskReport with the historical VaR and CVaR.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(portfolio, confidence)
    returns = _validate_returns_history(portfolio)
    weights = portfolio.weights
    portfolio_returns = returns @ weights
    portfolio_returns = _scale_to_holding_period(
        portfolio_returns, holding_period_days
    )
    var_pct = _compute_var_from_distribution(portfolio_returns, confidence)
    cvar_pct = _compute_cvar_from_distribution(
        portfolio_returns, confidence
    )
    total_val = portfolio.total_value
    return RiskReport(
        portfolio_value=total_val,
        var_value=var_pct * total_val,
        cvar_value=cvar_pct * total_val,
        method=VaRMethod.HISTORICAL,
        confidence=confidence,
        holding_period_days=holding_period_days,
    )


def parametric_var(
    portfolio: Portfolio,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    holding_period_days: int = 1,
) -> RiskReport:
    """Compute Value-at-Risk using the parametric (variance-covariance) method.

    Assumes portfolio returns are normally distributed. Uses the
    covariance matrix and a z-score to derive VaR analytically.

    Args:
        portfolio: Portfolio with positions and returns_history.
        confidence: Confidence level.
        holding_period_days: Holding period in trading days.

    Returns:
        A RiskReport with parametric VaR and estimated CVaR.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(portfolio, confidence)
    returns = _validate_returns_history(portfolio)
    weights = portfolio.weights
    cov_matrix = estimate_covariance(returns)
    port_variance = float(weights @ cov_matrix @ weights)
    port_std = np.sqrt(port_variance)
    z_score = _z_score_for_confidence(confidence)
    var_pct = -z_score * port_std * np.sqrt(holding_period_days)
    cvar_pct = _parametric_cvar(port_std, confidence, holding_period_days)
    total_val = portfolio.total_value
    return RiskReport(
        portfolio_value=total_val,
        var_value=var_pct * total_val,
        cvar_value=cvar_pct * total_val,
        method=VaRMethod.PARAMETRIC,
        confidence=confidence,
        holding_period_days=holding_period_days,
    )


def monte_carlo_var(
    portfolio: Portfolio,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    holding_period_days: int = 1,
    num_simulations: int = DEFAULT_MONTE_CARLO_SAMPLES,
    seed: Optional[int] = None,
) -> RiskReport:
    """Compute Value-at-Risk via Monte Carlo simulation.

    Draws correlated random returns from a multivariate normal
    distribution fitted to the historical covariance, then reads
    the loss quantile from the simulated P&L distribution.

    Args:
        portfolio: Portfolio with positions and returns_history.
        confidence: Confidence level.
        holding_period_days: Holding period in trading days.
        num_simulations: Number of Monte Carlo draws.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A RiskReport with Monte Carlo VaR and CVaR.

    Raises:
        ValueError: On invalid inputs.
    """
    _validate_inputs(portfolio, confidence)
    returns = _validate_returns_history(portfolio)
    weights = portfolio.weights
    simulated_returns = _generate_simulated_returns(
        returns, num_simulations, seed
    )
    portfolio_returns = simulated_returns @ weights
    portfolio_returns = _scale_to_holding_period(
        portfolio_returns, holding_period_days
    )
    var_pct = _compute_var_from_distribution(portfolio_returns, confidence)
    cvar_pct = _compute_cvar_from_distribution(
        portfolio_returns, confidence
    )
    total_val = portfolio.total_value
    return RiskReport(
        portfolio_value=total_val,
        var_value=var_pct * total_val,
        cvar_value=cvar_pct * total_val,
        method=VaRMethod.MONTE_CARLO,
        confidence=confidence,
        holding_period_days=holding_period_days,
    )


def stress_test(
    portfolio: Portfolio,
    scenarios: list[StressScenario],
) -> list[StressTestResult]:
    """Apply stress scenarios to the portfolio and compute P&L impact.

    Args:
        portfolio: The portfolio to stress test.
        scenarios: List of stress scenarios to apply.

    Returns:
        List of StressTestResult, one per scenario.

    Raises:
        ValueError: If portfolio has no positions.
    """
    if not portfolio.positions:
        raise ValueError("Portfolio has no positions")
    results: list[StressTestResult] = []
    for scenario in scenarios:
        result = _apply_single_scenario(portfolio, scenario)
        results.append(result)
    return results


def _apply_single_scenario(
    portfolio: Portfolio,
    scenario: StressScenario,
) -> StressTestResult:
    """Apply one stress scenario and return the result.

    Args:
        portfolio: Portfolio to stress.
        scenario: The scenario with per-asset shocks.

    Returns:
        StressTestResult with P&L breakdown.
    """
    asset_pnls: dict[str, float] = {}
    total_pnl = 0.0
    for pos in portfolio.positions:
        shock = scenario.shocks.get(pos.symbol, 0.0)
        pnl = pos.market_value * shock
        asset_pnls[pos.symbol] = pnl
        total_pnl += pnl
    logger.info(
        "Stress scenario '%s': portfolio P&L = %.2f",
        scenario.name, total_pnl,
    )
    return StressTestResult(
        scenario_name=scenario.name,
        portfolio_pnl=total_pnl,
        asset_pnls=asset_pnls,
    )


# --- Private helpers ---

def _compute_var_from_distribution(
    returns: np.ndarray,
    confidence: float,
) -> float:
    """Extract VaR quantile from a returns distribution.

    Args:
        returns: 1-D array of portfolio returns.
        confidence: Confidence level.

    Returns:
        VaR as a (negative) return value.
    """
    percentile = (1.0 - confidence) * 100.0
    return float(np.percentile(returns, percentile))


def _compute_cvar_from_distribution(
    returns: np.ndarray,
    confidence: float,
) -> float:
    """Compute Conditional VaR (Expected Shortfall) from a distribution.

    Args:
        returns: 1-D array of portfolio returns.
        confidence: Confidence level.

    Returns:
        CVaR as a (negative) return value — the mean of returns
        below the VaR threshold.
    """
    var_threshold = _compute_var_from_distribution(returns, confidence)
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) == 0:
        return var_threshold
    return float(np.mean(tail_returns))


def _scale_to_holding_period(
    daily_returns: np.ndarray,
    holding_period_days: int,
) -> np.ndarray:
    """Scale daily returns to a multi-day holding period via sqrt-time.

    Args:
        daily_returns: 1-D array of single-period returns.
        holding_period_days: Number of days to scale to.

    Returns:
        Scaled returns array.
    """
    if holding_period_days == 1:
        return daily_returns
    return daily_returns * np.sqrt(holding_period_days)


def _z_score_for_confidence(confidence: float) -> float:
    """Approximate the z-score for a given confidence level.

    Uses the Beasley-Springer-Moro approximation for the inverse
    normal CDF (rational approximation, accurate to ~1e-6).

    Args:
        confidence: Confidence level in (0, 1).

    Returns:
        The z-score (positive) such that P(Z <= z) = confidence.
    """
    # Rational approximation constants
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    p = confidence
    # Use symmetry: for p > 0.5, compute for 1-p and negate
    if p > 0.5:
        p_low = 1.0 - p
    else:
        p_low = p
    q = np.sqrt(-2.0 * np.log(p_low))
    numerator = (
        ((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]
    )
    denominator = (
        ((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4]) * q + 1.0
    )
    result = numerator / denominator
    if confidence > 0.5:
        return -result
    return result


def _parametric_cvar(
    port_std: float,
    confidence: float,
    holding_period_days: int,
) -> float:
    """Compute parametric CVaR under normality assumption.

    CVaR = -sigma * phi(z) / (1 - confidence) * sqrt(T)
    where phi is the standard normal PDF.

    Args:
        port_std: Portfolio standard deviation of daily returns.
        confidence: Confidence level.
        holding_period_days: Holding period in days.

    Returns:
        CVaR as a negative return value.
    """
    z = _z_score_for_confidence(confidence)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    cvar = -(port_std * phi_z / (1.0 - confidence)) * np.sqrt(
        holding_period_days
    )
    return float(cvar)


def _generate_simulated_returns(
    historical_returns: np.ndarray,
    num_simulations: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate correlated simulated returns via multivariate normal.

    Args:
        historical_returns: Array of shape (T, N).
        num_simulations: Number of simulated observations.
        seed: Optional RNG seed.

    Returns:
        Simulated returns of shape (num_simulations, N).
    """
    rng = np.random.default_rng(seed)
    mean_returns = np.mean(historical_returns, axis=0)
    cov_matrix = np.cov(historical_returns, rowvar=False)
    # Handle single-asset edge case where cov is scalar
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[float(cov_matrix)]])
        mean_returns = np.array([float(mean_returns)])
    return rng.multivariate_normal(
        mean_returns, cov_matrix, size=num_simulations
    )
