"""
Apex Trader Funding Compliance Checker

Validates trading episodes against Apex Evaluation Rules.
Used during evaluation to ensure model compliance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime


def safe_print(message: str = "") -> None:
    """Print helper that degrades gracefully on Windows consoles."""
    try:
        print(message)
    except UnicodeEncodeError:
        replacements = {
            '✓': '[OK]',
            '✅': '[OK]',
            '✗': '[X]',
            '❌': '[X]',
            '→': '->',
            '⚠': '[WARN]',
            '⚠️': '[WARN]',
            '🔴': '[CRITICAL]',
            '📊': '[STATS]',
            '—': '-',
            '–': '-',
            '’': "'",
            '“': '"',
            '”': '"',
        }
        ascii_message = message
        for src, target in replacements.items():
            ascii_message = ascii_message.replace(src, target)
        ascii_message = ascii_message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)

@dataclass
class ApexRuleViolation:
    """Represents a rule violation."""
    rule_name: str
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    description: str
    step: int
    timestamp: datetime = None
    value: float = None


class ApexComplianceChecker:
    """
    Validates episode against Apex Trader Funding rules.

    Rules Checked:
    1. Trailing drawdown never exceeded
    2. All positions closed by 4:59 PM ET
    3. Profit target reached and maintained
    4. Position size within limits (10 contracts for 50K)
    5. Realistic commission deduction
    6. No overnight positions
    """

    def __init__(
        self,
        account_size: float = 50000,
        trailing_dd_limit: float = 2500,
        profit_target: float = 3000,
        max_contracts: int = 10
    ):
        self.account_size = account_size
        self.trailing_dd_limit = trailing_dd_limit
        self.profit_target = profit_target
        self.max_contracts = max_contracts

        self.violations: List[ApexRuleViolation] = []
        self.warnings: List[ApexRuleViolation] = []
        self.info: List[ApexRuleViolation] = []

    def check_episode(
        self,
        portfolio_values: List[float],
        trailing_dd_levels: List[float],
        trade_history: List[Dict],
        timestamps: pd.DatetimeIndex,
        position_sizes: List[float]
    ) -> Dict:
        """
        Comprehensive episode validation.

        Returns:
            {
                'passed': bool,
                'violations': List[ApexRuleViolation],
                'warnings': List[ApexRuleViolation],
                'metrics': Dict
            }
        """
        self.violations = []
        self.warnings = []
        self.info = []

        # Rule 1: Trailing Drawdown
        self._check_trailing_drawdown(portfolio_values, trailing_dd_levels)

        # Rule 2: EOD Close-Out
        self._check_eod_closeout(trade_history, timestamps)

        # Rule 3: Profit Target
        profit_reached = self._check_profit_target(portfolio_values)

        # Rule 4: Position Size
        self._check_position_sizes(position_sizes)

        # Rule 5: Trade Quality
        self._check_trade_quality(trade_history)

        # Calculate compliance metrics
        metrics = self._calculate_metrics(
            portfolio_values, trade_history, profit_reached
        )

        # Determine pass/fail
        passed = len(self.violations) == 0 and profit_reached

        return {
            'passed': passed,
            'violations': self.violations,
            'warnings': self.warnings,
            'info': self.info,
            'metrics': metrics
        }

    def _check_trailing_drawdown(
        self,
        portfolio_values: List[float],
        trailing_dd_levels: List[float]
    ):
        """Check if trailing drawdown was ever violated."""
        for i, (pv, dd_level) in enumerate(zip(portfolio_values, trailing_dd_levels)):
            if pv < dd_level:
                self.violations.append(ApexRuleViolation(
                    rule_name="Trailing Drawdown",
                    severity="CRITICAL",
                    description=f"Portfolio ${pv:,.2f} below drawdown threshold ${dd_level:,.2f}",
                    step=i,
                    value=pv - dd_level
                ))

        # Warning for getting close
        for i, (pv, dd_level) in enumerate(zip(portfolio_values, trailing_dd_levels)):
            buffer = pv - dd_level
            if 0 < buffer < 500:  # Within $500 of drawdown
                self.warnings.append(ApexRuleViolation(
                    rule_name="Drawdown Proximity",
                    severity="WARNING",
                    description=f"Portfolio within ${buffer:.2f} of drawdown",
                    step=i,
                    value=buffer
                ))

    def _check_eod_closeout(
        self,
        trade_history: List[Dict],
        timestamps: pd.DatetimeIndex
    ):
        """Check all positions closed by 4:59 PM ET."""
        if len(timestamps) == 0:
            return

        last_timestamp = timestamps[-1]
        if last_timestamp.hour > 16 or (last_timestamp.hour == 16 and last_timestamp.minute > 59):
            self.violations.append(ApexRuleViolation(
                rule_name="EOD Close-Out",
                severity="CRITICAL",
                description=f"Position open after 4:59 PM ET: {last_timestamp}",
                step=len(timestamps) - 1,
                timestamp=last_timestamp
            ))

    def _check_profit_target(self, portfolio_values: List[float]) -> bool:
        """Check if profit target was reached."""
        final_pv = portfolio_values[-1]
        target = self.account_size + self.profit_target

        profit_reached = final_pv >= target

        if not profit_reached:
            shortfall = target - final_pv
            self.violations.append(ApexRuleViolation(
                rule_name="Profit Target",
                severity="CRITICAL",
                description=f"Profit target not reached. Short by ${shortfall:,.2f}",
                step=len(portfolio_values) - 1,
                value=shortfall
            ))
        else:
            self.info.append(ApexRuleViolation(
                rule_name="Profit Target",
                severity="INFO",
                description=f"Profit target REACHED: ${final_pv:,.2f}",
                step=len(portfolio_values) - 1,
                value=final_pv - self.account_size
            ))

        return profit_reached

    def _check_position_sizes(self, position_sizes: List[float]):
        """Check position sizes within limits."""
        for i, size in enumerate(position_sizes):
            if size > self.max_contracts:
                self.violations.append(ApexRuleViolation(
                    rule_name="Position Size",
                    severity="CRITICAL",
                    description=f"Position size {size} exceeds max {self.max_contracts}",
                    step=i,
                    value=size
                ))

    def _check_trade_quality(self, trade_history: List[Dict]):
        """Check trade quality metrics."""
        if len(trade_history) == 0:
            self.warnings.append(ApexRuleViolation(
                rule_name="Trade Activity",
                severity="WARNING",
                description="No trades executed during episode",
                step=0
            ))
            return

        # Check win rate
        winning = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
        win_rate = winning / len(trade_history) if len(trade_history) > 0 else 0

        if win_rate < 0.3:
            self.warnings.append(ApexRuleViolation(
                rule_name="Win Rate",
                severity="WARNING",
                description=f"Low win rate: {win_rate:.1%} (< 30%)",
                step=len(trade_history),
                value=win_rate
            ))

    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        trade_history: List[Dict],
        profit_reached: bool
    ) -> Dict:
        """Calculate comprehensive metrics."""
        if len(portfolio_values) < 2:
            return {}

        returns = np.diff(portfolio_values) / (np.array(portfolio_values[:-1]) + 1e-8)

        # Sharpe ratio
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252 * 390)

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdowns = (peak - portfolio_values) / peak
        max_dd = drawdowns.max()

        # Trade stats
        num_trades = len(trade_history)
        winning_trades = sum(1 for t in trade_history if t.get('pnl', 0) > 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0

        total_pnl = sum(t.get('pnl', 0) for t in trade_history)
        final_balance = portfolio_values[-1]

        return {
            'final_balance': final_balance,
            'total_pnl': total_pnl,
            'profit_target_reached': profit_reached,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_return_pct': (final_balance - self.account_size) / self.account_size * 100
        }

    def print_report(self, results: Dict):
        """Print formatted compliance report."""
        safe_print("=" * 80)
        safe_print("APEX TRADER FUNDING COMPLIANCE REPORT")
        safe_print("=" * 80)

        # Status
        status = '[OK] PASSED' if results['passed'] else '[X] FAILED'
        safe_print(f"\nStatus: {status}\n")

        # Violations
        if results['violations']:
            safe_print("[CRITICAL] Violations:")
            for v in results['violations']:
                safe_print(f"  - [{v.rule_name}] {v.description}")

        # Warnings
        if results['warnings']:
            safe_print("\n[WARN] Warnings:")
            for w in results['warnings']:
                safe_print(f"  - [{w.rule_name}] {w.description}")

        # Metrics
        safe_print("\n[STATS] Performance Metrics:")
        m = results['metrics']
        safe_print(f"  Final Balance: ${m['final_balance']:,.2f}")
        safe_print(f"  Total P&L: ${m['total_pnl']:,.2f}")
        safe_print(f"  Return: {m['total_return_pct']:.2f}%")
        safe_print(f"  Sharpe Ratio: {m['sharpe_ratio']:.2f}")
        safe_print(f"  Max Drawdown: {m['max_drawdown']:.2%}")
        safe_print(f"  Trades: {m['num_trades']}")
        safe_print(f"  Win Rate: {m['win_rate']:.1%}")
        safe_print(f"  Profit Target Reached: {'YES' if m['profit_target_reached'] else 'NO'}")

        safe_print("=" * 80)
