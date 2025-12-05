"""
Training Metrics Tracker for JAX
Real-time P&L, win rate, drawdown, and compliance monitoring during training

Mirrors PyTorch's MetricTrackingEvalCallback functionality for JAX training.
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class TrainingMetricsTracker:
    """
    Track actual trading metrics during JAX training.
    
    Monitors:
    - P&L (total, per-trade, gross profit/loss)
    - Win rate and profit factor
    - Maximum trailing drawdown
    - Apex compliance violations (4:59 PM close, drawdown limit)
    
    Usage:
        tracker = TrainingMetricsTracker(market='NQ', phase=1)
        
        # After each episode completes
        tracker.update(episode_info, timestep=current_update)
        
        # Periodically log metrics
        if update % 10 == 0:
            tracker.log_to_console(update, total_updates)
            tracker.save_to_json('results/metrics.json')
    """
    
    def __init__(
        self,
        market: str,
        phase: int,
        checkpoint_dir: str = "models/phase1_jax",  # NEW: for saving metrics
        initial_balance: float = 50000.0,
        drawdown_limit: float = 2500.0,
        enable_tensorboard: bool = False
    ):
        """
        Initialize metrics tracker.

        Args:
            market: Market symbol (ES, NQ, etc.)
            phase: Training phase (1 or 2)
            checkpoint_dir: Directory to save metrics JSON
            initial_balance: Starting balance for calculations
            drawdown_limit: Max allowed trailing drawdown (Apex: $2,500)
            enable_tensorboard: Enable TensorBoard logging (future feature)
        """
        self.market = market
        self.phase = phase
        self.checkpoint_dir = checkpoint_dir
        self.initial_balance = initial_balance
        self.drawdown_limit = drawdown_limit
        self.tensorboard_enabled = enable_tensorboard
        
        # P&L tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        
        # Drawdown tracking
        self.peak_balance = initial_balance
        self.current_balance = initial_balance
        self.max_trailing_drawdown = 0.0
        
        # Compliance violations
        self.close_violations = 0  # Positions held past 4:59 PM
        self.dd_violations = 0  # Drawdown exceeded $2,500
        
        # Episode history
        self.episode_returns: List[float] = []
        self.episode_balances: List[float] = []
        self.episode_trade_counts: List[int] = []

        # Real-time series buffers (bounded)
        self.history_limit = 200
        self.win_rate_series: List[Dict[str, Any]] = []
        self.policy_health_series: List[Dict[str, Any]] = []
        self.network_stability_series: List[Dict[str, Any]] = []
        self.action_distribution: List[Dict[str, Any]] = []

        # Latest telemetry for dashboard
        self.sps: float = 0.0
        self.timesteps_current: int = 0
        self.timesteps_target: Optional[int] = None
        self.entropy: float = 0.0
        self.kl_divergence: float = 0.0
        self.learning_rate: Optional[float] = None
        self.status: str = "TRAINING"
        self.last_trade: Dict[str, Any] = {"pnl": 0.0, "direction": "Flat", "bars_held": 0}
        
        # Metadata
        self.start_time = datetime.now()
        self.total_episodes = 0
    
    def record_episode(
        self,
        final_balance: float,
        num_trades: int,
        win_rate: float,
        total_pnl: float
    ) -> None:
        """
        Simplified episode recording method (for JAX integration).

        Args:
            final_balance: Ending portfolio value
            num_trades: Number of trades executed
            win_rate: Win rate (0.0 to 1.0)
            total_pnl: Total P&L for episode
        """
        episode_info = {
            'final_balance': final_balance,
            'episode_return': total_pnl,
            'position_closed': num_trades > 0,
            'trade_pnl': total_pnl,
            'forced_close': False
        }
        self.update(episode_info, timestep=0)

    def log_summary(self) -> None:
        """Print condensed metrics summary (for training loop integration)."""
        metrics = self.get_metrics()
        print(f"  [Metrics] Trades: {metrics['total_trades']} | " +
              f"Win%: {metrics['win_rate']:.1%} | " +
              f"PnL: ${metrics['total_pnl']:,.0f} | " +
              f"DD: ${metrics['max_trailing_drawdown']:,.0f}")

    def save_metrics(self) -> None:
        """Save metrics to checkpoint directory."""
        from pathlib import Path
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        output_path = f"{self.checkpoint_dir}/training_metrics_{self.market}.json"
        self.save_to_json(output_path, include_history=True)

    def update(self, episode_info: Dict[str, Any], timestep: int) -> None:
        """
        Update metrics after an episode completes.
        
        Args:
            episode_info: Dictionary with episode results
                - 'final_balance': Ending balance
                - 'trade_pnl': P&L from last trade (if closed)
                - 'position_closed': Whether a trade closed
                - 'forced_close': Whether position was force-closed at 4:59 PM
                - 'episode_return': Total episode return
            timestep: Current training timestep/update number
        """
        self.total_episodes += 1
        
        # Extract episode data
        final_balance = float(episode_info.get('final_balance', self.initial_balance))
        trade_pnl = float(episode_info.get('trade_pnl', 0.0))
        position_closed = bool(episode_info.get('position_closed', False))
        forced_close = bool(episode_info.get('forced_close', False))
        episode_return = float(episode_info.get('episode_return', 0.0))
        
        # Update balance tracking
        self.current_balance = final_balance
        self.peak_balance = max(self.peak_balance, final_balance)
        
        # Calculate trailing drawdown
        trailing_dd = self.peak_balance - final_balance
        self.max_trailing_drawdown = max(self.max_trailing_drawdown, trailing_dd)
        
        # Track P&L from completed trades
        if position_closed:
            self.total_trades += 1
            
            if trade_pnl > 0:
                self.winning_trades += 1
                self.gross_profit += trade_pnl
            elif trade_pnl < 0:
                self.losing_trades += 1
                self.gross_loss += abs(trade_pnl)
            
            self.total_pnl += trade_pnl
        
        # Track compliance violations
        if forced_close:
            self.close_violations += 1
        
        # Store episode history
        self.episode_returns.append(episode_return)
        self.episode_balances.append(final_balance)
        self.episode_trade_counts.append(int(position_closed))

        # Trim histories to avoid unbounded growth
        self._trim_history()

    def record_update(
        self,
        timesteps: int,
        sps: float,
        entropy: float,
        approx_kl: float,
        action_dist: Any,
        policy_loss: float,
        value_loss: float,
        win_rate: float,
        mean_return: float,
        learning_rate: Optional[float] = None,
        timesteps_target: Optional[int] = None,
        last_trade: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record per-update telemetry for real-time dashboard."""
        self.sps = float(sps)
        self.timesteps_current = int(timesteps)
        if timesteps_target is not None:
            self.timesteps_target = int(timesteps_target)

        if entropy is not None:
            self.entropy = float(entropy)
        if approx_kl is not None:
            self.kl_divergence = float(approx_kl)
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        if last_trade is not None:
            self.last_trade = last_trade

        # Maintain bounded series for charts
        self._append_limited(self.win_rate_series, {
            'episode': timesteps,
            'win_rate': float(win_rate)
        })

        self._append_limited(self.policy_health_series, {
            'step': timesteps,
            'entropy': float(entropy) if entropy is not None else None,
            'kl': float(approx_kl) if approx_kl is not None else None
        })

        self._append_limited(self.network_stability_series, {
            'step': timesteps,
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'kl_div': float(approx_kl)
        })

        if action_dist is not None:
            # action_dist order expected: HOLD, BUY, SELL
            try:
                hold, buy, sell = float(action_dist[0]), float(action_dist[1]), float(action_dist[2])
            except Exception:
                hold = buy = sell = 0.0
            self._append_limited(self.action_distribution, {
                'episode': timesteps,
                'buy': buy,
                'sell': sell,
                'hold': hold,
                'pm': 0.0  # placeholder for portfolio manager actions
            })

        # Also track mean return progression for completeness
        self._append_limited(self.episode_returns, float(mean_return))
        self._append_limited(self.episode_balances, self.current_balance)

        self._trim_history()
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary with all current metrics
        """
        # Calculate derived metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        
        profit_factor = 0.0
        if self.gross_loss > 0:
            profit_factor = self.gross_profit / self.gross_loss
        
        avg_win = self.gross_profit / self.winning_trades if self.winning_trades > 0 else 0.0
        avg_loss = self.gross_loss / self.losing_trades if self.losing_trades > 0 else 0.0
        
        avg_return = float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
        std_return = float(np.std(self.episode_returns)) if len(self.episode_returns) > 1 else 0.0
        
        # Sharpe-like ratio (episode returns)
        sharpe = avg_return / std_return if std_return > 1e-8 else 0.0

        apex_distance = self.drawdown_limit - self.max_trailing_drawdown

        metrics = {
            # P&L metrics
            'total_pnl': self.total_pnl,
            'current_balance': self.current_balance,
            'total_return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            
            # Trade statistics
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            
            # Risk metrics
            'max_trailing_drawdown': self.max_trailing_drawdown,
            'max_dd_pct': (self.max_trailing_drawdown / self.peak_balance) * 100,
            'peak_balance': self.peak_balance,
            
            # Episode statistics
            'total_episodes': self.total_episodes,
            'avg_episode_return': avg_return,
            'episode_return_std': std_return,
            'sharpe_ratio': sharpe,
            
            # Compliance
            'close_violations': self.close_violations,
            'dd_violations': self.dd_violations,
            'apex_compliant': (self.max_trailing_drawdown < self.drawdown_limit and self.close_violations == 0),

            # Real-time telemetry
            'sps': self.sps,
            'timesteps': {
                'current': self.timesteps_current,
                'target': self.timesteps_target
            },
            'entropy': self.entropy,
            'kl_divergence': self.kl_divergence,
            'learning_rate': self.learning_rate,
            'status': self.status,
            'drawdown_limit': self.drawdown_limit,
            'apex_distance': apex_distance,
            'last_trade': self.last_trade,

            # Series for charts
            'win_rate_series': list(self.win_rate_series),
            'policy_health_series': list(self.policy_health_series),
            'network_stability_series': list(self.network_stability_series),
            'action_distribution': list(self.action_distribution),
        }

        return metrics
    
    def log_to_console(self, update_num: int, total_updates: int) -> None:
        """
        Print formatted metrics table to console.
        
        Args:
            update_num: Current update number
            total_updates: Total updates in training
        """
        metrics = self.get_metrics()
        
        # Progress bar
        progress_pct = (update_num / total_updates) * 100 if total_updates > 0 else 0
        
        print(f"\n{'=' * 80}")
        print(f"  TRAINING METRICS - {self.market} Phase {self.phase} | Update {update_num}/{total_updates} ({progress_pct:.1f}%)")
        print(f"{'=' * 80}")
        print(f"║ P&L SUMMARY                                                              ║")
        print(f"├──────────────────────────────────────────────────────────────────────────┤")
        print(f"║ Total P&L:              ${metrics['total_pnl']:>20,.2f}                  ║")
        print(f"║ Current Balance:        ${metrics['current_balance']:>20,.2f}           ║")
        print(f"║ Total Return:           {metrics['total_return_pct']:>20,.2f}%          ║")
        print(f"║ Peak Balance:           ${metrics['peak_balance']:>20,.2f}              ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║ TRADE STATISTICS                                                         ║")
        print(f"├──────────────────────────────────────────────────────────────────────────┤")
        print(f"║ Total Trades:           {metrics['total_trades']:>20}                   ║")
        print(f"║ Winning Trades:         {metrics['winning_trades']:>20}                 ║")
        print(f"║ Losing Trades:          {metrics['losing_trades']:>20}                  ║")
        print(f"║ Win Rate:               {metrics['win_rate']:>20.1%}                    ║")
        print(f"║ Profit Factor:          {metrics['profit_factor']:>20.2f}               ║")
        print(f"║ Avg Win:                ${metrics['avg_win']:>20,.2f}                   ║")
        print(f"║ Avg Loss:               ${metrics['avg_loss']:>20,.2f}                  ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║ RISK METRICS                                                             ║")
        print(f"├──────────────────────────────────────────────────────────────────────────┤")
        print(f"║ Max Trailing DD:        ${metrics['max_trailing_drawdown']:>20,.2f}     ║")
        print(f"║ Max DD %:               {metrics['max_dd_pct']:>20,.2f}%                ║")
        print(f"║ Sharpe Ratio:           {metrics['sharpe_ratio']:>20.2f}                ║")
        print(f"╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║ APEX COMPLIANCE                                                          ║")
        print(f"├──────────────────────────────────────────────────────────────────────────┤")
        print(f"║ 4:59 PM Violations:     {metrics['close_violations']:>20}               ║")
        print(f"║ DD > $2,500 Violations: {metrics['dd_violations']:>20}                  ║")
        
        # Compliance status
        if metrics['apex_compliant']:
            status = "✅ COMPLIANT"
        else:
            status = "❌ NOT COMPLIANT"
        print(f"║ Status:                 {status:>20}                                     ║")
        print(f"{'=' * 80}\n")
    
    def save_to_json(self, output_path: str, include_history: bool = True) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            output_path: Path to save JSON file
            include_history: Whether to include full episode history
        """
        metrics = self.get_metrics()
        
        # Add metadata
        metrics['market'] = self.market
        metrics['phase'] = self.phase
        metrics['initial_balance'] = self.initial_balance
        metrics['drawdown_limit'] = self.drawdown_limit
        metrics['start_time'] = self.start_time.isoformat()
        metrics['last_updated'] = datetime.now().isoformat()
        
        # Optionally include full history (bounded)
        if include_history:
            metrics['episode_history'] = {
                'returns': [float(r) for r in self.episode_returns[-self.history_limit:]],
                'balances': [float(b) for b in self.episode_balances[-self.history_limit:]],
                'trade_counts': [int(tc) for tc in self.episode_trade_counts[-self.history_limit:]],
            }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _append_limited(self, arr: List[Any], value: Any) -> None:
        arr.append(value)
        if len(arr) > self.history_limit:
            del arr[0:len(arr) - self.history_limit]

    def _trim_history(self) -> None:
        if len(self.episode_returns) > self.history_limit:
            self.episode_returns = self.episode_returns[-self.history_limit:]
        if len(self.episode_balances) > self.history_limit:
            self.episode_balances = self.episode_balances[-self.history_limit:]
        if len(self.episode_trade_counts) > self.history_limit:
            self.episode_trade_counts = self.episode_trade_counts[-self.history_limit:]
        if len(self.win_rate_series) > self.history_limit:
            self.win_rate_series = self.win_rate_series[-self.history_limit:]
        if len(self.policy_health_series) > self.history_limit:
            self.policy_health_series = self.policy_health_series[-self.history_limit:]
        if len(self.network_stability_series) > self.history_limit:
            self.network_stability_series = self.network_stability_series[-self.history_limit:]
        if len(self.action_distribution) > self.history_limit:
            self.action_distribution = self.action_distribution[-self.history_limit:]
    
    def reset(self) -> None:
        """Reset all metrics (useful for multi-phase training)."""
        self.__init__(
            market=self.market,
            phase=self.phase,
            initial_balance=self.initial_balance,
            drawdown_limit=self.drawdown_limit,
            enable_tensorboard=self.tensorboard_enabled
        )
