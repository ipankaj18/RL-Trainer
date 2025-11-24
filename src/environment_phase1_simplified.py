"""
Simplified reward function for Phase 1 training.

Phase 1 Focus: Entry Signal Quality
- Reward based on trade P&L
- Simple, direct feedback
- No complex portfolio management penalties
"""

def calculate_phase1_reward_simplified(env, exit_reason: str, trade_pnl: float, portfolio_value: float) -> float:
    """
    Simplified reward function for Phase 1 focusing on entry quality.

    Args:
        env: The trading environment instance
        exit_reason: Reason for position exit ('tp', 'sl', 'apex', 'eod', etc.)
        trade_pnl: P&L from the trade
        portfolio_value: Current portfolio value

    Returns:
        float: Reward value
    """
    reward = 0.0

    # If in position, small holding penalty to encourage action
    if env.position != 0:
        reward -= 0.01  # Small penalty for holding

    # If trade closed, reward based on P&L
    if exit_reason in ['tp', 'sl', 'eod']:
        if trade_pnl > 0:
            # Reward winning trades
            reward += trade_pnl / 100.0  # Scale down for stability
        else:
            # Penalty for losing trades
            reward += trade_pnl / 100.0  # Negative value

    # Heavy penalty for Apex violations (trailing drawdown)
    if exit_reason == 'apex':
        reward -= 10.0  # Strong penalty to avoid violations

    # Small reward for staying flat (encouraging selectivity)
    if env.position == 0 and exit_reason == '':
        reward += 0.001  # Tiny reward for patience

    return reward
