## Dashboard Metrics Schema (Phase 1/2)

This schema describes the JSON served at `/api/metrics` (and SSE `/api/stream`).
The frontend expects these fields; missing fields should be given sensible defaults to keep the UI stable.

### Top-level fields
- `status`: string, e.g. `TRAINING` | `IDLE`
- `market`: string symbol
- `phase`: int (1 or 2)
- `sps`: float, steps per second
- `timesteps`: object `{ current: int, target: int | null }`
- `mean_return` / `avg_episode_return`: float
- `win_rate`: float (0-1)
- `entropy`: float
- `kl_divergence`: float
- `learning_rate`: float
- `drawdown_limit`: float (e.g. 2500)
- `max_trailing_drawdown`: float
- `apex_distance`: float (`drawdown_limit - max_trailing_drawdown`)
- `total_pnl`: float
- `current_balance`: float
- `profit_factor`: float
- `sharpe_ratio`: float
- `avg_win`: float
- `avg_loss`: float
- `largest_win`: float (optional)
- `largest_loss`: float (optional)
- `daily_var`: float (optional)
- `total_trades`: int
- `winning_trades`: int
- `losing_trades`: int
- `close_violations`: int
- `dd_violations`: int
- `apex_compliant`: bool
- `last_trade`: object `{ pnl: float, direction: string, bars_held: int }`
- `last_updated`: ISO string or epoch seconds (file mtime added if missing)

### Series fields (bounded ~200 points)
- `episode_history`: `{ returns: float[], balances: float[], trade_counts: int[] }`
- `win_rate_series`: `[{ episode: int, win_rate: float }]`
- `policy_health_series`: `[{ step: int, entropy: float, kl: float }]`
- `network_stability_series`: `[{ step: int, policy_loss: float, value_loss: float, kl_div: float }]`
- `action_distribution`: `[{ episode: int, buy: float, sell: float, hold: float, pm: float }]`

### File locations
- Phase 1 JAX: `models/phase1_jax/training_metrics_{market}.json`
- Phase 2: `results/{market}_jax_phase2_realtime_metrics.json`

### Notes
- The backend adds `last_updated` from file mtime if not present.
- The frontend now prefers SSE at `/api/stream` and falls back to polling `/api/metrics` every 2s.
- Keep series trimmed to avoid unbounded JSON growth (history_limit = 200 by default).
