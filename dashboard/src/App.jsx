import React, { useState, useEffect, useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    BarChart,
    Bar
} from 'recharts';
import {
    AlertTriangle,
    Brain,
    SignalHigh,
    Waves,
    TrendingUp,
    BarChart3
} from 'lucide-react';

// Mock data for offline or early renders. Real metrics from /api/metrics will override these fields.
const MOCK_METRICS = {
    status: 'TRAINING',
    sps: 1150,
    timesteps: { current: 5_200_000, target: 20_000_000 },
    mean_return: 450.23,
    win_rate: 0.523,
    entropy: 0.063,
    entropy_vital: 0.05,
    max_trailing_drawdown: 340,
    drawdown_limit: 2500,
    apex_distance: 2160,
    last_trade: { pnl: 55, direction: 'Long', bars_held: 4 },
    profit_factor: 1.45,
    sharpe_ratio: 1.2,
    avg_win: 120,
    avg_loss: -85,
    largest_win: 450,
    largest_loss: -300,
    daily_var: 210,
    close_violations: 0,
    dd_violations: 0,
    kl_divergence: 0.008,
    learning_rate: 0.0003,
    apex_compliant: true,
    episode_history: {
        returns: Array.from({ length: 80 }, (_, i) => 150 + i * 8 + Math.sin(i / 4) * 40),
        balances: Array.from({ length: 80 }, (_, i) => 12_000 + i * 45 + Math.sin(i / 6) * 150)
    }
};

const formatMoney = (val, digits = 0) => {
    if (val === undefined || val === null || Number.isNaN(val)) return '--';
    return `${val >= 0 ? '+' : ''}$${Math.abs(val).toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits
    })}`;
};

const formatPercent = (val) => {
    if (val === undefined || val === null || Number.isNaN(val)) return '--';
    return `${(val * 100).toFixed(1)}%`;
};

const formatCompact = (val) => {
    if (!val && val !== 0) return '--';
    return val >= 1_000_000 ? `${(val / 1_000_000).toFixed(1)}m` : `${(val / 1_000).toFixed(1)}k`;
};

function App() {
    const [metrics, setMetrics] = useState(null);
    const [error, setError] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(new Date());

    useEffect(() => {
        let es;
        let interval;

        const fetchMetrics = async () => {
            try {
                const response = await fetch('api/metrics');
                if (!response.ok) throw new Error('Failed to fetch metrics');
                const data = await response.json();
                setMetrics(data);
                setError(null);
                setLastUpdate(new Date());
            } catch (err) {
                setError(err.message);
            }
        };

        const startPolling = () => {
            fetchMetrics();
            interval = setInterval(fetchMetrics, 2000);
        };

        const startSSE = () => {
            try {
                es = new EventSource('api/stream');
                es.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        setMetrics(data);
                        setError(null);
                        setLastUpdate(new Date());
                    } catch (parseErr) {
                        setError('Failed to parse stream data');
                    }
                };
                es.onerror = () => {
                    setError('Streaming connection lost, falling back to polling.');
                    es.close();
                    startPolling();
                };
            } catch (err) {
                setError('Streaming not available, using polling.');
                startPolling();
            }
        };

        startSSE();

        return () => {
            if (es) es.close();
            if (interval) clearInterval(interval);
        };
    }, []);

    const data = useMemo(() => ({ ...MOCK_METRICS, ...(metrics || {}) }), [metrics]);

    const drawdownLimit = data.drawdown_limit ?? 2500;
    const ddCurrent = data.max_trailing_drawdown ?? 0;
    const drawdownRatio = Math.min(ddCurrent / drawdownLimit, 1);
    const apexDistance = data.apex_distance ?? Math.max(drawdownLimit - ddCurrent, 0);

    // Profitability chart data
    const profitSeries = useMemo(() => {
        const returns = data.episode_history?.returns || [];
        const balances = data.episode_history?.balances || [];
        return returns.map((ret, i) => ({
            episode: i,
            return: ret,
            balance: balances[i] ?? ret + 10_000
        })).slice(-120);
    }, [data.episode_history]);

    // Trading consistency series (win rate % over episodes)
    const winRateSeries = useMemo(() => {
        if (data.win_rate_series) return data.win_rate_series;
        const base = data.win_rate ?? 0.5;
        return Array.from({ length: 40 }, (_, i) => ({
            episode: 4200 + i * 25,
            win_rate: Math.max(0.4, Math.min(0.6, base + Math.sin(i / 3) * 0.05 - Math.random() * 0.01))
        }));
    }, [data.win_rate, data.win_rate_series]);

    // Action distribution mock or provided
    const actionData = useMemo(() => {
        if (data.action_distribution) return data.action_distribution;
        return Array.from({ length: 28 }, (_, i) => {
            const buy = 3 + Math.random() * 4;
            const sell = 2 + Math.random() * 3;
            const hold = 1 + Math.random() * 2;
            const pm = 0.5 + Math.random() * 1.5;
            return {
                episode: 4200 + i * 40,
                buy,
                sell,
                hold,
                pm
            };
        });
    }, [data.action_distribution]);

    // Policy health: entropy + KL
    const policyHealth = useMemo(() => {
        if (data.policy_health_series) return data.policy_health_series;
        return Array.from({ length: 60 }, (_, i) => ({
            step: i * 90_000,
            entropy: Math.max(0.03, 0.2 - i * 0.0025 + Math.sin(i / 4) * 0.01),
            kl: Math.max(0, 0.08 - i * 0.001 + Math.sin(i / 3) * 0.008)
        }));
    }, [data.policy_health_series]);

    // KL divergence / losses panel
    const klPanel = useMemo(() => {
        if (data.network_stability_series) return data.network_stability_series;
        return Array.from({ length: 50 }, (_, i) => ({
            step: i * 110_000,
            policy_loss: -0.0015 - Math.sin(i / 5) * 0.0006,
            value_loss: 24 - i * 0.12 + Math.sin(i / 6) * 0.8,
            kl_div: 0.006 + Math.sin(i / 7) * 0.002
        }));
    }, [data.network_stability_series]);

    const topCards = [
        {
            title: 'MEAN RETURN',
            value: formatMoney(data.avg_episode_return ?? data.mean_return ?? data.total_pnl, 2),
            hint: '↗️ Rolling 100 ep'
        },
        {
            title: 'WIN RATE',
            value: formatPercent(data.win_rate ?? 0),
            hint: '>50% target'
        },
        {
            title: 'ENTROPY (EXPL)',
            value: `${(data.entropy ?? 0.05).toFixed(3)}`,
            hint: `>${data.entropy_vital ?? 0.05} vital`,
            tone: 'amber'
        },
        {
            title: 'CURR. DRAWDN',
            value: formatMoney(-(ddCurrent || 0)),
            hint: `Peak: $${(data.drawdown_peak ?? 12_000).toLocaleString()}`,
            tone: 'red'
        },
        {
            title: 'APEX DIST.',
            value: `$${apexDistance.toLocaleString()}`,
            hint: `To Limit: $${drawdownLimit.toLocaleString()}`,
            tone: 'amber'
        },
        {
            title: 'LAST TRADE',
            value: formatMoney(data.last_trade?.pnl ?? 55),
            hint: `${data.last_trade?.direction || 'Long'} | ${data.last_trade?.bars_held || 4} bars`,
            tone: 'green'
        }
    ];

    const statsStrip = [
        { label: 'Profit Factor', value: data.profit_factor?.toFixed(2) ?? '—' },
        { label: 'Avg Win', value: formatMoney(data.avg_win ?? 0) },
        { label: 'Avg Loss', value: formatMoney(data.avg_loss ?? 0) },
        { label: 'Sharpe Ratio', value: data.sharpe_ratio?.toFixed(2) ?? '—' },
        { label: 'Largest Win', value: formatMoney(data.largest_win ?? 0) },
        { label: 'Largest Loss', value: formatMoney(data.largest_loss ?? 0) },
        { label: 'Daily Var', value: formatMoney(data.daily_var ?? 0) }
    ];

    return (
        <div className="dashboard">
            <div className="scanlines" />

            <header className="top-bar">
                <div className="brand">
                    <div className="brand-dot" />
                    <div>
                        <div className="brand-title">RL TRADER v2 – CYBERPUNK CONTROL CENTER</div>
                        <div className="brand-sub">live telemetry & risk posture</div>
                    </div>
                </div>
                <div className="status-line">
                    <span className="status-pill">
                        Status: <span className="accent">{data.status || 'TRAINING'}</span>
                    </span>
                    <span className="muted">SPS</span> <strong>{data.sps || 0}</strong>
                    <span className="muted">Timesteps</span>
                    <strong>{formatCompact(data.timesteps?.current ?? 0)} / {formatCompact(data.timesteps?.target ?? 0)}</strong>
                    <span className="muted">Updated</span>
                    <strong>{lastUpdate.toLocaleTimeString()}</strong>
                </div>
            </header>

            <section className="stat-row">
                {topCards.map((card) => (
                    <SummaryCard key={card.title} {...card} />
                ))}
            </section>

            <section className="main-grid">
                <div className="panel large">
                    <PanelHeader title="Profitability Trend" subtitle="Rolling 100 Ep" icon={<TrendingUp size={16} />} />
                    <div className="chart-shell">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={profitSeries} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
                                <defs>
                                    <linearGradient id="profit" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#4ade80" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#4ade80" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2b2f3a" vertical={false} />
                                <XAxis dataKey="episode" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                <YAxis stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
                                <Area type="monotone" dataKey="return" stroke="#4ade80" strokeWidth={2.5} fill="url(#profit)" name="Mean Ep Return" />
                                <Line type="monotone" dataKey="balance" stroke="#67e8f9" strokeWidth={2} dot={false} name="Current Balance" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="panel stack">
                    <div className="panel block">
                        <PanelHeader title="Apex Risk Monitor" subtitle="Live drawdown to limit" icon={<AlertTriangle size={16} />} />
                        <div className="gauge-row">
                            <GaugeMeter ratio={drawdownRatio} />
                            <div className="gauge-meta">
                                <div className="gauge-label">Current Drawdown</div>
                                <div className="gauge-value">{formatMoney(-ddCurrent)}</div>
                                <div className="gauge-sub">Limit: ${drawdownLimit.toLocaleString()}</div>
                                <div className="gauge-sub">Remaining: ${apexDistance.toLocaleString()}</div>
                            </div>
                        </div>
                        <div className="violation-row">
                            <span>Violations</span>
                            <div className={`badge ${data.close_violations + data.dd_violations > 0 ? 'badge-red' : 'badge-green'}`}>
                                {data.close_violations + data.dd_violations}
                            </div>
                        </div>
                    </div>

                    <div className="panel block">
                        <PanelHeader title="Trading Consistency" subtitle="Rolling 100 Ep" icon={<SignalHigh size={16} />} />
                        <div className="chart-shell small">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={winRateSeries} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2b2f3a" vertical={false} />
                                    <XAxis dataKey="episode" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                    <YAxis domain={[0.38, 0.6]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                                    <Line type="monotone" dataKey="win_rate" stroke="#22d3ee" strokeWidth={2} dot={false} />
                                    <Line type="monotone" dataKey={() => 0.5} stroke="#f43f5e" strokeDasharray="5 5" dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </section>

            <section className="bottom-grid">
                <div className="panel">
                    <PanelHeader title="Policy Health" subtitle="Dual Axis" icon={<Brain size={16} />} />
                    <div className="chart-shell">
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedPolicyChart data={policyHealth} />
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="panel">
                    <PanelHeader title="Action Distribution & Total Trades" subtitle="Per 1000 steps" icon={<BarChart3 size={16} />} />
                    <div className="chart-shell">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={actionData} stackOffset="expand" margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2b2f3a" vertical={false} />
                                <XAxis dataKey="episode" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                <YAxis stroke="#a1a1aa" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                                <Bar dataKey="buy" stackId="a" fill="#34d399" name="Buy %" />
                                <Bar dataKey="sell" stackId="a" fill="#f87171" name="Sell %" />
                                <Bar dataKey="hold" stackId="a" fill="#a78bfa" name="Hold %" />
                                <Bar dataKey="pm" stackId="a" fill="#38bdf8" name="PM Actions %" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="panel">
                    <PanelHeader title="Network Stability" subtitle="KL divergence & losses" icon={<Waves size={16} />} />
                    <div className="kl-metrics">
                        <div className="kl-chip">
                            KL Divergence: <span className="accent">{(data.kl_divergence ?? 0.0).toFixed(3)}</span>
                        </div>
                        <div className="kl-chip">LR: {data.learning_rate ?? '—'}</div>
                    </div>
                    <div className="chart-shell">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={klPanel} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2b2f3a" vertical={false} />
                                <XAxis dataKey="step" stroke="#a1a1aa" tickFormatter={(v) => `${(v / 1_000_000).toFixed(1)}m`} tick={{ fontSize: 11 }} />
                                <YAxis yAxisId="left" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                <YAxis yAxisId="right" orientation="right" stroke="#a1a1aa" tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
                                <Line yAxisId="left" type="monotone" dataKey="kl_div" stroke="#22d3ee" strokeWidth={2} dot={false} name="KL Div" />
                                <Line yAxisId="right" type="monotone" dataKey="policy_loss" stroke="#f472b6" strokeWidth={1.5} dot={false} name="Policy Loss" />
                                <Line yAxisId="right" type="monotone" dataKey="value_loss" stroke="#67e8f9" strokeWidth={1.5} dot={false} name="Value Loss" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </section>

            <section className="stats-strip">
                <div className="tab-label">[ Trade Stats ]</div>
                <div className="stats-grid">
                    {statsStrip.map((item) => (
                        <div key={item.label} className="stat-chip">
                            <div className="chip-label">{item.label}</div>
                            <div className="chip-value">{item.value}</div>
                        </div>
                    ))}
                </div>
            </section>

            {error && (
                <div className="error-banner">
                    ⚠️ API connection issue: {error}
                </div>
            )}
        </div>
    );
}

function SummaryCard({ title, value, hint, tone = 'green' }) {
    return (
        <div className={`summary-card summary-${tone}`}>
            <div className="summary-title">{title}</div>
            <div className="summary-value">{value}</div>
            <div className="summary-hint">{hint}</div>
        </div>
    );
}

function PanelHeader({ title, subtitle, icon }) {
    return (
        <div className="panel-header">
            <div className="panel-title">
                {icon}
                <span>{title}</span>
            </div>
            <div className="panel-sub">{subtitle}</div>
        </div>
    );
}

function GaugeMeter({ ratio }) {
    const clamped = Math.min(Math.max(ratio, 0), 1);
    const deg = 180 * clamped;
    const hue = 120 - clamped * 120; // 120 (green) -> 0 (red)
    return (
        <div className="gauge">
            <div className="gauge-arc" />
            <div
                className="gauge-needle"
                style={{ transform: `rotate(${deg - 90}deg)`, background: `hsl(${hue}, 80%, 60%)` }}
            />
            <div className="gauge-center">Apex</div>
        </div>
    );
}

function ComposedPolicyChart({ data }) {
    return (
        <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2b2f3a" vertical={false} />
            <XAxis dataKey="step" stroke="#a1a1aa" tickFormatter={(v) => `${(v / 1_000_000).toFixed(1)}m`} tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" stroke="#a1a1aa" tick={{ fontSize: 11 }} domain={[0, 0.22]} />
            <YAxis yAxisId="right" orientation="right" stroke="#a1a1aa" tick={{ fontSize: 11 }} domain={[0, 0.18]} />
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155' }} />
            <Line yAxisId="left" type="monotone" dataKey="entropy" stroke="#a78bfa" strokeWidth={2} dot={false} name="Entropy" />
            <Line yAxisId="right" type="monotone" dataKey="kl" stroke="#f97316" strokeWidth={2} dot={false} name="Approx KL" />
        </LineChart>
    );
}

export default App;
