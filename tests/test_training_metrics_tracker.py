"""
Unit tests for TrainingMetricsTracker
Tests P&L tracking, drawdown calculation, and compliance monitoring
"""
import pytest
import numpy as np
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jax_migration.training_metrics_tracker import TrainingMetricsTracker


class TestTrainingMetricsTracker:
    """Test suite for TrainingMetricsTracker"""
    
    def test_initialization(self):
        """Test tracker initialization with default values"""
        tracker = TrainingMetricsTracker(market='ES', phase=1)
        
        assert tracker.market == 'ES'
        assert tracker.phase == 1
        assert tracker.initial_balance == 50000.0
        assert tracker.drawdown_limit == 2500.0
        assert tracker.total_pnl == 0.0
        assert tracker.total_trades == 0
        assert tracker.close_violations == 0
    
    def test_basic_trade_tracking(self):
        """Test basic P&L and win rate calculation"""
        tracker = TrainingMetricsTracker(market='NQ', phase=2, enable_tensorboard=False)
        
        # Simulate 3 winning trades
        for _ in range(3):
            tracker.update({
                'final_balance': 50100.0,
                'trade_pnl': 100.0,
                'position_closed': True,
                'forced_close': False,
                'episode_return': 100.0,
            }, timestep=1)
        
        # Simulate 1 losing trade
        tracker.update({
            'final_balance': 50050.0,
            'trade_pnl': -50.0,
            'position_closed': True,
            'forced_close': False,
            'episode_return': 50.0,
        }, timestep=2)
        
        metrics = tracker.get_metrics()
        
        assert metrics['total_trades'] == 4
        assert metrics['winning_trades'] == 3
        assert metrics['losing_trades'] == 1
        assert np.isclose(metrics['win_rate'], 0.75)
        assert metrics['close_violations'] == 0
        assert np.isclose(metrics['total_pnl'], 250.0)  # 100 + 100 + 100 - 50
    
    def test_profit_factor_calculation(self):
        """Test profit factor calculation"""
        tracker = TrainingMetricsTracker(market='ES', phase=1, enable_tensorboard=False)
        
        # $200 gross profit
        tracker.update({
            'trade_pnl': 200.0,
            'position_closed': True,
            'forced_close': False,
            'final_balance': 50200.0,
            'episode_return': 200.0,
        }, timestep=1)
        
        # -$100 gross loss
        tracker.update({
            'trade_pnl': -100.0,
            'position_closed': True,
            'forced_close': False,
            'final_balance': 50100.0,
            'episode_return': 100.0,
        }, timestep=2)
        
        metrics = tracker.get_metrics()
        assert np.isclose(metrics['profit_factor'], 2.0)  # 200 / 100
        assert np.isclose(metrics['total_pnl'], 100.0)
    
    def test_drawdown_tracking(self):
        """Test maximum trailing drawdown calculation"""
        tracker = TrainingMetricsTracker(market='NQ', phase=2, enable_tensorboard=False)
        
        # Peak at 51000
        tracker.update({
            'final_balance': 51000.0,
            'position_closed': False,
            'forced_close': False,
            'episode_return': 1000.0,
        }, timestep=1)
        
        # Drop to 48500 (DD = $2,500)
        tracker.update({
            'final_balance': 48500.0,
            'position_closed': False,
            'forced_close': False,
            'episode_return': -1500.0,
        }, timestep=2)
        
        metrics = tracker.get_metrics()
        assert np.isclose(metrics['max_trailing_drawdown'], 2500.0)
        # Apex compliant checks if DD < limit (not strict equality)
        assert not metrics['apex_compliant']  # DD = limit triggers non-compliance
    
    def test_compliance_violations(self):
        """Test 4:59 PM close violation tracking"""
        tracker = TrainingMetricsTracker(market='ES', phase=2, enable_tensorboard=False)
        
        # Normal close
        tracker.update({
            'final_balance': 50000.0,
            'position_closed': True,
            'forced_close': False,
            'episode_return': 0.0,
        }, timestep=1)
        
        # Forced close (4:59 PM violation)
        tracker.update({
            'final_balance': 50000.0,
            'position_closed': True,
            'forced_close': True,
            'episode_return': 0.0,
        }, timestep=2)
        
        metrics = tracker.get_metrics()
        assert metrics['close_violations'] == 1
        assert not metrics['apex_compliant']  # Has violations
    
    def test_json_export(self):
        """Test JSON export functionality"""
        tracker = TrainingMetricsTracker(market='NQ', phase=1, enable_tensorboard=False)
        
        # Add some data
        tracker.update({
            'final_balance': 50100.0,
            'trade_pnl': 100.0,
            'position_closed': True,
            'forced_close': False,
            'episode_return': 100.0,
        }, timestep=1)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            tracker.save_to_json(temp_path, include_history=True)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data['market'] == 'NQ'
            assert data['phase'] == 1
            assert data['total_trades'] == 1
            assert 'episode_history' in data
            assert len(data['episode_history']['returns']) == 1
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_sharpe_calculation(self):
        """Test Sharpe ratio calculation"""
        tracker = TrainingMetricsTracker(market='ES', phase=1, enable_tensorboard=False)
        
        # Add multiple episodes with varying returns
        returns = [100, 120, 90, 110, 95]
        for ret in returns:
            tracker.update({
                'final_balance': 50000 + ret,
                'episode_return': ret,
                'position_closed': False,
                'forced_close': False,
            }, timestep=1)
        
        metrics = tracker.get_metrics()
        
        # Sharpe = mean / std
        expected_mean = np.mean(returns)
        expected_std = np.std(returns)
        expected_sharpe = expected_mean / expected_std if expected_std > 0 else 0
        
        assert np.isclose(metrics['sharpe_ratio'], expected_sharpe, rtol=0.01)
    
    def test_reset_functionality(self):
        """Test reset clears all metrics"""
        tracker = TrainingMetricsTracker(market='NQ', phase=2, enable_tensorboard=False)
        
        # Add some data
        tracker.update({
            'final_balance': 50100.0,
            'trade_pnl': 100.0,
            'position_closed': True,
            'forced_close': False,
            'episode_return': 100.0,
        }, timestep=1)
        
        # Reset
        tracker.reset()
        
        metrics = tracker.get_metrics()
        assert metrics['total_trades'] == 0
        assert metrics['total_pnl'] == 0.0
        assert metrics['max_trailing_drawdown'] == 0.0
    
    def test_apex_compliance_check(self):
        """Test Apex compliance status calculation"""
        tracker = TrainingMetricsTracker(
            market='ES',
            phase=2,
            drawdown_limit=2500.0,
            enable_tensorboard=False
        )
        
        # Compliant scenario
        tracker.update({
            'final_balance': 50100.0,
            'episode_return': 100.0,
            'position_closed': False,
            'forced_close': False,
        }, timestep=1)
        
        metrics = tracker.get_metrics()
        assert metrics['apex_compliant'] == True
        
        # Add drawdown violation
        tracker.update({
            'final_balance': 47000.0, #  $3,100 DD from peak
            'episode_return': -3100.0,
            'position_closed': False,
            'forced_close': False,
        }, timestep=2)
        
        metrics = tracker.get_metrics()
        assert metrics['apex_compliant'] == False  # DD > $2,500
    
    def test_average_win_loss(self):
        """Test average win and average loss calculations"""
        tracker = TrainingMetricsTracker(market='NQ', phase=1, enable_tensorboard=False)
        
        # Add 2 wins and 1 loss
        tracker.update({'trade_pnl': 150.0, 'position_closed': True, 'forced_close': False, 'final_balance': 50150, 'episode_return': 150}, timestep=1)
        tracker.update({'trade_pnl': 250.0, 'position_closed': True, 'forced_close': False, 'final_balance': 50400, 'episode_return': 400}, timestep=2)
        tracker.update({'trade_pnl': -100.0, 'position_closed': True, 'forced_close': False, 'final_balance': 50300, 'episode_return': 300}, timestep=3)
        
        metrics = tracker.get_metrics()
        
        # Avg win = (150 + 250) / 2 = 200
        # Avg loss = 100 / 1 = 100
        assert np.isclose(metrics['avg_win'], 200.0)
        assert np.isclose(metrics['avg_loss'], 100.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
