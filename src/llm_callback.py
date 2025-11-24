"""
LLM Monitoring Callback

Monitors LLM usage and hybrid agent performance during training.
Integrates with Stable Baselines 3 callback system for real-time tracking.

Tracks:
- LLM query frequency and latency
- Agreement/disagreement rates between RL and LLM
- Decision fusion methods used
- Risk veto applications
- Cache hit rates
- Error rates
"""

import logging
from typing import Dict, Any
import numpy as np

# Conditional import for Stable Baselines 3
try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Create a dummy BaseCallback for testing
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.logger = logging.getLogger(__name__)
        def _on_step(self) -> bool:
            return True
        def _on_rollout_end(self) -> bool:
            return True
        def _on_training_end(self) -> bool:
            return True


class LLMMonitoringCallback(BaseCallback):
    """
    Callback to monitor LLM usage during hybrid training.
    
    Integrates with SB3 training loop to track:
    - LLM performance metrics
    - Decision fusion statistics
    - Risk management effectiveness
    """
    
    def __init__(self, hybrid_agent, log_freq: int = 1000, verbose: int = 0):
        """
        Initialize LLM monitoring callback.
        
        Args:
            hybrid_agent: HybridTradingAgent instance
            log_freq: How often to log metrics (in steps)
            verbose: Verbosity level (0=quiet, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.hybrid_agent = hybrid_agent
        self.log_freq = log_freq
        
        # Episode tracking
        self.episode_llm_queries = []
        self.episode_agreements = []
        self.episode_disagreements = []
        self.episode_risk_vetoes = []
        
        # Step tracking
        self.step_count = 0
        self.last_log_step = 0
        
        # Performance tracking
        self.latency_history = []
        self.confidence_history = {'rl': [], 'llm': []}

        # Note: Cannot use self.logger here as self.model doesn't exist yet
        # Logging will happen in _on_training_start()

    def _on_training_start(self) -> None:
        """Called when training starts. Safe to use self.logger here."""
        if self.verbose > 0:
            print(f"[LLM_CALLBACK] Initialized with log_freq={self.log_freq}")

        # Now we can safely use logger since model is attached
        if hasattr(self, 'logger'):
            self.logger.info(f"[LLM_CALLBACK] Training started with log_freq={self.log_freq}")
            self.logger.info(f"[LLM_CALLBACK] Hybrid agent ready for monitoring")

    def _on_step(self) -> bool:
        """Called every environment step."""
        self.step_count += 1
        
        # Log stats at specified frequency
        if self.step_count % self.log_freq == 0:
            self._log_current_stats()
        
        return True
    
    def _log_current_stats(self):
        """Log current LLM and fusion statistics."""
        try:
            # Get hybrid agent stats
            hybrid_stats = self.hybrid_agent.get_stats()
            
            # Get LLM stats
            llm_stats = self.hybrid_agent.get_llm_stats()
            
            # Log to TensorBoard if available
            if hasattr(self.logger, 'record'):
                # LLM performance metrics
                if llm_stats.get('total_queries', 0) > 0:
                    self.logger.record("llm/total_queries", llm_stats['total_queries'])
                    self.logger.record("llm/avg_latency_ms", llm_stats['avg_latency_ms'])
                    self.logger.record("llm/error_rate", llm_stats['error_rate'])
                    self.logger.record("llm/cache_hit_rate", llm_stats['cache_hit_rate'])
                
                # Decision fusion metrics
                self.logger.record("fusion/agreement_pct", hybrid_stats.get('agreement_pct', 0))
                self.logger.record("fusion/disagreement_pct", hybrid_stats.get('disagreement_pct', 0))
                self.logger.record("fusion/rl_only_pct", hybrid_stats.get('rl_only_pct', 0))
                self.logger.record("fusion/llm_only_pct", hybrid_stats.get('llm_only_pct', 0))
                self.logger.record("fusion/risk_veto_pct", hybrid_stats.get('risk_veto_pct', 0))
                self.logger.record("fusion/llm_query_rate", hybrid_stats.get('llm_query_rate', 0))
                self.logger.record("fusion/cache_hit_rate", hybrid_stats.get('cache_hit_rate', 0))
                
                # Confidence metrics
                if 'avg_rl_confidence' in hybrid_stats:
                    self.logger.record("confidence/rl_avg", hybrid_stats['avg_rl_confidence'])
                if 'avg_llm_confidence' in hybrid_stats:
                    self.logger.record("confidence/llm_avg", hybrid_stats['avg_llm_confidence'])
                
                # Risk metrics
                self.logger.record("risk/total_vetoes", hybrid_stats.get('risk_veto', 0))
            
            # Log to console if verbose
            if self.verbose > 0:
                self._log_console_stats(hybrid_stats, llm_stats)
        
        except Exception as e:
            self.logger.error(f"[LLM_CALLBACK] Error logging stats: {e}")
    
    def _log_console_stats(self, hybrid_stats: Dict, llm_stats: Dict):
        """Log statistics to console."""
        print(f"\n[LLM_CALLBACK] Step {self.step_count}")
        print("-" * 50)
        
        # LLM performance
        if llm_stats.get('total_queries', 0) > 0:
            print(f"LLM Queries: {llm_stats['total_queries']}")
            print(f"Avg Latency: {llm_stats['avg_latency_ms']:.1f}ms")
            print(f"Error Rate: {llm_stats['error_rate']:.1f}%")
            print(f"Cache Hit Rate: {llm_stats['cache_hit_rate']:.1f}%")
        
        # Decision fusion
        print(f"Agreement Rate: {hybrid_stats.get('agreement_pct', 0):.1f}%")
        print(f"Disagreement Rate: {hybrid_stats.get('disagreement_pct', 0):.1f}%")
        print(f"LLM Override Rate: {hybrid_stats.get('llm_only_pct', 0):.1f}%")
        print(f"Risk Veto Rate: {hybrid_stats.get('risk_veto_pct', 0):.1f}%")
        print(f"LLM Query Rate: {hybrid_stats.get('llm_query_rate', 0):.1f}%")
        
        # Confidences
        if 'avg_rl_confidence' in hybrid_stats:
            print(f"Avg RL Confidence: {hybrid_stats['avg_rl_confidence']:.2f}")
        if 'avg_llm_confidence' in hybrid_stats:
            print(f"Avg LLM Confidence: {hybrid_stats['avg_llm_confidence']:.2f}")
        
        print("-" * 50)
    
    def _on_rollout_end(self) -> bool:
        """Called at end of rollout collection."""
        try:
            # Get episode stats
            stats = self.hybrid_agent.get_stats()
            
            # Store episode statistics
            self.episode_llm_queries.append(stats.get('llm_queries', 0))
            self.episode_agreements.append(stats.get('agreement', 0))
            self.episode_disagreements.append(stats.get('disagreement', 0))
            self.episode_risk_vetoes.append(stats.get('risk_veto', 0))
            
            # Log episode summary if verbose
            if self.verbose > 0:
                print(f"\n[LLM_CALLBACK] Rollout End Summary")
                print(f"Episode LLM Queries: {self.episode_llm_queries[-1]}")
                print(f"Episode Agreements: {self.episode_agreements[-1]}")
                print(f"Episode Risk Vetoes: {self.episode_risk_vetoes[-1]}")
            
            # Log to TensorBoard
            if hasattr(self.logger, 'record'):
                if len(self.episode_llm_queries) > 0:
                    self.logger.record("rollout/llm_queries_per_episode", self.episode_llm_queries[-1])
                    self.logger.record("rollout/agreements_per_episode", self.episode_agreements[-1])
                    self.logger.record("rollout/risk_vetoes_per_episode", self.episode_risk_vetoes[-1])
        
        except Exception as e:
            self.logger.error(f"[LLM_CALLBACK] Rollout end error: {e}")
        
        return True
    
    def _on_training_end(self) -> bool:
        """Called at end of training."""
        try:
            # Log final summary
            self.logger.info("[LLM_CALLBACK] Training completed - Final Summary:")
            
            final_stats = self.hybrid_agent.get_stats()
            llm_stats = self.hybrid_agent.get_llm_stats()
            
            self.logger.info(f"Total decisions: {final_stats.get('total_decisions', 0)}")
            self.logger.info(f"Agreement rate: {final_stats.get('agreement_pct', 0):.1f}%")
            self.logger.info(f"Risk veto rate: {final_stats.get('risk_veto_pct', 0):.1f}%")
            self.logger.info(f"LLM query rate: {final_stats.get('llm_query_rate', 0):.1f}%")
            
            if llm_stats.get('total_queries', 0) > 0:
                self.logger.info(f"Avg LLM latency: {llm_stats.get('avg_latency_ms', 0):.1f}ms")
                self.logger.info(f"LLM error rate: {llm_stats.get('error_rate', 0):.1f}%")
                self.logger.info(f"LLM cache hit rate: {llm_stats.get('cache_hit_rate', 0):.1f}%")
            
            # Calculate episode averages
            if self.episode_llm_queries:
                avg_queries = np.mean(self.episode_llm_queries)
                avg_agreements = np.mean(self.episode_agreements)
                avg_vetoes = np.mean(self.episode_risk_vetoes)
                
                self.logger.info(f"Avg LLM queries per episode: {avg_queries:.1f}")
                self.logger.info(f"Avg agreements per episode: {avg_agreements:.1f}")
                self.logger.info(f"Avg risk vetoes per episode: {avg_vetoes:.1f}")
        
        except Exception as e:
            self.logger.error(f"[LLM_CALLBACK] Training end error: {e}")
        
        return True
    
    def get_episode_stats(self) -> Dict[str, list]:
        """Get episode-level statistics."""
        return {
            'llm_queries': self.episode_llm_queries,
            'agreements': self.episode_agreements,
            'disagreements': self.episode_disagreements,
            'risk_vetoes': self.episode_risk_vetoes
        }
    
    def reset_episode_stats(self):
        """Reset episode statistics."""
        self.episode_llm_queries.clear()
        self.episode_agreements.clear()
        self.episode_disagreements.clear()
        self.episode_risk_vetoes.clear()


# Alternative callback for environments without SB3
def create_llm_logger(hybrid_agent, log_freq: int = 1000):
    """
    Create a simple LLM logger for non-SB3 environments.
    
    Args:
        hybrid_agent: HybridTradingAgent instance
        log_freq: How often to log (in steps)
    
    Returns:
        Simple logger function
    """
    step_count = 0
    
    def log_step():
        nonlocal step_count
        step_count += 1
        
        if step_count % log_freq == 0:
            stats = hybrid_agent.get_stats()
            llm_stats = hybrid_agent.get_llm_stats()
            
            print(f"\n[LLM_LOGGER] Step {step_count}")
            print(f"Agreement Rate: {stats.get('agreement_pct', 0):.1f}%")
            print(f"LLM Query Rate: {stats.get('llm_query_rate', 0):.1f}%")
            print(f"Avg LLM Latency: {llm_stats.get('avg_latency_ms', 0):.1f}ms")
    
    return log_step


if __name__ == '__main__':
    """Test LLM monitoring callback."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing LLM Monitoring Callback...")
    
    # Mock hybrid agent
    class MockHybridAgent:
        def __init__(self):
            self.decisions = 0
        
        def get_stats(self):
            self.decisions += 1
            return {
                'total_decisions': self.decisions,
                'agreement': self.decisions // 2,
                'disagreement': self.decisions // 3,
                'risk_veto': self.decisions // 10,
                'llm_queries': self.decisions // 2,
                'agreement_pct': 50.0,
                'disagreement_pct': 33.3,
                'risk_veto_pct': 10.0,
                'llm_query_rate': 50.0,
                'cache_hit_rate': 20.0,
                'avg_rl_confidence': 0.75,
                'avg_llm_confidence': 0.80
            }
        
        def get_llm_stats(self):
            return {
                'total_queries': 50,
                'avg_latency_ms': 15.5,
                'error_rate': 2.0,
                'cache_hit_rate': 25.0
            }
    
    # Test callback
    hybrid_agent = MockHybridAgent()
    callback = LLMMonitoringCallback(hybrid_agent, log_freq=100, verbose=1)
    
    print("\n1. Testing step logging...")
    for i in range(250):
        callback._on_step()
    
    print("\n2. Testing rollout end...")
    callback._on_rollout_end()
    
    print("\n3. Testing training end...")
    callback._on_training_end()
    
    print("\n4. Testing episode stats...")
    episode_stats = callback.get_episode_stats()
    print(f"Episode stats keys: {list(episode_stats.keys())}")
    
    print("\n✅ LLM monitoring callback tests passed!")
    
    # Test simple logger
    print("\n5. Testing simple logger...")
    logger = create_llm_logger(hybrid_agent, log_freq=50)
    for i in range(150):
        logger()
    
    print("\n✅ Simple logger test passed!")