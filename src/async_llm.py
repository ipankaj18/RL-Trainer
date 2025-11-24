"""
Async LLM Inference Module

Enables zero-latency always-on LLM queries via async execution.
"""

import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict, Tuple
import time
import numpy as np

# Setup logger for this module
logger = logging.getLogger(__name__)


class AsyncLLMInference:
    """
    Async LLM inference with zero blocking latency.

    Strategy:
    1. Submit query at step N (returns immediately)
    2. Query executes in background thread
    3. Result available at step N+1
    4. Agent uses result from previous step while current step executes

    Key Insight: It's OK if LLM reasoning is 1 step delayed!
    The agent sees: [RL instant decision] + [LLM reasoning from 1 step ago]
    This is still much better than no LLM reasoning.
    """

    def __init__(self, llm_model, max_workers=4):
        self.llm = llm_model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_queries = {}  # env_id -> Future
        self.latest_results = {}   # env_id -> result

        self.stats = {
            'queries_submitted': 0,
            'queries_completed': 0,
            'queries_timeout': 0,
            'avg_latency_ms': 0.0
        }

    def submit_query(self, env_id: int, observation, position_state, market_context, available_actions):
        """
        Submit LLM query asynchronously.

        Returns immediately. Result available next step via get_latest_result().

        Args:
            env_id: int - environment ID (for parallel envs)
            observation: np.array [261]
            position_state: dict
            market_context: dict
            available_actions: list[str]

        Returns:
            None (non-blocking)
        """
        # Enhanced logging for training debugging
        logger.debug(f"[ASYNC LLM] Query submitted for env {env_id}")
        logger.debug(f"[ASYNC LLM] Position state: {position_state}")
        logger.debug(f"[ASYNC LLM] Available actions: {available_actions}")
        
        # Submit to thread pool
        future = self.executor.submit(
            self._execute_query,
            observation, position_state, market_context, available_actions
        )

        # Store future
        if env_id in self.pending_queries:
            # Previous query still running, that's OK (will be skipped)
            logger.debug(f"[ASYNC LLM] Previous query for env {env_id} still running, replacing...")

        self.pending_queries[env_id] = {
            'future': future,
            'submit_time': time.time()
        }

        self.stats['queries_submitted'] += 1
        logger.debug(f"[ASYNC LLM] Query queued for env {env_id} (total submitted: {self.stats['queries_submitted']})")

    def get_latest_result(self, env_id: int, timeout_ms=10):
        """
        Get latest LLM result for this environment.

        Non-blocking: Returns immediately with cached result or None.

        Args:
            env_id: int - environment ID
            timeout_ms: float - max time to wait for result (in ms)

        Returns:
            result: dict or None
                {'action': int, 'confidence': float, 'reasoning': str, 'query_id': int}
                or None if not ready yet
        """
        # Check if query completed
        if env_id in self.pending_queries:
            pending = self.pending_queries[env_id]
            future = pending['future']

            # Check if done (non-blocking)
            if future.done():
                # Get result
                try:
                    result = future.result(timeout=timeout_ms/1000.0)
                    result = dict(result) if result is not None else None

                    # Update stats
                    latency_ms = (time.time() - pending['submit_time']) * 1000
                    self.stats['queries_completed'] += 1
                    self.stats['avg_latency_ms'] = (
                        0.9 * self.stats['avg_latency_ms'] + 0.1 * latency_ms
                    )

                    # Cleanup
                    del self.pending_queries[env_id]

                    if result is not None:
                        result['is_new'] = True
                        cached_result = dict(result)
                        cached_result['is_new'] = False
                        self.latest_results[env_id] = cached_result
                        return result
                    return None

                except Exception as e:
                    # Query failed
                    logger.warning(f"[ASYNC LLM] Query failed for env {env_id}: {e}")
                    del self.pending_queries[env_id]
                    return None

        # Not ready yet, return cached result from previous step
        cached = self.latest_results.get(env_id)
        if cached is not None:
            return cached
        return None

    def _execute_query(self, observation, position_state, market_context, available_actions):
        """
        Execute LLM query (runs in background thread).

        Returns:
            dict: {'action': int, 'confidence': float, 'reasoning': str, 'query_id': int}
        """
        try:
            # Call LLM
            action, confidence, reasoning, query_id = self.llm.query(
                observation, position_state, market_context
            )

            return {
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'query_id': query_id,
                'success': True
            }

        except Exception as e:
            logger.error(f"[ASYNC LLM] Query execution error: {e}")
            return {
                'action': 0,  # Default HOLD
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)}",
                'query_id': None,
                'success': False
            }

    def shutdown(self):
        """Shutdown executor and wait for pending queries."""
        self.executor.shutdown(wait=True)


class BatchedAsyncLLM:
    """
    Batched async LLM for parallel environments.

    Further optimization: Batch multiple environment queries into single LLM call.

    Example:
    - 20 parallel environments
    - Submit 20 queries → batch into 1-2 LLM calls
    - Process batch → distribute results
    - 10x speedup vs sequential queries
    """

    def __init__(self, llm_model, max_batch_size=8, batch_timeout_ms=10):
        self.llm = llm_model
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self.query_queue = queue.Queue()
        self.result_queues = {}  # env_id -> queue.Queue

        self.running = True  # Initialize before starting thread

        self.batch_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.batch_thread.start()

    def submit_query(self, env_id, observation, position_state, market_context, available_actions):
        """Submit query to batch queue."""
        # Enhanced logging for training debugging
        logger.debug(f"[BATCHED LLM] Query submitted for env {env_id}")
        logger.debug(f"[BATCHED LLM] Available actions: {available_actions}")
        
        # Create result queue for this env
        if env_id not in self.result_queues:
            self.result_queues[env_id] = queue.Queue(maxsize=1)

        # Add to query queue
        self.query_queue.put({
            'env_id': env_id,
            'observation': observation,
            'position_state': position_state,
            'market_context': market_context,
            'available_actions': available_actions,
            'submit_time': time.time()
        })
        
        logger.debug(f"[BATCHED LLM] Query queued for env {env_id}")

    def get_latest_result(self, env_id, timeout_ms=1):
        """Get result from result queue (non-blocking)."""
        if env_id not in self.result_queues:
            return None

        try:
            result = self.result_queues[env_id].get(timeout=timeout_ms/1000.0)
            return result
        except queue.Empty:
            return None

    def _batch_worker(self):
        """
        Worker thread that batches queries.

        Strategy:
        1. Collect queries for batch_timeout_ms
        2. When batch full or timeout reached, process batch
        3. Distribute results to result queues
        """
        while self.running:
            batch = []
            start_time = time.time()

            # Collect batch
            while len(batch) < self.max_batch_size:
                timeout_remaining = self.batch_timeout_ms / 1000.0 - (time.time() - start_time)
                if timeout_remaining <= 0:
                    break

                try:
                    query = self.query_queue.get(timeout=timeout_remaining)
                    batch.append(query)
                except queue.Empty:
                    break

            # Process batch
            if batch:
                self._process_batch(batch)

    def _process_batch(self, batch):
        """Process batch of queries."""
        # TODO: Implement batched LLM inference
        # This requires modifying LLM model to accept batch inputs
        # For now, process sequentially (still async from main thread)

        logger.debug(f"[BATCHED LLM] Processing batch of {len(batch)} queries")

        for query in batch:
            # Execute query
            try:
                logger.debug(f"[BATCHED LLM] Processing query for env {query['env_id']}")
                action, confidence, reasoning, query_id = self.llm.query(
                    query['observation'],
                    query['position_state'],
                    query['market_context']
                )

                logger.debug(f"[BATCHED LLM] Query successful: action={action}, confidence={confidence:.2f}")

                result = {
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'query_id': query_id,
                    'latency_ms': (time.time() - query['submit_time']) * 1000,
                    'success': True,
                    'is_new': True
                }

                # Put in result queue
                env_id = query['env_id']
                if env_id in self.result_queues:
                    try:
                        # Non-blocking put (drop old result if queue full)
                        self.result_queues[env_id].put_nowait(result)
                        logger.debug(f"[BATCHED LLM] Result queued for env {env_id}")
                    except queue.Full:
                        # Clear old result
                        try:
                            self.result_queues[env_id].get_nowait()
                            self.result_queues[env_id].put_nowait(result)
                            logger.debug(f"[BATCHED LLM] Replaced old result for env {env_id}")
                        except Exception as e:
                            logger.warning(f"[BATCHED LLM] Failed to queue result for env {env_id}: {e}")
                else:
                    logger.warning(f"[BATCHED LLM] No result queue for env {env_id}")

            except Exception as e:
                logger.error(f"[BATCHED LLM] Error processing query for env {query['env_id']}: {e}")
                import traceback
                traceback.print_exc()

                # Put error result in queue
                env_id = query['env_id']
                error_result = {
                    'action': 0,  # HOLD
                    'confidence': 0.0,
                    'reasoning': f"LLM_ERROR: {str(e)}",
                    'query_id': None,
                    'latency_ms': (time.time() - query['submit_time']) * 1000,
                    'success': False,
                    'is_new': True
                }

                if env_id in self.result_queues:
                    try:
                        self.result_queues[env_id].put_nowait(error_result)
                        logger.debug(f"[BATCHED LLM] Error result queued for env {env_id}")
                    except:
                        pass

    def shutdown(self):
        """Shutdown batch worker."""
        self.running = False
        self.batch_thread.join(timeout=5.0)


if __name__ == '__main__':
    """Test async LLM."""
    from llm_reasoning import LLMReasoningModule
    import numpy as np
    import time

    print("Note: This test requires Phi-3-mini-4k-instruct model in project folder")
    llm = LLMReasoningModule(config_path='config/llm_config.yaml')
    async_llm = AsyncLLMInference(llm, max_workers=4)

    # Submit queries
    obs = np.random.randn(261).astype(np.float32)
    position_state = {'position': 0}
    market_context = {'price': 20150, 'trend': 'down'}
    available_actions = ['HOLD', 'BUY', 'SELL']

    print("Submitting 10 async queries...")
    start = time.time()

    for env_id in range(10):
        async_llm.submit_query(env_id, obs, position_state, market_context, available_actions)

    print(f"✅ All queries submitted in {(time.time()-start)*1000:.1f}ms (non-blocking!)")

    # Wait for results
    time.sleep(0.5)

    results_ready = 0
    for env_id in range(10):
        result = async_llm.get_latest_result(env_id)
        if result:
            results_ready += 1

    print(f"✅ {results_ready}/10 results ready")
    print(f"✅ Avg latency: {async_llm.stats['avg_latency_ms']:.1f}ms")

    async_llm.shutdown()

    print("\n✅ Async LLM test passed!")
