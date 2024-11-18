from typing import Dict, List, Optional, Any
import psutil
import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from prometheus_client import Counter, Gauge, Histogram
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import torch
from performance_optimizer import MemoryAwareLRUCache, ResourceMonitor, memory_optimized

# Prometheus metrics
RESOURCE_USAGE = Gauge('resource_usage', 'Resource usage by type', ['resource_type'])
SCALING_EVENTS = Counter('scaling_events', 'Auto-scaling events', ['direction'])
TASK_QUEUE_SIZE = Gauge('task_queue_size', 'Number of tasks in queue', ['queue_type'])
PROCESSING_TIME = Histogram('task_processing_time', 'Task processing time')

@dataclass
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    io_usage: float
    network_usage: float
    timestamp: datetime

@dataclass
class ResourceThresholds:
    cpu_high: float = 80.0
    cpu_low: float = 20.0
    memory_high: float = 85.0
    memory_low: float = 30.0
    queue_high: int = 100
    queue_low: int = 10

class ResourcePool:
    def __init__(self, pool_size: int = 4):
        self.pool = ThreadPoolExecutor(max_workers=pool_size)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.queue = queue.PriorityQueue()
        self.pool_size = pool_size

    def submit_task(self, priority: int, task_id: str, func, *args, **kwargs):
        """Submit task to thread pool with priority."""
        self.queue.put((priority, task_id, (func, args, kwargs)))
        TASK_QUEUE_SIZE.labels(queue_type='main').set(self.queue.qsize())

    def get_active_task_count(self) -> int:
        """Get number of currently active tasks."""
        return len(self.active_tasks)

class ResourceManager:
    def __init__(self, initial_pool_size: int = 4):
        self.logger = logging.getLogger(__name__)
        self.thresholds = ResourceThresholds()
        self.resource_pool = ResourcePool(initial_pool_size)
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scale_time = datetime.min
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start task processing thread
        self.processing_thread = threading.Thread(target=self._process_task_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Initialize memory-aware caching system
        self.cache = MemoryAwareLRUCache(maxsize=1000, max_memory_percent=0.2)
        self.resource_monitor = ResourceMonitor()
        self.optimization_interval = 30  # seconds
        self.last_optimization = time.time()
        self.warning_threshold = 3    # Number of consecutive warnings before taking action
        self.warning_count = defaultdict(int)
        
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            io = psutil.disk_io_counters()
            network = psutil.net_io_counters()
            
            RESOURCE_USAGE.labels(resource_type='cpu').set(cpu)
            RESOURCE_USAGE.labels(resource_type='memory').set(memory)
            
            return ResourceMetrics(
                cpu_usage=cpu,
                memory_usage=memory,
                io_usage=io.read_bytes + io.write_bytes,
                network_usage=network.bytes_sent + network.bytes_recv,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")
            return None

    def _monitor_resources(self):
        """Continuous resource monitoring and scaling decisions."""
        while True:
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._cleanup_old_metrics()
                    
                    # Check if scaling is needed
                    if self._should_scale():
                        self._adjust_resources()
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(10)

    def _cleanup_old_metrics(self):
        """Remove metrics older than 1 hour."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]

    def _should_scale(self) -> bool:
        """Determine if scaling is needed based on metrics."""
        if not self.metrics_history:
            return False
            
        # Check cooldown period
        if (datetime.now() - self.last_scale_time).total_seconds() < self.scaling_cooldown:
            return False
            
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        queue_size = self.resource_pool.queue.qsize()
        
        return (
            avg_cpu > self.thresholds.cpu_high or
            avg_memory > self.thresholds.memory_high or
            queue_size > self.thresholds.queue_high or
            (avg_cpu < self.thresholds.cpu_low and 
             queue_size < self.thresholds.queue_low)
        )

    def _adjust_resources(self):
        """Adjust resource allocation based on current metrics."""
        try:
            recent_metrics = self.metrics_history[-5:]
            avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
            queue_size = self.resource_pool.queue.qsize()
            
            current_size = self.resource_pool.pool_size
            new_size = current_size
            
            # Scale up
            if (avg_cpu > self.thresholds.cpu_high or 
                queue_size > self.thresholds.queue_high):
                new_size = min(current_size * 2, 16)  # Max 16 workers
                SCALING_EVENTS.labels(direction='up').inc()
                
            # Scale down
            elif (avg_cpu < self.thresholds.cpu_low and 
                  queue_size < self.thresholds.queue_low):
                new_size = max(current_size // 2, 2)  # Min 2 workers
                SCALING_EVENTS.labels(direction='down').inc()
            
            if new_size != current_size:
                self._resize_pool(new_size)
                self.last_scale_time = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error adjusting resources: {str(e)}")

    def _resize_pool(self, new_size: int):
        """Resize the thread pool."""
        try:
            # Create new pool
            new_pool = ThreadPoolExecutor(max_workers=new_size)
            
            # Shutdown old pool gracefully
            self.resource_pool.pool.shutdown(wait=False)
            
            # Update pool
            self.resource_pool.pool = new_pool
            self.resource_pool.pool_size = new_size
            
            self.logger.info(f"Resized thread pool to {new_size} workers")
        except Exception as e:
            self.logger.error(f"Error resizing pool: {str(e)}")

    def _process_task_queue(self):
        """Process tasks from the priority queue."""
        while True:
            try:
                # Get task from queue
                priority, task_id, (func, args, kwargs) = self.resource_pool.queue.get()
                
                # Submit to thread pool
                start_time = time.time()
                future = self.resource_pool.pool.submit(func, *args, **kwargs)
                
                # Add completion callback
                future.add_done_callback(
                    lambda f: self._task_completed(task_id, start_time)
                )
                
                # Update metrics
                TASK_QUEUE_SIZE.labels(queue_type='main').set(
                    self.resource_pool.queue.qsize()
                )
                
            except Exception as e:
                self.logger.error(f"Error processing task queue: {str(e)}")
                time.sleep(1)

    def _task_completed(self, task_id: str, start_time: float):
        """Handle task completion and update metrics."""
        try:
            # Record processing time
            processing_time = time.time() - start_time
            PROCESSING_TIME.observe(processing_time)
            
            # Remove from active tasks
            self.resource_pool.active_tasks.pop(task_id, None)
            
        except Exception as e:
            self.logger.error(f"Error handling task completion: {str(e)}")

    def submit_task(self, priority: int, task_id: str, func, *args, **kwargs):
        """Submit a task for execution with priority."""
        try:
            self.resource_pool.submit_task(priority, task_id, func, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error submitting task: {str(e)}")
            raise

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status."""
        try:
            if not self.metrics_history:
                return {"status": "unknown"}
                
            recent_metrics = self.metrics_history[-1]
            return {
                "status": "healthy",
                "metrics": {
                    "cpu_usage": recent_metrics.cpu_usage,
                    "memory_usage": recent_metrics.memory_usage,
                    "io_usage": recent_metrics.io_usage,
                    "network_usage": recent_metrics.network_usage
                },
                "pool_size": self.resource_pool.pool_size,
                "active_tasks": self.resource_pool.get_active_task_count(),
                "queue_size": self.resource_pool.queue.qsize()
            }
        except Exception as e:
            self.logger.error(f"Error getting resource status: {str(e)}")
            return {"status": "error", "message": str(e)}

    @memory_optimized
    def check_resources(self) -> Dict[str, bool]:
        """Check if system resources are within acceptable thresholds."""
        metrics = self.resource_monitor.get_metrics()
        status = {}
        
        # Check memory usage
        status['memory_ok'] = metrics['memory_percent'] < self.thresholds.memory_high
        if not status['memory_ok']:
            self.warning_count['memory'] += 1
        else:
            self.warning_count['memory'] = 0
            
        # Check CPU usage
        status['cpu_ok'] = metrics['cpu_percent'] < self.thresholds.cpu_high
        if not status['cpu_ok']:
            self.warning_count['cpu'] += 1
        else:
            self.warning_count['cpu'] = 0
            
        # Check GPU if available
        if torch.cuda.is_available():
            gpu_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            status['gpu_ok'] = gpu_percent < 90.0
            if not status['gpu_ok']:
                self.warning_count['gpu'] += 1
            else:
                self.warning_count['gpu'] = 0
                
        return status
        
    def optimize_resources(self):
        """Optimize resource usage based on current metrics and warning counts."""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
            
        self.last_optimization = current_time
        status = self.check_resources()
        
        # Handle memory warnings
        if self.warning_count['memory'] >= self.warning_threshold:
            logger.warning("Critical memory usage detected. Initiating emergency cleanup...")
            self._emergency_memory_cleanup()
            
        # Handle CPU warnings
        if self.warning_count['cpu'] >= self.warning_threshold:
            logger.warning("High CPU usage detected. Optimizing processing...")
            self._optimize_cpu_usage()
            
        # Handle GPU warnings
        if torch.cuda.is_available() and self.warning_count['gpu'] >= self.warning_threshold:
            logger.warning("High GPU usage detected. Optimizing GPU memory...")
            self._optimize_gpu_usage()
            
    def _emergency_memory_cleanup(self):
        """Perform emergency memory cleanup when usage is critical."""
        # Clear cache
        self.cache.cache.clear()
        self.cache._order.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Emergency memory cleanup completed")
        
    def _optimize_cpu_usage(self):
        """Optimize CPU usage when consistently high."""
        # Implement CPU optimization strategies
        # For example, adjusting thread pool sizes or deferring non-critical tasks
        pass
        
    def _optimize_gpu_usage(self):
        """Optimize GPU memory usage when consistently high."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Additional GPU optimization strategies can be implemented here
            
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get detailed resource usage metrics with historical data."""
        return self.resource_monitor.get_metrics()
        
    def cache_data(self, key: str, value: Any):
        """Cache data with memory-aware storage."""
        self.cache.put(key, value)
        
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data."""
        return self.cache.get(key)
        
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.cache.clear()
        self.cache._order.clear()
        logger.info("Cache cleared")
