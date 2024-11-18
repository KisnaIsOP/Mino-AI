import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import torch
from functools import lru_cache, wraps

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Track performance metrics for system optimization."""
    execution_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    memory_usage: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gpu_usage: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    cache_hits: int = 0
    cache_misses: int = 0
    total_requests: int = 0
    
    def measure_time(self, operation_name: str):
        """Context manager to measure execution time."""
        class Timer:
            def __init__(self, metrics, operation):
                self.metrics = metrics
                self.operation = operation
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                execution_time = time.time() - self.start_time
                self.metrics.execution_times[self.operation].append(execution_time)
                self.metrics.update_resource_usage(self.operation)
        
        return Timer(self, operation_name)
    
    def update_resource_usage(self, operation: str):
        """Update resource usage metrics."""
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage[operation].append(memory_info.rss / 1024 / 1024)  # MB
        
        # GPU usage if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            self.gpu_usage[operation].append(gpu_memory)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}
        
        for operation, times in self.execution_times.items():
            if times:
                metrics[f"{operation}_avg_time"] = sum(times) / len(times)
                metrics[f"{operation}_max_time"] = max(times)
        
        for operation, usage in self.memory_usage.items():
            if usage:
                metrics[f"{operation}_avg_memory"] = sum(usage) / len(usage)
                metrics[f"{operation}_max_memory"] = max(usage)
        
        if self.gpu_usage:
            for operation, usage in self.gpu_usage.items():
                if usage:
                    metrics[f"{operation}_avg_gpu"] = sum(usage) / len(usage)
                    metrics[f"{operation}_max_gpu"] = max(usage)
        
        if self.total_requests > 0:
            metrics["cache_hit_ratio"] = self.cache_hits / self.total_requests
        
        return metrics
    
    def reset(self):
        """Reset all metrics."""
        self.execution_times.clear()
        self.memory_usage.clear()
        self.gpu_usage.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0


class LRUCache:
    """Least Recently Used Cache implementation with size limit."""
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.maxsize = maxsize
        self.metrics = PerformanceMetrics()
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        self.metrics.total_requests += 1
        if key in self.cache:
            self.metrics.cache_hits += 1
            return self.cache[key]
        self.metrics.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with LRU eviction."""
        if len(self.cache) >= self.maxsize:
            # Remove least recently used item
            lru_key = next(iter(self.cache))
            del self.cache[lru_key]
        self.cache[key] = value


class MemoryAwareLRUCache:
    """Advanced LRU Cache with memory monitoring and adaptive size adjustment."""
    def __init__(self, maxsize: int = 1000, max_memory_percent: float = 0.2):
        self.cache = {}
        self.maxsize = maxsize
        self.max_memory_percent = max_memory_percent
        self.metrics = PerformanceMetrics()
        self._order = []
        
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        return memory_percent <= self.max_memory_percent * 100
    
    def _adjust_cache_size(self):
        """Dynamically adjust cache size based on memory usage."""
        while self._order and not self._check_memory_usage():
            oldest_key = self._order.pop(0)
            self.cache.pop(oldest_key, None)
            
    def get(self, key: str) -> Any:
        """Get item from cache with memory-aware retrieval."""
        self.metrics.total_requests += 1
        if key in self.cache:
            self._order.remove(key)
            self._order.append(key)
            self.metrics.cache_hits += 1
            return self.cache[key]
        self.metrics.cache_misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache with memory monitoring."""
        if key in self.cache:
            self._order.remove(key)
        elif len(self.cache) >= self.maxsize:
            oldest_key = self._order.pop(0)
            self.cache.pop(oldest_key, None)
            
        self.cache[key] = value
        self._order.append(key)
        self._adjust_cache_size()


class ResourceOptimizer:
    """Optimize system resources and manage performance."""
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.cache = MemoryAwareLRUCache()
        self.resource_monitor = ResourceMonitor()
        self.optimization_interval = 60  # seconds
        self.last_optimization = time.time()
        
    def optimize_resources(self):
        """Perform resource optimization."""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
            
        self.last_optimization = current_time
        metrics = self.resource_monitor.get_metrics()
        
        # Adjust cache size based on memory pressure
        if metrics['memory_percent'] > 80:
            self.cache.maxsize = max(100, self.cache.maxsize // 2)
        elif metrics['memory_percent'] < 50:
            self.cache.maxsize = min(10000, self.cache.maxsize * 2)
            
        # Clear cache if memory usage is critical
        if metrics['memory_percent'] > 90:
            self.cache.cache.clear()
            self.cache._order.clear()


class ResourceMonitor:
    """Monitor system resources and provide metrics."""
    def __init__(self):
        self.history_size = 60  # Keep 1 minute of history
        self.metrics_history = defaultdict(lambda: deque(maxlen=self.history_size))
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_percent = process.memory_percent()
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_mb': process.memory_info().rss / 1024 / 1024
        }
        
        # Update history
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
        # Add moving averages
        for key in list(metrics.keys()):
            history = list(self.metrics_history[key])
            if history:
                metrics[f'{key}_avg'] = sum(history) / len(history)
                
        return metrics


def memory_optimized(func):
    """Decorator for memory-optimized function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_memory = process.memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_memory = process.memory_info().rss
            memory_diff = end_memory - start_memory
            if memory_diff > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"High memory usage detected in {func.__name__}: {memory_diff / 1024 / 1024:.2f}MB")
                
    return wrapper


class ResourceManager:
    """Manage system resources and optimize performance."""
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.resource_thresholds = {
            'memory_threshold': 0.85,  # 85% of available memory
            'gpu_threshold': 0.90,     # 90% of GPU memory
            'cpu_threshold': 0.80      # 80% of CPU usage
        }
    
    def check_resources(self) -> Dict[str, bool]:
        """Check if system resources are within acceptable thresholds."""
        # Check memory usage
        memory = psutil.virtual_memory()
        memory_ok = memory.percent < (self.resource_thresholds['memory_threshold'] * 100)
        
        # Check CPU usage
        cpu_ok = psutil.cpu_percent() < (self.resource_thresholds['cpu_threshold'] * 100)
        
        # Check GPU if available
        gpu_ok = True
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_usage = (allocated + reserved) / total
            gpu_ok = gpu_usage < self.resource_thresholds['gpu_threshold']
        
        return {
            'memory_ok': memory_ok,
            'cpu_ok': cpu_ok,
            'gpu_ok': gpu_ok
        }
    
    def optimize_resources(self):
        """Optimize resource usage based on current metrics."""
        # Clear unused memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        import gc
        gc.collect()
        
        # Clear metrics older than 1 hour
        self.metrics.reset()


@lru_cache(maxsize=1000)
def cached_operation(func):
    """Decorator for caching operation results."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
