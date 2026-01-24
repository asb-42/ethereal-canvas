"""
Pipeline Monitor and Watchdog Utility for Ethereal Canvas.

Provides real-time monitoring of pipeline progress and deadlock detection.
Used by both T2I and I2I backends for debugging and status reporting.
"""

import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from pathlib import Path

# Try to import psutil for process monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
    STATUS_ZOMBIE = psutil.STATUS_ZOMBIE
except ImportError:
    print("Warning: psutil not available, process monitoring disabled")
    PSUTIL_AVAILABLE = False
    psutil = None
    STATUS_ZOMBIE = None


class PipelineMonitor:
    """Monitor pipeline execution with real-time status updates and watchdog."""
    
    def __init__(self, task_name: str, log_callback: Optional[Callable] = None):
        self.task_name = task_name
        self.log_callback = log_callback
        self.start_time = None
        self.last_activity = None
        self.watchdog_active = False
        self.watchdog_thread = None
        self.process_pid = None
        self.steps_completed = 0
        self.total_steps = 0
        self.current_step = ""
        
    def _log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}"
        print(formatted_message)  # Always print to console
        
        if self.log_callback:
            self.log_callback(formatted_message)
    
    def start_monitoring(self, total_steps: int = 0):
        """Start pipeline monitoring and watchdog."""
        self.start_time = time.time()
        self.last_activity = time.time()
        self.total_steps = total_steps
        self.steps_completed = 0
        self.watchdog_active = True
        
        if PSUTIL_AVAILABLE:
            self.process_pid = psutil.Process().pid
        else:
            self.process_pid = None
        
        self._log(f"🚀 Starting {self.task_name} pipeline monitoring")
        self._log(f"📍 Process PID: {self.process_pid}")
        if total_steps > 0:
            self._log(f"📊 Total steps: {total_steps}")
        
        # Start watchdog thread
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        self._log("🐕 Watchdog started (10-second intervals)")
    
    def update_step(self, step_name: str, progress: Optional[int] = None):
        """Update current processing step."""
        self.current_step = step_name
        self.last_activity = time.time()
        
        if progress is not None:
            self.steps_completed = progress
            percentage = (progress / self.total_steps * 100) if self.total_steps > 0 else 0
            self._log(f"⚡ Step: {step_name} ({progress}/{self.total_steps}) - {percentage:.1f}%")
        else:
            self._log(f"⚡ Step: {step_name}")
    
    def update_progress(self, current: int, total: int, description: str = ""):
        """Update progress within current step."""
        self.last_activity = time.time()
        percentage = (current / total * 100) if total > 0 else 0
        desc = f" - {description}" if description else ""
        self._log(f"📈 Progress: {current}/{total} ({percentage:.1f}%){desc}")
    
    def complete_step(self, step_name: str):
        """Mark a step as completed."""
        self.last_activity = time.time()
        self._log(f"✅ Step completed: {step_name}")
    
    def error(self, error_message: str, exception: Optional[Exception] = None):
        """Log error and stop monitoring."""
        self._log(f"❌ Error in {self.task_name}: {error_message}", "ERROR")
        if exception:
            self._log(f"❌ Exception details: {str(exception)}", "ERROR")
        self.stop_monitoring()
    
    def success(self, result_path: Optional[str] = None):
        """Mark pipeline as successful."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._log(f"✅ {self.task_name} completed successfully in {elapsed:.1f}s")
        if result_path:
            self._log(f"💾 Output: {result_path}")
        self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop watchdog monitoring."""
        self.watchdog_active = False
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            self.watchdog_thread.join(timeout=2)
        self._log("🛑 Watchdog stopped")
    
    def _watchdog_loop(self):
        """Watchdog thread - checks every 10 seconds."""
        while self.watchdog_active:
            try:
                time.sleep(10)
                if not self.watchdog_active:
                    break
                
                current_time = time.time()
                time_since_activity = current_time - self.last_activity
                
                # Check if process is still alive
                process_active = False
                cpu_percent = 0.0
                
                if PSUTIL_AVAILABLE and self.process_pid:
                    try:
                        process = psutil.Process(self.process_pid)
                        process_active = process.is_running() and process.status() != STATUS_ZOMBIE
                        
                        # Check CPU usage (active if > 0.1%)
                        cpu_percent = process.cpu_percent(interval=0.1)
                        if cpu_percent > 0.1:
                            process_active = True
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        process_active = False
                
                if process_active and time_since_activity < 30:
                    self._log(f"🐕 Watchdog: Process active (CPU: {cpu_percent:.1f}%, Idle: {time_since_activity:.1f}s)")
                elif process_active:
                    self._log(f"⚠️ Watchdog: Process running but no activity for {time_since_activity:.1f}s (might be stuck)")
                else:
                    self._log(f"💀 Watchdog: Process died or stuck! No activity for {time_since_activity:.1f}s", "ERROR")
                    if self.last_activity:
                        last_time_str = datetime.fromtimestamp(self.last_activity).strftime('%H:%M:%S')
                        self._log(f"🕐 Last activity was: {last_time_str}")
                    break
                    
            except Exception as e:
                self._log(f"⚠️ Watchdog error: {e}", "ERROR")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type:
            self.error(f"Pipeline failed: {exc_val}", exc_val)
        else:
            self.success()
        return False  # Don't suppress exceptions


class UIStatusLogger:
    """Simple UI status logger for Gradio integration."""
    
    def __init__(self):
        self.status_messages = []
        self.max_messages = 100
    
    def log(self, message: str):
        """Add message to status log."""
        self.status_messages.append(message)
        
        # Keep only recent messages
        if len(self.status_messages) > self.max_messages:
            self.status_messages = self.status_messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 20) -> list:
        """Get recent status messages."""
        return self.status_messages[-count:]
    
    def clear(self):
        """Clear all status messages."""
        self.status_messages.clear()


# Global UI status logger instance
ui_status_logger = UIStatusLogger()


def create_monitor(task_name: str) -> PipelineMonitor:
    """Create a new pipeline monitor with UI logging."""
    return PipelineMonitor(task_name, log_callback=ui_status_logger.log)


def monitor_pipeline_step(monitor: PipelineMonitor, step_name: str, func: Callable, *args, **kwargs):
    """Monitor a specific pipeline step."""
    monitor.update_step(step_name)
    try:
        result = func(*args, **kwargs)
        monitor.complete_step(step_name)
        return result
    except Exception as e:
        monitor.error(f"Step '{step_name}' failed", e)
        raise