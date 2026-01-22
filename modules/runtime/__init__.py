"""
Runtime utilities for Ethereal Canvas.

Provides centralized utilities for logging, monitoring, and operations.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .paths import (
    RUNTIME_ROOT, LOGS_DIR, OUTPUTS_DIR, CACHE_DIR, TMP_DIR,
    timestamp, output_image_path, output_edit_path, output_inpaint_path,
    log_file_path, cleanup_temp_files, get_cache_usage
)


# -------------------------------------------------
# Logging utilities
# -------------------------------------------------

class RuntimeLogger:
    """Centralized runtime logging with structured output."""
    
    def __init__(self, name: str = "app"):
        self.name = name
        self.log_file = None
        self.start_time = time.time()
    
    def start_session(self, operation: str) -> None:
        """Start a new logging session."""
        self.start_time = time.time()
        self.log_file = log_file_path(f"{self.name}_{timestamp()}")
        
        try:
            with open(self.log_file, "a") as f:
                f.write(f"=== SESSION START: {operation} ===\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
                f.write(f"Process ID: {os.getpid()}\n")
                f.write(f"Python: {sys.version}\n")
                f.write("=" * 50 + "\n")
                f.flush()
        except Exception as e:
            print(f"Failed to start logging session: {e}")
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message with specified level."""
        if not self.log_file:
            return
        
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        level_upper = level.upper()
        
        try:
            with open(self.log_file, "a") as f:
                f.write(f"[{timestamp}] {level_upper}: {message}\n")
                
                # Add structured data if provided
                if kwargs:
                    for key, value in kwargs.items():
                        f.write(f"  {key}: {value}\n")
                
                f.flush()
        except Exception as e:
            print(f"Failed to log message: {e}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message."""
        self.log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message."""
        self.log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level message."""
        self.log("ERROR", message, **kwargs)
    
    def success(self, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log successful operation completion."""
        duration = time.time() - self.start_time
        
        self.info(f"✓ {operation} completed in {duration:.2f}s")
        
        if details:
            self.info("Operation details:")
            for key, value in details.items():
                self.info(f"  {key}: {value}")
    
    def end_session(self) -> None:
        """End current logging session."""
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"\n=== SESSION END ===\n")
                    f.write(f"Duration: {time.time() - self.start_time:.2f}s\n")
                    f.write("=" * 50 + "\n")
                    f.flush()
                self.log_file = None
            except Exception as e:
                print(f"Failed to end logging session: {e}")


# -------------------------------------------------
# System monitoring utilities
# -------------------------------------------------

class SystemMonitor:
    """Monitor system resources and performance."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": round(memory.percent, 1)
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    @staticmethod
    def get_disk_usage(path: Path = Path(".")) -> Dict[str, Any]:
        """Get disk usage statistics for given path."""
        try:
            import psutil
            disk = psutil.disk_usage(str(path))
            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round(disk.percent, 1)
            }
        except ImportError:
            return {"error": "psutil not available"}
    
    @staticmethod
    def get_process_info() -> Dict[str, Any]:
        """Get current process information."""
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "executable": sys.executable,
            "working_directory": os.getcwd(),
            "environment": dict(os.environ),
        }


# -------------------------------------------------
# Operation tracking utilities
# -------------------------------------------------

class OperationTracker:
    """Track and report on long-running operations."""
    
    def __init__(self):
        self.operations = {}
    
    def start_operation(self, operation_id: str, description: str) -> None:
        """Start tracking a long-running operation."""
        self.operations[operation_id] = {
            "description": description,
            "start_time": time.time(),
            "status": "running"
        }
        print(f"🔄 Started: {description}")
    
    def complete_operation(self, operation_id: str, result: Any = None) -> None:
        """Mark operation as completed."""
        if operation_id in self.operations:
            duration = time.time() - self.operations[operation_id]["start_time"]
            self.operations[operation_id]["status"] = "completed"
            self.operations[operation_id]["result"] = result
            self.operations[operation_id]["duration"] = duration
            print(f"✅ Completed: {self.operations[operation_id]['description']} in {duration:.2f}s")
    
    def fail_operation(self, operation_id: str, error: str) -> None:
        """Mark operation as failed."""
        if operation_id in self.operations:
            duration = time.time() - self.operations[operation_id]["start_time"]
            self.operations[operation_id]["status"] = "failed"
            self.operations[operation_id]["error"] = error
            print(f"❌ Failed: {self.operations[operation_id]['description']} in {duration:.2f}s - {error}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all tracked operations."""
        return self.operations


# -------------------------------------------------
# Error handling utilities
# -------------------------------------------------

class ErrorHandler:
    """Centralized error handling and reporting."""
    
    def __init__(self, logger: Optional[RuntimeLogger] = None):
        self.logger = logger or RuntimeLogger("error_handler")
    
    def handle_exception(self, operation: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle and log exceptions with context."""
        error_type = type(exception).__name__
        error_message = str(exception)
        
        self.logger.error(f"Exception in {operation}:", {
            "type": error_type,
            "message": error_message,
            "context": context or {}
        })
        
        # Re-raise for calling code to handle
        raise exception
    
    def handle_validation_error(self, operation: str, message: str) -> ValueError:
        """Handle validation errors with structured logging."""
        self.logger.error(f"Validation error in {operation}: {message}")
        raise ValueError(message)


# -------------------------------------------------
# Configuration utilities
# -------------------------------------------------

class RuntimeConfig:
    """Runtime configuration management."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or RUNTIME_ROOT / "runtime_config.yaml"
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                import yaml
                with open(self.config_file, 'r') as f:
                    self.config = yaml.safe_load(f)
            except Exception as e:
                print(f"Failed to load runtime config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.save_config()
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            import yaml
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        except Exception as e:
            print(f"Failed to save runtime config: {e}")


# -------------------------------------------------
# Global instances
# -------------------------------------------------
app_logger = RuntimeLogger("app")
system_monitor = SystemMonitor()
operation_tracker = OperationTracker()
error_handler = ErrorHandler(app_logger)
runtime_config = RuntimeConfig()