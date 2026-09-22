# Ethereal Canvas — Memory Management Architecture Specification

## 1. Purpose

This document defines the comprehensive memory management architecture for handling Qwen-Image-Edit-2511 and other large diffusion models within Ethereal Canvas. The architecture is designed to prevent torch.OutOfMemoryErrors while maintaining optimal performance and image quality.

The memory management system provides:
- Automatic detection of memory constraints and capabilities
- Multiple loading strategies with automatic fallback
- Real-time memory profiling and monitoring
- OOM detection with intelligent retry mechanisms
- Advanced quantization and optimization support

This document extends the core architecture specification to address memory management as a first-class concern.

---

## 2. Memory Management Principles

### 2.1 Proactive Memory Management

The system takes a proactive approach to memory management:
- Estimate memory requirements before model loading
- Select optimal loading strategies based on available resources
- Monitor memory usage continuously during operation
- Apply optimizations automatically when needed

### 2.2 Graceful Degradation

When memory constraints are detected:
- Automatically switch to more memory-efficient strategies
- Maintain functionality even with reduced performance
- Provide clear feedback about current operating mode
- Allow manual strategy override when needed

### 2.3 Performance-Aware Optimization

Memory optimizations are balanced against performance:
- Quantization techniques that preserve image quality
- Selective CPU offloading to minimize performance impact
- Adaptive strategy selection based on workload patterns

---

## 3. Memory Management Components

### 3.1 MemoryProfiler (`modules/memory/profiler.py`)

**Responsibility**: Runtime memory inspection and profiling

**Core Capabilities**:
- System and GPU memory snapshot collection
- Model memory requirement estimation
- Loading strategy benchmarking
- Memory usage trend analysis
- Profile persistence and comparison

**Key Classes**:
```python
@dataclass
class MemorySnapshot:
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    system_ram_mb: float
    process_ram_mb: float
    vram_total_mb: float
    vram_free_mb: float

class MemoryProfiler:
    def take_snapshot(self) -> MemorySnapshot
    def profile_model_loading(self, model_name, load_fn) -> ModelMemoryProfile
    def estimate_model_memory(self, model_config) -> Dict[str, float]
    def compare_profiles(self, profiles) -> Dict[str, Any]
```

### 3.2 MemoryManager (`modules/memory/manager.py`)

**Responsibility**: Strategic memory management and OOM prevention

**Core Capabilities**:
- Multiple loading strategy management
- OOM error detection and classification
- Automatic fallback chain execution
- Memory optimization application
- Strategy recommendation based on hardware

**Loading Strategies**:
```python
class LoadStrategy(Enum):
    FP16_FULL = "fp16_full"                    # Full precision, no optimization
    FP8_OPTIMIZED = "fp8_optimized"            # FP8 quantization (hardware dependent)
    NF4_QUANTIZED = "nf4_quantized"            # 4-bit quantization
    CPU_OFFLOAD = "cpu_offload"                # Model CPU offloading
    SEQUENTIAL_OFFLOAD = "sequential_offload"   # Layer-by-layer offloading
```

**Key Methods**:
```python
class MemoryManager:
    def load_model_with_fallback(self, model_name, load_fn) -> Tuple[Any, LoadingConfig]
    def is_oom_error(self, error: Exception) -> bool
    def get_recommended_strategy(self, available_memory_mb) -> List[LoadStrategy]
    def monitor_inference_memory(self, inference_fn) -> Any
```

### 3.3 EnhancedImageEditBackend (`modules/backends/enhanced_image_edit.py`)

**Responsibility**: Memory-managed backend implementation

**Core Capabilities**:
- Integration with memory management layer
- Automatic strategy selection and fallback
- Memory-aware inference execution
- Performance benchmarking and optimization

**Key Features**:
- Seamless integration with existing backend interface
- Automatic switching between loading strategies
- Comprehensive error handling and recovery
- Performance monitoring and reporting

---

## 4. Memory Strategy Hierarchy

### 4.1 Strategy Selection Process

The memory manager follows a deterministic hierarchy:

1. **Hardware Assessment**
   - Detect GPU capabilities (FP8 support, VRAM size)
   - Evaluate system memory constraints
   - Check quantization support

2. **Memory Estimation**
   - Calculate base memory requirements
   - Estimate per-strategy memory usage
   - Apply safety margins and buffers

3. **Strategy Recommendation**
   - Rank strategies by memory efficiency
   - Consider performance implications
   - Account for workload characteristics

4. **Loading with Fallback**
   - Attempt recommended strategy first
   - Automatic fallback on failure
   - Detailed error logging and analysis

### 4.2 Strategy Characteristics

| Strategy | Memory Savings | Performance Impact | Quality Impact | Use Case |
|-----------|----------------|-------------------|----------------|-----------|
| FP16_FULL | 0% | Baseline | None | Maximum quality |
| FP8_OPTIMIZED | 50% | Minor (5-15% slower) | Minimal | Modern GPUs |
| NF4_QUANTIZED | 65% | Moderate (20-40% slower) | Noticeable | Memory constrained |
| CPU_OFFLOAD | 75% | Significant (2-3x slower) | None | Extreme constraints |
| SEQUENTIAL_OFFLOAD | 85% | Major (4-5x slower) | None | Minimal VRAM |

---

## 5. Advanced Optimizations

### 5.1 FP8/Lightning Path

**Components**:
- `FP8LightningOptimizer` class for advanced quantization
- Lightning LoRA integration for 4-8 step inference
- Hardware-aware FP8 support detection
- TorchAO integration when available

**Implementation**:
```python
class FP8LightningOptimizer:
    def create_fp8_config(self) -> Optional[Dict[str, Any]]
    def create_lightning_lora_config(self) -> Dict[str, Any]
    def load_model_with_optimizations(self, model_name, cache_dir) -> Tuple[Any, Dict]
    def benchmark_optimizations(self, model_name, cache_dir) -> Dict[str, Any]
```

### 5.2 Quantization Support

**FP8 Quantization**:
- Supported on Ada Lovelace (8.9+) and Hopper (9.0+) GPUs
- 50% memory reduction with minimal quality loss
- Requires recent PyTorch and CUDA versions

**NF4 Quantization**:
- Universal support across GPU generations
- 65% memory reduction with noticeable quality impact
- Uses bitsandbytes library

**Lightning LoRA**:
- 4-8 step inference (3-8x speedup)
- Minimal quality degradation
- Requires additional weight downloads

---

## 6. Integration Patterns

### 6.1 Backend Integration

Memory management integrates seamlessly with existing backends:

```python
# Standard backend usage with memory management
backend = EnhancedImageEditBackend()
success = backend.load()  # Automatic strategy selection
result = backend.edit(prompt, image_path)  # Memory-monitored inference
```

### 6.2 Configuration Integration

Memory management respects existing configuration patterns:

```yaml
# config.yaml
memory_management:
  preferred_strategies: ["fp16_full", "fp8_optimized", "nf4_quantized"]
  fallback_enabled: true
  oom_retry_attempts: 3
  memory_threshold_mb: 2000
  profiling_enabled: true
```

### 6.3 Logging Integration

Memory events are logged with structured format:

```markdown
## Memory Management Log Entry

### Loading Strategy: fp8_lightning
- Device: cuda
- Available VRAM: 24GB
- Selected Strategy: FP8 + Lightning LoRA
- Load Time: 3.2s
- Peak Memory: 8.7GB
- Success: True

### Optimization Applied:
- FP8 quantization (TorchAO)
- Lightning LoRA (8-step)
- Attention slicing enabled
- xFormers optimization
```

---

## 7. Performance Monitoring

### 7.1 Metrics Collection

The system continuously collects:
- Memory usage patterns over time
- Loading strategy success rates
- OOM occurrence frequency
- Performance impact measurements
- Quality degradation assessments

### 7.2 Adaptive Optimization

Based on collected metrics, the system:
- Adapts strategy preferences over time
- Identifies optimal configurations for specific workloads
- Provides recommendations for hardware upgrades
- Optimizes resource allocation patterns

---

## 8. Error Handling and Recovery

### 8.1 OOM Detection

The memory manager identifies OOM errors through:
- Exception message pattern matching
- Memory allocation failure detection
- System memory pressure monitoring
- GPU memory exhaustion detection

### 8.2 Recovery Strategies

When OOM is detected:
1. **Immediate Cleanup**: Aggressive memory garbage collection
2. **Strategy Fallback**: Switch to more memory-efficient strategy
3. **Inference Retry**: Attempt with reduced batch size or resolution
4. **User Notification**: Clear error messaging and suggestions

### 8.3 Degradation Modes

The system supports multiple degradation modes:
- **Performance Mode**: Prioritize speed over memory efficiency
- **Memory Mode**: Prioritize memory conservation
- **Quality Mode**: Prioritize output quality
- **Adaptive Mode**: Balance all factors automatically

---

## 9. Testing and Validation

### 9.1 Memory Testing Framework

Comprehensive testing suite includes:
- Strategy viability testing across GPU configurations
- Memory usage benchmarking and validation
- OOM scenario simulation and recovery testing
- Performance impact measurement and analysis

### 9.2 Automated Assessment

Automated tools provide:
- Hardware capability assessment
- Strategy recommendation engine
- Performance prediction models
- Optimization impact analysis

---

## 10. Configuration and Customization

### 10.1 Strategy Configuration

Users can customize strategy selection:

```python
# Custom strategy preference
backend = EnhancedImageEditBackend(
    preferred_strategies=[
        LoadStrategy.FP8_OPTIMIZED,
        LoadStrategy.NF4_QUANTIZED,
        LoadStrategy.CPU_OFFLOAD
    ]
)
```

### 10.2 Memory Thresholds

Configurable memory thresholds:

```python
# Custom memory management
memory_manager.configure(
    warning_threshold_mb=16000,
    critical_threshold_mb=22000,
    fallback_trigger_threshold=2000
)
```

---

## 11. Future Extensions

### 11.1 Advanced Quantization

Planned additions:
- INT8 weight-only quantization
- Dynamic quantization strategies
- Custom quantization kernel integration
- Hardware-specific optimizations

### 11.2 Distributed Memory Management

Future capabilities:
- Multi-GPU memory distribution
- Model parallelization support
- Cross-device memory sharing
- Cluster-wide memory optimization

---

## 12. Authority

This document defines the authoritative memory management architecture for Ethereal Canvas. Implementation must follow these specifications unless explicitly overridden by system requirements.

In case of conflict between implementation and this document, this document takes precedence and the implementation must be updated to conform.

---

## 13. Version History

- **v1.0** (2026-01-23): Initial specification with comprehensive memory management layer
- **v1.1** (Future): Planned additions for distributed memory management and advanced quantization