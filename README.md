# HangDetect

HangDetect is a CUDA kernel monitoring and hang-detection helper library built as a Linux ELF audit module. It intercepts CUDA and NCCL calls via `LD_AUDIT`, measures execution time using CUDA events, and emits structured logs for downstream analysis.

## Features

- **LD_AUDIT-based interception**  
  Uses the ELF auditing interface to hook CUDA Runtime / Driver APIs and NCCL collectives.
- **Kernel & NCCL monitoring**  
  Measures GPU execution time for kernel launches and NCCL operations.
- **Hang-detection oriented**  
  Produces events that can be used to implement hang detection and performance alerting.
- **Structured JSON logging**  
  Logs can be fed into existing log / tracing systems.
- **Runtime control via C/Python FFI**  
  Process-wide enable/disable switch and user labels for tagging executions, controllable from C/C++ or Python via a small FFI layer.

## Quick Start

### Build

```bash
cargo build --release
```

This produces a shared library such as:

```text
target/release/libhangdetect.so
```

### Run with LD_AUDIT

Run your CUDA / NCCL application with HangDetect loaded as an audit module:

```bash
LD_AUDIT=/full/path/to/target/release/libhangdetect.so ./your_cuda_application
```

Notes:

- Use an **absolute path** for `LD_AUDIT`.
- You can chain multiple audit modules if needed:

  ```bash
  LD_AUDIT="/path/to/libhangdetect.so:/path/to/another_audit.so" ./your_cuda_application
  ```

When the dynamic linker binds CUDA / NCCL symbols, HangDetect automatically wraps the supported APIs.

## Configuration

HangDetect is configured via environment variables and a small FFI-accessible control API.

### Environment variables

- `HANG_DETECTION_ENABLED`  
  Process-wide default for monitoring.  
  If set to `"1"` at process start (for example `HANG_DETECTION_ENABLED=1 LD_AUDIT=... python your_script.py`), monitoring starts enabled.  
  Otherwise, monitoring starts disabled and can be toggled at runtime via the control API.

- `HANGDETECT_LOG_FILE`  
  Base path of the log file.  
  If set, logs are written to:

  ```text
  <HANGDETECT_LOG_FILE>.<LOCAL_RANK>
  ```

  If unset, logging falls back to the default `env_logger` output (typically stderr).

- `HANGDETECT_LOG_LEVEL`  
  Log level for HangDetect internals.  
  Examples: `trace`, `debug`, `info`, `warn`, `error` (default: `info`).

- `HANGDETECT_KERNEL_FILTER`  
  Optional regular expression used to filter which kernels / NCCL operations are monitored.  
  If set, only names matching this regex are logged; if unset, everything is monitored.

- `LOCAL_RANK`  
  Optional rank identifier (often set by distributed training launchers).  
  Used only to suffix the log file name when `HANGDETECT_LOG_FILE` is set.

### Control API (C / Python)

At runtime, HangDetect exposes a small control API via FFI. The LD_AUDIT module publishes the addresses of these functions in a per-process shared-memory block, and higher-level code (for example Python) can attach to them.

```c
// Enable or disable monitoring for this process.
void hangdetect_set_enable(bool enabled);

// Set a user label for this process.
// The label will be attached to subsequent kernel/NCCL logs.
// Pass NULL to clear the label.
void hangdetect_set_kernel_exec_label(const char* label);
```

Typical usage (C/C++):

```c
#include <stdbool.h>

void run_training_step(void) {
    // Enable monitoring for this thread
    hangdetect_set_enable(true);

    // Tag all operations in this step
    hangdetect_set_kernel_exec_label("training_step_0");

    // Launch your CUDA kernels / NCCL collectives here
}
```

A similar pattern can be used from Python. In this repository, `tests/pytorch_test.py` demonstrates how to:

- Read the FFI function pointers exported by the LD_AUDIT module from `/dev/shm/hangdetect_ctl_<PID>`.
- Wrap them with `ctypes` into `hangdetect_set_enable` / `hangdetect_set_kernel_exec_label` helpers.
- Dynamically toggle monitoring and attach labels (e.g. around module forward passes) in a PyTorch training loop, including multi-process runs via `torchrun`.

## Log Output

HangDetect emits JSON lines describing kernel / NCCL execution. For example:

```json
{"type":"Base","data":{"pid":12345,"timestamp_ms":1702646453000.0}}
{"type":"Start","data":{"kern_label":"cudaLaunchKernel(Runtime)","user_label":"training_step_0","timestamp_ms":12.34}}
{"type":"Complete","data":{"kern_label":"cudaLaunchKernel(Runtime)","user_label":"training_step_0","duration_ms":1.23,"timestamp_ms":13.57}}
```

These events can be used to:

- Detect long-running or stuck operations.
- Correlate GPU activity with higher-level application phases.
- Build timelines for debugging hangs and performance issues.

## Python / High-level Bindings (Planned)

- Python wrapper around the C API for PyTorch / other Python workloads.
- Example integration for common training loops.

## License

This project is licensed under the MIT License – see the LICENSE file for details.