# Quasar

High performance C++ Rocket League bot training program.

# Performance tips

ROCM:
HSA_OVERRIDE_GFX_VERSION=10.3.0 PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

MPS:
Enable mixed precision with config.use_mixed_precision = true
