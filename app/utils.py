
import psutil
import torch

def get_torch_gpu_info():
    if not torch.cuda.is_available():
        return "CUDA is not available on this system."
    
    info = []
    info.append(f"PyTorch version: {torch.__version__}")
    info.append(f"CUDA version: {torch.version.cuda}")
    info.append(f"cuDNN version: {torch.backends.cudnn.version()}")
    info.append(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        info.append(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        info.append(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        info.append(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        info.append(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        info.append(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        info.append(f"  Max memory allocated: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")

    system_memory = psutil.virtual_memory()
    info.append(f"\nSystem Memory:")
    info.append(f"  Total: {system_memory.total / 1e9:.2f} GB")
    info.append(f"  Available: {system_memory.available / 1e9:.2f} GB")
    info.append(f"  Used: {system_memory.used / 1e9:.2f} GB")
    info.append(f"  Percentage used: {system_memory.percent}%")

    return info
