import torch

def test_gpu_availability():
    print("Testing GPU availability and configuration...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get current device info
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device index: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"GPU device name: {device_name}")
        
        # Get device properties
        device_properties = torch.cuda.get_device_properties(current_device)
        print(f"Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
        print(f"CUDA capability: {device_properties.major}.{device_properties.minor}")
        
        # Test GPU with a simple tensor operation
        print("\nTesting GPU with tensor operation...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("GPU tensor operation completed successfully")
        
    else:
        print("No GPU available - will use CPU instead")

if __name__ == "__main__":
    test_gpu_availability()
