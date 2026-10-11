# LIGHTNING

A high-performance Python application leveraging PyTorch for deep learning tasks, optimized for both CPU and CUDA-enabled GPUs.

## Architecture Overview

The `LIGHTNING` repository is structured with a single entry point, `main.py`, which serves as the core of the application. This file handles device configuration, hardware auto-tuning, and initializes the main execution flow. The architecture is designed to be modular, allowing for easy integration of additional components or modifications.

- **Device Configuration & Hardware Auto-Tuning**: The application dynamically detects available hardware (CPU or CUDA-enabled GPU) and configures settings such as parallelism based on the detected resources. This ensures optimal performance across different environments.

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **GUI Library**: Tkinter
- **Numerical Libraries**:
  - NumPy
  - Torch (for tensor operations)
- **Concurrency**: Threading for parallel processing
- **Data Structures**: Collections (e.g., `deque`)

## Getting Started / Setup Guide

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/LIGHTNING.git
   cd LIGHTNING
   ```

2. **Install Dependencies**:
   Ensure you have Python and pip installed. Then, install the required packages using pip:
   ```bash
   pip install torch numpy
   ```

3. **Run the Application**:
   Execute the main script to start the application:
   ```bash
   python main.py
   ```

4. **Environment Configuration**:
   - Ensure that your system has the necessary hardware (GPU with CUDA support) if you intend to leverage GPU acceleration.
   - The application will automatically detect and configure the device settings.

5. **Customization**:
   - Modify the `main.py` file to adjust parameters such as parallelism settings or integrate additional modules as needed.

## Additional Notes

- **Hardware Detection**: The application includes a robust hardware detection mechanism that configures settings based on available resources, ensuring efficient execution.
- **Scalability**: Designed with modularity in mind, the architecture allows for easy scaling and integration of new features or components.