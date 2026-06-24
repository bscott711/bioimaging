import subprocess
import time
import matplotlib.pyplot as plt
from collections import deque
import sys

def get_gpu_metrics():
    # Query utilization.gpu and memory.used for all GPUs
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True
    )
    lines = result.stdout.strip().split('\n')
    metrics = []
    for line in lines:
        if line:
            util, mem = line.split(',')
            metrics.append((int(util.strip()), int(mem.strip())))
    return metrics

def main(duration_seconds=600):
    print(f"Monitoring GPU utilization for {duration_seconds} seconds...")
    
    times = deque(maxlen=duration_seconds)
    gpu0_util = deque(maxlen=duration_seconds)
    gpu0_mem = deque(maxlen=duration_seconds)
    gpu1_util = deque(maxlen=duration_seconds)
    gpu1_mem = deque(maxlen=duration_seconds)
    
    start_time = time.time()
    
    try:
        while True:
            elapsed = int(time.time() - start_time)
            if elapsed > duration_seconds:
                break
                
            metrics = get_gpu_metrics()
            if len(metrics) >= 2:
                times.append(elapsed)
                gpu0_util.append(metrics[0][0])
                gpu0_mem.append(metrics[0][1] / 1024.0) # Convert MB to GB
                gpu1_util.append(metrics[1][0])
                gpu1_mem.append(metrics[1][1] / 1024.0) # Convert MB to GB
                
            sys.stdout.write(f"\rElapsed: {elapsed}s | GPU0: {metrics[0][0]}% ({metrics[0][1]/1024:.1f}GB) | GPU1: {metrics[1][0]}% ({metrics[1][1]/1024:.1f}GB)      ")
            sys.stdout.flush()
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped early.")
        
    print("\nGenerating plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Compute Utilization
    ax1.plot(times, gpu0_util, label="GPU 0 Compute", color='blue')
    ax1.plot(times, gpu1_util, label="GPU 1 Compute", color='orange')
    ax1.set_ylabel("Compute Utilization (%)")
    ax1.set_title("GPU Compute & VRAM Usage Over Time")
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot VRAM Usage
    ax2.plot(times, gpu0_mem, label="GPU 0 VRAM", color='darkblue')
    ax2.plot(times, gpu1_mem, label="GPU 1 VRAM", color='darkorange')
    ax2.axhline(y=97.8, color='red', linestyle='--', label='99.6GB Limit (OOM Crash Zone)')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("VRAM Usage (GB)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    output_file = "/home/SDSMT.LOCAL/bscott/.gemini/antigravity-ide/brain/796182a1-20db-4a95-bd08-7212a7e444d7/scratch/gpu_metrics_over_time.png"
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")

if __name__ == "__main__":
    main(duration_seconds=600)
