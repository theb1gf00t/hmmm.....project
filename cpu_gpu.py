from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import psutil
import subprocess
import threading
from collections import deque

class SystemMonitor:
    def __init__(self):
        self.cpu_readings = deque()
        self.ram_readings = deque()
        self.gpu_readings = deque()
        self.monitoring = False

    def get_gpu_usage(self):
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                if len(gpu_data) >= 3:
                    return {
                        'gpu_utilization': float(gpu_data[0]),
                        'memory_used': float(gpu_data[1]),
                        'memory_total': float(gpu_data[2]),
                        'memory_percent': (float(gpu_data[1]) / float(gpu_data[2])) * 100,
                        'temperature': float(gpu_data[3]) if len(gpu_data) >= 4 else 0
                    }
        except:
            pass
        return None

    def get_system_usage(self):
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1e9,
            'ram_total_gb': psutil.virtual_memory().total / 1e9
        }

    def monitor_continuously(self, duration=300):
        self.cpu_readings.clear()
        self.ram_readings.clear()
        self.gpu_readings.clear()
        self.monitoring = True
        start_time = time.time()
        while self.monitoring and (time.time() - start_time) < duration:
            system_usage = self.get_system_usage()
            self.cpu_readings.append(system_usage['cpu_percent'])
            self.ram_readings.append(system_usage['ram_percent'])
            gpu_usage = self.get_gpu_usage()
            if gpu_usage:
                self.gpu_readings.append(gpu_usage['gpu_utilization'])
            time.sleep(1)

    def get_average_usage(self):
        cpu_avg = np.mean(list(self.cpu_readings)) if self.cpu_readings else 0
        ram_avg = np.mean(list(self.ram_readings)) if self.ram_readings else 0
        gpu_avg = np.mean(list(self.gpu_readings)) if self.gpu_readings else 0
        return {
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
            'cpu_max': max(self.cpu_readings) if self.cpu_readings else 0,
            'ram_max': max(self.ram_readings) if self.ram_readings else 0,
            'gpu_max': max(self.gpu_readings) if self.gpu_readings else 0,
            'samples': len(self.cpu_readings)
        }

def get_test_images(num_images=1500):
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    if not test_images_path.exists():
        return None
    all_images = list(test_images_path.glob('*.jpg'))
    if len(all_images) < num_images:
        return all_images
    else:
        return all_images[:num_images]

def benchmark_inference(device='cuda', num_images=1500):
    model = YOLO('/mnt/34B471F7B471BBC4/CSO_project/runs/detect/train7/weights/best.pt')
    all_images = get_test_images(num_images)
    if not all_images:
        return None
    actual_count = len(all_images)
    print(f"Starting benchmark on {device.upper()} with {actual_count} images...")
    monitor = SystemMonitor()
    print("Warming up...")
    for img_path in all_images[:5]:
        _ = model(img_path, device=device, verbose=False)
    times = []
    estimated_duration = actual_count * 0.1
    monitor_thread = threading.Thread(target=monitor.monitor_continuously, args=(estimated_duration * 2,))
    monitor_thread.start()
    start_total = time.time()
    for i, img_path in enumerate(all_images):
        start_time = time.time()
        results = model(img_path, device=device, verbose=False)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        times.append(inference_time)
        if (i + 1) % 100 == 0 or (i + 1) == actual_count:
            elapsed = time.time() - start_total
            images_remaining = actual_count - (i + 1)
            avg_time_per_image = elapsed / (i + 1)
            eta_seconds = images_remaining * avg_time_per_image
            eta_minutes = eta_seconds / 60
            avg_inference_ms = np.mean(times)
            current_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0
            print(f"Progress: {i+1}/{actual_count} images | "
                  f"Avg: {avg_inference_ms:.1f}ms | "
                  f"FPS: {current_fps:.1f} | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta_minutes:.1f}min")
    total_time = time.time() - start_total
    monitor.monitoring = False
    monitor_thread.join()
    system_stats = monitor.get_average_usage()
    stats = {
        'device': device,
        'total_images': actual_count,
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'total_time_s': total_time,
        'fps': actual_count / total_time,
        'all_times': times,
        'system_usage': system_stats
    }
    print(f"\nCompleted! Total time: {total_time/60:.2f} minutes | FPS: {stats['fps']:.2f}")
    return stats

def run_cpu_gpu_comparison():
    results = {}
    gpu_stats = benchmark_inference(device='cuda', num_images=1500)
    if gpu_stats:
        results['GPU'] = gpu_stats
    cpu_stats = benchmark_inference(device='cpu', num_images=1500)
    if cpu_stats:
        results['CPU'] = cpu_stats
    return results

def plot_4k_comparison(results):
    if not results or len(results) < 1:
        return
    devices = list(results.keys())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Image Performance Comparison ({results[devices[0]]["total_images"]} images)', fontsize=16, fontweight='bold')
    colors = {'GPU': '#4ECDC4', 'CPU': '#FF6B6B'}
    avg_times = [results[device]['avg_time_ms'] for device in devices]
    bars1 = ax1.bar(devices, avg_times, color=[colors[d] for d in devices], alpha=0.8)
    ax1.set_title('Average Inference Time per Image', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f} ms', ha='center', va='bottom', fontweight='bold')
    fps_values = [results[device]['fps'] for device in devices]
    bars2 = ax2.bar(devices, fps_values, color=[colors[d] for d in devices], alpha=0.8)
    ax2.set_title('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FPS')
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    total_times_min = [results[device]['total_time_s'] / 60 for device in devices]
    bars3 = ax3.bar(devices, total_times_min, color=[colors[d] for d in devices], alpha=0.8)
    ax3.set_title('Total Processing Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (minutes)')
    ax3.grid(True, alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f} min', ha='center', va='bottom', fontweight='bold')
    if len(devices) == 2:
        speedup = results['CPU']['avg_time_ms'] / results['GPU']['avg_time_ms']
        bars4 = ax4.bar(['Speedup'], [speedup], color='#FFD166', alpha=0.8)
        ax4.set_title('GPU Speedup Factor', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Times Faster')
        ax4.grid(True, alpha=0.3)
        ax4.text(0, speedup/2, f'{speedup:.1f}x', ha='center', va='center', fontsize=24, fontweight='bold', color='white')
    else:
        device = devices[0]
        sys_usage = results[device]['system_usage']
        usage_data = [sys_usage['cpu_avg'], sys_usage['ram_avg']]
        if device == 'GPU':
            usage_data.append(sys_usage['gpu_avg'])
        labels = ['CPU', 'RAM', 'GPU'][:len(usage_data)]
        bars4 = ax4.bar(labels, usage_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax4.set_title(f'{device} System Usage', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Usage (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def save_detailed_results(results):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_{timestamp}.csv"
    data = []
    for device, stats in results.items():
        sys_usage = stats['system_usage']
        data.append({
            'device': device,
            'total_images': stats['total_images'],
            'avg_time_ms': stats['avg_time_ms'],
            'std_time_ms': stats['std_time_ms'],
            'min_time_ms': stats['min_time_ms'],
            'max_time_ms': stats['max_time_ms'],
            'total_time_s': stats['total_time_s'],
            'total_time_min': stats['total_time_s'] / 60,
            'fps': stats['fps'],
            'cpu_usage_avg': sys_usage['cpu_avg'],
            'ram_usage_avg': sys_usage['ram_avg'],
            'gpu_usage_avg': sys_usage['gpu_avg'] if device == 'GPU' else 0,
            'samples_collected': sys_usage['samples']
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    for device, stats in results.items():
        timing_filename = f"timing_data_{device}_{timestamp}.csv"
        timing_df = pd.DataFrame({'inference_time_ms': stats['all_times']})
        timing_df.to_csv(timing_filename, index=False)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    results = run_cpu_gpu_comparison()
    if results:
        plot_4k_comparison(results)
        save_detailed_results(results)
