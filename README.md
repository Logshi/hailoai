# Hailo AI RTSP Benchmark on Raspberry Pi 5

Real-time RTSP AI vision on a Raspberry Pi 5, comparing CPU inference against the Hailo AI HAT+ (NPU). The project focuses on measuring and tuning an edge inference pipeline — latency, throughput and thermal behavior.

## Overview

Edge AI lives and dies by what the hardware can actually sustain. This project sets up a real-time RTSP vision pipeline on a Raspberry Pi 5 and uses it to compare CPU-only inference with the Hailo AI HAT+ NPU, so the performance trade-offs of edge accelerators can be observed under realistic conditions.

## Motivation

I wanted hands-on understanding of what an NPU buys you on a constrained device — not from spec sheets, but from running the same RTSP pipeline both ways and observing latency, frame rate and thermals. This is core knowledge for deploying computer vision at the edge.

## Features

- Real-time RTSP video ingestion on Raspberry Pi 5
- CPU vs Hailo AI HAT+ (NPU) inference comparison
- Latency and FPS observation
- Thermal behavior monitoring
- Pipeline tuning experiments

## Architecture

```
RTSP Camera Stream
        |
  Raspberry Pi 5 (capture)
        |
   +----+----+
   |         |
 CPU Path   Hailo NPU Path
   |         |
   +----+----+
        |
  Metrics: latency / FPS / thermals
```

## Tech Stack

- **Language:** Python
- **Hardware:** Raspberry Pi 5, Hailo AI HAT+ (NPU)
- **Input:** RTSP streams
- **Focus:** edge inference, benchmarking, pipeline tuning

## Installation

> Adapt to the actual setup in this repository.

```bash
git clone https://github.com/Logshi/hailoai.git
cd hailoai
pip install -r requirements.txt
```

Requires a Raspberry Pi 5 with the Hailo AI HAT+ and the corresponding Hailo runtime/drivers installed for the NPU path.

## Usage

```bash
# Example — adapt to actual scripts
python run_benchmark.py --source rtsp://<camera-url> --backend cpu
python run_benchmark.py --source rtsp://<camera-url> --backend hailo
```

## Demo

> Placeholders — replace with real captures.

- `docs/screenshots/cpu-vs-npu.png`
- `docs/demo.gif`

## Results

Benchmark in progress. Latency, FPS and thermal measurements for CPU vs Hailo NPU will be published here once collected on the target hardware. No fabricated benchmark figures are included.

## Roadmap

- [ ] Publish reproducible CPU vs NPU benchmark tables
- [ ] Add multiple model sizes to the comparison
- [ ] Log thermal throttling behavior over sustained runs
- [ ] Document power consumption observations

## What I Learned

- Setting up real-time RTSP inference on the Raspberry Pi 5
- Working with the Hailo AI HAT+ NPU runtime
- Measuring latency, throughput and thermals on edge hardware
- Reasoning about CPU vs NPU trade-offs for edge deployment

## Security & Privacy

No API keys, private camera URLs, real footage or sensitive deployment details are included in this repository.

## License

MIT — see [LICENSE](LICENSE).
