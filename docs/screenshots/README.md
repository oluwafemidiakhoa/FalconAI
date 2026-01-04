# FALCON-AI Dashboard Screenshots

This directory contains screenshots of the FALCON-AI live dashboard.

## How to Add Screenshots

1. **Run the dashboard:**
   ```bash
   falcon-ai serve --config configs/wow.yaml --port 8002
   ```

2. **Take screenshots of:**
   - **dashboard_main.png** - Full dashboard view showing all panels
   - **flight_summary.png** - Flight Summary panel (100% success rate, trigger rate, sparkline)
   - **telemetry_swarm.png** - Telemetry charts and Swarm Consensus panels
   - **events_benchmarks.png** - Event Stream and Benchmark Snapshot panels

3. **Save screenshots here:**
   - Use PNG format for best quality
   - Recommended size: 1920x1080 or similar
   - Ensure text is readable

4. **Update paths in README.md** if needed

## Current Screenshots

Based on the running dashboard at `http://127.0.0.1:8002`, the system shows:

### Main Dashboard
- **Title:** "Selective Intelligence in Motion"
- **Subtitle:** "Real-time perception, decisions, energy, memory, and swarm consensus"
- **Layout:** Clean, professional design with teal accents

### Flight Summary (LIVE)
- Success Rate: **100.0%**
- Trigger Rate: **10.1%** (highly selective)
- False Positives: **0**
- False Negatives: **0**
- Live sparkline showing performance over time

### Decision Stack
- Current action display
- Confidence level
- Reasoning explanation
- Updates in real-time

### Energy + Memory
- Energy Used: **1900** operations (example)
- Energy Remaining: **86.0%**
- Memory Size: **14** experiences
- Memory Type: **EXPERIENCE**

### Telemetry
Four real-time charts:
1. Confidence levels over time
2. Trigger rate (selectivity)
3. Energy remaining
4. Memory size growth

### Swarm Consensus
- Action voting breakdown (Escalate, Intervene, Alert, Observe)
- Pool: **8** shared experiences
- Delta: **+0.0%** improvement

### Benchmark Snapshot
Performance matrix showing 100% success across:
- spike/threshold/simple
- spike/threshold/adaptive
- spike/threshold/multi_tier
- spike/hybrid/simple
- spike/hybrid/adaptive

### Event Stream
Live events with:
- ACTION (red background) - Events requiring action
- IGNORE (gray) - Filtered normal events
- Confidence scores (e.g., 0.244, 0.216)
- Reasoning (e.g., "escalate", "none")

## Screenshot Specifications

**Recommended Settings:**
- **Resolution:** 1920x1080 or higher
- **Format:** PNG (for quality) or JPG (for smaller size)
- **Browser:** Latest Chrome/Edge for best rendering
- **Zoom:** 100% (no browser zoom)
- **Theme:** Default light theme (as shown)

**Key Scenes to Capture:**

1. **dashboard_main.png** - Full page showing:
   - Header with title
   - All control panels (Scenario, Mode, Interval, Pause)
   - All three main sections (Flight Summary, Decision Stack, Energy+Memory)

2. **flight_summary.png** - Close-up of:
   - Flight Summary panel
   - LIVE indicator
   - Success rate, trigger rate
   - False positives/negatives
   - Sparkline chart

3. **telemetry_swarm.png** - Bottom section showing:
   - All four telemetry charts
   - Swarm Consensus panel
   - Action voting breakdown

4. **events_benchmarks.png** - Right side showing:
   - Live Event Stream
   - Benchmark Snapshot
   - Performance metrics

## Using Screenshots in Documentation

These screenshots demonstrate:
- ✅ Production-ready UI
- ✅ Real-time monitoring capabilities
- ✅ Swarm intelligence visualization
- ✅ Performance benchmarking
- ✅ Clean, professional design
- ✅ Comprehensive metrics

Perfect for:
- GitHub README
- Project presentations
- Documentation
- Portfolio showcases
- Technical demos
- Social media posts
