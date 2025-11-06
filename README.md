# Swallow-Bat Algorithm (SBA) for Threat-Avoidance Routing in Container Truck Routing Problems

## 📋 Overview

This repository contains the implementation of the **Swallow-Bat Algorithm (SBA)**, a novel bio-inspired hybrid metaheuristic designed for solving Threat-Aware Container Truck Routing Problems (CTRP). The algorithm uniquely combines swallow flocking behavior for global exploration with bat echolocation for local exploitation, incorporating a dedicated threat-aware evasion operator for risk mitigation in logistics routing.

## 🎯 Key Features

- **Hybrid Metaheuristic**: Merges swarm intelligence principles from swallow flocking and bat echolocation
- **Threat-Aware Routing**: Proactively avoids spatial threats (conflict zones, infrastructure risks, environmental hazards)
- **Multi-Objective Optimization**: Simultaneously minimizes cost, distance, and threat exposure
- **Real-World Applicability**: Validated on benchmark instances and East African Community logistics corridors
- **High Performance**: Outperforms established algorithms including ALNS and HGA

## 🚀 Algorithm Components

### 1. Swallow Flocking Operators
- **Separation**: Prevents overcrowding and maintains diversity
- **Alignment**: Synchronizes velocity among neighboring agents  
- **Cohesion**: Promotes movement toward local group center

### 2. Bat Echolocation Components
- **Frequency Adaptation**: Dynamic search frequency adjustment
- **Velocity Update**: Movement toward global best solutions
- **Local Search Refinement**: Gaussian perturbation around best positions

### 3. Threat-Aware Evasion
- **Proactive Risk Mitigation**: Dynamically steers solutions away from hazardous zones
- **Static Threat Modeling**: Circular restricted zones with fixed centers and radii

## 📊 Performance Highlights

### Benchmark Results
- **100% success rate** across all test instances
- **1.74% Coefficient of Variation** demonstrating exceptional stability
- **0 threat exposures** while maintaining cost efficiency
- **20.85% lower mean cost** compared to second-best algorithm

### Real-World Case Study (East African Community)
- **1.8% to 22.0% lower total costs** than competing algorithms
- **100% feasibility** in handling capacity and demand constraints
- **Perfect threat avoidance** in benchmark scenarios

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.8+
- NumPy 1.20+
- Matplotlib 3.3+
- OSRM (for real-world distance calculations)

### Installation
```bash
git clone https://github.com/your-username/swallow-bat-algorithm.git
cd swallow-bat-algorithm
pip install -r requirements.txt

## Quick Start
```bash
git clone https://github.com/YvesNDIKURIYO-2022/swallow-bat-algorithm.git
cd swallow-bat-algorithm
pip install -r requirements.txt
Basic Implementation
python
from sba_algorithm import SwallowBatAlgorithm
from problem_instance import ThreatAwareCTRP

# Initialize problem instance
problem = ThreatAwareCTRP(
    nodes=node_locations,
    threats=threat_zones,
    vehicle_capacity=100,
    demands=customer_demands
)

# Configure SBA parameters
sba = SwallowBatAlgorithm(
    population_size=50,
    max_iterations=500,
    risk_weight=1000,
    frequency_range=[0, 2]
)

# Solve the problem
solution = sba.solve(problem)
solution.visualize_routes()
Parameter Tuning Guidance
Small problems (≤20 customers): pop=35, W_risk=1000

Medium to large problems: Focus on W_risk calibration, consider pop=45

Runtime vs Quality: Larger populations improve quality but increase computation time linearly

📁 Project Structure
text
swallow-bat-algorithm/
├── src/
│   ├── sba_core.py          # Main algorithm implementation
│   ├── problem_model.py     # CTRP problem formulation
│   ├── operators.py         # Flocking and echolocation operators
│   ├── threat_evasion.py    # Threat-aware evasion mechanisms
│   └── solution.py          # Solution representation and decoding
├── data/
│   ├── benchmarks/          # Modified Augerat instances
│   ├── east_africa/        # EAC corridor data
│   └── threat_zones/       # Geographic threat definitions
├── experiments/
│   ├── benchmark_tests.py   # Performance validation
│   ├── sensitivity_analysis.py
│   └── case_study.py       # Real-world application
├── results/
│   ├── visualizations/      # Route plots and convergence graphs
│   └── statistical_analysis/
├── docs/
│   └── manuscript.pdf       # Research paper
├── LICENSE
└── README.md
🧪 Experimental Setup
Benchmark Instances
Small-scale: Modified A-n32-k5 (21 customers, 3 vehicles, 5 threat zones)

Medium-scale: Modified A-n53-k7 (33 customers, 5 vehicles, 5 threat zones)

Large-scale: Modified A-n80-k10 (51 customers, 7 vehicles, 6 threat zones)

Comparison Algorithms
Particle Swarm Optimization (PSO)

Bat Algorithm (BA)

Harris Hawks Optimization (HHO)

Hybrid Genetic Algorithm (HGA)

Adaptive Large Neighborhood Search (ALNS)

📈 Results Visualization
The implementation includes comprehensive visualization capabilities:

Route Mapping: Geographic display of optimized routes and threat zones

Convergence Analysis: Algorithm performance over iterations

Pareto Fronts: Multi-objective trade-off analysis

Cost Structure: Breakdown of routing cost components

🎓 Citation
If you use this algorithm in your research, please cite:

bibtex
@article{ndikuriyo2025,
  title={A Bio-Inspired Swallow-Bat Algorithm for Threat-Avoidance Routing in the Container Truck Routing Problem},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={Journal of Heuristics},
  year={2025},
  publisher={Springer}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

text
MIT License

Copyright (c) 2025 Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
👥 Authors
Yves Ndikuriyo - Lead Researcher & Algorithm Development

Yinggui Zhang - Research Supervision & Methodology

Dung Davou Fom - Experimental Analysis & Validation

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and bug reports.

📞 Contact
For questions and collaborations, please contact:

Lead Researcher: Yves Ndikuriyo - [yves.ndikuriyo@institution.edu]

Project Repository: [GitHub URL]

Note: This implementation is based on research published in the accompanying manuscript. Please refer to the full paper for detailed mathematical formulations, theoretical foundations, and comprehensive experimental results.

Research Reference: Ndikuriyo, Y., Zhang, Y., & Fom, D. D. (2025). A Bio-Inspired Swallow-Bat Algorithm for Threat-Avoidance Routing in the Container Truck Routing Problem. Journal of Heuristics.
