# Swallow-Bat Algorithm (SBA) for Threat-Avoidance Routing in Container Truck Routing Problems  


## 📋 Overview  
This repository contains the implementation of the **Swallow-Bat Algorithm (SBA)**, a novel bio-inspired hybrid metaheuristic designed for solving **Threat-Aware Container Truck Routing Problems (CTRP)**. The algorithm uniquely combines swallow flocking behavior (for global exploration) with bat echolocation (for local exploitation), incorporating a dedicated threat-aware evasion operator to mitigate risks in logistics routing.  


## 🎯 Key Features  
- **Hybrid Metaheuristic**: Merges swarm intelligence principles from swallow flocking and bat echolocation.  
- **Threat-Aware Routing**: Proactively avoids spatial threats (conflict zones, infrastructure risks, environmental hazards).  
- **Multi-Objective Optimization**: Simultaneously minimizes cost, distance, and threat exposure.  
- **Real-World Applicability**: Validated on benchmark instances and East African Community (EAC) logistics corridors.  
- **High Performance**: Outperforms established algorithms including ALNS (Adaptive Large Neighborhood Search) and HGA (Hybrid Genetic Algorithm).  


## 🚀 Algorithm Components  

### 1. Swallow Flocking Operators  
- **Separation**: Prevents overcrowding to maintain solution diversity.  
- **Alignment**: Synchronizes velocity among neighboring agents for coordinated search.  
- **Cohesion**: Promotes movement toward the local group center to balance exploration.  

### 2. Bat Echolocation Components  
- **Frequency Adaptation**: Dynamically adjusts search frequency to balance exploration and exploitation.  
- **Velocity Update**: Guides movement toward globally optimal solutions.  
- **Local Search Refinement**: Uses Gaussian perturbation around best positions to refine solutions.  

### 3. Threat-Aware Evasion  
- **Proactive Risk Mitigation**: Dynamically steers routes away from hazardous zones.  
- **Static Threat Modeling**: Represents threats as circular restricted zones with fixed centers and radii.  


## 📊 Performance Highlights  

### Benchmark Results  
- **100% success rate** across all test instances.  
- **1.74% Coefficient of Variation**, demonstrating exceptional stability.  
- **0 threat exposures** while maintaining cost efficiency.  
- **20.85% lower mean cost** compared to the second-best algorithm.  

### Real-World Case Study (East African Community)  
- **1.8% to 22.0% lower total costs** than competing algorithms.  
- **100% feasibility** in handling vehicle capacity and customer demand constraints.  
- **Perfect threat avoidance** in validated scenarios.  


## 🛠️ Installation & Requirements  

### Prerequisites  
- Python 3.8+  
- NumPy 1.20+  
- Matplotlib 3.3+  
- OSRM (Open Source Routing Machine, for real-world distance calculations).  

### Installation  
```bash
git clone https://github.com/YvesNDIKURIYO-2022/swallow-bat-algorithm.git
cd swallow-bat-algorithm
pip install -r requirements.txt
```  


## 💻 Usage  

### Basic Implementation  
```python
from sba_algorithm import SwallowBatAlgorithm
from problem_instance import ThreatAwareCTRP

# Initialize problem instance
problem = ThreatAwareCTRP(
    nodes=node_locations,       # Coordinates of customers and depots
    threats=threat_zones,       # Spatial threat zone definitions
    vehicle_capacity=100,       # Maximum load capacity per truck
    demands=customer_demands    # Demand values for each customer
)

# Configure SBA parameters
sba = SwallowBatAlgorithm(
    population_size=50,         # Number of solutions in the swarm
    max_iterations=500,         # Maximum optimization iterations
    risk_weight=1000,           # Weight for threat avoidance in objective function
    frequency_range=[0, 2]      # Range for bat echolocation frequency adaptation
)

# Solve the problem and visualize results
solution = sba.solve(problem)
solution.visualize_routes()    # Plots optimized routes and threat zones
```  

### Parameter Tuning Guidance  
- **Small problems (≤20 customers)**: Use `population_size=35, risk_weight=1000`.  
- **Medium to large problems**: Focus on calibrating `risk_weight`; consider `population_size=45` for better quality.  
- **Runtime vs. Quality**: Larger populations improve solution quality but increase computation time linearly.  


## 📁 Project Structure  
```
swallow-bat-algorithm/
├── src/
│   ├── sba_core.py          # Main SBA algorithm implementation
│   ├── problem_model.py     # CTRP problem formulation and constraints
│   ├── operators.py         # Swallow flocking and bat echolocation operators
│   ├── threat_evasion.py    # Threat-aware route adjustment mechanisms
│   └── solution.py          # Solution representation and decoding logic
├── data/
│   ├── benchmarks/          # Modified Augerat VRP benchmark instances
│   ├── east_africa/         # East African Community corridor data
│   └── threat_zones/        # Geographic threat zone definitions (coordinates, radii)
├── experiments/
│   ├── benchmark_tests.py   # Scripts to validate performance on benchmarks
│   ├── sensitivity_analysis.py # Impact analysis of key parameters
│   └── case_study.py        # Real-world EAC corridor application
├── results/
│   ├── visualizations/      # Route maps, convergence graphs, and Pareto fronts
│   └── statistical_analysis/ # Performance metrics (cost, stability, threat exposure)
├── docs/
│   └── manuscript.pdf       # Full research paper with theoretical foundations
├── LICENSE                  # MIT License details
└── README.md                # Project documentation (this file)
```  


## 🧪 Experimental Setup  

### Benchmark Instances  
- **Small-scale**: Modified A-n32-k5 (21 customers, 3 vehicles, 5 threat zones).  
- **Medium-scale**: Modified A-n53-k7 (33 customers, 5 vehicles, 5 threat zones).  
- **Large-scale**: Modified A-n80-k10 (51 customers, 7 vehicles, 6 threat zones).  

### Comparison Algorithms  
- Particle Swarm Optimization (PSO)  
- Bat Algorithm (BA)  
- Harris Hawks Optimization (HHO)  
- Hybrid Genetic Algorithm (HGA)  
- Adaptive Large Neighborhood Search (ALNS)  


## 📈 Results Visualization  
The implementation includes tools to generate key insights:  
- **Route Mapping**: Geographic visualization of optimized routes with overlaid threat zones.  
- **Convergence Analysis**: Plots of objective function values over iterations to assess algorithm progress.  
- **Pareto Fronts**: Multi-objective trade-offs between cost, distance, and threat exposure.  
- **Cost Structure**: Breakdown of total costs (transport, risk mitigation, etc.).  


## 🎓 Citation  
If you use this algorithm in your research, please cite:  

```bibtex
@article{ndikuriyo2025,
  title={A Bio-Inspired Swallow-Bat Algorithm for Threat-Avoidance Routing in the Container Truck Routing Problem},
  author={Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal={Transportation Research Part E: Logistics and Transportation Review},
  year={2025},
  publisher={Elsevier}
}
```  


## 📄 License  
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  

```
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
```  


## 👥 Authors  
- **Yves Ndikuriyo** - Lead Researcher & Algorithm Development  
- **Yinggui Zhang** - Research Supervision & Methodology  
- **Dung Davou Fom** - Experimental Analysis & Validation  


## 🤝 Contributing  
Contributions are welcome! Please submit pull requests or open issues for bug reports, feature suggestions, or improvements.  


## 📞 Contact  
For questions and collaborations:  
- **Lead Researcher**: Yves Ndikuriyo - [yvesndikuriyo@csu.edu.cn](mailto:yvesndikuriyo@csu.edu.cn)  
- **Project Repository**: [https://github.com/YvesNDIKURIYO-2022/swallow-bat-algorithm](https://github.com/YvesNDIKURIYO-2022/swallow-bat-algorithm)  


**Note**: This implementation is based on research published in the accompanying manuscript. Refer to [docs/manuscript.pdf](docs/manuscript.pdf) for detailed mathematical formulations, theoretical foundations, and comprehensive experimental results.  

**Research Reference**: Ndikuriyo, Y., Zhang, Y., & Fom, D. D. (2025). A Bio-Inspired Swallow-Bat Algorithm for Threat-Avoidance Routing in the Container Truck Routing Problem. *Transportation Research Part E: Logistics and Transportation Review*.
