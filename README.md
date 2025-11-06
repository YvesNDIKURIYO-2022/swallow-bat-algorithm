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
