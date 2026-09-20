# Minimal Dataset — Autonomous Multi-Drone Monitoring of Gregarious Animals

Supporting data for:

> Rolland, E. G. A., Costelloe, B. R., Lundquist, U. P. S., & Christensen, A. L. (2026).
> *Autonomous Multi-Drone Monitoring of Gregarious Animals: PSO-Based Surface-of-Interest Optimisation from Simulation to Field Deployment.* Drones.

This archive contains the data, metadata, and analysis scripts needed to reproduce the main results of the paper.

## Study overview

The study develops an autonomous multi-drone system for monitoring zebra herds from complementary viewpoints. Monitoring quality is defined using *surfaces of interest*: the dorsal surface and the two flanks.

The dataset covers three parts of the study:

1. PSO tuning and validation in simulation.
2. Four autonomous field missions at Ol Pejeta Conservancy, Kenya.
3. Scalability experiments with 1–9 monitoring drones and 10–100 animals.

## Dataset structure

```text
minimal_dataset/
├── README.md
├── requirements.txt
├── data/
│   ├── simulation/
│   ├── field/
│   └── metadata/
├── scripts/
└── outputs/
```

### Simulation data

The simulation data include:

* PSO tuning trials and selected parameters;
* controller validation results;
* herd configurations;
* scalability experiments across drone number, herd size, and spatial spread.

Statistical comparisons use scenario-level means. Individual PSO repetitions are also provided for transparency.

### Field data

The field data include:

* mission information;
* 187 online PSO re-optimisations;
* annotated scout frames and detections;
* localisation and body-axis estimates;
* surface coverage, flagged by whether the swarm was on the target herd (`on_target_herd`);
* realised and reference monitoring quality;
* drone–animal stand-off distances.

## Metadata

`metadata/variable_definitions.csv` describes the variables, units, types, and allowed values used in the released tables.

`metadata/experiment_metadata.csv` contains the main experimental settings, including objective parameters, PSO settings, camera parameters, detector settings, and field hardware.

## Reproducing the results

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd scripts
python3 reproduce_statistics.py
```

The scripts reproduce the statistical results and the main data-driven figures reported in the paper.
The definitions and parameters used in this archive follow the implementation and analysis used in the paper.
