# Pipeline Scheduling in Satellite Edge Computing with MAPPO and HEFT

> Multi-agent reinforcement learning (MAPPO) with HEFT sequencing for scheduling image-processing pipelines in LEO satellite edge computing, with per-satellite ground-link windows derived from real orbits.

**Bachelor's Thesis in Computer Science**, Sapienza University of Rome
**Author:** Gianluca Nerattini · **Advisor:** Prof. Emiliano Casalicchio · **Academic Year:** 2025/2026

---

## Overview

Earth observation satellites acquire far more data than they can send to the ground. Satellite Edge Computing (SEC) moves part of the processing into orbit, so that only the results need to be downlinked.

This work studies the scheduling of **image-processing pipelines** on a small LEO constellation. Each pipeline is a DAG of tasks belonging to four missions (IS, CD, GC, CF), and every task has to be placed on one of three kinds of node:

| Node | Role | Characteristics |
|---|---|---|
| SAT | ordinary satellite | slow onboard processor, always reachable through inter-satellite links |
| SAS | satellite server | more powerful processor, in orbit |
| GND | ground station | about 14 times faster than a satellite, reachable **only while the link with the ground is open** |

The goal is to minimize two costs together: the total completion time $C_T$ and the energy $C_E$ spent by the nodes in orbit.

The ground station is the key difficulty. Offloading to it is usually the best choice, but if the generating satellite is out of visibility the transfer has to wait for the next window, which can be tens of seconds away. A placement that would save time becomes the most expensive one.

## Contributions

1. **MAPPO + HEFT scheduler.** A multi-agent PPO policy decides *where* each task runs; an insertion-based HEFT scheduler with a global view over all DAGs decides *when*, filling the idle slots left by the precedence constraints.
2. **Per-satellite connectivity windows from real orbits.** Instead of a single online/offline state for the whole constellation, every satellite has its own ground-link windows, computed from real Starlink TLE data with Skyfield/SGP4. A scenario builder generates families of scenarios from these windows.
3. **Action mask on ground-link waits.** Without it, the trained policy abandons the ground station entirely whenever a generating satellite is out of visibility, even when offloading would save most of the cost. The mask removes from exploration the placements that would force a wait on a closed link.
4. **Generalization across scenarios.** Comparison of four ways to obtain a single policy for unseen scenarios: FedAvg, greedy soup, lambda scaling and domain randomization.
5. **Training on an abstract MDP.** Training does not need the orbital simulator: it runs on the same queues, costs and HEFT sequencing, so independent environments can run in parallel on a cluster.

## Architecture

The scheduling of a set of pipelines goes through three phases:

1. **Generation.** The generating satellites create their pipelines.
2. **Scheduling.** The policy assigns every task to a node; HEFT then orders the queue of each node in a global topological order, taking into account transfers and ground-link windows.
3. **Execution.** One runtime per node executes its queue in the discrete-event simulator, respecting the DAG precedences and deferring the transfers that fall outside a link window.

The MAPPO policy uses parameter sharing (one actor and one critic for all agents) and centralized training with decentralized execution: the actor sees a local observation, the critic sees the global state. Only the actor is needed at inference time.

## Experimental setup and key results

**Setup.** Seven nodes (five satellites, three of which generate pipelines, one satellite server and one ground station), the largest workload of the case study (pipeline P-8192), ten training scenarios and ten unseen test scenarios built from real orbits with disjoint seeds. Each model is trained for 10,000 episodes.

**Results.** Median variation with respect to the baseline (every pipeline processed on its own generating satellite: $C_T$ = 25,473 ms, $C_E$ = 76,013 J), computed scenario by scenario on the ten test scenarios:

| Method | $C_T$, no mask | $C_T$, with mask | $C_E$, no mask | $C_E$, with mask |
|---|---:|---:|---:|---:|
| Specialist (one model per scenario) | -47.8% | -76.9% | -27.9% | -88.5% |
| FedAvg | +2.4% | -73.2% | +0.0% | -71.9% |
| Greedy soup | -51.4% | -73.4% | -27.2% | -78.3% |
| Lambda scaling | -50.4% | -73.4% | -26.8% | -78.3% |
| Domain randomization | -41.5% | -74.6% | -17.8% | -68.7% |

Without the mask, plain weight averaging (FedAvg) stays at the level of the baseline. With the mask, every method reduces both costs: on $C_T$ the generalized policies come within a few points of the specialists, while on $C_E$ the gap is larger.

## Repository structure

| Path | Role |
|---|---|
| `main.py` | Single entry point; the mode is chosen in `config/config.json5` |
| `architecture/` | The discrete-event simulator: satellites, topology from TLE, task execution, routing |
| `devices/` | Wrappers for the three node types (SAT, SAS, GND) |
| `mission/` | Missions, pipelines as DAGs, and the three-phase dispatcher |
| `scheduler/` | Placement policies (baseline, random, MAPPO) and the HEFT scheduler |
| `onpolicy/` | MAPPO: actor, critic, buffer, runner and the training environment |
| `training/` | Training launcher, abstract devices and the generalization methods |
| `evaluation/` | Comparison of candidate policies across scenarios, with HTML reports |
| `scenario/` | Scenario loading and the scenario builder based on orbital windows |
| `simulation_utility/` | Runtime per node, result tables, CSV export and Gantt charts |
| `config/` | Main configuration, scenarios, pipelines, hyperparameters and evaluation suites |

Code comments are in Italian.

## Getting started

### Requirements

- **Python 3.10**
- Internet access on the first run, to download the satellite TLE data from [CelesTrak](https://celestrak.org)

### Installation

```bash
git clone https://github.com/COSMOSatSim/Pipeline-Scheduling-in-Satellite-Edge-Computing-with-MAPPO-and-HEFT.git
cd Pipeline-Scheduling-in-Satellite-Edge-Computing-with-MAPPO-and-HEFT
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a GPU machine, install the CUDA build of `torch` before the other requirements (see the comments in `requirements.txt`).

On Windows, set `PYTHONUTF8=1` before running (`$env:PYTHONUTF8="1"` in PowerShell).

### Running

Everything goes through `main.py`. The field `CONTROL_PANEL.run` in `config/config.json5` selects one of five modes; the sections that do not belong to the chosen mode are ignored:

| `run` | What it does |
|---|---|
| `simulation` | a single simulation on the scenario in `SIMULATION.SCENARIO` |
| `training` | trains a MAPPO policy on the scenario in `TRAINING` |
| `generalization` | builds a generalized policy with the method in `GENERALIZATION` |
| `evaluation` | compares several candidate policies on a suite of scenarios (`config/evaluation/`) |
| `scenarios` | generates new scenarios from the orbital windows (`SCENARIO_BUILDER`) |

```bash
python main.py
```

For a short test run of the training, `--ep_override` replaces the number of episodes without editing the configuration:

```bash
python main.py --ep_override 100
```

**First run.** The simulation needs the TLE data and the constellation configurations in `data/`. Set `Build_Configurations: true` in `config/config.json5` for one run, then set it back to `false`. Training does not need them.

Trained models and results are written under `data/`, which is not part of the repository.

## Acknowledgements

The simulator is built on top of the SECMotionModel framework developed in Prof. Casalicchio's group (originally by Vincenzo Salvatore). The problem setting, the image-processing case study and the hardware parameters follow Q. Jiang et al., *Efficient On-Orbit Remote Sensing Imagery Processing via Satellite Edge Computing Resource Scheduling Optimization*, IEEE Transactions on Geoscience and Remote Sensing, 2025. The MAPPO implementation in `onpolicy/` is adapted from the code released with C. Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022.

The MAPPO + HEFT scheduler on per-satellite connectivity windows, the action mask, the scenario builder, the generalization methods and the experimental evaluation are the contribution of this thesis.

## License

Released under the GNU General Public License v3.0, see [`LICENSE`](LICENSE).
