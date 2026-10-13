"""Benchmark riproducibile dell'ambiente, senza addestrare o toccare checkpoint."""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np


class _ReferenceModules(importlib.abc.MetaPathFinder):
    modules = {
        "scheduler.device.utility.eft_utility",
        "scheduler.device.eft",
        "onpolicy.envs.secors.cost.energy",
        "onpolicy.envs.secors.core.encoding",
        "onpolicy.envs.secors.SECORS_env",
    }

    def __init__(self, root):
        self.root = Path(root)
        for name in self.modules:
            path = self.root.joinpath(*name.split('.')).with_suffix('.py')
            if not path.is_file():
                raise FileNotFoundError(path)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            source = self.root.joinpath(*fullname.split('.')).with_suffix('.py')
            return importlib.util.spec_from_file_location(fullname, source)
        return None


def _digest(env, result):
    payload = [item.tolist() if isinstance(item, np.ndarray) else item
               for item in result]
    queues = []
    for device in env._device_list:
        queues.append([(task.dag.dag_id, task.node_id, task.d_cpu,
                        getattr(task, 'scheduled_execution_start', None),
                        getattr(task, 'scheduled_execution_end', None),
                        getattr(task, 'scheduled_transmission_start', None),
                        getattr(task, 'scheduled_transmission_end', None))
                       for task in device.assigned_task_queue])
    payload.append(queues)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', action='append')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--reference-dir')
    parser.add_argument('--output')
    parser.add_argument('--compare')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats deve essere positivo')
    if args.reference_dir:
        sys.meta_path.insert(0, _ReferenceModules(args.reference_dir))
    from onpolicy.envs.secors.SECORS_env import SECORSEnv

    scenarios = args.scenario or [
        'Scenarios-Builded-4096/scenario_0',
        'Scenarios-Builded-8192/scenario_0',
    ]
    records = {}
    for scenario in scenarios:
        env = SECORSEnv(SimpleNamespace(scenario=scenario))
        rng = np.random.RandomState(args.seed)
        shape = (env.T, env.n_agents)
        policies = {
            'random': rng.randint(0, env.n_devices, size=shape),
            'sas': np.full(shape, env.n_sat),
            'ground': np.full(shape, env.n_devices - 1),
        }
        for policy, actions in policies.items():
            key = scenario + ':' + policy
            hashes = [_digest(env, env.reset())]
            for action in actions:
                hashes.append(_digest(env, env.step(action)))
            seconds = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                env.reset()
                for action in actions:
                    result = env.step(action)
                seconds.append(time.perf_counter() - start)
                # Verifica anche il riuso dei DAG dopo reset.
                assert _digest(env, result) == hashes[-1], key + ': reset differente'
            records[key] = dict(steps=env.T, agents=env.n_agents,
                                median_seconds=statistics.median(seconds),
                                seconds=seconds, digests=hashes)
            print(f'{key}: {statistics.median(seconds):.3f} s', flush=True)
        env.close()
    if args.compare:
        reference = json.loads(Path(args.compare).read_text(encoding='utf-8'))
        for key, record in records.items():
            old = reference[key]
            assert old['digests'] == record['digests'], key + ': risultati diversi'
            speedup = old['median_seconds'] / record['median_seconds']
            print(f'IDENTICO {key}: {speedup:.2f}x', flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
