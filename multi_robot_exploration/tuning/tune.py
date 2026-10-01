"""
Tune the frontier formula's weights with CMA-ES on the headless Gazebo stack.

  ros2 run multi_robot_exploration tune_weights --num-envs 4

Each generation:
  1. CMA-ES proposes --popsize weight sets.
  2. Every weight set runs on the SAME --worlds random worlds (new worlds each
     generation). Scoring all candidates on identical worlds removes most of
     the world-to-world noise that swamps the signal otherwise.
  3. The current best estimate (the CMA-ES mean) and the heuristic
     (= frontier_coordinator) also run on those worlds, as a progress check.
  4. CMA-ES updates towards the higher-scoring weights.

Only 5 numbers are learned: crowding, other_half, unknown_around,
frontier_size, turning (log10-scaled, 0.01…1000); distance is fixed at 1
because only ratios matter. See features.py for the formula and episode.py
for the score. Outputs in --run-dir: best_weights.json (updated every
generation), generations.csv, evaluations.csv, cma_state.pkl (for --resume).
"""
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from multi_robot_exploration.tuning.features import HEURISTIC_WEIGHTS, TERMS

LEARNED = TERMS[1:]                   # distance is fixed at 1
LOG10_BOUNDS = (-2.0, 3.0)            # each learned weight in [0.01, 1000]
# Start: the heuristic's crowding/other_half (50), the new terms at 1 (turning 0.1)
X0 = np.array([math.log10(50.0), math.log10(50.0), 0.0, 0.0, -1.0])
SIGMA0 = 1.0                          # one decade in each direction
SIGMA_CONVERGED = 0.05                # stop when steps are < ~12 % weight changes
SIGMA_MAX = 2.0                       # never sample wider than two decades
MAX_FAILED_FRACTION = 0.5             # a generation with more failed episodes is re-run once
MAX_POOL_RESTARTS = 3                 # per generation, after workers die


def to_weights(x):
    """log10 vector of the learned weights -> {term: weight}."""
    w = {'distance': 1.0}
    w.update({t: round(10.0 ** float(v), 6) for t, v in zip(LEARNED, x)})
    return w


class CMAES:
    """
    (mu/mu_w, lambda)-CMA-ES with default parameters, following N. Hansen,
    "The CMA Evolution Strategy: A Tutorial" (arXiv:1604.00772), minimising.
    Out-of-bounds samples are clipped to the box before evaluation.
    """

    def __init__(self, x0, sigma0, popsize=None, bounds=None, seed=0):
        n = len(x0)
        self.n = n
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.mean = np.array(x0, dtype=float)
        self.sigma = float(sigma0)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.bounds = bounds
        self.gen = 0
        self.rng = np.random.default_rng(seed)

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        xs = self.mean + self.sigma * (z * self.D) @ self.B.T
        if self.bounds is not None:
            xs = np.clip(xs, *self.bounds)
        return xs

    def tell(self, xs, fitness):
        xs = np.asarray(xs)
        order = np.argsort(fitness)
        sel = xs[order[:self.mu]]
        old = self.mean
        self.mean = self.weights @ sel
        y_w = (self.mean - old) / self.sigma
        c_inv_sqrt = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * c_inv_sqrt @ y_w
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
                / self.chin) < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w
        art = (sel - old) / self.sigma
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * art.T @ np.diag(self.weights) @ art)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chin - 1))
        self.sigma = min(self.sigma, SIGMA_MAX)
        self.C = (self.C + self.C.T) / 2
        eigval, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(eigval, 1e-20))
        self.gen += 1


# ── parallel episode workers (one sim slot each) ─────────────────────────
_runner = None


def _on_sigterm(*_):
    """Stop this worker's sim, then exit hard. sys.exit() is not enough: the
    process-pool worker loop catches SystemExit and keeps taking jobs."""
    try:
        if _runner is not None:
            _runner.sim.stop()
    finally:
        os._exit(0)


def _init_worker(slots, first_instance, kwargs):
    global _runner
    # Ctrl+C is handled by the main process. A no-op handler (not SIG_IGN, which
    # the sim's processes would inherit) keeps episodes from being interrupted.
    signal.signal(signal.SIGINT, lambda *_: None)
    signal.signal(signal.SIGTERM, _on_sigterm)
    from multi_robot_exploration.tuning.episode import EpisodeRunner
    slot = slots.get()
    time.sleep(5.0 * (slot - first_instance))   # stagger the CPU-heavy first launches
    _runner = EpisodeRunner(instance_id=slot, **kwargs)


def _run_job(job):
    """job = (tag, weights dict, seed) -> (tag, weights, stats or None). Never raises."""
    tag, weights, seed = job
    slot = _runner.sim.instance_id
    for attempt in range(2):
        try:
            st = _runner.run(weights, seed)
            print(f"  [slot {slot}] {tag:9s} world {seed:>10}: score {st['score']:6.3f}, "
                  f"{st['end_reason']} after {st['sim_time_s']:.0f} s, "
                  f"{st['failed_goals']} failed goals", flush=True)
            return tag, weights, st
        except Exception as e:   # the runner has already torn its sim down
            print(f'  [slot {slot}] {tag} world {seed}: {type(e).__name__}: {e} '
                  f'(attempt {attempt + 1}/2)', flush=True)
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
    return tag, weights, None


class SimPool:
    """
    Worker processes, one sim slot each. Unlike multiprocessing.Pool, a worker
    that dies (crash, OOM kill) is detected instead of hanging forever: the
    pool is rebuilt, orphaned sims are killed, and unfinished jobs re-run.
    """

    def __init__(self, num_envs, first_instance, runner_kwargs):
        self.num_envs = num_envs
        self.first_instance = first_instance
        self.kwargs = runner_kwargs
        self.log_dir = runner_kwargs['log_dir']
        self.ctx = mp.get_context('spawn')
        self.ex = None
        self._start()

    def _start(self):
        from multi_robot_exploration.tuning.sim_manager import kill_stale_sims
        os.makedirs(self.log_dir, exist_ok=True)
        n = kill_stale_sims(self.log_dir)
        if n:
            print(f'Killed {n} leftover simulation(s) from a previous crash.', flush=True)
        slots = self.ctx.Queue()
        for i in range(self.num_envs):
            slots.put(self.first_instance + i)
        self.ex = ProcessPoolExecutor(self.num_envs, mp_context=self.ctx,
                                      initializer=_init_worker,
                                      initargs=(slots, self.first_instance, self.kwargs))

    def close(self):
        """Stop workers now (they stop their sims), then kill anything left."""
        from multi_robot_exploration.tuning.sim_manager import kill_stale_sims
        if self.ex is None:
            return
        procs = list(getattr(self.ex, '_processes', {}).values())
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        self.ex.shutdown(wait=False, cancel_futures=True)
        self.ex = None
        kill_stale_sims(self.log_dir)

    def map(self, jobs):
        results = [None] * len(jobs)
        pending = list(range(len(jobs)))
        restarts = 0
        while pending:
            futs = {self.ex.submit(_run_job, jobs[i]): i for i in pending}
            pending = []
            try:
                for f in as_completed(futs):
                    results[futs[f]] = f.result()
            except BrokenProcessPool:
                for f, i in futs.items():
                    if f.done() and f.exception() is None:
                        results[i] = f.result()
                    else:
                        pending.append(i)
                restarts += 1
                print(f'WARNING: a worker process died; restarting the pool '
                      f'({restarts}/{MAX_POOL_RESTARTS}), {len(pending)} episodes to redo', flush=True)
                self.close()
                if restarts > MAX_POOL_RESTARTS:
                    for i in pending:
                        results[i] = (jobs[i][0], jobs[i][1], None)
                    pending = []
                self._start()
        return results


def _atomic_write(path, data, binary=False):
    """Write via a temp file + rename, so a crash never leaves a half-written file."""
    tmp = path + '.tmp'
    with open(tmp, 'wb' if binary else 'w') as fh:
        fh.write(data)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--num-envs', type=int, default=1,
                   help='parallel simulations (each needs ~2 CPU cores and ~2 GB RAM)')
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--popsize', type=int, default=8, help='weight sets per generation')
    p.add_argument('--worlds', type=int, default=6,
                   help='random worlds per weight set (same weights on the same world can '
                        'score very differently between runs; more worlds = less noise)')
    p.add_argument('--max-episode-sim-s', type=float, default=300.0)
    p.add_argument('--no-goal-fix', action='store_true',
                   help='use raw frontier centroids as goals (A/B test)')
    p.add_argument('--no-baseline', action='store_true',
                   help='skip the per-generation heuristic + current-mean runs (faster, less insight)')
    p.add_argument('--first-instance', type=int, default=0,
                   help='first sim slot (ROS domain 40+slot, Gazebo port 11445+slot)')
    p.add_argument('--rtf', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run-dir', default=os.path.expanduser(
        f'~/swarm_tuning_runs/{time.strftime("%Y%m%d_%H%M%S")}'))
    p.add_argument('--resume', action='store_true', help='continue from --run-dir/cma_state.pkl')
    args = p.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    state_path = os.path.join(args.run_dir, 'cma_state.pkl')
    if args.resume:
        if not os.path.exists(state_path):
            sys.exit(f'--resume: no {state_path} (no generation finished yet?). '
                     'Start without --resume.')
        with open(state_path, 'rb') as fh:
            es = pickle.load(fh)
        print(f'Resumed at generation {es.gen}', flush=True)
    elif os.path.exists(state_path):
        sys.exit(f'{args.run_dir} already has a tuning state. Add --resume to continue it, '
                 'or use a new --run-dir.')
    else:
        es = CMAES(X0, SIGMA0, popsize=args.popsize, bounds=LOG10_BOUNDS, seed=args.seed)
    print(f'Run dir: {args.run_dir}', flush=True)

    runner_kwargs = dict(rtf=args.rtf, max_episode_sim_s=args.max_episode_sim_s,
                         log_dir=os.path.join(args.run_dir, 'sim_logs'),
                         goal_fix=not args.no_goal_fix)
    pool = SimPool(args.num_envs, args.first_instance, runner_kwargs)

    gen_csv = os.path.join(args.run_dir, 'generations.csv')
    eval_csv = os.path.join(args.run_dir, 'evaluations.csv')
    if not os.path.exists(gen_csv):
        with open(gen_csv, 'w', newline='') as fh:
            csv.writer(fh).writerow(
                ['generation', 'sigma', 'candidates_mean_score', 'candidates_best_score',
                 'mean_weights_score', 'heuristic_score', 'failed_episodes']
                + [f'w_{t}' for t in TERMS])
    if not os.path.exists(eval_csv):
        with open(eval_csv, 'w', newline='') as fh:
            csv.writer(fh).writerow(['generation', 'tag', 'world_seed', 'score', 'end_reason',
                                     'explored_m2', 'sim_time_s', 'decisions', 'failed_goals']
                                    + [f'w_{t}' for t in TERMS])
    try:
        while es.gen < args.generations and es.sigma > SIGMA_CONVERGED:
            g = es.gen
            seeds = np.random.default_rng(args.seed * 100003 + g).integers(0, 2**31 - 1, args.worlds)
            xs = es.ask()
            jobs = [(f'cand{i}', to_weights(x), int(s)) for i, x in enumerate(xs) for s in seeds]
            if not args.no_baseline:
                jobs += [('mean', to_weights(es.mean), int(s)) for s in seeds]
                jobs += [('heuristic', HEURISTIC_WEIGHTS, int(s)) for s in seeds]
            t0 = time.time()
            for attempt in range(2):
                print(f'gen {g}: running {len(jobs)} episodes on {args.num_envs} sims '
                      f'(one line per finished episode)...', flush=True)
                results = pool.map(jobs)
                n_failed = sum(1 for r in results if r[2] is None)
                if n_failed <= MAX_FAILED_FRACTION * len(jobs):
                    break
                print(f'WARNING: {n_failed}/{len(jobs)} episodes failed in generation {g}.',
                      flush=True)
            else:
                sys.exit(
                    f'Too many failed episodes twice in generation {g}; stopping without updating '
                    f'the weights (rerun with --resume after fixing). Check '
                    f'{runner_kwargs["log_dir"]}/sim_<slot>.log. Common causes: too many --num-envs '
                    'for the machine, or ROS discovery failing without a network '
                    '(fix: sudo ip link set lo multicast on).')

            scores = {}
            with open(eval_csv, 'a', newline='') as fh:
                w = csv.writer(fh)
                for tag, weights, st in results:
                    if st is None:
                        continue
                    scores.setdefault(tag, []).append(st['score'])
                    w.writerow([g, tag, st['world_seed'], round(st['score'], 3), st['end_reason'],
                                round(st['explored_m2'], 2), round(st['sim_time_s'], 1),
                                st['decisions'], st['failed_goals']]
                               + [weights.get(t, 0.0) for t in TERMS])

            # A candidate with no successful episode counts as the worst score seen
            worst = min(v for vals in scores.values() for v in vals)
            cand = [float(np.mean(scores.get(f'cand{i}', [worst]))) for i in range(len(xs))]
            es.tell(xs, [-c for c in cand])            # CMA-ES minimises
            mean_s = np.mean(scores['mean']) if 'mean' in scores else float('nan')
            heur_s = np.mean(scores['heuristic']) if 'heuristic' in scores else float('nan')
            best_w = to_weights(es.mean)

            with open(gen_csv, 'a', newline='') as fh:
                csv.writer(fh).writerow([g, round(es.sigma, 4), round(float(np.mean(cand)), 3),
                                         round(float(np.max(cand)), 3), round(float(mean_s), 3),
                                         round(float(heur_s), 3), n_failed]
                                        + [best_w[t] for t in TERMS])
            _atomic_write(os.path.join(args.run_dir, 'best_weights.json'),
                          json.dumps(best_w, indent=2))
            _atomic_write(state_path, pickle.dumps(es), binary=True)
            print(f'gen {g}: candidates {np.mean(cand):.3f} (best {np.max(cand):.3f}) | '
                  f'current mean {mean_s:.3f} vs heuristic {heur_s:.3f} on the same worlds | '
                  f'sigma {es.sigma:.3f} | {(time.time() - t0) / 60:.0f} min | '
                  f'weights {best_w}', flush=True)
        reason = 'converged' if es.sigma <= SIGMA_CONVERGED else 'generation limit reached'
        print(f'\nDone ({reason}). Weights: {os.path.join(args.run_dir, "best_weights.json")}',
              flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted: the current generation is discarded; '
              'continue later with --resume.', flush=True)
    finally:
        print('Stopping simulations...', flush=True)
        pool.close()


if __name__ == '__main__':
    main()
