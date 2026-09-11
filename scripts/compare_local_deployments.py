"""Locked paired gameplay test using the actual local GUI decision engine."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone
from minesweeper_ai.deployment_compare import _mcnemar_exact_p
from minesweeper_ai.game import MinesweeperGame
from minesweeper_ai.local_selfplay import SPECS, _continue
from minesweeper_ai.replay_deployment import load_replay_deployment

_AGENTS = None

def initialize(reference, candidate):
    global _AGENTS
    _AGENTS = (load_replay_deployment(reference, device="cpu", cpu_threads=1)[0],
               load_replay_deployment(candidate, device="cpu", cpu_threads=1)[0])

def paired(task):
    mode, index, master_seed, first_zero = task
    seed = int.from_bytes(hashlib.sha256(f"local-paired-test:{master_seed}:{mode}:{index}".encode()).digest()[:8], "big")
    width, height, mines = SPECS[mode]
    results = {}
    # Alternate run order to reduce systematic warm-cache timing bias.
    order = (0, 1) if index % 2 == 0 else (1, 0)
    for agent_index in order:
        game = MinesweeperGame(width, height, mines, seed=seed, first_click_zero=first_zero)
        start = time.perf_counter()
        won, guesses = _continue(game, _AGENTS[agent_index])
        results["reference" if agent_index == 0 else "candidate"] = dict(won=won, guesses=guesses,
                                                            seconds=time.perf_counter()-start)
    return dict(mode=mode,index=index,seed=seed,**results)

def read_completed(path):
    records = []
    seen = set()
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = (item["mode"], item["index"])
        if key in seen:
            raise ValueError("Duplicate saved pair")
        seen.add(key)
        if any(not isinstance(item[label]["won"], bool) for label in ("reference", "candidate")):
            raise ValueError("Incomplete pair must not be counted as a loss")
        records.append(item)
    return records


def add_result(counts, result):
    ref, cand = result["reference"], result["candidate"]
    tally = counts[result["mode"]]
    tally["games"] += 1
    for label, item in (("reference", ref), ("candidate", cand)):
        tally[label+"_wins"] += item["won"]
        tally[label+"_seconds"] += item["seconds"]
        tally[label+"_guesses"] += item["guesses"]
    tally["candidate_only"] += cand["won"] and not ref["won"]
    tally["reference_only"] += ref["won"] and not cand["won"]


def checkpoint_hash(config_path):
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return hashlib.sha256((config_path.parent / raw["checkpoint"]).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--games",type=int,default=1000)
    parser.add_argument("--modes",nargs="+",choices=SPECS,default=["expert","evil"])
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--safe-first-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.games<1 or args.workers<1 or len(set(args.modes))!=len(args.modes):
        raise ValueError("Invalid paired test counts or modes")
    expected=dict(reference=str(args.reference),candidate=str(args.candidate),seed=args.seed,
                  games_per_mode=args.games,modes=args.modes,first_click_zero=not args.safe_first_only,
                  reference_config_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_config_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  reference_checkpoint_sha256=checkpoint_hash(args.reference),
                  candidate_checkpoint_sha256=checkpoint_hash(args.candidate))
    manifest_path=args.output/"manifest.json"
    records=[]
    if args.resume:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        for key,value in expected.items():
            if key in manifest and manifest[key]!=value:
                raise ValueError(f"Cannot resume a changed test: {key}")
        records=read_completed(args.output/"pairs.jsonl")
        manifest.setdefault("resumes",[]).append(dict(completed_before=len(records),workers=args.workers,
                                            time_utc=datetime.now(timezone.utc).isoformat()))
        manifest.update(expected)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        manifest=dict(**expected,workers=args.workers,completed=False,
                      policy="No training or tuning on paired test boards; report all modes.")
    counts={mode:Counter() for mode in args.modes}
    for item in records:
        if item["mode"] not in args.modes or not 0 <= item["index"] < args.games:
            raise ValueError("Saved pair is outside the registered test")
        expected_seed=int.from_bytes(hashlib.sha256(
            f"local-paired-test:{args.seed}:{item['mode']}:{item['index']}".encode()).digest()[:8],"big")
        if item["seed"]!=expected_seed:
            raise ValueError("Saved board seed does not match the registered test")
        add_result(counts,item)
    seen={(item["mode"],item["index"]) for item in records}
    tasks=[(mode,index,args.seed,not args.safe_first_only) for index in range(args.games) for mode in args.modes
           if (mode,index) not in seen]
    start=time.perf_counter()
    done=len(records)
    total=args.games*len(args.modes)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(dict(resuming=args.resume,completed=done,total=total,pending=len(tasks))),flush=True)
    pending=tasks
    errors=[]
    with (args.output/"pairs.jsonl").open("a" if args.resume else "x",encoding="utf-8") as handle:
        # A runtime failure is retried once in a fresh process. It is never an outcome.
        for attempt in range(2):
            if not pending:
                break
            failed=[]
            with ProcessPoolExecutor(args.workers if attempt==0 else min(4,args.workers),
                mp_context=multiprocessing.get_context("spawn"),initializer=initialize,
                initargs=(str(args.reference.resolve()),str(args.candidate.resolve()))) as pool:
                futures={pool.submit(paired,task):task for task in pending}
                for future in as_completed(futures):
                    task=futures[future]
                    try:
                        result=future.result()
                    except Exception as error:
                        failed.append(task)
                        item=dict(mode=task[0],index=task[1],attempt=attempt+1,
                                  exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                        errors.append(item)
                        with (args.output/"errors.jsonl").open("a",encoding="utf-8") as err:
                            err.write(json.dumps(item)+"\n")
                        if len(errors) <= 3:
                            print(json.dumps(dict(runtime_error=True,mode=task[0],index=task[1],retry_pending=True)),flush=True)
                        continue
                    add_result(counts,result)
                    handle.write(json.dumps(result,separators=(",",":"))+"\n")
                    handle.flush()
                    done+=1
                    if done%max(1,total//20)==0 or done==total:
                        print(json.dumps(dict(completed=done,total=total,
                                              seconds=round(time.perf_counter()-start,2))),flush=True)
            pending=failed
            if failed:
                print(json.dumps(dict(retry_count=len(failed),attempt=attempt+1)),flush=True)
    for tally in counts.values():
        tally["mcnemar_exact_p"]=_mcnemar_exact_p(tally["reference_only"],tally["candidate_only"])
        tally["delta_pp"]=100*(tally["candidate_wins"]-tally["reference_wins"])/tally["games"] if tally["games"] else None
    manifest.update(completed=done==total,summary={key:dict(value) for key,value in counts.items()},
                    last_invocation_seconds=time.perf_counter()-start,
                    last_invocation_runtime_errors=len(errors),pending=[list(task[:2]) for task in pending])
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    if pending:
        raise SystemExit("Paired test incomplete; runtime errors were not counted as losses")


if __name__=="__main__":
    main()
