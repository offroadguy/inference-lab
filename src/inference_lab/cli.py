import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from .benchmark import execute, error_code
from .config import Plan
from .providers import provider_for
from .report import export, compare
from .store import Store


def require_approval(run, plan):
    plan.validate()
    if plan.provider == "simulated":
        return
    a = run.get("approval")
    if not a or a["fingerprint"] != plan.fingerprint():
        raise ValueError("Missing approval or materially changed plan; create a fresh plan and approve")
    if not run["deploy_intent"] and time.time() > a["expires_at"]:
        raise ValueError("Approval expired before deployment; approve again")
    if run.get("deadline") and time.time() >= run["deadline"] - 30:
        raise ValueError("Run lifetime exhausted; teardown, then create and approve a new plan")


def cleanup(provider, run, store):
    try:
        provider.teardown(run, store)
    except Exception as exc:
        run["shutdown"] = "unverified"
        store.event(run["id"], "cleanup_failure", error=error_code(exc))
        store.save(run)


def launch(store, run, command):
    plan = Plan(**run["plan"])
    require_approval(run, plan)
    if command == "deploy" and plan.provider == "simulated":
        raise ValueError("Use run for the simulator; deploy is for Modal only")
    if command == "benchmark" and not run["resource_id"]:
        raise ValueError("Deploy first, or use run")
    if run["status"] == "complete":
        print(export(store, run))
        return
    provider = provider_for(plan)
    store.recover(run["id"])
    leave_running = False
    try:
        run["status"] = "deploying"
        store.save(run)
        if run["resource_id"] or run["deploy_intent"]:
            provider.reconcile(run, plan, store)
        else:
            provider.deploy(run, plan, store)
        if command == "deploy":
            run["status"] = "deployed"
            store.save(run)
            leave_running = True
            print(f"Deployed {run['id']}; resource is billable until teardown or its lifetime expires.")
            return
        run["status"] = "running"
        store.save(run)
        asyncio.run(execute(run, plan, store, provider.endpoint()))
        run["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        run["status"] = "partial"
        store.event(run["id"], "execution_stopped", error=error_code(exc))
        print(f"Run stopped: {error_code(exc)}. Inspect the partial report.", file=sys.stderr)
    finally:
        if not leave_running:
            cleanup(provider, run, store)
            store.save(run)
            print(export(store, run))
    if run["status"] != "complete":
        raise ValueError("Run incomplete; completed and incomplete requests were preserved")


def main():
    signal.signal(signal.SIGTERM, _terminate)
    load_dotenv()
    parser = argparse.ArgumentParser(description="Local-first inference lab; no cloud spending without typed approval")
    parser.add_argument("--data", help="Local results directory (default data/)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path)
    p.add_argument("--provider", choices=["simulated", "modal"])
    p.add_argument("--workload", choices=["chat", "summary", "quality"])
    for cmd in ["approve", "deploy", "run", "benchmark", "resume", "teardown", "report"]:
        sub.add_parser(cmd).add_argument("run_id")
    sub.add_parser("list")
    p = sub.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = parser.parse_args()
    store = Store(args.data)
    try:
        if args.command == "plan":
            cfg = json.loads(args.config.read_text()) if args.config else {}
            for key in ("provider", "workload"):
                if getattr(args, key):
                    cfg[key] = getattr(args, key)
            plan = Plan(**cfg)
            plan.validate()
            run = store.create(plan)
            print(json.dumps({"run_id": run["id"], "plan": plan.to_dict(), "estimate": plan.estimate()}, indent=2))
        elif args.command == "list":
            print(json.dumps([{k: r.get(k) for k in ("id", "status", "shutdown")} for r in store.runs()], indent=2))
        elif args.command == "compare":
            print(json.dumps(compare(json.loads(args.baseline.read_text()),
                                     json.loads(args.candidate.read_text())), indent=2))
        else:
            with store.lock(args.run_id):
                run = store.get(args.run_id)
                plan = Plan(**run["plan"])
                if args.command == "approve":
                    plan.validate()
                    print(json.dumps({"plan": plan.to_dict(), "estimate": plan.estimate()}, indent=2))
                    if run["deploy_intent"] or run["resource_id"]:
                        raise ValueError("Cannot renew an already started run; create a new plan")
                    phrase = f"APPROVE {run['id']} USD {plan.max_spend_usd:.2f}"
                    if input(f"Type exactly: {phrase}\n> ") != phrase:
                        raise ValueError("Spending not approved")
                    run["approval"] = {"fingerprint": plan.fingerprint(), "approved_at": time.time(),
                                       "expires_at": time.time() + 3600, "max_spend_usd": plan.max_spend_usd,
                                       "runtime_s": plan.runtime_s}
                    store.save(run)
                    print("Approved exact plan for one deployment; approval expires in one hour.")
                elif args.command == "teardown":
                    cleanup(provider_for(plan), run, store)
                    print(export(store, run))
                    if run["shutdown"] == "unverified":
                        raise ValueError("Shutdown unverified; inspect the provider workspace")
                elif args.command == "report":
                    print(export(store, run))
                else:
                    launch(store, run, args.command)
    except (ValueError, TypeError, KeyError, OSError, EOFError) as exc:
        # Only locally generated validation strings are rendered; remote execution errors are class-only.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _terminate(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
