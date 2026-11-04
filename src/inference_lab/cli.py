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
from .datasets import create_run, validate_manifest
from .models import ALL_PROFILES, profile_config


def require_approval(run, plan):
    plan.validate()
    if run.get("manifest"):
        validate_manifest(run["manifest"])
    if plan.workload in {"tokens", "quality"} and not run.get("manifest"):
        raise ValueError("This workload needs a prepared manifest; create a new plan")
    if plan.provider == "simulated":
        return
    a = run.get("approval")
    if not a or a["fingerprint"] != plan.fingerprint():
        raise ValueError("Missing approval or materially changed plan; create a fresh plan and approve")
    if run.get("manifest") and a.get("manifest_sha256") != run["manifest"]["sha256"]:
        raise ValueError("Prepared workload is not covered by this approval")
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
    import platform
    from importlib.metadata import version
    run["environment"] = {"python": platform.python_version(), "system": platform.system(),
                          "machine": platform.machine(), "httpx": version("httpx"),
                          "modal": version("modal"), "server_image": plan.image}
    from .reference import preflight
    preflight(plan)
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
        asyncio.run(execute(run, plan, store, provider.endpoint(), provider))
        if plan.reference_s and "reference" not in run["stages"]:
            from .reference import run_reference
            asyncio.run(run_reference(plan, run, store, provider.endpoint()))
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
    p = sub.add_parser("preflight")
    p.add_argument("--config", type=Path, required=True)
    p = sub.add_parser("onboard")
    p.add_argument("model")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", choices=["L4", "A100-40GB", "A100-80GB"], default="A100-40GB")
    p.add_argument("--gpu-count", type=int, choices=[1,2,4], default=1)
    p.add_argument("--hf-secret-name")
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path)
    p.add_argument("--profile", choices=list(ALL_PROFILES))
    p.add_argument("--provider", choices=["simulated", "modal"])
    p.add_argument("--workload", choices=["chat", "summary", "quality", "tokens"])
    for cmd in ["approve", "deploy", "run", "benchmark", "resume", "teardown", "report"]:
        sub.add_parser(cmd).add_argument("run_id")
    sub.add_parser("list")
    sub.add_parser("models")
    p = sub.add_parser("compare")
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p = sub.add_parser("sweep-plan")
    p.add_argument("--profiles", default="qwen-0.5b,qwen-3b,qwen-7b,qwen-32b")
    p.add_argument("--rates", default="1,2,4")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-spend-usd", type=float, required=True)
    for cmd in ("sweep-approve", "sweep-run", "sweep-report"):
        sub.add_parser(cmd).add_argument("campaign_id")
    p = sub.add_parser("reconcile-cost")
    p.add_argument("run_id")
    p.add_argument("statement", type=Path)
    args = parser.parse_args()
    store = Store(args.data)
    try:
        if args.command == "preflight":
            from .onboarding import preflight_report
            print(json.dumps(preflight_report(Plan(**json.loads(args.config.read_text()))), indent=2))
        elif args.command == "onboard":
            import os
            from .onboarding import inspect_model, HARDWARE
            if args.output.exists():
                raise ValueError("Output already exists; choose a new path")
            try:
                metadata = inspect_model(args.model, args.revision, token=os.getenv("HF_TOKEN") or False)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                raise ValueError(f"Hugging Face inspection failed ({type(exc).__name__}); check access and metadata") from None
            plan = Plan(provider="modal", model=args.model, revision=metadata['revision'],
                        model_metadata=metadata, gpu=args.gpu, gpu_count=args.gpu_count,
                        gpu_usd_s=HARDWARE[args.gpu][1], cpu_cores=4, ram_gib=32,
                        hf_secret_name=args.hf_secret_name, workload="tokens", max_spend_usd=10,
                        max_model_len=2048, server_max_sequences=4,
                        input_lengths=(128,), output_lengths=(32,), requests=60, rate=1)
            from dataclasses import replace
            replace(plan, provider="simulated").validate()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(plan.to_dict(), indent=2))
            print(json.dumps({"config":str(args.output), "model":plan.model, "revision":plan.revision,
                              "status":"metadata_checked_not_gpu_validated", "estimate":plan.estimate()}, indent=2))
        elif args.command == "sweep-plan":
            from .campaigns import create
            campaign = create(store, args.profiles.split(","), [float(x) for x in args.rates.split(",")],
                              args.repetitions, json.loads(args.config.read_text()), args.max_spend_usd)
            print(json.dumps(campaign, indent=2))
        elif args.command in {"sweep-approve", "sweep-run", "sweep-report"}:
            from .campaigns import load, approve, execute_campaign, write_report
            with store.lock(args.campaign_id):
                campaign = load(store, args.campaign_id)
                if args.command == "sweep-approve":
                    print(json.dumps({"campaign": campaign, "plans": [
                        {"run_id": rid, "plan": store.get(rid)["plan"],
                         "estimate": Plan(**store.get(rid)["plan"]).estimate()}
                        for rid in campaign["run_ids"]]}, indent=2))
                    phrase = f"APPROVE SWEEP {campaign['id']} USD {campaign['max_spend_usd']:.2f}"
                    approve(store, campaign, input(f"Type exactly: {phrase}\n> "))
                elif args.command == "sweep-run":
                    execute_campaign(store, campaign)
                else:
                    print(json.dumps(write_report(store, campaign), indent=2))
        elif args.command == "plan":
            cfg = profile_config(args.profile) if args.profile else {}
            cfg.update(json.loads(args.config.read_text()) if args.config else {})
            for key in ("provider", "workload"):
                if getattr(args, key):
                    cfg[key] = getattr(args, key)
            plan = Plan(**cfg)
            plan.validate()
            run = create_run(store, plan)
            print(json.dumps({"run_id": run["id"], "plan": plan.to_dict(), "estimate": plan.estimate()}, indent=2))
        elif args.command == "models":
            print(json.dumps([p.to_dict() for p in ALL_PROFILES.values()], indent=2))
        elif args.command == "list":
            print(json.dumps([{k: r.get(k) for k in ("id", "status", "shutdown")} for r in store.runs()], indent=2))
        elif args.command == "compare":
            print(json.dumps(compare(json.loads(args.baseline.read_text()),
                                     json.loads(args.candidate.read_text())), indent=2))
        else:
            with store.lock(args.run_id):
                run = store.get(args.run_id)
                plan = Plan(**run["plan"])
                if args.command == "reconcile-cost":
                    from .billing import reconcile
                    print(json.dumps(reconcile(store, run, args.statement), indent=2))
                    print(export(store, run))
                elif args.command == "approve":
                    plan.validate()
                    print(json.dumps({"plan": plan.to_dict(), "estimate": plan.estimate()}, indent=2))
                    if run["deploy_intent"] or run["resource_id"]:
                        raise ValueError("Cannot renew an already started run; create a new plan")
                    phrase = f"APPROVE {run['id']} USD {plan.max_spend_usd:.2f}"
                    if input(f"Type exactly: {phrase}\n> ") != phrase:
                        raise ValueError("Spending not approved")
                    run["approval"] = {"fingerprint": plan.fingerprint(), "approved_at": time.time(),
                                       "expires_at": time.time() + 3600, "max_spend_usd": plan.max_spend_usd,
                                       "runtime_s": plan.runtime_s, "manifest_sha256": run.get("manifest", {}).get("sha256")}
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
