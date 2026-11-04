"""Small adapter contract. Cloud SDK imports and calls stay behind CLI approval."""
from dataclasses import dataclass
import os
import socket
import subprocess
import sys
import time
from typing import Protocol


@dataclass
class Endpoint:
    url: str
    headers: dict


class Provider(Protocol):
    def deploy(self, run: dict, plan, store): ...
    def reconcile(self, run: dict, plan, store): ...
    def endpoint(self) -> Endpoint: ...
    def teardown(self, run: dict, store): ...


class ModalProvider:
    APP = "inference-lab"

    def __init__(self):
        self.sb = None

    def reconcile(self, run, plan, store):
        import modal
        # A pending creation is never automatically retried: an RPC may have succeeded remotely.
        if run["resource_id"]:
            self.sb = modal.Sandbox.from_id(run["resource_id"])
        elif run["deploy_intent"]:
            self.sb = modal.Sandbox.from_name(self.APP, "run-" + run["id"], include_terminated=True)
            run["resource_id"] = self.sb.object_id
            store.save(run)
        else:
            return False
        if self.sb.poll() is not None:
            raise ValueError("Remote sandbox has stopped. Create a new plan/approval; this run is never redeployed")
        return True

    def deploy(self, run, plan, store):
        import modal
        if run["deploy_intent"]:
            raise ValueError("Creation already attempted; reconcile instead of duplicating resources")
        run.update(deploy_intent=True, started_at=time.time(), deadline=time.time() + plan.runtime_s,
                   shutdown="unverified")
        store.save(run)  # Write-ahead intent before any provisioning call.
        app = modal.App.lookup(self.APP, create_if_missing=True)
        image = modal.Image.from_registry(plan.image).entrypoint([])
        args = ["vllm", "serve", plan.model, "--revision", plan.revision,
                "--tokenizer-revision", plan.revision, "--host", "0.0.0.0", "--port", "8000",
                "--dtype", plan.dtype, "--max-model-len", str(plan.max_model_len), "--max-num-seqs", str(plan.server_max_sequences),
                "--tensor-parallel-size", str(plan.gpu_count),
                "--gpu-memory-utilization", "0.80", "--seed", str(plan.seed), "--enforce-eager",
                "--enable-prefix-caching" if plan.prefix_caching else "--no-enable-prefix-caching"]
        self.sb = modal.Sandbox.create(
            *args, app=app, name="run-" + run["id"], image=image, gpu=f"{plan.gpu}:{plan.gpu_count}",
            cpu=(plan.cpu_cores, plan.cpu_cores), memory=(plan.ram_gib * 1024, plan.ram_gib * 1024), timeout=plan.runtime_s,
        )
        run["resource_id"] = self.sb.object_id
        store.save(run)

    async def gpu_telemetry(self):
        from .telemetry import parse_gpu_csv
        process = await self.sb.exec.aio(
            "nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits", timeout=5)
        text = await process.stdout.read.aio()
        return parse_gpu_csv(text)

    def endpoint(self):
        creds = self.sb.create_connect_token(port=8000)
        return Endpoint(creds.url.rstrip("/"), {"Authorization": "Bearer " + creds.token})

    def teardown(self, run, store):
        import modal
        if self.sb is None:
            if run["resource_id"]:
                self.sb = modal.Sandbox.from_id(run["resource_id"])
            elif run["deploy_intent"]:
                self.sb = modal.Sandbox.from_name(self.APP, "run-" + run["id"], include_terminated=True)
            else:
                run["shutdown"] = "not_started"
                store.save(run)
                return
        self.sb.terminate()
        for _ in range(10):
            if self.sb.poll() is not None:
                run.update(shutdown="verified", stopped_at=time.time())
                store.save(run)
                return
            time.sleep(1)
        raise RuntimeError("Shutdown not verified within 10 seconds")


class SimulatedProvider:
    def __init__(self):
        self.process = None
        self.port = None

    def deploy(self, run, plan, store):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        env = {**os.environ, "SIM_FAULT": plan.simulated_fault}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "inference_lab.simulator", str(self.port), str(plan.runtime_s)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        run.update(started_at=time.time(), deadline=time.time() + plan.runtime_s,
                   resource_id=f"simulated:{self.port}", shutdown="unverified")
        store.save(run)

    def reconcile(self, run, plan, store):
        # Simulator restarts are free. Previously persisted requests are never replayed.
        self.deploy(run, plan, store)
        return True

    def endpoint(self):
        return Endpoint(f"http://127.0.0.1:{self.port}", {})

    def teardown(self, run, store):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        run.update(shutdown="verified" if self.process else "unverified", stopped_at=time.time())
        store.save(run)


def provider_for(plan):
    return ModalProvider() if plan.provider == "modal" else SimulatedProvider()
