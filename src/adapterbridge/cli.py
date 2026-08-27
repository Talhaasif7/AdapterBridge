"""CLI entry points using Typer and Rich formatting with next-gen diagnostic tree UI."""

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from adapterbridge.core.inspector import AdapterInspector
from adapterbridge.core.schema_sync import SchemaSyncManager
from adapterbridge.models.report import IssueSeverity

app = typer.Typer(
    name="adapterbridge",
    help="AdapterBridge: LoRA Checkpoint & Config Compatibility Engine for Enterprise Inference Runtimes.",
    add_completion=False,
)

console = Console()


@app.command(name="check")
def check_cmd(
    path: str = typer.Option(..., "--path", "-p", help="Path to checkpoint directory"),
    target: str = typer.Option(..., "--target", "-t", help="Target serving engine (e.g. vllm, sglang@0.2.0, ollama)"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, sarif, markdown, pr-comment"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional output file destination"),
    canary: bool = typer.Option(False, "--canary", help="Execute zero-GPU activation canary probe"),
    chat_test: bool = typer.Option(False, "--chat-test", help="Execute chat completion round-trip rendering test"),
):
    """Static linter, dynamic schema validator, and canary probing engine for LoRA checkpoints."""
    try:
        inspector = AdapterInspector(checkpoint_path=path, target_engine=target)
        report = inspector.run_diagnostics()

        canary_res = None
        if canary:
            canary_res = inspector.run_canary_probe()

        chat_res = None
        if chat_test:
            chat_res = inspector.run_chat_test()

    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

    if format == "json":
        data = report.model_dump()
        if canary_res:
            data["canary_probe"] = canary_res.model_dump()
        if chat_res:
            data["chat_test"] = chat_res.model_dump()
        formatted_output = json.dumps(data, indent=2)

    elif format == "sarif":
        formatted_output = json.dumps(report.to_sarif(), indent=2)

    elif format == "pr-comment":
        formatted_output = report.to_pr_comment()

    elif format == "markdown":
        status_str = "PASSED" if report.is_compatible else "FAILED"
        md_lines = [
            "# AdapterBridge Compatibility Report",
            f"**Checkpoint:** `{report.checkpoint_path}`  ",
            f"**Target Engine:** `{report.target_engine}`  ",
            f"**Status:** `{status_str}`  ",
            "",
            "## Diagnostic Issues",
        ]
        for issue in report.issues:
            md_lines.append(f"- **[{issue.severity.value.upper()}]** `[{issue.code}]`: {issue.message}")
        if canary_res:
            md_lines.append(f"\n### Canary Probe: {'PASSED' if canary_res.passed else 'FAILED'}\n{canary_res.message}")
        if chat_res:
            md_lines.append(f"\n### Chat Template Roundtrip: {'PASSED' if chat_res.success else 'FAILED'}\nPrompt:\n```\n{chat_res.rendered_prompt}\n```")
        formatted_output = "\n".join(md_lines)

    else:
        # Rich Terminal One-Command DX Output
        table = Table(title=f"AdapterBridge Diagnostic Report ({target.upper()})")
        table.add_column("Severity", style="bold")
        table.add_column("Code", style="cyan")
        table.add_column("Message")

        for issue in report.issues:
            color = "red" if issue.severity == IssueSeverity.ERROR else "yellow"
            table.add_row(f"[{color}]{issue.severity.value.upper()}[/{color}]", issue.code, issue.message)

        console.print(table)

        if not report.is_compatible:
            console.print(f"\n[bold red]❌ Incompatible Checkpoint Detected for {target.upper()}:[/]")
            tree = Tree(f"[bold white]{path}[/]")
            for issue in report.issues:
                prefix = "[bold red][MISSING][/]" if "MISSING" in issue.code else ("[bold red][MISMATCH][/]" if issue.severity == IssueSeverity.ERROR else "[bold yellow][WARNING][/]")
                node_text = f"{prefix} [bold cyan]{issue.code}:[/] {issue.message}"
                if issue.quick_fix:
                    node_text += f"\n   [dim gray]Fix hint: {issue.quick_fix}[/]"
                tree.add(node_text)
            console.print(tree)

            fix_dst = f"{path}-repaired"
            console.print(Panel(
                f"[bold yellow]💡 Quick Fix Available! Run:[/\n"
                f"  [bold green]adapterbridge fix --src {path} --dst {fix_dst} --target {target}[/]",
                title="Actionable Remediation",
                style="yellow",
            ))
        else:
            console.print(Panel(
                f"[bold green]✔ Checkpoint '{path}' is fully compatible with {target.upper()}![/]",
                title=f"AdapterBridge Diagnostic Report ({target.upper()})",
                style="green"
            ))

        if canary_res:
            c_style = "green" if canary_res.passed else "red"
            console.print(Panel(f"Canary Activation Probe: {canary_res.message}", style=c_style, title="Zero-GPU Canary Probe"))

        if chat_res:
            ch_style = "green" if chat_res.success else "yellow"
            console.print(Panel(f"Chat Roundtrip Test: {'Passed' if chat_res.success else 'Warnings Detected'}\nRendered Output:\n{chat_res.rendered_prompt}", style=ch_style, title="Chat Template Roundtrip"))

        formatted_output = None

    if output and formatted_output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(formatted_output)
        console.print(f"[bold green]Report written to:[/] {output}")
    elif formatted_output and format != "text":
        console.print(formatted_output)

    if not report.is_compatible:
        raise typer.Exit(code=1)


@app.command(name="fix")
def fix_cmd(
    src: str = typer.Option(..., "--src", "-s", help="Source checkpoint path"),
    dst: Optional[str] = typer.Option(None, "--dst", "-d", help="Destination path for repaired checkpoint"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Alias for --dst output path"),
    target: str = typer.Option(..., "--target", "-t", help="Target engine (vllm, sglang, ollama, tensorrt)"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Fallback base model ID"),
    auto_download_lineage: bool = typer.Option(True, "--auto-download-lineage", help="Auto fetch base model lineage from Hugging Face Hub"),
):
    """Automated normalizer, metadata synthesizer, & prefix remapper for LoRA checkpoints."""
    destination = dst or out or f"{src}-fixed"
    try:
        inspector = AdapterInspector(checkpoint_path=src, target_engine=target)
        console.print(f"[bold blue]Analyzing source checkpoint:[/] {src}")
        plan = inspector.auto_repair(destination_path=destination, fallback_base_model=base_model)

        tree = Tree(f"[bold green]Executed Remediation Plan -> {plan.output_path}[/]")
        for op in plan.operations:
            tree.add(f"[cyan]{op.action.value.upper()}:[/] target=[bold]{op.target_path}[/] details={op.details}")
        console.print(tree)

        console.print(Panel(
            f"[bold green]✔ Repaired checkpoint successfully saved to:[/] {plan.output_path}\n"
            f"Ready for serving on [bold cyan]{target.upper()}[/] runtime!",
            style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Remediation Error:[/] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="doctor")
def doctor_cmd(
    endpoint: str = typer.Option(..., "--endpoint", "-e", help="HTTP endpoint of active inference cluster (e.g. http://localhost:8000/v1)"),
    adapter_id: str = typer.Option(..., "--adapter-id", "-a", help="Adapter ID or model name configured on cluster"),
    prompt: str = typer.Option("Test adapter execution.", "--prompt", "-p", help="Test prompt for live generation verification"),
):
    """Operational health doctor for live, running inference clusters."""
    console.print(f"[bold blue]Probing live inference service at:[/] {endpoint}")
    endpoint_url = endpoint.rstrip("/")
    if not endpoint_url.endswith("/v1"):
        models_url = f"{endpoint_url}/v1/models"
        chat_url = f"{endpoint_url}/v1/chat/completions"
    else:
        models_url = f"{endpoint_url}/models"
        chat_url = f"{endpoint_url}/chat/completions"

    # 1. Query /v1/models
    models_found = []
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "AdapterBridge-Doctor/0.1.1"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models_found = [m.get("id") for m in data.get("data", [])]
    except Exception as e:
        console.print(f"[bold yellow]Warning:[/] Could not query models endpoint ({models_url}): {str(e)}")

    # 2. Test live inference chat request
    payload = json.dumps({
        "model": adapter_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
    }).encode("utf-8")

    req = urllib.request.Request(
        chat_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "AdapterBridge-Doctor/0.1.1"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            usage = res_data.get("usage", {})
            choices = res_data.get("choices", [])
            output_text = choices[0].get("message", {}).get("content", "") if choices else ""

            console.print(Panel(
                f"[bold green]✔ Doctor Check PASSED for endpoint {endpoint}[/]\n"
                f"Active Models Registered: {models_found}\n"
                f"Target Adapter ID: {adapter_id}\n"
                f"Generated Output Snippet: '{output_text.strip()}'\n"
                f"Token Usage: {usage}",
                title="Live Endpoint Diagnostic",
                style="green"
            ))
    except urllib.error.HTTPError as e:
        console.print(Panel(
            f"[bold red]❌ Doctor Check FAILED (HTTP {e.code})[/]\n"
            f"Endpoint URL: {chat_url}\n"
            f"Response: {e.read().decode('utf-8')[:300]}",
            style="red"
        ))
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(Panel(
            f"[bold red]❌ Doctor Check Failed:[/] {str(e)}",
            style="red"
        ))
        raise typer.Exit(code=1)


@app.command(name="sync")
def sync_cmd(
    force: bool = typer.Option(False, "--force", "-f", help="Force refresh local target schema cache"),
):
    """Synchronize latest target engine rule specifications to local schema cache."""
    mgr = SchemaSyncManager()
    synced = mgr.sync_schemas(force=force)
    console.print("[bold blue]Target Engine Schema Sync Results:[寿司]")
    for engine, ver in synced.items():
        console.print(f"  ├── [bold cyan]{engine}:[/] {ver}")
    console.print("[bold green]Local schema cache updated successfully![/]")


@app.command(name="verify")
def verify_cmd(
    path: str = typer.Option(..., "--path", "-p", help="Path to checkpoint directory"),
    target: str = typer.Option(..., "--target", "-t", help="Target engine"),
    tensor_parallel_size: int = typer.Option(1, "--tensor-parallel-size", "-tp", help="Tensor Parallelism degree"),
):
    """Zero-GPU mock serving dry-run engine."""
    try:
        inspector = AdapterInspector(checkpoint_path=path, target_engine=target)
        res = inspector.verify_dry_run(tensor_parallel_size=tensor_parallel_size)

        if res.success:
            console.print(Panel(f"SUCCESS: {res.reason}\nEstimated Adapter RAM: {res.memory_estimate_mb} MB", style="green"))
        else:
            console.print(Panel(f"FAILURE: {res.reason}\nIssues: {res.sharding_issues}", style="red"))
            raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Verification Error:[/] {str(e)}")
        raise typer.Exit(code=1)


@app.command(name="export")
def export_cmd(
    path: str = typer.Option(..., "--path", "-p", help="Path to checkpoint directory"),
    target: str = typer.Option(..., "--target", "-t", help="Target engine"),
    output: str = typer.Option(..., "--output", "-o", help="Export target destination directory"),
):
    """Serving-targeted bundler."""
    try:
        inspector = AdapterInspector(checkpoint_path=path, target_engine=target)
        plan = inspector.auto_repair(destination_path=output)
        console.print(f"[bold green]Successfully exported checkpoint bundle to:[/] {output}")
    except Exception as e:
        console.print(f"[bold red]Export Error:[/] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
