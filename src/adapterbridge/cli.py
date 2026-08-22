"""CLI entry points using Typer and Rich formatting."""

import json
import os
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from adapterbridge.core.inspector import AdapterInspector
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
    target: str = typer.Option(..., "--target", "-t", help="Target serving engine (vllm, sglang, ollama, tensorrt)"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, sarif, markdown, pr-comment"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Optional output file destination"),
):
    """Static linter and compatibility validator for LoRA checkpoints."""
    try:
        inspector = AdapterInspector(checkpoint_path=path, target_engine=target)
        report = inspector.run_diagnostics()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise typer.Exit(code=1)

    if format == "json":
        formatted_output = report.model_dump_json(indent=2)
    elif format == "sarif":
        formatted_output = json.dumps(report.to_sarif(), indent=2)
    elif format == "pr-comment":
        formatted_output = report.to_pr_comment()
    elif format == "markdown":
        status_str = "PASSED" if report.is_compatible else "FAILED"
        md_lines = [
            f"# AdapterBridge Compatibility Report",
            f"**Checkpoint:** `{report.checkpoint_path}`  ",
            f"**Target Engine:** `{report.target_engine}`  ",
            f"**Status:** `{status_str}`  ",
            "",
            "## Diagnostic Issues",
        ]
        for issue in report.issues:
            md_lines.append(f"- **[{issue.severity.value.upper()}]** `[{issue.code}]`: {issue.message}")
        formatted_output = "\n".join(md_lines)
    else:
        # Rich Terminal Output
        table = Table(title=f"AdapterBridge Diagnostic Report ({target.upper()})")
        table.add_column("Severity", style="bold")
        table.add_column("Code", style="cyan")
        table.add_column("Message")

        for issue in report.issues:
            color = "red" if issue.severity == IssueSeverity.ERROR else "yellow"
            table.add_row(f"[{color}]{issue.severity.value.upper()}[/{color}]", issue.code, issue.message)

        console.print(table)
        status_panel_style = "green" if report.is_compatible else "red"
        console.print(Panel(report.summary, style=status_panel_style))
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
    dst: str = typer.Option(..., "--dst", "-d", help="Destination path for repaired checkpoint"),
    target: str = typer.Option(..., "--target", "-t", help="Target engine (vllm, sglang, ollama, tensorrt)"),
    base_model: Optional[str] = typer.Option(None, "--base-model", "-b", help="Fallback base model ID"),
):
    """Automated normalizer & metadata synthesizer for LoRA checkpoints."""
    try:
        inspector = AdapterInspector(checkpoint_path=src, target_engine=target)
        console.print(f"[bold blue]Analyzing source checkpoint:[/] {src}")
        plan = inspector.auto_repair(destination_path=dst, fallback_base_model=base_model)
        
        console.print(f"[bold green]Repaired checkpoint saved to:[/] {plan.output_path}")
        console.print(f"Executed {len(plan.operations)} remediation operation(s).")
    except Exception as e:
        console.print(f"[bold red]Remediation Error:[/] {str(e)}")
        raise typer.Exit(code=1)


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
