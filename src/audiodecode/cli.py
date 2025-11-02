"""
Command-line interface for AudioDecode transcription.

Provides easy-to-use commands for transcribing audio files, batch processing,
and model management.
"""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="AudioDecode: Fast audio transcription with Whisper")
console = Console()


@app.command()
def transcribe(
    files: List[Path] = typer.Argument(..., help="Audio file(s) to transcribe"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (e.g., 'en')"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("txt", "--format", "-f", help="Output format (txt, srt, vtt, json)"),
    device: str = typer.Option("auto", "--device", "-d", help="Device (cpu, cuda, auto)"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process multiple files in parallel"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers for batch mode"),
):
    """
    Transcribe audio file(s) to text.

    Examples:
        audiodecode transcribe podcast.mp3
        audiodecode transcribe video.mp4 --output subtitles.srt --format srt
        audiodecode transcribe *.mp3 --batch --workers 4
    """
    try:
        from audiodecode.inference import transcribe_file, WhisperInference
    except ImportError:
        console.print("[red]Error:[/red] faster-whisper not installed.")
        console.print("Install with: [cyan]pip install audiodecode[inference][/cyan]")
        raise typer.Exit(1)

    if len(files) == 0:
        console.print("[red]Error:[/red] No files specified")
        raise typer.Exit(1)

    # Single file mode
    if len(files) == 1 and not batch:
        file = files[0]
        if not file.exists():
            console.print(f"[red]Error:[/red] File not found: {file}")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(f"Transcribing {file.name} with {model} model...", total=None)

            result = transcribe_file(
                str(file),
                model_size=model,
                language=language,
                device=device
            )

        # Output handling
        if output:
            result.save(output)
            console.print(f"[green]✓[/green] Saved to {output}")
        else:
            # Print to stdout
            if format == "txt":
                console.print(result.text)
            elif format == "srt":
                console.print(result.to_srt())
            elif format == "vtt":
                console.print(result.to_vtt())
            elif format == "json":
                console.print(result.to_json())

        # Stats
        console.print(f"\n[dim]Duration: {result.duration:.1f}s | Language: {result.language} | Segments: {len(result.segments)}[/dim]")

    # Batch mode
    else:
        console.print(f"[cyan]Processing {len(files)} files with {workers} workers...[/cyan]")

        # For now, just process sequentially (batch processing will be added later)
        whisper = WhisperInference(model_size=model, device=device)

        with Progress(console=console) as progress:
            task = progress.add_task(f"Transcribing files...", total=len(files))

            for file in files:
                if not file.exists():
                    console.print(f"[yellow]Warning:[/yellow] Skipping {file} (not found)")
                    continue

                result = whisper.transcribe_file(str(file), language=language)

                # Auto-generate output filename
                if output:
                    out_file = output
                else:
                    out_file = file.with_suffix(f".{format}")

                result.save(out_file)
                console.print(f"[green]✓[/green] {file.name} -> {out_file.name}")

                progress.update(task, advance=1)

        console.print(f"\n[green]✓ Completed {len(files)} files[/green]")


@app.command("list-models")
def list_models():
    """List available Whisper models."""
    table = Table(title="Available Whisper Models")

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Parameters", style="magenta")
    table.add_column("Speed", style="green")
    table.add_column("Accuracy", style="yellow")
    table.add_column("Recommended For")

    models = [
        ("tiny", "39M", "⚡⚡⚡⚡", "⭐⭐", "Quick drafts, testing"),
        ("base", "74M", "⚡⚡⚡", "⭐⭐⭐", "General use (recommended)"),
        ("small", "244M", "⚡⚡", "⭐⭐⭐⭐", "Good balance"),
        ("medium", "769M", "⚡", "⭐⭐⭐⭐⭐", "High accuracy"),
        ("large-v3", "1550M", "⚡", "⭐⭐⭐⭐⭐", "Best accuracy"),
    ]

    for model_name, params, speed, accuracy, use_case in models:
        table.add_row(model_name, params, speed, accuracy, use_case)

    console.print(table)
    console.print("\n[dim]Use with: audiodecode transcribe audio.mp3 --model <model>[/dim]")


@app.command()
def info():
    """Show AudioDecode version and configuration."""
    try:
        import audiodecode
        from audiodecode.inference import _FASTER_WHISPER_AVAILABLE
        import numpy as np
        import av
        import soundfile as sf
    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[cyan]AudioDecode Information[/cyan]\n")
    console.print(f"Version: [green]{audiodecode.__version__}[/green]")
    console.print(f"NumPy: [green]{np.__version__}[/green]")
    console.print(f"PyAV: [green]{av.__version__}[/green]")
    console.print(f"SoundFile: [green]{sf.__version__}[/green]")
    console.print(f"faster-whisper: [{'green' if _FASTER_WHISPER_AVAILABLE else 'red'}]{'Available' if _FASTER_WHISPER_AVAILABLE else 'Not installed'}[/]")

    if not _FASTER_WHISPER_AVAILABLE:
        console.print("\n[yellow]Install STT support:[/yellow] [cyan]pip install audiodecode[inference][/cyan]")


@app.command()
def version():
    """Show AudioDecode version."""
    import audiodecode
    console.print(f"audiodecode {audiodecode.__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
