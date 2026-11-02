"""Python entry point for the Windows desktop acne monitoring workbench."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    launcher = Path(__file__).resolve().with_name("start_acne_monitor.pyw")
    runpy.run_path(str(launcher), run_name="__main__")
