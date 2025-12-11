"""Allow running the CLI as: python -m deepagents.cli."""

from deepagents_cli.main import cli_main

if __name__ == "__main__":
    import os
    print(os.path.abspath(__file__))
    cli_main()
