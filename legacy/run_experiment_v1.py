"""
run_experiment.py

Entry point for the Voynich Modular-23 Experiment.
Delegates to experiment.cli.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiment import cli

if __name__ == "__main__":
    cli.main()
