"""
voynich — a computational decipherment workbench for the Voynich Manuscript.

Core ideas:
  * Hypotheses are expressed as token -> letter substitution keys
    (optionally position-dependent within a line).
  * Candidate decipherments are scored with a smoothed character n-gram
    language model trained on a reference corpus (medieval Latin,
    14th-century Italian, or early-modern English).
  * The key space is searched with simulated annealing (MCMC), with
    multiple restarts and held-out validation on alternating lines.
  * The solver is validated on synthetic ciphers with known answers,
    so "the machinery works" is a testable claim even though the
    manuscript itself remains undeciphered.
"""

__version__ = "3.0.0"
