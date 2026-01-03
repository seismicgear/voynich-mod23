# CODEBOOK.md

> **⚠️ NOTE: Version 2.0 Dynamic Mapping**
>
> In Version 2.0, mappings are **dynamically generated** by the `solver.py` script using Simulated Annealing. There is no single static "codebook".
>
> The table below represents a **legacy hypothesis** from Version 1.0 or a specific output snapshot. To see the current best mapping found by the solver, inspect the console output of `solver.py`.

---

## Legacy / Hypothetical Codebook (V1)

This table encodes a working hypothesis about glyph roles/meanings based on positional and co‑occurrence patterns.

| Glyph     | Latin | Role (Hypothesis)         | Meaning (Hypothesis)      | Evidence |
|-----------|--------|---------------------------|---------------------------|----------|
| chedy     | H      | Line Anchor               | title / begin / use       | freq + pos |
| dar       | R      | Verb / Action             | apply / use               | colloc (after 'chedy') |
| shedy     | F      | Modifier                  | boil / prepare            | colloc (precedes nouns) |
| air       | I      | Suffix / Modifier         | of / in / to              | pos (end of phrase) |
| ar        | R      | Suffix / Particle         | again / plural            | pos (end of line) |
| qokedy    | M      | Root / Noun               | plant / root              | colloc (object of 'dar') |
| shol      | G      | Root / Noun               | herb / substance          | freq + colloc |
| chody     | B      | Root / Noun               | tool / material           | colloc |
| qoty      | O      | Structural Tag            | section / category        | pos (after nouns) |
| chol      | S      | Tag / Qualifier           | leaf / part               | colloc |
| daiin     | K      | Noun / Title              | target / patient          | colloc (after verbs) |
| aiin      | T      | Noun / Title              | subject / body            | structure |
| chedal    | Q      | Phrase Initiator          | for / regarding           | pos (mid-line) |
| cheedy    | X      | Emphatic Anchor           | also / again              | freq |
| oty       | D      | Structural / Tag          | about / related to        | pos |
| qokeedy   | A      | Root / Noun               | compound plant / object   | colloc |
