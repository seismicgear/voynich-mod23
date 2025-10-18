# INSTRUCTIONS.md

This file explains how to run the modular‑23 decoding experiment from start to finish.

---

## Step 1: Clone the Repo

```bash
git clone https://github.com/yourname/voynich-mod23.git
cd voynich-mod23
````

---

## Step 2: Set Up the Environment

Install dependencies:

```bash
pip install pandas numpy scipy nltk
python -m nltk.downloader punkt
```

---

## Step 3: Prepare the Data

Place these in the `/data/` folder:

* `voynich_eva_takahashi.txt`
  A plain-text file of EVA transcription (one token per line).

* `latin_reference.txt`
  A 15th-century Latin corpus (\~100 KB) for trigram comparison.

---

## Step 4: Run the Experiment

```bash
python run_experiment.py
```

This will:

* Decode all EVA words using the modular‑23 inverse cipher
* Compute:

  * gzip compression size (bytes)
  * Trigram cosine similarity vs Latin
* Run 10,000 Monte Carlo randomizations
* Output p-values showing whether your mapping outperforms chance

---

## Sample Output

```
Observed entropy: 3.612 bits/char  
Observed trigram‑cosine similarity vs Latin: 0.2715  
p‑value (gzip smaller)  : 0.0082
p‑value (cosine higher) : 0.0034  
```

---

## Step 5: Tweak the Mapping (Optional)

Edit `glyph_to_num` in `decoder.py`:

```python
glyph_to_num = {
    'q': 1, 'o': 2, 'k': 3, ...
}
```

---

## Contribute or Replicate

If you rerun the pipeline, feel free to open a pull request or share your output from `/results/` for comparison.

```
