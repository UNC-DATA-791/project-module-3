# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.2.2",
#     "marimo>=0.25.0",
#     "numpy==2.4.6",
#     "polars==1.42.1",
#     "pyfamsa==0.7.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import io
    import urllib.parse
    import urllib.request

    import altair as alt
    import numpy as np
    import polars as pl
    from pyfamsa import Aligner, Sequence

    alt.data_transformers.disable_max_rows()
    return Aligner, Sequence, alt, io, np, pl, urllib


@app.cell
def _(mo):
    mo.md(r"""
    # Unit 9 assignment

    In this assignment you build a **training set** for fine tuning a generative protein language model.

    You will:

    1. Pick one protein family and download its sequences.
    2. Remove sequences with non-canonical amino acids.
    3. Remove redundant sequences.
    4. Make three charts to summarize your sequence set.
    5. Save the output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick a protein family

    A **protein family** is a group of related proteins. They have the same function. Choose one of the families below.

    | Family | What it does |
    | --- | --- |
    | SH3 domain | Helps proteins stick to each other |
    | Ubiquitin | Tags other proteins for the cell to remove |
    | Thioredoxin | Fixes broken bonds in other proteins |
    | Response regulator | Helps bacteria react to their world |
    """)
    return


@app.cell
def _(mo):
    FAMILIES = {
        "SH3 domain": ("PF00018", 50, 70),
        "Ubiquitin": ("PF00240", 70, 80),
        "Thioredoxin": ("PF00085", 100, 120),
        "Response regulator": ("PF00072", 115, 130),
    }
    family_picker = mo.ui.dropdown(
        options=list(FAMILIES), value="Thioredoxin", label="Protein family"
    )
    family_picker
    return FAMILIES, family_picker


@app.cell
def _(FAMILIES, family_picker, io, pl, urllib):
    # This function hits the uniprot API to request the protein family you selected
    def fetch_family(pfam_id, min_len, max_len, n_max=2000):
        """Download up to n_max sequences of one Pfam family from UniProt."""
        query = f"(xref:pfam-{pfam_id}) AND (length:[{min_len} TO {max_len}])"
        url = (
            "https://rest.uniprot.org/uniprotkb/search?"
            + urllib.parse.urlencode(
                {
                    "query": query,
                    "format": "tsv",
                    "fields": "accession,sequence,length",
                    "size": 500,
                }
            )
        )
        pages = []
        while url and sum(len(p) for p in pages) < n_max:
            request = urllib.request.Request(
                url, headers={"User-Agent": "data791"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                pages.append(
                    pl.read_csv(io.BytesIO(response.read()), separator="\t")
                )
                link = response.headers.get("Link", "")
            url = (
                link.split(";")[0].strip("<> ")
                if 'rel="next"' in link
                else None
            )
        return (
            pl.concat(pages)
            .head(n_max)
            .rename(
                {
                    "Entry": "accession",
                    "Sequence": "sequence",
                    "Length": "length",
                }
            )
        )

    pfam_id, min_len, max_len = FAMILIES[family_picker.value]
    df = fetch_family(pfam_id, min_len, max_len)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: remove non-canonical amino acids

    We will remove proteins that have AAs not in the set of 20 canonical AAs.

    The cell below lists the 20 canonical letters.
    """)
    return


@app.cell
def _():
    CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    "".join(sorted(CANONICAL_AAS))
    return (CANONICAL_AAS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Keep only the rows of `df` where every letter of `sequence` is in `CANONICAL_AAS`. Call the result `df_clean`.

    <details>
    <summary>Hint</summary>

    ```python
    df_clean = df.filter(
        pl.col("sequence").map_elements(
            lambda s: set(s).issubset(CANONICAL_AAS), return_dtype=pl.Boolean
        )
    )
    f"{len(df)} -> {len(df_clean)} sequences"
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Line up the sequences for comparison

    To compare proteins, you line them up. This is a **multiple sequence alignment (MSA)**. In an MSA, each row is one protein. Each column is one position. A dash (`-`) means a gap: that protein has no letter at that position.

    The cell below does the alignment for you with [FAMSA](https://github.com/refresh-bio/FAMSA), a fast tool.
    """)
    return


@app.cell
def _(Aligner, Sequence, df_clean, mo, np):
    def align(sequences):
        """Align sequences. Returns an array: one row per sequence, one column per MSA position."""
        famsa_seqs = [
            Sequence(str(i).encode(), s.encode())
            for i, s in enumerate(sequences)
        ]
        aligned = Aligner(guide_tree="upgma").align(famsa_seqs)
        rows = {int(s.id): s.sequence.decode() for s in aligned}
        return np.array([list(rows[i]) for i in range(len(sequences))])

    # This cell needs `df_clean` from Exercise 1. It waits until it exists.
    try:
        df_clean
    except NameError:
        mo.stop(
            True,
            mo.callout(
                mo.md("**Finish Exercise 1 first.** This cell needs `df_clean`."),
                kind="warn",
            ),
        )

    msa = align(df_clean["sequence"].to_list())
    msa.shape
    return (msa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: remove redundant sequences

    Some sequences are very similar. If the model sees many copies, it may learn to repeat them.

    **Identity** is the share of positions where two aligned sequences have the same letter. Use the slider to set the limit. A sequence is dropped if it has more than this identity with a sequence we already kept.
    """)
    return


@app.cell
def _(mo, np):
    max_identity = mo.ui.slider(
        0.5,
        1.0,
        step=0.05,
        value=0.9,
        label="Maximum identity",
        show_value=True,
    )

    def drop_redundant(msa, max_identity):
        """Return the row numbers to keep. A row stays if it is less similar than max_identity to every kept row."""
        kept = []
        for i, row in enumerate(msa):
            if kept:
                other = msa[kept]
                both = (other != "-") & (row != "-")
                same = ((other == row) & both).sum(axis=1)
                shorter = np.minimum(
                    (other != "-").sum(axis=1), (row != "-").sum()
                )
                if (same / np.maximum(shorter, 1)).max() > max_identity:
                    continue
            kept.append(i)
        return kept

    max_identity
    return drop_redundant, max_identity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the function `drop_redundant` above. Then use it:

    1. Call it with `msa` and `max_identity.value`. Call the result `keep`.
    2. Use `keep` to select rows from `df_clean`. Call the result `df_final`.
    3. Use `keep` to select rows from `msa`. Call the result `msa_final`.

    Move the slider and run again. How many sequences are left?


    <details>
    <summary>Hint</summary>

    ```python
    keep = drop_redundant(msa, max_identity.value)
    df_final = df_clean[keep]
    msa_final = msa[keep]
    f"{len(df_clean)} -> {len(df_final)} sequences"
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chart 1: how long are the proteins?

    Make a histogram of `length` in `df_final`. Use about 30 bins.


    <details>
    <summary>Hint</summary>

    ```python
    alt.Chart(df_final, width=600).mark_bar().encode(
        x=alt.X("length:Q").bin(maxbins=30),
        y=alt.Y("count()"),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: remove gap columns

    Many MSA columns are mostly gaps. Only a few proteins have a letter there. It's helpful to remove these "gappy" columns for visualization.
    """)
    return


@app.cell
def _(mo):
    max_gap = mo.ui.slider(
        0.0,
        1.0,
        step=0.05,
        value=0.5,
        label="Maximum gap share",
        show_value=True,
    )
    max_gap
    return (max_gap,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Find the share of gaps in each column of `msa_final`. Call it `gap_share`. It has one number per column.
    2. Keep the columns where `gap_share` is at most `max_gap.value`. Call the result `msa_trim`.

    <br>

    The share of gaps is the mean of `msa_final == "-"` down each column.


    <details>
    <summary>Hint</summary>

    ```python
    gap_share = (msa_final == "-").mean(axis=0)
    msa_trim = msa_final[:, gap_share <= max_gap.value]
    f"{msa_final.shape[1]} -> {msa_trim.shape[1]} columns"
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chart 2: the MSA

    Draw the first 100 rows of `msa_trim` as a grid of colored squares. Each square is one letter. Gaps are gray. Look for columns with one color. Those columns are the same in many proteins.

    First make a long table `msa_long` with one row per square. Then plot it.


    <details>
    <summary>Hint</summary>

    ```python
    _sample = msa_trim[:100]
    msa_long = pl.DataFrame({
        "sequence": np.repeat(np.arange(len(_sample)), _sample.shape[1]),
        "position": np.tile(np.arange(_sample.shape[1]), len(_sample)),
        "residue": _sample.ravel(),
    })

    alt.Chart(msa_long, width=700, height=300).mark_rect().encode(
        x=alt.X("position:O", axis=None),
        y=alt.Y("sequence:O", axis=None),
        color=alt.when(alt.datum.residue == "-")
        .then(alt.value("#e5e5e5"))
        .otherwise(alt.Color("residue:N").title("Amino acid")),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Chart 3: which positions are conserved?

    **Conservation** tells how similar proteins are at a given position. A value of 1 means every protein has the same letter. A value near 0 means the letters differ a lot.

    We use entropy to measure it. Entropy is high when the letters are mixed. Conservation is `1 - entropy / log2(20)`.

    Make a line chart of conservation along `msa_trim`. Put the position on the x axis. Positions with high conservation matter most for the protein.


    <details>
    <summary>Hint</summary>

    ```python
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    counts = np.stack([(msa_trim == aa).sum(axis=0) for aa in AMINO_ACIDS])  # 20 x columns
    freqs = counts / np.maximum(counts.sum(axis=0), 1)
    entropy = -(freqs * np.log2(np.where(freqs > 0, freqs, 1))).sum(axis=0)

    conservation = pl.DataFrame({
        "position": np.arange(msa_trim.shape[1]),
        "conservation": 1 - entropy / np.log2(20),
    })

    alt.Chart(conservation, width=700).mark_line().encode(
        x=alt.X("position:Q"),
        y=alt.Y("conservation:Q").scale(domain=[0, 1]),
    )
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4: check and save your set

    Write two `assert` checks on `df_final`:

    1. Every sequence uses only the 20 canonical letters (`CANONICAL_AAS`).
    2. No sequence appears twice.

    Then save `accession` and `sequence` to `protein_set.csv`. You will use this file next week.


    <details>
    <summary>Hint</summary>

    ```python
    assert all(set(s).issubset(CANONICAL_AAS) for s in df_final["sequence"])
    assert df_final["sequence"].n_unique() == len(df_final)

    df_final.select("accession", "sequence").write_csv("protein_set.csv")
    ```

    </details>
    """)
    return


@app.cell
def _():
    # Code goes here
    return