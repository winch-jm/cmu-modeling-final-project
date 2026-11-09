# cmu-modeling-final-project
Biological Modeling and Simulation Final Project

Place the extracted Visium data in `data/raw/`, then convert to AnnData
(run from this directory):

```sh
python3 -m pip install -r requirements-anndata.txt
python3 scripts/format_anndata.py
```

Writes eight files to `data/anndata/`: raw and filtered counts for each of
the four samples. Use `--matrix filtered` or `--matrix raw` to select one
matrix type, `--output-dir PATH` to change the destination, and `--overwrite`
to replace existing outputs.

Each file keeps sparse, unnormalized counts in `X`, feature IDs as `var_names`,
gene symbols in `var['gene_symbol']`, spot annotations in `obs`, and full-resolution
pixel coordinates (x, y) in `obsm['spatial']`. Images and scale factors are in
`uns['spatial']`; complete source CSV tables and Excel sheets are retained in
`uns['source_tables']` and `uns['workbook']`. No spots or features are removed.
The tar archive duplicates the extracted inputs and is not converted separately.
