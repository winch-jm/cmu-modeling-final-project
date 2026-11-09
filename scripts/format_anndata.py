#!/usr/bin/env python3
"""Convert the extracted GSE193460 Visium files to spatial AnnData objects."""

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csc_matrix


def strings(dataset):
    return dataset.asstr()[:]


def read_counts(path):
    """Read 10x v3 CSC (features x barcodes), retaining every feature."""
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        counts = csc_matrix(
            (matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
            shape=tuple(matrix["shape"][:]),
        ).T.tocsr()
        features = matrix["features"]
        var = pd.DataFrame(index=pd.Index(strings(features["id"]), name="gene_id"))
        for key, value in features.items():
            if isinstance(value, h5py.Dataset) and not key.startswith("_"):
                if value.ndim == 1 and len(value) == counts.shape[1]:
                    var[key] = strings(value) if value.dtype.kind in "OSU" else value[:]
        var = var.rename(columns={"id": "feature_id", "name": "gene_symbol"})
        result = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(index=pd.Index(strings(matrix["barcodes"]), name="barcode")),
            var=var,
        )
    result.var_names_make_unique()
    return result


def convert(path, data_dir, workbook):
    sample, kind = None, None
    # Use exact suffix removal: sample identifiers themselves contain underscores.
    for candidate in ("filtered", "raw"):
        suffix = f"_{candidate}_feature_bc_matrix.h5"
        if path.name.endswith(suffix):
            sample, kind = path.name[:-len(suffix)], candidate
            break
    if kind is None:
        raise ValueError(f"Unrecognized matrix filename: {path}")
    a = read_counts(path)
    positions = pd.read_csv(
        data_dir / f"{sample}_tissue_positions_list.csv", header=None,
        names=["barcode", "in_tissue", "array_row", "array_col",
               "pxl_row_in_fullres", "pxl_col_in_fullres"], index_col=0,
    )
    annotations = pd.read_csv(data_dir / f"{sample}_spot_annotation.csv", index_col=0)
    off_tissue = pd.read_csv(data_dir / f"{sample}_off_tissue.csv", index_col=0)
    # AnnData does not automatically sanitize DataFrames nested inside uns.
    # Categoricals encode text and missing values without mixed object arrays.
    for table in (annotations, off_tissue):
        for column in table.select_dtypes(include=["object", "string"]).columns:
            table[column] = pd.Categorical(
                table[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
            )
    for name, table in (("positions", positions), ("annotations", annotations),
                        ("off_tissue", off_tissue)):
        if not table.index.is_unique:
            raise ValueError(f"Duplicate barcodes in {sample} {name}")
    if not a.obs_names.isin(positions.index).all():
        raise ValueError(f"Missing spatial coordinates for {sample}")
    a.obs = a.obs.join(positions).join(annotations)
    a.obs["listed_off_tissue"] = a.obs_names.isin(off_tissue.index)
    a.obs["has_spot_annotation"] = a.obs_names.isin(annotations.index)
    a.obs["sample"] = sample
    a.obs["matrix_type"] = kind
    a.obsm["spatial"] = a.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    a.obsm["array_coords"] = a.obs[["array_row", "array_col"]].to_numpy()
    images = {}
    for key, suffix in (("hires", "tissue_hires_image.png"),
                        ("lowres", "tissue_lowres_image.png"),
                        ("aligned_fiducials", "aligned_fiducials.jpg")):
        with Image.open(data_dir / f"{sample}_{suffix}") as img:
            images[key] = np.asarray(img).copy()
    with (data_dir / f"{sample}_scalefactors_json.json").open() as stream:
        scalefactors = json.load(stream)
    a.uns["spatial"] = {sample: {"images": images, "scalefactors": scalefactors}}
    # Preserve complete source tables, including barcodes absent from filtered X.
    a.uns["source_tables"] = {
        "positions": positions, "spot_annotation": annotations, "off_tissue": off_tissue,
    }
    a.uns["workbook"] = workbook
    a.uns["source_matrix"] = path.name
    a.uns["counts_location"] = "X (unaltered counts; no normalization or filtering)"
    return a, sample, kind


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "anndata")
    parser.add_argument("--matrix", choices=["all", "filtered", "raw"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    paths = sorted(args.data_dir.glob("*_feature_bc_matrix.h5"))
    if args.matrix != "all":
        paths = [p for p in paths if p.name.endswith(f"_{args.matrix}_feature_bc_matrix.h5")]
    if not paths:
        parser.error(f"No count matrices found in {args.data_dir}")
    workbook = {}
    for path in sorted(args.data_dir.glob("*.xlsx")):
        # Store sheets as text, preserving missing cells as empty strings and
        # avoiding mixed Excel column types that HDF5 cannot serialize.
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        workbook[path.name] = {
            sheet: frame.fillna("").to_numpy(dtype=str)
            for sheet, frame in sheets.items()
        }
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        destination = output / path.name.replace("_feature_bc_matrix.h5", ".h5ad")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; use --overwrite to replace it")
        a, sample, kind = convert(path, args.data_dir, workbook)
        temporary = destination.with_suffix(".h5ad.tmp")
        try:
            a.write_h5ad(temporary, compression="gzip")
            check = ad.read_h5ad(temporary, backed="r")
            try:
                if check.shape != a.shape or not check.obs_names.equals(a.obs_names):
                    raise ValueError(f"Output validation failed: {temporary}")
            finally:
                check.file.close()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(destination)
        print(f"{sample} ({kind}): {a.n_obs:,} spots x {a.n_vars:,} features -> {destination}")


if __name__ == "__main__":
    main()
