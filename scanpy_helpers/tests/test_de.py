import scanpy as sc
from scanpy_helpers.de import edger, glm_gam_poi, mast
import numpy as np
import pytest


@pytest.fixture
def adata():
    tmp_adata = sc.datasets.pbmc68k_reduced()
    """turn scaled data back into count integers. Doesn't matter if it isn't accurate"""
    tmp_adata.X = np.rint(tmp_adata.X - np.min(tmp_adata.X, axis=0)).astype(int)
    return tmp_adata


@pytest.mark.parametrize("cofactors", [None, ["n_genes", "percent_mito"]])
def test_edger(adata, cofactors):
    edger(
        adata,
        groupby="bulk_labels",
        cofactors=cofactors,
        groups=("Dendritic", "CD19+ B"),
    )


@pytest.mark.parametrize("cofactors", [None, ["n_genes", "percent_mito"]])
def test_glmgampoi(adata, cofactors):
    glm_gam_poi(
        adata,
        groupby="bulk_labels",
        cofactors=cofactors,
        groups=("Dendritic", "CD19+ B"),
    )


@pytest.mark.parametrize("cofactors", [None, ["n_genes", "percent_mito"]])
def test_mast(adata, cofactors):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    mast(
        adata,
        groupby="bulk_labels",
        cofactors=cofactors,
        groups=("Dendritic", "CD19+ B"),
    )
