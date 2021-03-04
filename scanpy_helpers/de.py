"""Module for helpers with differential gene expression"""

import numpy as np
import pandas as pd
from scanpy import logging, AnnData
from scipy.sparse import issparse
from typing import Sequence, Tuple, Optional
from threadpoolctl import threadpool_limits
from pandas.api.types import is_numeric_dtype
from multiprocessing import Pool
import itertools


def scvi_rank_genes_groups(adata: AnnData, model, *, groupby: str):
    """Perform differential expression with scVI and add the results
    to anndata in the same format as `sc.tl.rank_gene_groups` does.

    Requires that an scvi model is trained on `adata`.

    Returns a table with results, adds gene ranks to `adata.uns`
    """
    de_res = model.differential_expression(
        adata, groupby=groupby, batch_correction=True
    )
    de_res["score"] = np.sign(de_res["lfc_mean"]) * de_res["bayes_factor"]
    res_dict = {
        "params": {
            "groupby": groupby,
            "reference": "rest",
            "method": "scVI change",
            "use_raw": True,
            "layer": None,
            "corr_method": "scVI",
        },
        "names": [],
        "scores": [],
        "pvals": [],
        "pvals_adj": [],
        "logfoldchanges": [],
    }
    df_groupby = de_res.groupby("comparison")
    for comparison, tmp_df in df_groupby:
        tmp_df = tmp_df.sort_values("score", ascending=False)
        res_dict["names"].append(tmp_df.index.values)
        res_dict["scores"].append(tmp_df["score"].values)
        res_dict["pvals"].append(tmp_df["proba_not_de"].values)
        # scvi pvalues are already adjusted
        res_dict["pvals_adj"] = res_dict["pvals"]
        res_dict["logfoldchanges"].append(tmp_df["lfc_mean"].values)

    for key in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
        res_dict[key] = pd.DataFrame(
            np.vstack(res_dict[key]).T,
            columns=[c.split("vs")[0].strip() for c in df_groupby.groups.keys()],
        ).to_records(index=False, column_dtypes="O")
    adata.uns["rank_genes_groups"] = res_dict
    return de_res


def _fix_contrasts(contrasts, groupby):
    """
    Ensure that every contrast consists of two iterables.
    Preprends the 'groupby' colname to the contrast to fit the design matrix.
    """
    tmp_contrasts = []
    for a, b in contrasts:
        if isinstance(a, str):
            a = (a,)
        if isinstance(b, str):
            b = (b,)
        a = [f"{groupby}{x}" for x in _make_names(a)]
        b = [f"{groupby}{x}" for x in _make_names(b)]
        tmp_contrasts.append((a, b))
    return tmp_contrasts


def _make_names(seq: Sequence):
    return ["".join(e if e.isalnum() else "_" for e in string) for string in seq]


def edger_rank_genes_groups(
    adata: AnnData,
    *,
    groupby: str,
    contrasts: Sequence[Tuple[Sequence[str], Sequence[str]]],
    cofactors: Sequence[str] = None,
    layer: Optional[str] = None,
    n_cores_per_job: int = 4,
    n_jobs: int = 4,
):
    """
    Perform DE analysis using edgeR.

    Requires that an R installation and the following packages are available

        edgeR
        BiocParallel
        RhpcBLASctl

    Install them with `conda install bioconductor-edger bioconductor-biocparallel r-rhpcblasctl`.

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        The column in adata.obs to test for DE
    contrast
        Liste of tuples with tests to perform, e.g.
        `[('A', 'B'), (('A', 'B'), ('C', 'D','E'))]` which is equivalent to
        `[(('A', ), ('B', )), (('A', 'B'), ('C', 'D','E'))]
    cofactors
        Additional columns to include into the model
    layer
        layer in adata that contains raw counts. If None, use `X`.
    n_cores_per_job
        Number of cores to run per job (including BLAS parallelization)
    n_jobs
        Number of tests to run in parallel.
    """

    try:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2 import robjects as ro
    except ImportError:
        raise ImportError("edger requires rpy2 to be installed. ")

    try:
        base = importr("base")
        edger = importr("edgeR")
        stats = importr("stats")
        limma = importr("limma")
        blasctl = importr("RhpcBLASctl")
        bcparallel = importr("BiocParallel")
    except ImportError:
        raise ImportError(
            "edgeR requires a valid R installation with the following packages: "
            "edgeR"
        )

    blasctl.blas_set_num_threads(n_cores_per_job)
    blasctl.omp_set_num_threads(n_cores_per_job)
    logging.info("Preparing R objects")

    # Define model formula
    cofactors = [] if cofactors is None else _make_names(cofactors)
    groupby = _make_names([groupby])[0]
    model = f"~ 0 + {groupby} + {' + '.join(cofactors)}"
    contrasts = _fix_contrasts(contrasts, groupby)

    bcparallel.register(bcparallel.MulticoreParam(n_jobs))

    with localconverter(ro.default_converter + pandas2ri.converter):
        tmp_obs = adata.obs.loc[:, [groupby] + cofactors]
        tmp_obs.columns = _make_names(tmp_obs.columns)
        for col in tmp_obs.columns:
            if not is_numeric_dtype(tmp_obs[col]):
                tmp_obs[col] = _make_names(tmp_obs[col])

        obs_r = ro.conversion.py2rpy(tmp_obs)
        # just need the index
        var_r = ro.conversion.py2rpy(
            pd.DataFrame({"gene_symbol": adata.var_names}, index=adata.var_names)
        )

    with localconverter(ro.default_converter + numpy2ri.converter):
        expr = adata.X if layer is None else adata.layers[layer]
        if issparse(expr):
            expr = expr.T.toarray()
        else:
            expr = expr.T

        expr_r = ro.conversion.py2rpy(expr)

    design = stats.model_matrix(stats.as_formula(model), data=obs_r)
    dge = edger.DGEList(counts=expr_r, samples=obs_r, genes=var_r)

    contrasts_r = limma.makeContrasts(
        contrasts=[
            f'({"+".join(b)}) / {len(b)} - ({"+".join(a)}) / {len(a)}'
            for a, b in contrasts
        ],
        levels=base.colnames(design),
    )

    logging.info("Calculating NormFactors")
    dge = edger.calcNormFactors(dge)

    logging.info("Estimating Dispersions")
    dge = edger.estimateDisp(dge, design=design)

    logging.info("Fitting linear model")
    fit = edger.glmQLFit(dge, design=design)

    ro.globalenv["fit"] = fit
    ro.globalenv["contrasts"] = contrasts_r

    ro.r(
        """
        library(dplyr)
        de_res = BiocParallel::bplapply(1:ncol(contrasts), function(i) {
            test = edgeR::glmQLFTest(fit, contrast=contrasts[, i])
            edgeR::topTags(test, n=Inf, adjust.method="BH")$table %>%
                mutate(contrast_idx = i - 1)
        }) %>% bind_rows()
        """
    )

    with localconverter(
        ro.default_converter + numpy2ri.converter + pandas2ri.converter
    ):
        return ro.conversion.rpy2py(ro.globalenv["de_res"])


def glmgampoi_rank_genes_groups(
    adata: AnnData,
    *,
    groupby: str,
    contrasts: Sequence[Tuple[Sequence[str], Sequence[str]]],
    cofactors: Sequence[str] = None,
    layer: Optional[str] = None,
    subsample_disp=2000,
    n_cores_per_job: int = 4,
    n_jobs: int = 4,
):
    """
    Perform DE analysis using edgeR.

    Requires that an R installation and the following packages are available

        GlmGamPoi
        BiocParallel
        RhpcBLASctl

    Install them with `conda install bioconductor-glmgampoi bioconductor-biocparallel r-rhpcblasctl`.

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        The column in adata.obs to test for DE
    contrast
        Liste of tuples with tests to perform, e.g.
        `[('A', 'B'), (('A', 'B'), ('C', 'D','E'))]` which is equivalent to
        `[(('A', ), ('B', )), (('A', 'B'), ('C', 'D','E'))]
    cofactors
        Additional columns to include into the model
    layer
        layer in adata that contains raw counts. If None, use `X`.
    subsample_disp
        Subsample cells to this nubmer during estimation of overdispersion.
    n_cores_per_job
        Number of cores to run per job (including BLAS parallelization)
    n_jobs
        Number of tests to run in parallel.
    """

    try:
        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri, numpy2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2 import robjects as ro
    except ImportError:
        raise ImportError("edger requires rpy2 to be installed. ")

    try:
        base = importr("base")
        glm = importr("glmGamPoi")
        stats = importr("stats")
        blasctl = importr("RhpcBLASctl")
        bcparallel = importr("BiocParallel")
    except ImportError:
        raise ImportError(
            "GlmGamPoi requires a valid R installation with the following packages: "
            "glmGamPoi, BiocParallal, RhpcBLASctl"
        )

    blasctl.blas_set_num_threads(n_cores_per_job)
    blasctl.omp_set_num_threads(n_cores_per_job)
    logging.info("Preparing R objects")

    # Define model formula
    cofactors = [] if cofactors is None else _make_names(cofactors)
    groupby = _make_names([groupby])[0]
    model = f"~ 0 + {groupby} + {' + '.join(cofactors)}"
    contrasts = _fix_contrasts(contrasts, groupby)

    bcparallel.register(bcparallel.MulticoreParam(n_jobs))

    with localconverter(ro.default_converter + pandas2ri.converter):
        tmp_obs = adata.obs.loc[:, [groupby] + cofactors]
        tmp_obs.columns = _make_names(tmp_obs.columns)
        for col in tmp_obs.columns:
            if not is_numeric_dtype(tmp_obs[col]):
                tmp_obs[col] = _make_names(tmp_obs[col])

        obs_r = ro.conversion.py2rpy(tmp_obs)

    with localconverter(ro.default_converter + pandas2ri.converter):
        expr = adata.X if layer is None else adata.layers[layer]
        if issparse(expr):
            expr = expr.T.toarray()
        else:
            expr = expr.T

        expr = pd.DataFrame(expr)
        expr.index = adata.var_names
        expr.columns = adata.obs_names

        expr_r = ro.conversion.py2rpy(expr)

    # convert as dataframe and then convert to matrix - didn't keep rownames otherwise.
    expr_r = base.as_matrix(expr_r)

    design = stats.model_matrix(stats.as_formula(model), data=obs_r)
    contrasts = [
        f'({"+".join(b)}) / {len(b)} - ({"+".join(a)}) / {len(a)}' for a, b in contrasts
    ]

    logging.info("Fitting GLM")
    fit = glm.glm_gp(expr_r, design=design, subsample=subsample_disp)
    ro.globalenv["fit"] = fit
    ro.globalenv["contrasts"] = contrasts

    ro.r(
        """
        library(dplyr)
        de_res = BiocParallel::bplapply(contrasts, function(contrast) {
            glmGamPoi::test_de(fit, contrast) %>%
                mutate(comparision = contrast)
        }) %>% bind_rows()
        """
    )

    with localconverter(
        ro.default_converter + numpy2ri.converter + pandas2ri.converter
    ):
        return ro.conversion.rpy2py(ro.globalenv["de_res"])
