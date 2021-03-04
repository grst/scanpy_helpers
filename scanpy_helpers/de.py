"""Module for helpers with differential gene expression"""

import numpy as np
import pandas as pd
from scanpy import logging, AnnData
from scipy.sparse import issparse
from typing import Sequence, Tuple, Optional, Union, Literal, List
from threadpoolctl import threadpool_limits
from pandas.api.types import is_numeric_dtype
from multiprocessing import Pool
import itertools
from statsmodels.stats.multitest import fdrcorrection


def _make_names(seq: Union[str, Sequence[str]]) -> Union[str, List[str]]:
    # TODO put this in util and make it explicit - just raise an error if the valuese
    # don't match my requirements
    """Turn a value into a valid R name.

    Replaces all non-alphanumeric characters with an underscore.

    Raises
    ------
    ValueError
        if the number of unique elements in `seq` has been modified by the operation.
    """

    def _make_name(string):
        return "".join(e if e.isalnum() else "_" for e in string)

    if isinstance(seq, str):
        return _make_name(seq)
    else:
        n_unique = len(set(seq))
        seq_corrected = [_make_name(string) for string in seq]
        if len(set(seq_corrected)) != n_unique:
            raise ValueError(
                "The object contains names that are ambiguous after replacing invalid "
                "characters. Only alphanumeric characters and underscores are allowed. "
            )
        return seq_corrected


def de_res_to_anndata(
    adata: AnnData,
    de_res: pd.DataFrame,
    *,
    groupby: str,
    gene_id_col: str = "gene_symbol",
    score_col: str = "score",
    pval_col: str = "pvalue",
    pval_adj_col: Optional[str] = None,
    lfc_col: str = "lfc",
    key_added: str = "rank_genes_groups",
) -> None:
    """Add a tabular differential expression result to AnnData as
    if it was produced by scanpy.tl.rank_genes_groups.

    Parameters
    ----------
    adata
        annotated data matrix
    de_res
        Tablular de result
    groupby
        column in `de_res` that indicates the group. This column must
        also exist in `adata.obs`.
    gene_id_col
        column in `de_res` that holds the gene identifiers
    score_col
        column in `de_res` that holds the score (results will be ordered by score).
    pval_col
        column in `de_res` that holds the unadjusted pvalue
    pval_adj_col
        column in `de_res` that holds the adjusted pvalue. If not specified, the
        unadjusted pvalues will be FDR-adjusted.
    lfc_col
        column in `de_res` that holds the log fold change
    key_added
        key under which the results will be stored in adata.uns
    """
    if groupby not in adata.obs.columns or groupby not in de_res.columns:
        raise ValueError("groupby column must exist in both adata and de_res. ")
    res_dict = {
        "params": {
            "groupby": groupby,
            "reference": "rest",
            "method": "other",
            "use_raw": True,
            "layer": None,
            "corr_method": "other",
        },
        "names": [],
        "scores": [],
        "pvals": [],
        "pvals_adj": [],
        "logfoldchanges": [],
    }
    df_groupby = de_res.groupby(groupby)
    for _, tmp_df in df_groupby:
        tmp_df = tmp_df.sort_values(score_col, ascending=False)
        res_dict["names"].append(tmp_df[gene_id_col].values)
        res_dict["scores"].append(tmp_df[score_col].values)
        res_dict["pvals"].append(tmp_df[pval_col].values)
        if pval_adj_col is not None:
            res_dict["pvals_adj"].append(tmp_df[pval_adj_col].values)
        else:
            res_dict["pvals_adj"].append(fdrcorrection(tmp_df[pval_col].values)[1])
        res_dict["logfoldchanges"].append(tmp_df[lfc_col].values)

    for key in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
        res_dict[key] = pd.DataFrame(
            np.vstack(res_dict[key]).T,
            columns=list(df_groupby.groups.keys()),
        ).to_records(index=False, column_dtypes="O")
    adata.uns[key_added] = res_dict


def scvi(
    adata: AnnData, model, *, groupby: str, groups="all", **kwargs
) -> pd.DataFrame:
    """Perform differential expression with scVI.

    Essentially a wrapper around `scVI.differential_expression`.

    Parameters
    ----------
    adata
        annotated data matrix
    model
        scVI model trained on adata
    groupby
        column in adata.obs to use for the grouping.
    groups
        Subset of groups, e.g. `['g1', 'g2', 'g3']`, to which comparison shall
        be restricted, or `'all'` (default), for all groups.
    **kwargs
        arguments passed to `scVI.differential_expression`
    """
    de_res = model.differential_expression(
        adata,
        groupby=groupby,
        batch_correction=True,
        group1=None if groups == "all" else groups,
        **kwargs,
    )
    return de_res


def edger(
    adata: AnnData,
    *,
    groupby: str,
    groups: Union[Literal["all"], Sequence[str]],
    cofactors: Sequence[str] = None,
    layer: Optional[str] = None,
    n_cores_per_job: int = 4,
    n_jobs: int = 4,
) -> pd.DataFrame:
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
    groups
        Subset of groups, e.g. `['g1', 'g2', 'g3']`, to which comparison shall
        be restricted, or `'all'` (default), for all groups.
    cofactors
        Additional columns to include into the model
    layer
        layer in adata that contains raw counts. If None, use `X`.
    n_cores_per_job
        Number of cores to run per job (including BLAS parallelization)
    n_jobs
        Number of tests to run in parallel.

    Returns
    -------
    DataFrame with differential expression results
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
            "edgeR, BiocParallel, RhpcBLASctl"
        )

    # Set parallelism
    blasctl.blas_set_num_threads(n_cores_per_job)
    blasctl.omp_set_num_threads(n_cores_per_job)
    bcparallel.register(bcparallel.MulticoreParam(n_jobs))

    logging.info("Preparing R objects")
    cofactor_formula = (
        "" if cofactors is None else f"+ {' + '.join(_make_names(cofactors))}"
    )
    groupby = _make_names(groupby)
    model = f"~ 0 + {groupby} {cofactor_formula}"
    tmp_adata = (
        adata if groups == "all" else adata[adata.obs[groupby].isin(groups), :]
    ).copy()
    tmp_adata.obs.columns = _make_names(tmp_adata.obs.columns)
    for col in tmp_adata.obs.columns:
        if not is_numeric_dtype(tmp_adata.obs[col]):
            tmp_adata.obs[col] = _make_names(tmp_adata.obs[col])
    groups = tmp_adata.obs[groupby].unique()
    if len(groups) < 2:
        raise ValueError("Need at least two groups to compare. ")

    with localconverter(ro.default_converter + pandas2ri.converter):
        obs_r = ro.conversion.py2rpy(
            tmp_adata.obs.loc[:, [groupby] + ([] if cofactors is None else cofactors)]
        )
        # just need the index
        var_r = ro.conversion.py2rpy(
            pd.DataFrame(
                {"gene_symbol": tmp_adata.var_names}, index=tmp_adata.var_names
            )
        )

    with localconverter(ro.default_converter + numpy2ri.converter):
        expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]
        if issparse(expr):
            expr = expr.T.toarray()
        else:
            expr = expr.T

        expr_r = ro.conversion.py2rpy(expr)

    design = stats.model_matrix(stats.as_formula(model), data=obs_r)
    dge = edger.DGEList(counts=expr_r, samples=obs_r, genes=var_r)

    contrasts_r = limma.makeContrasts(
        contrasts=[
            f'({"+".join([f"{groupby}{g}" for g in groups if g != group])})'
            f" / {len(groups) - 1}"
            f" - {groupby}{group}"
            for group in groups
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
        de_res = ro.conversion.rpy2py(ro.globalenv["de_res"])

    # TODO fix this
    #  de_res["group"] = [groups[i] for i in de_res["contrast_idx"]]

    return de_res


def glm_gam_poi(
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


def mast(
    adata: AnnData,
    *,
    groupby: str,
    groups: Union[Literal["all"], Sequence[str]],
    cofactors: Sequence[str] = None,
    layer: Optional[str] = None,
    n_cores_per_job: int = 4,
    n_jobs: int = 4,
):
    """
    Perform DE analysis using edgeR.

    Requires that an R installation and the following packages are available

        MAST
        BiocParallel

    Install them with `conda install bioconductor-mast bioconductor-biocparallel`.

    Parameters
    ----------
    adata
        annotated data matrix. X must contain normalized and log-transformed values.
    groupby
        The column in adata.obs to test for DE
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
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2 import robjects as ro
        import anndata2ri
    except ImportError:
        raise ImportError("MAST requires rpy2 and anndata2ri to be installed. ")

    try:
        mast = importr("MAST")
        bcparallel = importr("BiocParallel")
    except ImportError:
        raise ImportError(
            "MAST requires a valid R installation with the following packages: "
            "MAST, BiocParallel"
        )

    bcparallel.register(bcparallel.MulticoreParam(n_jobs))

    logging.info("Preparing AnnData")
    tmp_adata = AnnData(
        X=adata.X if layer is None else adata.layers[layer],
        obs=adata.obs,
        var=adata.var,
    )
    tmp_adata.obs.columns = _make_names(tmp_adata.obs.columns)
    tmp_adata.obs[groupby] = _make_names(tmp_adata.obs[groupby])
    contrasts = []
    for group in tmp_adata.obs[groupby].unique():
        contrasts.append(f"is_group_{group}")
        tmp_adata.obs[f"is_group_{group}"] = tmp_adata.obs[groupby] == group

    logging.info("Preparing R objects")
    with localconverter(anndata2ri.converter):
        sce = ro.conversion.py2rpy(tmp_adata)
    sca = mast.SceToSingleCellAssay(sce)
    groupby = _make_names([groupby])[0]
    cofactor_formula = (
        "" if cofactors is None else "+ " + " + ".join(_make_names(cofactors))
    )

    logging.info("Running MAST")
    ro.globalenv["cpus_per_thread"] = n_cores_per_job
    ro.globalenv["contrasts"] = contrasts
    ro.globalenv["cofactor_formula"] = cofactor_formula
    ro.globalenv["sca"] = sca
    ro.r(
        """
        library(dplyr)
        de_res = bplapply(contrasts, function(model_col) {
            op = options(mc.cores=cpus_per_thread)
            on.exit(options(op))
            contrast_to_test = paste0(model_col, "TRUE")
            fit = zlm(as.formula(paste0("~", model_col, cofactor_formula)), sca)
            res = summary(fit, doLRT=contrast_to_test)$datatable
            merge(
                res[contrast==contrast_to_test & component=='H', .(primerid, `Pr(>Chisq)`)], #P-vals
                res[contrast==contrast_to_test & component=='logFC', .(primerid, coef)],
                by='primerid'
            ) %>% mutate(comparison=model_col)                  
        }) %>% bind_rows()
        """
    )

    with localconverter(ro.default_converter + pandas2ri.converter):
        de_res = ro.conversion.rpy2py(ro.globalenv["de_res"])

    de_res["comparison"] = de_res["comparison"].str.replace("is_group_", "")
    return de_res