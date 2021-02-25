"""Module for helpers with differential gene expression"""

import numpy as np
import pandas as pd


def scvi_rank_genes_groups(adata, model, *, groupby):
    """Perform differential expression with scVI and add the results
    to anndata in the same format as `sc.tl.rank_gene_groups` does.

    Requires that an scvi model is trained on `adata`.
    """
    de_res = model.differential_expression(
        adata, groupby="leiden", batch_correction=True
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
            columns=[c.split()[0] for c in df_groupby.groups.keys()],
        ).to_records(index=False, column_dtypes="O")
    adata.uns["rank_genes_groups"] = res_dict
    return de_res
