# pipeline.py
# ============================================================
# CottonSeedQualityPipeline — shared class definition
#
# This file MUST be imported by BOTH:
#   • 09_serialization.py  (when saving the bundle)
#   • app.py               (when loading the bundle)
#
# Pickle stores the class location as "pipeline.CottonSeedQualityPipeline".
# As long as pipeline.py is on the Python path, joblib.load() can always
# resolve the class regardless of which script triggered the load.
# ============================================================

import datetime
import numpy as np
import pandas as pd
from lime import lime_tabular


class CottonSeedQualityPipeline:
    """
    End-to-end inference pipeline for cotton seed quality prediction.

    WG model : XGBoost Tuned           (Test R² = 0.9608)
    CT model : GradientBoosting CT  (Test R² = 0.9990)

    Usage
    -----
    bundle   = joblib.load("deployment_bundle/cotton_seed_quality_bundle.joblib")
    pipeline = bundle["pipeline"]
    results  = pipeline.predict(new_field_data_df)

    LIME note
    ---------
    LimeTabularExplainer contains lambda functions that cannot be pickled.
    This pipeline stores lime_params (a plain dict) instead of the explainer.
    Call pipeline.build_lime_explainer() to reconstruct it on demand.
    """

    VERSION    = "2.0.0"
    CREATED_AT = datetime.datetime.now().isoformat()

    # Industry quality benchmarks
    WG_HIGH = 80;  WG_POOR = 70
    CT_HIGH = 70;  CT_POOR = 55
    R_EXCL  = 0.85; R_MIN  = 0.65

    def __init__(self, preprocessor, wg_model, ct_model,
                 le_vigor, feature_names,
                 shap_wg=None, shap_ct=None, lime_params=None,
                 wg_model_name="XGBoost_WG",
                 ct_model_name="GradientBoosting_CT",
                 wg_test_r2=0.9608, ct_test_r2=0.9990):
        self.preprocessor    = preprocessor
        self.wg_model        = wg_model
        self.ct_model        = ct_model
        self.le_vigor        = le_vigor
        self.feature_names   = list(feature_names)
        self.shap_wg         = shap_wg
        self.shap_ct         = shap_ct
        self.lime_params     = lime_params   # dict, NOT the explainer object
        self.wg_model_name   = wg_model_name
        self.ct_model_name   = ct_model_name
        self.wg_test_r2      = wg_test_r2
        self.ct_test_r2      = ct_test_r2

    # ------------------------------------------------------------------
    def _classify(self, wg: float, ct: float) -> str:
        ratio = ct / wg if wg > 0 else 0.0
        if wg >= self.WG_HIGH and ct >= self.CT_HIGH and ratio >= self.R_EXCL:
            return "Excellent Vigor / Viable Seeds"
        if wg >= self.WG_POOR and ct >= self.CT_POOR and ratio >= self.R_MIN:
            return "Good Vigor / Viable Seeds"
        return "Poor Vigor / Non-Viable Risk"

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(
            X.reindex(columns=self.feature_names)
        )

    def _predict_preprocessed(self, X_t: np.ndarray) -> pd.DataFrame:
        """
        Predict directly from an already-preprocessed numpy array.
        Used by Block 9 smoke test and anywhere that already has
        scaled/encoded features (e.g. X_test from 05_data_splits.joblib).
        Bypasses self._preprocess() to avoid double-transformation.
        """
        wg    = np.clip(self.wg_model.predict(X_t), 0, 100)
        ct    = np.clip(self.ct_model.predict(X_t), 0, 100)
        cwvi  = wg * 0.90 + ct
        delta = wg - ct
        ratio = np.where(wg > 0, ct / wg, np.nan)
        wg_flag = pd.cut(wg, bins=[-np.inf,60,70,80,np.inf],
                         labels=["Poor","Marginal","Acceptable","High Quality"])
        ct_flag = pd.cut(ct, bins=[-np.inf,40,55,70,np.inf],
                         labels=["Poor","Marginal","Acceptable","High Quality"])
        return pd.DataFrame({
            "WG_Predicted"   : wg.round(2),
            "CT_Predicted"   : ct.round(2),
            "CWVI"           : cwvi.round(2),
            "Vigor_Gap"      : delta.round(2),
            "CT_WG_Ratio"    : np.round(ratio, 4),
            "Vigor_Class"    : [self._classify(w, c) for w, c in zip(wg, ct)],
            "WG_Quality_Flag": wg_flag,
            "CT_Quality_Flag": ct_flag,
        })

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame — columns must match self.feature_names

        Returns
        -------
        pd.DataFrame with:
            WG_Predicted, CT_Predicted, CWVI, Vigor_Gap,
            CT_WG_Ratio, Vigor_Class, WG_Quality_Flag, CT_Quality_Flag
        """
        X_t = self._preprocess(X)
        wg  = np.clip(self.wg_model.predict(X_t), 0, 100)
        ct  = np.clip(self.ct_model.predict(X_t), 0, 100)

        cwvi  = wg * 0.90 + ct
        delta = wg - ct
        ratio = np.where(wg > 0, ct / wg, np.nan)

        wg_flag = pd.cut(
            wg, bins=[-np.inf, 60, 70, 80, np.inf],
            labels=["Poor", "Marginal", "Acceptable", "High Quality"]
        )
        ct_flag = pd.cut(
            ct, bins=[-np.inf, 40, 55, 70, np.inf],
            labels=["Poor", "Marginal", "Acceptable", "High Quality"]
        )

        return pd.DataFrame({
            "WG_Predicted"   : wg.round(2),
            "CT_Predicted"   : ct.round(2),
            "CWVI"           : cwvi.round(2),
            "Vigor_Gap"      : delta.round(2),
            "CT_WG_Ratio"    : np.round(ratio, 4),
            "Vigor_Class"    : [self._classify(w, c) for w, c in zip(wg, ct)],
            "WG_Quality_Flag": wg_flag,
            "CT_Quality_Flag": ct_flag,
        })

    # ------------------------------------------------------------------
    def predict_with_shap(self, X: pd.DataFrame, n_top: int = 10) -> dict:
        """Return predictions annotated with top-N SHAP feature drivers."""
        results = self.predict(X)
        X_t     = self._preprocess(X)

        wg_sv = self.shap_wg.shap_values(X_t)
        ct_sv = self.shap_ct.shap_values(X_t)
        if isinstance(wg_sv, list): wg_sv = wg_sv[0]
        if isinstance(ct_sv, list): ct_sv = ct_sv[0]

        def top_drivers(sv_row):
            idx = np.argsort(np.abs(sv_row))[::-1][:n_top]
            return {self.feature_names[i]: round(float(sv_row[i]), 4) for i in idx}

        results["WG_SHAP_Drivers"] = [top_drivers(r) for r in wg_sv]
        results["CT_SHAP_Drivers"] = [top_drivers(r) for r in ct_sv]
        return results

    # ------------------------------------------------------------------
    def build_lime_explainer(self):
        """
        Reconstruct the LIME explainer on demand.

        LimeTabularExplainer cannot be pickled (contains lambdas).
        This method rebuilds an identical explainer from the stored
        constructor parameters every time it is called.
        The result is deterministic given the same training_data + random_state.
        """
        if self.lime_params is None:
            raise ValueError(
                "lime_params not set — bundle was saved without LIME params."
            )
        return lime_tabular.LimeTabularExplainer(**self.lime_params)

    def explain_lime(self, X: pd.DataFrame, idx: int,
                     target: str = "WG", num_features: int = 15):
        """Generate a LIME explanation for one row."""
        explainer = self.build_lime_explainer()
        X_t       = self._preprocess(X)
        model     = self.wg_model if target == "WG" else self.ct_model
        return explainer.explain_instance(
            data_row    = X_t[idx],
            predict_fn  = model.predict,
            num_features= num_features,
        )

    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        return {
            "version"    : self.VERSION,
            "created_at" : self.CREATED_AT,
            "wg_model"   : self.wg_model_name,
            "ct_model"   : self.ct_model_name,
            "wg_test_r2" : self.wg_test_r2,
            "ct_test_r2" : self.ct_test_r2,
            "n_features" : len(self.feature_names),
            "benchmarks" : {
                "WG_high"          : self.WG_HIGH,
                "WG_poor"          : self.WG_POOR,
                "CT_high"          : self.CT_HIGH,
                "CT_poor"          : self.CT_POOR,
                "ratio_excellent"  : self.R_EXCL,
                "ratio_min"        : self.R_MIN,
            },
        }
