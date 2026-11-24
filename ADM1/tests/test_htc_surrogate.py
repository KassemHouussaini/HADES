"""HTC surrogate (htc_surrogate.py): training set, regression values, bounds and domain flags."""
import importlib.util, json, os
import numpy as np, pandas as pd
import pytest

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "ML training scripts and models")
FW = dict(T=180, RT=60, WC=85, VM_feed=82.6, FC_feed=8.3, C_feed=38.25, H_feed=5.47,
          N_feed=3.66, O_feed=43.02, S_feed=0.5, Ash_feed=9.1)


@pytest.fixture(scope="module")
def ml():
    spec = importlib.util.spec_from_file_location("htc_surrogate", os.path.join(ML_DIR, "htc_surrogate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_training_set_size(ml):
    assert ml.META["n_rows"] == 938 and ml.META["n_studies"] == 122


def test_params_match_tuning_json(ml):
    best = json.load(open(os.path.join(ML_DIR, "htc_xgb_best_params.json")))
    for t, m in ml.META["targets"].items():
        assert m["params"] == best[t]["params"]


def test_regression_base_food_waste(ml):
    _, inv = ml.HTC_parametric_ML(**FW)
    frozen = {"Y_char (%)": 70.17056274414062, "HHV_char (MJ/kg)": 17.064329147338867,
              "C_char (%)": 39.9316520690918, "N_char (%)": 2.346208333969116,
              "Ash_char (%)": 5.76481819152832}
    for k, v in frozen.items():
        assert inv[k] == pytest.approx(v, rel=1e-5), k


def test_closure_and_bounds(ml):
    _, inv = ml.HTC_parametric_ML(**FW)
    tot = sum(inv[k] for k in ("C_char (%)", "H_char (%)", "O_char (%)", "N_char (%)",
                               "S_char (%)", "Ash_char (%)"))
    assert tot == pytest.approx(100.0)
    assert all(0 <= inv[k] <= 100 for k in ("Y_char (%)", "C_char (%)", "N_char (%)", "Ash_char (%)"))
    assert inv["in_training_range"]


def test_sv_bl_from_moisture(ml):
    # WC 85 % -> water/biomass = 0.85/0.15
    a = ml.HTC_parametric_ML(**FW)[1]
    b = ml.HTC_parametric_ML(**{**FW, "Sv_BL": 0.85 / 0.15})[1]
    assert a["Y_char (%)"] == b["Y_char (%)"]


def test_extrapolation_flagged(ml):
    _, inv = ml.HTC_parametric_ML(**{**FW, "T": 400})
    assert not inv["in_training_range"] and "temp(C)" in inv["domain_violations"]


def test_invalid_moisture_raises(ml):
    with pytest.raises(ValueError):
        ml.HTC_parametric_ML(**{**FW, "WC": 100})


def test_reproduces_training_data_fit(ml):
    # in-sample R2 on the corrected rows must exceed the 10-fold CV R2 (sanity)
    spec = importlib.util.spec_from_file_location("train_htc_surrogate", os.path.join(ML_DIR, "train_htc_surrogate.py"))
    tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
    from sklearn.metrics import r2_score
    df = tr.load_corrected(); X = df[tr.FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["Bio_char(%)"], errors="coerce"); m = y.notna()
    assert r2_score(y[m], ml.models["Bio_char(%)"].predict(X[m])) > 0.865


def test_training_dataset_size():
    """HTC_dataset_paper.xlsx: 938 rows from 122 studies, all at T <= 300 C (frozen training set)."""
    clean = pd.read_excel(os.path.join(ML_DIR, "HTC_dataset_paper.xlsx"), "data")
    assert len(clean) == 938 and clean["Ref"].nunique() == 122
    assert (pd.to_numeric(clean["temp(C)"], errors="coerce") <= 300).all()


def test_joint_domain_check():
    """Layer 2 of the applicability domain: nearest-neighbour distance in the
    standardized input space vs D_T (mean + 2 sd of between-study NN distances)."""
    import importlib.util, os, numpy as np, pandas as pd
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "ML training scripts and models")
    spec = importlib.util.spec_from_file_location("htc_surrogate_ad_test", os.path.join(d, "htc_surrogate.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.AD is not None and 2.0 < m.AD["threshold"] < 3.0
    x = pd.read_excel(os.path.join(d, "HTC_dataset_paper.xlsx"), "data")
    row = x.dropna(subset=m.AD["features"]).iloc[0]
    assert m.ad_distance({f: float(row[f]) for f in m.AD["features"]}) < 1e-6      # a training point
    _, inv = m.HTC_parametric_ML(T=180, RT=60, WC=85, VM_feed=82.6, FC_feed=8.3, C_feed=38.25,
                                 H_feed=5.47, N_feed=3.66, O_feed=43.82, S_feed=0.5, Ash_feed=9.1)
    assert inv["in_domain_knn"] is True and inv["ad_knn_distance"] < inv["ad_knn_threshold"]
    _, inv = m.HTC_parametric_ML(T=300, RT=1440, WC=50, VM_feed=40, FC_feed=0.5, C_feed=80,
                                 H_feed=2, N_feed=12, O_feed=0, S_feed=1.5, Ash_feed=4.5)
    assert inv["in_domain_knn"] is False and np.isfinite(inv["ad_knn_distance"])
