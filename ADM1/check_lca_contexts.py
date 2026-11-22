"""
check_lca_contexts.py
=====================
For each regional preset, list which ecoinvent dataset and location every
regional background role resolves to, and flag fallbacks (a dataset taken
from CA, RoW or GLO instead of the province). Read-only: writes nothing to
the Brightway project.

    python ADM1/check_lca_contexts.py --project my_hades_project \
        --db ecoinvent-3.12-cutoff [--presets QC ON AB] [--out results/lca_context_datasets.csv]
"""
import argparse
import contextlib
import importlib.util
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# same queries as notebooks/HADES.ipynb (section 2), location-dependent roles only
BG_ROLES = {
    "elec": ("market for electricity, medium voltage", "market for electricity, medium voltage"),
    "coal_heat": ("heat production, at hard coal industrial furnace 1-10MW", "hard coal industrial furnace"),
    "ad_heat": ("heat production, natural gas, at boiler condensing modulating >100kW", "heat production, natural gas, at boiler condensing modulating >100kW"),
    "ng_heat_market": ("market for heat, district or industrial, natural gas",
                       "market for heat, district or industrial, natural gas"),
    "tap_water": ("market for tap water", "market for tap water"),
    "wastewater_treat": ("treatment of wastewater, average", "treatment of wastewater, average"),
}


def check(ei, bwl, hce, lcx, keys):
    rows = []
    for key in keys:
        p = lcx.get_preset(key)
        loc = p["location_priority"]
        for role, (q, nf) in BG_ROLES.items():
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    a = bwl.find_process(ei, q, loc, name_filter=nf)
                    got, name = a.get("location"), a["name"]
                except (ValueError, LookupError) as e:
                    got, name = None, f"NOT FOUND: {e}"
            rows.append(dict(preset=key, role=role, dataset=name, location=got,
                             fallback=got != loc[0], in_priority=got in loc))
        for n, prod, _w in (lcx.ng_heat_datasets(p) or []):
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    got = bwl.find_exact(ei, n, loc, product=prod).get("location")
                except LookupError:
                    got, n = None, n + " (NOT FOUND)"
            rows.append(dict(preset=key, role="biomethane credit (NG heat)", dataset=n, location=got,
                             fallback=got != loc[0], in_priority=got in loc))
        ds = lcx.heat_mix_datasets(p)
        for grp in lcx.heat_mix_shares(p):
            for n, prod, _w in ds[grp]:
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        a = bwl.find_exact(ei, n, loc, product=prod)
                        got = a.get("location")
                    except LookupError:
                        got, n = None, n + " (NOT FOUND)"
                rows.append(dict(preset=key, role=f"heat mix: {grp}", dataset=n, location=got,
                                 fallback=got != loc[0], in_priority=got in loc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", default="ecoinvent-3.12-cutoff")
    ap.add_argument("--presets", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import bw2data as bd
    import pandas as pd
    bd.projects.set_current(a.project)
    sb = _load("system_builder", "system_builder.py")
    lcx = _load("lca_contexts", "lca_contexts.py")
    ei = bd.Database(a.db)
    keys = a.presets or [k for k in lcx.PRESETS if k != "QC_PAPER1"]
    df = pd.DataFrame(check(ei, sb.bwl, sb.hce, lcx, keys))
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    fb = df[df["fallback"]]
    print(f"\n{len(fb)} of {len(df)} lookups fall back from the province location.")
    bad = df[~df["in_priority"]]
    if len(bad):
        print(f"{len(bad)} lookups found NO dataset in the priority locations:")
        print(bad.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"written {a.out}")


if __name__ == "__main__":
    main()
