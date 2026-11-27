"""
find_flue_gas_datasets.py
=========================
Read-only search of the ecoinvent project for flue gas cleaning data, to model
the treatment a permitted hydrochar boiler needs to meet its emission limits
(decision log F19, F20). Writes nothing to the Brightway project.

Context. hydrochar_enduse.compute_enduse_quantities already records, per run,
the mass of SO2, NOx and dust that must be removed to meet the limits of
Directive (EU) 2015/2193 (enduse_*_removed_kg). The supervisor (meeting of
2026-11-27) advised linking that removal to ecoinvent flue gas cleaning data,
which are valid for combustion gases in general, rather than modelling a
treatment technology. This script lists what the database offers:

  A. activities whose name or reference product mentions flue gas cleaning,
     desulfurization, DeNOx (SCR, SNCR), scrubbers, or the residues they make;
  B. inputs of the main reagents (lime, limestone, sodium bicarbonate, urea,
     ammonia) inside the combustion datasets already used by HADES, such as
     the hard coal industrial furnace that models the hydrochar boiler: if the
     template already contains flue gas cleaning, adding it again would count
     it twice.

    python ADM1/find_flue_gas_datasets.py --project my_hades_project --db ecoinvent-3.12-cutoff
"""
import argparse
import re

# terms for part A (case-insensitive, matched on name and reference product)
TERMS = re.compile(r"flue gas|desulph|desulf|denox|\bSCR\b|\bSNCR\b|scrubb|"
                   r"residue from .*cleaning|gypsum.*FGD|FGD", re.I)
# reagents for part B
REAGENTS = re.compile(r"\blime\b|limestone|sodium bicarbonate|sodium hydrogen carbonate|"
                      r"\burea\b|\bammonia\b|activated carbon", re.I)
# combustion templates used by HADES (system_builder, hydrochar_enduse, sweep_engine)
TEMPLATES = [
    "heat production, at hard coal industrial furnace 1-10MW",
    "heat production, natural gas, at boiler condensing modulating >100kW",
    "heat production, softwood chips from forest, at furnace 1000kW",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--db", required=True)
    a = ap.parse_args()
    import bw2data as bd
    bd.projects.set_current(a.project)
    ei = bd.Database(a.db)

    print("A. activities about flue gas cleaning")
    seen = set()
    for act in ei:
        text = f"{act.get('name', '')} | {act.get('reference product', '')}"
        if TERMS.search(text):
            key = (act["name"], act.get("reference product"), act.get("unit"))
            if key in seen:
                continue
            seen.add(key)
            print(f"  {act['name']!r} -> {act.get('reference product')!r} [{act.get('unit')}] "
                  f"locations e.g. {act.get('location')}")
    print(f"  {len(seen)} distinct activities\n")

    print("B. reagent inputs inside the combustion templates used by HADES")
    for name in TEMPLATES:
        hits = [x for x in ei if x.get("name") == name]
        if not hits:
            print(f"  {name}: not found")
            continue
        act = hits[0]
        print(f"  {name} ({act.get('location')}):")
        found = False
        for exc in act.technosphere():
            if REAGENTS.search(exc.input.get("name", "")) or TERMS.search(exc.input.get("name", "")):
                found = True
                print(f"     {exc['amount']:.3e} {exc.input.get('unit')} of {exc.input['name']!r}")
        if not found:
            print("     no flue gas cleaning reagent or residue among its inputs")


if __name__ == "__main__":
    main()
