"""
setup_hades.py
==============
One-time setup of a Brightway project for HADES:
  1. create (or reuse) a Brightway project;
  2. import ecoinvent with your own ecoinvent licence;
  3. import the IMPACT World+ 2.2.1 methods used in the article;
  4. check that the project holds what app.py and case_study.ipynb need.

Usage (from the HADES folder, in the HADES Python environment):

  python setup_hades.py --project hades --version 3.12 --system-model cutoff
        downloads ecoinvent from the ecoinvent servers (asks for your
        ecoinvent username and password, or reads EI_USERNAME / EI_PASSWORD)

  python setup_hades.py --project hades --ecospold "C:/path/to/datasets"
        imports ecoinvent from ecoSpold2 files you downloaded and unzipped
        yourself (the "datasets" folder of the release)

  python setup_hades.py --project hades --iw "C:/path/to/impact_world_plus_2.2.1_....bw2package"
        imports IMPACT World+ from the Brightway file published by CIRAIG on
        Zenodo (take the file matching your ecoinvent version and your
        Brightway version, 2 or 2.5)

The steps can be combined in one call. Re-running is safe: existing
databases are kept unless --overwrite is given.
"""
import argparse
import getpass
import os
import sys

import bw2data as bd
import bw2io as bi

IW_MID = "IMPACT World+ Midpoint 2.2.1_regionalized for ecoinvent v3.12"
IW_DMG = "IMPACT World+ Damage 2.2.1_regionalized for ecoinvent v3.12"
ARTICLE_METHODS = [(IW_MID, "Midpoint", "Climate change, short term"),
                   (IW_DMG, "Human health", "Total human health"),
                   (IW_DMG, "Ecosystem quality", "Total ecosystem quality")]


def credentials():
    user = os.environ.get("EI_USERNAME") or input("ecoinvent username: ").strip()
    pwd = os.environ.get("EI_PASSWORD") or getpass.getpass("ecoinvent password: ")
    return user, pwd


def import_from_server(version, system_model, overwrite):
    name = f"ecoinvent-{version}-{system_model}"
    if name in bd.databases and not overwrite:
        print(f"[ecoinvent] '{name}' already in the project, kept (use --overwrite to replace)")
        return name
    if not hasattr(bi, "import_ecoinvent_release"):
        sys.exit("[ecoinvent] this bw2io version cannot download ecoinvent; update bw2io or use --ecospold")
    user, pwd = credentials()
    print(f"[ecoinvent] downloading and importing ecoinvent {version} {system_model} (10-30 min)...")
    bi.import_ecoinvent_release(version=version, system_model=system_model, username=user, password=pwd)
    return name


def import_from_ecospold(path, version, system_model, overwrite):
    name = f"ecoinvent-{version}-{system_model}"
    if name in bd.databases and not overwrite:
        print(f"[ecoinvent] '{name}' already in the project, kept (use --overwrite to replace)")
        return name
    if not os.path.isdir(path):
        sys.exit(f"[ecoinvent] folder not found: {path}")
    if "biosphere3" not in bd.databases:
        bi.bw2setup()
    if name in bd.databases:
        del bd.databases[name]
    imp = bi.SingleOutputEcospold2Importer(path, name)
    imp.apply_strategies()
    imp.statistics()
    if list(imp.unlinked):
        sys.exit("[ecoinvent] unlinked exchanges remain: check that the ecoSpold2 files match the biosphere database")
    imp.write_database()
    return name


def import_iw(path):
    if not os.path.isfile(path):
        sys.exit(f"[IMPACT World+] file not found: {path}")
    if not path.lower().endswith(".bw2package"):
        sys.exit("[IMPACT World+] expected a .bw2package file (Brightway files of the CIRAIG Zenodo record)")
    print(f"[IMPACT World+] importing {os.path.basename(path)}...")
    bi.BW2Package.import_file(path)


def check(ei_name):
    print("\nProject check")
    ok = True
    eis = [d for d in bd.databases if "ecoinvent" in d.lower() and "biosphere" not in d.lower()]
    print(f"  ecoinvent databases: {eis or 'NONE'}")
    ok &= bool(eis)
    if ei_name and ei_name in bd.databases:
        deps = bd.databases[ei_name].get("depends", [])
        print(f"  '{ei_name}' depends on: {deps}")
    for m in ARTICLE_METHODS:
        found = m in bd.methods
        ok &= found
        print(f"  {'found  ' if found else 'MISSING'} {' | '.join(m)}")
    print("\nReady: start the interface with  streamlit run app.py" if ok else
          "\nNot complete: import the missing items above (the app also lets you pick other indicators).")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--project", required=True, help="Brightway project to create or reuse")
    a.add_argument("--version", default="3.12", help="ecoinvent version (article: 3.12)")
    a.add_argument("--system-model", default="cutoff", help="cutoff, apos or consequential (article: cutoff)")
    a.add_argument("--ecospold", help="folder of ecoSpold2 files (instead of downloading)")
    a.add_argument("--skip-ecoinvent", action="store_true", help="do not import ecoinvent")
    a.add_argument("--iw", help="IMPACT World+ .bw2package file to import")
    a.add_argument("--overwrite", action="store_true", help="replace an existing ecoinvent database")
    args = a.parse_args()

    bd.projects.set_current(args.project)          # creates the project if it does not exist
    print(f"[project] '{args.project}' (Brightway data folder: {bd.projects.dir})")
    ei_name = None
    if not args.skip_ecoinvent:
        ei_name = (import_from_ecospold(args.ecospold, args.version, args.system_model, args.overwrite)
                   if args.ecospold else
                   import_from_server(args.version, args.system_model, args.overwrite))
    if args.iw:
        import_iw(args.iw)
    check(ei_name)


if __name__ == "__main__":
    main()
