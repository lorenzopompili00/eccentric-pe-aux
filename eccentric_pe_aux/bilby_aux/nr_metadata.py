#!/usr/bin/env python
"""Print the SXS metadata.

Examples
--------
    python -m eccentric_pe_aux.bilby_aux.nr_metadata --sim-file /path/to/Strain_N2.h5
    python -m eccentric_pe_aux.bilby_aux.nr_metadata --sxs-id SXS:BBH:1234
"""

import argparse

import numpy as np
import sxs


def _get(md, key, default="n/a"):
    try:
        return md[key]
    except (KeyError, AttributeError, TypeError):
        return default


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim-file", help="path to an SXS strain file")
    g.add_argument("--sxs-id", help="SXS catalog identifier")
    args = p.parse_args()

    if args.sim_file:
        wf = sxs.load(args.sim_file)
        modes = wf
    else:
        wf = sxs.load(args.sxs_id)
        modes = wf.h
    md = wf.metadata

    m1, m2 = _get(md, "reference_mass1"), _get(md, "reference_mass2")
    q = min(m1 / m2, m2 / m1)  # bilby mass_ratio convention (<= 1)
    ells = sorted({int(l) for l, m in modes.LM})

    print(f"reference_time         : {_get(md, 'reference_time')}")
    print(f"mass_ratio (q <= 1)    : {q:.6f}")
    print(
        f"reference spin1        : {np.round(_get(md, 'reference_dimensionless_spin1'), 4)}"
    )
    print(
        f"reference spin2        : {np.round(_get(md, 'reference_dimensionless_spin2'), 4)}"
    )
    print(f"reference_eccentricity : {_get(md, 'reference_eccentricity')}")
    print(f"reference_mean_anomaly : {_get(md, 'reference_mean_anomaly')}")
    print(
        f"orbits after reference : {_get(md, 'number_of_orbits_from_reference_time')}"
    )
    print(f"modes available        : ell {ells[0]}-{ells[-1]} ({len(modes.LM)} modes)")


if __name__ == "__main__":
    main()
