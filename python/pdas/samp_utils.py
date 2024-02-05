import os
from math import floor

import numpy as np

from pdas.data_utils import load_meshes, load_info_domain


def get_bound_indices(meshdims):

    samples_seed = []
    ndim = len(meshdims)

    if ndim == 1:
        samples_seed += [0, meshdims[0]]

    elif ndim == 2:
        samples_seed += [i * meshdims[0] for i in range(meshdims[1])]  # left
        samples_seed += [i for i in range((meshdims[1] - 1) * meshdims[0], meshdims[0] * meshdims[1])]  # top
        samples_seed += [(meshdims[0] - 1) + i * meshdims[0] for i in range(meshdims[1])]  # right
        samples_seed += [i for i in range(meshdims[0])]  # bottom
    elif ndim == 3:
        raise ValueError("3D seed samples not implemented yet")

    return list(np.unique(samples_seed))


def gen_sample_mesh(
    samptype,
    meshdir,
    outdir,
    samp_bounds=False,
    percpoints=None,
    npoints=None,
    randseed=0,
):
    # expand as necessary
    assert samptype in ["random"]

    # for monolithic, can specify percentage or number, not both
    assert (percpoints is not None) != (npoints is not None)

    if percpoints is not None:
        assert (percpoints > 0.0) and (percpoints <= 1.0)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # get monolithic mesh dimensions
    coords_full, coords_sub = load_meshes(meshdir)
    ndim = coords_full.shape[-1]

    # monolithic sample mesh
    if coords_sub is None:

        meshdims = coords_full.shape[:-1]
        ncells = np.prod(meshdims)

        # "seed" sample indices
        points_seed = []
        if samp_bounds:
            points_seed += get_bound_indices(meshdims)
        npoints_seed = len(points_seed)

        if percpoints is not None:
            npoints = floor(ncells * percpoints)

        assert npoints > 0
        assert npoints <= ncells
        assert npoints >= npoints_seed

        if samptype == "random":
            samples = gen_random_samples(0, ncells-1, npoints, randseed=randseed, points_seed=points_seed)

        outfile = os.path.join(outdir, "sample_mesh_gids.dat")
        print(f"Saving sample mesh global indices to {outfile}")
        np.savetxt(outfile, samples, fmt='%8i')

    # decomposed sample mesh
    else:
        # decomposed only works with percentage, for now
        assert percpoints is not None

        ndom_list, _ = load_info_domain(meshdir)
        ndomains = np.prod(ndom_list)

        for dom_idx in range(ndomains):
            i = dom_idx % ndom_list[0]
            j = int(dom_idx / ndom_list[0])
            k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

            coords_local = coords_sub[i][j][k]
            meshdims = coords_local.shape[:-1]
            ncells = np.prod(meshdims)

            # "seed" sample indices
            points_seed = []
            if samp_bounds:
                points_seed += get_bound_indices(meshdims)
            npoints_seed = len(points_seed)

            # TODO: adjust this if enabling using npoints
            npoints = floor(ncells * percpoints)

            if samptype == "random":
                # have to perturb random seed so sampling isn't the same in uniform subdomains
                samples = gen_random_samples(0, ncells-1, npoints, randseed=randseed+dom_idx, points_seed=points_seed)

            outdir_sub = os.path.join(outdir, f"domain_{dom_idx}")
            if not os.path.isdir(outdir_sub):
                os.mkdir(outdir_sub)
            outfile = os.path.join(outdir_sub, "sample_mesh_gids.dat")
            print(f"Saving sample mesh global indices to {outfile}")
            np.savetxt(outfile, samples, fmt='%8i')


def gen_random_samples(
    low,
    high,
    numsamps,
    randseed=0,
    points_seed=[],
):

    rng = np.random.default_rng(seed=randseed)

    all_samples = np.arange(low, high+1)
    rng.shuffle(all_samples)

    numsamps_seed = len(points_seed)
    if numsamps_seed == 0:
        samples = np.sort(all_samples[:numsamps])
    else:
        samples = points_seed
        idx = 0
        while len(samples) < numsamps:
            if all_samples[idx] not in samples:
                samples.append(all_samples[idx])
            idx += 1
        samples = np.sort(samples)

    return samples