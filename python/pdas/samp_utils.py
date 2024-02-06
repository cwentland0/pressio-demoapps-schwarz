import os
from math import floor

import numpy as np
from scipy.linalg import qr

from pdas.data_utils import load_meshes, load_info_domain
from pdas.prom_utils import load_pod_basis


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
    basis_dir=None,
    seed_qdeim=False,
    nmodes=None,
    samp_bounds=False,
    percpoints=None,
    npoints=None,
    randseed=0,
):
    # expand as necessary
    assert samptype in ["random", "eigenvec", "gnat"]

    # for monolithic, can specify percentage or number, not both
    assert (percpoints is not None) != (npoints is not None)

    if percpoints is not None:
        assert (percpoints > 0.0) and (percpoints <= 1.0)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    load_basis = False
    if seed_qdeim or (samptype in ["eigenvec", "gnat"]):
        load_basis = True
        assert basis_dir is not None
        assert os.path.isdir(basis_dir)
        assert nmodes is not None

    # get monolithic mesh dimensions
    coords_full, coords_sub = load_meshes(meshdir)
    ndim = coords_full.shape[-1]

    # monolithic sample mesh
    if coords_sub is None:

        meshdims = coords_full.shape[:-1]
        ncells = np.prod(meshdims)

        if load_basis:
            assert isinstance(nmodes, int)
            basis, = load_pod_basis(basis_dir, nmodes=nmodes)

        # "seed" sample indices
        points_seed = []
        if seed_qdeim:
            points_seed += calc_qdeim_samples(basis, ncells)
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
        elif samptype == "eigenvec":
            samples = calc_eigenvec_samples(basis, ncells, npoints, points_seed=points_seed)

        assert samples.shape[0] == npoints
        outfile = os.path.join(outdir, "sample_mesh_gids.dat")
        print(f"Saving sample mesh global indices to {outfile}")
        np.savetxt(outfile, samples, fmt='%8i')

    # decomposed sample mesh
    else:
        # decomposed only works with percentage, for now
        assert percpoints is not None

        ndom_list, _ = load_info_domain(meshdir)
        ndomains = np.prod(ndom_list)

        if load_basis:
            if isinstance(nmodes, int):
                nmodes = [nmodes] * ndomains
            else:
                assert isinstance(nmodes, list)
            assert len(nmodes) == ndomains
            basis_list, = load_pod_basis(basis_dir, nmodes=nmodes)

        for dom_idx in range(ndomains):
            i = dom_idx % ndom_list[0]
            j = int(dom_idx / ndom_list[0])
            k = int(dom_idx / (ndom_list[0] * ndom_list[1]))

            coords_local = coords_sub[i][j][k]
            meshdims = coords_local.shape[:-1]
            ncells = np.prod(meshdims)

            # "seed" sample indices
            points_seed = []
            if seed_qdeim:
                points_seed += calc_qdeim_samples(basis_list[dom_idx], ncells)
            if samp_bounds:
                points_seed += get_bound_indices(meshdims)
            npoints_seed = len(points_seed)

            # TODO: adjust this if enabling using npoints
            npoints = floor(ncells * percpoints)

            if samptype == "random":
                # have to perturb random seed so sampling isn't the same in uniform subdomains
                samples = gen_random_samples(0, ncells-1, npoints, randseed=randseed+dom_idx, points_seed=points_seed)
            elif samptype == "eigenvec":
                samples = calc_eigenvec_samples(basis_list[dom_idx], ncells, npoints, points_seed=points_seed)

            assert samples.shape[0] == npoints
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


def calc_qdeim_samples(basis, ncells):
    nmodes = basis.shape[1]

    _, _, dof_ids = qr(basis.T, pivoting=True, mode="economic")
    samp_ids = np.unique(dof_ids[:nmodes] % ncells)

    return list(samp_ids)


def calc_eigenvec_samples(
    basis,
    ncells,
    numsamps,
    points_seed=[],
):
    ndof_per_cell = round(basis.shape[0] / ncells)

    assert len(points_seed) > 0
    samp_ids = np.array(points_seed, dtype=np.int32)

    while samp_ids.shape[0] < numsamps:
        dof_ids = np.concatenate([samp_ids * (i + 1) for i in range(ndof_per_cell)])
        basis_samp = basis[dof_ids, :]
        _, _, svecs_right = np.linalg.svd(basis_samp, full_matrices=False)
        resvec = np.squeeze(np.square(svecs_right[:, [-1]].T @ basis.T))
        sort_idxs = np.flip(np.argsort(resvec))

        for sort_idx in sort_idxs:
            cell_id_samp = sort_idx % ncells
            if cell_id_samp not in samp_ids:
                samp_ids = np.append(samp_ids, [cell_id_samp])
                samp_ids = np.sort(samp_ids)
                break

    return samp_ids

