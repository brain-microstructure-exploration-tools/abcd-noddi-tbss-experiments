from pathlib import Path
from dipy.reconst.shm import sph_harm_ind_list
from typing import List

def get_unique_file_with_extension(directory_path : Path, extension : str) -> Path:
    l = list(directory_path.glob(f'*.{extension}'))
    if len(l) == 0 :
        raise Exception(f"No {extension} file was found in {directory_path}")
    if len(l) > 1 :
        raise Exception(f"Multiple {extension} files were found in {directory_path}")
    return l[0]

def get_dipy_to_mrtrix_permutation(sh_degree:int) -> List[int]:
    """Get a permutation of indices that transforms between the dipy and mrtrix real spherical harmonic bases.
    Because the change of basis happens to be an involution, this same permutation can be used to go from
    dipy to mrtrix or from mrtrix to dipy.

    See https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675

    Args:
        sh_degree: the \ell_max, i.e. the maximum degree of spherical harmonic to include in the basis

    Returns: A list of integers P representing change of basis permutation, i.e. with the property that
        if f[i] is the i^th coefficient of a function expressed in the dipy basis then f[P[i]] is the i^th
        coefficient of the same function expressed in the mrtrix basis (or vice versa because it happens
        that P is involutive).

    Example usage with dipy:

        ```
        csd_model = ConstrainedSphericalDeconvModel(gtab, response_mrtrix, sh_order=8)
        csd_fit = csd_model.fit(data)
        csd_shm_coeff = csd_fit.shm_coeff
        csd_shm_coeff_mrtrix = csd_shm_coeff[:,:,:,get_dipy_to_mrtrix_permutation(8)]
        save_nifti('output_fod_in_mrtrix_format.nii.gz', csd_shm_coeff_mrtrix, affine, header)
        ```
    """
    m,l = sph_harm_ind_list(sh_degree)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    return permutation
