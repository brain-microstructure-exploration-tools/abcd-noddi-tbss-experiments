from pathlib import Path
from dipy.reconst.shm import sph_harm_ind_list
from typing import List
import json
import numpy as np

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


def write_dipy_response(response, filename):
    """
    Write a dipy DTI response function to file.
    
    A dipy DTI response function has the form
        (evals, S0)
    where S0 is a non-diffusion weighted signal and where evals is a numpy array of the form
        (lambda_1, lambda_2, lambda_2)
    which represents a prolate diffusion tensor with unspecified orientation.

    This sort of representation is output by dipy.reconst.csdeconv.response_from_mask_ssst
    for example.
    """

    if not isinstance(response, tuple) or len(response) != 2:
        raise ValueError("response should be a tuple of length two consisting of an eigenvalue array and an S0 value")
    if not isinstance(response[0], np.ndarray) or response[0].shape != (3,):
        raise ValueError("the first element of response should be a numpy array listing three eigenvalues")

    with open(filename, 'w') as file:
        json.dump(
            [
                [eval.item() for eval in response[0]],
                response[1].item() # json won't serialize numpy float type, it has to be native python float, hence ".item()"
            ],
            file
        )

def read_dipy_response(filename):
    """
    Read a dipy DTI response function from file. See the docstring of `write_dipy_response`
    to understand the form of the returned response function.
    """
    with open(filename, 'r') as file:
        response_list = json.load(file)
        return (
            np.array(response_list[0], dtype=float),
            np.float32(response_list[1])
        )