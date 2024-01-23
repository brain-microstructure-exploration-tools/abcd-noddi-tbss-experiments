from pathlib import Path
from dipy.reconst.shm import calculate_max_order, sph_harm_ind_list
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

def aggregate_dipy_response_functions(response_functions):
    """ Aggregate a collection of dipy dti response functions.

    This is useful for taking an average response function over a group when doing a population study.

    The method is to take the geometric mean of the eigenvalues and the arithmetic mean of the non-diffusion-weighted signal.
    The reason for taking the geometric mean of the eigenvalues is that this is exactly taking the Frechet mean of the diffusion
    tensors that are represented by the eigenvalues when the tensors are treated as elements of the standard log-euclidean metric space
    described in 

        Arsigny, Vincent, et al. "Log‐Euclidean metrics for fast and simple calculus on diffusion tensors."
        Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance
        in Medicine 56.2 (2006): 411-421.

    Args:
        response_functions: a sequence of response functions. a response function is the sort of thing
            returned by dipy.reconst.csdeconv.response_from_mask_ssst
    Returns: an aggregate response function.
    """
    
    evals_array = np.array([rf[0] for rf in response_functions])
    S0_array = np.array([rf[1] for rf in response_functions])
    return np.exp(np.log(evals_array).mean(axis=0)), np.mean(S0_array)

def aggregate_dipy_response_functions_workflow(response_dir, output_path):
    """ Aggregate a collection of dipy dti response functions in a directory and write the output to a file.

    This is useful for taking an average response function over a group when doing a population study.

    See aggregate_dipy_response_functions for more details.

    Args:
        response_dir: a directory containing response functions as text files. (for example they
            could be written out by write_dipy_response)
        output_path: file path at which to save the aggregate response
    """
    response_file_paths = list(response_dir.glob("*.txt"))
    if len(response_file_paths) == 0 :
        raise FileNotFoundError(f"No response files found in {response_dir}")
    response_functions = [read_dipy_response(response_file_path) for response_file_path in response_file_paths]
    aggregate_response_function =  aggregate_dipy_response_functions(response_functions)
    write_dipy_response(aggregate_response_function, output_path)

def get_degree_powers(fod_array):
    """Compute "degree powers" of FODs expressed in terms of spherical harmonics.

    The "power" at a specific degree l is the square-sum over all indices m of the spherical harmonic coefficinets c^m_l
    at that given l.

    The idea comes from
       Bloy, Luke, and Ragini Verma. "Demons registration of high angular resolution diffusion images."
       2010 IEEE International Symposium on Biomedical Imaging: From Nano to Macro. IEEE, 2010.

    Args:
        fod_array: an array of even degree spherical harmonic coefficients in the last axis, in the standard ordering 
            l=0, m= 0
            l=2, m=-2,
            l=2, m=-1,
            l=2, m= 0,
            l=2, m= 1,
            l=2, m= 2,
            l=4, m= -4,
            ...

    Retuns: l_values, degree_powers
        l_values: a 1D array listing the degrees
        degree_powers: an array with the same shape as fod_array in all but the final axis. The final axis contains the powers
            at the degrees in the ordering in which they are listed in l_values. So degree_powers[...,i] is the power of fod_array
            in degree l=l_values[i].
    """
    sh_degree_max = calculate_max_order(fod_array.shape[-1])
    m,l = sph_harm_ind_list(sh_degree_max)
    l_values = np.unique(l)
    return l_values, np.stack(
        [
            (fod_array[...,l==l_value]**2).sum(axis=-1)
            for l_value in l_values
        ],
        axis=-1
    )