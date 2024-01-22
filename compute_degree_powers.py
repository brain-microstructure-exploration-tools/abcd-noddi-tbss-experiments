from dipy.reconst.shm import calculate_max_order, sph_harm_ind_list
import numpy as np

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

# TODO add argparse and then processing code