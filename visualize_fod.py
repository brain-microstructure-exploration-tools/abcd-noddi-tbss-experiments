# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# The goal of this notebook is to examine a FOD output and visualize it without using any purpose made visualization tool, requiring me to prove that I understand how to interpret the FOD coefficients. It succeeds in using vtk.js to render FODs that look exactly like the ones shown by mrview.

# %%
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.special import sph_harm
import spherical, quaternionic
from IPython.display import display, Javascript
import pandas as pd

# %% [markdown]
# # Create spherical meshgrid and sample spherical harmonic basis

# %%
num_theta = 30
num_phi = 60
thetas = np.linspace(0,np.pi,num=num_theta,endpoint=True) # Note that we include endpoint for theta! The total number of them is still num_theta
phis = np.linspace(0,2*np.pi,num=num_phi,endpoint=False)
th, ph = np.meshgrid(thetas, phis)

# %%
sphere_points = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1).reshape(-1,3)

# %%
f = lambda i,j : i+j*num_theta # map from pair of theta_index,phi_index to a flattened single index
polys_list = []
for i in range(num_theta):
    for j in range(num_phi):
        polys_list += [4,f(i,j), f((i+1)%num_theta, j), f((i+1)%num_theta, (j+1)%num_phi), f(i, (j+1)%num_phi)]


# %%
def sph_harm_l_m(l_max):
    for l in range(0,l_max+1,2):
        for m in range(-l,l+1):
            yield l,m
l, m = np.array(list(sph_harm_l_m(8)),dtype=int).T


# %%
# follows formula at https://mrtrix.readthedocs.io/en/latest/concepts/spherical_harmonics.html#storage-conventions
def sph_harm_real(m,l,ph,th):
    y = sph_harm(m,l,ph,th)
    ynegm = sph_harm(-m,l,ph,th)
    y = np.where(m<0,np.sqrt(2)*np.imag(ynegm),y)
    y = np.where(m>0,np.sqrt(2)*np.real(y),y)
    return np.real_if_close(y)


# %%
sph_harm_vals = sph_harm_real(m[:,np.newaxis],l[:,np.newaxis],ph.reshape(1,-1),th.reshape(1,-1))

# %% [markdown]
# # Dipy-mrtrix conversion
#
# [See my findings here.](https://github.com/dipy/dipy/discussions/2959#discussioncomment-7481675)

# %%
from dipy.reconst.shm import sph_harm_ind_list
def get_mrtrix_to_dipy_conversion_matrix(sh_degree_max):
    m,l = sph_harm_ind_list(sh_degree_max)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    
    permutation_matrix = np.eye(dimensionality)[:,permutation]
    # This matrix can be applied from the left (i.e. summing over dim 1) to coefficients in the mrtrix basis to obtain cofficients in the dipy basis
    # (Because the permutation is idempotent, it doesn't actually matter if you apply from left or right, and in fact the conversion
    # from mrtrix coefficients to dipy coefficients is the same as the conversion the other way around)
    
    return permutation_matrix

# I found it easier and faster to work with the permutation directly instead of a permutation matrix:

def get_dipy_to_mrtrix_permutation(sh_degree_max):
    m,l = sph_harm_ind_list(sh_degree_max)
    basis_indices = list(zip(l,m)) # dipy basis ordering
    dimensionality = len(basis_indices)
    basis_indices_permuted = list(zip(l,-m)) # mrtrix basis ordering
    permutation = [basis_indices.index(basis_indices_permuted[i]) for i in range(dimensionality)] # dipy to mrtrix permution
    
    return permutation


# %% [markdown]
# # Load example mrtrix and dipy CSD results

# %%
fod_path = Path('csd_output_mrtrix/fod/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_fod.nii.gz')
fod_dipy_path = Path('csd_output_dipy/fod/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_fod.nii.gz')
dwi_path = Path('extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi.nii')

# %%
dwi = nib.load(dwi_path)
dwi_array = dwi.get_fdata()

# %%
fod = nib.load(fod_path)
fod_array = fod.get_fdata()
fod_dipy = nib.load(fod_dipy_path)
fod_dipy_array = fod_dipy.get_fdata()

# %% [markdown]
# The mrtrix image can be viewed in mrview as follows:
#
# ```sh
# mrview extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi.nii -odf.load_sh csd_output_mrtrix_msmt/fod/WM/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_wmfod.nii.gz
# ```
#
# The dipy generated fods (from ordinary CSD) can be viewed as follows:
# ```sh
# mrview extracted_images/NDARINV1JXDFV9Z_baselineYear1Arm1_ABCD-MPROC-DTI_20161206184105/sub-NDARINV1JXDFV9Z/ses-baselineYear1Arm1/dwi/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi.nii -odf.load_sh csd_output_dipy/fod/sub-NDARINV1JXDFV9Z_ses-baselineYear1Arm1_run-01_dwi_fod_mrtrixResponse.nii.gz
# ```
#
# In order to get the voxel indices displayed in the mrview voxel info area to match array indices of `dwi_array`, the first axis needs to be reversed:

# %%
dwi_array = dwi_array[::-1]
fod_array = fod_array[::-1]
fod_dipy_array = fod_dipy_array[::-1]

# %% [markdown]
# I do this transformation because I want to use mrview to compare my FOD renders to the FOD shapes as they are intepreted by the framework that generated them (mrtrix). I don't know why the first index gets reversed. Maybe it has to do with the minus sign in the affine:

# %%
dwi.affine

# %%
dwi.header['srow_z']


# %% [markdown]
# 🤷

# %% [markdown]
# # Wigner matrices

# %%
def wigner_D_mrtrixbasis(R, ell_max):
    """Get analogue of wigner D matrix for the real spherical harmonic basis, using the mrtrix convention for the basis.
    The returned array is flat and indexed similarly to spherical.Wigner.
    The argument R is a quanternionic array representing a rotation.
    """
    wigner = spherical.Wigner(ell_max)
    D = wigner.D(R)
    Dr = lambda l,m,mp : np.real(D[spherical.WignerDindex(l,m,mp)])
    Di = lambda l,m,mp : np.imag(D[spherical.WignerDindex(l,m,mp)])
    DM = np.zeros_like(D,dtype=np.double)
    for l in range(ell_max+1):
        # m < 0
        for m in range(-l,0):
            for mp in range(-l,0):
                DM[spherical.WignerDindex(l,m,mp)] = (-1)**(mp+1) * Dr(l,-m,mp) + Dr(l,-m,-mp)
            DM[spherical.WignerDindex(l,m,0)] = np.sqrt(2) * Di(l,-m,0)
            for mp in range(1,l+1):
                DM[spherical.WignerDindex(l,m,mp)] = (-1)**mp * Di(l,-m,-mp) + Di(l,-m,mp)
    
        # m == 0
        for mp in range(-l,0):
            DM[spherical.WignerDindex(l,0,mp)] = - np.sqrt(2) * Di(l,0,-mp)
        DM[spherical.WignerDindex(l,0,0)] = np.real_if_close(D[spherical.WignerDindex(l,0,0)])
        for mp in range(1,l+1):
            DM[spherical.WignerDindex(l,0,mp)] = np.sqrt(2) * Dr(l,0,mp)
        
        # m > 0
        for m in range(1,l+1):
            for mp in range(-l,0):
                DM[spherical.WignerDindex(l,m,mp)] = -( (-1)**(mp+1) * Di(l,m,mp) + Di(l,m,-mp) )
            DM[spherical.WignerDindex(l,m,0)] = np.sqrt(2) * Dr(l,m,0)
            for mp in range(1,l+1):
                DM[spherical.WignerDindex(l,m,mp)] = (-1)**mp * Dr(l,m,-mp) + Dr(l,m,mp)
    return DM

def get_rotation_matrix_mrtrixbasis(R, ell_max):
    l, m = np.array(list(sph_harm_l_m(ell_max)),dtype=int).T
    DM = wigner_D_mrtrixbasis(R, ell_max)
    DM_mat = np.zeros((len(l),len(l)),dtype=np.double)
    for i in range(len(l)):
        for j in range(len(l)):
            if l[i] == l[j]:
                DM_mat[i,j] = DM[spherical.WignerDindex(l[i],m[i],m[j])]
    return DM_mat


# %% [markdown]
# # Visualize FODs

# %%
vtk_js_viewer_code = """
const script = document.createElement('script');
script.src = 'https://unpkg.com/vtk.js';
script.onload = () => {
  const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
  const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [0,0,0] });
  const actor = vtk.Rendering.Core.vtkActor.newInstance();
  const mapper = vtk.Rendering.Core.vtkMapper.newInstance();

  const polydata = vtk.Common.DataModel.vtkPolyData.newInstance();
  polydata.getPoints().setData(Float32Array.from(pointsDataFromPython), 3);
  polydata.getPolys().setData(Uint32Array.from(polysDataFromPython));

  const num_points = pointsDataFromPython.length/3;
  const colorsArray = new Float32Array(num_points*3);
  for (let j = 0; j < num_points; j++) {
    colorsArray[j * 3] = colorsFromPython[j][0];     // R
    colorsArray[j * 3 + 1] = colorsFromPython[j][1]; // G
    colorsArray[j * 3 + 2] = colorsFromPython[j][2]; // B
  }
  const colorData = vtk.Common.Core.vtkDataArray.newInstance({
    numberOfComponents: 3, // RGB
    values: colorsArray,
    name: 'Colors',
  });
  polydata.getPointData().setScalars(colorData);
  
  const normalsFilter = vtk.Filters.Core.vtkPolyDataNormals.newInstance();
  normalsFilter.setInputData(polydata);

  // (vtkAxesActor is not as useful in these visuals because the arrows are too fat and cannot be customized)
  /* const axesActor = vtk.Rendering.Core.vtkAxesActor.newInstance();
  axesActor.setScale([0.1,0.1,0.1])
  renderer.addActor(axesActor); */

  // Create three arrows for axes.
  [[1,0,0],[0,1,0],[0,0,1]].forEach( direction => {
    const arrowSource = vtk.Filters.Sources.vtkArrowSource.newInstance({
      direction: direction,
      tipRadius: 0.01,
      shaftRadius: 0.003,
      tipLength: 0.01,
    });
    const axisMapper = vtk.Rendering.Core.vtkMapper.newInstance();
    axisMapper.setInputConnection(arrowSource.getOutputPort());
    const axisActor = vtk.Rendering.Core.vtkActor.newInstance();
    axisActor.setScale([arrowScaleFromPython,arrowScaleFromPython,arrowScaleFromPython]);
    axisActor.setMapper(axisMapper);
    axisActor.getProperty().setColor(...direction);
    renderer.addActor(axisActor);
  });
    
  
  renderWindow.addRenderer(renderer);
  renderer.addActor(actor);
  actor.setMapper(mapper);
  mapper.setInputConnection(normalsFilter.getOutputPort());
  mapper.setScalarVisibility(true);
  mapper.setColorModeToDirectScalars();
  renderer.resetCamera();
  
  const openGLRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
  renderWindow.addView(openGLRenderWindow);
  
  const container = document.createElement('div');
  container.style.width = '800px';
  container.style.height = '600px';
  element.appendChild(container);
  openGLRenderWindow.setContainer(container);
  
  const { width, height } = container.getBoundingClientRect();
  openGLRenderWindow.setSize(width, height);
  
  const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
  interactor.setView(openGLRenderWindow);
  interactor.initialize();
  interactor.bindEvents(container);
  
  const interactorStyle = vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
  interactor.setInteractorStyle(interactorStyle);
  
  renderWindow.render();
};
document.head.appendChild(script);
"""

def view_voxel(i,j,k,fod_array):
    fod_vals = (fod_array[i,j,k] @ sph_harm_vals)
    scaled_sphere_pts = fod_vals[:,np.newaxis] * sphere_points

    colors = np.zeros_like(scaled_sphere_pts,dtype=float)
    neg_fod_mask = fod_vals<0
    colors[neg_fod_mask] = np.array([205, 92, 92])/255 # negative: red
    colors[~neg_fod_mask] = np.array([92,205,92])/255 # positive: blue
    
    js_code = f"""
    const pointsDataFromPython = {list(scaled_sphere_pts.reshape(-1))};
    const polysDataFromPython = {polys_list};
    const colorsFromPython = {list(map(list,colors))};
    const arrowScaleFromPython = {3*scaled_sphere_pts.max()};
    """
    js_code += vtk_js_viewer_code
    display(Javascript(js_code))


# %% [markdown]
# ## Compare mrtrix and dipy FODs, using our change of basis to convert dipy to mrtrix

# %%
view_voxel(70,72,70,fod_array)

# %%
view_voxel(70,72,70,fod_dipy_array[:,:,:,get_dipy_to_mrtrix_permutation(8)])

# %% [markdown]
# ## Verify that wigner matrices achieve rotation

# %%
vtk_js_viewer_code_rotations = """
const script = document.createElement('script');
script.src = 'https://unpkg.com/vtk.js';
script.onload = () => {
  const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
  const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [0,0,0] });
  const actor = vtk.Rendering.Core.vtkActor.newInstance();
  const mapper = vtk.Rendering.Core.vtkMapper.newInstance();

  let i = 0

  const polydata = vtk.Common.DataModel.vtkPolyData.newInstance();
  polydata.getPoints().setData(Float32Array.from(pointsDataFromPython[i]), 3);
  polydata.getPolys().setData(Uint32Array.from(polysDataFromPython));

  const num_points = pointsDataFromPython[i].length/3;
  const colorsArray = new Float32Array(num_points*3);
  for (let j = 0; j < num_points; j++) {
    colorsArray[j * 3] = colorsFromPython[i][j][0];     // R
    colorsArray[j * 3 + 1] = colorsFromPython[i][j][1]; // G
    colorsArray[j * 3 + 2] = colorsFromPython[i][j][2]; // B
  }
  const colorData = vtk.Common.Core.vtkDataArray.newInstance({
    numberOfComponents: 3, // RGB
    values: colorsArray,
    name: 'Colors',
  });
  polydata.getPointData().setScalars(colorData);
  
  const normalsFilter = vtk.Filters.Core.vtkPolyDataNormals.newInstance();
  normalsFilter.setInputData(polydata);

  const arrowSource = vtk.Filters.Sources.vtkArrowSource.newInstance({
    direction: axisFromPython,
    tipRadius: 0.01,
    shaftRadius: 0.003,
    tipLength: 0.01,
  });
  const axisMapper = vtk.Rendering.Core.vtkMapper.newInstance();
  axisMapper.setInputConnection(arrowSource.getOutputPort());
  const axisActor = vtk.Rendering.Core.vtkActor.newInstance();
  axisActor.setScale([arrowScaleFromPython,arrowScaleFromPython,arrowScaleFromPython]);
  axisActor.setMapper(axisMapper);

  renderWindow.addRenderer(renderer);
  renderer.addActor(actor);
  renderer.addActor(axisActor);
  actor.setMapper(mapper);
  mapper.setInputConnection(normalsFilter.getOutputPort());
  mapper.setScalarVisibility(true);
  mapper.setColorModeToDirectScalars();
  renderer.resetCamera();
  
  const openGLRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
  renderWindow.addView(openGLRenderWindow);
  
  const container = document.createElement('div');
  container.style.width = '800px';
  container.style.height = '600px';
  element.appendChild(container);
  openGLRenderWindow.setContainer(container);
  
  const { width, height } = container.getBoundingClientRect();
  openGLRenderWindow.setSize(width, height);
  
  const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
  interactor.setView(openGLRenderWindow);
  interactor.initialize();
  interactor.bindEvents(container);
  
  const interactorStyle = vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
  interactor.setInteractorStyle(interactorStyle);
  
  renderWindow.render();

  function animate() {
    i = (i+1)%pointsDataFromPython.length;

    for (let j = 0; j < num_points; j++) {
      colorsArray[j * 3] = colorsFromPython[i][j][0];     // R
      colorsArray[j * 3 + 1] = colorsFromPython[i][j][1]; // G
      colorsArray[j * 3 + 2] = colorsFromPython[i][j][2]; // B
    }
    colorData.modified();
    
    polydata.getPoints().setData(Float32Array.from(pointsDataFromPython[i]), 3);
    polydata.modified();
    renderWindow.render();
  }
  setInterval(animate, 400);
};
document.head.appendChild(script);
"""

def view_rotation(fod, rot_axis):
    
    m,l = sph_harm_ind_list(8)

    fods_list = []
    rot_axis = np.array(rot_axis, dtype=np.double)
    rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
    for rot_angle in np.linspace(0,2*np.pi,50,endpoint=False):
        rot = get_rotation_matrix_mrtrixbasis(quaternionic.array.from_axis_angle(rot_angle*rot_axis), 8)
        fods_list.append(rot @ fod)
    fods = np.stack(fods_list,axis=0)
    fod_vals = (fods @ sph_harm_vals)
    
    scaled_sphere_pts = fod_vals[...,np.newaxis] * sphere_points[np.newaxis]
    
    colors = np.zeros_like(scaled_sphere_pts,dtype=float)
    neg_fod_mask = fod_vals<0
    colors[neg_fod_mask] = np.array([205, 92, 92])/255 # negative: red
    colors[~neg_fod_mask] = np.array([92,205,92])/255 # positive: blue
    
    js_code = f"""
    const pointsDataFromPython = {scaled_sphere_pts.reshape(scaled_sphere_pts.shape[0],-1).tolist()};
    const polysDataFromPython = {polys_list};
    const colorsFromPython = {colors.tolist()};
    const axisFromPython = {rot_axis.tolist()};
    const arrowScaleFromPython = {3*scaled_sphere_pts.max()};
    """
    js_code += vtk_js_viewer_code_rotations
    display(Javascript(js_code))


# %%
view_rotation(fod_array[76,73,70], [2,3,-5])

# %% [markdown]
# # Compute "degree powers" or "band energies"

# %%
from dipy.reconst.shm import sph_harm_ind_list
def get_degree_powers(fod_array, sh_degree_max:int):
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
        sh_degree_max: the maximum degree

    Retuns: l_values, degree_powers
        l_values: a 1D array listing the degrees
        degree_powers: an array with the same shape as fod_array in all but the final axis. The final axis contains the powers
            at the degrees in the ordering in which they are listed in l_values. So degree_powers[...,i] is the power of fod_array
            in degree l=l_values[i].
    """
    m,l = sph_harm_ind_list(sh_degree_max)
    l_values = np.unique(l)
    return l_values, np.stack(
        [
            (fod_array[...,l==l_value]**2).sum(axis=-1)
            for l_value in l_values
        ],
        axis=-1
    )


# %% [markdown]
# ## Verify that these are invariant to rotation

# %%
i,j,k = [76,73,70] # voxel indices to test
rot_axis = np.array([1,2,3], dtype=np.double) # rotation axis to test
rot_angle = np.pi/6 # rotation angle to test


rot_axis = rot_axis / np.sqrt(np.sum(rot_axis**2))
rot = get_rotation_matrix_mrtrixbasis(quaternionic.array.from_axis_angle(rot_angle*rot_axis), 8)

fod = fod_array[76,73,70]
fod_rotated = rot @ fod

dp_l_vals, dp = get_degree_powers(fod, 8)
_, dp2 = get_degree_powers(fod_rotated, 8)

# %%
# Observe that the coefficients are definitely quite different after rotation
pd.DataFrame({'coeffs before rot':fod, 'coeffs after rot':fod_rotated}, index=zip(l,m))

# %%
# However the degree-powers/band-energies are preserved by the rotation
pd.DataFrame({'degree powers before rot':dp, 'degree powers after rot':dp2},index=dp_l_vals)

# %% [markdown]
# # Visualize basis

# %%
vtk_js_viewer_code_basis = """
const script = document.createElement('script');
script.src = 'https://unpkg.com/vtk.js';
script.onload = () => {
  const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
  const renderer = vtk.Rendering.Core.vtkRenderer.newInstance({ background: [0,0,0] });
  const actor = vtk.Rendering.Core.vtkActor.newInstance();
  const mapper = vtk.Rendering.Core.vtkMapper.newInstance();

  let i = 0

  const polydata = vtk.Common.DataModel.vtkPolyData.newInstance();
  polydata.getPoints().setData(Float32Array.from(pointsDataFromPython[i]), 3);
  polydata.getPolys().setData(Uint32Array.from(polysDataFromPython));

  const num_points = pointsDataFromPython[i].length/3;
  const colorsArray = new Float32Array(num_points*3);
  for (let j = 0; j < num_points; j++) {
    colorsArray[j * 3] = colorsFromPython[i][j][0];     // R
    colorsArray[j * 3 + 1] = colorsFromPython[i][j][1]; // G
    colorsArray[j * 3 + 2] = colorsFromPython[i][j][2]; // B
  }
  const colorData = vtk.Common.Core.vtkDataArray.newInstance({
    numberOfComponents: 3, // RGB
    values: colorsArray,
    name: 'Colors',
  });
  polydata.getPointData().setScalars(colorData);
  
  const normalsFilter = vtk.Filters.Core.vtkPolyDataNormals.newInstance();
  normalsFilter.setInputData(polydata);

  renderWindow.addRenderer(renderer);
  renderer.addActor(actor);
  actor.setMapper(mapper);
  mapper.setInputConnection(normalsFilter.getOutputPort());
  mapper.setScalarVisibility(true);
  mapper.setColorModeToDirectScalars();
  renderer.resetCamera();
  
  const openGLRenderWindow = vtk.Rendering.OpenGL.vtkRenderWindow.newInstance();
  renderWindow.addView(openGLRenderWindow);
  
  const container = document.createElement('div');
  container.style.width = '800px';
  container.style.height = '600px';
  element.appendChild(container);
  openGLRenderWindow.setContainer(container);
  
  const { width, height } = container.getBoundingClientRect();
  openGLRenderWindow.setSize(width, height);
  
  const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
  interactor.setView(openGLRenderWindow);
  interactor.initialize();
  interactor.bindEvents(container);
  
  const interactorStyle = vtk.Interaction.Style.vtkInteractorStyleTrackballCamera.newInstance();
  interactor.setInteractorStyle(interactorStyle);
  
  renderWindow.render();

  function animate() {
    i = (i+1)%pointsDataFromPython.length;

    for (let j = 0; j < num_points; j++) {
      colorsArray[j * 3] = colorsFromPython[i][j][0];     // R
      colorsArray[j * 3 + 1] = colorsFromPython[i][j][1]; // G
      colorsArray[j * 3 + 2] = colorsFromPython[i][j][2]; // B
    }
    colorData.modified();
    
    polydata.getPoints().setData(Float32Array.from(pointsDataFromPython[i]), 3);
    polydata.modified();
    renderWindow.render();
  }
  setInterval(animate, 500);
};
document.head.appendChild(script);
"""

def view_basis():
    
    m,l = sph_harm_ind_list(8)

    # This is fun to play with if you want to randomly generate n_fods FODs with given degree_powers 
    # random_fods = np.random.normal(size=(n_fods,)+l.shape)
    # l_values, random_degree_powers = get_degree_powers(random_fods, 8)
    # for i,l_value in enumerate(l_values):
    #     random_fods[...,l==l_value] *= np.sqrt(degree_powers[i]/random_degree_powers[...,i]).reshape(n_fods,1)
    # fod_vals = (random_fods @ sph_harm_vals)

    fods = np.eye(len(l))
    fod_vals = (fods @ sph_harm_vals)
    
    scaled_sphere_pts = fod_vals[...,np.newaxis] * sphere_points[np.newaxis]

    colors = np.zeros_like(scaled_sphere_pts,dtype=float)
    neg_fod_mask = fod_vals<0
    colors[neg_fod_mask] = np.array([205, 92, 92])/255 # negative: red
    colors[~neg_fod_mask] = np.array([92,205,92])/255 # positive: blue
    
    js_code = f"""
    const pointsDataFromPython = {scaled_sphere_pts.reshape(scaled_sphere_pts.shape[0],-1).tolist()};
    const polysDataFromPython = {polys_list};
    const colorsFromPython = {colors.tolist()};
    """
    js_code += vtk_js_viewer_code_basis
    display(Javascript(js_code))


# %%
view_basis()
