"""
"""
import os
import numpy as np
import jax.numpy as jnp 
from jax import grad as jax_grad
from jax import vmap as jax_vmap
from jax import jit as jax_jit
from ..sigmoid_smhm import logsm_from_logmhalo_jax, DEFAULT_PARAM_VALUES
#from ..kernel_weighted_hist import triweighted_kernel_histogram_with_derivs as twhist
from ..sigmoid_smhm import _logsm_from_logmhalo_jax_kern
import time

from timeit import default_timer as timer
from jax import make_jaxpr
from jax import local_devices
from jax import local_device_count
from jax import device_put
from jax.debug import visualize_array_sharding

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


NM = int(os.environ.get("GORDON_NM", 1000))
#LOGM = np.linspace(8, 15, NM)
LOGM = jnp.linspace(8, 15, NM)  # optimzied
PARAMS = np.array(list(DEFAULT_PARAM_VALUES.values()))
#print(local_devices())
#print("Local device count: ",local_device_count())

def test_logsm_from_logmhalo_evaluates_device():

  print("LOGSM 1 PVC ===>")
  LOGM_SINGLE_DEVICE = device_put(LOGM,device=local_devices()[0]) 
  PARAMS_SINGLE_DEVICE = device_put(PARAMS,device=local_devices()[0]) 
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM_SINGLE_DEVICE, PARAMS_SINGLE_DEVICE))
  start = timer()
  for i in range(0,1000):
      logsm = logsm_from_logmhalo_jax(LOGM_SINGLE_DEVICE, PARAMS_SINGLE_DEVICE)
  end = timer()
  print("<===LOGSM 1 PVC: "+str(end-start))
#  assert np.all(np.isfinite(logsm))

def test_logsm_from_logmhalo_evaluates_two_devices():

  print("LOGSM 2 PVC ===>")
  sharding = PositionalSharding(mesh_utils.create_device_mesh((2,),devices=local_devices()[0:2]))
  LOGM_TWO_DEVICES = device_put(LOGM,sharding.reshape(2,)) 
  visualize_array_sharding(LOGM_TWO_DEVICES)
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM_TWO_DEVICES, PARAMS))
  start = timer()
  for i in range(0,1000):
     logsm = logsm_from_logmhalo_jax(LOGM_TWO_DEVICES, PARAMS)
#  logsm = logsm_from_logmhalo_jax(LOGM_TWO_DEVICES, PARAMS)
  end = timer()
  print("<===LOGSM 2 PVC: "+str(end-start))
#  assert np.all(np.isfinite(logsm))

def test_logsm_from_logmhalo_evaluates_four_devices():

  print("LOGSM 4 PVC ===>")
  sharding = PositionalSharding(mesh_utils.create_device_mesh((4,),devices=local_devices()[0:4]))
  LOGM_FOUR_DEVICES = device_put(LOGM,sharding.reshape(4,)) 
  visualize_array_sharding(LOGM_FOUR_DEVICES)
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM_FOUR_DEVICES, PARAMS))
  start = timer()
  for i in range(0,1000):
      logsm = logsm_from_logmhalo_jax(LOGM_FOUR_DEVICES, PARAMS)
  end = timer()
  print("<===LOGSM 4 PVC: "+str(end-start))

def test_logsm_from_logmhalo_evaluates_eight_devices():

  print("LOGSM 8 PVC ===>")
  sharding = PositionalSharding(mesh_utils.create_device_mesh((8,),devices=local_devices()[0:8]))
  LOGM_EIGHT_DEVICES = device_put(LOGM,sharding.reshape(8,)) 
  visualize_array_sharding(LOGM_EIGHT_DEVICES)
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM_EIGHT_DEVICES, PARAMS))
  start = timer()
  for i in range(0,1000):
      logsm = logsm_from_logmhalo_jax(LOGM, PARAMS)
  end = timer()
  print("<===LOGSM 8 PVC: "+str(end-start))

def test_logsm_from_logmhalo_evaluates_twelve_devices():

  print("LOGSM 12 PVC ===>")
  sharding = PositionalSharding(mesh_utils.create_device_mesh((12,),devices=local_devices()[0:12]))
  LOGM_TWELVE_DEVICES = device_put(LOGM,sharding.reshape(12,)) 
  visualize_array_sharding(LOGM_TWELVE_DEVICES)
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM_TWELVE_DEVICES, PARAMS))
  start = timer()
  for i in range(0,1000):
     logsm = logsm_from_logmhalo_jax(LOGM, PARAMS)
#  logsm = logsm_from_logmhalo_jax(LOGM_TWELVE_DEVICES, PARAMS)
  end = timer()
  print("<===LOGSM 12 PVC: "+str(end-start))

def test_logsm_from_logmhalo_evaluates_default():

  print("LOGSM X PVC===>")
  print(make_jaxpr(logsm_from_logmhalo_jax)(LOGM, PARAMS))
  start = timer()
#  for i in range(0,3000):
#      logsm = logsm_from_logmhalo_jax(LOGM, PARAMS)
  logsm = logsm_from_logmhalo_jax(LOGM, PARAMS)
  end = timer()
  print("<===LOGSM X PVC "+str(end-start))


#    assert np.all(np.isfinite(logsm))


def test_kernel_hist_evaluates():
    logsm = np.array(logsm_from_logmhalo_jax(LOGM, PARAMS))
#    logsm_bins = np.linspace(10, 11, 10)
#    n_halos, n_params = logsm.size, len(PARAMS)
#    sigma = 0.25
#    jac = np.random.uniform(0, 1, size=n_halos * n_params).reshape((n_halos, n_params))

#    h, h_jac = twhist(logsm, np.array(jac), logsm_bins, sigma)
#    twhist(logsm, np.array(jac), logsm_bins, sigma)
#    assert h.shape == (logsm_bins.size - 1,)
#    assert h_jac.shape == (logsm_bins.size - 1, n_params)


def test_kernel_weighted_hist_of_model_gradients():
    """Compute model gradients with jax and propagate through histogram with numba."""
    #logsm = np.array(logsm_from_logmhalo_jax(LOGM, PARAMS))
    #logsm_bins = np.linspace(10, 11, 10)
    #scatter = 0.25
    _gradfunc = jax_grad(_logsm_from_logmhalo_jax_kern, argnums=1)
    gradfunc = jax_jit(jax_vmap(_gradfunc, in_axes=(0, None)))
    smhm_jac = np.array(gradfunc(LOGM, PARAMS))
    for i in range(0,300):
        np.array(gradfunc(LOGM, PARAMS))
#    np.array(gradfunc(LOGM, PARAMS))
#    assert np.shape(smhm_jac) == (LOGM.size, len(PARAMS))

#    h, h_jac = twhist(logsm, smhm_jac, logsm_bins, scatter)
