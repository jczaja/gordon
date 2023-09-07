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

#NM = int(os.environ.get("GORDON_NM",  12*5010000000))  # peak mem  for 12 JAX devices
#NM = int(os.environ.get("GORDON_NM", 20000100000))     # peak mem  for 4 JAX devices
#NM = int(os.environ.get("GORDON_NM",  10000002000))    # peak mem  for 2 JAX devices
NM = int(os.environ.get("GORDON_NM",  5010000000))      # peak mem  for 1 JAX device

NUM_DEVICES = int(os.environ.get("DEVICES_NM",  1))     # Number of XPUs to be used 

LOGM = np.linspace(8, 15, NM)
#LOGM = jnp.linspace(8, 15, NM)  # optimized, but result is located on single XPU
PARAMS = np.array(list(DEFAULT_PARAM_VALUES.values()))

def test_logsm_from_logmhalo_evaluates():

  if NUM_DEVICES > local_device_count():
      print("Error: More device("+str(NUM_DEVICES)+") was requested than supported("+str(local_device_count)+")")
      exit(-1)

  sharding = PositionalSharding(mesh_utils.create_device_mesh((NUM_DEVICES,),devices=local_devices()[0:NUM_DEVICES]))
  LOGM_DEVICES = device_put(LOGM,sharding.reshape(NUM_DEVICES,)) 
  PARAMS_DEVICES = device_put(PARAMS,sharding.replicate()) 
  visualize_array_sharding(LOGM_DEVICES)
  start = timer()
  for i in range(0,NUM_ITERS):
     logsm = logsm_from_logmhalo_jax(LOGM_DEVICES, PARAMS_DEVICES)
  end = timer()

  average_execution_time = ((end - start)*1000.0)/NUM_ITERS
  print("<===LOGSM JAX devices: "+str(NUM_DEVICES)+", average execution time[ms]: "+str(average_execution_time))

# assert np.all(np.isfinite(logsm))
