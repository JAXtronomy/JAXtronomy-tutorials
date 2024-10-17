import jax
import jax.numpy as jnp
from functools import partial

import os
os.environ['XLA_FLAGS']='--xla_cpu_enable_xprof_traceme'

def func1(x):
    return x**2 + jnp.sin(x)

# named scope doesn't work here :(
@partial(jax.jit,static_argnums=0)
def func2(x):
    arr1 = jnp.ones(x)
    arr2 = jnp.arange(0,x)
    return arr1+arr2

# named scope works here... sort of.
# need to have a jit associated with a named_scope
# @partial(jax.jit,static_argnums=1)
def func3(value,len):
    return func1(value) * func2(len)

mem = jax.jit(func3,static_argnums=1).lower(12.,3).compile().memory_analysis()
print(mem)
exit()
print(func3(12.2,3)) # can optionally compile inside the profile call
# with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False):
with jax.profiler.trace("/tmp/tensorboard", create_perfetto_link=False):
  # Run the operations to be profiled
    print(func3(12.2,3))