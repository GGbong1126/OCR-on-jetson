"""Initialize the primary runtime context before mixing Driver PTX module APIs."""
import ctypes as ct

cuda=ct.CDLL('libcudart.so.10.2')
cuda.cudaFree.argtypes=[ct.c_void_p]; cuda.cudaFree.restype=ct.c_int
assert cuda.cudaFree(None)==0  # Initialize only; null pointer frees no allocation.

from positive import main

if __name__=='__main__': main()
