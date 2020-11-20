import numpy as np
from mpc2c import nmf

# from cylang import cylang
# cylang.compile()
from Cython.Build import Cythonize
Cythonize.main(["mpc2c/**.py", "-3", "--inplace"])

nmf_tools = nmf.NMFTools(np.random.rand(10, 10), 10, 10)

def main():
    print("Hello, wrodl!")


if __name__ == "__main__":
    main()
