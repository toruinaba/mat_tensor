from src.shell.materials import Elastic_sh
import numpy as np

E = 205000.0
n = 0.3
a = Elastic_sh(E, n)
print(a.De)
print(a.De_inv)

print(De)
