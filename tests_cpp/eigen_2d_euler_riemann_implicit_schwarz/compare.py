
import numpy as np

gamma = (5.+2.)/5.

def computePressure(rho, u, v, E):
  vel = u**2 + v**2
  return (gamma - 1.) * (E - rho*vel*0.5)

if __name__== "__main__":
  nx=30
  ny=30
  fomTotDofs = nx*ny*4

  for dom_idx in range(4):
    D = np.fromfile(f"riemann2d_solution_{dom_idx}.bin")
    nt = int(np.size(D)/fomTotDofs)
    D = np.reshape(D, (nt, fomTotDofs))
    D = D[-1, :]
    D = np.reshape(D, (nx*ny, 4))
    rho = D[:,0]
    u   = D[:,1]/rho
    v   = D[:,2]/rho
    p   = computePressure(rho, u, v, D[:,3])
    np.savetxt(f"rho_{dom_idx}.txt", rho)
    np.savetxt(f"p_{dom_idx}.txt", p)

    goldR = np.loadtxt(f"rho_gold_{dom_idx}.txt")
    assert(rho.shape == goldR.shape)
    assert(np.isnan(rho).all() == False)
    assert(np.isnan(goldR).all() == False)
    assert(np.allclose(rho, goldR,rtol=1e-10, atol=1e-12))

    goldP = np.loadtxt(f"p_gold_{dom_idx}.txt")
    assert(p.shape == goldP.shape)
    assert(np.isnan(p).all() == False)
    assert(np.isnan(goldP).all() == False)
    assert(np.allclose(p, goldP,rtol=1e-10, atol=1e-12))
