from time import time
import numpy as np
from MCEq.misc import info
import mceq_config as config


def solve_rhs(self, int_grid=None, grid_var="X", **kwargs):
        """
        Modified `MCEqRun.solve()` function to accept rhs 
        source function for each slant depth step. `rhs_source` 
        is array of state vectors for each slant depth step. 
        
        Launches the solver.

        The setting `integrator` in the config file decides which solver
        to launch.

        Args:
          int_grid (list): list of depths at which results are recorded
          grid_var (str): Can be depth `X` or something else (currently
            only `X` supported)
          kwargs (dict): Arguments are passed directly to the solver methods.

        """
        info(2, "Launching {0} solver".format(config.integrator))

        if not kwargs.pop("skip_integration_path", False):
            if int_grid is not None and np.any(np.diff(int_grid) < 0):
                raise Exception(
                    "The X values in int_grid are required to be strickly",
                    "increasing.",
                )

            # Calculate integration path if not yet happened
            self._calculate_integration_path(int_grid, grid_var)
        else:
            info(2, "Warning: integration path calculation skipped.")

        phi0 = np.zeros_like(self._phi0)
        nsteps, dX, rho_inv, grid_idcs = self.integration_path

        info(2, "for {0} integration steps.".format(nsteps))
        
        rhs_source =  kwargs.pop("rhs_source")

        start = time()
        
        kernel = solv_numpy
        args = (nsteps, dX, rho_inv, 
                self.int_m, self.dec_m, 
                phi0, grid_idcs,
                rhs_source)



        self._solution, self.grid_sol = kernel(*args)

        info(2, "time elapsed during integration: {0:5.2f}sec".format(time() - start))
        
def solv_numpy(nsteps, dX, rho_inv, int_m, dec_m, phi, grid_idcs, rhs_source):
    """:mod:`numpy` implementation of forward-euler integration.

    Args:
      nsteps (int): number of integration steps
      dX (numpy.array[nsteps]): vector of step-sizes
        :math:`\\Delta X_i` in g/cm**2
      rho_inv (numpy.array[nsteps]): vector of density values
        :math:`\\frac{1}{\\rho(X_i)}`
      int_m (numpy.array): interaction matrix :eq:`int_matrix`
        in dense or sparse representation
      dec_m (numpy.array): decay  matrix :eq:`dec_matrix` in dense
        or sparse representation
      phi (numpy.array): initial state vector :math:`\\Phi(X_0)`
    Returns:
      numpy.array: state vector :math:`\\Phi(X_{nsteps})` after integration
    """

    grid_sol = []
    grid_step = 0

    imc = int_m
    dmc = dec_m
    dxc = dX
    ric = rho_inv
    phc = phi

    dXaccum = 0.0

    from time import time

    start = time()

    for step in range(nsteps):
        phc += ((imc.dot(phc) + dmc.dot(ric[step] * phc)) * dxc[step] 
                + rhs_source[step, :])

        dXaccum += dxc[step]

        if grid_idcs and grid_step < len(grid_idcs) and grid_idcs[grid_step] == step:
            grid_sol.append(np.copy(phc))
            grid_step += 1

    info(
        2,
        "Performance: {0:6.2f}ms/iteration".format(
            1e3 * (time() - start) / float(nsteps)
        ),
    )

    return phc, np.array(grid_sol)        