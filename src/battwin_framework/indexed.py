"""Original indexed orchestration and forward sensitivity mathematics, unchanged."""

import numpy as np


# ============================================================
# 1. NUMERICAL JACOBIANS
# ------------------------------------------------------------
# Used only when sensitivity analysis is turned on.
#
# Returns:
#   J_r      = dM/dr
#   J_theta  = dM/dtheta
# ============================================================

def get_jacobians(func, r, u, theta, epsilon=1e-6):
    """
    Compute numerical Jacobians of a static stage model using
    forward finite differences.

    Parameters
    ----------
    func : callable
        Stage model: y = func(r, u, theta)
    r : np.ndarray
        Local read vector
    u : np.ndarray
        Control vector
    theta : np.ndarray
        Local parameter vector
    epsilon : float
        Small perturbation for finite differences

    Returns
    -------
    J_r : np.ndarray
        Jacobian dM/dr with shape (dim_y, dim_r)
    J_theta : np.ndarray
        Jacobian dM/dtheta with shape (dim_y, dim_theta)
    """
    y_base = func(r, u, theta)
    dim_y = len(y_base)

    # Jacobian with respect to r
    J_r = np.zeros((dim_y, len(r)))
    for j in range(len(r)):
        r_pert = r.copy()
        r_pert[j] += epsilon
        y_pert = func(r_pert, u, theta)
        J_r[:, j] = (y_pert - y_base) / epsilon

    # Jacobian with respect to theta
    J_theta = np.zeros((dim_y, len(theta)))
    for j in range(len(theta)):
        theta_pert = theta.copy()
        theta_pert[j] += epsilon
        y_pert = func(r, u, theta_pert)
        J_theta[:, j] = (y_pert - y_base) / epsilon

    return J_r, J_theta


# ============================================================
# 2. FEEDBACK LAW
# ------------------------------------------------------------
# This is a simple proportional correction law.
#
# It updates the stage-specific feedback memory zeta_k based on
# KPI error observed at the end of the chain.
#
# You can replace this later with any custom learning/controller law.
# ============================================================

def proportional_feedback_update(zeta_list, error, gains, target_stage_idx=1):
    """
    Update feedback memory using a simple proportional law.

    Parameters
    ----------
    zeta_list : list of np.ndarray
        Current list of stage-specific feedback blocks:
        [zeta_1, zeta_2, ..., zeta_N]
    error : float
        KPI deviation = KPI_current - KPI_target
    gains : list of float
        One gain per stage feedback block.
        gains[k] is used to update zeta_k.
    target_stage_idx : int
        Index of stage to which correction is applied.
        Default 1 means stage 2 in Python indexing convention.

    Returns
    -------
    zeta_next : list of np.ndarray
        Updated list of feedback blocks
    """
    zeta_next = [z.copy() for z in zeta_list]

    # Only update the selected stage block if it exists and has dimension > 0
    if len(zeta_next[target_stage_idx]) > 0:
        zeta_next[target_stage_idx][0] += gains[target_stage_idx] * error

    return zeta_next


# ============================================================
# 3. STATE-SPACE LINE CLASS
# ------------------------------------------------------------
# This class handles:
#   - compiling the stage sequence
#   - executing one item through the full line
#   - executing multiple items with delayed feedback
#   - optional sensitivity analysis
#
# The coding is intentionally kept simple and explicit.
# ============================================================

class StateSpaceLine:
    """
    Static-map hybrid state-space framework with:
      - global feature store eta
      - stage-specific feedback memory zeta_k
      - optional forward sensitivity propagation

    Notes
    -----
    This implementation matches the simplified static-map framework:
        y_k = M_k(r_k, u_k, theta_k)

        r_k = W_k [eta ; zeta]

        eta_k = (I - Pi_k) eta_{k-1} + Pi_k psi_k(y_k)

    where psi_k is implemented as direct placement into a write slice.
    """

    def __init__(self, dim_eta_0, zeta_dims):
        """
        Parameters
        ----------
        dim_eta_0 : int
            Dimension of the initial feature vector eta_0
        zeta_dims : list of int
            Dimensions of the stage-specific feedback blocks.
            Example: [0, 1, 0, 0]
        """
        self.chain_def = []
        self.dim_eta_0 = dim_eta_0
        self.zeta_dims = zeta_dims
        self.dim_zeta_total = sum(zeta_dims)
        self.compiled = False

    def load_sequence(self, chain_definition):
        """
        Load the ordered list of stage definitions.

        Each stage is a dictionary with keys:
            name         : stage name
            func         : stage model function
            read_indices : list of global feature indices or -1 for zeta_k access
            out_dim      : dimension of stage output
            u            : control vector
            theta        : parameter vector
        """
        self.chain_def = chain_definition

    def compile(self):
        """
        Compile the framework:
          - total feature dimension P
          - total parameter dimension p_theta
          - read matrices W_k
          - write masks Pi_k
          - slices for output placement and parameter embedding
        """
        dim_P = self.dim_eta_0 + sum(stage["out_dim"] for stage in self.chain_def)
        p_theta = sum(len(stage["theta"]) for stage in self.chain_def)

        self.compiled_chain = []

        current_write_idx = self.dim_eta_0
        current_theta_idx = 0
        current_zeta_idx = 0

        for i, stage in enumerate(self.chain_def):
            num_reads = len(stage["read_indices"])

            # ----------------------------------------------------
            # Build read selection matrix W_k
            # W_k maps [eta ; zeta_flat] -> r_k
            # ----------------------------------------------------
            W_k = np.zeros((num_reads, dim_P + self.dim_zeta_total))

            for local_idx, global_idx in enumerate(stage["read_indices"]):
                if global_idx == -1:
                    # Read from THIS stage's zeta block
                    # For simplicity, this code assumes one feedback scalar
                    # is read if the block exists.
                    zeta_offset = dim_P + current_zeta_idx
                    if self.zeta_dims[i] > 0:
                        W_k[local_idx, zeta_offset] = 1.0
                else:
                    # Read from global eta
                    W_k[local_idx, global_idx] = 1.0

            # ----------------------------------------------------
            # Build write mask Pi_k
            # This stage writes into its dedicated output slice
            # ----------------------------------------------------
            mask = np.zeros(dim_P)
            out_dim = stage["out_dim"]
            mask[current_write_idx: current_write_idx + out_dim] = 1.0
            Pi_k = np.diag(mask)

            # Save compiled stage information
            self.compiled_chain.append({
                **stage,
                "W_k": W_k,
                "Pi_k": Pi_k,
                "write_slice": slice(current_write_idx, current_write_idx + out_dim),
                "theta_slice": slice(current_theta_idx, current_theta_idx + len(stage["theta"])),
                "zeta_slice": slice(current_zeta_idx, current_zeta_idx + self.zeta_dims[i]),
            })

            current_write_idx += out_dim
            current_theta_idx += len(stage["theta"])
            current_zeta_idx += self.zeta_dims[i]

        self.dim_P = dim_P
        self.p_theta = p_theta
        self.I = np.eye(dim_P)
        self.compiled = True

    # ========================================================
    # Single-item execution
    # --------------------------------------------------------
    # This runs ONE item through the full line using the
    # current feedback memory zeta.
    #
    # If calc_sensitivity=True, it also computes d(eta_N)/dtheta.
    # ========================================================
    def simulate_item(self, eta_0, zeta_list, calc_sensitivity=False):
        """
        Simulate one item through the process chain.

        Parameters
        ----------
        eta_0 : np.ndarray
            Initial feature vector (base product descriptor)
        zeta_list : list of np.ndarray
            Stage-specific feedback memory blocks
        calc_sensitivity : bool
            If True, compute forward sensitivity of eta_N wrt theta

        Returns
        -------
        result : dict
            {
              "eta_final": final feature vector,
              "zeta_flat": flattened feedback used in this run,
              "stage_outputs": list of y_k,
              "S_eta": d(eta_final)/dtheta or None
            }
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before simulation.")

        # ----------------------------------------------------
        # Initialize eta and flattened zeta
        # ----------------------------------------------------
        eta = np.zeros(self.dim_P)
        eta[0:self.dim_eta_0] = eta_0

        zeta_flat = np.concatenate(zeta_list) if self.dim_zeta_total > 0 else np.array([])

        # ----------------------------------------------------
        # Optional sensitivities
        # ----------------------------------------------------
        S_eta = np.zeros((self.dim_P, self.p_theta)) if calc_sensitivity else None

        # zeta is treated as fixed input during a single chain execution
        S_zeta = np.zeros((self.dim_zeta_total, self.p_theta)) if calc_sensitivity else None

        stage_outputs = []

        # ----------------------------------------------------
        # Stage loop
        # ----------------------------------------------------
        for stage in self.compiled_chain:
            # ---------- FORWARD ----------
            global_state = np.concatenate([eta, zeta_flat])
            r_k = stage["W_k"] @ global_state

            y_k = stage["func"](
                r_k,
                np.asarray(stage["u"], dtype=float),
                np.asarray(stage["theta"], dtype=float)
            )

            # Store stage output
            stage_outputs.append(y_k.copy())

            # eta update using write slice as the embedding psi_k
            update_vec = np.zeros(self.dim_P)
            update_vec[stage["write_slice"]] = y_k

            # eta_k = (I - Pi_k) eta_{k-1} + Pi_k psi_k(y_k)
            # Here Pi_k @ update_vec is equivalent to update_vec because
            # update_vec is already zero outside the write slice.
            eta = (self.I - stage["Pi_k"]) @ eta + update_vec

            # ---------- SENSITIVITY ----------
            if calc_sensitivity:
                J_r, J_theta_local = get_jacobians(
                    stage["func"],
                    r_k,
                    np.asarray(stage["u"], dtype=float),
                    np.asarray(stage["theta"], dtype=float)
                )

                # Expand local theta Jacobian to global theta space
                J_theta_global = np.zeros((len(y_k), self.p_theta))
                J_theta_global[:, stage["theta_slice"]] = J_theta_local

                # S_rk = W_k [S_eta ; S_zeta]
                S_global = np.vstack((S_eta, S_zeta))
                S_rk = stage["W_k"] @ S_global

                # S_yk = J_r * S_rk + J_theta_global
                S_yk = J_r @ S_rk + J_theta_global

                # Embed S_yk into global eta sensitivity
                S_update = np.zeros((self.dim_P, self.p_theta))
                S_update[stage["write_slice"], :] = S_yk

                # S_eta update
                S_eta = (self.I - stage["Pi_k"]) @ S_eta + S_update

        return {
            "eta_final": eta,
            "zeta_flat": zeta_flat,
            "stage_outputs": stage_outputs,
            "S_eta": S_eta
        }

    # ========================================================
    # Multi-item execution with delayed feedback
    # --------------------------------------------------------
    # This runs several items sequentially.
    # After each item completes:
    #   - a KPI is read
    #   - deviation from target is computed
    #   - zeta is updated for the NEXT item
    # ========================================================
    def simulate_items(
        self,
        num_items,
        eta_0,
        zeta_init,
        kpi_stage_index,
        kpi_output_index,
        kpi_target,
        feedback_gains,
        feedback_stage_index=1,
        calc_sensitivity=False
    ):
        """
        Simulate multiple items with one-step delayed feedback.

        Parameters
        ----------
        num_items : int
            Number of items / chain executions
        eta_0 : np.ndarray
            Initial base feature vector for each item
        zeta_init : list of np.ndarray
            Initial stage-specific feedback blocks
        kpi_stage_index : int
            Which stage output contains the KPI
        kpi_output_index : int
            Which component of that stage output is the KPI
        kpi_target : float
            Desired target KPI value
        feedback_gains : list of float
            Gain list for stage-specific feedback update
        feedback_stage_index : int
            Which stage zeta_k gets updated by the feedback law
        calc_sensitivity : bool
            If True, compute d(eta_N)/dtheta for each item

        Returns
        -------
        history : list of dict
            One record per item containing outputs, KPI, error, eta_final, etc.
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before simulation.")

        zeta_current = [z.copy() for z in zeta_init]
        history = []

        for i in range(num_items):
            # Run one item using current zeta^i
            result = self.simulate_item(
                eta_0=eta_0,
                zeta_list=zeta_current,
                calc_sensitivity=calc_sensitivity
            )

            eta_final = result["eta_final"]
            stage_outputs = result["stage_outputs"]
            S_eta = result["S_eta"]

            # ------------------------------------------------
            # Compute KPI and deviation
            # ------------------------------------------------
            kpi_value = stage_outputs[kpi_stage_index][kpi_output_index]
            error = kpi_value - kpi_target

            # ------------------------------------------------
            # Update feedback memory for NEXT item
            # ------------------------------------------------
            zeta_next = proportional_feedback_update(
                zeta_list=zeta_current,
                error=error,
                gains=feedback_gains,
                target_stage_idx=feedback_stage_index
            )

            # Save history for inspection
            history.append({
                "item_index": i + 1,
                "eta_final": eta_final.copy(),
                "stage_outputs": [y.copy() for y in stage_outputs],
                "kpi_value": float(kpi_value),
                "kpi_target": float(kpi_target),
                "error": float(error),
                "zeta_used": [z.copy() for z in zeta_current],
                "zeta_next": [z.copy() for z in zeta_next],
                "S_eta": None if S_eta is None else S_eta.copy()
            })

            # Shift zeta forward by one step
            zeta_current = zeta_next

        return history