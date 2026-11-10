from copy import deepcopy

import numpy as np

def default_objective_from_kpi(features_final, kpi_key):
    """
    Default objective: J = KPI value.

    Parameters
    ----------
    features_final : dict
        Final feature store after chain execution
    kpi_key : str
        Key of KPI in features_final

    Returns
    -------
    float
        KPI value
    """
    if kpi_key not in features_final:
        raise KeyError(f"KPI key '{kpi_key}' not found in final features.")
    return float(features_final[kpi_key])

def is_numeric_scalar(x):
    """
    Check whether x behaves like a numeric scalar for perturbation purposes.
    """
    return isinstance(x, (int, float, np.integer, np.floating))

def proportional_feedback_update(zeta_dict, error, gains, target_stage_name):
    """
    Update stage-specific feedback memory using a proportional law.

    Parameters
    ----------
    zeta_dict : dict
        Current feedback memory
    error : float
        KPI error = KPI_current - KPI_target
    gains : dict
        Nested dict of gains per stage and feedback key
    target_stage_name : str
        Which stage feedback block is updated

    Returns
    -------
    dict
        Updated feedback memory
    """
    zeta_next = deepcopy(zeta_dict)

    if target_stage_name in zeta_next and target_stage_name in gains:
        for key, gain in gains[target_stage_name].items():
            if key in zeta_next[target_stage_name]:
                zeta_next[target_stage_name][key] += gain * error

    return zeta_next

class NamedStateSpaceLine:
    """
    Named-feature hybrid state-space framework.

    Main methods
    ------------
    compile()
        Check that stage names are unique.

    simulate_item(...)
        Run one item through the whole chain.

    simulate_items(...)
        Run many items with optional delayed feedback.

    sensitivity_one_item(...)
        Finite-difference sensitivity analysis for one item.

    sensitivity_items(...)
        Finite-difference sensitivity analysis for many items,
        with optional delayed feedback between items.
    """

    def __init__(self):
        self.chain_def = []
        self.compiled = False

    def load_sequence(self, chain_definition):
        """
        Load ordered chain definition.
        """
        self.chain_def = chain_definition

    def compile(self):
        """
        Compile simple structural checks.
        """
        stage_names = [stage["name"] for stage in self.chain_def]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("Stage names must be unique.")
        self.compiled = True

    def get_required_input_columns(self):
        """
        Return the set of columns required in the ORIGINAL input dataframe
        before stage-generated outputs are added.

        Each stage declares 'required_input_columns', excluding columns that
        are expected to be produced by upstream stages.
        """
        required = set()
        for stage in self.chain_def:
            stage_required = stage.get("required_input_columns", [])
            required.update(stage_required)
        return sorted(required)

    def validate_input_dataframe(self, df_input):
        """
        Validate that the input dataframe contains all required base columns.
        """
        required_cols = self.get_required_input_columns()
        missing_cols = [c for c in required_cols if c not in df_input.columns]

        if missing_cols:
            raise ValueError(
                "Input dataframe is missing required columns:\n"
                + "\n".join(f"  - {c}" for c in missing_cols)
            )

    def simulate_item(self, row_features, zeta_dict=None):
        """
        Simulate one item through the whole chain.

        Parameters
        ----------
        row_features : dict
            One input row converted to a dictionary
        zeta_dict : dict or None
            Stage-specific feedback memory

        Returns
        -------
        dict
            {
                "features_final": final evolving feature store,
                "stage_outputs": list of stage outputs,
                "zeta_used": feedback memory used for this item
            }
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before simulation.")

        if zeta_dict is None:
            zeta_dict = {}

        features = deepcopy(row_features)
        stage_outputs = []

        for stage in self.chain_def:
            stage_name = stage["name"]
            func = stage["func"]
            control = stage.get("control", {})
            theta = stage.get("theta", {})
            zeta_stage = deepcopy(zeta_dict.get(stage_name, {}))

            y_dict = func(
                features=features,
                zeta=zeta_stage,
                control=control,
                theta=theta,
            )

            if not isinstance(y_dict, dict):
                raise TypeError(f"Stage '{stage_name}' must return a dict.")

            # Write outputs back into the global named feature store
            features.update(y_dict)

            stage_outputs.append({
                "stage_name": stage_name,
                "outputs": deepcopy(y_dict),
            })

        return {
            "features_final": features,
            "stage_outputs": stage_outputs,
            "zeta_used": deepcopy(zeta_dict),
        }

    def simulate_items(
        self,
        df_input,
        use_feedback=False,
        zeta_init=None,
        kpi_key=None,
        kpi_target=None,
        feedback_gains=None,
        feedback_stage_name=None,
        row_id_column=None,
    ):
        """
        Simulate multiple rows / items.

        Parameters
        ----------
        df_input : pd.DataFrame
            Input dataframe. Each row is one item / chain execution.
        use_feedback : bool
            If True, delayed feedback is enabled.
        zeta_init : dict or None
            Initial feedback memory.
        kpi_key : str or None
            Final feature key used as KPI.
        kpi_target : float or None
            KPI target.
        feedback_gains : dict or None
            Gains for proportional feedback update.
        feedback_stage_name : str or None
            Which stage feedback block is updated.
        row_id_column : str or None
            Optional row identifier column.

        Returns
        -------
        list of dict
            History of all item simulations.
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before simulation.")

        self.validate_input_dataframe(df_input)

        if zeta_init is None:
            zeta_init = {}

        if feedback_gains is None:
            feedback_gains = {}

        if use_feedback:
            if kpi_key is None:
                raise ValueError("kpi_key must be provided when use_feedback=True.")
            if kpi_target is None:
                raise ValueError("kpi_target must be provided when use_feedback=True.")
            if feedback_stage_name is None:
                raise ValueError("feedback_stage_name must be provided when use_feedback=True.")

        zeta_current = deepcopy(zeta_init)
        history = []

        for i, (_, row) in enumerate(df_input.iterrows(), start=1):
            row_features = row.to_dict()

            result = self.simulate_item(
                row_features=row_features,
                zeta_dict=zeta_current if use_feedback else {},
            )

            features_final = result["features_final"]
            stage_outputs = result["stage_outputs"]

            kpi_value = None
            error = None
            zeta_next = deepcopy(zeta_current) if use_feedback else {}

            if use_feedback:
                if kpi_key not in features_final:
                    raise KeyError(f"KPI key '{kpi_key}' not found in final feature store.")

                kpi_value = float(features_final[kpi_key])
                error = kpi_value - float(kpi_target)

                zeta_next = proportional_feedback_update(
                    zeta_dict=zeta_current,
                    error=error,
                    gains=feedback_gains,
                    target_stage_name=feedback_stage_name,
                )

            history.append({
                "item_index": i,
                "row_id": row_features.get(row_id_column) if row_id_column else None,
                "features_final": deepcopy(features_final),
                "stage_outputs": deepcopy(stage_outputs),
                "kpi_key": kpi_key if use_feedback else None,
                "kpi_value": kpi_value,
                "kpi_target": float(kpi_target) if (use_feedback and kpi_target is not None) else None,
                "error": error,
                "zeta_used": deepcopy(zeta_current) if use_feedback else {},
                "zeta_next": deepcopy(zeta_next) if use_feedback else {},
            })

            if use_feedback:
                zeta_current = zeta_next

        return history

    def sensitivity_one_item(
        self,
        row_features,
        zeta_dict=None,
        sensitivity_columns=None,
        kpi_key=None,
        objective_fn=None,
        epsilon=1e-5,
    ):
        """
        Finite-difference sensitivity analysis for one item.

        Parameters
        ----------
        row_features : dict
            One input row as a feature dictionary.
        zeta_dict : dict or None
            Feedback memory for this item. Treated as FIXED during sensitivity analysis.
        sensitivity_columns : list of str
            Numeric input columns to perturb.
        kpi_key : str or None
            KPI key used if objective_fn is not provided.
        objective_fn : callable or None
            Optional custom scalar objective:
                J = objective_fn(features_final)
        epsilon : float
            Perturbation size for finite differences.

        Returns
        -------
        dict
            {
                "base_features_final": ...,
                "base_stage_outputs": ...,
                "J_base": ...,
                "d_eta_final": {input_col: {feature_key: derivative}},
                "dJ": {input_col: derivative}
            }
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before sensitivity analysis.")

        if zeta_dict is None:
            zeta_dict = {}

        if sensitivity_columns is None or len(sensitivity_columns) == 0:
            raise ValueError("sensitivity_columns must be a non-empty list.")

        # Base simulation
        base_result = self.simulate_item(
            row_features=deepcopy(row_features),
            zeta_dict=deepcopy(zeta_dict),
        )

        base_features_final = base_result["features_final"]
        base_stage_outputs = base_result["stage_outputs"]

        if objective_fn is not None:
            J_base = float(objective_fn(base_features_final))
        else:
            if kpi_key is None:
                raise ValueError("Provide either objective_fn or kpi_key.")
            J_base = default_objective_from_kpi(base_features_final, kpi_key)

        # Finite-difference derivatives
        d_eta_final = {}
        dJ = {}

        for col in sensitivity_columns:
            if col not in row_features:
                raise KeyError(f"Sensitivity column '{col}' not found in row_features.")

            base_value = row_features[col]

            if not is_numeric_scalar(base_value):
                raise TypeError(
                    f"Column '{col}' is not numeric. "
                    "Finite-difference sensitivity is only supported for numeric columns."
                )

            perturbed_row = deepcopy(row_features)
            perturbed_row[col] = float(base_value) + epsilon

            pert_result = self.simulate_item(
                row_features=perturbed_row,
                zeta_dict=deepcopy(zeta_dict),
            )

            pert_features_final = pert_result["features_final"]

            # d(eta_final)/d(col)
            feature_sens = {}
            for key, base_val in base_features_final.items():
                pert_val = pert_features_final.get(key, base_val)

                if is_numeric_scalar(base_val) and is_numeric_scalar(pert_val):
                    feature_sens[key] = (float(pert_val) - float(base_val)) / epsilon

            d_eta_final[col] = feature_sens

            # dJ/d(col)
            if objective_fn is not None:
                J_pert = float(objective_fn(pert_features_final))
            else:
                J_pert = default_objective_from_kpi(pert_features_final, kpi_key)

            dJ[col] = (J_pert - J_base) / epsilon

        return {
            "base_features_final": deepcopy(base_features_final),
            "base_stage_outputs": deepcopy(base_stage_outputs),
            "J_base": J_base,
            "d_eta_final": d_eta_final,
            "dJ": dJ,
        }

    def sensitivity_items(
        self,
        df_input,
        sensitivity_columns,
        use_feedback=False,
        zeta_init=None,
        kpi_key=None,
        kpi_target=None,
        feedback_gains=None,
        feedback_stage_name=None,
        objective_fn=None,
        epsilon=1e-5,
        row_id_column=None,
    ):
        """
        Finite-difference sensitivity analysis over multiple items.

        If feedback is enabled:
            - zeta is updated sequentially between items
            - during sensitivity analysis of one item, zeta is treated as fixed

        Parameters
        ----------
        df_input : pd.DataFrame
            Input rows to simulate
        sensitivity_columns : list of str
            Numeric columns to perturb
        use_feedback : bool
            If True, delayed feedback is enabled between items
        zeta_init : dict or None
            Initial feedback memory
        kpi_key : str or None
            KPI key used for default objective and feedback
        kpi_target : float or None
            KPI target used for feedback
        feedback_gains : dict or None
            Gains for feedback law
        feedback_stage_name : str or None
            Stage block to update in zeta
        objective_fn : callable or None
            Optional custom scalar objective
        epsilon : float
            Finite-difference perturbation
        row_id_column : str or None
            Optional row identifier

        Returns
        -------
        list of dict
            Sensitivity history item by item
        """
        if not self.compiled:
            raise RuntimeError("Framework must be compiled before sensitivity analysis.")

        self.validate_input_dataframe(df_input)

        if zeta_init is None:
            zeta_init = {}

        if feedback_gains is None:
            feedback_gains = {}

        if use_feedback:
            if kpi_key is None:
                raise ValueError("kpi_key must be provided when use_feedback=True.")
            if kpi_target is None:
                raise ValueError("kpi_target must be provided when use_feedback=True.")
            if feedback_stage_name is None:
                raise ValueError("feedback_stage_name must be provided when use_feedback=True.")

        zeta_current = deepcopy(zeta_init)
        history = []

        for i, (_, row) in enumerate(df_input.iterrows(), start=1):
            row_features = row.to_dict()

            sens_result = self.sensitivity_one_item(
                row_features=row_features,
                zeta_dict=deepcopy(zeta_current) if use_feedback else {},
                sensitivity_columns=sensitivity_columns,
                kpi_key=kpi_key,
                objective_fn=objective_fn,
                epsilon=epsilon,
            )

            features_final = sens_result["base_features_final"]

            kpi_value = None
            error = None
            zeta_next = deepcopy(zeta_current) if use_feedback else {}

            if use_feedback:
                if kpi_key not in features_final:
                    raise KeyError(f"KPI key '{kpi_key}' not found in final feature store.")

                kpi_value = float(features_final[kpi_key])
                error = kpi_value - float(kpi_target)

                zeta_next = proportional_feedback_update(
                    zeta_dict=zeta_current,
                    error=error,
                    gains=feedback_gains,
                    target_stage_name=feedback_stage_name,
                )

            history.append({
                "item_index": i,
                "row_id": row_features.get(row_id_column) if row_id_column else None,
                "features_final": deepcopy(features_final),
                "J_base": sens_result["J_base"],
                "dJ": deepcopy(sens_result["dJ"]),
                "d_eta_final": deepcopy(sens_result["d_eta_final"]),
                "zeta_used": deepcopy(zeta_current) if use_feedback else {},
                "zeta_next": deepcopy(zeta_next) if use_feedback else {},
                "kpi_value": kpi_value,
                "kpi_target": float(kpi_target) if (use_feedback and kpi_target is not None) else None,
                "error": error,
            })

            if use_feedback:
                zeta_current = zeta_next

        return history
