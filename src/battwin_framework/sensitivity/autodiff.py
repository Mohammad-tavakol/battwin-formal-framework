"""TensorFlow chain rule through the saved preprocessing, networks and inverses."""

from pathlib import Path
import numpy as np
import pandas as pd
from ..schema import ALIASES, chain_inputs
from ..models.preprocessing import model_frame
from .finite_difference import finite_difference_sensitivity


class DifferentiableDNNChain:
    """TensorFlow equivalent of the saved, independent-row DNN forward chain."""

    def __init__(self, line, frame, *, chain_inputs, model_frame, aliases):
        import tensorflow as tf
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        if getattr(line, "family", None) != "DNN":
            raise ValueError("Automatic differentiation requires the DNN chain.")
        self.tf = tf
        self.inputs = chain_inputs(line, frame)
        self.features = list(self.inputs.select_dtypes("number").columns)
        self.numeric = self.inputs[self.features].to_numpy(dtype=np.float64)
        if len(self.inputs) == 0 or not np.isfinite(self.numeric).all():
            raise ValueError(
                "Comparison requires nonempty inputs with finite numeric values."
            )
        self.aliases = aliases
        self.stages, self.outputs, self.reference_encoded = ([], [], [])
        reference = self.inputs.copy()
        for stage in line.chain_def:
            model = stage["func"].model
            if model.preprocessor.transformer_weights:
                raise TypeError(
                    "Weighted column transformers need an explicit differentiable adapter."
                )
            if any(
                (
                    type(layer).__name__ not in {"InputLayer", "Dense", "Dropout"}
                    for layer in model.model.layers
                )
            ):
                raise TypeError(
                    "Verify row independence before adding other neural layer types."
                )
            aligned = model_frame(reference, model.columns, model.preprocessor)
            blocks = []
            for name, transformer, columns in model.preprocessor.transformers_:
                if not len(columns) or (
                    isinstance(transformer, str) and transformer == "drop"
                ):
                    continue
                columns = list(columns)
                canonical = [aliases.get(c, c) for c in columns]
                if all((pd.api.types.is_numeric_dtype(aligned[c]) for c in columns)):
                    if not hasattr(transformer, "steps"):
                        raise TypeError(f"Unsupported numeric preprocessing: {name}")
                    operations = []
                    for _, operation in transformer.steps:
                        if isinstance(operation, SimpleImputer):
                            if (
                                operation.add_indicator
                                or not np.isfinite(operation.statistics_).all()
                            ):
                                raise TypeError(
                                    "Unsupported imputer indicator or dropped numeric column."
                                )
                            operations.append(
                                ("impute", np.asarray(operation.statistics_, float))
                            )
                        elif isinstance(operation, StandardScaler):
                            operations.append(
                                (
                                    "scale",
                                    (
                                        (
                                            np.asarray(operation.mean_, float)
                                            if operation.with_mean
                                            else np.zeros(len(columns))
                                        ),
                                        (
                                            np.asarray(operation.scale_, float)
                                            if operation.with_std
                                            else np.ones(len(columns))
                                        ),
                                    ),
                                )
                            )
                        else:
                            raise TypeError(
                                f"Unsupported numeric operation: {type(operation).__name__}"
                            )
                    blocks.append(("numeric", canonical, operations))
                else:
                    if any(
                        (pd.api.types.is_numeric_dtype(aligned[c]) for c in columns)
                    ):
                        raise TypeError(
                            "Mixed numeric/categorical blocks must be handled explicitly."
                        )
                    fixed = transformer.transform(aligned[columns])
                    if hasattr(fixed, "toarray"):
                        fixed = fixed.toarray()
                    blocks.append(
                        (
                            "fixed",
                            None,
                            tf.constant(np.asarray(fixed), dtype=tf.float64),
                        )
                    )
            self.stages.append((model, blocks))
            self.reference_encoded.append(
                np.asarray(model.preprocessor.transform(aligned), dtype=np.float32)
            )
            predicted = model.predict_frame(reference)
            for column in predicted:
                reference[column] = predicted[column]
            self.outputs.extend(model.outputs)
        self.reference = reference[self.outputs]

    def _inverse(self, raw, specification):
        from sklearn.preprocessing import StandardScaler

        tf = self.tf
        values = tf.cast(tf.reshape(raw, [-1]), tf.float64)
        scaler = specification.get("scaler")
        if scaler is not None:
            if not isinstance(scaler, StandardScaler):
                raise TypeError("Unsupported fitted target inverse.")
            if scaler.with_std:
                values = values * tf.constant(float(scaler.scale_[0]), tf.float64)
            if scaler.with_mean:
                values = values + tf.constant(float(scaler.mean_[0]), tf.float64)
        elif "mean" in specification and "scale" in specification:
            values = values * tf.constant(float(specification["scale"]), tf.float64)
            values = values + tf.constant(float(specification["mean"]), tf.float64)
        elif specification.get("kind") == "standard":
            raise ValueError("Missing saved standard target inverse.")
        kind = specification.get("kind")
        if kind == "log1p":
            values = tf.math.expm1(values)
        elif kind not in {"standard", "identity"}:
            raise ValueError(f"Unsupported target inverse: {kind}")
        return values

    def __call__(self, numeric, *, return_encoded=False):
        tf = self.tf
        store = {column: numeric[:, j] for j, column in enumerate(self.features)}
        encoded_stages = []
        for model, blocks in self.stages:
            encoded = []
            for kind, columns, operations in blocks:
                if kind == "fixed":
                    encoded.append(operations)
                    continue
                values = tf.stack([store[c] for c in columns], axis=1)
                for operation, parameters in operations:
                    if operation == "impute":
                        values = tf.where(
                            tf.math.is_nan(values),
                            tf.constant(parameters, tf.float64),
                            values,
                        )
                    else:
                        mean, scale = parameters
                        values = (values - tf.constant(mean, tf.float64)) / tf.constant(
                            scale, tf.float64
                        )
                encoded.append(values)
            model_input = tf.cast(tf.concat(encoded, axis=1), tf.float32)
            encoded_stages.append(model_input)
            raw = model.model(model_input, training=False)
            heads = list(raw) if isinstance(raw, (list, tuple)) else [raw]
            for column, head, inverse in zip(model.outputs, heads, model.targets):
                store[column] = self._inverse(head, inverse)
        outputs = tf.stack([store[c] for c in self.outputs], axis=1)
        return (outputs, encoded_stages) if return_encoded else outputs

    def jacobian(self):
        """Per-row physical-output derivatives w.r.t. physical numeric inputs."""
        tf = self.tf
        x = tf.constant(self.numeric, tf.float64)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x)
            outputs, encoded = self(x, return_encoded=True)
            sums = tf.unstack(tf.reduce_sum(outputs, axis=0))
        gradients = []
        for total in sums:
            gradient = tape.gradient(total, x)
            if gradient is None:
                raise RuntimeError("Disconnected TensorFlow input-to-output gradient.")
            gradients.append(gradient.numpy())
        del tape
        jacobian = np.stack(gradients, axis=1)
        prediction = outputs.numpy()
        np.testing.assert_allclose(
            prediction, self.reference.to_numpy(), rtol=2e-05, atol=0.0002
        )
        encoding_errors = []
        for actual, expected in zip(encoded, self.reference_encoded):
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-05, atol=2e-05)
            encoding_errors.append(float(np.max(np.abs(actual.numpy() - expected))))
        if not np.isfinite(jacobian).all():
            raise ValueError("Non-finite automatic derivative.")
        diagnostics = {
            "forward_parity_passed": True,
            "forward_max_absolute_error": float(
                np.max(np.abs(prediction - self.reference.to_numpy()))
            ),
            "per_stage_encoded_max_absolute_error": encoding_errors,
            "numeric_features": self.features,
            "outputs": self.outputs,
            "categories_held_fixed": True,
            "feedback": False,
            "saved_network_dtype": "float32",
            "preprocessing_and_target_inverse_dtype": "float64",
        }
        return (prediction, jacobian, diagnostics)


def autodiff_finite_difference_comparison(
    line,
    frame,
    reference,
    *,
    finite_difference_detail=None,
    step_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
):
    """Compare identical cases, perturbations, physical units and SD scaling."""
    graph = DifferentiableDNNChain(
        line, frame, chain_inputs=chain_inputs, model_frame=model_frame, aliases=ALIASES
    )
    prediction, jacobian, diagnostics = graph.jacobian()
    reference_inputs = chain_inputs(line, reference)
    scales = reference_inputs[graph.features].std(ddof=0)
    steps = {c: max(0.001, 0.001 * float(scales[c])) for c in graph.features}
    fd_function = finite_difference_sensitivity
    finite = finite_difference_detail
    if finite is None:
        finite = fd_function(
            line, frame, graph.features, output_cols=graph.outputs, epsilon=steps
        )
    if set(finite.feature) != set(graph.features):
        raise ValueError(
            "The supplied finite differences use different features or step sizes."
        )
    for feature, data in finite.groupby("feature"):
        np.testing.assert_allclose(
            data.epsilon.to_numpy(), steps[feature], rtol=1e-12, atol=0
        )
    paired = finite.copy().rename(columns={"derivative": "finite_difference"})
    rows = paired.item_index.to_numpy(int) - 1
    output_ids = paired.output.map(
        {c: i for i, c in enumerate(graph.outputs)}
    ).to_numpy(int)
    feature_ids = paired.feature.map(
        {c: i for i, c in enumerate(graph.features)}
    ).to_numpy(int)
    if len(paired) != len(frame) * len(graph.outputs) * len(graph.features):
        raise ValueError("Incomplete finite-difference comparison grid.")
    if paired.duplicated(["item_index", "feature", "output"]).any():
        raise ValueError("Duplicate finite-difference comparison cases.")
    np.testing.assert_allclose(
        paired.input_value, graph.numeric[rows, feature_ids], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        paired.base_output, prediction[rows, output_ids], rtol=2e-05, atol=0.0002
    )
    paired["automatic_derivative"] = jacobian[rows, output_ids, feature_ids]
    paired["reference_input_sd"] = paired.feature.map(scales)
    paired["scaled_ad"] = paired.automatic_derivative * paired.reference_input_sd
    paired["scaled_fd"] = paired.finite_difference * paired.reference_input_sd
    paired["signed_error"] = paired.finite_difference - paired.automatic_derivative
    paired["absolute_error"] = paired.signed_error.abs()
    summary = (
        paired.groupby(["output", "feature"], sort=False)
        .agg(
            n=("automatic_derivative", "size"),
            ad_mean_signed=("automatic_derivative", "mean"),
            fd_mean_signed=("finite_difference", "mean"),
            ad_importance=("scaled_ad", lambda x: x.abs().mean()),
            fd_importance=("scaled_fd", lambda x: x.abs().mean()),
            derivative_mae=("absolute_error", "mean"),
        )
        .reset_index()
    )
    summary["ad_rank"] = summary.groupby("output").ad_importance.rank(
        method="min", ascending=False
    )
    summary["fd_rank"] = summary.groupby("output").fd_importance.rank(
        method="min", ascending=False
    )
    agreement = []
    for output, data in paired.groupby("output", sort=False):
        ranked = summary[summary.output.eq(output)]
        scale = float(data.scaled_ad.abs().mean())
        error = float((data.scaled_fd - data.scaled_ad).abs().mean())
        agreement.append(
            {
                "output": output,
                "n_derivatives": len(data),
                "scaled_derivative_mae": error,
                "relative_scaled_derivative_mae": error / scale if scale else np.nan,
                "signed_derivative_pearson": data.scaled_ad.corr(data.scaled_fd),
                "importance_spearman": ranked.ad_importance.corr(
                    ranked.fd_importance, method="spearman"
                ),
                "top5_overlap": len(
                    set(ranked.nlargest(5, "ad_importance").feature)
                    & set(ranked.nlargest(5, "fd_importance").feature)
                ),
            }
        )
    stability = []
    for multiplier in step_multipliers:
        detail = (
            finite
            if multiplier == 1
            else fd_function(
                line,
                frame,
                graph.features,
                output_cols=graph.outputs,
                epsilon={c: step * multiplier for c, step in steps.items()},
            )
        )
        for output, data in detail.groupby("output", sort=False):
            ids = data.item_index.to_numpy(int) - 1
            cols = data.feature.map(
                {c: i for i, c in enumerate(graph.features)}
            ).to_numpy(int)
            ad = (
                jacobian[ids, graph.outputs.index(output), cols]
                * data.feature.map(scales).to_numpy()
            )
            fd = data.derivative.to_numpy() * data.feature.map(scales).to_numpy()
            norm = np.mean(np.abs(ad))
            stability.append(
                {
                    "output": output,
                    "step_multiplier": multiplier,
                    "scaled_derivative_mae": float(np.mean(np.abs(fd - ad))),
                    "relative_scaled_derivative_mae": (
                        float(np.mean(np.abs(fd - ad)) / norm) if norm else np.nan
                    ),
                }
            )
    diagnostics.update(
        {
            "rows": len(frame),
            "reference_rows": len(reference),
            "finite_difference_steps": steps,
            "step_multipliers": list(step_multipliers),
            "models_refitted": False,
            "interpretation": "Local whole-chain sensitivities at fixed categories and fixed/no feedback; not causal effects.",
        }
    )
    return {
        "derivatives": paired,
        "importance": summary,
        "agreement": pd.DataFrame(agreement),
        "step_size": pd.DataFrame(stability),
        "diagnostics": diagnostics,
    }


def verify_known_autodiff_jacobian():
    """Check scaling, categories, upstream connections and log inverse analytically."""
    import tensorflow as tf
    from types import SimpleNamespace
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    frame = pd.DataFrame(
        {"x": [1.25, 2.5, 3.0], "z": [0.5, 2.0, 4.0], "kind": ["A", "B", "A"]}
    )
    input_scaler = StandardScaler().fit([[-1.0], [3.0]])
    first = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer().fit(frame[["x"]])),
                        ("scaler", input_scaler),
                    ]
                ),
                ["x"],
            ),
            ("cat", OneHotEncoder(sparse_output=False).fit(frame[["kind"]]), ["kind"]),
        ]
    )
    first.fit(frame)
    first.named_transformers_["num"].named_steps["scaler"].fit([[-1.0], [3.0]])
    target_a = StandardScaler().fit([[2.0], [12.0]])
    second = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                ["a", "z"],
            )
        ]
    )
    second.fit(pd.DataFrame({"a": [-1.0, 7.0], "z": [-3.0, 7.0]}))
    target_b = StandardScaler().fit([[-0.5], [0.9]])

    def network(width, weights):
        inputs = tf.keras.Input(shape=(width,))
        outputs = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.Constant(
                np.asarray(weights, np.float32).reshape(-1, 1)
            ),
        )(inputs)
        return tf.keras.Model(inputs, outputs)

    class ToyStage:

        def __init__(self, prep, columns, output, model, target):
            self.preprocessor, self.columns, self.outputs, self.model, self.targets = (
                prep,
                columns,
                [output],
                model,
                [target],
            )

        def predict_frame(self, data):
            x = np.asarray(self.preprocessor.transform(data[self.columns]), np.float32)
            value = np.asarray(self.model(x, training=False)).reshape(-1, 1)
            value = self.targets[0]["scaler"].inverse_transform(value.astype(float))
            if self.targets[0]["kind"] == "log1p":
                value = np.expm1(value)
            return pd.DataFrame(value, index=data.index, columns=self.outputs)

    a = ToyStage(
        first,
        ["x", "kind"],
        "a",
        network(3, [2.0, 1.0, -1.0]),
        {"kind": "standard", "scaler": target_a},
    )
    b = ToyStage(
        second,
        ["a", "z"],
        "y",
        network(2, [0.2, 0.3]),
        {"kind": "log1p", "scaler": target_b},
    )
    line = SimpleNamespace(
        family="DNN",
        chain_def=[
            {"func": SimpleNamespace(model=a)},
            {"func": SimpleNamespace(model=b)},
        ],
    )
    graph = DifferentiableDNNChain(
        line,
        frame,
        chain_inputs=lambda line, data: data[["x", "z", "kind"]].copy(),
        model_frame=lambda data, columns, prep: data[columns],
        aliases={},
    )
    prediction, gradient, _ = graph.jacobian()
    expected = np.zeros_like(gradient)
    expected[:, 0, 0] = 5.0
    expected[:, 1, 0] = (prediction[:, 1] + 1) * 0.175
    expected[:, 1, 1] = (prediction[:, 1] + 1) * 0.042
    np.testing.assert_allclose(gradient, expected, rtol=2e-06, atol=1e-07)
    assert graph.features == ["x", "z"]
    return {
        "analytic_two_stage_jacobian_passed": True,
        "checks": [
            "numeric input scaling",
            "fixed category effects",
            "upstream chain rule",
            "saved standard target inverse",
            "saved log1p target inverse",
            "zero disconnected partials",
        ],
    }
