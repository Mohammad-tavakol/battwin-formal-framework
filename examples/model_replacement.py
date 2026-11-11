"""Demonstrate a custom stage adapter and restore it; no artifacts are changed."""

from battwin_framework import Framework
from battwin_framework.models import CallableStageModel


def main():
    framework = Framework.from_bundle()
    family, stage = "RF", "Electrochemical"
    original = framework.models.stage(family, stage)
    schema = framework.models.schema(family, stage)
    # Replace original.predict_frame below with your own physical-unit predictor.
    replacement = CallableStageModel(
        stage, family, schema["inputs"], original.predict_frame
    )
    previous = framework.replace_model(family, stage, replacement)
    try:
        predictions = framework.predict(
            framework.load_dataset("test"), families=(family,)
        )
        print(predictions[family][schema["outputs"]].head().to_string(index=False))
    finally:
        framework.replace_model(family, stage, previous)


if __name__ == "__main__":
    main()
