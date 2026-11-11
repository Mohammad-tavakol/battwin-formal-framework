"""Filenames of the original, bundled trained artifacts."""

RF_FILES = {
    "Mixing": ["RF_Viscosity__log1p__intended_tuned__r_s=42.joblib"],
    "CoatingDrying": [
        "RF_Mass_loading_dry_mg_cm2__standard__intended_tuned__r_s=42.joblib",
        "RF_Dry_Coating_Thickness_m__standard__intended_tuned__r_s=42.joblib",
        "RF_Dried_Porosity__standard__intended_tuned__r_s=42.joblib",
    ],
    "Calendering": [
        "RF_Calendered_thickness_um.joblib",
        "RF_Calendered_porosity.joblib",
    ],
    "Electrochemical": [
        "RF_Gravimetric_capacity_mA_h_g__log1p__intended_tuned__r_s=42.joblib",
        "RF_Volumetric_capacity_mA_h_cm3__standard__intended_tuned__r_s=42.joblib",
    ],
}
DNN_FILES = {
    "Mixing": ("mixing_NN.keras", "Mixing_preprocessor.joblib"),
    "CoatingDrying": ("coat_dry_NN.keras", "coat_dry_preprocessor.joblib"),
    "Calendering": (
        "calendering_DNN__log1p__r_s=42.keras",
        "Calendering_DNN_matched_preprocessor.joblib",
    ),
    "Electrochemical": (
        "electrocchemical_DNN__log1p__r_s=42.keras",
        "Electrochemical_DNN_matched_preprocessor.joblib",
    ),
}
DNN_TARGET_FILES = {
    "Mixing": "Mixing_DNN_target_transform.joblib",
    "CoatingDrying": "Ccoating_DNN_target_transform.joblib",
    "Calendering": "Calendering_DNN_target_transform.joblib",
    "Electrochemical": "Electrochemical_DNN_matched_target_transform.joblib",
}
