"""
EarlyAlert — data_loader.py
============================
Chargement des fichiers CLIF (.parquet) depuis le dossier data/raw/.
Responsabilité unique : lire, merger, et retourner un DataFrame brut unifié
prêt pour le preprocessing.

Fichiers CLIF utilisés :
    - clif_vitals.parquet          ← constantes vitales (FC, TA, SpO2, FR, Temp)
    - clif_labs.parquet            ← biologie (lactate, créatinine, Hb...)
    - clif_hospitalization.parquet ← séjours hospitaliers (admission/sortie)
    - clif_patient.parquet         ← démographie (âge, sexe)
    - clif_adt.parquet             ← mouvements patient (entrée réa, sortie réa)
    - clif_respiratory_support.parquet ← ventilation mécanique (proxy de sévérité)
    - clif_code_status.parquet     ← statut DNR (important pour le label)

Fichiers non utilisés ici (utiles pour extensions futures) :
    - clif_medication_admin_*      → vasopresseurs (extension V2)
    - clif_patient_assessments*    → scores GCS (extension V2)
    - clif_crrt_therapy            → dialyse (extension V2)
    - clif_position                → position patient
    - clif_patient_procedures      → procédures invasives
    - clif_hospital_diagnosis      → diagnostic principal
"""

import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# Configuration du logger
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_loader")


# ─────────────────────────────────────────────
# Constantes — noms des colonnes CLIF standard
# ─────────────────────────────────────────────

# Identifiants patient/séjour dans CLIF
PATIENT_ID_COL = "patient_id"
ENCOUNTER_ID_COL = "hospitalization_id"  # = séjour hospitalier
ICU_STAY_COL = "location_category"  # valeur "ICU" dans clif_adt

# Colonnes temporelles
TIME_COL = "recorded_dttm"  # timestamp dans vitals / labs
LAB_TIME_COL = "lab_result_dttm"

# Features vitales à extraire depuis clif_vitals
VITAL_FEATURES = {
    "heart_rate": "fc",               # Fréquence cardiaque (bpm)
    "sbp": "tas",                     # Tension artérielle systolique (mmHg)
    "dbp": "tad",                     # Tension artérielle diastolique (mmHg)
    "resp_rate": "fr",                # Fréquence respiratoire (/min)
    "spo2": "spo2",                   # Saturation O2 (%)
    "temp_c": "temperature",          # Température (°C)
}

# Features biologiques à extraire depuis clif_labs
LAB_FEATURES = {
    "lactate": "lactate",             # Lactate (mmol/L) — marqueur de choc
    "creatinine": "creatinine",       # Créatinine (mg/dL) — insuffisance rénale
    "hemoglobin": "hemoglobin",       # Hémoglobine (g/dL)
    "wbc": "wbc",                     # Globules blancs — sepsis
}


# ─────────────────────────────────────────────
# Classe principale DataLoader
# ─────────────────────────────────────────────

class CLIFDataLoader:
    """
    Charge et fusionne les tables CLIF parquet en un DataFrame
    patient-timestep unifié.

    Paramètres
    ----------
    raw_dir : str | Path
        Chemin vers le dossier contenant les fichiers .parquet
    min_icu_hours : int
        Durée minimale de séjour en réa pour inclure un patient (défaut : 6h)
        → On a besoin d'au moins 6h de données pour construire une fenêtre
    """

    def __init__(self, raw_dir: str = "data/raw", min_icu_hours: int = 6):
        self.raw_dir = Path(raw_dir)
        self.min_icu_hours = min_icu_hours
        self._check_files()

    def _check_files(self):
        """Vérifie que les fichiers parquet nécessaires existent."""
        required = [
            "clif_vitals.parquet",
            "clif_hospitalization.parquet",
            "clif_adt.parquet",
            "clif_patient.parquet",
        ]
        for f in required:
            path = self.raw_dir / f
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier requis manquant : {path}\n"
                    f"Vérifie que ton dossier raw_dir={self.raw_dir} est correct."
                )
        log.info("✅ Tous les fichiers requis sont présents.")

    # ──────────────────────────────────────────
    # Loaders individuels
    # ──────────────────────────────────────────

    def _load_parquet(self, filename: str, cols: Optional[list] = None) -> pd.DataFrame:
        """Charge un fichier parquet avec log de forme."""
        path = self.raw_dir / filename
        df = pd.read_parquet(path, columns=cols)
        log.info(f"  Chargé {filename} → {df.shape[0]:,} lignes × {df.shape[1]} cols")
        return df

    def load_hospitalization(self) -> pd.DataFrame:
        """
        clif_hospitalization.parquet
        Colonnes clés : encounter_id, patient_id, admission_dttm, discharge_dttm
        → Donne les bornes temporelles du séjour hospitalier global.
        """
        df = self._load_parquet(
            "clif_hospitalization.parquet",
            cols=[ENCOUNTER_ID_COL, PATIENT_ID_COL,
                  "admission_dttm", "discharge_dttm"]
        )
        df["admission_dttm"] = pd.to_datetime(df["admission_dttm"])
        df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"])
        df = df.rename(columns={"age_at_admission": "age"})
        return df

    def load_icu_stays(self) -> pd.DataFrame:
        """
        clif_adt.parquet (Admission-Discharge-Transfer)
        On filtre sur adt_location_category == 'ICU' pour extraire
        uniquement les périodes de réanimation.
        → Colonnes retournées : encounter_id, in_dttm (entrée réa), out_dttm (sortie réa)
        """
        df = self._load_parquet(
            "clif_adt.parquet",
            cols=[ENCOUNTER_ID_COL, ICU_STAY_COL,
                  "in_dttm", "out_dttm"]
        )
        df = df[df[ICU_STAY_COL].str.upper() == "ICU"].copy()
        df["in_dttm"] = pd.to_datetime(df["in_dttm"])
        df["out_dttm"] = pd.to_datetime(df["out_dttm"])

        # Durée du séjour en heures
        df["icu_duration_h"] = (df["out_dttm"] - df["in_dttm"]).dt.total_seconds() / 3600

        # Filtre : garder uniquement les séjours suffisamment longs
        before = len(df)
        df = df[df["icu_duration_h"] >= self.min_icu_hours].copy()
        log.info(
            f"  Séjours ICU filtrés : {before:,} → {len(df):,} "
            f"(durée ≥ {self.min_icu_hours}h)"
        )
        return df.drop(columns=[ICU_STAY_COL])

    def load_vitals(self) -> pd.DataFrame:
        """
        clif_vitals.parquet
        Colonnes clés : encounter_id, recorded_dttm, vital_category, vital_value
        Format LONG → on pivote en format WIDE (1 ligne = 1 timestamp × 1 patient).
        """
        df = self._load_parquet(
            "clif_vitals.parquet",
            cols=[ENCOUNTER_ID_COL, TIME_COL,
                  "vital_category", "vital_value"]
        )
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df["vital_value"] = pd.to_numeric(df["vital_value"], errors="coerce")

        # Garder uniquement les features d'intérêt
        vital_keys = list(VITAL_FEATURES.keys())
        df = df[df["vital_category"].isin(vital_keys)].copy()

        # Pivot : format LONG → WIDE
        df_wide = df.pivot_table(
            index=[ENCOUNTER_ID_COL, TIME_COL],
            columns="vital_category",
            values="vital_value",
            aggfunc="mean"  # si plusieurs mesures au même timestamp → moyenne
        ).reset_index()

        # Renommer les colonnes avec nos noms français standardisés
        df_wide = df_wide.rename(columns=VITAL_FEATURES)
        df_wide.columns.name = None

        log.info(f"  Vitals pivotés → {df_wide.shape[0]:,} timesteps, "
                 f"{df_wide.shape[1]-2} features")
        return df_wide

    def load_labs(self) -> pd.DataFrame:
        """
        clif_labs.parquet
        Même logique que vitals : format long → wide.
        Les labs sont moins fréquents (toutes les 4-12h en général).
        → Ils seront forward-fillés dans le preprocessing.
        """
        path = self.raw_dir / "clif_labs.parquet"
        if not path.exists():
            log.warning("  clif_labs.parquet absent — labs ignorés.")
            return pd.DataFrame()

        df = self._load_parquet(
            "clif_labs.parquet",
            cols=[ENCOUNTER_ID_COL, LAB_TIME_COL,
                  "lab_category", "lab_value"]
        )
        df = df.rename(columns={LAB_TIME_COL: TIME_COL})  # <--- AJOUTÉ pour uniformiser
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        df["lab_value"] = pd.to_numeric(df["lab_value"], errors="coerce")

        lab_keys = list(LAB_FEATURES.keys())
        df = df[df["lab_category"].isin(lab_keys)].copy()

        if df.empty:
            log.warning("  Aucun lab trouvé avec les catégories attendues.")
            return pd.DataFrame()

        df_wide = df.pivot_table(
            index=[ENCOUNTER_ID_COL, TIME_COL],
            columns="lab_category",
            values="lab_value",
            aggfunc="mean"
        ).reset_index()
        df_wide = df_wide.rename(columns=LAB_FEATURES)
        df_wide.columns.name = None

        log.info(f"  Labs pivotés → {df_wide.shape[0]:,} timesteps")
        return df_wide

    def load_patient_demographics(self) -> pd.DataFrame:
        """
        clif_patient.parquet
        → age, sex (features statiques ajoutées à chaque timestep).
        """
        df = self._load_parquet(
            "clif_patient.parquet",
            cols=[PATIENT_ID_COL, "sex_category"]
        )
        # Encodage binaire du sexe
        df["sex_male"] = (df["sex_category"].str.upper() == "MALE").astype(int)
        return df[[PATIENT_ID_COL, "sex_male"]]

    def load_code_status(self) -> pd.DataFrame:
        """
        clif_code_status.parquet
        → Statut DNR (Do Not Resuscitate).
        Important : les patients DNR ont un pronostic différent.
        On les gardera dans les données mais on les marque.
        """
        path = self.raw_dir / "clif_code_status.parquet"
        if not path.exists():
            log.warning("  clif_code_status.parquet absent — ignoré.")
            return pd.DataFrame()

        df = self._load_parquet("clif_code_status.parquet")
        return df

    def load_respiratory_support(self) -> pd.DataFrame:
        """
        clif_respiratory_support.parquet
        → Présence d'une ventilation mécanique invasive (= signe de sévérité majeur).
        On crée un flag binaire : ventilé oui/non à chaque timestep.
        """
        path = self.raw_dir / "clif_respiratory_support.parquet"
        if not path.exists():
            log.warning("  clif_respiratory_support.parquet absent — ignoré.")
            return pd.DataFrame()

        df = self._load_parquet(
            "clif_respiratory_support.parquet",
            cols=[ENCOUNTER_ID_COL, TIME_COL, "device_category"]
        )
        df[ENCOUNTER_ID_COL] = df[ENCOUNTER_ID_COL].astype(str)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        # Flag ventilation invasive
        df["on_ventilator"] = (
            df["device_category"].str.upper()
            .str.contains("INVASIVE|INTUBAT|VENT", na=False)
        ).astype(int)
        return df[[ENCOUNTER_ID_COL, TIME_COL, "on_ventilator"]]

    # ──────────────────────────────────────────
    # Fusion principale
    # ──────────────────────────────────────────

    def load_all(self, max_patients=25000) -> pd.DataFrame:
        log.info(f"📂 Pipeline de chargement (Limite : {max_patients} patients)")

        # 1. On utilise load_hospitalization pour identifier les patients disponibles
        hosp_full = self.load_hospitalization()
        all_available = hosp_full[ENCOUNTER_ID_COL].unique()

        if len(all_available) > max_patients:
            # On sélectionne aléatoirement les IDs
            selected_ids = np.random.choice(all_available, size=max_patients, replace=False)
            # On s'assure que ce sont des strings pour éviter les bugs de merge
            selected_ids = selected_ids.astype(str)
            log.info(f"   🎯 Échantillonnage de {max_patients} patients effectué.")
        else:
            selected_ids = all_available.astype(str)
            log.info(f"   ℹ️ Nombre de patients ({len(all_available)}) inférieur à la limite.")

        # Filtrer Hospitalization immédiatement
        hosp = hosp_full[hosp_full[ENCOUNTER_ID_COL].astype(str).isin(selected_ids)].copy()
        del hosp_full  # Libère la mémoire du gros DataFrame initial

        # 2. Charger et filtrer ICU
        icu = self.load_icu_stays()
        icu[ENCOUNTER_ID_COL] = icu[ENCOUNTER_ID_COL].astype(str)
        icu = icu[icu[ENCOUNTER_ID_COL].isin(selected_ids)].copy()

        # Fusion hosp + icu pour avoir les bornes temporelles et les IDs patients
        icu = icu.merge(
            hosp[[ENCOUNTER_ID_COL, PATIENT_ID_COL]],
            on=ENCOUNTER_ID_COL,
            how="left"
        )

        # 3. Charger et filtrer VITALS (C'est ici qu'on sauve ta RAM !)
        vitals = self.load_vitals()
        vitals[ENCOUNTER_ID_COL] = vitals[ENCOUNTER_ID_COL].astype(str)
        vitals = vitals[vitals[ENCOUNTER_ID_COL].isin(selected_ids)].copy()

        # Merge vitals + ICU
        df = vitals.merge(icu, on=ENCOUNTER_ID_COL, how="inner")

        # Filtrage temporel (période ICU uniquement)
        mask_in_icu = (
                (df[TIME_COL] >= df["in_dttm"]) &
                (df[TIME_COL] <= df["out_dttm"])
        )
        df = df[mask_in_icu].copy()

        # Calcul du temps relatif
        df["minutes_from_icu_in"] = (
                (df[TIME_COL] - df["in_dttm"]).dt.total_seconds() / 60
        ).round(0).astype(int)

        log.info(f"  ✅ Base de travail prête : {len(df):,} lignes")

        # 4. Labs
        labs = self.load_labs()
        if not labs.empty:
            labs[ENCOUNTER_ID_COL] = labs[ENCOUNTER_ID_COL].astype(str)
            labs = labs[labs[ENCOUNTER_ID_COL].isin(selected_ids)]

            df = df.sort_values([TIME_COL])
            labs = labs.sort_values([TIME_COL])

            df = pd.merge_asof(
                df, labs, on=TIME_COL, by=ENCOUNTER_ID_COL,
                direction="backward", tolerance=pd.Timedelta("12h")
            )

        # 5. Démographie
        demo = self.load_patient_demographics()
        if not demo.empty:
            df[PATIENT_ID_COL] = df[PATIENT_ID_COL].astype(str)
            demo[PATIENT_ID_COL] = demo[PATIENT_ID_COL].astype(str)
            df = df.merge(demo, on=PATIENT_ID_COL, how="left")

        # 6. Support respiratoire
        resp = self.load_respiratory_support()
        if not resp.empty:
            resp[ENCOUNTER_ID_COL] = resp[ENCOUNTER_ID_COL].astype(str)
            resp = resp[resp[ENCOUNTER_ID_COL].isin(selected_ids)]

            df = df.sort_values([TIME_COL])
            resp = resp.sort_values([TIME_COL])

            df = pd.merge_asof(
                df, resp, on=TIME_COL, by=ENCOUNTER_ID_COL,
                direction="backward", tolerance=pd.Timedelta("1h")
            )
            df["on_ventilator"] = df["on_ventilator"].fillna(0).astype(int)

        # Tri final
        df = df.sort_values([ENCOUNTER_ID_COL, TIME_COL]).reset_index(drop=True)

        log.info("=" * 55)
        log.info(f"✅ DataFrame final : {df.shape[0]:,} lignes")
        log.info(f"   Patients uniques : {df[ENCOUNTER_ID_COL].nunique():,}")
        log.info("=" * 55)

        return df

# ─────────────────────────────────────────────
# Point d'entrée standalone (test rapide)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    loader = CLIFDataLoader(raw_dir="data/raw", min_icu_hours=6)
    df_raw = loader.load_all()

    # Sauvegarde intermédiaire
    out_path = Path("data/processed/df_raw_merged.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(out_path, index=False)
    log.info(f"💾 Sauvegardé → {out_path}")

    # Aperçu
    print("\n--- Aperçu ---")
    print(df_raw.head(3).to_string())
    print("\n--- Valeurs manquantes (%) ---")
    print((df_raw.isnull().mean() * 100).round(1).sort_values(ascending=False))