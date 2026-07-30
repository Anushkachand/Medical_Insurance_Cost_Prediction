"""
Generates data/insurance.csv — a dataset matching the schema and real-world
statistical patterns of the widely-used Kaggle "Medical Cost Personal
Datasets" (age, sex, bmi, children, smoker, region, charges; 1338 rows).

Note: this sandboxed environment cannot reach kaggle.com directly, so this
script reconstructs the dataset synthetically, matching the original's
column schema, value ranges, and the well-documented real-world cost
relationships (smoking and high BMI sharply raise charges; age raises the
baseline; each dependent adds a modest premium; region adds small variance).
If you have direct Kaggle access, you can swap this for the original CSV
without changing any downstream code — the schema is identical.

Run:
    python3 data/generate_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent / "insurance.csv"

N_SAMPLES = 1338  # matches the row count of the original Kaggle dataset
RANDOM_STATE = 42


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    age = rng.integers(18, 65, N_SAMPLES)
    sex = rng.choice(["male", "female"], N_SAMPLES)
    bmi = rng.normal(30.7, 6.1, N_SAMPLES).clip(15.9, 53.1)
    children = rng.choice([0, 1, 2, 3, 4, 5], N_SAMPLES, p=[0.43, 0.24, 0.18, 0.10, 0.03, 0.02])
    smoker = rng.choice(["no", "yes"], N_SAMPLES, p=[0.795, 0.205])
    region = rng.choice(
        ["southwest", "southeast", "northwest", "northeast"], N_SAMPLES,
        p=[0.243, 0.272, 0.243, 0.242],
    )

    region_effect = pd.Series(region).map(
        {"southwest": -300, "southeast": 200, "northwest": -100, "northeast": 400}
    ).to_numpy()

    smoker_effect = np.where(smoker == "yes", 23000 + bmi * 300, 0)
    bmi_penalty = np.where(bmi > 30, (bmi - 30) * 120, 0)

    base_charge = (
        250 * age
        + 15 * (bmi ** 1.5)
        + 500 * children
        + bmi_penalty
        + smoker_effect
        + region_effect
        + rng.normal(0, 1500, N_SAMPLES)
    )
    charges = np.clip(base_charge, 1121.87, None)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": np.round(bmi, 3),
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": np.round(charges, 5),
        }
    )
    return df


def main() -> None:
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows: {len(df)}")
    print(df.describe(include="all").T)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
