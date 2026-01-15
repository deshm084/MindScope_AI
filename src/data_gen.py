from pathlib import Path
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
np.random.seed(42)
Faker.seed(42)

def generate_multimodal_data(num_samples=4000):
    print(f"Generating {num_samples} synthetic patient records...")

    data = []

    for _ in range(num_samples):
        age = np.random.randint(18, 65)
        sleep_hours = np.random.normal(6.5, 1.5)
        stress_level = np.random.randint(1, 10)
        family_history = np.random.choice([0, 1], p=[0.7, 0.3])

        sleep_hours = float(np.clip(sleep_hours, 2.0, 10.0))

        risk_score = (stress_level * 0.4) + ((10 - sleep_hours) * 0.3) + (family_history * 2)
        noise = np.random.normal(0, 0.5)  # less noise = clearer signal
        final_score = risk_score + noise

        if final_score > 7.5:
            label = 2
            text_base = np.random.choice([
                "I feel overwhelming hopelessness and cannot get out of bed.",
                "The anxiety is constant, I can't breathe sometimes.",
                "Nothing brings me joy anymore, just darkness.",
                "I am struggling to cope with daily tasks, everything feels heavy."
            ])
        elif final_score > 4.5:
            label = 1
            text_base = np.random.choice([
                "I've been feeling a bit down lately due to work stress.",
                "Sleep has been rough, and I feel irritable.",
                "Not feeling like myself, just tired all the time.",
                "Worried about the future, having trouble focusing."
            ])
        else:
            label = 0
            text_base = np.random.choice([
                "I feel pretty good, just normal work fatigue.",
                "Life is going well, enjoying my hobbies.",
                "Sleeping well and feeling energetic.",
                "Had a busy week but feeling accomplished."
            ])

        full_text = f"{text_base} {fake.sentence()}"

        data.append({
            "age": int(age),
            "sleep_hours": round(sleep_hours, 1),
            "stress_level": int(stress_level),
            "family_history": int(family_history),
            "clinical_note": full_text,
            "risk_label": int(label)
        })

    df = pd.DataFrame(data)

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    save_path = raw_dir / "mental_health_multimodal.csv"
    df.to_csv(save_path, index=False)

    print(f"Data saved to: {save_path}")
    print(df["risk_label"].value_counts().sort_index())
    return df

if __name__ == "__main__":
    generate_multimodal_data()
