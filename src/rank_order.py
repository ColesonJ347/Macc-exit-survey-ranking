import os
import re
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/Grad Program Exit Survey Data.xlsx"
OUT_DIR = "outputs"
OUT_CSV = os.path.join(OUT_DIR, "ranking.csv")
OUT_PNG = os.path.join(OUT_DIR, "rank_order.png")


def parse_course_list(cell) -> list[str]:
    """Parse Qualtrics 'Groups' cell values like: 'ACC 6300...,ACC 6510...'."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    # Qualtrics sometimes includes an ImportId row like {"ImportId":"..."}
    if s.startswith("{") and "ImportId" in s:
        return []
    # Split by comma and clean
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def get_group_cols(columns, keyword_base):
    """Return dict of group columns for Most/Neutral/Least for a section."""
    def find_col(suffix):
        matches = [c for c in columns if keyword_base in c and suffix in c]
        return matches[0] if matches else None

    return {
        "most": find_col("Groups - Most Beneficial"),
        "neutral": find_col("Groups - Neutral"),
        "least": find_col("Groups - Least Beneficial"),
    }


def build_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns

    core_base = "Please identify which MAcc CORE courses"
    elect_base = "Please identify which MAcc Elective courses"

    core = get_group_cols(cols, core_base)
    elect = get_group_cols(cols, elect_base)

    rows = []

    for track_name, group_cols in [("Core", core), ("Elective", elect)]:
        if not all(group_cols.values()):
            # If columns aren’t found, skip that track
            continue

        for _, r in df.iterrows():
            most = parse_course_list(r[group_cols["most"]])
            neutral = parse_course_list(r[group_cols["neutral"]])
            least = parse_course_list(r[group_cols["least"]])

            for c in most:
                rows.append({"track": track_name, "course": c, "bucket": "Most", "score": 1})
            for c in neutral:
                rows.append({"track": track_name, "course": c, "bucket": "Neutral", "score": 0})
            for c in least:
                rows.append({"track": track_name, "course": c, "bucket": "Least", "score": -1})

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No course responses found. Check dataset columns and names.")

    # Clean course names a little (fix double spaces, etc.)
    long["course"] = long["course"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    summary = (
        long.groupby(["track", "course"], as_index=False)
            .agg(
                score=("score", "sum"),
                n=("score", "count"),
                n_most=("bucket", lambda s: (s == "Most").sum()),
                n_neutral=("bucket", lambda s: (s == "Neutral").sum()),
                n_least=("bucket", lambda s: (s == "Least").sum()),
            )
    )

    # Sort: highest score first, then most responses
    summary = summary.sort_values(["score", "n"], ascending=[False, False]).reset_index(drop=True)
    summary.insert(0, "rank", range(1, len(summary) + 1))

    return summary


def make_plot(summary: pd.DataFrame):
    top = summary.head(10).copy()
    # Reverse for barh so #1 shows at top
    top = top.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top["course"], top["score"])
    plt.title("Rank Order of Courses (Most=+1, Neutral=0, Least=-1)")
    plt.xlabel("Net Score")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_excel(DATA_PATH, sheet_name=0)
    summary = build_rank_table(df)

    summary.to_csv(OUT_CSV, index=False)
    make_plot(summary)

    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
