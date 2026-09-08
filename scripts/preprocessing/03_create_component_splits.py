from pathlib import Path
from collections import defaultdict
from itertools import permutations

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fakeavceleb_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fakeavceleb_manifest_split.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "splits"
)


# =========================================================
# UNION FIND
# =========================================================

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_linked_ids(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(";")
        if x.strip()
    ]


# =========================================================
# BUILD COMPONENTS
# =========================================================

def build_components(df):
    uf = UnionFind()

    for identity in df["identity"].unique():
        uf.add(identity)

    for _, row in df.iterrows():
        source_identity = row["identity"]

        linked_ids = parse_linked_ids(
            row["linked_identities"]
        )

        for linked_identity in linked_ids:
            uf.union(
                source_identity,
                linked_identity,
            )

    components = defaultdict(set)

    for identity in uf.parent:
        root = uf.find(identity)
        components[root].add(identity)

    component_list = list(
        components.values()
    )

    component_list.sort(
        key=lambda x: sorted(x)[0]
    )

    return component_list


# =========================================================
# COMPONENT STATISTICS
# =========================================================

def get_component_stats(
    df,
    components,
):
    stats = []

    for component_id, identities in enumerate(
        components
    ):
        part = df[
            df["identity"].isin(
                identities
            )
        ]

        counts = (
            part["semantic_class"]
            .value_counts()
            .to_dict()
        )

        stats.append(
            {
                "component_id": component_id,
                "identities": identities,
                "videos": len(part),

                "REAL": counts.get(
                    "REAL",
                    0,
                ),

                "VISUAL_MANIPULATION": counts.get(
                    "VISUAL_MANIPULATION",
                    0,
                ),

                "AUDIO_MANIPULATION": counts.get(
                    "AUDIO_MANIPULATION",
                    0,
                ),

                "AUDIO_VISUAL_MANIPULATION": counts.get(
                    "AUDIO_VISUAL_MANIPULATION",
                    0,
                ),
            }
        )

    return stats


# =========================================================
# FIND BEST 7/1/1/1 ASSIGNMENT
# =========================================================

def choose_component_assignment(stats):
    """
    There are 10 components.

    Assign:
      7 train
      1 validation
      1 calibration
      1 test

    Exhaustively search the possible validation/calibration/test
    choices and choose an assignment whose class proportions
    are closest to the overall dataset.
    """

    component_ids = [
        s["component_id"]
        for s in stats
    ]

    classes = [
        "REAL",
        "VISUAL_MANIPULATION",
        "AUDIO_MANIPULATION",
        "AUDIO_VISUAL_MANIPULATION",
    ]

    totals = {
        cls: sum(
            s[cls]
            for s in stats
        )
        for cls in classes
    }

    total_videos = sum(
        s["videos"]
        for s in stats
    )

    target_fraction = {
        "train": 0.70,
        "validation": 0.10,
        "calibration": 0.10,
        "test": 0.10,
    }

    best_score = float("inf")
    best_assignment = None

    # Ordered permutations because validation/calibration/test
    # are distinct roles.
    for val_id, cal_id, test_id in permutations(
        component_ids,
        3,
    ):

        held_out = {
            val_id,
            cal_id,
            test_id,
        }

        train_ids = [
            cid
            for cid in component_ids
            if cid not in held_out
        ]

        assignment = {
            "train": train_ids,
            "validation": [val_id],
            "calibration": [cal_id],
            "test": [test_id],
        }

        score = 0.0

        for split_name, ids in assignment.items():

            split_stats = [
                s
                for s in stats
                if s["component_id"] in ids
            ]

            split_videos = sum(
                s["videos"]
                for s in split_stats
            )

            video_fraction = (
                split_videos
                / total_videos
            )

            score += (
                video_fraction
                - target_fraction[split_name]
            ) ** 2

            for cls in classes:

                split_class_count = sum(
                    s[cls]
                    for s in split_stats
                )

                class_fraction = (
                    split_class_count
                    / totals[cls]
                )

                score += (
                    class_fraction
                    - target_fraction[split_name]
                ) ** 2

        if score < best_score:
            best_score = score
            best_assignment = assignment

    return (
        best_assignment,
        best_score,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    df = pd.read_csv(
        MANIFEST_PATH
    )

    components = build_components(
        df
    )

    print("=" * 80)
    print("CONNECTED COMPONENTS")
    print("=" * 80)

    print(
        f"Components: {len(components)}"
    )

    print(
        "Sizes:",
        [
            len(c)
            for c in components
        ],
    )

    if len(components) != 10:
        raise RuntimeError(
            "Expected 10 identity components "
            "from the audit."
        )

    stats = get_component_stats(
        df,
        components,
    )

    print()
    print("Component statistics:")
    print()

    stats_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in s.items()
                if key != "identities"
            }
            for s in stats
        ]
    )

    print(
        stats_df.to_string(
            index=False
        )
    )

    assignment, score = (
        choose_component_assignment(
            stats
        )
    )

    print()
    print("=" * 80)
    print("SELECTED ASSIGNMENT")
    print("=" * 80)

    for split_name, component_ids in (
        assignment.items()
    ):
        print(
            f"{split_name:12s}: "
            f"{component_ids}"
        )

    print(
        f"\nBalance score: {score:.8f}"
    )

    # Build component -> split lookup

    component_to_split = {}

    for split_name, component_ids in (
        assignment.items()
    ):
        for cid in component_ids:
            component_to_split[cid] = (
                split_name
            )

    identity_to_component = {}

    for component_id, identities in enumerate(
        components
    ):
        for identity in identities:
            identity_to_component[
                identity
            ] = component_id

    df["component_id"] = (
        df["identity"]
        .map(identity_to_component)
    )

    df["split"] = (
        df["component_id"]
        .map(component_to_split)
    )

    if df["split"].isna().any():
        raise RuntimeError(
            "Some samples were not assigned "
            "to a split."
        )

    # =====================================================
    # STRICT LEAKAGE AUDIT
    # =====================================================

    identity_split = (
        df[
            [
                "identity",
                "split",
            ]
        ]
        .drop_duplicates()
        .set_index("identity")["split"]
        .to_dict()
    )

    leakage_rows = []

    for _, row in df.iterrows():

        sample_split = row["split"]

        related_ids = {
            row["identity"],
            *parse_linked_ids(
                row["linked_identities"]
            ),
        }

        for identity in related_ids:

            other_split = identity_split.get(
                identity
            )

            if other_split is None:
                raise RuntimeError(
                    f"Unknown identity: "
                    f"{identity}"
                )

            if other_split != sample_split:

                leakage_rows.append(
                    {
                        "sample_id": row[
                            "sample_id"
                        ],
                        "identity": row[
                            "identity"
                        ],
                        "linked_identity": (
                            identity
                        ),
                        "sample_split": (
                            sample_split
                        ),
                        "linked_split": (
                            other_split
                        ),
                    }
                )

    if leakage_rows:

        leakage_df = pd.DataFrame(
            leakage_rows
        )

        print(
            leakage_df.head(20)
        )

        raise RuntimeError(
            f"LEAKAGE DETECTED: "
            f"{len(leakage_rows)} rows"
        )

    # =====================================================
    # SAVE
    # =====================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name in [
        "train",
        "validation",
        "calibration",
        "test",
    ]:

        subset = df[
            df["split"] == split_name
        ].copy()

        subset.to_csv(
            SPLIT_DIR
            / f"{split_name}.csv",
            index=False,
        )

    # =====================================================
    # REPORT
    # =====================================================

    print()
    print("=" * 80)
    print("FINAL SPLIT SUMMARY")
    print("=" * 80)

    summary = (
        df.groupby("split")
        .agg(
            videos=(
                "sample_id",
                "count",
            ),
            identities=(
                "identity",
                "nunique",
            ),
            components=(
                "component_id",
                "nunique",
            ),
        )
    )

    print()
    print(summary)

    print()
    print("=" * 80)
    print("4-CLASS DISTRIBUTION")
    print("=" * 80)

    class_table = pd.crosstab(
        df["split"],
        df["semantic_class"],
    )

    print()
    print(class_table)

    print()
    print("=" * 80)
    print("CLASS DISTRIBUTION (%)")
    print("=" * 80)

    class_pct = (
        pd.crosstab(
            df["split"],
            df["semantic_class"],
            normalize="index",
        )
        * 100
    )

    print()
    print(
        class_pct.round(2)
    )

    print()
    print("=" * 80)
    print("BINARY LABEL DISTRIBUTIONS")
    print("=" * 80)

    print("\nVisual:")
    print(
        pd.crosstab(
            df["split"],
            df["visual_label"],
        )
    )

    print("\nAudio:")
    print(
        pd.crosstab(
            df["split"],
            df["audio_label"],
        )
    )

    print()
    print(
        "STRICT IDENTITY-LINK LEAKAGE CHECK: PASSED"
    )

    print()
    print(
        f"Saved master split manifest:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()