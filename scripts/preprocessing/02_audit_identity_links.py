from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "fakeavceleb_manifest.csv"
)


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


def main():
    df = pd.read_csv(MANIFEST_PATH)

    folder_ids = set(
        df["identity"]
        .dropna()
        .astype(str)
    )

    linked_ids = set()

    uf = UnionFind()

    for identity in folder_ids:
        uf.add(identity)

    rows_with_links = 0
    total_links = 0

    link_frequency = Counter()

    for _, row in df.iterrows():

        target_id = str(
            row["identity"]
        )

        links = parse_linked_ids(
            row["linked_identities"]
        )

        if links:
            rows_with_links += 1

        for linked_id in links:

            linked_ids.add(linked_id)
            link_frequency[linked_id] += 1
            total_links += 1

            uf.union(
                target_id,
                linked_id,
            )

    all_ids = set(
        uf.parent.keys()
    )

    linked_only = (
        linked_ids
        - folder_ids
    )

    components = defaultdict(set)

    for identity in all_ids:
        root = uf.find(identity)
        components[root].add(identity)

    component_sizes = sorted(
        [
            len(members)
            for members
            in components.values()
        ],
        reverse=True,
    )

    folder_component_sizes = []

    for members in components.values():

        count = len(
            members & folder_ids
        )

        if count:
            folder_component_sizes.append(
                count
            )

    folder_component_sizes.sort(
        reverse=True
    )

    print("=" * 80)
    print("FAKEAVCELEB IDENTITY-LINK AUDIT")
    print("=" * 80)

    print()
    print(
        f"Videos: {len(df):,}"
    )

    print(
        f"Folder identities: "
        f"{len(folder_ids):,}"
    )

    print(
        f"Linked identities: "
        f"{len(linked_ids):,}"
    )

    print(
        f"Linked-only identities: "
        f"{len(linked_only):,}"
    )

    print(
        f"All identity tokens: "
        f"{len(all_ids):,}"
    )

    print()
    print(
        f"Rows containing linked identities: "
        f"{rows_with_links:,}"
    )

    print(
        f"Total identity references: "
        f"{total_links:,}"
    )

    print()
    print(
        f"Connected components: "
        f"{len(components):,}"
    )

    print()

    print(
        "Largest component sizes "
        "(all identity tokens):"
    )

    print(
        component_sizes[:20]
    )

    print()

    print(
        "Largest component sizes "
        "(folder identities only):"
    )

    print(
        folder_component_sizes[:20]
    )

    print()

    if folder_component_sizes:

        largest = (
            folder_component_sizes[0]
        )

        pct = (
            largest
            / len(folder_ids)
            * 100
        )

        print(
            f"Largest connected component contains "
            f"{largest}/{len(folder_ids)} "
            f"folder identities "
            f"({pct:.2f}%)."
        )

    print()
    print(
        "Most frequently referenced "
        "linked identities:"
    )

    for identity, count in (
        link_frequency.most_common(20)
    ):
        folder_marker = (
            "folder+linked"
            if identity in folder_ids
            else "linked-only"
        )

        print(
            f"{identity:12s} "
            f"{count:6d} "
            f"{folder_marker}"
        )

    print()

    print(
        "Example rows containing "
        "cross-identity references:"
    )

    examples = df[
        df["linked_identities"]
        .notna()
    ][
        [
            "identity",
            "linked_identities",
            "semantic_class",
            "filename",
        ]
    ].head(20)

    print(
        examples.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()