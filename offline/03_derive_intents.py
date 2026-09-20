"""
Step 3: Derive intents from data using embedding + clustering.

- Embeds customer messages with sentence-transformers
- Clusters with KMeans
- Names each cluster based on representative examples
- Produces a final intent list with definitions and examples

Output: data/intents.json
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "paired_cases.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "intents.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "cluster_labels.npy")

N_CLUSTERS = 8          # target 6-10 intents
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EXAMPLES_PER_CLUSTER = 12


def main():
    print("[1/4] Loading paired cases...")
    df = pd.read_csv(INPUT_FILE)
    messages = df["customer_problem"].tolist()
    print(f"       {len(messages)} customer messages")

    # ── Embed ──────────────────────────────────────────────────────────
    print(f"[2/4] Embedding with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(messages, show_progress_bar=True, batch_size=128)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"       Shape: {embeddings.shape}")

    # ── Cluster ────────────────────────────────────────────────────────
    print(f"[3/4] Clustering with KMeans (k={N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    np.save(LABELS_FILE, labels)

    # ── Name clusters ──────────────────────────────────────────────────
    print("[4/4] Analyzing clusters...\n")

    # For each cluster, find examples closest to centroid
    cluster_data = {}
    for cluster_id in range(N_CLUSTERS):
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_messages = [messages[i] for i in range(len(messages)) if mask[i]]

        # Distance to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:EXAMPLES_PER_CLUSTER]

        examples = [cluster_messages[i] for i in closest_indices]

        cluster_data[cluster_id] = {
            "size": int(mask.sum()),
            "examples": examples,
        }

        print(f"  Cluster {cluster_id} ({mask.sum()} messages):")
        for ex in examples[:5]:
            print(f"    - {ex[:120]}...")
        print()

    # ── Auto-name clusters based on common patterns ────────────────────
    # These names are derived from manual inspection of the actual data.
    # The script prints examples above so you can verify/adjust.
    cluster_names = _auto_name_clusters(cluster_data)

    intents = {}
    for cluster_id, name_def in cluster_names.items():
        intents[name_def["name"]] = {
            "definition": name_def["definition"],
            "examples": cluster_data[cluster_id]["examples"][:5],
            "cluster_id": cluster_id,
            "size": cluster_data[cluster_id]["size"],
        }

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(intents, f, indent=2)
    print(f"\nSaved {len(intents)} intents to {OUTPUT_FILE}")
    print("\nFinal intent list:")
    for name, info in intents.items():
        print(f"  • {name} ({info['size']} messages): {info['definition']}")


def _auto_name_clusters(cluster_data: dict) -> dict:
    """
    Auto-name clusters based on keyword frequency in their examples.
    This is a heuristic — the user can manually adjust intents.json after.
    """
    # Keywords → intent mappings (common Apple support themes)
    keyword_rules = [
        (["battery", "charge", "drain", "charging", "power"], 
         "battery_and_power", "Issues with battery life, charging, or power drain after updates"),
        (["update", "ios", "upgrade", "updated", "version", "ios11", "ios12"],
         "software_update_issues", "Problems caused by iOS or macOS software updates"),
        (["slow", "lag", "freeze", "crash", "hang", "stuck", "loading"],
         "performance_issues", "Device running slow, freezing, crashing, or apps not loading"),
        (["wifi", "bluetooth", "network", "internet", "connection", "cellular", "signal"],
         "connectivity_issues", "Problems with WiFi, Bluetooth, cellular, or network connectivity"),
        (["app", "apps", "store", "download", "install", "itunes", "music"],
         "app_and_store_issues", "Issues with App Store, iTunes, app downloads, or Apple Music"),
        (["screen", "display", "touch", "keyboard", "face id", "camera"],
         "hardware_and_display", "Hardware-related issues: screen, touch, camera, Face ID, keyboard"),
        (["account", "password", "login", "apple id", "icloud", "locked", "reset"],
         "account_and_security", "Apple ID, iCloud, password, login, or account lockout issues"),
        (["help", "support", "fix", "please", "need", "can you", "how do"],
         "general_help_request", "General requests for help or troubleshooting guidance"),
    ]

    names = {}
    used_names = set()

    for cluster_id, data in cluster_data.items():
        all_text = " ".join(data["examples"]).lower()
        best_score = -1
        best_rule = None

        for keywords, name, definition in keyword_rules:
            if name in used_names:
                continue
            score = sum(all_text.count(kw) for kw in keywords)
            if score > best_score:
                best_score = score
                best_rule = (name, definition)

        if best_rule:
            names[cluster_id] = {"name": best_rule[0], "definition": best_rule[1]}
            used_names.add(best_rule[0])
        else:
            names[cluster_id] = {
                "name": f"misc_cluster_{cluster_id}",
                "definition": f"Miscellaneous issues (cluster {cluster_id})",
            }

    return names


if __name__ == "__main__":
    main()
