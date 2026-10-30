import os
import glob
import json
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

LABELS = ["TR", "EN", "MIXED", "NE", "AMBIGUOUS", "OTHER"]

def load_table(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        best_df, max_rows = None, -1
        for _, df in sheets.items():
            if len(df) > max_rows:
                max_rows, best_df = len(df), df
        return best_df
    return pd.read_csv(path)

def find_annotator_file(name):
    for pat in [f"annotation/*{name}*", f"data/**/*{name}*", f"**/*{name}*"]:
        matches = [
            f for f in glob.glob(pat, recursive=True)
            if f.endswith((".xlsx", ".csv", ".xls")) and "key" not in os.path.basename(f).lower()
        ]
        if matches:
            return sorted(matches)[0]
    return None

def detect_label_col(df):
    if "reference_label" in df.columns:
        return "reference_label"
    if "your label" in df.columns:
        return "your label"
    best_col, best_matches = df.columns[-1], -1
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip().str.upper()
        match_count = vals.isin(LABELS).sum()
        if match_count > best_matches:
            best_matches, best_col = match_count, col
    return best_col

def detect_text_col(df, label_col):
    best_col, max_score = df.columns[0], -1
    for col in df.columns:
        if col == label_col:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        bracket_hits = s.str.contains(r"\[\[", regex=True).sum()
        score = s.str.len().mean() + (bracket_hits * 100)
        if score > max_score:
            max_score, best_col = score, col
    return best_col

def safe_kappa(y1, y2, labels=None):
    try:
        if len(y1) == 0:
            return 0.0
        if set(y1) == set(y2) and len(set(y1)) == 1:
            return 1.0
        val = cohen_kappa_score(y1, y2, labels=labels)
        return 0.0 if np.isnan(val) else float(val)
    except Exception:
        return 0.0

def per_label_kappa(y1, y2):
    out = {}
    for lbl in LABELS:
        b1 = [1 if x == lbl else 0 for x in y1]
        b2 = [1 if x == lbl else 0 for x in y2]
        out[lbl] = None if (sum(b1) == 0 and sum(b2) == 0) else round(safe_kappa(b1, b2), 4)
    return out

def majority_vote(l1, l2, l3):
    counts = Counter([l1, l2, l3])
    top_label, top_count = counts.most_common(1)[0]
    return (top_label, top_count) if top_count >= 2 else (l1, 1)

def compute_pair_metrics(df_sub):
    pairs = [("tugra", "ibrahim"), ("tugra", "mahmut"), ("ibrahim", "mahmut")]
    overall_kappas, per_label_pair = {}, {}

    for a1, a2 in pairs:
        pname = f"{a1}_vs_{a2}"
        y1, y2 = df_sub[a1].tolist(), df_sub[a2].tolist()
        overall_kappas[pname] = round(safe_kappa(y1, y2, labels=LABELS), 4)
        per_label_pair[pname] = per_label_kappa(y1, y2)

    avg_overall = round(float(np.mean(list(overall_kappas.values()))), 4)
    avg_per_label = {}
    for lbl in LABELS:
        vals = [per_label_pair[p][lbl] for p in per_label_pair if per_label_pair[p][lbl] is not None]
        avg_per_label[lbl] = round(float(np.mean(vals)), 4) if vals else None

    label_counts = {
        lbl: {
            "tugra": int((df_sub["tugra"] == lbl).sum()),
            "ibrahim": int((df_sub["ibrahim"] == lbl).sum()),
            "mahmut": int((df_sub["mahmut"] == lbl).sum()),
            "majority": int((df_sub["majority"] == lbl).sum()),
        }
        for lbl in LABELS
    }
    if "gold" in df_sub.columns:
        for lbl in LABELS:
            label_counts[lbl]["reference_label"] = int((df_sub["gold"] == lbl).sum())

    return {
        "n_items": int(len(df_sub)),
        "unanimous_3_of_3": int((df_sub["agreement_count"] == 3).sum()),
        "majority_2_of_3": int((df_sub["agreement_count"] == 2).sum()),
        "disagreement_1_1_1": int((df_sub["agreement_count"] == 1).sum()),
        "pairwise_overall_kappa": overall_kappas,
        "average_overall_kappa": avg_overall,
        "pairwise_per_label_kappa": per_label_pair,
        "average_per_label_kappa": avg_per_label,
        "label_counts": label_counts,
    }

def compute_vs_reference(df_sub, ref_col="gold"):
    ref = df_sub[ref_col].tolist()
    res = {}
    for who in ["tugra", "ibrahim", "mahmut", "majority"]:
        pred = df_sub[who].tolist()
        per_lbl_f1 = {}
        for lbl in LABELS:
            b_ref = [1 if x == lbl else 0 for x in ref]
            b_prd = [1 if x == lbl else 0 for x in pred]
            if sum(b_ref) == 0 and sum(b_prd) == 0:
                per_lbl_f1[lbl] = None
            else:
                per_lbl_f1[lbl] = round(float(f1_score(b_ref, b_prd, zero_division=0)), 4)
        res[who] = {
            "kappa": round(safe_kappa(ref, pred, labels=LABELS), 4),
            "macro_f1": round(float(f1_score(ref, pred, labels=LABELS, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(ref, pred)), 4),
            "per_label_f1": per_lbl_f1,
        }
    return res

def format_split_section(title, metrics):
    lines = [f"### {title} (n = {metrics['n_items']})\n"]
    lines.append(
        f"- **Pairwise Overall κ:** Tuğra–İbrahim `{metrics['pairwise_overall_kappa']['tugra_vs_ibrahim']:.4f}` · "
        f"Tuğra–Mahmut `{metrics['pairwise_overall_kappa']['tugra_vs_mahmut']:.4f}` · "
        f"İbrahim–Mahmut `{metrics['pairwise_overall_kappa']['ibrahim_vs_mahmut']:.4f}` → "
        f"**Mean κ = `{metrics['average_overall_kappa']:.4f}`**"
    )
    lines.append(
        f"- **Consensus:** 3/3 Unanimous `{metrics['unanimous_3_of_3']}` ({metrics['unanimous_3_of_3']/metrics['n_items']*100:.1f}%) · "
        f"2/3 Majority `{metrics['majority_2_of_3']}` ({metrics['majority_2_of_3']/metrics['n_items']*100:.1f}%) · "
        f"1/1/1 Disagreement `{metrics['disagreement_1_1_1']}` ({metrics['disagreement_1_1_1']/metrics['n_items']*100:.1f}%)\n"
    )
    has_ref = "reference_label" in metrics["label_counts"]["TR"]
    hdr = "| Label | Tuğra–İbrahim κ | Tuğra–Mahmut κ | İbrahim–Mahmut κ | **Mean κ** | Counts (Tuğra / İbrahim / Mahmut / Maj" + (" / Ref) |" if has_ref else ") |")
    lines.append(hdr)
    lines.append("|---|---|---|---|---|---|")
    for lbl in LABELS:
        k_ti = metrics["pairwise_per_label_kappa"]["tugra_vs_ibrahim"][lbl]
        k_tm = metrics["pairwise_per_label_kappa"]["tugra_vs_mahmut"][lbl]
        k_im = metrics["pairwise_per_label_kappa"]["ibrahim_vs_mahmut"][lbl]
        k_av = metrics["average_per_label_kappa"][lbl]
        c = metrics["label_counts"][lbl]
        s_ti = f"{k_ti:.4f}" if k_ti is not None else "N/A"
        s_tm = f"{k_tm:.4f}" if k_tm is not None else "N/A"
        s_im = f"{k_im:.4f}" if k_im is not None else "N/A"
        s_av = f"**{k_av:.4f}**" if k_av is not None else "N/A"
        cnt_str = f"{c['tugra']} / {c['ibrahim']} / {c['mahmut']} / **{c['majority']}**"
        if has_ref:
            cnt_str += f" / `{c['reference_label']}`"
        lines.append(f"| **`{lbl}`** | {s_ti} | {s_tm} | {s_im} | {s_av} | {cnt_str} |")
    lines.append("")
    return lines

def main():
    os.makedirs("results", exist_ok=True)
    annotators = ["tugra", "ibrahim", "mahmut"]
    dfs, base_df, detected_lcol = {}, None, None

    for name in annotators:
        fpath = find_annotator_file(name)
        df = load_table(fpath)
        lcol = detect_label_col(df)
        detected_lcol = lcol
        dfs[name] = df[lcol].fillna("MISSING").astype(str).str.strip().str.upper().tolist()
        if base_df is None:
            base_df = df.copy()

    master = pd.DataFrame({
        "row_id": range(1, len(dfs["tugra"]) + 1),
        "tugra": dfs["tugra"],
        "ibrahim": dfs["ibrahim"],
        "mahmut": dfs["mahmut"],
    })

    tcol = detect_text_col(base_df, detected_lcol)
    master["text"] = base_df[tcol].astype(str).tolist()

    maj_labels, maj_counts = [], []
    for r in master.itertuples():
        ml, mc = majority_vote(r.tugra, r.ibrahim, r.mahmut)
        maj_labels.append(ml)
        maj_counts.append(mc)
    master["majority"] = maj_labels
    master["agreement_count"] = maj_counts

    kpath = "annotation/key.csv"
    has_key = os.path.isfile(kpath)
    if has_key:
        kdf = pd.read_csv(kpath)
        master["gold"] = kdf["reference_label"].astype(str).str.strip().str.upper().tolist()
        master["source"] = kdf["source"].astype(str).str.strip().str.lower().tolist()
        master["word"] = kdf["word"].astype(str).tolist() if "word" in kdf.columns else [""] * len(master)
        master["reference_id"] = kdf["reference"].astype(str).tolist() if "reference" in kdf.columns else [""] * len(master)

    report_data = {
        "has_answer_key": has_key,
        "overall_all_items": compute_pair_metrics(master),
    }

    md = ["# Human Agreement Study (`src/agreement.py`)\n"]
    md.extend(format_split_section("1. Overall Inter-Annotator Agreement — All Items", report_data["overall_all_items"]))

    if has_key:
        real_df = master[master["source"] == "real"].copy()
        gen_df = master[master["source"] == "generated"].copy()

        report_data["real_items_agreement"] = compute_pair_metrics(real_df)
        report_data["generated_items_agreement"] = compute_pair_metrics(gen_df)
        report_data["human_ceiling_vs_turengmix_gold"] = compute_vs_reference(real_df, "gold")
        report_data["generated_quality_vs_our_labels"] = compute_vs_reference(gen_df, "gold")

        md.extend(format_split_section("2. Inter-Annotator Agreement — Real TurEngMix Items Only", report_data["real_items_agreement"]))
        md.extend(format_split_section("3. Inter-Annotator Agreement — Generated (Synthetic) Items Only", report_data["generated_items_agreement"]))

        md.append("## 4. Annotators vs. TurEngMix Gold on Real Items (Measured Human Ceiling)\n")
        md.append("| Annotator | Cohen's Kappa (κ) | Macro-F1 (Human Ceiling) | Accuracy | MIXED F1 | NE F1 |")
        md.append("|---|---|---|---|---|---|")
        for who, m in report_data["human_ceiling_vs_turengmix_gold"].items():
            mf1 = m["per_label_f1"].get("MIXED")
            nf1 = m["per_label_f1"].get("NE")
            md.append(
                f"| **{who.title()}** | `{m['kappa']:.4f}` | **`{m['macro_f1']:.4f}`** | `{m['accuracy']:.4f}` | "
                f"`{mf1:.4f}` | `{nf1:.4f}` |"
            )
        md.append("")

        md.append("## 5. Annotators & Majority Vote vs. Our Labels on Generated Items (Synthetic Data QA)\n")
        md.append("| Annotator | Cohen's Kappa (κ) | Macro-F1 | Accuracy | MIXED F1 | NE F1 |")
        md.append("|---|---|---|---|---|---|")
        for who, m in report_data["generated_quality_vs_our_labels"].items():
            mf1 = m["per_label_f1"].get("MIXED")
            nf1 = m["per_label_f1"].get("NE")
            md.append(
                f"| **{who.title()}** | `{m['kappa']:.4f}` | **`{m['macro_f1']:.4f}`** | `{m['accuracy']:.4f}` | "
                f"`{mf1:.4f}` | `{nf1:.4f}` |"
            )
        md.append("")

    disagreements = master[master["agreement_count"] == 1]
    if len(disagreements) > 0:
        md.append("## 6. All 1/1/1 Total Disagreements (Edge Cases)\n")
        if has_key:
            md.append("| Row | Source | Target Word | Gold/Ref | Tuğra | İbrahim | Mahmut | Context Snippet |")
            md.append("|---|---|---|---|---|---|---|---|")
            for r in disagreements.itertuples():
                raw = str(r.text).replace("|", " ").replace("\n", " ")
                idx = raw.find("[[")
                start = max(0, idx - 30) if idx != -1 else 0
                snippet = ("..." if start > 0 else "") + raw[start:start + 80] + ("..." if start + 80 < len(raw) else "")
                md.append(f"| {r.row_id} | `{r.source}` | **`{r.word}`** | `{r.gold}` | `{r.tugra}` | `{r.ibrahim}` | `{r.mahmut}` | {snippet} |")
        else:
            md.append("| Row | Text Snippet | Tuğra | İbrahim | Mahmut |")
            md.append("|---|---|---|---|---|")
            for r in disagreements.itertuples():
                md.append(f"| {r.row_id} | {r.text[:80]} | `{r.tugra}` | `{r.ibrahim}` | `{r.mahmut}` |")

    with open("results/human_agreement.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    with open("results/human_agreement.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md))

if __name__ == "__main__":
    main()
