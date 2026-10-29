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
        best_sheet, best_df, max_rows = None, None, -1
        for sname, df in sheets.items():
            if len(df) > max_rows:
                max_rows, best_df, best_sheet = len(df), df, sname
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
    best_col, best_matches = None, -1
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.strip().str.upper()
        match_count = vals.isin(LABELS).sum()
        if match_count > best_matches:
            best_matches, best_col = match_count, col
    return best_col if best_matches > 0 else df.columns[-1]

def detect_text_col(df, label_col):
    # En uzun ortalama string uzunluğuna veya [[ ]] işaretine sahip sütunu bulur (ID sütunlarını atlar)
    best_col, max_avg_len = df.columns[0], -1
    for col in df.columns:
        if col == label_col:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        bracket_hits = s.str.contains(r"\[\[", regex=True).sum()
        avg_len = s.str.len().mean() + (bracket_hits * 100)
        if avg_len > max_avg_len:
            max_avg_len, best_col = avg_len, col
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
            label_counts[lbl]["gold"] = int((df_sub["gold"] == lbl).sum())

    return {
        "n_items": int(len(df_sub)),
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
        res[who] = {
            "kappa": round(safe_kappa(ref, pred, labels=LABELS), 4),
            "macro_f1": round(float(f1_score(ref, pred, labels=LABELS, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(ref, pred)), 4),
        }
    return res

def main():
    os.makedirs("results", exist_ok=True)
    annotators = ["tugra", "ibrahim", "mahmut"]
    dfs, base_df, detected_lcol = {}, None, None

    for name in annotators:
        fpath = find_annotator_file(name)
        if not fpath:
            raise FileNotFoundError(f"'{name}' icin annotasyon dosyasi bulunamadi!")
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

    key_candidates = [
        f for f in (glob.glob("annotation/*key*.csv") + glob.glob("data/**/*key*.csv", recursive=True) + glob.glob("**/*key*.csv", recursive=True))
        if os.path.isfile(f)
    ]
    has_key = False
    if key_candidates:
        kpath = sorted(key_candidates)[0]
        print(f"=== CEVAP ANAHTARI BULUNDU: {kpath} ===")
        kdf = load_table(kpath)
        gcol = detect_label_col(kdf)
        scol = next((c for c in kdf.columns if any(k in str(c).lower() for k in ["source", "split", "origin", "type", "dataset", "is_real"])), None)
        master["gold"] = kdf[gcol].astype(str).str.strip().str.upper().tolist()
        master["source"] = kdf[scol].astype(str).str.strip().str.lower().tolist() if scol else ["unknown"] * len(master)
        has_key = True

    report_data = {
        "has_answer_key": has_key,
        "overall_all_items": compute_pair_metrics(master),
        "agreement_breakdown": {
            "unanimous_3_of_3": int((master["agreement_count"] == 3).sum()),
            "majority_2_of_3": int((master["agreement_count"] == 2).sum()),
            "total_disagreement_1_1_1": int((master["agreement_count"] == 1).sum()),
        }
    }

    if has_key and "source" in master.columns:
        real_mask = master["source"].str.contains("real|tureng|gold|orig", case=False, na=False)
        gen_mask = master["source"].str.contains("gen|synth|ours|fake", case=False, na=False)
        if real_mask.any():
            report_data["real_items_agreement"] = compute_pair_metrics(master[real_mask])
            report_data["human_ceiling_vs_turengmix_gold"] = compute_vs_reference(master[real_mask], "gold")
        if gen_mask.any():
            report_data["generated_items_agreement"] = compute_pair_metrics(master[gen_mask])
            report_data["generated_quality_vs_our_labels"] = compute_vs_reference(master[gen_mask], "gold")
        if not real_mask.any() and not gen_mask.any():
            report_data["all_items_vs_gold"] = compute_vs_reference(master, "gold")

    with open("results/human_agreement.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    md = ["# Human Agreement Study (Inter-Annotator Agreement)\n"]
    ov = report_data["overall_all_items"]
    ab = report_data["agreement_breakdown"]
    md.append(f"- **Total Annotated Items:** {ov['n_items']}")
    md.append(f"- **3/3 Unanimous Agreement:** {ab['unanimous_3_of_3']} ({ab['unanimous_3_of_3']/ov['n_items']*100:.1f}%)")
    md.append(f"- **2/3 Majority Agreement:** {ab['majority_2_of_3']} ({ab['majority_2_of_3']/ov['n_items']*100:.1f}%)")
    md.append(f"- **1/1/1 Total Disagreement:** {ab['total_disagreement_1_1_1']} ({ab['total_disagreement_1_1_1']/ov['n_items']*100:.1f}%)\n")

    md.append("## 1. Pairwise Cohen's Kappa (All Items)\n")
    md.append("| Annotator Pair | Overall Cohen's Kappa (κ) |")
    md.append("|---|---|")
    for pair, val in ov["pairwise_overall_kappa"].items():
        md.append(f"| **{pair.replace('_vs_', ' vs. ').title()}** | `{val:.4f}` |")
    md.append(f"| **Average (Mean κ)** | **`{ov['average_overall_kappa']:.4f}`** |\n")

    md.append("## 2. Per-Label Binary Cohen's Kappa & Label Counts\n")
    md.append("| Label | Tuğra vs İbrahim | Tuğra vs Mahmut | İbrahim vs Mahmut | **Mean κ** | Count (Tuğra / İbrahim / Mahmut / Maj) |")
    md.append("|---|---|---|---|---|---|")
    for lbl in LABELS:
        k_ti = ov["pairwise_per_label_kappa"]["tugra_vs_ibrahim"][lbl]
        k_tm = ov["pairwise_per_label_kappa"]["tugra_vs_mahmut"][lbl]
        k_im = ov["pairwise_per_label_kappa"]["ibrahim_vs_mahmut"][lbl]
        k_av = ov["average_per_label_kappa"][lbl]
        c = ov["label_counts"][lbl]
        s_ti = f"{k_ti:.4f}" if k_ti is not None else "N/A"
        s_tm = f"{k_tm:.4f}" if k_tm is not None else "N/A"
        s_im = f"{k_im:.4f}" if k_im is not None else "N/A"
        s_av = f"**{k_av:.4f}**" if k_av is not None else "N/A"
        md.append(f"| **`{lbl}`** | {s_ti} | {s_tm} | {s_im} | {s_av} | {c['tugra']} / {c['ibrahim']} / {c['mahmut']} / **{c['majority']}** |")
    md.append("")

    if "real_items_agreement" in report_data:
        rv = report_data["real_items_agreement"]
        md.append(f"## 3. Real Items Agreement (n={rv['n_items']}, Mean κ = `{rv['average_overall_kappa']:.4f}`)\n")
        md.append("| Annotator | vs. TurEngMix Gold (κ) | Macro-F1 (Human Ceiling) | Accuracy |")
        md.append("|---|---|---|---|")
        for who, m in report_data["human_ceiling_vs_turengmix_gold"].items():
            md.append(f"| **{who.title()}** | `{m['kappa']:.4f}` | `{m['macro_f1']:.4f}` | `{m['accuracy']:.4f}` |")
        md.append("")

    if "generated_items_agreement" in report_data:
        gv = report_data["generated_items_agreement"]
        md.append(f"## 4. Generated Items Agreement (n={gv['n_items']}, Mean κ = `{gv['average_overall_kappa']:.4f}`)\n")
        md.append("| Annotator | vs. Our Synthetic Labels (κ) | Macro-F1 | Accuracy |")
        md.append("|---|---|---|---|")
        for who, m in report_data["generated_quality_vs_our_labels"].items():
            md.append(f"| **{who.title()}** | `{m['kappa']:.4f}` | `{m['macro_f1']:.4f}` | `{m['accuracy']:.4f}` |")
        md.append("")

    disagreements = master[master["agreement_count"] == 1]
    if len(disagreements) > 0:
        md.append("## 5. All 1/1/1 Total Disagreements (Edge Cases)\n")
        md.append("| Row | Text Snippet | Tuğra | İbrahim | Mahmut |")
        md.append("|---|---|---|---|---|")
        for r in disagreements.itertuples():
            raw = str(r.text).replace("|", " ").replace("\n", " ")
            idx = raw.find("[[")
            start = max(0, idx - 35) if idx != -1 else 0
            snippet = ("..." if start > 0 else "") + raw[start:start + 90] + ("..." if start + 90 < len(raw) else "")
            md.append(f"| {r.row_id} | {snippet} | `{r.tugra}` | `{r.ibrahim}` | `{r.mahmut}` |")

    with open("results/human_agreement.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n".join(md[-30:]))

if __name__ == "__main__":
    main()
