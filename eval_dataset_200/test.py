import json
import csv


def flatten_evidence(evidence_list):
    if not evidence_list:
        return [{}]
    flat = []
    for ev in evidence_list:
        row = {}
        for k, v in ev.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v, ensure_ascii=False)
            else:
                row[k] = v
        flat.append(row)
    return flat


def collect_all_fieldnames(rows):
    """收集所有可能的字段名"""
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    return sorted(fieldnames)


def jsonl_to_csv(jsonl_path, csv_path):
    rows = []

    # 第一遍：读取所有数据
    with open(jsonl_path, "r", encoding="utf-8-sig") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print("⚠️ JSON 解析失败，已跳过：", e)
                continue

            evidences = flatten_evidence(data.get("evidence", []))

            for ev in evidences:
                row = {}

                # 顶层字段
                for k in [
                    "id", "task_type", "modality", "question",
                    "image_path", "gold_answer",
                    "expected_entities", "expected_relations",
                    "must_include", "must_not_include",
                    "difficulty"
                ]:
                    v = data.get(k)
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v

                # evidence
                for ek, evv in ev.items():
                    row[f"evidence_{ek}"] = evv

                # quality
                q = data.get("quality", {})
                for qk, qv in q.items():
                    row[f"quality_{qk}"] = qv

                # metadata
                m = data.get("metadata", {})
                for mk, mv in m.items():
                    row[f"metadata_{mk}"] = mv

                rows.append(row)

    if not rows:
        print("❌ 没有可写入的数据")
        return

    # 统一表头
    fieldnames = collect_all_fieldnames(rows)

    # 第二遍：写入 CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("✅ CSV 写入完成：", csv_path)


if __name__ == "__main__":
    jsonl_to_csv(r"eval_dataset_200\rejected.jsonl", "rejected.csv")