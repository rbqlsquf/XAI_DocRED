import json
import numpy as np


def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(prediction)
    gold_sp_pred = set(gold)
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if cur_sp_pred == gold_sp_pred else 0.0
    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall
    return em, prec, recall


def eval(file):
    with open(file) as f:
        json_data = json.load(f)

    metrics = {
        "sp_em": 0,
        "sp_f1": 0,
        "sp_prec": 0,
        "sp_recall": 0,
    }

    for data in json_data:
        data_number = data["data_number"]
        real_answer = data["real_answer"]
        gpt_answer = data["gpt_answer"]
        real_support = data["real_support"]
        gpt_support = data["gpt_support"]

        # supporting fact 점수 측정
        sp_em, sp_prec, sp_recall = update_sp(metrics, gpt_support, real_support)

    N = len(json_data)
    for k in metrics.keys():
        metrics[k] /= N
    with open("output_f1.txt", "w", encoding="UTF-8") as out_file:
        out_file.write(json.dumps(metrics, indent=4))
        print(metrics)


def exact_match_score(prediction, ground_truth):
    return int(prediction == ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(prediction_tokens)
    recall = num_common / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


if __name__ == "__main__":

    pred_path = "predict_8000.json"

    gold_data = []

    with open(pred_path, "r", encoding="UTF8") as f2:
        pred_data = json.load(f2)

    metrics = {
        "sp_em": 0,
        "sp_f1": 0,
        "sp_prec": 0,
        "sp_recall": 0,
    }
    all_data_id = []
    all_sp_em = []
    all_sp_prec = []
    all_sp_recall = []
    for i, data in enumerate(pred_data):
        pred_sp = data["evidence_index"]
        gold_sp = data["evidence_sentence"]
        if len(gold_sp) == 0:
            continue
        sp_em, sp_prec, sp_recall = update_sp(metrics, pred_sp, gold_sp)
        ######틀린거 제외하고 해보기
        if data['predict'] != data['answer']:
            continue
        
        all_data_id.append(data["data_id"])
        all_sp_em.append(sp_em)
        all_sp_prec.append(sp_prec)
        all_sp_recall.append(sp_recall)

    for k in metrics.keys():
        metrics[k] /= len(all_data_id)

    with open("output_sp_correct.jsonl", "w", encoding="UTF-8") as out_file:
        for i in range(len(all_sp_em)):
            json.dump(
                {
                    "data_id": all_data_id[i],
                    "em": all_sp_em[i],
                    "sp_prec": all_sp_prec[i],
                    "recall": all_sp_recall[i],
                },
                out_file,
            )
            out_file.write("\n")
        json.dump(metrics, out_file)
        print(metrics)
