import json

with open("data/train_data.json", "r") as f:
    json_data = json.load(f)
na_triple = []
evidence = []
title = []

for data in json_data:
    na_triple.append(data["na_triple"])
    evidence.append(data["evidence"])
    title.append(data["title"])


check_dict_evidence = {}
check_dict_title = {}
all_result = []
state = False
for idx, triple in enumerate(na_triple):
    triple_tuple = tuple(triple)
    if triple_tuple in check_dict_evidence:
        skip_outer_loop = False
        for evidence_ in check_dict_evidence[triple_tuple]:
            if evidence_ == evidence[idx]:
                for title_ in check_dict_title[triple_tuple]:
                    if title_ == title[idx]:
                        skip_outer_loop = True
                        break
            if skip_outer_loop:
                break
        if skip_outer_loop:
            continue

    else:
        check_dict_evidence[triple_tuple] = []
        check_dict_title[triple_tuple] = []
    check_dict_evidence[triple_tuple].append(evidence[idx])
    check_dict_title[triple_tuple].append(title[idx])

    all_result.append(json_data[idx])
