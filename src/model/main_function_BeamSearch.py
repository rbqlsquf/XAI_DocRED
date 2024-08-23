from torch.nn import functional as F
import os
import torch
import timeit
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import Module
from torch import Tensor
from src.functions.utils import load_examples, set_seed, to_list, load_input_data
from src.functions.processor_sent import SquadResult
from src.functions.evaluate_v1_0 import eval_during_train, f1_score
from src.functions.hotpotqa_metric import eval
from src.functions.squad_metric import (
    compute_predictions_logits,
    restore_prediction,
    restore_prediction2,
    save_file_with_evidence,
)
from torch import nn


def batch_features(features, batch_size):
    for i in range(0, len(features), batch_size):
        yield features[i : i + batch_size]


def macro_f1_score(y_true, y_pred, num_classes):
    f1_scores = []

    for i in range(num_classes):
        # True positives, False positives, False negatives
        tp = torch.sum((y_true == i) & (y_pred == i)).item()
        fp = torch.sum((y_true != i) & (y_pred == i)).item()
        fn = torch.sum((y_true == i) & (y_pred != i)).item()

        # Precision and Recall
        precision = tp / (tp + fp + 1e-8)  # Adding epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    # Macro F1 Score
    macro_f1 = sum(f1_scores) / num_classes
    return macro_f1


def train(args, model, tokenizer, logger):
    # 학습에 사용하기 위한 dataset Load
    # examples, features = load_examples(args, tokenizer, evaluate=False, output_examples=True)
    dataset, examples, features = load_examples(args, tokenizer, evaluate=False, output_examples=True)
    # train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.train_batch_size)

    # #########feature 수 뽑기
    # shuffled_indices = list(train_sampler)

    # # Shuffle the features using the same indices
    # shuffled_features = [features[idx] for idx in shuffled_indices]
    ###########feature sampler###########################
    shuffled_features_batches = list(batch_features(features, args.train_batch_size))

    # optimization 최적화 schedule 을 위한 전체 training step 계산
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Layer에 따른 가중치 decay 적용
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # optimizer 및 scheduler 선언
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Training Step
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Train batch size per GPU = %d", args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1 if args.from_init_weight else int(args.checkpoint) + 1

    tr_loss, logging_loss = 0.0, 0.0

    # loss buffer 초기화
    model.zero_grad()
    # a = nn.GRU(768, 768, bidirectional=True)
    set_seed(args)
    # 하나의 example에는 10개의 문서 정보가 모두 담겨있고 10개의 feature (=features)로 분할됨
    # 메모리 문제때문에 실시간으로 tensor로 바꿔줌
    # 전처리 부분은 어차피 데이터 따라서 죄다 다시해야하는거라 중요하게 보진 않아도 됨
    for epoch in range(args.num_train_epochs):
        for step, (batch, batch_for_feature) in enumerate(zip(train_dataloader, shuffled_features_batches)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # 모델에 입력할 입력 tensor 저장
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "sent_masks": batch[3],
                "answer": batch[4],
                # "question_type": all_question_type
            }

            outputs = model(**inputs)
            loss, sampled_evidence_scores, mask, label_logits, sampled_evidence_sentence = outputs
            """
            loss : (num_samples)
                    Output Layer에 적용된 Evidence Path에 따라 Loss가 return
            sampled_evidence_scores : (num_samples, max_dec_length, num_docs * num_sent)
                    Evidence Path 별로 추출된 문장의 확률 값
            mask : (1, max_dec_length, num_docs * num_sent)
                    문서 별 문장에 대한 Padding Mask
            start_logits : (num_docs, max_length(=512), num_samples)
            end_logits : (num_docs, max_length(=512), num_samples)
            sampled_evidence_sentence : (num_sample, max_dec_length)
                    reasoning path 별 문장 번호
           """
            r_batch_size = label_logits.size(0)
            sampled_evidence_scores = sampled_evidence_scores.view(
                -1, args.num_samples, args.max_dec_len, args.max_sent_num
            )
            #####label_logits는 확률로 사용할 거임
            soft_label_logits = F.softmax(label_logits, dim=1)

            predicted_answer = []
            evidence_predicted_answer = []
            # print("\n".join([str(e) for e in sampled_evidence_sentence.tolist()]))

            # (Evidence Path 별로) Evidence Vector로부터 Predict Answer 추론
            for path in range(args.num_samples):
                all_results = []
                # label_logtis : [batch, 긍부정label 수(2), path수?? 근거 문장수] -> label logits : [10, 2] -> 근거문장수에 대한거
                label_logit = soft_label_logits[:, :, path]

                batch_size = soft_label_logits.size(0)

                # unique_id : [batch]
                unique_id = batch[5]  # 각 데이터 별 unique_id

                # predicted_answer : [batch]
                predicted_answer.append(label_logit)
                # evidence_path : [batch, 근거문장수 3]

                evidence_path = (
                    sampled_evidence_sentence.view(batch_size, -1, args.max_dec_len).transpose(0, 1)[path].tolist()
                )

                batch_tmp_input_ids = []
                batch_tmp_attention_mask = []
                batch_tmp_sentence_mask = []
                for k in range(batch_size):

                    tmp_sentence_mask = [0] * len(batch_for_feature[k].cur_sent_range[0])
                    see_tokens = batch_for_feature[k].cur_sent_range[0]  # 질문에 해당하는 값
                    for j in range(args.max_dec_len):
                        see_tokens.extend(batch_for_feature[k].cur_sent_range[evidence_path[k][j]])
                        #!!!!! j+1 인거 왜 인지 확인해보기
                        tmp_sentence_mask = tmp_sentence_mask + [j + 1] * len(
                            batch_for_feature[k].cur_sent_range[evidence_path[k][j]]
                        )

                    sentences = batch[0][k][see_tokens]
                    # 일단 내용 다 복사한 다음
                    # if batch_for_feature[k].cur_sent_rage
                    tmp_input_ids = sentences.tolist()

                    tmp_input_ids = tmp_input_ids[: args.max_seq_length - 1] + [tokenizer.sep_token_id]
                    tokens = tokenizer.convert_ids_to_tokens(tmp_input_ids)
                    tmp_attention_mask = torch.zeros([1, args.max_seq_length], dtype=torch.long)
                    input_mask = [e for e in range(len(tmp_input_ids))]
                    tmp_attention_mask[:, input_mask] = 1
                    tmp_input_ids = tmp_input_ids + [tokenizer.pad_token_id] * (
                        args.max_seq_length - len(tmp_input_ids)
                    )
                    tmp_sentence_mask = tmp_sentence_mask[: args.max_seq_length] + [0] * (
                        args.max_seq_length - len(tmp_sentence_mask)
                    )
                    batch_tmp_input_ids.append(tmp_input_ids)
                    batch_tmp_attention_mask.append(tmp_attention_mask.squeeze(dim=0).tolist())
                    batch_tmp_sentence_mask.append(tmp_sentence_mask)

                input_tmp_input_ids = torch.tensor(batch_tmp_input_ids, dtype=torch.long).cuda()
                input_tmp_attention_mask = torch.tensor(batch_tmp_attention_mask, dtype=torch.long).cuda()
                input_tmp_sentence_mask = torch.tensor(batch_tmp_sentence_mask, dtype=torch.long).cuda()
                sample_inputs = {
                    "input_ids": input_tmp_input_ids,
                    "attention_mask": input_tmp_attention_mask,
                    "sent_masks": input_tmp_sentence_mask,
                }
                e_label_logits, e_sampled_evidence_sentence = model(**sample_inputs)
                ################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # label_logits : [batch, 긍부정, path]
                softmax_e_label_logits = F.softmax(e_label_logits, dim=1)

                label = softmax_e_label_logits[:, :, 0]
                # end = e_end_logits[0, :, 0]
                evidence_predicted_answer.append(label)
            ############################################여기서 부터!!!!!!!!!!!!!!!!!!!!
            # Evidence Path 별로 점수 측정
            ####이제 predicted_answer 와 evidence_predicted_answer를 가지고 점수 측정을 진행해야함
            # num_sample, batch 사이즈 만큼의 listy
            pred_prob_list = [[1e-3 for _ in range(r_batch_size)] for _ in range(args.num_samples)]
            g_pred_prob_list = [[1e-3 for _ in range(r_batch_size)] for _ in range(args.num_samples)]
            gold_list = batch[4]  # 실제 라벨링 정답 값 [batch, 1] -> one-hot으로 만들어서 뻬야함
            gold_list = F.one_hot(gold_list, num_classes=args.num_label)  # [batch, 2] 로 변경
            for path in range(args.num_samples):
                prob = predicted_answer[path]  # 사이즈가 batch, 2일 것임
                e_prob = evidence_predicted_answer[path]
                prob_label = torch.argmax(prob, dim=1)
                e_prob_label = torch.argmax(e_prob, dim=1)
                diff = prob_label != e_prob_label  # 다른 부분 찾는 코드

                # prob : 예측과 evidence 예측, e_prob : evidence예측과 gold !!!
                prob = abs(evidence_predicted_answer[path] - predicted_answer[path])
                e_prob = gold_list - evidence_predicted_answer[path]
                mased_prob = prob * gold_list
                mased_e_prob = e_prob * gold_list

                prob = mased_prob.sum(
                    dim=1, keepdim=True
                )  # -> 확률값들 중에서 1에 해당하는 값만 가지고온 값,  [batch, 1]
                e_prob = mased_e_prob.sum(
                    dim=1, keepdim=True
                )  # -> 확률값들 중에서 1에 해당하는 값만 가지고온 값,  [batch, 1]

                pred_prob_list[path] = (1 - prob).tolist()
                g_pred_prob_list[path] = (1 - e_prob).tolist()
            # pred_prob_list : [path, batch, 1]
            pred_prob_list = torch.tensor(pred_prob_list, dtype=torch.float).cuda()
            g_pred_prob_list = torch.tensor(g_pred_prob_list, dtype=torch.float).cuda()
            # 가장 성능이 높은 Path 선정
            ll = to_list(pred_prob_list + g_pred_prob_list)
            ll = torch.tensor(ll, dtype=torch.float).cuda()
            best_path = torch.max(ll, dim=0).indices

            # Evidence Path Loss 계산을 위한 Mask 생성 : [batch, max_dec_len, max_sentences]
            s_sampled_evidence_sentence = torch.zeros(
                [args.num_samples, r_batch_size, args.max_dec_len, sampled_evidence_scores.size(-1)],
                dtype=torch.long,
            ).cuda()
            g_sampled_evidence_sentence = torch.zeros(
                [args.num_samples, r_batch_size, args.max_dec_len, sampled_evidence_scores.size(-1)],
                dtype=torch.long,
            ).cuda()
            # r_sampled_evidence_sentence : [batch, path, 근거 문장]
            r_sampled_evidence_sentence = sampled_evidence_sentence.view(-1, args.num_samples, args.max_dec_len)
            for idx in range(args.num_samples):

                # r_sampled_evidence_sentence : [batch, 근거문장수, 40] -> one-hot 벡터로 문장 위치표시
                sampled_sampled_evidence_sentence = F.one_hot(
                    r_sampled_evidence_sentence[:, idx, :], num_classes=sampled_evidence_scores.size(-1)
                )  # 최대 문장에 대한 one-hot 인코딩 진행

                # negative_sampled_evidence_sentence : [batch, 1, 40] -> 실제 근거 문장들 표시!
                negative_sampled_evidence_sentence = torch.sum(sampled_sampled_evidence_sentence, 1, keepdim=True)
                prob = pred_prob_list[idx]  # [batch, 1]
                g_prob = g_pred_prob_list[idx]  # [batch, 1]

                # Evidence Vector based Answer <=> Evidence Sentence based Answer
                #                  Gold Answer <=> Evidence Sentence based Answer

                # 점수가 낮은 경우 추론된 Evidence Sentence를 제외한 모든 문장의 확률을 높이도록
                # mask : [batch, 근거문장수, 40] // negative_sampled_evidence_sentence : [batch, 1, 40]
                # s_sampled_evidence_sentence[idx, : , :] = [batch, 근거문장, 40] -> 그렇다면 이것도 사이즈를 늘려야지
                for batch_idx in range(len(prob)):
                    if prob[batch_idx] < 0.5:
                        s_sampled_evidence_sentence[idx, batch_idx, :, :] = (
                            mask[batch_idx] - negative_sampled_evidence_sentence[batch_idx]
                        )
                    else:
                        s_sampled_evidence_sentence[idx, batch_idx, :, :] = sampled_sampled_evidence_sentence[
                            batch_idx, :, :
                        ]
                    if g_prob[batch_idx] < 0.5:
                        g_sampled_evidence_sentence[idx, batch_idx, :, :] = (
                            mask[batch_idx] - negative_sampled_evidence_sentence[batch_idx]
                        )
                    else:
                        g_sampled_evidence_sentence[idx, batch_idx, :, :] = sampled_sampled_evidence_sentence[
                            batch_idx, :, :
                        ]

            e_div = torch.sum(s_sampled_evidence_sentence, -1)
            g_div = torch.sum(g_sampled_evidence_sentence, -1)
            # evidence_nll : [batch, path, 근거문장수, 40] -> [path, batch, 근거문장, 40]
            evidence_nll = -F.log_softmax(sampled_evidence_scores, -1).transpose(0, 1)
            g_evidence_nll = -F.log_softmax(sampled_evidence_scores, -1).transpose(0, 1)

            evidence_nll = evidence_nll * s_sampled_evidence_sentence
            g_evidence_nll = g_evidence_nll * g_sampled_evidence_sentence

            evidence_nll = torch.mean(torch.sum(evidence_nll, -1) / e_div, -1)
            evidence_nll = evidence_nll * pred_prob_list.squeeze(dim=-1)
            # evidence_nll : [path, batch]
            g_evidence_nll = torch.mean(torch.sum(g_evidence_nll, -1) / g_div, -1)
            g_evidence_nll = g_evidence_nll * g_pred_prob_list.squeeze(dim=-1)

            column_indices = torch.arange(best_path.size(0), device="cuda:0")
            # loss = loss.unsqueeze(dim=1).repeat(1, r_batch_size)
            if torch.mean(evidence_nll).item() != 0 and torch.mean(evidence_nll).item() < 1000:
                loss = loss + 0.1 * evidence_nll

            if torch.mean(g_evidence_nll).item() != 0 and torch.mean(evidence_nll).item() < 1000:
                loss = loss + 0.1 * g_evidence_nll

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if torch.mean(loss).item() > 0 and torch.mean(loss).item() < 1000:
                loss = loss[best_path.squeeze(dim=1), column_indices]
                loss.mean().backward()
            else:
                # outputs = model(**inputs)
                print("Loss Error - ", step)
                continue

            tr_loss += loss.mean()

            # Loss 출력
            if (global_step + 1) % 50 == 0:
                print("{} step processed.. Current Loss : {}".format((global_step + 1), loss.mean().item()))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # model save
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 모델 저장 디렉토리 생성
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # 학습된 가중치 및 vocab 저장
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Validation Test!!
                    logger.info("***** Eval results *****")
                    evaluate(args, model, tokenizer, logger, global_step=global_step)

    return global_step, tr_loss / global_step


# 정답이 사전부착된 데이터로부터 평가하기 위한 함수
def evaluate(args, model, tokenizer, logger, global_step=""):
    # 데이터셋 Load
    dataset, examples, features = load_examples(args, tokenizer, evaluate=True, output_examples=True)
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)
    shuffled_features_batches = list(batch_features(features, args.eval_batch_size))
    # 최종 출력 파일 저장을 위한 디렉토리 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(global_step))
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 모델 출력을 저장하기위한 리스트 선언
    all_results = []

    # 평가 시간 측정을 위한 time 변수
    start_time = timeit.default_timer()
    model.eval()
    tmp_scores = []
    answer_list = []
    pred_list = []
    accuracy = 0

    for batch_idx, (batch, batch_for_feature) in enumerate(tqdm(zip(test_dataloader, shuffled_features_batches))):
        # 모델을 평가 모드로 변경
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # 평가에 필요한 입력 데이터 저장
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "sent_masks": batch[3],
            }
            # outputs = (start_logits, end_logits)
            # start_logits: [batch_size, max_length]
            # end_logits: [batch_size, max_length]
            outputs = model(**inputs)

            label_logits, sampled_evidence_sentence = outputs
            sampled_evidence_sentence = sampled_evidence_sentence.view(-1, args.num_samples, args.max_dec_len)
            optim_sample_id = 0

        batch_size = args.eval_batch_size
        eval_feature = batch_for_feature

        # 입력 질문에 대한 N개의 결과 저장하기위해 q_id 저장
        unique_id = batch[5]
        label_logit, evidence, pred_label = (
            to_list(label_logits[:, :, 0]),
            # to_list(sampled_evidence_sentence[:, optim_sample_id, 0]),
            to_list(sampled_evidence_sentence[:, 0, :]),
            to_list(torch.argmax(label_logits[:, :, 0], dim=1)),
        )
        predicted_answer = torch.argmax(label_logits[:, :, 0], dim=1)

        answer_label = batch[4]
        accuracy += int(torch.sum(predicted_answer == answer_label))
        # q_id에 대한 예측 정답 시작/끝 위치 확률 저장
        result = SquadResult(unique_id, label_logit, evidence, pred_label=pred_label)

        # feature에 종속되는 최종 출력 값을 리스트에 저장
        all_results.append(result)
        answer_list.append(answer_label)
    accuracy = float(accuracy) / len(examples)
    print("accuracy : ", accuracy)

    ######################이제 파일 작성하는 거해야함

    # 평가 시간 측정을 위한 time 변수
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(features))

    # 최종 예측 값을 저장하기 위한 파일 생성
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))

    #####파일 작성하는 함수만들기
    save_file_with_evidence(examples, shuffled_features_batches, all_results, output_prediction_file, tokenizer)
