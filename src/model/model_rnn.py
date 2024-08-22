from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn as nn
import torch
from torch.nn import functional as F
from transformers.models.electra import ElectraPreTrainedModel, ElectraModel
from transformers import AutoTokenizer, AutoModelForPreTraining
from torch.nn import TransformerEncoderLayer
from transformers.activations import gelu
from random import random, randint, randrange
from torch.nn import MSELoss
import math
import operator
import numpy as np

import torch.nn.init as weight_init


class BeamSearchAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, num_sent, topk=5):
        super(BeamSearchAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_sent = num_sent
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)
        self.topk = topk

    def forward(
        self,
        last_hidden,
        decoder_inputs,
        encoder_outputs,
        attention_scores,
        attention_mask,
        evidence_scores=None,
        evidence_sentence_index=None,
        is_training=True,
    ):
        """
        :param last_hidden: (1, batch, hidden)
        :param decoder_inputs: (batch, 1, hidden)
        :param encoder_outputs: (batch, seq_len, hidden)
        :return:
        """
        batch_size = decoder_inputs.size(0)
        indexes = [e for e in range(batch_size)]
        key_encoder_outputs = self.dense1(encoder_outputs)
        value_encoder_outputs = self.dense2(encoder_outputs)

        # key : (batch, seq, hidden)
        # value : (batch, seq, hidden)

        output, hidden = self.decoder(decoder_inputs, hx=last_hidden)
        # output : (batch(20), 1, hidden)
        # hidden : (1, batch(20), hidden)
        # t_encoder_outputs : (batch, hidden, seq)
        t_encoder_outputs = key_encoder_outputs.transpose(1, 2)

        # attn_outputs : (batch, 1, max_sent 40), attention_mask : 필요없는 부분이 -1이기 때문에 확률 이빠이 낮춰
        attn_outputs = output.bmm(t_encoder_outputs) / self.div_term + attention_mask

        # attn_alignment : [batch, 1, max_sent 40]
        attn_alignment = F.softmax(attn_outputs, -1)
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch, 1, hidden)

        hidden_states = torch.cat([context, output], -1)
        # result : [batch, 1, hidden]
        result = self.dense3(hidden_states)  # context와 output을 concat한 값
        tmp_result = []
        tmp_hidden = []
        tmp_attention_mask = []
        tmp_attn_outputs = []

        flag = False

        # tmp_attn_alignment = torch.zero ####~~!!! 여기서 부터 질문, attb_alignment는 지금 attention 2에 대해서 진행을 해야하나? 그냥 batch 단위로 하면 되는 거 아닌가?
        top_n_logit_indices = attn_alignment.topk(k=self.topk, dim=-1, sorted=True)
        scores = top_n_logit_indices.values.squeeze(1)
        sentences = top_n_logit_indices.indices.squeeze(1)
        #
        # sentences = torch.argmax(attn_alignment, -1)
        # scores = attn_alignment[:, :, torch.argmax(attn_alignment)]
        if evidence_scores is not None:
            evidence_scores_sum = evidence_scores.unsqueeze(1).repeat(1, self.topk)
            log_scores = -torch.log(scores) + evidence_scores_sum
            l = log_scores.view(-1, self.topk * self.topk).tolist()
            index_and_scores = [sorted(enumerate(e), key=lambda x: x[1], reverse=False) for e in l]
            nodes = {}

            tmp_evidence_scores = []
            refine_evidence_sentences = []
            refine_attention_scores = []
            evidence_sentences = []
            for batch_id, index_and_score in enumerate(index_and_scores):
                tmp_evidence_scores.append([])
                refine_attention_scores.append([])
                evidence_sentences.append([])
                tmp_result.append([])
                tmp_hidden.append([])
                tmp_attention_mask.append([])
                tmp_attn_outputs.append([])
                for sample_id, sorted_node in enumerate(index_and_score[: self.topk]):
                    s, r = int(sorted_node[0] / self.topk), sorted_node[0] % self.topk
                    s = s + batch_id * self.topk
                    tmp_evidence_scores[-1].append(log_scores[s][r])
                    tmp_result[-1].append(result[s])
                    tmp_hidden[-1].append(hidden[0, s])
                    refine_evidence_sentences.append(evidence_sentence_index[s] + [sentences[s][r].item()])
                    refine_attention_scores[-1].append(
                        torch.cat([attention_scores[:, s, :, :], attn_outputs[s, :, :].unsqueeze(0)], 0)
                    )
                    evidence_sentences[-1].append(sentences[s][r])
                    tmp_attention_mask[-1].append(attention_mask[s])
                    tmp_attn_outputs[-1].append(attn_outputs[s])

                tmp_evidence_scores[-1] = torch.stack(tmp_evidence_scores[-1])
                refine_attention_scores[-1] = torch.stack(refine_attention_scores[-1])
                tmp_result[-1] = torch.stack(tmp_result[-1])
                tmp_hidden[-1] = torch.stack(tmp_hidden[-1])
                evidence_sentences[-1] = torch.stack(evidence_sentences[-1])
                tmp_attention_mask[-1] = torch.stack(tmp_attention_mask[-1])
                tmp_attn_outputs[-1] = torch.stack(tmp_attn_outputs[-1])

            evidence_scores = torch.stack(tmp_evidence_scores).view(
                -1,
            )
            attention_scores = (
                torch.stack(refine_attention_scores, 0).view(batch_size, -1, 1, self.num_sent).transpose(0, 1)
            )
            result = torch.stack(tmp_result, 0).view(-1, 1, self.hidden_size)
            hidden = torch.stack(tmp_hidden, 0).view(-1, self.hidden_size).unsqueeze(0)
            evidence_sentence_index = refine_evidence_sentences
            evidence_sentences = torch.stack(evidence_sentences, 0).view(
                -1,
            )
            attention_mask = torch.stack(tmp_attention_mask, 0).view(-1, 1, self.num_sent)
            attn_outputs = torch.stack(tmp_attn_outputs, 0).view(-1, 1, self.num_sent)

        else:
            flag = True
            evidence_scores = -torch.log(scores[0 :: self.topk].reshape(-1))
            evidence_sentences = sentences[0 :: self.topk].reshape(-1)  # 고친거 맞을까?
        # if is_training:
        attention_mask[indexes, 0, evidence_sentences] = -1e10

        if flag:
            evidence_sentence_index = []
            for item in evidence_sentences:
                evidence_sentence_index.append([item.item()])
            attention_scores = attn_outputs.unsqueeze(0)

        return result, hidden, evidence_sentence_index, attention_scores, attention_mask, evidence_scores


class ElectraForQuestionAnsweringBeamSearch(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraForQuestionAnsweringBeamSearch, self).__init__(config)
        # 분류 해야할 라벨 개수 (start/end)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        # ELECTRA 모델 선언
        self.max_seq_length = config.max_position_embeddings
        self.electra = ElectraModel(config)
        self.max_sent_num = config.max_sent_num
        self.num_samples = config.num_samples

        # 추가한 부분
        self.classifier = nn.Linear(in_features=config.hidden_size * 2, out_features=config.num_label)

        # 삭제해야할 부분
        self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.max_dec_len = config.max_dec_len
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.gru = BeamSearchAttentionDecoder(self.hidden_size, self.max_sent_num, self.num_samples)
        self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True, num_layers=1)
        # ELECTRA weight 초기화
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        ##################
        attention_mask=None,
        token_type_ids=None,
        sent_masks=None,
        ##############
        answer=None,
        head=None,
        tail=None,
    ):

        # ELECTRA output 저장
        # outputs : [1, batch_size, seq_length, hidden_size]
        # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
        # outputs : (last-layer hidden state, all hidden states, all attentions)
        # last-layer hidden state : [batch, seq_length, hidden_size]
        # all hidden states : [13, batch, seq_length, hidden_size]
        # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
        # all attentions : [12, batch, num_heads, seq_length, seq_length]
        batch_size = input_ids.size(0)
        sequence_length = input_ids.size(1)
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        is_training = False
        if answer is not None:
            is_training = True

        # sequence_output : [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]
        # cls_output : [batch, hidden]
        cls_outputs = sequence_output[:, 0, :]

        # cls_output : [batch, hidden]

        sentence_masks = F.one_hot(sent_masks, num_classes=self.max_sent_num).transpose(1, 2).float()
        # 0번째 문장은 질문에 해당하는 부분 지정
        # [batch, max_sent, max_length], attention_size : [batch, max_length] !!!cls랑 sep도 들어가는 거 같음 확인...ㅜ
        sentence_masks[:, 0, :] = sentence_masks[:, 0, :] * attention_mask
        # sentence_masks : [batch, seq_length] ==> [batch, seq_len, sent_num] ==> [batch, sent_num, seq_len]
        # sentence_masks : [10, 512] ==> [10, 512, 40] ==> [10, 40, 512]
        # [sentence_masks] = [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, ...]]

        div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
        # div_term : [batch, sent_num, 1]
        # 불필요한 거면 작은 정수값, 값이 있으면 남겨두기, 0번째 문장은 무조건 무시해야하는 거아님? 그래서 무시해야함 !!!
        div_term = div_term.masked_fill(div_term == 0, 1e-10)
        # attention_masks : [1, batch*max_sent], 버릴 부분을 1로 채우고 아닌 부분은 소수로 채운다? 무시할 부분이 1이된느낌?
        # attention_masks = div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).view(1,batch_size * self.max_sent_num).bool()
        attention_masks = (
            div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).squeeze(dim=2).bool()
        )
        sentence_representation = sentence_masks.bmm(sequence_output)
        # 문장별 벡터값을 모은 다음, 전체 토큰수로 나누어주는 부분
        # sentence_representation : [batch, sent_num, hidden]
        sentence_representation = sentence_representation / div_term
        # sentence_representation : [batch, sent_num, hidden]  #!!!기존 코드는 batch 사이즈를 없애는 부분이 있었음. batch 다시 살려주기
        sentence_representation = self.dropout(sentence_representation)

        attention_masks = attention_masks.float()

        # 질문 벡터만 모아서 초기 디코더 input값으로 설정
        last_hidden = None

        # sentence_representation[:, 0::self.max_sent_num, :] 의 사이즈 : [1, batch, hidden]

        # !!! cls_output으로 초기 input 설정하라고 하심

        ######decoder input 구성하기
        # decoder_input : [batch, 1, hidden] #여기를 바꿔주면 되지 않을까?
        decoder_inputs = cls_outputs.unsqueeze(dim=1)

        # encoder_outputs: [batch, max_sent, hidde]
        encoder_outputs = sentence_representation.unsqueeze(dim=1)
        attention_masks[:, 0] = 1  # 질문 부분도 함께 무시해야하는 부분임
        mm = 1 - attention_masks  # 그럼 얘는 신경써야하는 부분임
        mm = mm.unsqueeze(1).expand(-1, self.max_dec_len, -1)
        attention_masks = (
            attention_masks.masked_fill(attention_masks == 1, -1e10).masked_fill(attention_masks == 0, 0).unsqueeze(1)
        )  # 신경써야할 부분은 0으로, 필요없는 부분은 -1e10으로 설정

        # Evidence Path 수만큼 Expand -> 이걸 하는 이유는 beam search 할 때 코드 효율성을 위해 진행하는 거임
        # decoder_input : [batch*5, 1, hidden]
        # encoder_outputs : [batch*5, max_sent, hidden]
        # attention_masks : [batch*5, 1, max_sent]
        decoder_inputs = decoder_inputs.repeat(1, self.num_samples, 1).view(-1, 1, self.hidden_size)
        encoder_outputs = encoder_outputs.repeat(1, self.num_samples, 1, 1).view(
            -1, self.max_sent_num, self.hidden_size
        )
        attention_masks = attention_masks.repeat(1, self.num_samples, 1).view(-1, 1, self.max_sent_num)
        evidence_sentences = []
        evidence_scores = None
        attention_scores = []
        for evidence_step in range(self.max_dec_len):  # max_dec_len : 근거 문장 수
            decoder_inputs, last_hidden, evidence_sentences, attention_scores, attention_masks, evidence_scores = (
                self.gru(
                    last_hidden,
                    decoder_inputs,
                    encoder_outputs,
                    attention_scores,
                    attention_masks,
                    evidence_scores,
                    evidence_sentences,
                )
            )
        # evidence_vector : [hidden, batch*5] -> path 5개를 함께 가지고 온거임
        evidence_vector = decoder_inputs.view(-1, self.num_samples, self.hidden_size)
        # evidence_sentences : [batch * 5, 3]
        evidence_sentences = torch.tensor(evidence_sentences, dtype=torch.long).cuda()

        # attention_scores : [batch*5, 3, max_sent]
        attention_scores = attention_scores.squeeze(2).transpose(0, 1)

        if is_training:
            evidence = evidence_sentences
        else:
            evidence = evidence_sentences

        ######수정하는 부분 최종 output은 batch, 2, num_sent
        # sequence_output : [batch, max_length, hidden] -> [batch, 2, hidden]
        expanded_cls = cls_outputs.unsqueeze(1).expand(-1, self.num_samples, -1)
        sentence_vector = torch.cat([expanded_cls, evidence_vector], -1)

        # linear_classifier : [batch, path, hidden* 2] -> [batch, path, 97]
        # label_logits : [batch, 2, hidden] * [batch, hidden, num_sent] = [batch, 2, num_sent]
        label_logits = self.classifier(sentence_vector).permute(0, 2, 1)
        # label_logits = linear_classfier.matmul(evidence_vector)

        # outputs = (start_logits, end_logits)
        # outputs = (label_logits)
        outputs = (label_logits, evidence) + outputs[1:]

        # 학습 시
        if answer is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # logg_fct 선언

            loss_fct = CrossEntropyLoss(reduction="none")

            # loss 계산
            label_losses = []

            for e in range(self.num_samples):
                label_lose = loss_fct(label_logits[:, :, e], answer)

                label_losses.append(label_lose)

            label_losses = torch.stack(label_losses, 0)

            # 최종 loss 계산
            span_loss = label_losses
            # outputs : (total_loss, start_logits, end_logits)
            outputs = (span_loss, attention_scores, mm) + outputs

        return outputs  # (loss), start_logits, end_logits


# class ElectraForQuestionAnsweringBeamSearch(ElectraPreTrainedModel):
#     def __init__(self, config):
#         super(ElectraForQuestionAnsweringBeamSearch, self).__init__(config)
#         # 분류 해야할 라벨 개수 (start/end)
#         self.num_labels = config.num_labels
#         self.hidden_size = config.hidden_size

#         # ELECTRA 모델 선언
#         self.max_seq_length = config.max_position_embeddings
#         self.electra = ElectraModel(config)
#         self.max_sent_num = config.max_sent_num
#         self.num_samples = config.num_samples

#         # 추가한 부분
#         self.classifier = nn.Linear(in_features=config.hidden_size, out_features=config.num_label)

#         # 삭제해야할 부분
#         self.start_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
#         self.end_dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
#         self.max_dec_len = config.max_dec_len
#         self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.gru = BeamSearchAttentionDecoder(self.hidden_size, self.max_sent_num, self.num_samples)
#         self.gru2 = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True, num_layers=1)
#         # ELECTRA weight 초기화
#         self.init_weights()

#     def forward(
#         self,
#         input_ids=None,
#         ##################
#         attention_mask=None,
#         token_type_ids=None,
#         sent_masks=None,
#         ##############
#         answer=None,
#         head=None,
#         tail=None,
#     ):

#         # ELECTRA output 저장
#         # outputs : [1, batch_size, seq_length, hidden_size]
#         # electra 선언 부분에 특정 옵션을 부여할 경우 출력은 다음과 같음
#         # outputs : (last-layer hidden state, all hidden states, all attentions)
#         # last-layer hidden state : [batch, seq_length, hidden_size]
#         # all hidden states : [13, batch, seq_length, hidden_size]
#         # 12가 아닌 13인 이유?? ==> 토큰의 임베딩도 output에 포함되어 return
#         # all attentions : [12, batch, num_heads, seq_length, seq_length]
#         batch_size = input_ids.size(0)
#         sequence_length = input_ids.size(1)
#         outputs = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#         )
#         is_training = False
#         if answer is not None:
#             is_training = True

#         # sequence_output : [batch_size, seq_length, hidden_size]
#         sequence_output = outputs[0]
#         # cls_output : [batch, hidden]
#         cls_outputs = sequence_output[:, 0, :]

#         # linear_classifier : [batch, path, hidden* 2] -> [batch, path, 97]
#         # label_logits : [batch, 2, hidden] * [batch, hidden, num_sent] = [batch, 2, num_sent]
#         label_logits = self.classifier(cls_outputs)
#         # label_logits = linear_classfier.matmul(evidence_vector)

#         # outputs = (start_logits, end_logits)
#         # outputs = (label_logits)
#         outputs = (label_logits,) + outputs[1:]

#         # 학습 시
#         if answer is not None:
#             # sometimes the start/end positions are outside our model inputs, we ignore these terms
#             # logg_fct 선언

#             loss_fct = CrossEntropyLoss()

#             # loss 계산
#             label_lose = loss_fct(label_logits, answer)

#             # outputs : (total_loss, start_logits, end_logits)
#             outputs = (label_lose,) + outputs

#         return outputs  # (loss), start_logits, end_logits
