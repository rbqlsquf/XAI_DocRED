import json
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count

# logging.basicConfig(level=logging.ERROR)
import numpy as np
from tqdm import tqdm
import nltk
from transformers.file_utils import is_tf_available, is_torch_available

# from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.data.processors.utils import DataProcessor
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset
max_sent_num = 0
if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    tok_to_sent_index = []
    token_index = 0
    for i, sentence in enumerate(example.sentences):
        # doc_tokens ?? ==> 어절 단위의 문서(context)
        # token ==> 어절
        # i = 어절 index
        for token in sentence:
            orig_to_tok_index.append(len(all_doc_tokens))
            # ??에 길이를 저장
            token_index += 1
            sub_tokens = tokenizer.tokenize(token)
            # sub tokens?? => 어절을 wordpiece
            for sub_token in sub_tokens:
                tok_to_orig_index.append(token_index)
                all_doc_tokens.append(sub_token)
                tok_to_sent_index.append(i + 1)  # 토큰들에 해당하는 문장번호 넣기

    ##########head랑 tail의 시작 인덱스 값을 찾아나서는 과정...

    spans = []
    head = example.na_triple[0]
    tail = example.na_triple[1]
    query = head + "[SEP]" + tail
    truncated_query = tokenizer.encode(query, add_special_tokens=False, max_length=max_query_length)

    #############!!!그냥 1로 지정..
    # sequence_added_tokens = 1
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )

    #####그냥 2로 지정
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    # sequence_pair_added_tokens = 2

    # []
    span_doc_tokens = all_doc_tokens

    if tokenizer.padding_side == "right":
        texts = truncated_query
        pairs = span_doc_tokens
        truncation = TruncationStrategy.ONLY_SECOND.value
    else:
        texts = span_doc_tokens
        pairs = truncated_query
        truncation = TruncationStrategy.ONLY_FIRST.value

    encoded_dict = tokenizer.encode_plus(
        texts,
        pairs,
        truncation=truncation,
        padding=padding_strategy,
        max_length=max_seq_length,
        return_overflowing_tokens=True,
        # stride=max_seq_length - doc_stride - sequence_pair_added_tokens,d
        return_token_type_ids=True,
    )

    paragraph_len = min(
        len(all_doc_tokens) - len(spans),
        max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
    )

    if tokenizer.pad_token_id in encoded_dict["input_ids"]:
        if tokenizer.padding_side == "right":
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            last_padding_id_position = (
                len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
            )
            non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

    else:
        non_padded_ids = encoded_dict["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

    token_to_orig_map = {}

    cur_sent_to_orig_sent_map = {}
    sent_mask = [0] * (len(truncated_query) + (sequence_added_tokens))
    sent_offset = tok_to_sent_index[0]

    cur_sent_range = [[] for _ in range(40)]
    cur_sent_range[0] = [e for e in range(len(sent_mask))]
    for i in range(paragraph_len):
        cur_sent_num = tok_to_sent_index[i] - sent_offset + 1
        orig_sent_num = tok_to_sent_index[i]

        index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
        token_to_orig_map[index] = tok_to_orig_index[i]

        sent_mask.append(cur_sent_num)
        if cur_sent_num >= 40:
            print("hi")
        cur_sent_range[cur_sent_num].append(len(sent_mask) - 1)
        cur_sent_to_orig_sent_map[cur_sent_num] = orig_sent_num

    encoded_dict["paragraph_len"] = paragraph_len
    encoded_dict["tokens"] = tokens
    encoded_dict["token_to_orig_map"] = token_to_orig_map
    encoded_dict["truncated_query_with_special_tokens_length"] = len(query) + sequence_added_tokens
    encoded_dict["token_is_max_context"] = {}
    encoded_dict["start"] = 0
    encoded_dict["length"] = paragraph_len

    encoded_dict["sent_mask"] = sent_mask + [0] * (max_seq_length - len(sent_mask))
    encoded_dict["cur_sent_to_orig_sent"] = cur_sent_to_orig_sent_map
    encoded_dict["cur_sent_range"] = cur_sent_range
    encoded_dict["truncated_query"] = truncated_query

    spans.append(encoded_dict)

    ######################################################################################################

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        features.append(
            SquadFeatures(
                span["input_ids"],
                None,  # span["attention_mask"],
                span["token_type_ids"],
                span["cur_sent_range"],
                cls_index,
                example_index=0,
                # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                label=example.label,
                sent_mask=span["sent_mask"],
                cur_sent_to_orig_sent=span["cur_sent_to_orig_sent"],
                truncated_query=span["truncated_query"],
                evidence_sentence=example.evidence,
                data_id=example.data_id,
            )
        )
    return features


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset
        threads: multiple processing threadsa-smi


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    features = []
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )
    new_features = []
    unique_id = 1000000000

    example_index = 0
    for example_features in features:

        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1

    features = new_features
    del new_features
    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).cuda()
        all_attention_masks = all_input_ids.masked_fill(all_input_ids != 0, 1).cuda()
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).cuda()
        all_sent_masks = torch.tensor([f.sent_mask for f in features], dtype=torch.long).cuda()
        all_answer = torch.tensor([f.label for f in features], dtype=torch.long).cuda()
        all_unique_id = torch.tensor([f.unique_id for f in features], dtype=torch.long).cuda()

        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_sent_masks,
            all_answer,
            all_unique_id,
        )

        return features, dataset


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None, tokenizer=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "train", tokenizer)

    def get_dev_examples(self, data_dir, filename=None, tokenizer=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)
        return self._create_examples(input_data, "dev", tokenizer)

    def example_from_input(self, question, context):
        return SquadExample(
            qas_id="sample",
            question_text=question,
            context_text=context,
            answer_text=None,
            start_position_character=None,
            title="sample",
            is_impossible=False,
            answers=[],
        )

    def get_example_from_input(self, input_dictionary):
        # context, question, id, title
        context_text = input_dictionary["context"]
        question_text = input_dictionary["question"]
        qas_id = input_dictionary["id"]
        start_position_character = None
        is_impossible = False
        answer_text = None
        answers = []

        examples = [
            SquadExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title="",
                is_impossible=is_impossible,
                answers=answers,
            )
        ]
        return examples

    def _create_examples(self, input_data, set_type, tokenizer):
        is_training = set_type == "train"
        num = 0
        examples = []
        for entry in tqdm(input_data):
            # qas_id = entry["_id"]
            # question_text = entry["question"]
            # level = entry["level"]
            # question_type = entry["type"]
            # if 'question_type' in entry.keys():
            #     q_type = entry['question_type']
            # else:
            #     q_type = None
            data_examples = []
            # support_facts = {e[0]:e[1] for e in entry["supporting_facts"]}
            na_triple = entry["na_triple"]
            head = entry["head"]
            tail = entry["tail"]
            relation = entry["relation"]
            label = entry["labels"]
            title = entry["title"]
            evidence = entry["evidence"]
            sentence_start = entry["sentence_start"]
            sentences = entry["sentences"]
            data_id = entry["data_id"]

            example = SquadExample(
                na_triple=na_triple,
                head=head,
                tail=tail,
                relation=relation,
                label=label,
                title=title,
                evidence=evidence,
                sentence_start=sentence_start,
                sentences=sentences,
                data_id=data_id,
            )

            examples.append(example)
            if len(examples) >= 100:
                break

        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(self, na_triple, head, tail, relation, label, title, evidence, sentence_start, sentences, data_id):
        # 데이터셋 꼭 까서 보고 example을 읽어주길바람,,
        self.na_triple = na_triple
        self.head = head
        self.tail = tail
        self.relation = relation
        self.label = label
        self.title = title
        self.evidence = evidence
        self.sentence_start = sentence_start
        self.sentences = sentences
        self.data_id = data_id


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index
        end_position: end of the answer token index
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cur_sent_range,
        cls_index,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        label,
        sent_mask,
        cur_sent_to_orig_sent,
        truncated_query,
        evidence_sentence,
        data_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cur_sent_range = cur_sent_range
        self.cls_index = cls_index

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.label = label
        self.sent_mask = sent_mask
        self.cur_sent_to_orig_sent = cur_sent_to_orig_sent
        self.truncated_query = truncated_query

        self.data_id = data_id
        self.evidence_sentence = evidence_sentence


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, label_logits, evidence=None, pred_label=None):
        self.label_logits = label_logits
        self.unique_id = unique_id
        self.evidence = evidence
        self.pred_labels = pred_label
