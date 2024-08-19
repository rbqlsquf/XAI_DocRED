import argparse
import os
import logging
from attrdict import AttrDict
import torch
from transformers import ElectraTokenizer, ElectraConfig
from transformers import AutoTokenizer, AutoModelForPreTraining

from src.model.model_rnn import ElectraForQuestionAnsweringBeamSearch
from src.model.main_function_BeamSearch import train, evaluate
from src.functions.utils import init_logger, set_seed


def create_model(args):
    # 모델 파라미터 Load
    config = ElectraConfig.from_pretrained(
        (
            args.model_name_or_path
            if args.from_init_weight
            else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint))
        ),
        # os.path.join("./first", "checkpoint-{}".format(args.checkpoint)) if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
    )

    # tokenizer는 pre-trained된 것을 불러오는 과정이 아닌 불러오는 모델의 vocab 등을 Load
    tokenizer = ElectraTokenizer.from_pretrained(
        (
            args.model_name_or_path
            if args.from_init_weight
            else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint))
        ),
        # os.path.join("./first", "checkpoint-{}".format(args.checkpoint)) if args.from_init_weight else os.path.join( args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        do_lower_case=args.do_lower_case,
    )
    config.max_sent_num = args.max_sent_num
    config.max_dec_len = args.max_dec_len
    config.num_samples = args.num_samples
    config.num_label = args.num_label
    model = ElectraForQuestionAnsweringBeamSearch.from_pretrained(
        # os.path.join("./first", "checkpoint-{}".format(args.checkpoint)) if args.from_init_weight else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint)),
        (
            args.model_name_or_path
            if args.from_init_weight
            else os.path.join(args.output_dir, "checkpoint-{}".format(args.checkpoint))
        ),
        config=config,
        # from_tf= True if args.from_init_weight else False
    )

    # vocab 추가
    # 중요 단어의 UNK 방지 및 tokenize를 방지해야하는 경우(HTML 태그 등)에 활용
    # "세종대왕"이 OOV인 경우 ['세종대왕'] --tokenize-->  ['UNK'] (X)
    # html tag인 [td]는 tokenize가 되지 않아야 함. (완전한 tag의 형태를 갖췄을 때, 의미를 갖기 때문)
    #                             ['[td]'] --tokenize-->  ['[', 't', 'd', ']'] (X)

    if args.from_init_weight and args.add_vocab:
        if args.from_init_weight:
            add_token = {"additional_special_tokens": ["[td]", "추가 단어 1", "추가 단어 2"]}
            # 추가된 단어는 tokenize 되지 않음
            # ex
            # '[td]' vocab 추가 전 -> ['[', 't', 'd', ']']
            # '[td]' vocab 추가 후 -> ['[td]']
            tokenizer.add_special_tokens(add_token)
            model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    return model, tokenizer


def main(cli_args):
    # 파라미터 업데이트
    args = AttrDict(vars(cli_args))
    args.device = "cuda"
    logger = logging.getLogger(__name__)

    # logger 및 seed 지정
    init_logger()
    set_seed(args)

    # 모델 불러오기
    model, tokenizer = create_model(args)
    # def get_n_params(model):
    #     pp = 0
    #     for p in list(model.parameters()):
    #         nn = 1
    #         for s in list(p.size()):
    #             nn = nn * s
    #         pp += nn
    #     return pp
    #
    # print(get_n_params(model))
    # exit(1)

    # Running mode에 따른 실행
    if args.do_train:
        train(args, model, tokenizer, logger)
    elif args.do_eval:
        model, tokenizer = create_model(args)
        evaluate(args, model, tokenizer, logger, global_step=args.checkpoint)


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    # Directory
    cli_parser.add_argument("--data_dir", type=str, default="./data")
    # cli_parser.add_argument("--data_dir", type=str, default="./all_data")
    cli_parser.add_argument("--model_name_or_path", type=str, default="google/electra-base-discriminator")

    cli_parser.add_argument("--output_dir", type=str, default="./beamserach32")
    # cli_parser.add_argument("--output_dir", type=str, default="./uppperbound")
    # cli_parser.add_argument("--output_dir", type=str, default="./baseline0601")

    cli_parser.add_argument("--train_file", type=str, default="train_data.json")
    cli_parser.add_argument("--predict_file", type=str, default="dev_data.json")
    # cli_parser.add_argument("--predict_file", type=str, default="refine_hotpot_dev_distractor_v1.json")
    # cli_parser.add_argument("--predict_file", type=str, default="refine_hotpot_dev_fullwiki_v1.json")
    cli_parser.add_argument("--checkpoint", type=str, default="3000")

    # Model Hyper Parameter
    cli_parser.add_argument("--max_seq_length", type=int, default=512)
    cli_parser.add_argument("--doc_stride", type=int, default=128)
    cli_parser.add_argument("--max_query_length", type=int, default=64)
    cli_parser.add_argument("--max_answer_length", type=int, default=30)
    cli_parser.add_argument("--n_best_size", type=int, default=20)

    # 한규빈이 막내인 이유가 뭡니까!
    # 연구실 사람들은 다 00년 이전에 태어났는데 규빈이는 01입니다.
    # 규빈이는 01입니다.
    # 아 ㅋㅋ 그건말이죠~ 연구실 사람들은 다 00년 이전에 태어났는데 규빈이는 01입니다. 가릿?
    # Training Parameter
    cli_parser.add_argument("--learning_rate", type=float, default=5e-5)
    cli_parser.add_argument("--train_batch_size", type=int, default=8)
    cli_parser.add_argument("--eval_batch_size", type=int, default=8)
    cli_parser.add_argument("--max_sent_num", type=int, default=40)
    ####추가한 부분
    #############!!!
    ### num samples : beamsearch 수 , max_dec_len : 근거 문장 몇개 뽑을지
    cli_parser.add_argument("--num_samples", type=int, default=3)
    cli_parser.add_argument("--max_dec_len", type=int, default=2)
    cli_parser.add_argument("--num_label", type=int, default=97)  # 라벨수
    cli_parser.add_argument("--num_train_epochs", type=int, default=5)

    cli_parser.add_argument("--save_steps", type=int, default=1000)
    cli_parser.add_argument("--logging_steps", type=int, default=1000)
    cli_parser.add_argument("--seed", type=int, default=42)
    cli_parser.add_argument("--threads", type=int, default=8)

    cli_parser.add_argument("--weight_decay", type=float, default=0.0)
    cli_parser.add_argument("--adam_epsilon", type=int, default=1e-10)

    cli_parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    cli_parser.add_argument("--warmup_steps", type=int, default=0)
    cli_parser.add_argument("--max_steps", type=int, default=-1)
    cli_parser.add_argument("--max_grad_norm", type=int, default=1.0)

    cli_parser.add_argument("--verbose_logging", type=bool, default=False)
    cli_parser.add_argument("--do_lower_case", type=bool, default=True)
    cli_parser.add_argument("--no_cuda", type=bool, default=False)

    # For SQuAD v2.0 (Yes/No Question)
    cli_parser.add_argument("--version_2_with_negative", type=bool, default=False)
    cli_parser.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    # Running Mode
    cli_parser.add_argument("--from_init_weight", type=bool, default=True)
    cli_parser.add_argument("--add_vocab", type=bool, default=False)
    cli_parser.add_argument("--do_train", type=bool, default=True)
    cli_parser.add_argument("--do_eval", type=bool, default=True)
    cli_parser.add_argument("--do_predict", type=bool, default=False)
    cli_args = cli_parser.parse_args()

    main(cli_args)
