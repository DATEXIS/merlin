import ast
import asyncio
import logging
from argparse import ArgumentParser
from pathlib import Path

from src.pipeline.pipeline import start_pipeline
from src.pipeline.verifier_args import VerifierArgs
from src.exp_args import ExpArgs
from src.vllm_client import get_api_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data', help="path to pvc storage")
    parser.add_argument("--file_name", type=str, default='patients_3055')
    parser.add_argument("--server_name", type=str, default='vllm-server')
    parser.add_argument("--namespace", type=str, default='anon')
    parser.add_argument("--chief_complaint", type=str, default='abdominal_pain')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--num_choices", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--budget", type=str)
    parser.add_argument("--start_verifier", type=int, default=1)
    parser.add_argument("--temperatures", type=str, help="")
    parser.add_argument("--max_tokens", type=str, help="")
    parser.add_argument("--thresholds", type=str, help="")
    parser.add_argument("--load_from_checkpoint", type=str, default='False')
    parser.add_argument("--eval_mode", type=str, default='False')
    parser.add_argument("--think_about_labs", type=str, default='True')
    parser.add_argument("--guided_decoding", type=str, default='False')
    parser.add_argument("--merlin_mode", type=str, default='True')
    parser.add_argument("--lora", type=str, default='False')
    parser.add_argument("--lora_name", type=str)
    parser.add_argument("--split", type=str, default='',
                        help="Comma-separated splits to eval, e.g. 'dev,val'. Empty = no filter.")
    parser.add_argument("--config_string", type=str)
    args = parser.parse_args()

    splits = [s for s in args.split.split(',') if s] or None

    exp_args = ExpArgs(
        data_dir=Path(args.data_dir),
        file_name=args.file_name,
        api_config=get_api_config(args.server_name, args.namespace),
        eval_mode=args.eval_mode.lower() == 'true',
        merlin_mode=args.merlin_mode.lower() == 'true',
        think_about_labs=args.think_about_labs.lower() == 'true',
        guided_decoding=args.guided_decoding.lower() == 'true',
        load_from_checkpoint=args.load_from_checkpoint.lower() == 'true',
        lora=args.lora.lower() == 'true',
        lora_name=args.lora_name,
        config_str=ast.literal_eval(args.config_string),
        seed=args.seed,
        num_samples=args.num_samples,
        splits=splits,
    )

    v_args = VerifierArgs(
        exp_args=exp_args,
        chief_complaint=args.chief_complaint,
        start_verifier=args.start_verifier,
        budgets=ast.literal_eval(args.budget),
        temperatures=ast.literal_eval(args.temperatures),
        max_tokens=ast.literal_eval(args.max_tokens),
        verifier_thresholds=ast.literal_eval(args.thresholds),
        num_choices=args.num_choices,
        concurrency=args.concurrency,
    )

    asyncio.run(start_pipeline(exp_args, v_args))
