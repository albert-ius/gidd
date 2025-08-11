import json

import numpy as np
import hydra
import tqdm
import torch
from transformers import AutoModelForCausalLM

from gidd.data import get_dataloaders
from gidd.utils import parse_dtype
from gidd.loss import get_loss
from gidd.checkpoints import load_checkpoint
from gidd.trainer import get_trainer
from gidd import GiddPipeline
import argparse
import sys


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-m", "--model_name", type=str, default="dvruette/gidd-base-p_unif-0.0",
                        help="Model name from hf")
    parser.add_argument("-o", "--output", type=str, default="metrics.json",
                        help="Path to output file")
    parser.add_argument("-b", "--batch", type=int, default=32,
                        help="Batch size")
    
    args = parser.parse_args(argv)

    pipe = GiddPipeline.from_pretrained(args.model_name, trust_remote_code=True)
    pipe.to(device)

    model, noise_schedule, tokenizer, config = pipe.model, pipe.noise_schedule, pipe.tokenizer, pipe.config
    model.eval()
    print(config)
    config.training.eval_batch_size = args.batch
    dtype = parse_dtype(config.training.dtype)

    loss_fn = get_loss(config, tokenizer, noise_schedule)
    _, test_dl = get_dataloaders(config, tokenizer)

    trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    trainer.to(device)
    trainer = torch.compile(trainer)
    model.eval()

    eval_metrics = {}
    with torch.no_grad():
        eval_loss = 0
        num_eval_samples = 0
        for test_batch in tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True):
            bs = test_batch["input_ids"].size(0)

            test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
            loss, metrics = trainer(test_batch)

            for k, v in metrics.items():
                eval_metrics[k] = eval_metrics.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v) * bs

            eval_loss += loss.item() * bs
            num_eval_samples += bs

    eval_metrics = {
        "loss": eval_loss / num_eval_samples,
        **{k: v / num_eval_samples for k, v in eval_metrics.items()},
    }
    eval_metrics["ppl"] = np.exp(eval_metrics["elbo"])

    eval_metrics["path"] = args.hf_model

    print(json.dumps(eval_metrics, indent=2))

    with open(args.output, "a") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
