import json
from pathlib import Path

import hydra
import numpy as np
import tqdm
import torch
import mauve
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_mauve(generated_texts, reference_texts):
  """Compute MAUVE metric."""
  if mauve is None:
    print("MAUVE not available. Skipping MAUVE computation.")
    return None
  
  result = mauve.compute_mauve(
    p_text=reference_texts,
    q_text=generated_texts,
    device_id=0 if torch.cuda.is_available() else -1,
    verbose=False
  )
  return result.mauve


@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    def get_texts_from_format(texts_path, input_format, model_tokenizer):
        if texts_path is None:
            return None
        texts_path = hydra.utils.to_absolute_path(texts_path)
        if input_format == 'pt':
            z_ts = torch.load(texts_path, weights_only=True)
            # fix for bug in self-correct script:
            if z_ts.shape[1] == 1:
                z_ts = z_ts.squeeze(1)
            texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)
        elif args.input_format == 'json':
            with open(texts_path, 'r') as f:
                texts_enumerated = json.load(f)
                texts = [v for k, v in texts_enumerated.items()]
        return texts
    
    samples_texts = get_texts_from_format(args.samples_path, args.input_format, model_tokenizer)
    validation_texts = get_texts_from_format(args.validation_path, args.input_format, model_tokenizer)
    conditions_texts = get_texts_from_format(args.conditions_path, args.input_format, model_tokenizer)

    metrics = {}

    if 'gen_ppl' in args.metrics: 
        print(f"Loding model {args.pretrained_model}")

        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        if args.torch_compile:
            model = torch.compile(model)

        total_acc = 0
        total_nll = 0
        total_tokens = 0
        all_nlls = []
        with torch.no_grad():
            for i in tqdm.trange(0, len(samples_texts), args.batch_size, desc="Inference", dynamic_ncols=True):
                xs = samples_texts[i:i + args.batch_size]

                batch = tokenizer(xs, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
                attn_mask = batch["attention_mask"]
        
                logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]

                labels = batch["input_ids"][:, 1:]
                loss_mask = attn_mask[:, :-1]

                nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
                all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
                total_nll += (nll * loss_mask).sum().item()

                acc = (logits.argmax(-1) == labels).float()
                total_acc += (acc * loss_mask).sum().item()

                total_tokens += loss_mask.sum().item()


        nll = total_nll / total_tokens
        ppl = np.exp(total_nll / total_tokens)
        acc = total_acc / total_tokens

        metrics_ppl = {
            "file": Path(args.samples_path).stem,
            "pretrained_model": args.pretrained_model,
            "median_nll": np.median(all_nlls),
            "avg_nll": nll,
            "ppl": ppl,
            "acc": acc,
            "tokens": total_tokens,
        }

        metrics['gen_ppl'] = metrics_ppl
    
    if 'mauve' in args.metrics:
        mauve_score = compute_mauve(samples_texts, validation_texts)
        mauve_metrics = {
            "file": Path(args.samples_path).stem,
            "validation_file" : Path(args.validation_path).stem,
            "mauve_score": mauve_score
        }

        metrics['mauve'] = mauve_metrics
    print(json.dumps(metrics, indent=4))
    print("=== RESULTS ===")
    print(",".join(map(str, metrics.values())))
    print("===============")

    with open(hydra.utils.to_absolute_path(args.metrics_path), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
