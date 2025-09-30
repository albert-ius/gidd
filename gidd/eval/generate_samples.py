import hydra
import tqdm
import torch
import json

from gidd.utils import parse_dtype
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler


@hydra.main(config_path="../configs", config_name="generate", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Generating {args.num_samples} samples from {args.path}")

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config, cond_texts_embedder = load_checkpoint(ckpt_path, device=device)

    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    cond_texts = None

    if args.use_emb_cond:
        texts_path = args.cond_texts_path
        with open(texts_path, 'r') as f:
            cond_texts = json.load(f)

    sampler = get_sampler(config, model, tokenizer, noise_schedule, min_p=args.min_p, cond_texts_embedder=cond_texts_embedder, compile_step=False)

    model.eval()

    samples = []

    decode_samples = args.output_format != 'pt'

    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, args.num_samples, args.batch_size):
                bs = min(args.batch_size, args.num_samples - i)
                if args.use_emb_cond:
                    z_t = sampler.generate(bs, args.num_denoising_steps, decode=decode_samples, show_progress=False, cond_texts=cond_texts[i:i + bs])
                else:
                    z_t = sampler.generate(bs, args.num_denoising_steps, decode=decode_samples, show_progress=False, cond_texts=None)
                if decode_samples:
                    samples.extend(z_t)
                else:
                    samples.append(z_t)
                pbar.update(bs)

    if args.output_format == 'pt':
        samples = torch.cat(samples, dim=0).cpu()
        torch.save(samples, hydra.utils.to_absolute_path(args.samples_path))
    elif args.output_format == 'json':
        enumerated_samples = {ind: sample for ind, sample in enumerate(samples)}
        print(f'!!!HERE!!!')
        with open(hydra.utils.to_absolute_path(args.samples_path), 'w') as f:
            json.dump(enumerated_samples, f, indent=4)




if __name__ == "__main__":
    main()
