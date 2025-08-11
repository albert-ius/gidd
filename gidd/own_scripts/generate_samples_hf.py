from gidd import GiddPipeline
import argparse
import torch
import tqdm
import sys


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-m", "--model_name", type=str, default="dvruette/gidd-base-p_unif-0.0",
                        help="Model name from hf")
    parser.add_argument("-n", "--num_samples", type=int, default=32,
                        help="Num samples")
    parser.add_argument("-o", "--output", type=str, default="samples.pt",
                        help="Path to output file")
    parser.add_argument("-d", "--num_denoising_steps", type=int, default=128,
                        help="Num denoising steps")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size")
    
    args = parser.parse_args(argv)

    pipe = GiddPipeline.from_pretrained(args.model_name, trust_remote_code=True)
    pipe.to(device)

    samples = []
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for i in range(0, args.num_samples, args.batch_size):
                bs = min(args.batch_size, args.num_samples - i)
                z_t = pipe.generate(bs, args.num_denoising_steps, decode=False, show_progress=False)
                samples.append(z_t)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0).cpu()

    torch.save(samples, args.output)

if __name__ == "__main__":
    main(sys.argv[1:])

