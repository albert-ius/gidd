import argparse
import torch
import tqdm
import sys
from transformers import AutoTokenizer
from pathlib import Path

def load_text_from_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    return "".join(lines)

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--generations_dir", type=str,
                        help="Path to dir with generated samples")
    parser.add_argument("-t", "--tokenizer", type=str,
                        help="Tokenizer")
    parser.add_argument("-o", "--output", type=str, default="samples.pt",
                        help="Path to output file")
    
    args = parser.parse_args(argv)

    print(f'Tokenizer:{args.tokenizer}')
    print(f'Generations dir:{args.generations_dir}')
    print(f'Output dir:{args.output}')

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    pathlist = list(Path(args.generations_dir).rglob('*.txt'))

    output_batch = []

    print('tokenizing')

    with tqdm.tqdm(total=len(pathlist)) as pbar:
        for path in pathlist:
            path_in_str = str(path)
            sample = load_text_from_file(path_in_str)
            output_batch.append(tokenizer.encode(sample, max_length=2761, padding='max_length', truncation=True, return_tensors='pt'))
            pbar.update(1)


    output_tensor = torch.vstack(output_batch)

    print('saving')
    torch.save(output_tensor, args.output)


if __name__ == "__main__":
    main(sys.argv[1:])