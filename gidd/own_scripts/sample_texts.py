import hydra
import tqdm
import json

from gidd.modeling import get_tokenizer
from gidd.data import get_shuffled_test_dataloader


@hydra.main(config_path="../configs", config_name="generate_condition_texts", version_base="1.1")
def main(args):
    tokenizer = get_tokenizer(args)

    test_dl = get_shuffled_test_dataloader(args, tokenizer, test_batch_size=args.batch_size, random_seed=args.random_seed)
    batch_iterator = iter(test_dl)
    num_samples = args.num_samples
    batch_size = args.batch_size
    samples_sampled = 0
    all_texts = []
    with tqdm.tqdm(total=num_samples) as pbar:
        while samples_sampled < num_samples:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(test_dl)
                batch = next(batch_iterator)
            batch_texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            all_texts.extend(batch_texts[:min(num_samples - samples_sampled, batch_size)])
            batch_size = len(batch['input_ids'])
            samples_sampled += batch_size
            pbar.update(batch_size)

    enumerated_texts = {ind: text for ind, text in enumerate(all_texts)}
    with open(args.samples_path, 'w') as f:
        json.dump(enumerated_texts, f, indent=4)
        

if __name__ == "__main__":
    main()