import torch
import hydra

from gidd.checkpoints import load_checkpoint


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")


if __name__ == "__main__":
    main()
