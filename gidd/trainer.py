import torch
import torch.nn as nn
import torch.distributed as dist
import embeddings

from gidd.diffusion_process import sample_t, NoiseSchedule
from gidd.loss import Loss
import torch._dynamo
torch._dynamo.config.suppress_errors = True


class DiffusionTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.condition_on_text_embeds = False

        if self.config.cond_embeddings.use_text_embedder:
            # Note: TextEmbedder is not an nn.Module to keep its parameters frozen
            # during training. Device movement is handled
            # manually in training hooks.
            self.condition_on_text_embeds = True
            self.text_embedder = embeddings.TextEmbedder(
                model_name=self.config.cond_embeddings.model_name, device=self.device)
            self.text_condition_dim = self.config.cond_embeddings.text_condition_dim

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        if device is not None:
            self._move_text_embedder_to_device(device)
        return super().to(device, dtype)

    def forward(self, batch):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            t = sample_t(self.config, batch_size, device=self.device)
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)

            if self.condition_on_text_embeds:
                with torch.no_grad():
                    cond_text = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    cond = self.text_embedder(cond_text)
                    if isinstance(cond, torch.Tensor):
                        cond = cond.to(self.device)
                logits = self.model(z_t, t, cond=cond)
            else:
                logits = self.model(z_t, t)

            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                z_t=z_t,
                t=t,
                reduction=self.config.loss.reduction,
            )
        return loss, metrics
    
    def _move_text_embedder_to_device(self, device):
        """Move text embedder to the current device."""
        if self.condition_on_text_embeds and self.text_embedder is not None:
            self.text_embedder = self.text_embedder.to(device)
            #self.text_embedder.tokenizer = self.text_embedder.tokenizer.to(device)


class AutoregressiveTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, loss_fn, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.device = next(model.parameters()).device
    
    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            labels = batch["input_ids"][:, 1:]
            loss_mask = batch["attention_mask"][:, :-1]

            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            logits = logits[:, :-1]
            loss = self.loss_fn(logits.transpose(1, 2), labels)
            total_loss = (loss * loss_mask).sum()
            total_tokens = loss_mask.sum().float()

            if self.world_size > 1:
                dist.all_reduce(total_tokens)
                total_tokens /= self.world_size

            loss = total_loss / total_tokens

        return loss, {
            "elbo": loss.detach(),
            "nll": loss.detach(),
            "ppl": loss.detach().exp(),
        }


def get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype=None):
    if config.model.type == "diffusion":
        return DiffusionTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    elif config.model.type == "autoregressive":
        return AutoregressiveTrainer(config, model, tokenizer, loss_fn, dtype)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
