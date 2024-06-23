import torch


class WarmUpScheduler:

    def __init__(self, optimizer, embed_dim, n_warmup, n_training_steps):
        self._optimizer = optimizer
        self.embed_dim = embed_dim
        self.n_warmup = n_warmup
        self.n_training_steps = n_training_steps
        self.step = 0

    def step(self):
        self._update_lr()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _precalculate_lr(self):

        self._lr_val = torch.tensor(
            [
                ((self.d_model**-0.05) * min(i**-0.5, i * self.n_warmup ** (-1.5)))
                for i in range(self.n_training_steps)
            ],
            device=self._optimizer.device,
        )

    def _update_lr(self):
        self.step += 1
        new_lr = self._lr_val[self.step].item()
        self.curr_lr = new_lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_curr_lr(self):
        return self.curr_lr
