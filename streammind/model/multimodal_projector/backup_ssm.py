import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from torchmetrics.functional import accuracy
from dataclasses import dataclass
from functools import partial
import math
import einops

def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    delta_A = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    
    if is_variable_B:
        delta_B_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    else:
        delta_B_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    
    x = torch.zeros((batch, dim, dstate), device=u.device)
    ys = []
    
    delta_A = delta_A.permute(2, 0, 1, 3)
    delta_B_u = delta_B_u.permute(2, 0, 1, 3)
    
    for i in range(delta_A.shape[0]):
        x = delta_A[i] * x + delta_B_u[i]
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        ys.append(y)
        
    y = torch.stack(ys, dim=2)
    
    if D is not None:
        y = y + u * D[..., None]
    
    return y.to(dtype=dtype_in)

class PureMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, hidden_states, residual=None, inference_params=None):
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
            
        x = residual
        batch, seqlen, dim = x.shape

        xz = self.in_proj(x)
        xz = xz.transpose(1, 2)
        x, z = xz.chunk(2, dim=1)

        x = self.conv1d(x)[:, :, :seqlen]
        x = F.silu(x)

        x_dbl = self.x_proj(x.transpose(1, 2))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt).transpose(1, 2)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)
        
        A = -torch.exp(self.A_log.float())

        y = selective_scan_ref(x, dt, A, B, C, self.D.float(), delta_bias=None, delta_softplus=True)
        
        y = y * F.silu(z)
        out = self.out_proj(y.transpose(1, 2))

        return out, residual

try:
    from mamba_ssm.models.mixer_seq_simple import create_block as create_block_cuda
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False
    _init_weights = None

@dataclass
class SSMConfig:
    d_code = 1024
    d_model = 2048
    n_ssm = 1
    n_classes = 400
    lr = 1.4e-4
    lr_min = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.02
    scheduler = "plateau"

class VideoMamba(L.LightningModule):
    @staticmethod
    def get_default_config():
        return SSMConfig()

    def __init__(self, config: SSMConfig, omit_in_proj: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        use_cuda_kernel = HAS_MAMBA_CUDA and torch.cuda.is_available()
        
        if use_cuda_kernel:
            self.ssms = nn.ModuleList(
                [create_block_cuda(config.d_model, d_intermediate=0, layer_idx=i) for i in range(config.n_ssm)]
            )
        else:
            self.ssms = nn.ModuleList(
                [PureMambaBlock(config.d_model) for i in range(config.n_ssm)]
            )
        
        self.norm_fn = nn.LayerNorm(config.d_model)
        self.loss_fn = nn.CrossEntropyLoss()
        
        if _init_weights is not None:
            self.apply(partial(_init_weights, n_layer=config.n_ssm))

    def load_checkpoint(self, ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            "loaded checkpoint {}, missing: {}, unexpected: {}".format(
                ckpt_path, missing, unexpected
            )
        )

    def forward(self, embeds, inference_params=None):
        hidden_states = embeds
        residual = None

        for ssm in self.ssms:
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=inference_params
            )
            
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))
        
        logits = hidden_states
        return logits

    def on_after_backward(self):
        param_norm_dict = {}
        grad_norm_dict = {}
        for pn, p in self.named_parameters():
            param_norm_dict["train_param/" + pn] = p.norm()
            if p.grad is not None:
                grad_norm_dict["train_grad/" + pn] = p.grad.norm()
        self.log_dict(
            param_norm_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True
        )
        self.log_dict(
            grad_norm_dict, logger=True, on_step=True, on_epoch=True, sync_dist=True
        )

    def training_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict(
            {"train_loss": loss, "train_acc": acc, "train_acc5": acc5},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict(
            {"val_loss": loss, "val_acc": acc, "val_acc5": acc5}, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        embeds, labels = batch
        outputs = self(embeds)
        loss = self.loss_fn(outputs, labels)
        acc = accuracy(
            outputs, labels, "multiclass", num_classes=self.hparams.config.n_classes
        )
        acc5 = accuracy(
            outputs,
            labels,
            "multiclass",
            num_classes=self.hparams.config.n_classes,
            top_k=5,
        )
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_acc5": acc5})

    def configure_optimizers(self):
        cfg = self.hparams.config
        optimizer = optim.AdamW(
            self.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
        )
        
        scheduler_fn = {
            "plateau": partial(
                optim.lr_scheduler.ReduceLROnPlateau,
                optimizer,
                "min",
                0.1,
                10,
                min_lr=cfg.lr_min,
            ),
            "cosine": partial(
                optim.lr_scheduler.CosineAnnealingWarmRestarts,
                optimizer,
                50,
                1,
                cfg.lr_min,
            ),
        }[cfg.scheduler]
        
        scheduler = scheduler_fn()
        
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": "learning_rate",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor("step"),
        ]
        return callbacks