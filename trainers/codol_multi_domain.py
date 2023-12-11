import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerMULTI3
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint, load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

from tqdm import tqdm


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoDoL',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)       
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CoDoL.N_CTX
        n_dmx = cfg.TRAINER.CoDoL.N_DMX
        n = n_ctx + n_dmx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        # for domain
        if cfg.EVAL_ONLY:
            domainnames = cfg.DATASET.TARGET_DOMAINS
        else:
            domainnames = cfg.DATASET.SOURCE_DOMAINS
        n_dm = len(domainnames)
        
        print("Initializing a generic context")
        dmx_vectors = torch.empty(n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(dmx_vectors, std=0.02)
  
        self.dmx = nn.Parameter(dmx_vectors)

        # for class
        print("Initializing a generic context")
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n)
        
        self.ctx = nn.Parameter(ctx_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of context words (tokens): {n_dmx}") 
        
        # for class meta_net 
        self.dmx_meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        if cfg.TRAINER.CoDoL.PREC == "fp16":
            self.dmx_meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        domainnames = [name.replace("_", " ") for name in domainnames]
        domainnames = [", a type of {}".format(domain) for domain in domainnames]
        
        if cfg.TRAINER.CoDoL.PROMPT_TYPE == "each":
            prompts = [prompt_prefix + " " + name + " " + domain + "." for domain in domainnames for name in classnames]
        elif cfg.TRAINER.CoDoL.PROMPT_TYPE == "mean":
            prompts = [prompt_prefix + " " + name + " " + domain + "." for name  in classnames for domain in domainnames]
        
        print(f"prompts.{prompts}")

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS
        
        self.n = n
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        ctx_dim = ctx.size(-1)
        dmx = self.dmx  # (n_dmx, ctx_dim)
        
        bias = self.dmx_meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        # for dmx
        dmx = dmx.unsqueeze(0) # (1, n_dmx, ctx_dim)
        dmx_shifted = dmx + bias # (batch, n_dmx, ctx_dim)
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for dmx_shifted_i in dmx_shifted:
            ctx_i = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_i = ctx_i.unsqueeze(0).expand(self.n_dm, -1, -1, -1)
 
            dmx_i = dmx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            dmx_i = dmx_i.unsqueeze(0).expand(self.n_dm, -1, -1, -1)
            
            ctxdmx_i = torch.cat([ctx_i, dmx_i], dim=2).reshape(self.n_cls*self.n_dm, self.n, ctx_dim)
            pts_i = self.construct_prompts(ctxdmx_i, prefix, suffix)  
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
    
        self.cfg = cfg
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        prompts = self.prompt_learner(image_features) 
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        return logits

@TRAINER_REGISTRY.register()
class CoDoL_multi_domain(TrainerMULTI3):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CoDoL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CoDoL.PREC == "fp32" or cfg.TRAINER.CoDoL.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)


        self.scaler = GradScaler() if cfg.TRAINER.CoDoL.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
       
    def forward_backward(self, batch0, batch1, batch2):
        image0, label0, image1, label1, image2, label2  = self.parse_batch_train(batch0, batch1, batch2)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CoDoL.PREC
        if prec == "amp":
            with autocast():
                loss0 = model(image0, label0)
                loss1 = model(image1, label1)
                loss2 = model(image2, label2)
            loss = torch.mean(torch.stack([loss0, loss1, loss2]))
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
            loss_summary = {
            "loss": loss.item(),
            }
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()
        else:
            output0 = model(image0)
            output1 = model(image1)
            output2 = model(image2)
            
            loss0 = F.cross_entropy(output0[:, :self.model.prompt_learner.n_cls], label0)
            acc0 = compute_accuracy(output0[:, :self.model.prompt_learner.n_cls], label0)[0]
            
            loss1 = F.cross_entropy(output1[:, self.model.prompt_learner.n_cls: 2*self.model.prompt_learner.n_cls], label1)
            acc1 = compute_accuracy(output1[:, self.model.prompt_learner.n_cls: 2*self.model.prompt_learner.n_cls], label1)[0]
            
            loss2 = F.cross_entropy(output2[:, -self.model.prompt_learner.n_cls:], label2) 
            acc2 = compute_accuracy(output2[:, -self.model.prompt_learner.n_cls:], label2)[0] 
            
            loss = loss0 + loss1 + loss2
            acc = torch.mean(torch.stack([acc0, acc1, acc2]))

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_summary = {
                "loss": loss.item(),
                "acc": acc.item(),
            }

            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()     
            
        return loss_summary

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            print(f"Test model using model.pth.tar-{epoch}")
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
        
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
  
        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def parse_bath_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
