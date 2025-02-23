import os.path as osp
import json  
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


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

    model = clip.build_model(state_dict or model.state_dict())

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
    def __init__(self, cfg, classnames, clip_model, attributes):
        super().__init__()
        n_cls = len(classnames)
        # Split context tokens between positions if needed
        n_ctx = cfg.TRAINER.AGL.N_CTX
        ctx_init = cfg.TRAINER.AGL.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        # Validate image sizes
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # Store number of attributes per class we'll use
        self.n_attrs = 3  # Fixed to use top-3 attributes per class
        self.attributes = {cls: attrs[:3] for cls, attrs in attributes.items()}

        # Initialize context vectors based on position strategy
        if ctx_init:
            # Initialize from given text (similar to original)
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Initialize based on positioning strategy
            pos_strategy = cfg.TRAINER.AGL.PROMPT_POSITION
            if pos_strategy == "SP-CLS-SP-ATTR":
                # Two sets of soft prompts
                n_ctx_pre = n_ctx // 2
                n_ctx_mid = n_ctx - n_ctx_pre
                ctx_vectors_pre = torch.empty(n_ctx_pre, ctx_dim, dtype=dtype)
                ctx_vectors_mid = torch.empty(n_ctx_mid, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors_pre, std=0.02)
                nn.init.normal_(ctx_vectors_mid, std=0.02)
                self.ctx_pre = nn.Parameter(ctx_vectors_pre)
                self.ctx_mid = nn.Parameter(ctx_vectors_mid)
            else:
                # Single set of soft prompts
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx = nn.Parameter(ctx_vectors)

        print(f"Prompt position strategy: {cfg.TRAINER.AGL.PROMPT_POSITION}")
        print(f"Number of context tokens: {n_ctx}")

        # Process class names and attributes
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = []
        prompts = []
        
        # Create prompts for each class-attribute combination
        for class_idx, name in enumerate(classnames):
            class_attrs = self.attributes[name]
            for attr in class_attrs:
                prompts.append(self.build_prompt_text(name, attr))
                name_lens.append(len(_tokenizer.encode(f"{name}{attr}")))

        # Tokenize prompts
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # Store prefix and suffix
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.prompt_position = cfg.TRAINER.AGL.PROMPT_POSITION

    def build_prompt_text(self, classname, attribute):
        """Helper to maintain consistent prompt structure"""
        # Just concatenate with spaces - actual structure handled in forward
        return f"{classname} {attribute}"

    def forward(self):
        """
        Generates prompts based on the configured position strategy.
        Each class will have 3 different prompts, one for each of its attributes.
        """
        if self.prompt_position == "CLS-SP-ATTR":
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls * self.n_attrs, -1, -1)
            
            prompts = []
            for i in range(self.n_cls * self.n_attrs):
                name_len = self.name_lens[i]
                class_token = self.token_suffix[i:i+1, :name_len, :]
                attr_token = self.token_suffix[i:i+1, name_len:, :]
                
                # Structure: [SOS][Class][Soft Prompts][Attribute][EOS]
                prompt = torch.cat([
                    self.token_prefix[i:i+1, :, :],  # SOS token
                    class_token,                      # Class name
                    ctx[i:i+1, :, :],                # Learnable soft prompts
                    attr_token                        # Attribute token
                ], dim=1)
                prompts.append(prompt)
            
            # Combine all prompts into a single tensor
            return torch.cat(prompts, dim=0)

        elif self.prompt_position == "SP-CLS-SP-ATTR":
            # Expand both context vectors for batch processing
            ctx_pre = self.ctx_pre.unsqueeze(0).expand(self.n_cls * self.n_attrs, -1, -1)
            ctx_mid = self.ctx_mid.unsqueeze(0).expand(self.n_cls * self.n_attrs, -1, -1)
            
            prompts = []
            for i in range(self.n_cls * self.n_attrs):
                name_len = self.name_lens[i]
                class_token = self.token_suffix[i:i+1, :name_len, :]
                attr_token = self.token_suffix[i:i+1, name_len:, :]
                
                # Structure: [SOS][SP1][Class][SP2][Attribute][EOS]
                prompt = torch.cat([
                    self.token_prefix[i:i+1, :, :],  # SOS token
                    ctx_pre[i:i+1, :, :],            # First set of soft prompts
                    class_token,                      # Class name
                    ctx_mid[i:i+1, :, :],            # Second set of soft prompts
                    attr_token                        # Attribute token
                ], dim=1)
                prompts.append(prompt)
            
            return torch.cat(prompts, dim=0)

        elif self.prompt_position == "SP-ATTR-SP-CLS":
            # Similar structure but different ordering
            ctx_pre = self.ctx_pre.unsqueeze(0).expand(self.n_cls * self.n_attrs, -1, -1)
            ctx_mid = self.ctx_mid.unsqueeze(0).expand(self.n_cls * self.n_attrs, -1, -1)
            
            prompts = []
            for i in range(self.n_cls * self.n_attrs):
                name_len = self.name_lens[i]
                class_token = self.token_suffix[i:i+1, :name_len, :]
                attr_token = self.token_suffix[i:i+1, name_len:, :]
                
                # Structure: [SOS][SP1][Attribute][SP2][Class][EOS]
                prompt = torch.cat([
                    self.token_prefix[i:i+1, :, :],  # SOS token
                    ctx_pre[i:i+1, :, :],            # First set of soft prompts
                    attr_token,                       # Attribute token
                    ctx_mid[i:i+1, :, :],            # Second set of soft prompts
                    class_token                       # Class name
                ], dim=1)
                prompts.append(prompt)
            
            return torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Unknown prompt position: {self.prompt_position}")


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, attributes):
        super().__init__()
        # Initialize with attributes for enhanced prompts
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, attributes)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Store dimensions for handling attribute-based prompts
        self.n_cls = len(classnames)
        self.n_attrs = 3  # We're using 3 attributes per class

    def forward(self, image):
        # Get image features
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Get enhanced prompts (now includes attribute variations)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Scale logits
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)  # Prevent overflow

        # Return a dictionary containing all necessary components for loss computation
        return {
            'image_features': image_features,                 # [batch_size, feature_dim]
            'text_features': text_features,                   # [n_cls * n_attrs, feature_dim]
            'logit_scale': logit_scale,                      # scalar
            'logits': logit_scale * image_features @ text_features.t(),  # [batch_size, n_cls * n_attrs]
            'raw_logits': image_features @ text_features.t()  # Unscaled logits for temperature scaling
        }

@TRAINER_REGISTRY.register()
class AGL(TrainerX):
    """
    Attribute-Enhanced Context Optimization with Multi-Positive Contrastive Learning.
    Extends AGL to handle attribute-based prompts and contrastive learning.
    """
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def check_cfg(self, cfg):
        assert cfg.TRAINER.AGL.PREC in ["fp16", "fp32", "amp"]
        assert cfg.TRAINER.AGL.TEMPERATURE > 0, "Temperature must be positive"
        assert cfg.TRAINER.AGL.CONTRAST_WEIGHT > 0, "Contrastive loss weight must be positive"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Load class attributes from config-specified path
        print(f"Loading attributes from {cfg.DATASET.ATTRIBUTE_FILE}")
        with open(cfg.DATASET.ATTRIBUTE_FILE, 'r') as f:
            attributes = json.load(f)
            
        # Validate attributes exist for all classes
        for name in classnames:
            assert name in attributes, f"Missing attributes for class {name}"
            assert len(attributes[name]) >= 3, f"Need at least 3 attributes for class {name}"

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.AGL.PREC == "fp32" or cfg.TRAINER.AGL.PREC == "amp":
            clip_model.float()

        print("Building attribute-enhanced CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, attributes)

        # Freeze all parameters except prompt learner
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # Move model to device and set up optimization
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.AGL.PREC == "amp" else None

        # Handle multi-GPU setup
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def compute_attribute_contrastive_loss(self, output_dict, labels):
        """
        Multi-Positive Contrastive Loss implementation that properly handles
        multiple attribute-based positives per class.
        """
        image_features = output_dict['image_features']  # [batch_size, dim]
        text_features = output_dict['text_features']    # [n_cls * n_attrs, dim]
        temperature = self.cfg.TRAINER.AGL.TEMPERATURE
        n_attrs = self.model.module.n_attrs if isinstance(self.model, nn.DataParallel) else self.model.n_attrs
        batch_size = image_features.size(0)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t())  # [batch_size, n_cls * n_attrs]
        
        # Create ground truth distribution mask
        device = image_features.device
        mask = torch.zeros(batch_size, text_features.size(0), device=device)
        for i in range(batch_size):
            # Set 1s for all attributes of the correct class
            pos_start = labels[i] * n_attrs
            pos_end = pos_start + n_attrs
            mask[i, pos_start:pos_end] = 1.0
        
        # Normalize mask to create proper distribution
        p = mask / mask.sum(dim=1, keepdim=True)
        
        # Apply temperature scaling and compute log probabilities
        logits = logits / temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()  # Numerical stability
        q = F.log_softmax(logits, dim=1)
        
        # Compute cross entropy between distributions
        loss = -torch.sum(p * q, dim=1).mean()
        
        return loss

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        if self.cfg.TRAINER.AGL.PREC == "amp":
            with autocast():
                # Forward pass
                output_dict = self.model(image)
                
                # Classification loss using mean over attribute predictions
                logits = output_dict['logits']  # [batch_size, n_cls * n_attrs]
                n_attrs = self.model.module.n_attrs if isinstance(self.model, nn.DataParallel) else self.model.n_attrs
                cls_logits = logits.view(logits.size(0), -1, n_attrs).mean(dim=2)
                cls_loss = F.cross_entropy(cls_logits, label)
                
                # Contrastive loss
                cont_loss = self.compute_attribute_contrastive_loss(output_dict, label)
                
                # Combined loss
                loss = cls_loss + self.cfg.TRAINER.AGL.CONTRAST_WEIGHT * cont_loss
            
            # Optimization step with gradient scaling
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # Non-AMP training path
            output_dict = self.model(image)
            
            logits = output_dict['logits']
            n_attrs = self.model.module.n_attrs if isinstance(self.model, nn.DataParallel) else self.model.n_attrs
            cls_logits = logits.view(logits.size(0), -1, n_attrs).mean(dim=2)
            cls_loss = F.cross_entropy(cls_logits, label)
            
            cont_loss = self.compute_attribute_contrastive_loss(output_dict, label)
            loss = cls_loss + self.cfg.TRAINER.CONTRAST_WEIGHT * cont_loss
            
            self.model_backward_and_update(loss)

        # Return detailed loss summary
        loss_summary = {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "cont_loss": cont_loss.item(),
            "acc": compute_accuracy(cls_logits, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

   