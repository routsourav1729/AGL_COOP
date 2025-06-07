import os.path as osp
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

# Import components from CoCoOp trainer
from .cocoop import CoCoOp, CustomCLIP, PromptLearner, TextEncoder, load_clip_to_cpu
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class EnhancedAttributeProcessor(nn.Module):
    def __init__(self, cfg, classnames, clip_model, attributes):
        super().__init__()
        self.dtype = clip_model.dtype
        self.n_attrs = cfg.DATASET.NUM_ATTRIBUTES
        self.classnames = classnames
        self.max_attr_length = 10  # Maximum tokens per attribute to keep
        
        # Store processed attributes
        self.class_attributes = {
            cls: attrs[:self.n_attrs] for cls, attrs in attributes.items()
        }
        
        # Initialize attribute embeddings preserving token sequences
        self.setup_attribute_embeddings(clip_model)
        
    def setup_attribute_embeddings(self, clip_model):
        """Stores full token sequences for attributes without averaging"""
        attr_embeddings = []
        attr_attention_masks = []
        
        for class_name in self.classnames:
            class_attrs = self.class_attributes[class_name]
            
            class_attr_embeddings = []
            class_attr_masks = []
            for attr in class_attrs:
                attr = attr.replace("_", " ")
                tokens = clip.tokenize(attr)
                
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokens).type(self.dtype)
                    # Extract tokens (excluding SOS and EOS)
                    attr_tokens = embedding[0, 1:-1, :]
                    
                    # Create attention mask (1 for real tokens, 0 for padding)
                    mask = torch.ones(self.max_attr_length, dtype=torch.bool)
                    
                    # Handle variable length by truncating or padding
                    if attr_tokens.size(0) > self.max_attr_length:
                        attr_tokens = attr_tokens[:self.max_attr_length]
                    else:
                        mask[attr_tokens.size(0):] = 0
                        # Pad with zeros to max length
                        pad_size = self.max_attr_length - attr_tokens.size(0)
                        attr_tokens = torch.cat([
                            attr_tokens, 
                            torch.zeros(pad_size, attr_tokens.size(1), dtype=self.dtype)
                        ], dim=0)
                    
                    class_attr_embeddings.append(attr_tokens)
                    class_attr_masks.append(mask)
            
            class_attr_embeddings = torch.stack(class_attr_embeddings)
            class_attr_masks = torch.stack(class_attr_masks)
            
            attr_embeddings.append(class_attr_embeddings)
            attr_attention_masks.append(class_attr_masks)
        
        # Shape: [n_classes, n_attrs, max_attr_length, dim]
        self.register_buffer("attribute_embeddings", torch.stack(attr_embeddings))
        # Shape: [n_classes, n_attrs, max_attr_length]
        self.register_buffer("attention_masks", torch.stack(attr_attention_masks))
        
    def get_attributes(self, class_indices=None):
        """Returns attribute embeddings and attention masks"""
        if class_indices is None:
            return self.attribute_embeddings, self.attention_masks
            
        return (
            self.attribute_embeddings[class_indices],
            self.attention_masks[class_indices]
        )
    
    
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention module that allows image features to attend to
    different parts of multiple attribute descriptions.
    """
    def __init__(self, img_dim, attr_dim, ctx_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = ctx_dim // num_heads
        assert self.head_dim * num_heads == ctx_dim, "ctx_dim must be divisible by num_heads"
        
        # Image feature projection
        self.img_proj = nn.Linear(img_dim, ctx_dim)
        
        # Projections for keys, queries, and values
        self.query_proj = nn.Linear(ctx_dim, ctx_dim)
        self.key_proj = nn.Linear(attr_dim, ctx_dim)
        self.value_proj = nn.Linear(attr_dim, ctx_dim)
        
        # Output projection
        self.output_proj = nn.Linear(ctx_dim, ctx_dim)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(ctx_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, img_features, attr_features, attention_mask=None):
    
        if attr_features.dim() == 5:
            attr_features = attr_features.squeeze(1)
        
        # Get model's dtype
        dtype = self.query_proj.weight.dtype
        
        # Convert inputs to correct dtype
        img_features = img_features.to(dtype)
        attr_features = attr_features.to(dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype)
        
        batch_size = img_features.size(0)
        n_attrs = attr_features.size(1)
        max_length = attr_features.size(2)
        
        # Project image features to queries
        img_ctx = self.img_proj(img_features)
        queries = self.query_proj(img_ctx)
        queries = queries.view(batch_size, 1, self.num_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)  # [batch, heads, 1, head_dim]
        
        # Flatten attributes for processing
        flat_attrs = attr_features.reshape(batch_size, -1, attr_features.size(-1))
        
        # Process keys and values (ensuring dtype consistency)
        keys = self.key_proj(flat_attrs)
        keys = keys.reshape(batch_size, -1, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)  # [batch, heads, tokens, head_dim]
        
        values = self.value_proj(flat_attrs)
        values = values.reshape(batch_size, -1, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)  # [batch, heads, tokens, head_dim]
        
        # Process attention mask
        if attention_mask is not None:
            flat_mask = attention_mask.reshape(batch_size, 1, 1, -1)
            attn_mask = (1.0 - flat_mask.float()) * -10000.0
        else:
            attn_mask = None
        
        # Compute attention (ensure consistent dtype throughout)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values (explicit cast to ensure matching dtype)
        attn_probs = attn_probs.to(dtype)
        values = values.to(dtype)
        
        context = torch.matmul(attn_probs, values)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, self.num_heads * self.head_dim)
        
        # Final projection
        context_output = self.output_proj(context)
        context_output = self.output_dropout(context_output)
        context_output = self.layer_norm(img_ctx + context_output)
        
        return context_output

class EnhancedCrossPromptLearner(PromptLearner):
    def __init__(self, cfg, classnames, clip_model, attributes):
        # Initialize parent PromptLearner first
        super().__init__(cfg, classnames, clip_model)
        
        # Create enhanced attribute processor
        self.attribute_processor = EnhancedAttributeProcessor(
            cfg, classnames, clip_model, attributes
        )
        
        # Get dimensions from parent and config
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        num_heads = cfg.TRAINER.CROSSCOCOOP.NUM_HEADS
        attn_dropout = cfg.TRAINER.CROSSCOCOOP.ATTN_DROPOUT
        
        # Remove original meta_net
        del self.meta_net
        
        # Create enhanced cross-attention module
        self.cross_attention = MultiHeadCrossAttention(
            img_dim=vis_dim,
            attr_dim=ctx_dim,
            ctx_dim=ctx_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        # Add attribute selection mechanism (optional)
        self.attr_selection = nn.Sequential(
            nn.Linear(vis_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.attribute_processor.n_attrs)
        )
        
        # Match dtype with CLIP model
        if cfg.TRAINER.CROSSCOCOOP.PREC == "fp16":
            self.cross_attention = self.cross_attention.half()
            self.attr_selection = self.attr_selection.half()
    
    def forward(self, im_features):
        """Enhanced forward pass using token-level cross-attention"""
        # Get token prefix, suffix, and context vectors from parent
        prefix = self.token_prefix  # SOS token embeddings
        suffix = self.token_suffix  # Class name + EOS token embeddings
        ctx = self.ctx              # Learnable context vectors
        
        batch_size = im_features.size(0)
        
        # Generate prompts for all images and classes
        all_prompts = []
        
        for i, img_feat in enumerate(im_features):
            # Get attribute importance weights for this image (optional)
            attr_weights = F.softmax(self.attr_selection(img_feat), dim=0)
            
            # Process each class
            class_prompts = []
            for cls_idx in range(self.n_cls):
                # Get attributes and attention masks for this class
                cls_attrs, cls_masks = self.attribute_processor.get_attributes(
                    torch.tensor([cls_idx], device=img_feat.device)
                )
                cls_attrs = cls_attrs[0]  # Remove batch dimension
                cls_masks = cls_masks[0]  # Remove batch dimension
                
                # Weight attributes by predicted importance (optional)
                # [n_attrs, 1, 1] * [n_attrs, max_length, dim]
                weighted_attrs = attr_weights[None, :, None, None] * cls_attrs
                
                # Apply cross-attention between image and attributes
                ctx_bias = self.cross_attention(
                    img_feat.unsqueeze(0),      # [1, vis_dim]
                    weighted_attrs.unsqueeze(0), # [1, n_attrs, max_length, dim]
                    cls_masks.unsqueeze(0)      # [1, n_attrs, max_length]
                )  # [1, ctx_dim]
                
                # Apply bias to context vectors
                ctx_bias = ctx_bias.unsqueeze(1)  # [1, 1, ctx_dim]
                ctx_shifted = ctx.unsqueeze(0) + ctx_bias  # [1, n_ctx, ctx_dim]
                
                # Construct prompt
                class_prompt = self.construct_prompts(
                    ctx_shifted,
                    prefix[cls_idx:cls_idx+1],
                    suffix[cls_idx:cls_idx+1]
                )
                class_prompts.append(class_prompt)
            
            # Stack prompts for all classes for this image
            image_prompts = torch.cat(class_prompts, dim=0)
            all_prompts.append(image_prompts)
        
        # Stack prompts for all images [batch_size, n_cls, n_tkn, ctx_dim]
        return torch.stack(all_prompts)

class CrossCustomCLIP(CustomCLIP):
    """
    Extends CoCoOp's CustomCLIP to use our cross-attention based prompt learner.
    Maintains the same interface as CustomCLIP for seamless integration.
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        # We'll initialize the parent but then override the prompt_learner
        super().__init__(cfg, classnames, clip_model)
        
        # Replace the standard prompt learner with our cross-attention version
        # Store the original to avoid issues during state_dict loading
        original_prompt_learner = self.prompt_learner
        
        # Create our enhanced prompt learner with attribute support
        self.prompt_learner = EnhancedCrossPromptLearner(
            cfg, classnames, clip_model, attributes
        )
        
        if cfg.TRAINER.CROSSCOCOOP.PREC == "fp16":
            self.prompt_learner = self.prompt_learner.half()
        
        # Make sure tokenized_prompts points to the new prompt learner's version
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Transfer any weights that can be transferred from original to new prompt learner
        # (especially the context tokens which we want to keep)
        with torch.no_grad():
            if hasattr(original_prompt_learner, 'ctx') and hasattr(self.prompt_learner, 'ctx'):
                self.prompt_learner.ctx.copy_(original_prompt_learner.ctx)
        
        # Delete the original to free memory
        del original_prompt_learner
    
    def forward(self, image, label=None):
        """
        Forward pass that maintains the same interface as CustomCLIP.
        
        Args:
            image: Input images
            label: Ground truth labels (None during inference)
            
        Returns:
            During training: Cross-entropy loss
            During inference: Classification logits
        """
        # Get tokenized prompts
        tokenized_prompts = self.tokenized_prompts
        
        # Get logit scale
        logit_scale = self.logit_scale.exp()
        
        # Encode images
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Generate prompts using our cross-attention based prompt learner
        prompts = self.prompt_learner(image_features)
        
        # Calculate logits for each image
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            # Encode text features
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        
        # Stack logits from all images
        logits = torch.stack(logits)
        
        # Return loss during training, logits during inference
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits

@TRAINER_REGISTRY.register()
class CrossCoCoOp(CoCoOp):
    """
    Cross-Attention Conditional Context Optimization.
    Extends CoCoOp with attribute-guided cross-attention for prompt learning.
    """
    def check_cfg(self, cfg):
        """
        Check configuration and add CrossCoCoOp-specific checks.
        """
        # First run CoCoOp's configuration checks
        super().check_cfg(cfg)
        
        # Add CrossCoCoOp-specific checks
        assert cfg.TRAINER.CROSSCOCOOP.PREC in ["fp16", "fp32", "amp"], \
            "Precision must be one of fp16, fp32, or amp"
        assert cfg.TRAINER.CROSSCOCOOP.NUM_HEADS > 0, \
            "Number of attention heads must be positive"
        assert cfg.TRAINER.CROSSCOCOOP.ATTN_DROPOUT >= 0, \
            "Attention dropout must be non-negative"
        assert cfg.TRAINER.CROSSCOCOOP.ATTRIBUTE_DROPOUT >= 0, \
            "Attribute dropout must be non-negative"
        assert cfg.DATASET.ATTRIBUTE_FILE, \
            "Must specify attribute file path in DATASET.ATTRIBUTE_FILE"
    
    def build_model(self):
        """
        Build the CrossCoCoOp model with attribute-guided cross-attention.
        """
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        # Load attributes
        print(f"Loading attributes from {cfg.DATASET.ATTRIBUTE_FILE}")
        try:
            with open(cfg.DATASET.ATTRIBUTE_FILE, 'r') as f:
                attributes = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load attributes: {e}")
        
        # Load CLIP model (reusing CoCoOp's function)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        # Set precision based on config
        if cfg.TRAINER.CROSSCOCOOP.PREC == "fp32" or cfg.TRAINER.CROSSCOCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        
        # Build CrossCustomCLIP model
        print("Building Cross-Attention Conditional CLIP")
        self.model = CrossCustomCLIP(cfg, classnames, clip_model, attributes)
        
        # Freeze all parameters except prompt learner
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check which parameters will be updated
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        # Initialize weights if specified
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        
        # Set up model, optimizer, and scheduler (same as CoCoOp) 
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        # Set up mixed precision training
        self.scaler = GradScaler() if cfg.TRAINER.CROSSCOCOOP.PREC == "amp" else None
        
        # Handle multi-GPU setup
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    def forward_backward(self, batch):
        """
        Forward and backward pass, using the correct precision mode.
        Reuses CoCoOp's implementation for compatibility.
        """
        image, label = self.parse_batch_train(batch)
        
        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        # Use the precision mode specified in CrossCoCoOp config
        prec = self.cfg.TRAINER.CROSSCOCOOP.PREC
        
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        loss_summary = {"loss": loss.item()}
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        
        return loss_summary
    
