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

# Import components from CoOp trainer
from .coop import CoOp, CustomCLIP, PromptLearner, TextEncoder, load_clip_to_cpu
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class AttributeEmbedding(nn.Module):
    """
    Handles the creation and management of attribute embeddings.
    This module is responsible for:
    1. Converting attribute text to CLIP token embeddings
    2. Managing attribute-class associations
    3. Providing attribute embeddings for prompt construction
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        super().__init__()
        self.dtype = clip_model.dtype
        self.n_attrs = cfg.DATASET.NUM_ATTRIBUTES
        self.classnames = classnames
        
        # Validate attributes exist for all classes
        for name in classnames:
            assert name in attributes, f"Missing attributes for class {name}"
            assert len(attributes[name]) >= self.n_attrs, \
                f"Need at least {self.n_attrs} attributes for class {name}"
        
        # Store processed attributes - limit to top n_attrs per class
        self.class_attributes = {
            cls: attrs[:self.n_attrs] for cls, attrs in attributes.items()
        }
        
        # Initialize attribute embeddings
        self.setup_attribute_embeddings(clip_model)
        
    def setup_attribute_embeddings(self, clip_model):
        """
        Creates and stores CLIP embeddings for all attributes.
        Uses CLIP's token embedding layer to convert attribute text to embeddings.
        """
        attribute_tokens = []  # Will store tokenized attributes
        self.attr_lens = []   # Store length of each attribute
        
        # Process each class and its attributes
        for class_name in self.classnames:
            class_attrs = self.class_attributes[class_name]
            
            for attr in class_attrs:
                # Clean and tokenize the attribute text
                attr = attr.replace("_", " ")
                tokens = clip.tokenize(attr)
                
                # Get embeddings using CLIP's token embedding
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokens).type(self.dtype)
                    # Remove SOS and EOS tokens, keep only attribute tokens
                    attr_tokens = embedding[0, 1:-1, :]
                    
                # Store embeddings and lengths
                attribute_tokens.append(attr_tokens)
                self.attr_lens.append(attr_tokens.size(0))
        
        # Stack all attribute embeddings into a single tensor
        # Register as buffer since we don't need gradients for these
        self.register_buffer(
            "attribute_embeddings", 
            torch.stack(attribute_tokens)
        )
        
    def get_attribute_embeddings(self, class_idx, attr_idx):
        """
        Retrieve attribute embeddings for a specific class and attribute index.
        
        Args:
            class_idx (int): Index of the class
            attr_idx (int): Index of the attribute (0 to n_attrs-1)
            
        Returns:
            torch.Tensor: Embedding tensor for the specified attribute
        """
        # Calculate combined index into our attribute embeddings
        idx = class_idx * self.n_attrs + attr_idx
        return self.attribute_embeddings[idx]
    
    def get_attribute_length(self, class_idx, attr_idx):
        """
        Get the token length of a specific attribute.
        
        Args:
            class_idx (int): Index of the class
            attr_idx (int): Index of the attribute
            
        Returns:
            int: Number of tokens in the attribute
        """
        idx = class_idx * self.n_attrs + attr_idx
        return self.attr_lens[idx]
 
 
    
class EnhancedPromptLearner(PromptLearner):
    """
    Extends CoOp's PromptLearner to support attribute-enhanced prompts without 
    modifying the original prompt learning mechanism.
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        # Initialize the original PromptLearner first - this sets up all CoOp functionality
        super().__init__(cfg, classnames, clip_model)
        
        # Create attribute embedding module
        self.attribute_embeddings = AttributeEmbedding(cfg, classnames, clip_model, attributes)
        
        # Store reference to class token position for attribute placement
        self.position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        # Get original CoOp prompts using parent class
        original_prompts = super().forward()
        
        # Generate attribute-enhanced prompts based on position
        if self.position == "end":
            enhanced_prompts = self._create_end_prompts()
        elif self.position == "middle":
            enhanced_prompts = self._create_middle_prompts()
        elif self.position == "front":
            enhanced_prompts = self._create_front_prompts()
        else:
            raise ValueError(f"Unknown prompt position: {self.position}")
        
        return original_prompts, enhanced_prompts

    def _pad_prompt(self, prompt, max_seq_length):
        """Helper method to pad a prompt to the specified length"""
        curr_len = prompt.size(1)
        if curr_len < max_seq_length:
            embed_dim = prompt.size(2)
            padding = torch.zeros(
                1, max_seq_length - curr_len, embed_dim, 
                device=prompt.device, 
                dtype=prompt.dtype
            )
            prompt = torch.cat([prompt, padding], dim=1)
        return prompt

    def _create_end_prompts(self):
        """Creates prompts with attributes after class token at the end position"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # First calculate maximum sequence length needed
        max_seq_length = 0
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            ctx_len = ctx.size(1)
            for j in range(self.attribute_embeddings.n_attrs):
                attr_len = self.attribute_embeddings.get_attribute_embeddings(i, j).size(0)
                total_len = 1 + ctx_len + name_len + attr_len + 1  # SOS + CTX + CLS + ATTR + EOS
                max_seq_length = max(max_seq_length, total_len)
        
        # Create prompts with padding
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :]
            class_i = self.token_suffix[i:i+1, :name_len, :]
            suffix_i = self.token_suffix[i:i+1, name_len:, :]
            
            for j in range(self.attribute_embeddings.n_attrs):
                attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                attr_j = attr_j.unsqueeze(0)
                
                # Construct prompt: [SOS][Context][Class][Attribute][EOS]
                prompt = torch.cat([
                    prefix_i,         # SOS
                    ctx[i:i+1, :, :], # Context
                    class_i,          # Class
                    attr_j,           # Attribute
                    suffix_i[:, -1:, :]  # EOS
                ], dim=1)
                
                # Pad to max length
                prompt = self._pad_prompt(prompt, max_seq_length)
                prompts.append(prompt)
        
        return torch.cat(prompts, dim=0)

    def _create_middle_prompts(self):
        """Creates prompts with attributes after middle context"""
        half_n_ctx = self.n_ctx // 2
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # First calculate maximum sequence length needed
        max_seq_length = 0
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            ctx_len = ctx.size(1)
            for j in range(self.attribute_embeddings.n_attrs):
                attr_len = self.attribute_embeddings.get_attribute_embeddings(i, j).size(0)
                # 1 (SOS) + half_ctx + name_len + half_ctx + attr_len + 1 (EOS)
                total_len = 1 + half_n_ctx + name_len + (ctx_len - half_n_ctx) + attr_len + 1
                max_seq_length = max(max_seq_length, total_len)
        
        # Create prompts with padding
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :]
            class_i = self.token_suffix[i:i+1, :name_len, :]
            suffix_i = self.token_suffix[i:i+1, name_len:, :]
            
            # Split context into two halves
            ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
            
            for j in range(self.attribute_embeddings.n_attrs):
                attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                attr_j = attr_j.unsqueeze(0)
                
                # Construct prompt: [SOS][CTX1][Class][CTX2][Attribute][EOS]
                prompt = torch.cat([
                    prefix_i,      # SOS
                    ctx_i_half1,   # First half context
                    class_i,       # Class
                    ctx_i_half2,   # Second half context
                    attr_j,        # Attribute
                    suffix_i[:, -1:, :]  # EOS
                ], dim=1)
                
                # Pad to max length
                prompt = self._pad_prompt(prompt, max_seq_length)
                prompts.append(prompt)
        
        return torch.cat(prompts, dim=0)

    def _create_front_prompts(self):
        """Creates prompts with attributes after context in front position"""
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # First calculate maximum sequence length needed
        max_seq_length = 0
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            ctx_len = ctx.size(1)
            for j in range(self.attribute_embeddings.n_attrs):
                attr_len = self.attribute_embeddings.get_attribute_embeddings(i, j).size(0)
                # 1 (SOS) + name_len + ctx_len + attr_len + 1 (EOS)
                total_len = 1 + name_len + ctx_len + attr_len + 1
                max_seq_length = max(max_seq_length, total_len)
        
        # Create prompts with padding
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :]
            class_i = self.token_suffix[i:i+1, :name_len, :]
            suffix_i = self.token_suffix[i:i+1, name_len:, :]
            
            for j in range(self.attribute_embeddings.n_attrs):
                attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                attr_j = attr_j.unsqueeze(0)
                
                # Construct prompt: [SOS][Class][CTX][Attribute][EOS]
                prompt = torch.cat([
                    prefix_i,          # SOS
                    class_i,           # Class
                    ctx[i:i+1, :, :],  # Context
                    attr_j,            # Attribute
                    suffix_i[:, -1:, :]   # EOS
                ], dim=1)
                
                # Pad to max length
                prompt = self._pad_prompt(prompt, max_seq_length)
                prompts.append(prompt)
        
        return torch.cat(prompts, dim=0)


class AGLCustomCLIP(CustomCLIP):
    """
    Extends CoOp's CustomCLIP to support both original and attribute-enhanced prompts.
    Maintains original CLIP functionality while adding attribute-guided learning.
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        # Initialize parent class first (sets up CLIP model, image encoder, etc.)
        super().__init__(cfg, classnames, clip_model)
        
        # Replace the prompt learner with our enhanced version
        # Note: This keeps all other CustomCLIP components intact
        self.prompt_learner = EnhancedPromptLearner(cfg, classnames, clip_model, attributes)
        
        # Store class-related dimensions for loss computation
        self.n_cls = len(classnames)
        self.n_attrs = cfg.DATASET.NUM_ATTRIBUTES

    def forward(self, image):
        """
        Forward pass that handles both original and attribute-enhanced prompts.
        
        Args:
            image: Input image tensor
            
        Returns:
            dict: Contains original logits, attribute logits, and features needed
                 for loss computation
        """
        # Get image features (same as original CLIP)
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Get both original and attribute-enhanced prompts
        original_prompts, enhanced_prompts = self.prompt_learner()
        
        # Get tokenized prompts (both original and attribute-based)
        orig_tokenized = self.prompt_learner.tokenized_prompts
        # Create expanded tokenized prompts for attributes (repeat for each attribute)
        attr_tokenized = orig_tokenized.repeat_interleave(self.n_attrs, dim=0)
        
        # Encode both types of prompts using CLIP's text encoder
        text_features_orig = self.text_encoder(original_prompts, orig_tokenized)
        text_features_attr = self.text_encoder(enhanced_prompts, attr_tokenized)
        
        # Normalize all features (exactly as in CLIP)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
        text_features_attr = text_features_attr / text_features_attr.norm(dim=-1, keepdim=True)
        
        # Compute logit scale (with safety clamp as in CLIP)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # Return dictionary with all necessary components for loss computation
        return {
            # Original CoOp logits
            'logits_orig': logit_scale * image_features @ text_features_orig.t(),
            
            # Attribute-enhanced logits
            'logits_attr': logit_scale * image_features @ text_features_attr.t(),
            
            # Features for potential additional computations in the trainer
            'image_features': image_features,
            'text_features_orig': text_features_orig,
            'text_features_attr': text_features_attr,
            
            # Scale factor for temperature scaling
            'logit_scale': logit_scale
        }

    def get_text_features(self, prompts, tokenized_prompts):
        """
        Helper method to get text features using CLIP's text encoder.
        Useful for computing features for different prompt variations.
        """
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features / text_features.norm(dim=-1, keepdim=True)
    
@TRAINER_REGISTRY.register()
class AGLTrainer(CoOp):
    """
    Attribute-Guided Learning trainer that extends CoOp.
    This implementation maintains CoOp's original functionality while adding
    attribute-based contrastive learning capabilities.
    """
    def check_cfg(self, cfg):
        # First run CoOp's configuration checks
        super().check_cfg(cfg)
        
        # Add our AGL-specific configuration checks
        assert cfg.TRAINER.AGL.TEMPERATURE > 0, "Temperature must be positive"
        assert cfg.TRAINER.AGL.CONTRAST_WEIGHT >= 0, "Contrast weight must be non-negative"
        assert cfg.DATASET.ATTRIBUTE_FILE, "Must specify attribute file path"

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Load and validate attributes
        print(f"Loading attributes from {cfg.DATASET.ATTRIBUTE_FILE}")
        try:
            with open(cfg.DATASET.ATTRIBUTE_FILE, 'r') as f:
                attributes = json.load(f)
            
            # Validate attribute coverage
            for name in classnames:
                assert name in attributes, f"Missing attributes for class {name}"
                assert len(attributes[name]) >= cfg.DATASET.NUM_ATTRIBUTES, \
                    f"Need at least {cfg.DATASET.NUM_ATTRIBUTES} attributes for class {name}"
        except Exception as e:
            raise Exception(f"Failed to load attributes: {e}")

        # Load CLIP model (same as CoOp)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        # Build AGLCustomCLIP model
        print("Building attribute-enhanced CLIP")
        self.model = AGLCustomCLIP(cfg, classnames, clip_model, attributes)

        # Freeze all parameters except prompt learner (same as CoOp)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # Initialize weights if specified
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # Setup model and optimization (same as CoOp)
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        # Setup mixed precision training
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Handle multi-GPU setup
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def compute_attribute_contrastive_loss(self, output_dict, labels):
        """
        Implements multi-positive contrastive loss that handles multiple attribute-based 
        positives per class. This creates a proper distribution over positive pairs
        and computes cross-entropy between predicted and target distributions.
        """
        image_features = output_dict['image_features']  # [batch_size, dim]
        text_features = output_dict['text_features_attr']  # [n_cls * n_attrs, dim]
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
        # Subtract max for numerical stability
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        q = F.log_softmax(logits, dim=1)
        
        # Compute cross entropy between distributions
        loss = -torch.sum(p * q, dim=1).mean()
        
        return loss

    def compute_loss(self, output_dict, label):
        """
        Computes combined loss from CoOp's classification loss and 
        attribute-guided contrastive loss.
        """
        # Original CoOp classification loss
        orig_logits = output_dict['logits_orig']
        loss_coop = F.cross_entropy(orig_logits, label)

        # If contrast weight is 0, behave exactly like CoOp
        if self.cfg.TRAINER.AGL.CONTRAST_WEIGHT == 0:
            return loss_coop

        # Add attribute contrastive loss
        cont_loss = self.compute_attribute_contrastive_loss(output_dict, label)
        
        # Combine losses
        total_loss = loss_coop + self.cfg.TRAINER.AGL.CONTRAST_WEIGHT * cont_loss
        
        # Store individual losses for logging
        self.loss_summary = {
            "loss_coop": loss_coop.item(),
            "loss_cont": cont_loss.item()
        }
        
        return total_loss

    def forward_backward(self, batch):
        """
        Performs forward and backward passes, inheriting CoOp's precision handling
        while adding our attribute-guided learning.
        """
        image, label = self.parse_batch_train(batch)
        
        # Handle different precision modes
        if self.cfg.TRAINER.COOP.PREC == "amp":
            with autocast():
                output_dict = self.model(image)
                loss = self.compute_loss(output_dict, label)
            
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_dict = self.model(image)
            loss = self.compute_loss(output_dict, label)
            self.model_backward_and_update(loss)

        # Prepare loss summary for logging
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output_dict['logits_orig'], label)[0].item(),
        }
        # Add individual losses if attribute learning is enabled
        if hasattr(self, 'loss_summary'):
            loss_summary.update(self.loss_summary)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary