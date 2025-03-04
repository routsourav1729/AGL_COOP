import wandb
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
        # for class_name in attributes:
        #     for i, attr in enumerate(attributes[class_name]):
        #         tokens = clip.tokenize(attr)
        #         print(f"Class: {class_name}, Attr: {attr}")
        #         print(f"Token count: {tokens.shape[1] - 2}")
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
        print(f"Max attribute length: {max(self.attr_lens)} tokens")
        print(f"Average attribute length: {sum(self.attr_lens)/len(self.attr_lens):.2f} tokens")
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
        
        # CLIP's max context length
        max_allowed_length = 77
        
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :]  # SOS
            class_i = self.token_suffix[i:i+1, :name_len, :]  # Class name
            suffix_i = self.token_suffix[i:i+1, name_len:, :]  # EOS
            
            for j in range(self.attribute_embeddings.n_attrs):
                attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                attr_j = attr_j.unsqueeze(0)
                
                # Calculate available space for attribute
                fixed_tokens = 1 + ctx.size(1) + name_len + 1  # SOS + CTX + CLASS + EOS
                avail_attr_tokens = max_allowed_length - fixed_tokens
                
                # Take only as many attribute tokens as will fit
                if attr_j.size(1) > avail_attr_tokens and avail_attr_tokens > 0:
                    attr_j = attr_j[:, :avail_attr_tokens, :]
                
                # Construct prompt: [SOS][Context][Class][Attribute][EOS]
                prompt = torch.cat([
                    prefix_i,         # SOS
                    ctx[i:i+1, :, :], # Context
                    class_i,          # Class
                    attr_j,           # Attribute (now properly sized)
                    suffix_i[:, -1:, :]  # EOS
                ], dim=1)
                
                # Ensure the total length doesn't exceed CLIP's limit
                if prompt.size(1) > max_allowed_length:
                    prompt = prompt[:, :max_allowed_length, :]
                
                prompts.append(prompt)
        
        # Now pad all prompts to the same length (max_allowed_length)
        padded_prompts = []
        for prompt in prompts:
            curr_len = prompt.size(1)
            if curr_len < max_allowed_length:
                embed_dim = prompt.size(2)
                padding = torch.zeros(
                    1, max_allowed_length - curr_len, embed_dim, 
                    device=prompt.device, 
                    dtype=prompt.dtype
                )
                prompt = torch.cat([prompt, padding], dim=1)
            padded_prompts.append(prompt)
        
        return torch.cat(padded_prompts, dim=0)

    def _create_middle_prompts(self):
        """Creates prompts with attributes between the class token and second half of context"""
        half_n_ctx = self.n_ctx // 2
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # CLIP's max context length
        max_allowed_length = 77
        
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :]  # SOS
            class_i = self.token_suffix[i:i+1, :name_len, :]  # Class name
            suffix_i = self.token_suffix[i:i+1, name_len:, :]  # EOS
            
            # Split context into two halves
            ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
            ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
            
            for j in range(self.attribute_embeddings.n_attrs):
                attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                attr_j = attr_j.unsqueeze(0)
                
                # Calculate available space for attribute
                fixed_tokens = 1 + half_n_ctx + name_len + (ctx.size(1) - half_n_ctx) + 1  # SOS + CTX1 + CLASS + CTX2 + EOS
                avail_attr_tokens = max_allowed_length - fixed_tokens
                
                # Take only as many attribute tokens as will fit
                if attr_j.size(1) > avail_attr_tokens and avail_attr_tokens > 0:
                    attr_j = attr_j[:, :avail_attr_tokens, :]
                
                # Construct prompt: [SOS][CTX1][Class][Attribute][CTX2][EOS]
                prompt = torch.cat([
                    prefix_i,         # SOS
                    ctx_i_half1,      # First half context
                    class_i,          # Class
                    attr_j,           # Attribute (now properly sized)
                    ctx_i_half2,      # Second half context
                    suffix_i[:, -1:, :]  # EOS
                ], dim=1)
                
                # Ensure the total length doesn't exceed CLIP's limit
                if prompt.size(1) > max_allowed_length:
                    prompt = prompt[:, :max_allowed_length, :]
                
                prompts.append(prompt)
        
        # Now pad all prompts to the same length (max_allowed_length)
        padded_prompts = []
        for prompt in prompts:
            curr_len = prompt.size(1)
            if curr_len < max_allowed_length:
                embed_dim = prompt.size(2)
                padding = torch.zeros(
                    1, max_allowed_length - curr_len, embed_dim, 
                    device=prompt.device, 
                    dtype=prompt.dtype
                )
                prompt = torch.cat([prompt, padding], dim=1)
            padded_prompts.append(prompt)
        
        return torch.cat(padded_prompts, dim=0)

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
        # Get image features
        image_features = self.image_encoder(image.type(self.dtype))
        original_prompts, enhanced_prompts = self.prompt_learner()
        
        # Get text features for original CoOp behavior (unchanged)
        orig_tokenized = self.prompt_learner.tokenized_prompts
        text_features_orig = self.text_encoder(original_prompts, orig_tokenized)
        
        # Normalize exactly as CoOp does
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_orig = text_features_orig / text_features_orig.norm(dim=-1, keepdim=True)
        
        # Apply logit scale exactly as CoOp does
        logit_scale = self.logit_scale.exp()
        
        # Calculate original logits exactly as CoOp would
        logits_orig = logit_scale * image_features @ text_features_orig.t()
        
        # Check if we're in test mode by examining the call stack
        import inspect
        caller_functions = [frame.function for frame in inspect.stack()]
        in_test_mode = 'test' in caller_functions
        
        # In test mode, return only the original logits (CoOp behavior)
        if in_test_mode:
            return logits_orig
        
        # For training, also compute the attribute-based features
        attr_tokenized = orig_tokenized.repeat_interleave(self.n_attrs, dim=0)
        text_features_attr = self.text_encoder(enhanced_prompts, attr_tokenized)
        text_features_attr = text_features_attr / text_features_attr.norm(dim=-1, keepdim=True)
        
        # Return both the original logits and the data needed for attribute learning
        return {
            'logits_orig': logits_orig,  # Original CoOp logits
            'image_features': image_features,  # For contrastive loss
            'text_features_orig': text_features_orig,
            'text_features_attr': text_features_attr,
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
    
        # Initialize wandb
        try:
            if not hasattr(cfg.TRAINER, 'NO_WANDB') or not cfg.TRAINER.NO_WANDB:
                wandb.init(
                    project="attribute-guided-learning",
                    name=f"{cfg.DATASET.NAME}_{cfg.MODEL.BACKBONE.NAME}_shots{cfg.DATASET.NUM_SHOTS}_seed{cfg.SEED}",
                    config={
                        "dataset": cfg.DATASET.NAME, 
                        "num_shots": cfg.DATASET.NUM_SHOTS,
                        "backbone": cfg.MODEL.BACKBONE.NAME,
                        "contrast_weight": cfg.TRAINER.AGL.CONTRAST_WEIGHT,
                        "temperature": cfg.TRAINER.AGL.TEMPERATURE
                    }
                )
                self.use_wandb = True
            else:
                self.use_wandb = False
        except Exception as e:
            print(f"Wandb initialization error: {e}")
            self.use_wandb = False   

    # def compute_attribute_contrastive_loss(self, output_dict, labels):
    #     """
    #     Computes a multi-positive contrastive loss using unique attribute-enhanced
    #     soft prompt embeddings with improved stability and scaling.
        
    #     For each image, only the attribute embeddings for its own class are considered
    #     positives and the embeddings for all other classes serve as negatives.
        
    #     This implementation includes:
    #     - Feature normalization for proper cosine similarity
    #     - Improved temperature scaling
    #     - Loss scaling for better integration with classification loss
    #     - Numerical stability improvements
    #     """
    #     # Extract features
    #     image_features = output_dict['image_features']  # [batch_size, dim]
    #     full_text_features = output_dict['text_features_attr']  # [n_cls * n_attrs, dim]
        
    #     # Normalize features to ensure proper cosine similarity calculation
    #     image_features = F.normalize(image_features, p=2, dim=1)
    #     full_text_features = F.normalize(full_text_features, p=2, dim=1)
        
    #     # Get configuration parameters
    #     temperature = self.cfg.TRAINER.AGL.TEMPERATURE
    #     n_attrs = (self.model.module.n_attrs if isinstance(self.model, nn.DataParallel)
    #             else self.model.n_attrs)
    #     batch_size = image_features.size(0)
        
    #     # Identify unique classes present in the batch
    #     unique_classes = torch.unique(labels)
    #     num_unique = unique_classes.size(0)
        
    #     # Build a mapping from original class label to its index in the reduced set
    #     class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
        
    #     # Build a reduced text feature matrix that contains attribute embeddings only
    #     # for the unique classes in the batch (each class contributes n_attrs embeddings)
    #     reduced_text_features = []
    #     for cls in unique_classes:
    #         start_idx = cls.item() * n_attrs
    #         end_idx = start_idx + n_attrs
    #         reduced_text_features.append(full_text_features[start_idx:end_idx])
    #     reduced_text_features = torch.cat(reduced_text_features, dim=0)  # [num_unique * n_attrs, dim]
        
    #     # Ensure text features are normalized after concatenation
    #     reduced_text_features = F.normalize(reduced_text_features, p=2, dim=1)
        
    #     # Compute similarity logits between each image and the reduced attribute embeddings
    #     logits = torch.matmul(image_features, reduced_text_features.t())  # [batch_size, num_unique * n_attrs]
        
    #     # Create a ground truth mask where for each image, only the embeddings corresponding
    #     # to its own class are marked as positives
    #     device = image_features.device
    #     mask = torch.zeros(batch_size, num_unique * n_attrs, device=device)
    #     for i in range(batch_size):
    #         cls = labels[i].item()
    #         unique_idx = class_to_index[cls]  # index in the reduced unique set
    #         pos_start = unique_idx * n_attrs
    #         pos_end = pos_start + n_attrs
    #         mask[i, pos_start:pos_end] = 1.0
        
    #     # Add a small smoothing factor to avoid numerical issues
    #     # This provides a slight regularization effect
    #     smoothing = 0.05
    #     smoothed_mask = (1 - smoothing) * mask + smoothing / (num_unique * n_attrs)
        
    #     # Normalize mask to yield a valid distribution (each row sums to 1)
    #     p = smoothed_mask / smoothed_mask.sum(dim=1, keepdim=True)
        
    #     # Scale logits by temperature and improve numerical stability
    #     logits = logits / temperature
        
    #     # Subtract max for numerical stability (only along specified dimension)
    #     logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
        
    #     # Compute log probabilities over the unique attribute embeddings
    #     log_probs = F.log_softmax(logits, dim=1)
        
    #     # Compute the cross-entropy between the ground truth (p) and predicted (log_probs) distributions
    #     loss = -torch.sum(p * log_probs, dim=1).mean()
        
    #     # Apply a scaling factor to make the loss magnitude more manageable and 
    #     # better aligned with the classification loss
    #     loss_scale = 0.5
    #     loss = loss * loss_scale
        
    #     return loss

    def sup_con_loss_with_hn(self, output_dict, labels):
        """
        Implements supervised contrastive loss with hard negative mining to
        address class imbalance issues.
        """
        # Extract features and normalize
        image_features = F.normalize(output_dict['image_features'], p=2, dim=1)
        text_features = F.normalize(output_dict['text_features_attr'], p=2, dim=1)
        
        # Get configuration parameters
        temperature = self.cfg.TRAINER.AGL.TEMPERATURE
        
        n_attrs = (self.model.module.n_attrs if isinstance(self.model, nn.DataParallel)
                else self.model.n_attrs)
        batch_size = image_features.size(0)
        
        # Create unique class mask and identify class indices
        unique_classes = torch.unique(labels)
        class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
        
        # Gather attribute features for classes in this batch
        class_attr_features = []
        for cls in unique_classes:
            start_idx = cls.item() * n_attrs
            end_idx = start_idx + n_attrs
            class_attr_features.append(text_features[start_idx:end_idx])
        
        # Combine attribute features
        attr_features = torch.cat(class_attr_features, dim=0)
        
        # Compute logits between images and all attribute features
        logits = torch.matmul(image_features, attr_features.t()) / temperature
        
        # Create class-based mask (1 if same class, 0 otherwise)
        device = image_features.device
        num_attrs_total = attr_features.size(0)
        mask = torch.zeros(batch_size, num_attrs_total, device=device)
        
        # Fill in mask based on class membership
        for i in range(batch_size):
            cls = labels[i].item()
            unique_idx = class_to_index[cls]
            pos_start = unique_idx * n_attrs
            pos_end = pos_start + n_attrs
            mask[i, pos_start:pos_end] = 1.0
        
        # Hard negative mining: For each sample, select k hardest negatives
        k_hard_negatives = min(n_attrs * 3, (num_attrs_total - n_attrs))  # 3x positives
        hard_negative_mask = torch.zeros_like(mask)
        
        for i in range(batch_size):
            # Get logits for this sample
            sample_logits = logits[i]
            # Create mask of negative samples (where mask is 0)
            negative_mask = (1.0 - mask[i])
            # Set positive logits to large negative value
            negative_logits = sample_logits - 1000.0 * (1.0 - negative_mask)
            # Get indices of top-k hard negatives (highest similarity among negatives)
            _, hard_negative_indices = torch.topk(negative_logits, k_hard_negatives)
            # Mark these as the hard negatives to use
            hard_negative_mask[i, hard_negative_indices] = 1.0
        
        # Final mask: combine positives and hard negatives
        final_mask = mask + hard_negative_mask
        
        # Compute log-softmax over the combinations of positives and hard negatives
        # First, create a mask to only include positives and selected hard negatives
        valid_logits = logits.clone()
        valid_logits[final_mask == 0] = -1000.0  # Effectively remove these from softmax
        
        # Compute log probabilities
        log_probs = F.log_softmax(valid_logits, dim=1)
        
        # We only care about positives for the loss calculation
        # Get mean log-prob for positive samples
        pos_log_probs = (mask * log_probs).sum(1) / mask.sum(1)
        
        # Loss is negative of mean log-probs (to maximize log-probs)
        loss = -pos_log_probs.mean()
        
        return loss
    
    
    
    def compute_loss(self, output_dict, label):
        """
        Computes combined loss from CoOp's classification loss and 
        attribute-guided contrastive loss.
        """
        # Original CoOp classification loss
        orig_logits = output_dict['logits_orig']
        loss_coop = F.cross_entropy(orig_logits, label)

       

        # Add attribute contrastive loss
        # cont_loss = self.compute_attribute_contrastive_loss(output_dict, label)
        cont_loss = self.sup_con_loss_with_hn(output_dict, label)
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
        # Log to wandb if enabled
        if hasattr(self, 'use_wandb') and self.use_wandb:
            wandb_log = {
                "train/loss": loss_summary["loss"],
                "train/acc": loss_summary["acc"],
                "train/lr": self.get_current_lr()
            }
            if "loss_coop" in loss_summary:
                wandb_log["loss/classification"] = loss_summary["loss_coop"]
            if "loss_cont" in loss_summary:
                wandb_log["loss/contrastive"] = loss_summary["loss_cont"]
            wandb.log(wandb_log)
        return loss_summary