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

# Import components from CoCoOp trainer
from .cocoop import CoCoOp, CustomCLIP, PromptLearner, TextEncoder, load_clip_to_cpu
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



class EnhancedCoCoOpPromptLearner(PromptLearner):
    """
    Extends CoCoOp's PromptLearner to support attribute-enhanced prompts
    while preserving image-conditioning capability.
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        # Initialize the original PromptLearner from CoCoOp first
        # This sets up the context vectors, meta-network, and tokenized prompts
        super().__init__(cfg, classnames, clip_model)
        
        # Create attribute embedding module
        self.attribute_embeddings = AttributeEmbedding(cfg, classnames, clip_model, attributes)
        
        # Store class token position for attribute placement
        self.position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, im_features):
        """
        Forward pass that generates both original CoCoOp prompts and
        attribute-enhanced prompts using image conditioning.
        
        Args:
            im_features: Image features from CLIP's image encoder
            
        Returns:
            tuple: (original_prompts, enhanced_prompts)
        """
        # Get token prefix, suffix, and context vectors
        prefix = self.token_prefix  # SOS token embeddings
        suffix = self.token_suffix  # Class name + EOS token embeddings
        ctx = self.ctx              # Learnable context vectors
        
        # Apply image conditioning using the meta-network (CoCoOp's approach)
        bias = self.meta_net(im_features)  # Generate image-specific bias
        bias = bias.unsqueeze(1)           # Reshape to apply to context vectors
        ctx = ctx.unsqueeze(0)             # Prepare context vectors for batch
        ctx_shifted = ctx + bias           # Apply image-specific conditioning
        
        # Generate original CoCoOp prompts
        # This follows CoCoOp's implementation
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            # Expand the context for all classes
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            # Construct prompts using parent class method
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        original_prompts = torch.stack(prompts)
        
        # Generate attribute-enhanced prompts based on position
        if self.position == "end":
            enhanced_prompts = self._create_end_prompts(ctx_shifted)
        elif self.position == "middle":
            enhanced_prompts = self._create_middle_prompts(ctx_shifted)
        elif self.position == "front":
            enhanced_prompts = self._create_front_prompts(ctx_shifted)
        else:
            raise ValueError(f"Unknown prompt position: {self.position}")
        
        return original_prompts, enhanced_prompts

    def _create_end_prompts(self, ctx_shifted):
        """
        Creates prompts with attributes after the class token at the end position,
        while preserving image conditioning.
        """
        # CLIP's max context length
        max_allowed_length = 77
        
        # For each image in the batch, create attribute-enhanced prompts
        batch_size = ctx_shifted.size(0)
        all_prompts = []
        
        for b in range(batch_size):
            # Get image-specific context for this batch item
            ctx_image = ctx_shifted[b]  # Image-conditioned context
            
            # Create prompts for each class and each attribute
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :, :]  # SOS
                class_i = self.token_suffix[i:i+1, :name_len, :]  # Class name
                suffix_i = self.token_suffix[i:i+1, name_len:, :]  # EOS
                
                # Expand context for this class
                ctx_i = ctx_image.unsqueeze(0)  # (1, n_ctx, ctx_dim)
                
                # Create a prompt for each attribute
                for j in range(self.attribute_embeddings.n_attrs):
                    attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                    attr_j = attr_j.unsqueeze(0)  # (1, attr_len, ctx_dim)
                    
                    # Calculate available space for attribute
                    fixed_tokens = 1 + ctx_i.size(1) + name_len + 1  # SOS + CTX + CLASS + EOS
                    avail_attr_tokens = max_allowed_length - fixed_tokens
                    
                    # Take only as many attribute tokens as will fit
                    if attr_j.size(1) > avail_attr_tokens and avail_attr_tokens > 0:
                        attr_j = attr_j[:, :avail_attr_tokens, :]
                    
                    # Construct prompt: [SOS][Context][Class][Attribute][EOS]
                    prompt = torch.cat([
                        prefix_i,     # SOS
                        ctx_i,        # Context (image-conditioned)
                        class_i,      # Class
                        attr_j,       # Attribute
                        suffix_i[:, -1:, :]  # EOS
                    ], dim=1)
                    
                    # Ensure the total length doesn't exceed CLIP's limit
                    if prompt.size(1) > max_allowed_length:
                        prompt = prompt[:, :max_allowed_length, :]
                    
                    all_prompts.append(prompt)
        
        # Stack all prompts for all images, classes, and attributes
        stacked_prompts = torch.cat(all_prompts, dim=0)
        
        return stacked_prompts

    def _create_middle_prompts(self, ctx_shifted):
        """
        Creates prompts with attributes between the class token and second half
        of context, while preserving image conditioning.
        """
        # Split context in half for middle positioning
        half_n_ctx = self.n_ctx // 2
        max_allowed_length = 77
        
        batch_size = ctx_shifted.size(0)
        all_prompts = []
        
        for b in range(batch_size):
            # Get image-specific context for this batch item
            ctx_image = ctx_shifted[b]
            
            # Split the context into two halves
            ctx_half1 = ctx_image[:half_n_ctx, :]
            ctx_half2 = ctx_image[half_n_ctx:, :]
            
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :, :]
                class_i = self.token_suffix[i:i+1, :name_len, :]
                suffix_i = self.token_suffix[i:i+1, name_len:, :]
                
                # Expand context halves for this class
                ctx_half1_i = ctx_half1.unsqueeze(0)
                ctx_half2_i = ctx_half2.unsqueeze(0)
                
                for j in range(self.attribute_embeddings.n_attrs):
                    attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                    attr_j = attr_j.unsqueeze(0)
                    
                    # Calculate available space for attribute
                    fixed_tokens = 1 + half_n_ctx + name_len + (self.n_ctx - half_n_ctx) + 1
                    avail_attr_tokens = max_allowed_length - fixed_tokens
                    
                    if attr_j.size(1) > avail_attr_tokens and avail_attr_tokens > 0:
                        attr_j = attr_j[:, :avail_attr_tokens, :]
                    
                    # Construct prompt: [SOS][CTX1][Class][Attribute][CTX2][EOS]
                    prompt = torch.cat([
                        prefix_i,     # SOS
                        ctx_half1_i,  # First half context (image-conditioned)
                        class_i,      # Class
                        attr_j,       # Attribute
                        ctx_half2_i,  # Second half context (image-conditioned)
                        suffix_i[:, -1:, :]  # EOS
                    ], dim=1)
                    
                    if prompt.size(1) > max_allowed_length:
                        prompt = prompt[:, :max_allowed_length, :]
                    
                    all_prompts.append(prompt)
        
        stacked_prompts = torch.cat(all_prompts, dim=0)
        return stacked_prompts

    def _create_front_prompts(self, ctx_shifted):
        """
        Creates prompts with attributes after the class token in front position,
        while preserving image conditioning.
        """
        max_allowed_length = 77
        
        batch_size = ctx_shifted.size(0)
        all_prompts = []
        
        for b in range(batch_size):
            # Get image-specific context for this batch item
            ctx_image = ctx_shifted[b]  # Image-conditioned context
            
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :, :]
                class_i = self.token_suffix[i:i+1, :name_len, :]
                suffix_i = self.token_suffix[i:i+1, name_len:, :]
                
                # Expand context for this class
                ctx_i = ctx_image.unsqueeze(0)
                
                for j in range(self.attribute_embeddings.n_attrs):
                    attr_j = self.attribute_embeddings.get_attribute_embeddings(i, j)
                    attr_j = attr_j.unsqueeze(0)
                    
                    # Calculate available space for attribute
                    fixed_tokens = 1 + name_len + ctx_i.size(1) + 1
                    avail_attr_tokens = max_allowed_length - fixed_tokens
                    
                    if attr_j.size(1) > avail_attr_tokens and avail_attr_tokens > 0:
                        attr_j = attr_j[:, :avail_attr_tokens, :]
                    
                    # Construct prompt: [SOS][Class][CTX][Attribute][EOS]
                    prompt = torch.cat([
                        prefix_i,     # SOS
                        class_i,      # Class
                        ctx_i,        # Context (image-conditioned)
                        attr_j,       # Attribute
                        suffix_i[:, -1:, :]  # EOS
                    ], dim=1)
                    
                    if prompt.size(1) > max_allowed_length:
                        prompt = prompt[:, :max_allowed_length, :]
                    
                    all_prompts.append(prompt)
        
        stacked_prompts = torch.cat(all_prompts, dim=0)
        return stacked_prompts


class AGCoCoOpCustomCLIP(CustomCLIP):
    """
    Extends CoCoOp's CustomCLIP to support both image-conditioned and
    attribute-enhanced prompts with margin-based contrastive learning.
    """
    def __init__(self, cfg, classnames, clip_model, attributes):
        # Initialize parent class first (sets up CLIP model components)
        super().__init__(cfg, classnames, clip_model)
        
        # Replace the prompt learner with our enhanced version that handles attributes
        self.prompt_learner = EnhancedCoCoOpPromptLearner(cfg, classnames, clip_model, attributes)
        
        # Store class-related dimensions for loss computation
        self.n_cls = len(classnames)
        self.n_attrs = cfg.DATASET.NUM_ATTRIBUTES
        
        # Store the dtype from CLIP model
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        """
        Forward pass that handles both original CoCoOp behavior and
        attribute-enhanced prompts for training.
        
        Args:
            image: Input images
            label: Ground truth labels (None during inference)
            
        Returns:
            During training: Dict with logits and features for loss computation
            During inference: Original CoCoOp logits
        """
        # Get tokenized prompts for the original prompts
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Get logit scale to be applied to the similarities
        logit_scale = self.logit_scale.exp()
        
        # Encode images with CLIP's image encoder
        image_features = self.image_encoder(image.type(self.dtype))
        
        # Normalize image features for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Get both original and attribute-enhanced prompts from our prompt learner
        original_prompts, attr_prompts = self.prompt_learner(image_features)
        
        # Process original CoCoOp prompts (same as in CoCoOp)
        # For each image, we use its image-conditioned prompt for all classes
        logits = []
        for pts_i, imf_i in zip(original_prompts, image_features):
            # Encode the text prompts
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            
            # Normalize text features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute logits for this image
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        # During inference or evaluation, return only the original logits
        # We detect test mode by checking if we're training and if labels are provided
        if not self.prompt_learner.training or label is None:
            return logits
        
        # For training, we also need to process attribute-enhanced prompts
        # Create extended tokenized prompts for all attributes
        batch_size = image_features.size(0)
        
        # We need to create tokenized prompts for all attribute-enhanced prompts
        # Each class has n_attrs attributes, and we create prompts for each batch item
        attr_tokenized = tokenized_prompts.repeat_interleave(self.n_attrs, dim=0)
        attr_tokenized = attr_tokenized.repeat(batch_size, 1)
        
        # Encode all attribute-enhanced prompts
        text_features_attr = self.text_encoder(attr_prompts, attr_tokenized)
        
        # Normalize attribute text features
        text_features_attr = text_features_attr / text_features_attr.norm(dim=-1, keepdim=True)
        
        # Return all components needed for computing the losses
        return {
            'logits_orig': logits,  # Original CoCoOp logits for classification loss
            'image_features': image_features,  # For contrastive loss
            'text_features_attr': text_features_attr,  # Attribute-enhanced text features
            'logit_scale': logit_scale,  # For scaling similarities
            'batch_size': batch_size,  # For organizing the attribute features
            'n_cls': self.n_cls,  # Number of classes
            'n_attrs': self.n_attrs  # Number of attributes per class
        }

    def get_text_features(self, prompts, tokenized_prompts):
        """
        Helper method to get text features using CLIP's text encoder.
        Useful for computing features for different prompt variations.
        """
        text_features = self.text_encoder(prompts, tokenized_prompts)
        return text_features / text_features.norm(dim=-1, keepdim=True)    


@TRAINER_REGISTRY.register()
class AGCoCoOpTrainer(CoCoOp):
    """
    Attribute-Guided CoCoOp trainer that extends CoCoOp with
    attribute-based contrastive learning capabilities and margin loss.
    """
    def check_cfg(self, cfg):
        # First run CoCoOp's configuration checks
        super().check_cfg(cfg)
        
        # Add AGCoCoOp-specific configuration checks
        assert cfg.TRAINER.AGCOCOOP.TEMPERATURE_CONTRAST > 0, "Contrastive temperature must be positive"
        assert cfg.TRAINER.AGCOCOOP.TEMPERATURE_MARGIN > 0, "Margin temperature must be positive"
        assert cfg.TRAINER.AGCOCOOP.CONTRAST_WEIGHT >= 0, "Contrastive weight must be non-negative"
        assert cfg.TRAINER.AGCOCOOP.MARGIN_WEIGHT >= 0, "Margin weight must be non-negative"
        assert cfg.TRAINER.AGCOCOOP.MARGIN >= 0, "Margin must be non-negative"
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

        # Load CLIP model
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            clip_model.float()

        # Build AGCoCoOpCustomCLIP model
        print("Building attribute-enhanced CoCoOp model")
        self.model = AGCoCoOpCustomCLIP(cfg, classnames, clip_model, attributes)

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

        # Setup model and optimization
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        # Setup mixed precision training
        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Handle multi-GPU setup
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
        # Initialize wandb
        try:
            if not hasattr(cfg.TRAINER, 'NO_WANDB') or not cfg.TRAINER.NO_WANDB:
                wandb.init(
                    project="attribute-guided-cocoop",
                    name=f"{cfg.DATASET.NAME}_{cfg.MODEL.BACKBONE.NAME}_shots{cfg.DATASET.NUM_SHOTS}_seed{cfg.SEED}",
                    config={
                        "dataset": cfg.DATASET.NAME, 
                        "num_shots": cfg.DATASET.NUM_SHOTS,
                        "backbone": cfg.MODEL.BACKBONE.NAME,
                        "contrast_weight": cfg.TRAINER.AGCOCOOP.CONTRAST_WEIGHT,
                        "margin_weight": cfg.TRAINER.AGCOCOOP.MARGIN_WEIGHT,
                        "temperature_contrast": cfg.TRAINER.AGCOCOOP.TEMPERATURE_CONTRAST,
                        "temperature_margin": cfg.TRAINER.AGCOCOOP.TEMPERATURE_MARGIN,
                        "margin": cfg.TRAINER.AGCOCOOP.MARGIN
                    }
                )
                self.use_wandb = True
            else:
                self.use_wandb = False
        except Exception as e:
            print(f"Wandb initialization error: {e}")
            self.use_wandb = False

    def multi_positive_contrastive_loss(self, output_dict, labels):
        """
        Implements multi-positive contrastive loss for attribute-guided learning,
        inspired by https://arxiv.org/pdf/2306.00984.pdf.
        
        This loss treats all attributes of the same class as positive examples.
        """
        # Extract features and normalize
        image_features = F.normalize(output_dict['image_features'], p=2, dim=1)
        text_features = F.normalize(output_dict['text_features_attr'], p=2, dim=1)
        
        # Get configuration parameters
        temperature = self.cfg.TRAINER.AGCOCOOP.TEMPERATURE_CONTRAST
        
        # Get dimensions for reshaping
        n_attrs = (self.model.module.n_attrs if isinstance(self.model, nn.DataParallel)
                else self.model.n_attrs)
        n_cls = (self.model.module.n_cls if isinstance(self.model, nn.DataParallel)
                else self.model.n_cls)
        batch_size = image_features.size(0)
        
        # Reshape text features to [batch_size, n_cls, n_attrs, dim]
        # First, reshape to separate batch items
        text_features = text_features.view(batch_size, n_cls, n_attrs, -1)
        
        # Compute similarity between each image and all class-attribute combinations
        # [batch_size, batch_size, n_cls, n_attrs]
        similarities = torch.matmul(
            image_features.unsqueeze(1), 
            text_features.view(batch_size, n_cls, n_attrs, -1).permute(0, 1, 3, 2)
        ) / temperature
        
        # Create mask to identify positive pairs
        # A positive pair is an image and any attribute from its ground truth class
        mask = torch.zeros(batch_size, batch_size, n_cls, n_attrs, device=image_features.device)
        for i in range(batch_size):
            # Get the true class of this image
            cls_idx = labels[i].item()
            # All attributes of this class are positives for this image
            mask[i, i, cls_idx, :] = 1.0
        
        # Reshape similarities for softmax across all class-attribute combinations
        flat_similarities = similarities.view(batch_size, batch_size * n_cls * n_attrs)
        
        # Reshape mask to match
        flat_mask = mask.view(batch_size, batch_size * n_cls * n_attrs)
        
        # Normalize mask to create a valid probability distribution
        flat_mask = flat_mask / flat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Compute log probabilities
        log_probs = F.log_softmax(flat_similarities, dim=1)
        
        # Compute cross-entropy loss (negative of the expected log probability)
        loss = -(flat_mask * log_probs).sum(dim=1).mean()
        
        return loss

    def margin_loss(self, output_dict, labels):
        """
        Implements a margin-based contrastive loss to enforce a minimum 
        separation between positive and negative pairs.
        """
        # Extract features and normalize
        image_features = F.normalize(output_dict['image_features'], p=2, dim=1)
        text_features = F.normalize(output_dict['text_features_attr'], p=2, dim=1)
        
        # Get configuration parameters
        temperature = self.cfg.TRAINER.AGCOCOOP.TEMPERATURE_MARGIN
        margin = self.cfg.TRAINER.AGCOCOOP.MARGIN
        
        # Get dimensions for reshaping
        n_attrs = (self.model.module.n_attrs if isinstance(self.model, nn.DataParallel)
                else self.model.n_attrs)
        n_cls = (self.model.module.n_cls if isinstance(self.model, nn.DataParallel)
                else self.model.n_cls)
        batch_size = image_features.size(0)
        
        # Compute similarity between each image and all text features
        # [batch_size, batch_size * n_cls * n_attrs]
        text_features_reshaped = text_features.view(batch_size, n_cls, n_attrs, -1)
        
        # For each image and each attribute of its true class,
        # we want to ensure it's margin higher than any other class's attributes
        margin_violations = []
        
        for i in range(batch_size):
            # Get the true class for this image
            cls_idx = labels[i].item()
            
            # Get all positive attributes for this class
            positive_attrs = text_features_reshaped[i, cls_idx]
            
            # Compute similarity with this image
            image_feat = image_features[i].unsqueeze(0)  # [1, dim]
            pos_similarities = (image_feat @ positive_attrs.transpose(0, 1)).squeeze(0) / temperature
            
            # For each positive attribute
            for a in range(n_attrs):
                pos_sim = pos_similarities[a]
                
                # Compute similarities with all negative attributes (from other classes)
                neg_similarities = []
                for c in range(n_cls):
                    if c == cls_idx:
                        continue  # Skip the true class
                    
                    # Get attributes for this negative class
                    neg_attrs = text_features_reshaped[i, c]
                    neg_sims = (image_feat @ neg_attrs.transpose(0, 1)).squeeze(0) / temperature
                    neg_similarities.append(neg_sims)
                
                # Stack all negative similarities
                if neg_similarities:
                    neg_similarities = torch.cat(neg_similarities)
                    
                    # For each positive similarity, find the hardest negative
                    hardest_neg_sim = neg_similarities.max()
                    
                    # Compute margin violation: max(0, margin - (pos_sim - neg_sim))
                    violation = F.relu(margin - (pos_sim - hardest_neg_sim))
                    margin_violations.append(violation)
        
        # If no violations were found, return zero loss
        if not margin_violations:
            return torch.tensor(0.0, device=image_features.device)
        
        # Average margin violations across all positive pairs
        margin_loss = torch.stack(margin_violations).mean()
        
        return margin_loss

    def compute_loss(self, output_dict, label):
        """
        Computes combined loss from classification, multi-positive contrastive,
        and margin losses.
        """
        # Original CoCoOp classification loss
        orig_logits = output_dict['logits_orig']
        loss_cocoop = F.cross_entropy(orig_logits, label)

        # Add multi-positive contrastive loss
        cont_loss = self.multi_positive_contrastive_loss(output_dict, label)
        
        # Add margin loss
        marg_loss = self.margin_loss(output_dict, label)
        
        # Combine losses with their respective weights
        total_loss = loss_cocoop + \
                    self.cfg.TRAINER.AGCOCOOP.CONTRAST_WEIGHT * cont_loss + \
                    self.cfg.TRAINER.AGCOCOOP.MARGIN_WEIGHT * marg_loss
        
        # Store individual losses for logging
        self.loss_summary = {
            "loss_cocoop": loss_cocoop.item(),
            "loss_cont": cont_loss.item(),
            "loss_margin": marg_loss.item()
        }
        
        return total_loss

    def forward_backward(self, batch):
        """
        Performs forward and backward passes, inheriting CoCoOp's precision handling
        while adding our attribute-guided learning with margin loss.
        """
        image, label = self.parse_batch_train(batch)
        
        # Handle different precision modes
        if self.cfg.TRAINER.COCOOP.PREC == "amp":
            with autocast():
                output_dict = self.model(image, label)
                loss = self.compute_loss(output_dict, label)
            
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_dict = self.model(image, label)
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
            if "loss_cocoop" in loss_summary:
                wandb_log["loss/cocoop"] = loss_summary["loss_cocoop"]
            if "loss_cont" in loss_summary:
                wandb_log["loss/contrastive"] = loss_summary["loss_cont"]
            if "loss_margin" in loss_summary:
                wandb_log["loss/margin"] = loss_summary["loss_margin"]
            wandb.log(wandb_log)
            
        return loss_summary
