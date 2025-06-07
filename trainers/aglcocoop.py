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
        
        if torch.cuda.is_available():
            print(f"Using GPU via CUDA_VISIBLE_DEVICES")
        # Handle multi-GPU setup
        
        # if torch.cuda.device_count() > 1:
        #     print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            
        #     self.model = nn.DataParallel(self.model)
    
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
        adapted to work with accumulated features from multiple batches.
        
        This version works with a larger effective batch size through gradient accumulation
        and removes hard negative mining, relying on batch diversity instead.
        """
        # Extract features and normalize
        image_features = F.normalize(output_dict['image_features'], p=2, dim=1)
        text_features = F.normalize(output_dict['text_features_attr'], p=2, dim=1)
        
        # Get configuration parameters
        temperature = self.cfg.TRAINER.AGCOCOOP.TEMPERATURE_CONTRAST
        
        # Get dimensions for reshaping
        n_attrs = self.model.module.n_attrs if isinstance(self.model, nn.DataParallel) else self.model.n_attrs
        n_cls = self.model.module.n_cls if isinstance(self.model, nn.DataParallel) else self.model.n_cls
        batch_size = image_features.size(0)  # This is now the accumulated batch size
        
        # Calculate total number of text features per batch item
        text_features_per_item = n_cls * n_attrs
        
        # Reshape text features to ensure they align with the batch structure
        # Each batch item has text features for all classes and all attributes
        text_features = text_features.view(batch_size, text_features_per_item, -1)
        
        # Compute similarity matrix between all images and all text features
        # [batch_size, batch_size * text_features_per_item]
        logits = torch.matmul(
            image_features, 
            text_features.view(batch_size * text_features_per_item, -1).t()
        ) / temperature
        
        # Create a mask indicating which text features are positive for each image
        pos_mask = torch.zeros(batch_size, batch_size * text_features_per_item, 
                            device=image_features.device)
        
        # For each image, mark its true class attributes as positives
        for i in range(batch_size):
            # Get the ground truth class for this image
            label = labels[i].item()
            
            # Calculate which text features correspond to this image
            start_idx = i * text_features_per_item
            
            # Mark all attributes of the correct class as positives
            for j in range(n_attrs):
                attr_idx = start_idx + (label * n_attrs) + j
                pos_mask[i, attr_idx] = 1.0
        
        # Add small positive value to all positions to avoid numerical issues
        # This can help stabilize learning by preventing log(0) errors
        eps = 1e-6
        pos_mask = pos_mask + eps
        
        # Normalize the mask to create a valid probability distribution for each image
        pos_mask = pos_mask / pos_mask.sum(dim=1, keepdim=True)
        
        # Compute log softmax along the text feature dimension
        log_softmax = F.log_softmax(logits, dim=1)
        
        # Compute the negative log likelihood (cross-entropy) loss
        # This maximizes the likelihood of the positive pairs
        loss = -(pos_mask * log_softmax).sum(dim=1).mean()
        
        return loss

    def margin_loss(self, output_dict, labels):
        """
        Implements a batch-based margin loss for accumulated features.
        This version works with larger effective batch sizes and doesn't require
        explicit hard negative mining.
        """
        # Extract features and normalize
        image_features = F.normalize(output_dict['image_features'], p=2, dim=1)
        text_features = F.normalize(output_dict['text_features_attr'], p=2, dim=1)
        
        # Get configuration parameters
        temperature = self.cfg.TRAINER.AGCOCOOP.TEMPERATURE_MARGIN
        margin = self.cfg.TRAINER.AGCOCOOP.MARGIN
        
        # Get dimensions for reshaping
        n_attrs = self.model.module.n_attrs if isinstance(self.model, nn.DataParallel) else self.model.n_attrs
        n_cls = self.model.module.n_cls if isinstance(self.model, nn.DataParallel) else self.model.n_cls
        batch_size = image_features.size(0)
        
        # Calculate total number of text features per batch item
        text_features_per_item = n_cls * n_attrs
        
        # Reshape text features to ensure they align with the batch structure
        text_features = text_features.view(batch_size, text_features_per_item, -1)
        
        # Compute all pairwise similarities in one go
        # [batch_size, batch_size * text_features_per_item]
        all_similarities = torch.matmul(
            image_features, 
            text_features.view(batch_size * text_features_per_item, -1).t()
        ) / temperature
        
        # Initialize loss accumulator
        margin_loss = 0.0
        num_positives = 0
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get the ground truth class for this image
            label = labels[i].item()
            
            # Find the feature indices for this batch item
            batch_start_idx = i * text_features_per_item
            
            # Get indices for positive attributes (same class)
            pos_indices = [batch_start_idx + (label * n_attrs) + j for j in range(n_attrs)]
            
            # Get all positive similarities for this image
            pos_similarities = all_similarities[i, pos_indices]
            
            # Create mask for negative attributes (all other classes)
            neg_mask = torch.ones(batch_size * text_features_per_item, 
                                device=image_features.device, dtype=torch.bool)
            
            # Mark positives as False in the negative mask
            for idx in pos_indices:
                if idx < neg_mask.size(0):  # Safety check
                    neg_mask[idx] = False
            
            # Get all negative similarities
            neg_similarities = all_similarities[i][neg_mask]
            
            if neg_similarities.numel() == 0:
                continue  # Skip if no negatives (unlikely but possible)
            
            # For each positive similarity, compute margin violations against all negatives
            for pos_sim in pos_similarities:
                # Find the violation against the hardest negative (highest similarity)
                # This is simpler and more efficient than computing all pairwise violations
                hardest_neg_sim = neg_similarities.max()
                violation = F.relu(margin - (pos_sim - hardest_neg_sim))
                
                if violation > 0:
                    margin_loss += violation
                    num_positives += 1
        
        # Return average margin violation (or zero if no positives)
        if num_positives > 0:
            return margin_loss / num_positives
        else:
            return torch.tensor(0.0, device=image_features.device)

    def forward_backward(self, batch):
        """
        Performs forward and backward passes with gradient accumulation over 8 images.
        """
        image, label = self.parse_batch_train(batch)
        
        # Initialize accumulation variables if not already done
        if not hasattr(self, 'accumulated_features'):
            self.accumulated_features = {
                'image_features': [],
                'text_features_attr': [],
                'logits_orig': []
            }
            self.accumulated_labels = []
            self.accumulated_count = 0
            self.accumulation_steps = 8  # Accumulate over 8 images
        
        # Handle different precision modes
        if self.cfg.TRAINER.COCOOP.PREC == "amp":
            with autocast():
                # Forward pass
                output_dict = self.model(image, label)
                
                # Store just the loss for CoCoOp (apply immediately)
                loss_cocoop = F.cross_entropy(output_dict['logits_orig'], label)
                
                # Store features for later contrastive/margin loss computation
                self.accumulated_features['image_features'].append(output_dict['image_features'].detach())
                self.accumulated_features['text_features_attr'].append(output_dict['text_features_attr'].detach())
                self.accumulated_features['logits_orig'].append(output_dict['logits_orig'].detach())
                self.accumulated_labels.append(label)
                self.accumulated_count += 1
                
                # Only compute contrastive losses after accumulating enough samples
                if self.accumulated_count >= self.accumulation_steps:
                    # Combine accumulated features
                    combined_dict = {
                        'image_features': torch.cat(self.accumulated_features['image_features'], dim=0),
                        'text_features_attr': torch.cat(self.accumulated_features['text_features_attr'], dim=0),
                        'logits_orig': torch.cat(self.accumulated_features['logits_orig'], dim=0),
                    }
                    combined_labels = torch.cat(self.accumulated_labels, dim=0)
                    
                    # Compute contrastive and margin losses on accumulated batch
                    cont_loss = self.multi_positive_contrastive_loss(combined_dict, combined_labels)
                    marg_loss = self.margin_loss(combined_dict, combined_labels)
                    
                    # Combine with the current CoCoOp loss
                    total_loss = loss_cocoop + \
                                self.cfg.TRAINER.AGCOCOOP.CONTRAST_WEIGHT * cont_loss + \
                                self.cfg.TRAINER.AGCOCOOP.MARGIN_WEIGHT * marg_loss
                    
                    # Store for logging
                    self.loss_summary = {
                        "loss_cocoop": loss_cocoop.item(),
                        "loss_cont": cont_loss.item(),
                        "loss_margin": marg_loss.item()
                    }
                else:
                    # Just use CoCoOp loss when still accumulating
                    total_loss = loss_cocoop
                    self.loss_summary = {
                        "loss_cocoop": loss_cocoop.item(),
                        "loss_cont": 0.0,
                        "loss_margin": 0.0
                    }
            
            # Backward pass and optimization
            self.optim.zero_grad()
            self.scaler.scale(total_loss).backward()
            
            # Only update weights after accumulating enough samples
            if self.accumulated_count >= self.accumulation_steps:
                self.scaler.step(self.optim)
                self.scaler.update()
                
                # Reset accumulation
                self.accumulated_features = {
                    'image_features': [],
                    'text_features_attr': [],
                    'logits_orig': []
                }
                self.accumulated_labels = []
                self.accumulated_count = 0
        else:
            # Same logic but without mixed precision
            output_dict = self.model(image, label)
            
            # Store just the loss for CoCoOp (apply immediately)
            loss_cocoop = F.cross_entropy(output_dict['logits_orig'], label)
            
            # Store features for later contrastive/margin loss computation
            self.accumulated_features['image_features'].append(output_dict['image_features'].detach())
            self.accumulated_features['text_features_attr'].append(output_dict['text_features_attr'].detach())
            self.accumulated_features['logits_orig'].append(output_dict['logits_orig'].detach())
            self.accumulated_labels.append(label)
            self.accumulated_count += 1
            
            # Only compute contrastive losses after accumulating enough samples
            if self.accumulated_count >= self.accumulation_steps:
                # Combine accumulated features
                combined_dict = {
                    'image_features': torch.cat(self.accumulated_features['image_features'], dim=0),
                    'text_features_attr': torch.cat(self.accumulated_features['text_features_attr'], dim=0),
                    'logits_orig': torch.cat(self.accumulated_features['logits_orig'], dim=0),
                }
                combined_labels = torch.cat(self.accumulated_labels, dim=0)
                
                # Compute contrastive and margin losses on accumulated batch
                cont_loss = self.multi_positive_contrastive_loss(combined_dict, combined_labels)
                marg_loss = self.margin_loss(combined_dict, combined_labels)
                
                # Combine with the current CoCoOp loss
                total_loss = loss_cocoop + \
                            self.cfg.TRAINER.AGCOCOOP.CONTRAST_WEIGHT * cont_loss + \
                            self.cfg.TRAINER.AGCOCOOP.MARGIN_WEIGHT * marg_loss
                
                # Store for logging
                self.loss_summary = {
                    "loss_cocoop": loss_cocoop.item(),
                    "loss_cont": cont_loss.item(),
                    "loss_margin": marg_loss.item()
                }
            else:
                # Just use CoCoOp loss when still accumulating
                total_loss = loss_cocoop
                self.loss_summary = {
                    "loss_cocoop": loss_cocoop.item(),
                    "loss_cont": 0.0,
                    "loss_margin": 0.0
                }
            
            # Backward pass
            self.model_backward_and_update(total_loss)
            
            # Reset accumulation if needed
            if self.accumulated_count >= self.accumulation_steps:
                self.accumulated_features = {
                    'image_features': [],
                    'text_features_attr': [],
                    'logits_orig': []
                }
                self.accumulated_labels = []
                self.accumulated_count = 0

        # Prepare loss summary for logging
        loss_summary = {
            "loss": total_loss.item(),
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