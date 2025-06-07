#!/bin/bash

# Default values
DATA_DIR="data"
ATTRIBUTES_DIR="attributes"
SAMPLED_DIR="sampled_attributes"
CLIP_MODEL="ViT-B/32"
NUM_CLUSTERS=3
FORCE=""

# Function to display usage
function display_usage {
    echo "Usage: $0 -k OPENAI_API_KEY [options]"
    echo ""
    echo "Required arguments:"
    echo "  -k, --api-key KEY     Your OpenAI API key"
    echo ""
    echo "Optional arguments:"
    echo "  -d, --data-dir DIR    Root directory containing datasets (default: data)"
    echo "  -o, --output-dir DIR  Directory to save generated attributes (default: attributes)"
    echo "  -s, --sampled-dir DIR Directory to save sampled attributes (default: sampled_attributes)"
    echo "  -c, --clip-model MODEL CLIP model to use (default: ViT-B/32)"
    echo "  -n, --num-clusters N  Number of clusters for attribute sampling (default: 3)"
    echo "  -D, --datasets LIST   Comma-separated list of datasets to process (default: all)"
    echo "  -g, --generation-only Only generate attributes, skip sampling"
    echo "  -S, --sampling-only   Only sample attributes, skip generation"
    echo "  -f, --force           Force regeneration of attributes even if they already exist"
    echo "  -h, --help            Display this help message"
    echo ""
    echo "Available datasets: imagenet, caltech-101, oxford_pets, stanford_cars, oxford_flowers,"
    echo "                   food-101, fgvc_aircraft, sun397, dtd, eurosat, ucf101"
    echo ""
    echo "Example: $0 -k sk-yourapikey123 -D oxford_flowers,dtd -n 5 -f"
}

# Parse command line arguments
POSITIONAL_ARGS=()
DATASETS_ARG=""
GENERATION_ONLY=false
SAMPLING_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--api-key)
            OPENAI_API_KEY="$2"
            shift
            shift
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift
            shift
            ;;
        -o|--output-dir)
            ATTRIBUTES_DIR="$2"
            shift
            shift
            ;;
        -s|--sampled-dir)
            SAMPLED_DIR="$2"
            shift
            shift
            ;;
        -c|--clip-model)
            CLIP_MODEL="$2"
            shift
            shift
            ;;
        -n|--num-clusters)
            NUM_CLUSTERS="$2"
            shift
            shift
            ;;
        -D|--datasets)
            DATASETS_ARG="$2"
            shift
            shift
            ;;
        -g|--generation-only)
            GENERATION_ONLY=true
            shift
            ;;
        -S|--sampling-only)
            SAMPLING_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE="--force"
            shift
            ;;
        -h|--help)
            display_usage
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            display_usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if API key is provided
if [ -z "$OPENAI_API_KEY" ] && [ "$SAMPLING_ONLY" = false ]; then
    echo "Error: OpenAI API key is required for attribute generation"
    display_usage
    exit 1
fi

# Create directories
mkdir -p "$ATTRIBUTES_DIR"
mkdir -p "$SAMPLED_DIR"

echo "===== ArGue Attribute Pipeline ====="
echo ""
echo "Data directory: $DATA_DIR"
echo "Attributes directory: $ATTRIBUTES_DIR"
echo "Sampled attributes directory: $SAMPLED_DIR"
echo ""

# Process datasets argument
if [ -n "$DATASETS_ARG" ]; then
    # Convert comma-separated list to space-separated list
    DATASETS_LIST=$(echo "$DATASETS_ARG" | tr ',' ' ')
else
    # Default: all datasets
    DATASETS_LIST="oxford_flowers oxford_pets food-101 dtd eurosat fgvc_aircraft stanford_cars sun397 caltech-101 ucf101"
fi

echo "Processing datasets: $DATASETS_LIST"
echo ""

# Step 1: Generate attributes
if [ "$SAMPLING_ONLY" = false ]; then
    echo "Step 1: Generating attributes for selected datasets..."
    
    DATASETS_ARGS=""
    for dataset in $DATASETS_LIST; do
        DATASETS_ARGS="$DATASETS_ARGS \"$dataset\""
    done
    
    # Use eval to properly handle the quoted dataset names
    eval "python /raid/biplab/souravr/thesis/multimodal/AGL/scripts/attribute/gen_attr.py \
      --data_dir \"$DATA_DIR\" \
      --output_dir \"$ATTRIBUTES_DIR\" \
      --openai_key \"$OPENAI_API_KEY\" \
      --datasets $DATASETS_ARGS \
      $FORCE"
    
    echo "Attribute generation completed!"
else
    echo "Skipping attribute generation (sampling only)..."
fi

echo ""

# Step 2: Sample attributes
if [ "$GENERATION_ONLY" = false ]; then
    echo "Step 2: Sampling attributes for selected datasets..."
    
    # Build list of attribute files to process
    ATTRIBUTE_FILES=""
    for dataset in $DATASETS_LIST; do
        if [ -f "$ATTRIBUTES_DIR/${dataset}_attributes.json" ]; then
            ATTRIBUTE_FILES="$ATTRIBUTE_FILES \"$ATTRIBUTES_DIR/${dataset}_attributes.json\""
        else
            echo "Warning: Attribute file for $dataset not found, skipping sampling"
        fi
    done
    
    if [ -n "$ATTRIBUTE_FILES" ]; then
        # Use eval to properly handle the quoted file paths
        eval "python /raid/biplab/souravr/thesis/multimodal/AGL/scripts/attribute/sample_attr.py \
          --attributes_files $ATTRIBUTE_FILES \
          --data_dir \"$DATA_DIR\" \
          --output_dir \"$SAMPLED_DIR\" \
          --clip_model \"$CLIP_MODEL\" \
          --num_clusters $NUM_CLUSTERS"
        
        echo "Attribute sampling completed!"
    else
        echo "No attribute files found to sample from!"
    fi
else
    echo "Skipping attribute sampling (generation only)..."
fi

echo ""
echo "Pipeline completed!"
echo "Generated attributes are in: $ATTRIBUTES_DIR"
echo "Sampled attributes are in: $SAMPLED_DIR"