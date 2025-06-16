"""
Constants and configuration values for HuggingFace model analysis.
"""

# Network visualization constants
DEFAULT_WIDTH = "100%"
DEFAULT_HEIGHT = "800px"
NETWORK_BGCOLOR = "#222222"
FONT_COLOR = "white"

# Node sizing constants
MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 50
NODE_SIZE_SCALE = 5

# String truncation
DEFAULT_MAX_LENGTH = 200
UNKNOWN = "unknown"

# Model type patterns
MODEL_TYPE_PATTERNS = {
    'quantized': r'base_model:quantized:(.+)',
    'finetune': r'base_model:finetune:(.+)',
    'merge': r'base_model:merge:(.+)',
    'adapter': r'base_model:adapter:(.+)',
    'generic_base': r'base_model:([^:]+)$'
}

# Color palettes
ARCH_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
               '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
               '#F8C471', '#82E0AA', '#F1948A', '#85929E', '#D2B4DE']

EDGE_COLORS = {
    'finetuned': '#e74c3c',
    'quantized': '#f39c12',
    'merged': '#9b59b6',
    'adapter': '#2ecc71',
    'derived': '#34495e',
    'unknown': '#95a5a6'
}

# Request timeout
REQUEST_TIMEOUT = 10
REQUEST_DELAY = 0.1 