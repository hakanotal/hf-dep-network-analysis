# Exploring Hugging Face LLM Metadata 

The primary focus of the project centered around the metadata analysis of open-source Large Language Models (LLMs) available on Hugging Face. The core methodology involves constructing and analyzing graph networks to understand the relationships and evolution of these models. 

The initial concept involves mapping connections between models based on their lineage: 

    Base models: Foundational models upon which others are built. 

    Fine-tuned models: Versions of base models adapted for specific tasks or datasets. 

    Quantized models: Versions of models optimized for size and computational efficiency. 

1.1. Key Discussion Points & Advisor Feedback 

Several avenues for enriching the analysis by incorporating more detailed metadata were discussed: 

    Model Origin & Language: 

    Incorporating data on the country of origin of the model or the developing institution. 

    Analyzing the primary language(s) the model is designed for or trained on. 

    Community Engagement & Reputation: 

    Filtering or weighting models based on community engagement metrics like download counts and likes. This could potentially be used to develop a "reputation score" or "impact score" for models or contributors. 

    Vendor/Community Analysis: 

    Exploring the ecosystems around specific vendors, research labs, or individual contributors, mapping their model release patterns and interconnections within their own model families. 

    Temporal Dynamics: 

    Leveraging model upload times to create temporal graphs, allowing for the study of how the LLM landscape evolves, how quickly fine-tuning or quantization occurs after a base model release, and the lifespan or popularity trends of models over time. 

1.2. Additional Ideas for Project Expansion 

Building upon the advisor's suggestions, the following additional dimensions could further enhance the research: 

    License Analysis: 

    Tracking the types of licenses associated with models (e.g., Apache 2.0, MIT, GPL, OpenRAIL). 

    Analyzing the propagation of certain license types across model families or communities. 

    Investigating potential correlations between license type and model popularity, downstream adoption, or modification. 

    Task & Application Focus: 

    Categorizing models by their intended tasks (e.g., text generation, translation, summarization, sentiment analysis) based on metadata tags or model card descriptions. 

    Mapping how different base models are adapted for various tasks and identifying trends in task-specific fine-tuning. 

    Model Architecture & Parameter Insights: 

    Where available, incorporating metadata about model architecture (e.g., Transformer, Llama, T5) and number of parameters. 

    Analyzing the evolution of model sizes and architectures over time within the graph. 

    Dataset Linkages (if available): 

    Identifying if metadata links to specific datasets used for pre-training or fine-tuning. 

    Mapping the influence of popular datasets on model development. 

    Fine-tuning Technique Analysis: 

    If discernible from metadata or model cards, identifying common fine-tuning techniques (e.g., LoRA, QLoRA, full fine-tuning) and their prevalence. 

    Cross-lingual Transfer & Multilinguality: 

    Investigating the relationships between multilingual models and their monolingual fine-tuned versions, or how models are adapted for new languages. 

    Ethical Considerations & Bias Documentation: 

    Examining how often and in what detail model cards discuss ethical considerations, limitations, and potential biases. This might be a qualitative overlay on the quantitative graph analysis. 

## 🚀 Features

- **🔍 Comprehensive Model Metadata Extraction** - Extracts all available metadata using Context7 documentation
- **🤖 Advanced Model Type Detection** - Automatically detects:
  - 🔵 **Base Models** - Original foundation models
  - 🎯 **Finetuned Models** - Models trained on specific datasets
  - ⚡ **Quantized Models** - Compressed versions of base models
  - 🔗 **Merged Models** - Combinations of multiple models
  - 🔧 **Adapter Models** - Parameter-efficient fine-tuning adaptations
- **📊 Batch Analysis** - Analyze multiple models with intelligent filtering
- **💾 Export Capabilities** - Save results as CSV and JSON
- **🎛️ Flexible Verbosity** - Control output detail level
- **📈 Statistical Insights** - Model type distribution and popularity analysis
- **🔗 Network Graph Analysis** - Build and visualize model dependency networks
- **📊 Interactive Visualizations** - Multiple layout algorithms and export formats

## 📦 Installation

```bash
uv run main_single.py
uv run main_multi.py
```

## 📁 File Structure

```
├── hf_models.py          # Main library with all functions
├── main.py              # Comprehensive usage examples  
├── network_example.py   # Network analysis examples
├── README.md           # This documentation
└── model_data/         # Output directory for analysis files
    ├── *.csv           # Model metadata and edge lists
    ├── *.json          # Raw data and network structures
    ├── *.png           # Network visualizations
    └── *.graphml       # Network data for external tools
```


## Visualization

- [CosmoGraph](https://cosmograph.app/run/?data=https://raw.githubusercontent.com/hakanotal/hf-dep-network-analysis/refs/heads/main/network_data/edges.csv&meta=https://raw.githubusercontent.com/hakanotal/hf-dep-network-analysis/refs/heads/main/network_data/metadata.csv&source=source&target=target&gravity=0.25&repulsion=1&repulsionTheta=1.15&linkSpring=1&linkDistance=10&friction=0.85&renderLabels=true&renderHoveredLabel=true&renderLinks=true&curvedLinks=true&nodeSizeScale=1&linkWidthScale=2&linkArrowsSizeScale=1&nodeSize=size-total%20links&nodeColor=color-type&linkWidth=width-default&linkColor=color-default&)