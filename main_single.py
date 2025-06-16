from src import (
    analyze_single_model, 
    HfApi
)
import os
from dotenv import load_dotenv
load_dotenv()


def main():
    # Initialize HuggingFace API
    hf_api = HfApi(
        endpoint="https://huggingface.co",
        token=os.getenv("HF_TOKEN")
    )
    
    print("🤖 HuggingFace Model Analysis Demo")
    print("=" * 50)
    
    # Analyze a single model (verbose)
    print("\n📋 Example 1: Single Model Analysis (Verbose)")
    print("-" * 50)
    
    single_result = analyze_single_model(
        model_name="THU-KEG/Llama3-Crab-SFT",
        hf_api=hf_api,
        verbose=True
    )
    
    # if single_result['success']:
    #     # Optionally save single model results
    #     save_analysis_results(single_result, "single_model_analysis")
   

    print("\n✅ Analysis Complete!")


if __name__ == "__main__":
    main() 