from src import (
    analyze_model_list, 
    save_analysis_results,
    build_model_network,
    visualize_model_network,
    analyze_network_metrics,
    print_network_summary,
    export_network_csv,
    HfApi
)
import json
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
    
    # Analyze multiple models (brief summary)
    print("\n📋 Example 2: Multiple Models Analysis (Brief)")
    print("-" * 50)
    
    list_result = analyze_model_list(
        hf_api=hf_api,
        pipeline_tag=["text-generation"],
        sort="downloads",
        limit=50000,
        verbose=False  # Brief summary for each model
    )
    
    if list_result['success']:
        # Optionally save the results
        save_analysis_results(list_result, "multi_model_analysis")
        
        # Show additional insights
        print(f"\n🔍 Additional Insights:")
        df = list_result['dataframe']
        
        # Find models with dependencies
        dependent_models = df[df['base_models_count'] > 0]
        if not dependent_models.empty:
            print(f"📊 Models with dependencies: {len(dependent_models)}")
            for _, model in dependent_models.iterrows():
                print(f"  🔗 {model['id']} ({model['model_type']}) -> {', '.join(model['base_models'][:2])}")
        
        # Show model type distribution
        type_dist = list_result['type_distribution']
        print(f"\n📈 Model Type Summary:")
        for model_type, count in type_dist.items():
            if count > 0:
                print(f"  {model_type.upper()}: {count} models")
        
        # Build and analyze network graph
        print("\n📋 Network Graph Analysis")
        print("-" * 50)
        
        # Create the model network
        print("🔗 Building model dependency network...")
        G = build_model_network(list_result, include_isolated=False, propagate_metadata_flag=True)
        
        # Print network summary
        print_network_summary(G)
        
        # Analyze network metrics
        metrics = analyze_network_metrics(G)
        
        # Save network data
        print(f"\n💾 Saving network data...")

        # Save network data in CSV format (edges.csv + metadata.csv)
        export_network_csv(G)

        if False:
            # Visualize the network (this will save and display the graph)
            print(f"\n📊 Creating network visualization...")
            visualize_model_network()
        
        # Save network metrics
        with open("network_data/network_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"💾 Saved metrics: network_data/network_metrics.json")
        
        # Show some interesting insights
        if 'pagerank' in metrics.get('centrality', {}):
            pagerank = metrics['centrality']['pagerank']
            top_influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\n🌟 Most Influential Models (PageRank):")
            for i, (model, score) in enumerate(top_influential, 1):
                model_short = model.split('/')[-1]
                print(f"  {i}. {model_short}: {score:.4f}")
    
    print("\n✅ Analysis Complete!")


if __name__ == "__main__":
    main() 