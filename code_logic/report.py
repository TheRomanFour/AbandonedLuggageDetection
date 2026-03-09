import pandas as pd
import numpy as np
import os

def generate_benchmark_report(file_path='master_results_dynamic_v3.csv'):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() # Clean column names

    # Summary list
    summary_data = []

    # Get unique methods
    methods = df['Method'].unique()

    for method in methods:
        m_df = df[df['Method'] == method]
        
        # Count Confusion Matrix components
        tp = (m_df['Status'] == 'TP (Success)').sum()
        tn = (m_df['Status'] == 'TN (Correct Safe)').sum()
        fp = (m_df['Status'] == 'FP (False Alarm)').sum()
        fn = (m_df['Status'] == 'FN (Miss)').sum()
        
        total = len(m_df)
        
        # Calculate Scientific Metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_radius = m_df['Avg_Radius'].mean()
        
        summary_data.append({
            'Method': method,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1, 4),
            'Avg_Radius_PX': round(avg_radius, 2),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'Total_Videos': total
        })

    # Create Summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by F1 Score to show the best method first
    summary_df = summary_df.sort_values(by='F1_Score', ascending=False)
    
    # Save to CSV
    output_file = 'final_benchmark_analysis.csv'
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Acc':<8} | {'F1':<8} | {'Recall':<8} | {'Avg Rad':<10}")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['Method']:<20} | {row['Accuracy']:>7.2%} | {row['F1_Score']:>7.2%} | {row['Recall']:>7.2%} | {row['Avg_Radius_PX']:>8.1f}px")
    print("="*80)
    print(f"Full analysis saved to: {output_file}")

if __name__ == "__main__":
    generate_benchmark_report()