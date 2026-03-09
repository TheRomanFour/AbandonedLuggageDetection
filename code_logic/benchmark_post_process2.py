import pandas as pd
import os

def generate_scientific_report(csv_path="/mnt/SamsungSSD/IvanVrsalovic/Prtljaga/master_results_final_clean.csv"):
    """
    Processes the raw benchmark CSV and generates a summary table with 
    standard scientific performance metrics (F1, Recall, Precision, Accuracy).
    """
    # 1. Check if source file exists
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Please ensure the CSV is in the same folder.")
        return

    # 2. LOAD AND CLEAN DATA
    # skipinitialspace handles potential spaces after commas in CSV
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    
    # Identify the radius column (handles 'Avg_R' or 'Avg_Radius')
    radius_col = 'Avg_R' if 'Avg_R' in df.columns else 'Avg_Radius'
    
    # Remove duplicates to ensure clean statistics
    df = df.drop_duplicates(subset=['Method', 'Video'], keep='first')

    methods = df['Method'].unique()
    summary_stats = []

    # 3. CALCULATE METRICS PER METHOD
    for m in methods:
        m_df = df[df['Method'] == m]
        
        # Binary Classification components
        tp = m_df['Status'].str.contains('TP', na=False).sum()
        tn = m_df['Status'].str.contains('TN', na=False).sum()
        fp = m_df['Status'].str.contains('FP', na=False).sum()
        fn = m_df['Status'].str.contains('FN', na=False).sum()
        
        total = len(m_df)
        
        # Calculate Scientific Metrics 
        # (Using standard formulas with zero-division protection)
        accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_radius = m_df[radius_col].mean()

        summary_stats.append({
            "Method": m,
            "F1_Score(%)": round(f1, 2),
            "Recall(%)": round(recall, 2),
            "Precision(%)": round(precision, 2),
            "Accuracy(%)": round(accuracy, 2),
            "Avg_Radius(px)": round(avg_radius, 1),
            "TP": tp, 
            "FP": fp, 
            "TN": tn, 
            "FN": fn
        })

    # 4. EXPORT RESULTS
    # Create the DataFrame and sort by the primary scientific metric (F1-Score)
    summary_df = pd.DataFrame(summary_stats).sort_values(by="F1_Score(%)", ascending=False)
    
    output_name = "scientific_summary_report.csv"
    summary_df.to_csv(output_name, index=False)
    
    print("-" * 30)
    print(f"PROCESS COMPLETE")
    print(f"Source: {csv_path}")
    print(f"Output: {output_name}")
    print("-" * 30)
    
    # Show the table in the console for immediate review
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    generate_scientific_report()