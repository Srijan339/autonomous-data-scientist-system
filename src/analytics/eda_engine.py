import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class EDAEngine:

    def __init__(self, df, output_folder="reports", target_column="Score"):
        self.df = df
        self.output_folder = output_folder
        self.target_column = target_column
        self.summary_data = {}

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def correlation_heatmap(self):

        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")

        path = f"{self.output_folder}/correlation_heatmap.png"

        plt.savefig(path)
        plt.close()

        print(f"Saved: {path}")

    def histograms(self):

        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:

            plt.figure()

            sns.histplot(self.df[col], kde=True)

            plt.title(f"Distribution of {col}")

            path = f"{self.output_folder}/hist_{col}.png"

            plt.savefig(path)
            plt.close()

            print(f"Saved: {path}")

    def boxplots(self):

        numeric_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in numeric_cols:

            plt.figure()
            
            # Drop NaN values for boxplot
            data_clean = self.df[col].dropna()
            
            if len(data_clean) > 0:
                plt.boxplot(data_clean)
                plt.title(f"Boxplot of {col}")
                plt.ylabel(col)

                path = f"{self.output_folder}/boxplot_{col}.png"

                plt.savefig(path)
                plt.close()

                print(f"Saved: {path}")
            else:
                plt.close()
                print(f"Skipped {col}: No valid data")

    def get_top_correlations(self, top_n=5):
        """Get top correlations with target column"""
        if self.target_column not in self.df.columns:
            return {}
        
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])
        correlations = numeric_df.corr()[self.target_column].sort_values(ascending=False)
        
        # Exclude the target column itself
        correlations = correlations[correlations.index != self.target_column]
        
        return correlations.head(top_n)

    def generate_summary_report(self):
        """Generate text summary report"""
        # Only select numeric columns
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        
        # Filter out columns with all NaN values
        numeric_cols = [col for col in numeric_cols if self.df[col].notna().sum() > 0]
        
        summary = []
        summary.append("=" * 60)
        summary.append("EDA SUMMARY REPORT")
        summary.append("=" * 60)
        summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Dataset Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        
        # Top Correlations
        summary.append(f"\nTOP CORRELATIONS WITH {self.target_column}")
        summary.append("-" * 40)
        
        top_corrs = self.get_top_correlations(top_n=5)
        for col, corr_value in top_corrs.items():
            summary.append(f"  {col} -> {corr_value:.2f}")
        
        # Column Statistics (only for numeric columns with valid data)
        summary.append(f"\nNUMERIC COLUMNS STATISTICS")
        summary.append("-" * 40)
        for col in numeric_cols:
            summary.append(f"\n{col}:")
            summary.append(f"  Mean: {self.df[col].mean():.2f}")
            summary.append(f"  Std Dev: {self.df[col].std():.2f}")
            summary.append(f"  Min: {self.df[col].min():.2f}")
            summary.append(f"  Max: {self.df[col].max():.2f}")
            summary.append(f"  Missing: {self.df[col].isna().sum()}")
        
        summary.append("\n" + "=" * 60)
        
        # Save summary with UTF-8 encoding
        report_path = f"{self.output_folder}/EDA_SUMMARY.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
        
        print(f"\n✅ Summary saved: {report_path}")
        
        # Print summary
        print("\n".join(summary))
        
        self.summary_data = {"correlations": top_corrs.to_dict()}

    def run_full_eda(self):

        print("\n📊 Running Automated EDA...\n")

        self.correlation_heatmap()

        self.histograms()

        self.boxplots()

        self.generate_summary_report()

        print("\nEDA Completed. Reports saved in 'reports/' folder.")