import json
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentTracker:
    """
    A simple experiment tracking system for mental manipulation detection dialogue_models.
    Tracks performance metrics, hyperparameters, and model configurations.
    """

    def __init__(self, base_dir="./dialogue_experiment_tracking"):
        """Initialize the experiment tracker with a base directory for storing results."""
        self.base_dir = base_dir
        self.results_file = os.path.join(base_dir, "experiment_results.json")

        # Create the base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        # Initialize or load existing results
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []

    def log_experiment(self, model_name, level, hyperparams, metrics, notes=""):
        """
        Log a new experiment with its hyperparameters and performance metrics.

        Args:
            model_name (str): Name of the model architecture
            level (str): 'sentence' or 'dialogue'
            hyperparams (dict): Dictionary of hyperparameters
            metrics (dict): Dictionary of performance metrics
            notes (str): Any additional notes about the experiment
        """
        experiment = {
            "id": len(self.results) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "level": level,
            "hyperparameters": hyperparams,
            "metrics": metrics,
            "notes": notes
        }

        self.results.append(experiment)
        self._save_results()

        # Print a confirmation with the experiment ID
        print(f"Experiment #{experiment['id']} logged successfully.")
        return experiment["id"]

    def _save_results(self):
        """Save all results to the JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_experiment(self, experiment_id):
        """Retrieve a specific experiment by ID."""
        for exp in self.results:
            if exp["id"] == experiment_id:
                return exp
        return None

    def get_all_experiments(self):
        """Retrieve all experiments."""
        return self.results

    def get_experiments_by_level(self, level):
        """Retrieve all experiments for a specific level (sentence or dialogue)."""
        return [exp for exp in self.results if exp["level"] == level]

    def get_best_experiment(self, metric="f1", level=None):
        """Get the best experiment based on a specific metric."""
        if not self.results:
            return None

        filtered_results = self.results
        if level:
            filtered_results = [exp for exp in filtered_results if exp["level"] == level]

        # Sort by the specified metric (descending)
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x["metrics"].get(metric, 0),
            reverse=True
        )

        return sorted_results[0] if sorted_results else None

    def to_dataframe(self):
        """Convert results to a pandas DataFrame for easier analysis."""
        if not self.results:
            return pd.DataFrame()

        # Flatten the nested dictionaries
        flattened_data = []
        for exp in self.results:
            flat_exp = {
                "id": exp["id"],
                "timestamp": exp["timestamp"],
                "model_name": exp["model_name"],
                "level": exp["level"],
                "notes": exp["notes"]
            }

            # Add hyperparameters with 'hp_' prefix
            for hp_key, hp_value in exp["hyperparameters"].items():
                flat_exp[f"hp_{hp_key}"] = hp_value

            # Add metrics with 'metric_' prefix
            for metric_key, metric_value in exp["metrics"].items():
                flat_exp[f"metric_{metric_key}"] = metric_value

            flattened_data.append(flat_exp)

        return pd.DataFrame(flattened_data)

    def visualize_results(self, metric="f1", by_level=True, save_path=None, include_carbon=False):
        """
        Visualize the results of experiments using the specified metric.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No experiments to visualize.")
            return

        metric_col = f"metric_{metric}"
        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found in experiments.")
            return

        # Check how many unique levels exist
        unique_levels = df["level"].unique()

        # Create appropriate figure layout
        if include_carbon and "metric_total_emissions_kg" in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

        # Create the main bar plot
        if by_level and len(unique_levels) > 1:
            sns.barplot(x="id", y=metric_col, hue="level", data=df, ax=ax1)
            ax1.set_title(f"Comparison of {metric.upper()} Across Experiments by Level")
        else:
            sns.barplot(x="id", y=metric_col, data=df, ax=ax1)
            if by_level and len(unique_levels) == 1:
                ax1.set_title(f"Comparison of {metric.upper()} Across Experiments ({unique_levels[0].title()})")
            else:
                ax1.set_title(f"Comparison of {metric.upper()} Across All Experiments")

        ax1.set_xlabel("Experiment ID")
        ax1.set_ylabel(f"{metric.upper()} Score")
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on top of the bars
        for i, p in enumerate(ax1.patches):
            ax1.annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, rotation=0)

        # Add carbon emissions plot if requested and available
        if include_carbon and "metric_total_emissions_kg" in df.columns:
            carbon_col = "metric_total_emissions_kg"
            if by_level and len(unique_levels) > 1:
                sns.barplot(x="id", y=carbon_col, hue="level", data=df, ax=ax2)
            else:
                sns.barplot(x="id", y=carbon_col, data=df, ax=ax2)

            ax2.set_title("Carbon Emissions (kg CO2) by Experiment")
            ax2.set_ylabel("CO2 Emissions (kg)")
            ax2.set_xlabel("Experiment ID")
            ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")


        plt.close()  # Close the figure to free memory

    def compare_hyperparams(self, metric="f1", hyperparam="learning_rate", level=None):
        """
        Visualize the effect of a specific hyperparameter on model performance.

        Args:
            metric (str): The metric to visualize
            hyperparam (str): The hyperparameter to analyze
            level (str, optional): Filter by sentence/dialogue level
        """
        df = self.to_dataframe()
        if df.empty:
            print("No experiments to compare.")
            return

        metric_col = f"metric_{metric}"
        hyperparam_col = f"hp_{hyperparam}"

        if metric_col not in df.columns:
            print(f"Metric '{metric}' not found in experiments.")
            return

        if hyperparam_col not in df.columns:
            print(f"Hyperparameter '{hyperparam}' not found in experiments.")
            return

        if level:
            df = df[df["level"] == level]
            if df.empty:
                print(f"No experiments found for level '{level}'.")
                return

        plt.figure(figsize=(10, 6))

        # Sort by hyperparameter value
        df = df.sort_values(by=hyperparam_col)

        # Plot the hyperparameter vs metric relationship
        ax = sns.lineplot(x=hyperparam_col, y=metric_col, marker='o', data=df)

        # Add model name as labels for each point
        for i, row in df.iterrows():
            ax.annotate(row["model_name"],
                        (row[hyperparam_col], row[metric_col]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)

        plt.title(f"Effect of {hyperparam} on {metric.upper()}")
        plt.xlabel(hyperparam)
        plt.ylabel(f"{metric.upper()} Score")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.show()

    def generate_report(self, output_file=None):
        """
        Generate a comprehensive report of all experiments.

        Args:
            output_file (str, optional): Path to save the report
        """
        if not self.results:
            print("No experiments to report.")
            return "No experiments to report."

        # Get dataframe of results
        df = self.to_dataframe()

        # Start building the report
        report = []
        report.append("# Mental Manipulation Detection Experiments Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total experiments: {len(self.results)}")
        report.append("")

        # Add summary by level
        report.append("## Summary by Level")
        for level in df["level"].unique():
            level_df = df[df["level"] == level]
            report.append(f"### {level.capitalize()} Level")
            report.append(f"Number of experiments: {len(level_df)}")

            # Find best experiment for each metric
            for metric in [col for col in level_df.columns if col.startswith("metric_")]:
                clean_metric = metric.replace("metric_", "")
                # Check if there are any non-NaN values in this metric column
                if level_df[metric].notna().any():
                    # Get the best experiment, ignoring NaN values
                    best_idx = level_df[metric].idxmax()
                    if not pd.isna(best_idx):
                        best_exp = level_df.loc[best_idx]
                        report.append(
                            f"Best {clean_metric.upper()}: {best_exp[metric]:.4f} (Experiment #{best_exp['id']} - {best_exp['model_name']})")
                    else:
                        report.append(f"Best {clean_metric.upper()}: No experiments with valid values")
                else:
                    report.append(f"Best {clean_metric.upper()}: No experiments with this metric")

            report.append("")

        report.append("## Carbon Footprint Summary")
        for level in df["level"].unique():
            level_df = df[df["level"] == level]

            if "metric_total_emissions_kg" in level_df.columns:
                total_emissions = level_df["metric_total_emissions_kg"].sum()
                avg_emissions = level_df["metric_total_emissions_kg"].mean()
                max_emissions = level_df["metric_total_emissions_kg"].max()

                report.append(f"### {level.capitalize()} Level Carbon Metrics")
                report.append(f"Total emissions: {total_emissions:.6f} kg CO2")
                report.append(f"Average emissions per experiment: {avg_emissions:.6f} kg CO2")
                report.append(f"Maximum emissions: {max_emissions:.6f} kg CO2")
                report.append(f"Equivalent to: {total_emissions * 2.24:.2f} miles driven")
                report.append("")

        # Add detailed experiment results
        report.append("## Detailed Experiment Results")
        for exp in sorted(self.results, key=lambda x: x["id"]):
            report.append(f"### Experiment #{exp['id']}")
            report.append(f"Timestamp: {exp['timestamp']}")
            report.append(f"Model: {exp['model_name']}")
            report.append(f"Level: {exp['level']}")

            report.append("#### Hyperparameters")
            for hp_key, hp_value in exp["hyperparameters"].items():
                report.append(f"- {hp_key}: {hp_value}")

            report.append("#### Metrics")
            for metric_key, metric_value in exp["metrics"].items():
                if "emissions" in metric_key.lower() or "carbon" in metric_key.lower():
                    report.append(f"- {metric_key}: {metric_value:.6f}")
                else:
                    report.append(f"- {metric_key}: {metric_value:.4f}")

            if exp["notes"]:
                report.append("#### Notes")
                report.append(exp["notes"])

            report.append("")

        # Join the report into a single string
        full_report = "\n".join(report)

        # Save the report if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(full_report)
            print(f"Report saved to {output_file}")

        return full_report


    def visualize_carbon_efficiency(self, performance_metric="f1", save_path=None):
        """
        Visualize carbon efficiency (performance per CO2 kg).
        """
        df = self.to_dataframe()
        if df.empty or "metric_total_emissions_kg" not in df.columns:
            print("No carbon data to visualize.")
            return

        # Calculate efficiency: performance/emissions
        efficiency_col = "efficiency"
        df[efficiency_col] = df[f"metric_{performance_metric}"] / df["metric_total_emissions_kg"]

        plt.figure(figsize=(12, 6))

        # Create scatter plot with size proportional to absolute performance
        scatter = plt.scatter(
            df["metric_total_emissions_kg"],
            df[f"metric_{performance_metric}"],
            s=df[f"metric_{performance_metric}"] * 500,  # Size based on performance
            c=df[efficiency_col],
            cmap='viridis'
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Efficiency (F1/kg CO2)')

        # Annotate points with experiment IDs
        for i, row in df.iterrows():
            plt.annotate(f"#{row['id']}", (row["metric_total_emissions_kg"], row[f"metric_{performance_metric}"]))

        plt.xlabel('CO2 Emissions (kg)')
        plt.ylabel(f'{performance_metric.upper()} Score')
        plt.title(f'Carbon Efficiency Analysis (Bubble size = {performance_metric.upper()})')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Carbon efficiency visualization saved to {save_path}")

        plt.show()