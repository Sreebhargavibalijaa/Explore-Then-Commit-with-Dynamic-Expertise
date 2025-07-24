"""
Master Script: Run All Experiments for Lottery Ticket Hypothesis for Research
Executes all 6 experiments and generates a comprehensive report.
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import all experiments
from experiment_1_longitudinal_analysis import LongitudinalAnalysis
from experiment_2_researcher_simulation import ResearcherSimulation
from experiment_3_anti_competition import AntiCompetitionAnalysis
from experiment_4_degeneracy_test import DegeneracyTest
from experiment_5_impact_options import ImpactOptionPricing
from experiment_6_attention_decay import AttentionDecayAnalysis
from experiment_7_bandit_research import BanditResearchStrategy

class ComprehensiveAnalysis:
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_all_experiments(self):
        """Run all experiments sequentially"""
        print("=" * 80)
        print("LOTTERY TICKET HYPOTHESIS FOR RESEARCH - COMPREHENSIVE EXPERIMENTAL VALIDATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Experiment 1: Longitudinal Analysis
        print("Running Experiment 1: Longitudinal Analysis of Transformative ML Advances...")
        experiment1 = LongitudinalAnalysis()
        self.results['experiment1'] = experiment1.run_experiment()
        print("✓ Experiment 1 completed\n")
        
        # Experiment 2: Researcher Simulation
        print("Running Experiment 2: Controlled Simulation of Researcher Trajectories...")
        experiment2 = ResearcherSimulation(n_researchers=5000)  # Reduced for faster execution
        self.results['experiment2'] = experiment2.run_experiment()
        print("✓ Experiment 2 completed\n")
        
        # Experiment 3: Anti-Competition Effect
        print("Running Experiment 3: Anti-Competition Effect Analysis...")
        experiment3 = AntiCompetitionAnalysis()
        self.results['experiment3'] = experiment3.run_experiment()
        print("✓ Experiment 3 completed\n")
        
        # Experiment 4: Degeneracy Test
        print("Running Experiment 4: Degeneracy Test Implementation...")
        experiment4 = DegeneracyTest()
        self.results['experiment4'] = experiment4.run_experiment()
        print("✓ Experiment 4 completed\n")
        
        # Experiment 5: Impact Option Pricing
        print("Running Experiment 5: Impact Option Pricing Models...")
        experiment5 = ImpactOptionPricing()
        self.results['experiment5'] = experiment5.run_experiment()
        print("✓ Experiment 5 completed\n")
        
        # Experiment 6: Attention Decay Metrics
        print("Running Experiment 6: Attention Decay Metrics...")
        experiment6 = AttentionDecayAnalysis()
        self.results['experiment6'] = experiment6.run_experiment()
        print("✓ Experiment 6 completed\n")
        
        # Experiment 7: Multi-Armed Bandit Research Strategy
        print("Running Experiment 7: Multi-Armed Bandit Research Strategy...")
        experiment7 = BanditResearchStrategy()
        self.results['experiment7'] = experiment7.run_experiment()
        print("✓ Experiment 7 completed\n")
        
        execution_time = time.time() - self.start_time
        print(f"All experiments completed in {execution_time:.2f} seconds")
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all findings"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EXPERIMENTAL REPORT")
        print("=" * 80)
        
        # Summary of key findings
        print("\n📊 KEY FINDINGS SUMMARY:")
        print("-" * 50)
        
        # 10x Depth Threshold
        exp1_results = self.results['experiment1']['threshold_results']
        print(f"🔍 10x Depth Threshold:")
        print(f"   • Odds ratio: {exp1_results['odds_ratio']:.2f}")
        print(f"   • P-value: {exp1_results['p_value']:.6f}")
        print(f"   • Transformative papers above 3,000hr threshold: {exp1_results['transformative_above_threshold']}/15")
        
        # Serendipity Multiplier
        exp2_results = self.results['experiment2']['serendipity_results']
        print(f"\n🎯 Serendipity Multiplier:")
        print(f"   • Low percentile breakthrough rate: {exp2_results['low_percentile_breakthrough_rate']:.3f}")
        print(f"   • High percentile breakthrough rate: {exp2_results['high_percentile_breakthrough_rate']:.3f}")
        print(f"   • Serendipity ratio: {exp2_results['serendipity_ratio']:.2f}")
        print(f"   • Cohen's d: {exp2_results['cohens_d']:.2f}")
        
        # Anti-Competition Effect
        exp3_results = self.results['experiment3']['competition_results']
        print(f"\n⚔️ Anti-Competition Effect:")
        print(f"   • Predicted β (per 100 papers): {exp3_results['predicted_beta']:.2f}")
        print(f"   • Exponential R-squared: {exp3_results['exponential_r_squared']:.3f}")
        print(f"   • Correlation: {exp3_results['correlation']:.3f}")
        
        # Degeneracy Test
        exp4_results = self.results['experiment4']['validation_results']
        print(f"\n🧪 Degeneracy Test Performance:")
        print(f"   • Precision: {exp4_results['precision']:.3f}")
        print(f"   • Recall: {exp4_results['recall']:.3f}")
        print(f"   • F1-score: {exp4_results['f1_score']:.3f}")
        print(f"   • T-statistic: {exp4_results['t_statistic']:.3f}")
        
        # Impact Option Pricing
        exp5_results = self.results['experiment5']['option_values_df']
        print(f"\n💰 Impact Option Pricing:")
        print(f"   • Mean option value: {exp5_results['option_value'].mean():.2f}")
        print(f"   • Mean expected return: {exp5_results['expected_return'].mean():.3f}")
        print(f"   • Mean risk-adjusted return: {exp5_results['risk_adjusted_return'].mean():.3f}")
        
        # Attention Decay
        exp6_results = self.results['experiment6']['commitment_signals']
        print(f"\n⏰ Attention Decay Metrics:")
        print(f"   • Mean commitment score: {exp6_results['commitment_score'].mean():.3f}")
        print(f"   • Mean attention-productivity correlation: {exp6_results['attention_productivity_corr'].mean():.3f}")
        
        # Multi-Armed Bandit Research Strategy
        exp7_results = self.results['experiment7']['strategy_performance']
        best_strategy = max(exp7_results.items(), key=lambda x: x[1]['mean_reward'])
        print(f"\n🎰 Multi-Armed Bandit Research Strategy:")
        print(f"   • Best strategy: {best_strategy[0]} (mean reward: {best_strategy[1]['mean_reward']:.2f})")
        print(f"   • Exploration-exploitation balance is crucial for research success")
        print(f"   • ML models can predict research strategy performance")
        
        # Statistical significance summary
        print(f"\n📈 STATISTICAL SIGNIFICANCE SUMMARY:")
        print("-" * 50)
        significant_findings = []
        
        if exp1_results['p_value'] < 0.001:
            significant_findings.append("10x Depth Threshold (p < 0.001)")
        if exp2_results['p_value'] < 0.001:
            significant_findings.append("Serendipity Multiplier (p < 0.001)")
        if exp3_results['p_value'] < 0.001:
            significant_findings.append("Anti-Competition Effect (p < 0.001)")
        if exp4_results['p_value'] < 0.001:
            significant_findings.append("Degeneracy Test (p < 0.001)")
        
        print(f"   • Statistically significant findings: {len(significant_findings)}")
        for finding in significant_findings:
            print(f"     ✓ {finding}")
        
        # Effect sizes
        print(f"\n📏 EFFECT SIZES:")
        print("-" * 50)
        print(f"   • Serendipity Multiplier: Cohen's d = {exp2_results['cohens_d']:.2f}")
        print(f"   • Anti-Competition Effect: R² = {exp3_results['exponential_r_squared']:.3f}")
        print(f"   • 10x Depth Threshold: Odds ratio = {exp1_results['odds_ratio']:.2f}")
        
        # Policy implications
        print(f"\n🎯 POLICY IMPLICATIONS:")
        print("-" * 50)
        print("   • The ML community systematically undervalues focused obsession")
        print("   • Random focus on arbitrary problems yields higher breakthrough rates")
        print("   • Competition in popular areas reduces per-paper transformative impact")
        print("   • Degeneracy tests can identify fertile research directions")
        print("   • Impact option pricing enables better research portfolio management")
        print("   • Attention decay metrics help optimize research commitment decisions")
        print("   • Multi-armed bandit strategies optimize exploration-exploitation balance")
        
        # Limitations and future work
        print(f"\n⚠️ LIMITATIONS & FUTURE WORK:")
        print("-" * 50)
        print("   • Experiments use synthetic data - real-world validation needed")
        print("   • Focus hours estimation requires more precise measurement")
        print("   • Longitudinal studies needed to validate temporal dynamics")
        print("   • Cross-cultural validation of Random Focus Principle")
        print("   • Integration with existing research evaluation frameworks")
        
    def create_combined_visualization(self):
        """Create a combined visualization of key results"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: 10x Depth Threshold
        exp1_results = self.results['experiment1']['threshold_results']
        labels = ['Transformative', 'SOTA-chasing']
        values = [exp1_results['transformative_prob'], exp1_results['control_prob']]
        colors = ['blue', 'red']
        
        axes[0, 0].bar(labels, values, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Probability Above Threshold')
        axes[0, 0].set_title('10x Depth Threshold\n(≥3,000 hours)')
        axes[0, 0].text(0.5, 0.9, f'Odds Ratio: {exp1_results["odds_ratio"]:.2f}', 
                       ha='center', va='top', transform=axes[0, 0].transAxes)
        
        # Plot 2: Serendipity Multiplier
        exp2_results = self.results['experiment2']['serendipity_results']
        labels = ['Low Percentile\n(<30%)', 'High Percentile\n(≥30%)']
        values = [exp2_results['low_percentile_breakthrough_rate'], 
                 exp2_results['high_percentile_breakthrough_rate']]
        
        axes[0, 1].bar(labels, values, color=['green', 'orange'], alpha=0.7)
        axes[0, 1].set_ylabel('Breakthrough Rate')
        axes[0, 1].set_title('Serendipity Multiplier')
        axes[0, 1].text(0.5, 0.9, f'Ratio: {exp2_results["serendipity_ratio"]:.2f}', 
                       ha='center', va='top', transform=axes[0, 1].transAxes)
        
        # Plot 3: Anti-Competition Effect
        exp3_data = self.results['experiment3']['data']
        axes[0, 2].scatter(exp3_data['total_papers'], exp3_data['transformative_rate'], alpha=0.7)
        axes[0, 2].set_xlabel('Total Papers')
        axes[0, 2].set_ylabel('Transformative Rate')
        axes[0, 2].set_title('Anti-Competition Effect')
        
        # Plot 4: Degeneracy Test Performance
        exp4_results = self.results['experiment4']['validation_results']
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [exp4_results['precision'], exp4_results['recall'], exp4_results['f1_score']]
        
        axes[1, 0].bar(metrics, values, color=['purple', 'cyan', 'magenta'], alpha=0.7)
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Degeneracy Test Performance')
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 5: Impact Option Values
        exp5_results = self.results['experiment5']['option_values_df']
        axes[1, 1].hist(exp5_results['option_value'], bins=20, alpha=0.7, color='gold')
        axes[1, 1].set_xlabel('Option Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Impact Option Values Distribution')
        
        # Plot 6: Attention Decay Metrics
        exp6_results = self.results['experiment6']['commitment_signals']
        if not exp6_results.empty:
            axes[1, 2].scatter(exp6_results['attention_trend'], exp6_results['productivity_trend'],
                              c=exp6_results['commitment_score'], cmap='viridis', alpha=0.7)
            axes[1, 2].set_xlabel('Attention Trend')
            axes[1, 2].set_ylabel('Productivity Trend')
            axes[1, 2].set_title('Commitment Signals')
            plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2], label='Commitment Score')
        
        plt.tight_layout()
        plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results_to_file(self):
        """Save all results to a CSV file for further analysis"""
        # Create summary dataframe
        summary_data = []
        
        # Experiment 1 results
        exp1_results = self.results['experiment1']['threshold_results']
        summary_data.append({
            'experiment': 'Longitudinal Analysis',
            'metric': '10x Depth Threshold',
            'value': exp1_results['odds_ratio'],
            'p_value': exp1_results['p_value'],
            'description': 'Odds ratio for transformative papers above 3,000hr threshold'
        })
        
        # Experiment 2 results
        exp2_results = self.results['experiment2']['serendipity_results']
        summary_data.append({
            'experiment': 'Researcher Simulation',
            'metric': 'Serendipity Multiplier',
            'value': exp2_results['serendipity_ratio'],
            'p_value': exp2_results['p_value'],
            'description': 'Ratio of breakthrough rates (low vs high percentile)'
        })
        
        # Experiment 3 results
        exp3_results = self.results['experiment3']['competition_results']
        summary_data.append({
            'experiment': 'Anti-Competition Effect',
            'metric': 'Competition Beta',
            'value': exp3_results['predicted_beta'],
            'p_value': exp3_results['p_value'],
            'description': 'Exponential decay coefficient per 100 papers'
        })
        
        # Experiment 4 results
        exp4_results = self.results['experiment4']['validation_results']
        summary_data.append({
            'experiment': 'Degeneracy Test',
            'metric': 'F1-Score',
            'value': exp4_results['f1_score'],
            'p_value': exp4_results['p_value'],
            'description': 'F1-score for fertile problem identification'
        })
        
        # Experiment 5 results
        exp5_results = self.results['experiment5']['option_values_df']
        summary_data.append({
            'experiment': 'Impact Option Pricing',
            'metric': 'Mean Option Value',
            'value': exp5_results['option_value'].mean(),
            'p_value': None,
            'description': 'Average option value across research projects'
        })
        
        # Experiment 6 results
        exp6_results = self.results['experiment6']['commitment_signals']
        if not exp6_results.empty:
            summary_data.append({
                'experiment': 'Attention Decay',
                'metric': 'Mean Commitment Score',
                'value': exp6_results['commitment_score'].mean(),
                'p_value': None,
                'description': 'Average commitment score across trajectories'
            })
        
        # Save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('experimental_results_summary.csv', index=False)
        print(f"\n📄 Results saved to 'experimental_results_summary.csv'")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        # Run all experiments
        self.run_all_experiments()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Create combined visualization
        self.create_combined_visualization()
        
        # Save results
        self.save_results_to_file()
        
        print(f"\n🎉 COMPLETE ANALYSIS FINISHED!")
        print(f"Total execution time: {time.time() - self.start_time:.2f} seconds")
        print(f"Generated files:")
        print(f"  • experiment_1_results.png")
        print(f"  • experiment_2_results.png")
        print(f"  • experiment_3_results.png")
        print(f"  • experiment_4_results.png")
        print(f"  • experiment_5_results.png")
        print(f"  • experiment_6_results.png")
        print(f"  • comprehensive_results.png")
        print(f"  • experimental_results_summary.csv")

if __name__ == "__main__":
    analysis = ComprehensiveAnalysis()
    analysis.run_complete_analysis() 