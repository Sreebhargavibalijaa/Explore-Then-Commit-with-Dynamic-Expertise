"""
Experiment 5: Impact Option Pricing Models
Implements impact option pricing models for evaluating long-term research bets,
allowing researchers to "price" their direction's option value.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import random
from datetime import datetime, timedelta

class ImpactOptionPricing:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.research_volatility = 0.4  # 40% annual volatility for research outcomes
        
    def _generate_research_projects(self, n_projects=50):
        """Generate diverse research projects with different characteristics"""
        projects = []
        
        project_types = [
            'Theoretical Breakthrough', 'Methodological Innovation', 'Empirical Discovery',
            'System Integration', 'Algorithm Optimization', 'Dataset Creation',
            'Benchmark Development', 'Tool Development', 'Survey/Review',
            'Reproducibility Study', 'Cross-Domain Application', 'Novel Architecture'
        ]
        
        for i in range(n_projects):
            project_type = random.choice(project_types)
            
            # Generate project characteristics
            initial_impact = random.uniform(10, 100)  # Initial citation potential
            time_to_maturity = random.randint(1, 10)  # Years to potential breakthrough
            volatility = random.uniform(0.2, 0.8)  # Project-specific volatility
            research_cost = random.uniform(1000, 50000)  # Annual research cost
            success_probability = random.uniform(0.1, 0.9)  # Probability of success
            
            # Calculate option parameters
            strike_price = research_cost * time_to_maturity  # Total investment needed
            current_value = initial_impact * success_probability
            
            projects.append({
                'id': i,
                'type': project_type,
                'initial_impact': initial_impact,
                'time_to_maturity': time_to_maturity,
                'volatility': volatility,
                'research_cost': research_cost,
                'success_probability': success_probability,
                'strike_price': strike_price,
                'current_value': current_value,
                'option_value': 0,  # Will be calculated
                'expected_return': 0,  # Will be calculated
                'risk_adjusted_return': 0  # Will be calculated
            })
        
        return pd.DataFrame(projects)
    
    def _black_scholes_option_pricing(self, S, K, T, r, sigma):
        """Calculate option value using Black-Scholes model"""
        # S: current value, K: strike price, T: time to maturity, r: risk-free rate, sigma: volatility
        
        # Handle edge cases
        if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
            return max(0, S - K)  # Intrinsic value
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call option value
        call_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return max(0, call_value)  # Option value cannot be negative
    
    def _calculate_option_values(self, projects_df):
        """Calculate option values for all research projects"""
        option_values = []
        
        for _, project in projects_df.iterrows():
            S = project['current_value']
            K = project['strike_price']
            T = project['time_to_maturity']
            r = self.risk_free_rate
            sigma = project['volatility']
            
            # Calculate option value
            option_value = self._black_scholes_option_pricing(S, K, T, r, sigma)
            
            # Calculate expected return
            expected_return = (option_value - K) / K if K > 0 else 0
            
            # Calculate risk-adjusted return (Sharpe ratio)
            risk_adjusted_return = expected_return / sigma if sigma > 0 else 0
            
            option_values.append({
                'option_value': option_value,
                'expected_return': expected_return,
                'risk_adjusted_return': risk_adjusted_return,
                'option_premium': option_value / S if S > 0 else 0,
                'break_even_probability': K / (S + option_value) if (S + option_value) > 0 else 1
            })
        
        return pd.DataFrame(option_values)
    
    def _monte_carlo_simulation(self, projects_df, n_simulations=10000):
        """Run Monte Carlo simulation for research outcomes"""
        simulation_results = []
        
        for _, project in projects_df.iterrows():
            project_simulations = []
            
            for _ in range(n_simulations):
                # Simulate research outcome
                success = random.random() < project['success_probability']
                
                if success:
                    # Successful outcome - impact follows log-normal distribution
                    impact = np.random.lognormal(
                        mean=np.log(project['initial_impact']),
                        sigma=project['volatility']
                    )
                else:
                    # Failed outcome - minimal impact
                    impact = project['initial_impact'] * 0.1
                
                # Calculate net value
                total_cost = project['research_cost'] * project['time_to_maturity']
                net_value = impact - total_cost
                
                project_simulations.append({
                    'success': success,
                    'impact': impact,
                    'net_value': net_value,
                    'roi': net_value / total_cost if total_cost > 0 else 0
                })
            
            # Aggregate results
            simulations_df = pd.DataFrame(project_simulations)
            
            simulation_results.append({
                'project_id': project['id'],
                'mean_impact': simulations_df['impact'].mean(),
                'std_impact': simulations_df['impact'].std(),
                'mean_net_value': simulations_df['net_value'].mean(),
                'std_net_value': simulations_df['net_value'].std(),
                'success_rate': simulations_df['success'].mean(),
                'positive_roi_rate': (simulations_df['roi'] > 0).mean(),
                'var_95': np.percentile(simulations_df['net_value'], 5),  # Value at Risk
                'expected_shortfall': simulations_df[simulations_df['net_value'] < np.percentile(simulations_df['net_value'], 5)]['net_value'].mean()
            })
        
        return pd.DataFrame(simulation_results)
    
    def _portfolio_optimization(self, projects_df, option_values_df, budget=1000000):
        """Optimize research portfolio given budget constraints"""
        # Combine data by adding option values as new columns
        combined_df = projects_df.copy()
        combined_df['option_value'] = option_values_df['option_value']
        combined_df['expected_return'] = option_values_df['expected_return']
        combined_df['risk_adjusted_return'] = option_values_df['risk_adjusted_return']
        combined_df['option_premium'] = option_values_df['option_premium']
        combined_df['break_even_probability'] = option_values_df['break_even_probability']
        
        # Calculate metrics for optimization
        combined_df['value_per_cost'] = combined_df['option_value'] / combined_df['research_cost']
        combined_df['impact_per_cost'] = combined_df['initial_impact'] / combined_df['research_cost']
        combined_df['risk_adjusted_value'] = combined_df['option_value'] * combined_df['risk_adjusted_return']
        
        # Sort by different criteria
        value_optimized = combined_df.nlargest(20, 'value_per_cost')
        impact_optimized = combined_df.nlargest(20, 'impact_per_cost')
        risk_adjusted_optimized = combined_df.nlargest(20, 'risk_adjusted_value')
        
        # Calculate portfolio metrics
        portfolios = {
            'value_optimized': self._calculate_portfolio_metrics(value_optimized, budget),
            'impact_optimized': self._calculate_portfolio_metrics(impact_optimized, budget),
            'risk_adjusted_optimized': self._calculate_portfolio_metrics(risk_adjusted_optimized, budget)
        }
        
        return portfolios
    
    def _calculate_portfolio_metrics(self, portfolio_df, budget):
        """Calculate portfolio-level metrics"""
        total_cost = portfolio_df['research_cost'].sum()
        total_option_value = portfolio_df['option_value'].sum()
        total_impact = portfolio_df['initial_impact'].sum()
        
        # Weight by budget allocation
        if total_cost > 0:
            budget_allocation = min(1.0, budget / total_cost)
            adjusted_option_value = total_option_value * budget_allocation
            adjusted_impact = total_impact * budget_allocation
        else:
            budget_allocation = 0
            adjusted_option_value = 0
            adjusted_impact = 0
        
        return {
            'total_cost': total_cost,
            'total_option_value': total_option_value,
            'total_impact': total_impact,
            'budget_allocation': budget_allocation,
            'adjusted_option_value': adjusted_option_value,
            'adjusted_impact': adjusted_impact,
            'portfolio_efficiency': adjusted_option_value / budget if budget > 0 else 0,
            'diversification_score': 1.0 - (portfolio_df['type'].value_counts().max() / len(portfolio_df))
        }
    
    def _analyze_temporal_dynamics(self, projects_df, option_values_df):
        """Analyze how option values change over time"""
        temporal_data = []
        
        for _, project in projects_df.iterrows():
            for year in range(1, project['time_to_maturity'] + 1):
                # Adjust parameters for current year
                remaining_time = project['time_to_maturity'] - year
                remaining_cost = project['research_cost'] * remaining_time
                
                # Assume some progress has been made
                progress_factor = year / project['time_to_maturity']
                current_value = project['current_value'] * (1 + progress_factor * 0.5)
                
                # Calculate option value at this point
                option_value = self._black_scholes_option_pricing(
                    current_value, remaining_cost, remaining_time, 
                    self.risk_free_rate, project['volatility']
                )
                
                temporal_data.append({
                    'project_id': project['id'],
                    'year': year,
                    'remaining_time': remaining_time,
                    'current_value': current_value,
                    'remaining_cost': remaining_cost,
                    'option_value': option_value,
                    'option_premium': option_value / current_value if current_value > 0 else 0
                })
        
        return pd.DataFrame(temporal_data)
    
    def plot_results(self, projects_df, option_values_df, simulation_results, portfolios, temporal_data):
        """Create visualizations of option pricing results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Option values vs project characteristics
        combined_df = projects_df.copy()
        combined_df['option_value'] = option_values_df['option_value'].values
        combined_df['expected_return'] = option_values_df['expected_return'].values
        combined_df['risk_adjusted_return'] = option_values_df['risk_adjusted_return'].values
        
        axes[0, 0].scatter(combined_df['initial_impact'], combined_df['option_value'], 
                          c=combined_df['volatility'], cmap='viridis', alpha=0.7, s=100)
        axes[0, 0].set_xlabel('Initial Impact')
        axes[0, 0].set_ylabel('Option Value')
        axes[0, 0].set_title('Option Values vs Impact')
        plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Volatility')
        
        # Plot 2: Risk-return scatter
        axes[0, 1].scatter(combined_df['volatility'], combined_df['expected_return'], 
                          c=combined_df['option_value'], cmap='plasma', alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Volatility')
        axes[0, 1].set_ylabel('Expected Return')
        axes[0, 1].set_title('Risk-Return Profile')
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Option Value')
        
        # Plot 3: Portfolio comparison
        portfolio_names = list(portfolios.keys())
        portfolio_efficiencies = [portfolios[name]['portfolio_efficiency'] for name in portfolio_names]
        portfolio_diversification = [portfolios[name]['diversification_score'] for name in portfolio_names]
        
        x = np.arange(len(portfolio_names))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, portfolio_efficiencies, width, label='Efficiency', alpha=0.7)
        axes[0, 2].bar(x + width/2, portfolio_diversification, width, label='Diversification', alpha=0.7)
        axes[0, 2].set_xlabel('Portfolio Strategy')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Portfolio Performance')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([name.replace('_', ' ').title() for name in portfolio_names])
        axes[0, 2].legend()
        
        # Plot 4: Temporal dynamics
        if not temporal_data.empty:
            for project_id in temporal_data['project_id'].unique()[:5]:  # Show first 5 projects
                project_data = temporal_data[temporal_data['project_id'] == project_id]
                axes[1, 0].plot(project_data['year'], project_data['option_value'], 
                               marker='o', label=f'Project {project_id}')
            
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Option Value')
            axes[1, 0].set_title('Option Value Evolution')
            axes[1, 0].legend()
        
        # Plot 5: Simulation results
        if not simulation_results.empty:
            axes[1, 1].scatter(simulation_results['mean_net_value'], simulation_results['std_net_value'], 
                              c=simulation_results['success_rate'], cmap='RdYlGn', alpha=0.7, s=100)
            axes[1, 1].set_xlabel('Mean Net Value')
            axes[1, 1].set_ylabel('Std Net Value')
            axes[1, 1].set_title('Risk-Reward from Simulations')
            plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Success Rate')
        
        # Plot 6: Project type analysis
        type_analysis = combined_df.groupby('type').agg({
            'option_value': 'mean',
            'expected_return': 'mean',
            'volatility': 'mean'
        }).reset_index()
        
        top_types = type_analysis.nlargest(8, 'option_value')
        axes[1, 2].barh(range(len(top_types)), top_types['option_value'])
        axes[1, 2].set_yticks(range(len(top_types)))
        axes[1, 2].set_yticklabels(top_types['type'], fontsize=8)
        axes[1, 2].set_xlabel('Average Option Value')
        axes[1, 2].set_title('Option Values by Project Type')
        
        plt.tight_layout()
        plt.savefig('experiment_5_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """Run the complete impact option pricing experiment"""
        print("=== Experiment 5: Impact Option Pricing Models ===\n")
        
        # Generate research projects
        projects_df = self._generate_research_projects()
        print(f"Generated {len(projects_df)} research projects")
        print(f"Project types: {projects_df['type'].value_counts().to_dict()}")
        
        # Calculate option values
        option_values_df = self._calculate_option_values(projects_df)
        print(f"\nOption Pricing Results:")
        print(f"  - Mean option value: {option_values_df['option_value'].mean():.2f}")
        print(f"  - Mean expected return: {option_values_df['expected_return'].mean():.3f}")
        print(f"  - Mean risk-adjusted return: {option_values_df['risk_adjusted_return'].mean():.3f}")
        
        # Run Monte Carlo simulation
        simulation_results = self._monte_carlo_simulation(projects_df)
        print(f"\nMonte Carlo Simulation Results:")
        print(f"  - Mean success rate: {simulation_results['success_rate'].mean():.3f}")
        print(f"  - Mean positive ROI rate: {simulation_results['positive_roi_rate'].mean():.3f}")
        print(f"  - Mean VaR (95%): {simulation_results['var_95'].mean():.2f}")
        
        # Portfolio optimization
        portfolios = self._portfolio_optimization(projects_df, option_values_df)
        print(f"\nPortfolio Optimization Results:")
        for strategy, metrics in portfolios.items():
            print(f"  {strategy}:")
            print(f"    - Portfolio efficiency: {metrics['portfolio_efficiency']:.4f}")
            print(f"    - Diversification score: {metrics['diversification_score']:.3f}")
            print(f"    - Total option value: {metrics['total_option_value']:.2f}")
        
        # Temporal dynamics
        temporal_data = self._analyze_temporal_dynamics(projects_df, option_values_df)
        print(f"\nTemporal Dynamics:")
        print(f"  - Generated {len(temporal_data)} temporal data points")
        
        # Create visualizations
        self.plot_results(projects_df, option_values_df, simulation_results, portfolios, temporal_data)
        
        return {
            'projects_df': projects_df,
            'option_values_df': option_values_df,
            'simulation_results': simulation_results,
            'portfolios': portfolios,
            'temporal_data': temporal_data
        }

if __name__ == "__main__":
    experiment = ImpactOptionPricing()
    results = experiment.run_experiment() 