"""
Experiment 6: Attention Decay Metrics
Implements attention decay metrics to quantify when to abandon/commit to research directions,
based on diminishing returns and opportunity costs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import random
from datetime import datetime, timedelta

class AttentionDecayAnalysis:
    def __init__(self):
        self.decay_functions = {
            'exponential': lambda t, a, b: a * np.exp(-b * t),
            'power_law': lambda t, a, b: a * (t + 1) ** (-b),
            'logistic': lambda t, a, b, c: a / (1 + np.exp(b * (t - c))),
            'weibull': lambda t, a, b, c: a * np.exp(-((t / b) ** c))
        }
        
    def _generate_research_trajectories(self, n_trajectories=100):
        """Generate research trajectories with different attention decay patterns"""
        trajectories = []
        
        research_types = [
            'Incremental Improvement', 'Novel Approach', 'Cross-Domain Application',
            'Theoretical Development', 'Empirical Validation', 'Methodological Innovation',
            'System Integration', 'Benchmark Optimization', 'Reproducibility Study',
            'Survey/Review', 'Tool Development', 'Dataset Creation'
        ]
        
        for i in range(n_trajectories):
            research_type = random.choice(research_types)
            
            # Generate trajectory parameters
            initial_attention = random.uniform(0.5, 1.0)  # Initial attention level
            decay_rate = random.uniform(0.1, 0.8)  # How fast attention decays
            breakthrough_probability = random.uniform(0.01, 0.2)  # Probability of breakthrough
            opportunity_cost = random.uniform(0.1, 0.5)  # Cost of staying vs switching
            
            # Choose decay function
            decay_function = random.choice(list(self.decay_functions.keys()))
            
            # Generate time series data
            max_time = random.randint(10, 50)  # Months
            time_points = np.arange(max_time)
            
            # Generate attention decay
            if decay_function == 'exponential':
                attention = self.decay_functions[decay_function](time_points, initial_attention, decay_rate)
            elif decay_function == 'power_law':
                attention = self.decay_functions[decay_function](time_points, initial_attention, decay_rate)
            elif decay_function == 'logistic':
                attention = self.decay_functions[decay_function](time_points, initial_attention, decay_rate, max_time/2)
            else:  # weibull
                attention = self.decay_functions[decay_function](time_points, initial_attention, max_time/3, decay_rate)
            
            # Add noise
            attention += np.random.normal(0, 0.05, len(attention))
            attention = np.clip(attention, 0, 1)
            
            # Generate productivity (correlated with attention but with lag)
            productivity = np.zeros_like(attention)
            for t in range(len(attention)):
                if t == 0:
                    productivity[t] = attention[t]
                else:
                    # Productivity follows attention with some lag and persistence
                    productivity[t] = 0.7 * attention[t] + 0.3 * productivity[t-1] + np.random.normal(0, 0.1)
            
            productivity = np.clip(productivity, 0, 1)
            
            # Determine if breakthrough occurs
            breakthrough_time = None
            for t in range(len(time_points)):
                if random.random() < breakthrough_probability * attention[t]:
                    breakthrough_time = t
                    break
            
            trajectories.append({
                'id': i,
                'type': research_type,
                'initial_attention': initial_attention,
                'decay_rate': decay_rate,
                'decay_function': decay_function,
                'breakthrough_probability': breakthrough_probability,
                'opportunity_cost': opportunity_cost,
                'time_points': time_points,
                'attention': attention,
                'productivity': productivity,
                'breakthrough_time': breakthrough_time,
                'max_time': max_time
            })
        
        return trajectories
    
    def _fit_decay_models(self, trajectory):
        """Fit different decay models to a trajectory"""
        time_points = trajectory['time_points']
        attention = trajectory['attention']
        
        models = {}
        
        # Exponential decay: A * exp(-b*t)
        def exp_loss(params):
            a, b = params
            predicted = a * np.exp(-b * time_points)
            return np.mean((attention - predicted) ** 2)
        
        try:
            exp_result = minimize(exp_loss, [1.0, 0.1], bounds=[(0, None), (0, None)])
            models['exponential'] = {
                'params': exp_result.x,
                'mse': exp_result.fun,
                'aic': len(time_points) * np.log(exp_result.fun) + 2 * 2
            }
        except:
            models['exponential'] = {'params': [0, 0], 'mse': float('inf'), 'aic': float('inf')}
        
        # Power law decay: A * (t+1)^(-b)
        def power_loss(params):
            a, b = params
            predicted = a * (time_points + 1) ** (-b)
            return np.mean((attention - predicted) ** 2)
        
        try:
            power_result = minimize(power_loss, [1.0, 0.5], bounds=[(0, None), (0, None)])
            models['power_law'] = {
                'params': power_result.x,
                'mse': power_result.fun,
                'aic': len(time_points) * np.log(power_result.fun) + 2 * 2
            }
        except:
            models['power_law'] = {'params': [0, 0], 'mse': float('inf'), 'aic': float('inf')}
        
        # Logistic decay: A / (1 + exp(b*(t-c)))
        def logistic_loss(params):
            a, b, c = params
            predicted = a / (1 + np.exp(b * (time_points - c)))
            return np.mean((attention - predicted) ** 2)
        
        try:
            logistic_result = minimize(logistic_loss, [1.0, 0.1, len(time_points)/2], 
                                     bounds=[(0, None), (0, None), (0, len(time_points))])
            models['logistic'] = {
                'params': logistic_result.x,
                'mse': logistic_result.fun,
                'aic': len(time_points) * np.log(logistic_result.fun) + 2 * 3
            }
        except:
            models['logistic'] = {'params': [0, 0, 0], 'mse': float('inf'), 'aic': float('inf')}
        
        return models
    
    def _calculate_attention_metrics(self, trajectory):
        """Calculate various attention decay metrics"""
        attention = trajectory['attention']
        time_points = trajectory['time_points']
        
        # Basic metrics
        initial_attention = attention[0]
        final_attention = attention[-1]
        attention_decay = initial_attention - final_attention
        
        # Rate of decay
        decay_rate = np.mean(np.diff(attention))
        
        # Half-life (time to reach 50% of initial attention)
        half_attention = initial_attention / 2
        half_life = None
        for t, att in enumerate(attention):
            if att <= half_attention:
                half_life = t
                break
        
        # Attention stability (inverse of variance)
        attention_stability = 1 / (np.var(attention) + 1e-6)
        
        # Attention recovery (ability to bounce back)
        attention_recovery = 0
        for t in range(1, len(attention)):
            if attention[t] > attention[t-1]:
                attention_recovery += attention[t] - attention[t-1]
        
        # Cumulative attention
        cumulative_attention = np.sum(attention)
        
        # Attention efficiency (cumulative attention per unit time)
        attention_efficiency = cumulative_attention / len(time_points)
        
        return {
            'initial_attention': initial_attention,
            'final_attention': final_attention,
            'attention_decay': attention_decay,
            'decay_rate': decay_rate,
            'half_life': half_life,
            'attention_stability': attention_stability,
            'attention_recovery': attention_recovery,
            'cumulative_attention': cumulative_attention,
            'attention_efficiency': attention_efficiency
        }
    
    def _calculate_abandonment_thresholds(self, trajectory, models):
        """Calculate optimal abandonment thresholds"""
        attention = trajectory['attention']
        time_points = trajectory['time_points']
        opportunity_cost = trajectory['opportunity_cost']
        
        # Find best fitting model
        best_model = min(models.items(), key=lambda x: x[1]['aic'])
        model_name, model_info = best_model
        
        # Calculate predicted future attention
        future_time = np.arange(len(time_points), len(time_points) + 12)  # 12 months ahead
        
        if model_name == 'exponential':
            a, b = model_info['params']
            future_attention = a * np.exp(-b * future_time)
        elif model_name == 'power_law':
            a, b = model_info['params']
            future_attention = a * (future_time + 1) ** (-b)
        elif model_name == 'logistic':
            a, b, c = model_info['params']
            future_attention = a / (1 + np.exp(b * (future_time - c)))
        else:
            future_attention = np.zeros_like(future_time)
        
        # Calculate expected value of continuing vs switching
        current_attention = attention[-1]
        expected_continuation_value = np.mean(future_attention)
        expected_switch_value = 1.0 - opportunity_cost  # Assume new direction has full attention
        
        # Abandonment threshold based on expected value
        abandonment_threshold = expected_switch_value - expected_continuation_value
        
        # Time-based threshold (when attention drops below certain level)
        time_threshold = None
        for t, att in enumerate(future_attention):
            if att < 0.3:  # 30% attention threshold
                time_threshold = t
                break
        
        # Opportunity cost threshold
        opportunity_threshold = opportunity_cost
        
        return {
            'best_model': model_name,
            'model_params': model_info['params'],
            'current_attention': current_attention,
            'expected_continuation_value': expected_continuation_value,
            'expected_switch_value': expected_switch_value,
            'abandonment_threshold': abandonment_threshold,
            'time_threshold': time_threshold,
            'opportunity_threshold': opportunity_threshold,
            'should_abandon': abandonment_threshold > 0
        }
    
    def _analyze_commitment_signals(self, trajectories):
        """Analyze signals that indicate when to commit to a direction"""
        commitment_data = []
        
        for trajectory in trajectories:
            attention = trajectory['attention']
            productivity = trajectory['productivity']
            breakthrough_time = trajectory['breakthrough_time']
            
            # Calculate commitment signals
            attention_trend = np.polyfit(range(len(attention)), attention, 1)[0]
            productivity_trend = np.polyfit(range(len(productivity)), productivity, 1)[0]
            
            # Attention-productivity correlation
            attention_productivity_corr = np.corrcoef(attention, productivity)[0, 1]
            
            # Stability metrics
            attention_stability = 1 / (np.var(attention) + 1e-6)
            productivity_stability = 1 / (np.var(productivity) + 1e-6)
            
            # Breakthrough indicators
            breakthrough_occurred = breakthrough_time is not None
            time_to_breakthrough = breakthrough_time if breakthrough_time else len(attention)
            
            # Commitment score
            commitment_score = (
                0.3 * attention_trend +
                0.3 * productivity_trend +
                0.2 * attention_productivity_corr +
                0.1 * attention_stability +
                0.1 * productivity_stability
            )
            
            commitment_data.append({
                'trajectory_id': trajectory['id'],
                'attention_trend': attention_trend,
                'productivity_trend': productivity_trend,
                'attention_productivity_corr': attention_productivity_corr,
                'attention_stability': attention_stability,
                'productivity_stability': productivity_stability,
                'breakthrough_occurred': breakthrough_occurred,
                'time_to_breakthrough': time_to_breakthrough,
                'commitment_score': commitment_score
            })
        
        return pd.DataFrame(commitment_data)
    
    def plot_results(self, trajectories, decay_models, attention_metrics, abandonment_thresholds, commitment_signals):
        """Create visualizations of attention decay analysis"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Sample attention decay curves
        sample_trajectories = random.sample(trajectories, 5)
        for trajectory in sample_trajectories:
            axes[0, 0].plot(trajectory['time_points'], trajectory['attention'], 
                           label=f"Type: {trajectory['type'][:15]}...", alpha=0.7)
        
        axes[0, 0].set_xlabel('Time (months)')
        axes[0, 0].set_ylabel('Attention Level')
        axes[0, 0].set_title('Sample Attention Decay Curves')
        axes[0, 0].legend(fontsize=8)
        
        # Plot 2: Decay model comparison
        model_performance = {}
        for trajectory in trajectories:
            models = decay_models[trajectory['id']]
            for model_name, model_info in models.items():
                if model_name not in model_performance:
                    model_performance[model_name] = []
                model_performance[model_name].append(model_info['aic'])
        
        model_names = list(model_performance.keys())
        model_aics = [np.mean(model_performance[name]) for name in model_names]
        
        axes[0, 1].bar(model_names, model_aics, alpha=0.7)
        axes[0, 1].set_xlabel('Decay Model')
        axes[0, 1].set_ylabel('Average AIC')
        axes[0, 1].set_title('Model Performance Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Attention metrics distribution
        attention_decay_values = [metrics['attention_decay'] for metrics in attention_metrics.values()]
        half_life_values = [metrics['half_life'] for metrics in attention_metrics.values() if metrics['half_life'] is not None]
        
        axes[0, 2].hist(attention_decay_values, bins=20, alpha=0.7, label='Attention Decay')
        axes[0, 2].set_xlabel('Attention Decay')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Distribution of Attention Decay')
        
        # Plot 4: Abandonment thresholds
        abandonment_values = [thresh['abandonment_threshold'] for thresh in abandonment_thresholds.values()]
        should_abandon = [thresh['should_abandon'] for thresh in abandonment_thresholds.values()]
        
        colors = ['red' if abandon else 'blue' for abandon in should_abandon]
        axes[1, 0].scatter(range(len(abandonment_values)), abandonment_values, c=colors, alpha=0.7)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Trajectory ID')
        axes[1, 0].set_ylabel('Abandonment Threshold')
        axes[1, 0].set_title('Abandonment Thresholds')
        
        # Plot 5: Commitment signals
        if not commitment_signals.empty:
            axes[1, 1].scatter(commitment_signals['attention_trend'], 
                              commitment_signals['productivity_trend'],
                              c=commitment_signals['commitment_score'], cmap='viridis', alpha=0.7)
            axes[1, 1].set_xlabel('Attention Trend')
            axes[1, 1].set_ylabel('Productivity Trend')
            axes[1, 1].set_title('Commitment Signals')
            plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Commitment Score')
        
        # Plot 6: Breakthrough analysis
        breakthrough_times = [traj['breakthrough_time'] for traj in trajectories if traj['breakthrough_time'] is not None]
        no_breakthrough = [traj for traj in trajectories if traj['breakthrough_time'] is None]
        
        if breakthrough_times:
            axes[1, 2].hist(breakthrough_times, bins=15, alpha=0.7, label='Breakthroughs')
        axes[1, 2].axvline(x=np.mean(breakthrough_times) if breakthrough_times else 0, 
                          color='red', linestyle='--', label='Mean')
        axes[1, 2].set_xlabel('Time to Breakthrough (months)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Breakthrough Timing')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('experiment_6_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """Run the complete attention decay experiment"""
        print("=== Experiment 6: Attention Decay Metrics ===\n")
        
        # Generate research trajectories
        trajectories = self._generate_research_trajectories()
        print(f"Generated {len(trajectories)} research trajectories")
        
        # Fit decay models
        decay_models = {}
        for trajectory in trajectories:
            decay_models[trajectory['id']] = self._fit_decay_models(trajectory)
        
        # Calculate attention metrics
        attention_metrics = {}
        for trajectory in trajectories:
            attention_metrics[trajectory['id']] = self._calculate_attention_metrics(trajectory)
        
        # Calculate abandonment thresholds
        abandonment_thresholds = {}
        for trajectory in trajectories:
            models = decay_models[trajectory['id']]
            abandonment_thresholds[trajectory['id']] = self._calculate_abandonment_thresholds(trajectory, models)
        
        # Analyze commitment signals
        commitment_signals = self._analyze_commitment_signals(trajectories)
        
        # Print summary statistics
        print(f"\nAttention Decay Analysis:")
        attention_decay_values = [metrics['attention_decay'] for metrics in attention_metrics.values()]
        print(f"  - Mean attention decay: {np.mean(attention_decay_values):.3f}")
        print(f"  - Std attention decay: {np.std(attention_decay_values):.3f}")
        
        half_life_values = [metrics['half_life'] for metrics in attention_metrics.values() if metrics['half_life'] is not None]
        print(f"  - Mean half-life: {np.mean(half_life_values):.1f} months")
        
        abandonment_values = [thresh['abandonment_threshold'] for thresh in abandonment_thresholds.values()]
        should_abandon_count = sum(1 for thresh in abandonment_thresholds.values() if thresh['should_abandon'])
        print(f"  - Trajectories recommended for abandonment: {should_abandon_count}/{len(trajectories)}")
        
        breakthrough_count = sum(1 for traj in trajectories if traj['breakthrough_time'] is not None)
        print(f"  - Breakthroughs occurred: {breakthrough_count}/{len(trajectories)}")
        
        print(f"\nCommitment Signal Analysis:")
        print(f"  - Mean commitment score: {commitment_signals['commitment_score'].mean():.3f}")
        print(f"  - Mean attention-productivity correlation: {commitment_signals['attention_productivity_corr'].mean():.3f}")
        
        # Create visualizations
        self.plot_results(trajectories, decay_models, attention_metrics, abandonment_thresholds, commitment_signals)
        
        return {
            'trajectories': trajectories,
            'decay_models': decay_models,
            'attention_metrics': attention_metrics,
            'abandonment_thresholds': abandonment_thresholds,
            'commitment_signals': commitment_signals
        }

if __name__ == "__main__":
    experiment = AttentionDecayAnalysis()
    results = experiment.run_experiment() 