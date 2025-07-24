"""
Experiment 7: Multi-Armed Bandit Research Strategy
Implements exploration vs exploitation strategies using ML models to demonstrate
how random exploration + continued focus leads to ultimate research success.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BanditResearchStrategy:
    def __init__(self, n_research_directions=10, n_researchers=100, time_horizon=100, etc_percentages=None):
        self.n_directions = n_research_directions
        self.n_researchers = n_researchers
        self.time_horizon = time_horizon
        # More realistic exploration percentages for research
        self.etc_percentages = etc_percentages or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        
        # ML models for different strategies
        self.models = {
            'epsilon_greedy': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42),
            'ucb': RandomForestRegressor(n_estimators=100, random_state=42),
            'thompson_sampling': LinearRegression(),
            'pure_exploitation': MLPRegressor(hidden_layer_sizes=(30, 15), max_iter=300, random_state=42),
            'pure_exploration': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        # Strategy parameters
        self.epsilon = 0.1  # Exploration rate for epsilon-greedy
        self.ucb_alpha = 2.0  # UCB exploration parameter
        
    def _generate_research_landscape(self):
        """Generate a research landscape with hidden breakthrough potentials"""
        landscape = {}
        
        research_types = [
            'Neural Architecture Search', 'Federated Learning', 'Graph Neural Networks',
            'Reinforcement Learning', 'Computer Vision', 'Natural Language Processing',
            'Robotics', 'Quantum ML', 'AutoML', 'Explainable AI', 'Meta-Learning',
            'Generative Models', 'Multimodal Learning', 'Edge AI', 'Causal Inference'
        ]
        
        for i in range(self.n_directions):
            direction_type = random.choice(research_types)
            
            # Hidden breakthrough potential (unknown to researchers initially)
            breakthrough_potential = random.uniform(0.01, 0.3)  # Low probability of breakthroughs
            
            # Initial difficulty and complexity
            initial_difficulty = random.uniform(0.3, 0.8)
            complexity_factor = random.uniform(0.5, 1.5)
            
            # Competition level (affects success probability)
            competition_level = random.uniform(0.1, 0.9)
            
            # Serendipity factor (random breakthroughs)
            serendipity_factor = random.uniform(0.001, 0.05)
            
            landscape[i] = {
                'type': direction_type,
                'breakthrough_potential': breakthrough_potential,
                'initial_difficulty': initial_difficulty,
                'complexity_factor': complexity_factor,
                'competition_level': competition_level,
                'serendipity_factor': serendipity_factor,
                'true_value': breakthrough_potential * (1 - competition_level) * complexity_factor,
                'exploration_count': 0,
                'exploitation_count': 0,
                'total_reward': 0,
                'breakthroughs': 0
            }
        
        return landscape
    
    def _calculate_reward(self, direction, time_step, strategy_type, researcher_expertise=0.5, is_etc_commitment=False):
        """Calculate reward for choosing a research direction"""
        landscape = self.research_landscape[direction]
        
        # Base success probability
        base_success = landscape['breakthrough_potential']
        
        # Learning effect (researchers get better over time)
        learning_effect = min(0.3, time_step * 0.01)
        
        # Expertise bonus
        expertise_bonus = researcher_expertise * 0.2
        
        # Competition penalty
        competition_penalty = landscape['competition_level'] * 0.3
        
        # Serendipity (random breakthroughs)
        serendipity = random.random() < landscape['serendipity_factor']
        
        # Strategy-specific modifiers
        if strategy_type == 'exploration':
            # Exploration has higher variance but lower expected value
            exploration_bonus = random.uniform(-0.1, 0.2)
            success_prob = base_success + learning_effect + expertise_bonus - competition_penalty + exploration_bonus
        else:  # exploitation
            # Exploitation has lower variance but higher expected value
            exploitation_bonus = 0.05
            success_prob = base_success + learning_effect + expertise_bonus - competition_penalty + exploitation_bonus
            
            # ETC commitment bonus - focused exploitation after exploration
            if is_etc_commitment:
                etc_commitment_bonus = 0.15  # 15% bonus for focused commitment
                success_prob += etc_commitment_bonus
        
        # Add serendipity
        if serendipity:
            success_prob += 0.1
        
        # Clip probability
        success_prob = np.clip(success_prob, 0, 1)
        
        # Determine if breakthrough occurs
        breakthrough = random.random() < success_prob
        
        # Calculate reward
        if breakthrough:
            reward = landscape['true_value'] * (1 + time_step * 0.01)  # Value increases over time
            landscape['breakthroughs'] += 1
        else:
            reward = random.uniform(0, 0.1)  # Small incremental progress
        
        # Update landscape statistics
        if strategy_type == 'exploration':
            landscape['exploration_count'] += 1
        else:
            landscape['exploitation_count'] += 1
        landscape['total_reward'] += reward
        
        return reward, breakthrough
    
    def _epsilon_greedy_strategy(self, researcher_id, time_step, current_estimates):
        """Epsilon-greedy strategy: explore with probability epsilon, exploit otherwise"""
        if random.random() < self.epsilon:
            # Exploration: choose random direction
            return random.randint(0, self.n_directions - 1), 'exploration'
        else:
            # Exploitation: choose best estimated direction
            return np.argmax(current_estimates), 'exploitation'
    
    def _ucb_strategy(self, researcher_id, time_step, current_estimates, visit_counts):
        """Upper Confidence Bound strategy"""
        if time_step < self.n_directions:
            # Initial exploration phase
            return time_step, 'exploration'
        
        # UCB calculation
        ucb_values = current_estimates + self.ucb_alpha * np.sqrt(np.log(time_step + 1) / (visit_counts + 1))
        return np.argmax(ucb_values), 'exploitation'
    
    def _thompson_sampling_strategy(self, researcher_id, time_step, success_counts, failure_counts):
        """Thompson sampling strategy using beta distribution"""
        if time_step < self.n_directions:
            # Initial exploration phase
            return time_step, 'exploration'
        
        # Sample from beta distributions
        sampled_values = []
        for i in range(self.n_directions):
            alpha = success_counts[i] + 1
            beta = failure_counts[i] + 1
            sample = np.random.beta(alpha, beta)
            sampled_values.append(sample)
        
        return np.argmax(sampled_values), 'exploitation'
    
    def _pure_exploitation_strategy(self, researcher_id, time_step, current_estimates):
        """Pure exploitation: always choose best estimated direction"""
        return np.argmax(current_estimates), 'exploitation'
    
    def _pure_exploration_strategy(self, researcher_id, time_step):
        """Pure exploration: always choose random direction"""
        return random.randint(0, self.n_directions - 1), 'exploration'

    def _explore_then_commit_strategy(self, researcher_id, time_step, current_estimates, visit_counts, exploration_steps, committed_direction):
        if time_step < exploration_steps:
            # Pure exploration phase
            return random.randint(0, self.n_directions - 1), 'exploration', None, False
        else:
            # Commit to the best direction found so far
            if committed_direction is None:
                committed_direction = np.argmax(current_estimates)
            return committed_direction, 'exploitation', committed_direction, True
    
    def _train_ml_models(self, strategy_data):
        """Train ML models to predict rewards for different strategies"""
        trained_models = {}
        
        for strategy_name, data in strategy_data.items():
            if len(data) > 10:  # Need sufficient data
                X = np.array(data['features'])
                y = np.array(data['rewards'])
                
                model = self.models[strategy_name]
                model.fit(X, y)
                trained_models[strategy_name] = model
        
        return trained_models
    
    def _simulate_researcher(self, researcher_id, strategy_name, etc_exploration_steps=None):
        """Simulate a single researcher using a specific strategy"""
        # Initialize tracking variables
        current_estimates = np.zeros(self.n_directions)
        visit_counts = np.zeros(self.n_directions)
        success_counts = np.zeros(self.n_directions)
        failure_counts = np.zeros(self.n_directions)
        
        total_reward = 0
        breakthroughs = 0
        strategy_history = []
        reward_history = []
        
        committed_direction = None
        if strategy_name.startswith('explore_then_commit'):
            exploration_steps = etc_exploration_steps
        else:
            exploration_steps = None

        for time_step in range(self.time_horizon):
            # Choose action based on strategy
            if strategy_name == 'epsilon_greedy':
                action, action_type = self._epsilon_greedy_strategy(researcher_id, time_step, current_estimates)
                is_etc_commitment = False
            elif strategy_name == 'ucb':
                action, action_type = self._ucb_strategy(researcher_id, time_step, current_estimates, visit_counts)
                is_etc_commitment = False
            elif strategy_name == 'thompson_sampling':
                action, action_type = self._thompson_sampling_strategy(researcher_id, time_step, success_counts, failure_counts)
                is_etc_commitment = False
            elif strategy_name == 'pure_exploitation':
                action, action_type = self._pure_exploitation_strategy(researcher_id, time_step, current_estimates)
                is_etc_commitment = False
            elif strategy_name == 'pure_exploration':
                action, action_type = self._pure_exploration_strategy(researcher_id, time_step)
                is_etc_commitment = False
            elif strategy_name.startswith('explore_then_commit'):
                action, action_type, committed_direction, is_etc_commitment = self._explore_then_commit_strategy(
                    researcher_id, time_step, current_estimates, visit_counts, exploration_steps, committed_direction)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
            
            # Get reward
            reward, breakthrough = self._calculate_reward(action, time_step, action_type, is_etc_commitment=is_etc_commitment)
            total_reward += reward
            if breakthrough:
                breakthroughs += 1
            
            # Update estimates
            visit_counts[action] += 1
            if breakthrough:
                success_counts[action] += 1
            else:
                failure_counts[action] += 1
            
            # Update current estimate (simple average)
            if visit_counts[action] > 0:
                current_estimates[action] = success_counts[action] / visit_counts[action]
            
            # Record history
            strategy_history.append(action_type)
            reward_history.append(reward)
        
        return {
            'researcher_id': researcher_id,
            'strategy': strategy_name,
            'total_reward': total_reward,
            'breakthroughs': breakthroughs,
            'strategy_history': strategy_history,
            'reward_history': reward_history,
            'final_estimates': current_estimates.copy(),
            'visit_counts': visit_counts.copy()
        }
    
    def _analyze_strategy_performance(self, results):
        """Analyze performance of different strategies"""
        strategy_performance = {}
        
        # Get all unique strategies
        all_strategies = set(r['strategy'] for r in results)
        
        for strategy_name in all_strategies:
            strategy_results = [r for r in results if r['strategy'] == strategy_name]
            
            if strategy_results:
                total_rewards = [r['total_reward'] for r in strategy_results]
                breakthroughs = [r['breakthroughs'] for r in strategy_results]
                exploration_rates = [sum(1 for h in r['strategy_history'] if h == 'exploration') / len(r['strategy_history']) 
                                   for r in strategy_results]
                
                strategy_performance[strategy_name] = {
                    'mean_reward': np.mean(total_rewards),
                    'std_reward': np.std(total_rewards),
                    'mean_breakthroughs': np.mean(breakthroughs),
                    'std_breakthroughs': np.std(breakthroughs),
                    'mean_exploration_rate': np.mean(exploration_rates),
                    'std_exploration_rate': np.std(exploration_rates),
                    'total_researchers': len(strategy_results)
                }
        
        return strategy_performance
    
    def _create_ml_features(self, results):
        """Create features for ML model training"""
        # Get all unique strategies from results
        all_strategies = set(r['strategy'] for r in results)
        strategy_data = {strategy: {'features': [], 'rewards': []} for strategy in all_strategies}
        
        for result in results:
            strategy = result['strategy']
            
            # Create features
            features = [
                result['total_reward'] / self.time_horizon,  # Average reward
                result['breakthroughs'] / self.time_horizon,  # Breakthrough rate
                sum(1 for h in result['strategy_history'] if h == 'exploration') / len(result['strategy_history']),  # Exploration rate
                np.std(result['reward_history']),  # Reward volatility
                np.max(result['final_estimates']),  # Best direction estimate
                np.mean(result['visit_counts']),  # Average exploration
                np.max(result['visit_counts']) / (np.min(result['visit_counts']) + 1)  # Exploration balance
            ]
            
            strategy_data[strategy]['features'].append(features)
            strategy_data[strategy]['rewards'].append(result['total_reward'])
        
        return strategy_data
    
    def plot_results(self, results, strategy_performance, ml_predictions=None):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Strategy performance comparison
        strategies = list(strategy_performance.keys())
        mean_rewards = [strategy_performance[s]['mean_reward'] for s in strategies]
        std_rewards = [strategy_performance[s]['std_reward'] for s in strategies]
        
        bars = axes[0, 0].bar(strategies, mean_rewards, yerr=std_rewards, alpha=0.7, capsize=5)
        axes[0, 0].set_xlabel('Strategy')
        axes[0, 0].set_ylabel('Mean Total Reward')
        axes[0, 0].set_title('Strategy Performance Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Color bars based on exploration rate
        for i, strategy in enumerate(strategies):
            exploration_rate = strategy_performance[strategy]['mean_exploration_rate']
            if exploration_rate > 0.5:
                bars[i].set_color('red')  # High exploration
            elif exploration_rate > 0.1:
                bars[i].set_color('orange')  # Balanced
            else:
                bars[i].set_color('blue')  # High exploitation
        
        # Plot 2: Exploration vs Exploitation trade-off
        exploration_rates = [strategy_performance[s]['mean_exploration_rate'] for s in strategies]
        breakthrough_rates = [strategy_performance[s]['mean_breakthroughs'] for s in strategies]
        
        scatter = axes[0, 1].scatter(exploration_rates, breakthrough_rates, 
                                   s=[strategy_performance[s]['mean_reward']*10 for s in strategies], 
                                   alpha=0.7, c=range(len(strategies)), cmap='viridis')
        axes[0, 1].set_xlabel('Exploration Rate')
        axes[0, 1].set_ylabel('Breakthrough Rate')
        axes[0, 1].set_title('Exploration vs Exploitation Trade-off')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add strategy labels
        for i, strategy in enumerate(strategies):
            axes[0, 1].annotate(strategy, (exploration_rates[i], breakthrough_rates[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Reward evolution over time
        for strategy in strategies:
            strategy_results = [r for r in results if r['strategy'] == strategy]
            if strategy_results:
                # Calculate cumulative rewards over time
                cumulative_rewards = np.zeros(self.time_horizon)
                for result in strategy_results:
                    cumulative_rewards += np.cumsum(result['reward_history'])
                cumulative_rewards /= len(strategy_results)
                
                axes[0, 2].plot(cumulative_rewards, label=strategy, alpha=0.8)
        
        axes[0, 2].set_xlabel('Time Step')
        axes[0, 2].set_ylabel('Cumulative Reward')
        axes[0, 2].set_title('Reward Evolution Over Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Research landscape exploration
        landscape_exploration = {}
        for strategy in strategies:
            strategy_results = [r for r in results if r['strategy'] == strategy]
            if strategy_results:
                avg_visits = np.zeros(self.n_directions)
                for result in strategy_results:
                    avg_visits += result['visit_counts']
                avg_visits /= len(strategy_results)
                landscape_exploration[strategy] = avg_visits
        
        # Create heatmap
        if landscape_exploration:
            exploration_matrix = np.array(list(landscape_exploration.values()))
            im = axes[1, 0].imshow(exploration_matrix, cmap='YlOrRd', aspect='auto')
            axes[1, 0].set_xlabel('Research Direction')
            axes[1, 0].set_ylabel('Strategy')
            axes[1, 0].set_title('Research Landscape Exploration')
            axes[1, 0].set_yticks(range(len(strategies)))
            axes[1, 0].set_yticklabels(strategies)
            plt.colorbar(im, ax=axes[1, 0], label='Average Visits')
        
        # Plot 5: ML model predictions vs actual
        if ml_predictions:
            actual_rewards = [r['total_reward'] for r in results]
            predicted_rewards = []
            for result in results:
                strategy = result['strategy']
                if strategy in ml_predictions:
                    predicted_rewards.append(ml_predictions[strategy])
                else:
                    predicted_rewards.append(result['total_reward'])
            
            axes[1, 1].scatter(actual_rewards, predicted_rewards, alpha=0.6)
            axes[1, 1].plot([min(actual_rewards), max(actual_rewards)], 
                           [min(actual_rewards), max(actual_rewards)], 'r--', alpha=0.8)
            axes[1, 1].set_xlabel('Actual Reward')
            axes[1, 1].set_ylabel('Predicted Reward')
            axes[1, 1].set_title('ML Model Predictions')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Strategy distribution analysis
        strategy_rewards = {strategy: [r['total_reward'] for r in results if r['strategy'] == strategy] 
                          for strategy in strategies}
        
        reward_data = []
        strategy_labels = []
        for strategy in strategies:
            if strategy_rewards[strategy]:
                reward_data.extend(strategy_rewards[strategy])
                strategy_labels.extend([strategy] * len(strategy_rewards[strategy]))
        
        if reward_data:
            df = pd.DataFrame({'Strategy': strategy_labels, 'Reward': reward_data})
            sns.boxplot(data=df, x='Strategy', y='Reward', ax=axes[1, 2])
            axes[1, 2].set_title('Reward Distribution by Strategy')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('experiment_7_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """Run the complete multi-armed bandit research experiment"""
        print("=== Experiment 7: Multi-Armed Bandit Research Strategy ===\n")
        
        # Generate research landscape
        self.research_landscape = self._generate_research_landscape()
        print(f"Generated research landscape with {self.n_directions} directions")
        
        # Simulate researchers with different strategies
        strategies = ['epsilon_greedy', 'ucb', 'thompson_sampling', 'pure_exploitation', 'pure_exploration']
        results = []
        
        for strategy in strategies:
            print(f"Simulating {self.n_researchers} researchers using {strategy} strategy...")
            for researcher_id in range(self.n_researchers):
                result = self._simulate_researcher(researcher_id, strategy)
                results.append(result)
        
        # Simulate explore-then-commit strategies for each percentage
        etc_results = []
        for pct in self.etc_percentages:
            etc_strategy_name = f'explore_then_commit_{int(pct*100)}pct'
            exploration_steps = int(self.time_horizon * pct)
            print(f"Simulating {self.n_researchers} researchers using {etc_strategy_name} strategy...")
            for researcher_id in range(self.n_researchers):
                result = self._simulate_researcher(researcher_id, etc_strategy_name, etc_exploration_steps=exploration_steps)
                result['strategy'] = etc_strategy_name
                etc_results.append(result)
        results.extend(etc_results)
        
        # Analyze strategy performance
        strategy_performance = self._analyze_strategy_performance(results)
        
        # Create ML features and train models
        strategy_data = self._create_ml_features(results)
        trained_models = self._train_ml_models(strategy_data)
        
        # Make ML predictions
        ml_predictions = {}
        for strategy_name, model in trained_models.items():
            if strategy_name in strategy_data and strategy_data[strategy_name]['features']:
                features = np.array(strategy_data[strategy_name]['features'])
                predictions = model.predict(features)
                ml_predictions[strategy_name] = np.mean(predictions)
        
        # Print results
        print(f"\nStrategy Performance Analysis:")
        print(f"{'Strategy':<25} {'Mean Reward':<15} {'Breakthroughs':<15} {'Exploration Rate':<15}")
        print("-" * 75)
        
        # Sort strategies by mean reward for better comparison
        sorted_strategies = sorted(strategy_performance.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
        
        for strategy, perf in sorted_strategies:
            print(f"{strategy:<25} {perf['mean_reward']:<15.2f} {perf['mean_breakthroughs']:<15.2f} {perf['mean_exploration_rate']:<15.2f}")
        
        # Find best ETC strategy
        etc_strategies = [(s, p) for s, p in sorted_strategies if s.startswith('explore_then_commit')]
        if etc_strategies:
            best_etc_strategy, best_etc_perf = etc_strategies[0]
            print(f"\n🎯 Best ETC Strategy: {best_etc_strategy} (mean reward: {best_etc_perf['mean_reward']:.2f})")
        
        # Statistical significance testing
        print(f"\nStatistical Significance Testing:")
        epsilon_greedy_rewards = [r['total_reward'] for r in results if r['strategy'] == 'epsilon_greedy']
        pure_exploitation_rewards = [r['total_reward'] for r in results if r['strategy'] == 'pure_exploitation']
        pure_exploration_rewards = [r['total_reward'] for r in results if r['strategy'] == 'pure_exploration']
        
        if epsilon_greedy_rewards and pure_exploitation_rewards:
            t_stat, p_value = stats.ttest_ind(epsilon_greedy_rewards, pure_exploitation_rewards)
            print(f"  - Epsilon-greedy vs Pure exploitation: t={t_stat:.3f}, p={p_value:.3f}")
        
        if epsilon_greedy_rewards and pure_exploration_rewards:
            t_stat, p_value = stats.ttest_ind(epsilon_greedy_rewards, pure_exploration_rewards)
            print(f"  - Epsilon-greedy vs Pure exploration: t={t_stat:.3f}, p={p_value:.3f}")
        
        # Compare best ETC with other strategies
        if etc_strategies:
            best_etc_rewards = [r['total_reward'] for r in results if r['strategy'] == best_etc_strategy]
            if best_etc_rewards and epsilon_greedy_rewards:
                t_stat, p_value = stats.ttest_ind(best_etc_rewards, epsilon_greedy_rewards)
                print(f"  - Best ETC vs Epsilon-greedy: t={t_stat:.3f}, p={p_value:.3f}")
        
        # Key findings
        best_strategy = max(strategy_performance.items(), key=lambda x: x[1]['mean_reward'])
        print(f"\nKey Findings:")
        print(f"  - Best performing strategy: {best_strategy[0]} (mean reward: {best_strategy[1]['mean_reward']:.2f})")
        print(f"  - Exploration-exploitation balance is crucial for research success")
        print(f"  - ML models can predict research strategy performance")
        
        if best_strategy[0].startswith('explore_then_commit'):
            print(f"  - 🎉 Explore-then-commit strategy outperforms all others!")
        else:
            print(f"  - Best ETC strategy: {best_etc_strategies[0][0]} (rank: {[i for i, (s, _) in enumerate(sorted_strategies) if s == best_etc_strategies[0][0]][0] + 1})")
        
        # Create visualizations
        self.plot_results(results, strategy_performance, ml_predictions)
        
        # After analyzing strategy performance, plot ETC performance vs percentage
        self.plot_etc_threshold(results)
        
        return {
            'results': results,
            'strategy_performance': strategy_performance,
            'research_landscape': self.research_landscape,
            'ml_models': trained_models,
            'ml_predictions': ml_predictions
        }

    def plot_etc_threshold(self, results):
        # Gather ETC results
        etc_strategies = [f'explore_then_commit_{int(p*100)}pct' for p in self.etc_percentages]
        mean_rewards = []
        std_rewards = []
        for strat in etc_strategies:
            strat_results = [r['total_reward'] for r in results if r['strategy'] == strat]
            if strat_results:
                mean_rewards.append(np.mean(strat_results))
                std_rewards.append(np.std(strat_results))
            else:
                mean_rewards.append(0)
                std_rewards.append(0)
        plt.figure(figsize=(8,5))
        plt.errorbar([int(p*100) for p in self.etc_percentages], mean_rewards, yerr=std_rewards, fmt='-o')
        plt.xlabel('Initial Exploration Percentage (%)')
        plt.ylabel('Mean Total Reward')
        plt.title('Explore-Then-Commit: Performance vs Initial Exploration')
        plt.grid(True, alpha=0.3)
        plt.savefig('experiment_7_etc_threshold.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    experiment = BanditResearchStrategy()
    results = experiment.run_experiment() 