"""
Experiment 4: Degeneracy Test Implementation
Implements the degeneracy test for identifying "random but fertile" problems that are
suitable for the Random Focus Principle.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
import random

class DegeneracyTest:
    def __init__(self):
        self.problem_characteristics = [
            'theoretical_depth', 'empirical_tractability', 'computational_complexity',
            'data_availability', 'evaluation_metrics', 'interdisciplinary_potential',
            'industrial_relevance', 'academic_interest', 'methodological_novelty',
            'scalability', 'interpretability', 'robustness', 'efficiency',
            'generalization', 'fairness', 'privacy', 'sustainability'
        ]
        
    def _generate_problem_space(self, n_problems=100):
        """Generate a diverse set of ML problems with various characteristics"""
        problems = []
        
        problem_types = [
            'Computer Vision', 'NLP', 'Reinforcement Learning', 'Generative Models',
            'Graph Neural Networks', 'Federated Learning', 'Meta Learning',
            'Neural Architecture Search', 'Explainable AI', 'Robotics',
            'Multimodal Learning', 'Self-Supervised Learning', 'Few-Shot Learning',
            'Continual Learning', 'Neural Rendering', 'Audio Processing',
            'Time Series', 'Anomaly Detection', 'Causal Inference', 'Quantum ML'
        ]
        
        for i in range(n_problems):
            problem_type = random.choice(problem_types)
            
            # Generate characteristics vector
            characteristics = {}
            for char in self.problem_characteristics:
                # Different distributions for different characteristics
                if char in ['theoretical_depth', 'methodological_novelty']:
                    # Bimodal distribution for fertile problems
                    if random.random() < 0.3:  # 30% chance of high value
                        characteristics[char] = random.uniform(0.7, 1.0)
                    else:
                        characteristics[char] = random.uniform(0.1, 0.6)
                elif char in ['empirical_tractability', 'data_availability']:
                    # Uniform distribution
                    characteristics[char] = random.uniform(0.1, 1.0)
                elif char in ['computational_complexity']:
                    # Inverse relationship with tractability
                    tractability = characteristics.get('empirical_tractability', random.uniform(0.1, 1.0))
                    characteristics[char] = 1.0 - tractability + random.normalvariate(0, 0.1)
                else:
                    # Normal distribution
                    characteristics[char] = random.normalvariate(0.5, 0.2)
                
                # Clamp to [0, 1]
                characteristics[char] = max(0.0, min(1.0, characteristics[char]))
            
            # Calculate degeneracy score
            degeneracy_score = self._calculate_degeneracy_score(characteristics)
            
            # Determine if fertile
            fertile = degeneracy_score > 0.6  # Threshold for fertile problems
            
            problems.append({
                'id': i,
                'type': problem_type,
                'characteristics': characteristics,
                'degeneracy_score': degeneracy_score,
                'fertile': fertile,
                'expected_impact': self._estimate_impact(characteristics, degeneracy_score)
            })
        
        return pd.DataFrame(problems)
    
    def _calculate_degeneracy_score(self, characteristics):
        """Calculate degeneracy score based on problem characteristics"""
        # Key factors for degeneracy (non-degenerate = fertile)
        theoretical_depth = characteristics['theoretical_depth']
        empirical_tractability = characteristics['empirical_tractability']
        methodological_novelty = characteristics['methodological_novelty']
        interdisciplinary_potential = characteristics['interdisciplinary_potential']
        
        # Computational complexity should be moderate (not too easy, not too hard)
        computational_complexity = characteristics['computational_complexity']
        complexity_score = 1.0 - abs(computational_complexity - 0.5) * 2
        
        # Data availability should be sufficient but not overwhelming
        data_availability = characteristics['data_availability']
        data_score = 1.0 - abs(data_availability - 0.7) * 1.5
        
        # Calculate degeneracy score
        degeneracy_score = (
            0.3 * theoretical_depth +
            0.2 * empirical_tractability +
            0.25 * methodological_novelty +
            0.15 * interdisciplinary_potential +
            0.05 * complexity_score +
            0.05 * data_score
        )
        
        return degeneracy_score
    
    def _estimate_impact(self, characteristics, degeneracy_score):
        """Estimate expected impact of a problem"""
        # Base impact from degeneracy
        base_impact = degeneracy_score * 100
        
        # Additional factors
        industrial_relevance = characteristics['industrial_relevance']
        academic_interest = characteristics['academic_interest']
        scalability = characteristics['scalability']
        
        # Impact multiplier
        impact_multiplier = (
            0.4 * industrial_relevance +
            0.4 * academic_interest +
            0.2 * scalability
        )
        
        return base_impact * (1 + impact_multiplier)
    
    def _apply_degeneracy_test(self, problems_df):
        """Apply the degeneracy test to identify fertile problems"""
        # Extract characteristics matrix
        char_matrix = np.array([list(p.values()) for p in problems_df['characteristics']])
        
        # Standardize features
        scaler = StandardScaler()
        char_matrix_scaled = scaler.fit_transform(char_matrix)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        char_pca = pca.fit_transform(char_matrix_scaled)
        
        # Cluster problems
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(char_pca)
        
        # Analyze clusters
        cluster_analysis = []
        for cluster_id in range(4):
            cluster_mask = clusters == cluster_id
            cluster_problems = problems_df[cluster_mask]
            
            avg_degeneracy = cluster_problems['degeneracy_score'].mean()
            avg_impact = cluster_problems['expected_impact'].mean()
            fertile_ratio = cluster_problems['fertile'].mean()
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'size': len(cluster_problems),
                'avg_degeneracy': avg_degeneracy,
                'avg_impact': avg_impact,
                'fertile_ratio': fertile_ratio,
                'cluster_quality': avg_degeneracy * fertile_ratio
            })
        
        return {
            'clusters': clusters,
            'cluster_analysis': pd.DataFrame(cluster_analysis),
            'pca_components': char_pca,
            'explained_variance': pca.explained_variance_ratio_
        }
    
    def _analyze_problem_network(self, problems_df):
        """Analyze problem relationships and identify clusters"""
        # Create similarity matrix
        n_problems = len(problems_df)
        similarity_matrix = np.zeros((n_problems, n_problems))
        
        for i in range(n_problems):
            for j in range(n_problems):
                if i != j:
                    char_i = list(problems_df.iloc[i]['characteristics'].values())
                    char_j = list(problems_df.iloc[j]['characteristics'].values())
                    similarity = 1.0 - np.mean(np.abs(np.array(char_i) - np.array(char_j)))
                    similarity_matrix[i, j] = similarity
        
        # Create network
        G = nx.Graph()
        for i in range(n_problems):
            G.add_node(i, **problems_df.iloc[i])
        
        # Add edges based on similarity
        threshold = 0.7
        for i in range(n_problems):
            for j in range(i+1, n_problems):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        # Analyze network properties
        network_analysis = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'avg_shortest_path': nx.average_shortest_path_length(G) if nx.is_connected(G) else float('inf'),
            'communities': list(nx.community.greedy_modularity_communities(G))
        }
        
        return G, network_analysis
    
    def _validate_degeneracy_test(self, problems_df, cluster_results):
        """Validate the degeneracy test performance"""
        # Test predictive power
        fertile_problems = problems_df[problems_df['fertile'] == True]
        non_fertile_problems = problems_df[problems_df['fertile'] == False]
        
        # Compare degeneracy scores
        fertile_scores = fertile_problems['degeneracy_score']
        non_fertile_scores = non_fertile_problems['degeneracy_score']
        
        t_stat, p_value = stats.ttest_ind(fertile_scores, non_fertile_scores)
        
        # Calculate AUC-like metric
        threshold = 0.6
        true_positives = sum(fertile_scores > threshold)
        false_positives = sum(non_fertile_scores > threshold)
        true_negatives = sum(non_fertile_scores <= threshold)
        false_negatives = sum(fertile_scores <= threshold)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fertile_mean_score': fertile_scores.mean(),
            'non_fertile_mean_score': non_fertile_scores.mean()
        }
    
    def plot_results(self, problems_df, cluster_results, network_analysis):
        """Create visualizations of degeneracy test results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Degeneracy score distribution
        fertile_scores = problems_df[problems_df['fertile']]['degeneracy_score']
        non_fertile_scores = problems_df[~problems_df['fertile']]['degeneracy_score']
        
        axes[0, 0].hist(non_fertile_scores, alpha=0.7, label='Non-fertile', bins=20, color='red')
        axes[0, 0].hist(fertile_scores, alpha=0.7, label='Fertile', bins=20, color='blue')
        axes[0, 0].axvline(x=0.6, color='black', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Degeneracy Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Degeneracy Score Distribution')
        axes[0, 0].legend()
        
        # Plot 2: PCA visualization
        pca_components = cluster_results['pca_components']
        clusters = cluster_results['clusters']
        fertile = problems_df['fertile'].values
        
        colors = ['red' if not f else 'blue' for f in fertile]
        axes[0, 1].scatter(pca_components[:, 0], pca_components[:, 1], c=colors, alpha=0.7)
        axes[0, 1].set_xlabel(f'PC1 ({cluster_results["explained_variance"][0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({cluster_results["explained_variance"][1]:.2%} variance)')
        axes[0, 1].set_title('PCA of Problem Characteristics')
        
        # Plot 3: Impact vs Degeneracy
        axes[0, 2].scatter(problems_df['degeneracy_score'], problems_df['expected_impact'], 
                          c=colors, alpha=0.7)
        axes[0, 2].set_xlabel('Degeneracy Score')
        axes[0, 2].set_ylabel('Expected Impact')
        axes[0, 2].set_title('Impact vs Degeneracy Score')
        
        # Plot 4: Cluster analysis
        cluster_analysis = cluster_results['cluster_analysis']
        x = range(len(cluster_analysis))
        width = 0.35
        
        axes[1, 0].bar([i - width/2 for i in x], cluster_analysis['avg_degeneracy'], 
                      width, label='Avg Degeneracy', alpha=0.7)
        axes[1, 0].bar([i + width/2 for i in x], cluster_analysis['fertile_ratio'], 
                      width, label='Fertile Ratio', alpha=0.7)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Cluster Analysis')
        axes[1, 0].set_xticks(x)
        axes[1, 0].legend()
        
        # Plot 5: Characteristic importance
        char_importance = []
        for char in self.problem_characteristics:
            char_values = [p[char] for p in problems_df['characteristics']]
            fertile_values = [char_values[i] for i in range(len(char_values)) if problems_df.iloc[i]['fertile']]
            non_fertile_values = [char_values[i] for i in range(len(char_values)) if not problems_df.iloc[i]['fertile']]
            
            if fertile_values and non_fertile_values:
                t_stat, p_val = stats.ttest_ind(fertile_values, non_fertile_values)
                char_importance.append({
                    'characteristic': char,
                    't_statistic': abs(t_stat),
                    'p_value': p_val,
                    'effect_size': abs(np.mean(fertile_values) - np.mean(non_fertile_values))
                })
        
        char_importance_df = pd.DataFrame(char_importance)
        char_importance_df = char_importance_df.sort_values('t_statistic', ascending=False)
        
        top_chars = char_importance_df.head(8)
        axes[1, 1].barh(range(len(top_chars)), top_chars['t_statistic'])
        axes[1, 1].set_yticks(range(len(top_chars)))
        axes[1, 1].set_yticklabels(top_chars['characteristic'], fontsize=8)
        axes[1, 1].set_xlabel('|t-statistic|')
        axes[1, 1].set_title('Most Discriminative Characteristics')
        
        # Plot 6: Problem type distribution
        fertile_types = problems_df[problems_df['fertile']]['type'].value_counts()
        non_fertile_types = problems_df[~problems_df['fertile']]['type'].value_counts()
        
        all_types = list(set(fertile_types.index) | set(non_fertile_types.index))
        fertile_counts = [fertile_types.get(t, 0) for t in all_types]
        non_fertile_counts = [non_fertile_types.get(t, 0) for t in all_types]
        
        x = np.arange(len(all_types))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, fertile_counts, width, label='Fertile', alpha=0.7)
        axes[1, 2].bar(x + width/2, non_fertile_counts, width, label='Non-fertile', alpha=0.7)
        axes[1, 2].set_xlabel('Problem Type')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Problem Type Distribution')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(all_types, rotation=45, ha='right', fontsize=8)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('experiment_4_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_experiment(self):
        """Run the complete degeneracy test experiment"""
        print("=== Experiment 4: Degeneracy Test Implementation ===\n")
        
        # Generate problem space
        problems_df = self._generate_problem_space()
        print(f"Generated {len(problems_df)} problems")
        print(f"Fertile problems: {problems_df['fertile'].sum()}")
        print(f"Non-fertile problems: {(~problems_df['fertile']).sum()}")
        
        # Apply degeneracy test
        cluster_results = self._apply_degeneracy_test(problems_df)
        print(f"\nCluster Analysis:")
        print(cluster_results['cluster_analysis'].to_string())
        
        # Analyze problem network
        G, network_analysis = self._analyze_problem_network(problems_df)
        print(f"\nNetwork Analysis:")
        for key, value in network_analysis.items():
            if key != 'communities':
                print(f"  {key}: {value}")
        
        # Validate degeneracy test
        validation_results = self._validate_degeneracy_test(problems_df, cluster_results)
        print(f"\nDegeneracy Test Validation:")
        print(f"  - T-statistic: {validation_results['t_statistic']:.3f}")
        print(f"  - P-value: {validation_results['p_value']:.6f}")
        print(f"  - Precision: {validation_results['precision']:.3f}")
        print(f"  - Recall: {validation_results['recall']:.3f}")
        print(f"  - F1-score: {validation_results['f1_score']:.3f}")
        print(f"  - Fertile mean score: {validation_results['fertile_mean_score']:.3f}")
        print(f"  - Non-fertile mean score: {validation_results['non_fertile_mean_score']:.3f}")
        
        # Create visualizations
        self.plot_results(problems_df, cluster_results, network_analysis)
        
        return {
            'problems_df': problems_df,
            'cluster_results': cluster_results,
            'network_analysis': network_analysis,
            'validation_results': validation_results
        }

if __name__ == "__main__":
    experiment = DegeneracyTest()
    results = experiment.run_experiment() 