"""
# Notebook 3: Automated Experiment Design and Active Learning
## Optimizing Scientific Discovery Through Intelligent Experimentation

### Learning Objectives
By the end of this notebook, you will be able to:
- Implement Bayesian optimization for experiment design
- Develop uncertainty quantification systems
- Create automated hypothesis generation frameworks
- Build active learning pipelines for scientific discovery
- Design adaptive experimental strategies
"""

import numpy as np
import pandas as pd
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')

"""
## Section 1: Bayesian Optimization Framework
We'll implement a comprehensive Bayesian optimization system for experimental design.
"""

class ExperimentOptimizer:
    def __init__(self, param_bounds, n_initial=10):
        """
        Initialize the experiment optimizer
        
        Parameters:
        -----------
        param_bounds : dict
            Dictionary of parameter bounds {param_name: (lower, upper)}
        n_initial : int
            Number of initial random experiments
        """
        self.param_bounds = param_bounds
        self.n_initial = n_initial
        self.X = []
        self.y = []
        self.model = None
        
    def suggest_experiments(self, n_suggestions=1):
        """
        Suggest next experiments to run
        
        Parameters:
        -----------
        n_suggestions : int
            Number of experiments to suggest
            
        Returns:
        --------
        suggestions : list
            List of parameter dictionaries for suggested experiments
        """
        if len(self.X) < self.n_initial:
            # Random sampling for initial experiments
            suggestions = []
            for _ in range(n_suggestions):
                params = {}
                for param_name, (lower, upper) in self.param_bounds.items():
                    params[param_name] = np.random.uniform(lower, upper)
                suggestions.append(params)
            return suggestions
        
        # Bayesian optimization for subsequent experiments
        X_torch = torch.tensor(self.X, dtype=torch.float32)
        y_torch = torch.tensor(self.y, dtype=torch.float32)
        
        # Initialize and fit GP model
        gp = SingleTaskGP(X_torch, y_torch.reshape(-1, 1))
        fit_gpytorch_model(gp)
        
        # Define acquisition function
        EI = ExpectedImprovement(gp, y_torch.max())
        
        # Optimize acquisition function
        bounds = torch.tensor([[b[0] for b in self.param_bounds.values()],
                             [b[1] for b in self.param_bounds.values()]])
        
        suggested_params = []
        for _ in range(n_suggestions):
            candidate, _ = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100
            )
            suggested_params.append(candidate.detach().numpy())
            
        # Convert to dictionary format
        suggestions = []
        for params in suggested_params:
            suggestion = {name: float(value) 
                        for name, value in zip(self.param_bounds.keys(), params[0])}
            suggestions.append(suggestion)
            
        return suggestions
    
    def add_result(self, parameters, result):
        """
        Add experimental result to the dataset
        
        Parameters:
        -----------
        parameters : dict
            Dictionary of experimental parameters
        result : float
            Experimental result/objective value
        """
        param_values = [parameters[name] for name in self.param_bounds.keys()]
        self.X.append(param_values)
        self.y.append(result)
        
    def get_optimization_status(self):
        """
        Get current optimization status and progress
        
        Returns:
        --------
        status : dict
            Dictionary containing optimization metrics and progress
        """
        if len(self.y) == 0:
            return {"status": "No experiments conducted yet"}
            
        status = {
            "n_experiments": len(self.y),
            "best_result": max(self.y),
            "best_params": dict(zip(self.param_bounds.keys(), 
                                  self.X[np.argmax(self.y)])),
            "mean_result": np.mean(self.y),
            "std_result": np.std(self.y)
        }
        
        return status

"""
## Section 2: Uncertainty Quantification
Implementation of uncertainty quantification methods for experimental results.
"""

class UncertaintyQuantifier:
    def __init__(self, kernel=None):
        """
        Initialize uncertainty quantifier
        
        Parameters:
        -----------
        kernel : gpytorch.kernels
            Kernel function for Gaussian Process
        """
        if kernel is None:
            kernel = C(1.0) * RBF([1.0] * 10)  # Default kernel
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=42
        )
        
    def fit(self, X, y):
        """
        Fit the uncertainty quantification model
        """
        self.gp.fit(X, y)
        
    def predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates
        
        Returns:
        --------
        mean : array
            Mean predictions
        std : array
            Standard deviation of predictions
        """
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std
    
    def compute_confidence_intervals(self, X, confidence=0.95):
        """
        Compute confidence intervals for predictions
        """
        mean, std = self.predict_with_uncertainty(X)
        z_score = norm.ppf((1 + confidence) / 2)
        lower = mean - z_score * std
        upper = mean + z_score * std
        return lower, upper
    
    def plot_uncertainty(self, X, y, X_test):
        """
        Plot predictions with uncertainty bands
        """
        mean, std = self.predict_with_uncertainty(X_test)
        lower, upper = self.compute_confidence_intervals(X_test)
        
        plt.figure(figsize=(12, 6))
        plt.scatter(X, y, c='black', label='Observations')
        plt.plot(X_test, mean, 'r-', label='Mean prediction')
        plt.fill_between(X_test.ravel(), lower, upper, color='r', alpha=0.2,
                        label='95% confidence interval')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title('Predictions with Uncertainty')
        plt.show()

"""
## Section 3: Automated Hypothesis Generation
Framework for generating and testing scientific hypotheses automatically.
"""

class HypothesisGenerator:
    def __init__(self):
        self.hypotheses = []
        self.evidence = {}
        
    def generate_hypotheses(self, data, target_variable):
        """
        Generate potential hypotheses from data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Experimental data
        target_variable : str
            Name of the target variable
        """
        # Correlation analysis
        correlations = data.corr()[target_variable].abs().sort_values(ascending=False)
        
        # Generate hypotheses based on correlations
        for variable, correlation in correlations.items():
            if variable != target_variable and correlation > 0.3:
                hypothesis = {
                    'type': 'correlation',
                    'variables': [variable, target_variable],
                    'strength': correlation,
                    'description': f"{variable} is correlated with {target_variable}"
                }
                self.hypotheses.append(hypothesis)
                
        # Generate interaction hypotheses
        for i, var1 in enumerate(data.columns):
            for var2 in data.columns[i+1:]:
                if var1 != target_variable and var2 != target_variable:
                    interaction = data[var1] * data[var2]
                    corr = abs(interaction.corr(data[target_variable]))
                    if corr > 0.3:
                        hypothesis = {
                            'type': 'interaction',
                            'variables': [var1, var2, target_variable],
                            'strength': corr,
                            'description': f"Interaction between {var1} and {var2} affects {target_variable}"
                        }
                        self.hypotheses.append(hypothesis)
                        
    def test_hypothesis(self, hypothesis, data):
        """
        Test a generated hypothesis
        
        Parameters:
        -----------
        hypothesis : dict
            Hypothesis dictionary
        data : pandas.DataFrame
            Data to test the hypothesis
        """
        if hypothesis['type'] == 'correlation':
            var1, var2 = hypothesis['variables']
            correlation = data[var1].corr(data[var2])
            p_value = self._calculate_correlation_p_value(data[var1], data[var2])
            
            evidence = {
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        elif hypothesis['type'] == 'interaction':
            var1, var2, target = hypothesis['variables']
            interaction = data[var1] * data[var2]
            correlation = interaction.corr(data[target])
            p_value = self._calculate_correlation_p_value(interaction, data[target])
            
            evidence = {
                'interaction_effect': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        self.evidence[str(hypothesis)] = evidence
        return evidence
    
    def _calculate_correlation_p_value(self, x, y):
        """Calculate p-value for correlation"""
        from scipy import stats
        correlation, p_value = stats.pearsonr(x, y)
        return p_value
    
    def rank_hypotheses(self):
        """
        Rank hypotheses based on evidence
        """
        ranked_hypotheses = []
        for hypothesis in self.hypotheses:
            evidence = self.evidence.get(str(hypothesis), {})
            score = 0
            if evidence.get('significant', False):
                score += 1
            score += abs(evidence.get('correlation', 0))
            
            ranked_hypotheses.append({
                'hypothesis': hypothesis,
                'evidence': evidence,
                'score': score
            })
            
        return sorted(ranked_hypotheses, key=lambda x: x['score'], reverse=True)

"""
## Section 4: Active Learning System
Implementation of active learning for efficient experimentation.
"""

class ActiveLearner:
    def __init__(self, base_model, acquisition_function='uncertainty'):
        """
        Initialize active learner
        
        Parameters:
        -----------
        base_model : object
            Base model with fit and predict methods
        acquisition_function : str
            Type of acquisition function to use
        """
        self.model = base_model
        self.acquisition_function = acquisition_function
        self.X_train = None
        self.y_train = None
        
    def initialize(self, X_pool, y_pool, n_initial=5):
        """
        Initialize the active learner with some labeled data
        """
        indices = np.random.choice(len(X_pool), n_initial, replace=False)
        self.X_train = X_pool[indices]
        self.y_train = y_pool[indices]
        
        mask = np.ones(len(X_pool), dtype=bool)
        mask[indices] = False
        self.X_pool = X_pool[mask]
        self.y_pool = y_pool[mask]
        
        self.model.fit(self.X_train, self.y_train)
        
    def select_instances(self, n_instances=1):
        """
        Select instances for labeling
        """
        if self.acquisition_function == 'uncertainty':
            # Uncertainty sampling
            probas = self.model.predict_proba(self.X_pool)
            uncertainties = -np.sum(probas * np.log(probas + 1e-10), axis=1)
            indices = np.argsort(uncertainties)[-n_instances:]
            
        elif self.acquisition_function == 'diversity':
            # Diversity sampling
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(self.X_pool, self.X_train)
            indices = np.argsort(distances.min(axis=1))[-n_instances:]
            
        return indices
    
    def update(self, indices, new_labels):
        """
        Update the model with new labeled data
        """
        # Add new instances to training set
        self.X_train = np.vstack([self.X_train, self.X_pool[indices]])
        self.y_train = np.concatenate([self.y_train, new_labels])
        
        # Remove labeled instances from pool
        mask = np.ones(len(self.X_pool), dtype=bool)
        mask[indices] = False
        self.X_pool = self.X_pool[mask]
        self.y_pool = self.y_pool[mask]
        
        # Retrain model
        self.model.fit(self.X_train, self.y_train)
        
    def get_performance_metrics(self):
        """
        Calculate performance metrics
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        y_pred = self.model.predict(self.X_pool)
        metrics = {
            'accuracy': accuracy_score(self.y_pool, y_pred),
            'f1_score': f1_score(self.y_pool, y_pred, average='weighted'),
            'n_labeled': len(self.X_train),
            'n_unlabeled': len(self.X_pool)
        }
        return metrics

"""
## Section 5: Practical Exercises

1. Bayesian Optimization Exercise:
   - Optimize a multi-parameter chemical reaction
   - Compare different acquisition functions
   - Analyze convergence behavior

2. Uncertainty Quantification Exercise:
   - Implement different uncertainty estimation methods
   - Compare confidence interval accuracy
   - Visualize uncertainty in experimental results

3. Hypothesis Generation Exercise:
   - Generate hypotheses from real experimental data
   - Test and rank generated hypotheses
   - Validate findings with domain knowledge

4. Active Learning Exercise:
   - Implement different sampling strategies
   - Compare learning efficiency
   - Analyze cost-benefit trade-offs

## Additional Resources

1. Bayesian Optimization:
   - "A Tutorial on Bayesian Optimization" (https://arxiv.org/abs/1807.02811)
   - BoTorch documentation (https://botorch.org/)

2. Active Learning:
   - "Active Learning Literature Survey" (http://burrsettles.com/pub/settles.activelearning.pdf)
   - "Active Learning with Python" (https://activelearning.ai/)

3. Experimental Design:
   - "Design and Analysis of Experiments" by Douglas C. Montgomery
   - "Optimal Experimental Design with R" by Dieter Rasch
"""

# Example usage
if __name__ == "__main__":
    # Example: Optimize a simple 2D function
    def objective_function(x, y):
        return -(x**2 + y**2) + np.sin(3*x) + np.cos(3*y)
    