"""
# Notebook 4: Scientific Knowledge Discovery
## Automated Pattern Discovery and Theory Formation

### Learning Objectives
By the end of this notebook, you will be able to:
- Implement automated pattern recognition in scientific data
- Develop symbolic regression systems for equation discovery
- Create causal discovery frameworks
- Build automated theory generation systems
- Validate and interpret discovered patterns
"""

import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, solve, simplify
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
import warnings
warnings.filterwarnings('ignore')

"""
## Section 1: Automated Pattern Recognition
Implementation of advanced pattern detection algorithms for scientific data.
"""

class PatternDetector:
    def __init__(self):
        self.patterns = {}
        self.significance_threshold = 0.05
        
    def detect_periodic_patterns(self, time_series, sampling_rate=1.0):
        """
        Detect periodic patterns in time series data
        
        Parameters:
        -----------
        time_series : array-like
            Time series data
        sampling_rate : float
            Sampling rate of the data
        """
        from scipy.fft import fft, fftfreq
        
        # Compute FFT
        n = len(time_series)
        yf = fft(time_series)
        xf = fftfreq(n, 1/sampling_rate)
        
        # Find dominant frequencies
        peaks, properties = find_peaks(np.abs(yf[:n//2]), height=np.std(np.abs(yf))*2)
        
        periodic_patterns = []
        for peak, height in zip(peaks, properties['peak_heights']):
            frequency = xf[peak]
            period = 1/frequency if frequency != 0 else float('inf')
            
            pattern = {
                'type': 'periodic',
                'frequency': frequency,
                'period': period,
                'amplitude': height/n,
                'significance': height/np.mean(np.abs(yf))
            }
            periodic_patterns.append(pattern)
            
        self.patterns['periodic'] = periodic_patterns
        return periodic_patterns
    
    def detect_trends(self, time_series):
        """
        Detect trends and change points in time series
        """
        from scipy import stats
        
        # Linear trend
        x = np.arange(len(time_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
        
        trend = {
            'type': 'linear',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'significant': p_value < self.significance_threshold
        }
        
        # Change point detection
        from ruptures import Binseg
        
        algorithm = Binseg(model="l2").fit(time_series.reshape(-1, 1))
        change_points = algorithm.predict(n_bkps=3)
        
        self.patterns['trend'] = trend
        self.patterns['change_points'] = change_points
        
        return trend, change_points
    
    def detect_clusters(self, data, eps=0.5, min_samples=5):
        """
        Detect clusters in multivariate data
        """
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(data_scaled)
        
        # Analyze clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_info = []
        
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            info = {
                'size': len(cluster_points),
                'centroid': np.mean(cluster_points, axis=0),
                'variance': np.var(cluster_points, axis=0),
                'density': len(cluster_points) / np.prod(np.ptp(cluster_points, axis=0))
            }
            cluster_info.append(info)
            
        self.patterns['clusters'] = cluster_info
        return cluster_info
    
    def detect_correlations(self, data):
        """
        Detect significant correlations between variables
        """
        correlations = []
        n_vars = data.shape[1]
        
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                corr, p_value = pearsonr(data[:, i], data[:, j])
                if abs(corr) > 0.5 and p_value < self.significance_threshold:
                    correlation = {
                        'variables': (i, j),
                        'correlation': corr,
                        'p_value': p_value,
                        'type': 'positive' if corr > 0 else 'negative'
                    }
                    correlations.append(correlation)
                    
        self.patterns['correlations'] = correlations
        return correlations

"""
## Section 2: Symbolic Regression for Equation Discovery
Implementation of symbolic regression to discover mathematical relationships.
"""

class EquationDiscoverer:
    def __init__(self, complexity_penalty=0.1):
        self.complexity_penalty = complexity_penalty
        self.best_equations = []
        
    def discover_equations(self, X, y, max_depth=3):
        """
        Discover equations that describe relationships in data
        
        Parameters:
        -----------
        X : array-like
            Input variables
        y : array-like
            Target variable
        max_depth : int
            Maximum depth of equation trees
        """
        import gplearn
        from gplearn.genetic import SymbolicRegressor
        
        # Define function set
        function_set = ['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log']
        
        # Initialize symbolic regressor
        est = SymbolicRegressor(
            population_size=1000,
            generations=20,
            function_set=function_set,
            parsimony_coefficient=self.complexity_penalty,
            max_samples=0.9,
            verbose=1,
            random_state=42
        )
        
        # Fit model
        est.fit(X, y)
        
        # Extract and simplify equations
        equations = []
        for program in est._programs[-1]:
            try:
                eq_str = str(program)
                # Convert to SymPy expression
                expr = sp.sympify(eq_str)
                # Simplify expression
                simplified = simplify(expr)
                
                equations.append({
                    'original': eq_str,
                    'simplified': str(simplified),
                    'fitness': program.raw_fitness_,
                    'complexity': program.length_
                })
            except:
                continue
                
        self.best_equations = sorted(equations, key=lambda x: x['fitness'])
        return self.best_equations
    
    def validate_equations(self, X, y, equations):
        """
        Validate discovered equations
        """
        validation_results = []
        
        for eq in equations:
            try:
                # Create lambda function from equation
                expr = sp.sympify(eq['simplified'])
                vars = list(expr.free_symbols)
                f = sp.lambdify(vars, expr)
                
                # Compute predictions
                y_pred = f(*[X[:, i] for i in range(X.shape[1])])
                
                # Calculate metrics
                mse = np.mean((y - y_pred)**2)
                r2 = 1 - mse/np.var(y)
                
                validation = {
                    'equation': eq['simplified'],
                    'mse': mse,
                    'r2': r2,
                    'complexity': eq['complexity']
                }
                validation_results.append(validation)
            except:
                continue
                
        return validation_results

"""
## Section 3: Causal Discovery
Implementation of causal discovery algorithms for scientific data.
"""

class CausalDiscoverer:
    def __init__(self):
        self.model = None
        self.causal_relationships = []
        
    def discover_causal_structure(self, data, threshold=0.1):
        """
        Discover causal relationships in data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        threshold : float
            Threshold for edge inclusion
        """
        # Create structural model
        sm = from_pandas(data, w_threshold=threshold)
        
        # Convert to NetworkX graph
        self.model = nx.DiGraph(sm)
        
        # Analyze causal relationships
        relationships = []
        for edge in self.model.edges(data=True):
            source, target, weight = edge
            relationship = {
                'cause': source,
                'effect': target,
                'strength': weight['weight'],
                'type': 'direct'
            }
            relationships.append(relationship)
            
        # Find indirect relationships
        for source in self.model.nodes():
            for target in self.model.nodes():
                if source != target:
                    paths = list(nx.all_simple_paths(self.model, source, target))
                    if len(paths) > 1:  # Multiple paths indicate indirect relationship
                        relationship = {
                            'cause': source,
                            'effect': target,
                            'paths': paths,
                            'type': 'indirect'
                        }
                        relationships.append(relationship)
                        
        self.causal_relationships = relationships
        return relationships
    
    def validate_causal_relationships(self, data, interventions=None):
        """
        Validate discovered causal relationships
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Validation data
        interventions : dict
            Dictionary of intervention experiments
        """
        from sklearn.metrics import mutual_info_score
        
        validation_results = []
        
        for relationship in self.causal_relationships:
            if relationship['type'] == 'direct':
                cause = relationship['cause']
                effect = relationship['effect']
                
                # Calculate mutual information
                mi = mutual_info_score(data[cause], data[effect])
                
                # Check temporal precedence if time information available
                temporal_score = self._check_temporal_precedence(data, cause, effect)
                
                # Check intervention effects if available
                intervention_score = self._check_interventions(
                    interventions, cause, effect) if interventions else None
                
                validation = {
                    'relationship': (cause, effect),
                    'mutual_information': mi,
                    'temporal_score': temporal_score,
                    'intervention_score': intervention_score,
                    'confidence': (mi + temporal_score + (intervention_score or 0))/3
                }
                validation_results.append(validation)
                
        return validation_results
    
    def _check_temporal_precedence(self, data, cause, effect):
        """Check if cause precedes effect in time"""
        if 'time' in data.columns:
            cause_times = data.groupby(cause)['time'].mean()
            effect_times = data.groupby(effect)['time'].mean()
            return float(cause_times.mean() < effect_times.mean())
        return 0.5  # Neutral score if no temporal information
    
    def _check_interventions(self, interventions, cause, effect):
        """Check if interventions support causal relationship"""
        if (cause, effect) in interventions:
            intervention = interventions[(cause, effect)]
            # Compare intervention effect with observational data
            return intervention['effect_size']
        return None

"""
## Section 4: Theory Generation
Implementation of automated theory generation and testing.
"""

class TheoryGenerator:
    def __init__(self):
        self.theories = []
        self.pattern_detector = PatternDetector()
        self.equation_discoverer = EquationDiscoverer()
        self.causal_discoverer = CausalDiscoverer()
        
    def generate_theories(self, data, context=None):
        """
        Generate scientific theories from data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Scientific data
        context : dict
            Domain knowledge and constraints
        """
        theories = []
        
        # Detect patterns
        patterns = self._analyze_patterns(data)
        
        # Discover equations
        equations = self._discover_equations(data)
        
        # Find causal relationships
        causal_relations = self._discover_causal_relations(data)
        
        # Generate theories by combining evidence
        for pattern in patterns:
            for equation in equations:
                for relation in causal_relations:
                    theory = self._formulate_theory(
                        pattern, equation, relation, context
                    )
                    if theory:
                        theories.append(theory)
                        
        # Rank theories by evidence strength
        theories = sorted(theories, key=lambda x: x['evidence_strength'], reverse=True)
        self.theories = theories
        return theories
    
    def _analyze_patterns(self, data):
        """Analyze patterns in data"""
        patterns = []
        
        # Time series patterns
        if 'time' in data.columns:
            for col in data.columns:
                if col != 'time':
                    periodic = self.pattern_detector.detect_periodic_patterns(data[col])
                    trends = self.pattern_detector.detect_trends(data[col])
                    patterns.extend(periodic)
                    patterns.append(trends[0])
                    
        # Multivariate patterns
        clusters = self.pattern_detector.detect_clusters(data.drop('time', axis=1))
        correlations = self.pattern_detector.detect_correlations(
            data.drop('time', axis=1).values
        )
        
        patterns.extend(clusters)
        patterns.extend(correlations)
        return patterns
    
    def _discover_equations(self, data):
        """Discover equations in data"""
        X = data.drop(['time', 'target'], axis=1).values
        y = data['target'].values
        return self.equation_discoverer.discover_equations(X, y)
    
    def _discover_causal_relations(self, data):
        """Discover causal relationships"""
        return self.causal_discoverer.discover_causal_structure(data)
    
    def _formulate_theory(self, pattern, equation, relation, context):
        """
        Formulate a scientific theory by combining evidence
        """
        # Check consistency with context/domain knowledge
        if context and not self._check_consistency(pattern, equation, relation, context):
            return None
            
        # Combine evidence into a theory
        theory = {
            'patterns': pattern,
            'mathematical_form': equation,
            'causal_structure': relation,
            'predictions': self._generate_predictions(equation, relation),
            'evidence_strength': self._evaluate_evidence(pattern, equation, relation),
            'testable_implications': self._generate_testable_implications(
                pattern, equation, relation
            )
        }
        
        return theory
    
    def _check_consistency(self, pattern, equation, relation, context):
        """Check consistency with domain knowledge"""
        # Implement domain-specific consistency checks
        return True
    
    def _generate_predictions(self, equation, relation):
        """Generate predictions from theory"""
        predictions = []
        # Implement prediction generation
        return predictions