import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class ChurnMLPipeline:
    
    def __init__(self, X, y, test_size=0.2, random_state=42, output_dir='powerbi_exports'):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        
        # Will be populated during execution
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = None
        self.all_results = {}
        self.trained_models = {}
        self.failed_models = []
        self.best_model_name = None
        self.cv_fold_results = []
        self.best_fold_models = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def verify_data(self):
        print("\n" + "="*80)
        print("STEP 1: VERIFYING DATA")
        print("="*80)
        
        print(f" X shape: {self.X.shape}")
        print(f" y shape: {self.y.shape}")
        print(f" Features: {self.X.shape[1]}")
        print(f" Samples: {self.X.shape[0]}")
        print(f"\nClass distribution:")
        print(self.y.value_counts())
        print(f"Churn rate: {self.y.mean():.2%}")
        missing = self.X.isnull().sum().sum()
        print(f"\n Missing values: {missing}")
        
        if missing > 0:
            print("  Filling missing values with median...")
            self.X = self.X.fillna(self.X.median())
        
        print("\n Data is ready for modeling!")
        return self
    
    def prepare_train_test(self):
        """Split and scale data"""
        print("\n" + "="*80)
        print("STEP 2: PREPARING TRAIN/TEST SPLIT")
        print("="*80)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, 
            random_state=self.random_state, stratify=self.y
        )
        
        # Scale data
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f" Train size: {len(self.X_train)}")
        print(f" Test size: {len(self.X_test)}")
        return self
    
    def get_all_models(self):
        
        try:
            class_ratio = (self.y == 0).sum() / (self.y == 1).sum()
        except:
            class_ratio = 2.77  
        
        return {
   
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=31,
                min_child_samples=25,      
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,            
                reg_lambda=1.0,             
                min_split_gain=0.01,
                random_state=self.random_state,
                verbose=-1,
                class_weight='balanced',
                n_jobs=-1,
                importance_type='gain'
            ),
            
            'Bagging': BaggingClassifier(
                n_estimators=150,
                max_samples=0.85,
                max_features=0.85,
                bootstrap=True,
                bootstrap_features=False,
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=False
            ),
            
          
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
                scale_pos_weight=class_ratio,
                n_jobs=-1,
                tree_method='hist'
            ),
            
            
            
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),  
                activation='relu',
                solver='adam',
                alpha=0.001,             
                batch_size='auto',
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                shuffle=True,
                random_state=self.random_state,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                tol=1e-4
            ),
            
            'DecisionTree': DecisionTreeClassifier(
                max_depth=12,
                min_samples_split=25,
                min_samples_leaf=12,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced',
                splitter='best',
                criterion='gini'
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=7,             
                weights='distance',         
                algorithm='auto',
                leaf_size=30,
                metric='minkowski',
                p=2,
                n_jobs=-1
            ),
            
            
            'QDA': QuadraticDiscriminantAnalysis(
                reg_param=0.1               
            ),
            
            'LDA': LinearDiscriminantAnalysis(
                solver='svd',
                shrinkage=None
            ),
            
            'NaiveBayes': GaussianNB(
                var_smoothing=1e-9
            ),
            
            'SGD': SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=2000,
                tol=1e-4,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1,
                learning_rate='optimal',
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10
            ),
            
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            'SVC': SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                random_state=self.random_state
            ),
            
            'SuperEnsemble': StackingClassifier(
                estimators=[
                    ('extra', ExtraTreesClassifier(
                        n_estimators=200, max_depth=15, min_samples_split=30,
                        min_samples_leaf=15, max_features='sqrt',
                        class_weight='balanced', n_jobs=-1,
                        random_state=self.random_state
                    )),
                    ('gb', GradientBoostingClassifier(
                        n_estimators=200, learning_rate=0.05, max_depth=5,
                        min_samples_split=25, min_samples_leaf=12,
                        subsample=0.85, random_state=self.random_state
                    )),
                    ('rf', RandomForestClassifier(
                        n_estimators=200, max_depth=15, min_samples_split=30,
                        min_samples_leaf=15, max_features='sqrt',
                        class_weight='balanced', n_jobs=-1,
                        random_state=self.random_state
                    )),
                    ('lgbm', LGBMClassifier(
                        n_estimators=200, learning_rate=0.05, max_depth=8,
                        num_leaves=31, min_child_samples=25,
                        class_weight='balanced', verbose=-1,
                        random_state=self.random_state
                    ))
                ],
                final_estimator=LogisticRegression(
                    class_weight='balanced',
                    random_state=self.random_state
                ),
                cv=5,
                stack_method='predict_proba',
                n_jobs=-1,
                passthrough=False
            )
        }
    
    def cross_validate_all_models(self, fold_configs=[2, 3, 4, 5]):   
        print("\n" + "="*80)
        print("STEP 3: CROSS-VALIDATION ANALYSIS")
        print("="*80)
        print(f"Testing fold configurations: {fold_configs}")
        
        all_models = self.get_all_models()
        
        for n_folds in fold_configs:
            print(f"\n{'='*60}")
            print(f"TESTING WITH {n_folds} FOLDS")
            print(f"{'='*60}")
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            
            for name, model in all_models.items():
                print(f"\n{name} ({n_folds} folds)...", end=' ')
                
                try:
                    start = time.time()
                    
                    # Cross-validation scores for each fold
                    fold_scores = cross_val_score(
                        model, self.X_train_scaled, self.y_train, 
                        cv=cv, scoring='f1', n_jobs=-1
                    )
                    
                    # Store results for each fold
                    for fold_idx, score in enumerate(fold_scores, 1):
                        self.cv_fold_results.append({
                            'model_name': name,
                            'n_folds': n_folds,
                            'fold_number': fold_idx,
                            'f1_score': score,
                            'training_time': time.time() - start
                        })
                    
                    print(f" Mean F1={fold_scores.mean():.4f}, Std={fold_scores.std():.4f}")
                    
                except Exception as e:
                    print(f" {str(e)[:50]}")
                    if name not in self.failed_models:
                        self.failed_models.append(name)
        
        cv_df = pd.DataFrame(self.cv_fold_results)
        cv_summary = cv_df.groupby(['model_name', 'n_folds']).agg({
            'f1_score': ['mean', 'std', 'min', 'max'],
            'training_time': 'mean'
        }).reset_index()
        
        cv_summary.columns = ['model_name', 'n_folds', 'f1_mean', 'f1_std', 
                              'f1_min', 'f1_max', 'avg_training_time']
        
        print(f"\n Cross-validation complete!")
        print(f" Total combinations tested: {len(cv_df)}")
        
        return cv_df
    
    def find_best_fold_config(self):
        """Find best fold configuration for each model"""
        print("\n" + "="*80)
        print("STEP 4: FINDING BEST FOLD CONFIGURATIONS")
        print("="*80)
        
        cv_df = pd.DataFrame(self.cv_fold_results)
        
        # Find best fold configuration for each model
        best_configs = cv_df.groupby(['model_name', 'n_folds']).agg({
            'f1_score': 'mean'
        }).reset_index()
        
        best_configs = best_configs.loc[best_configs.groupby('model_name')['f1_score'].idxmax()]
        
        print("\n Best Fold Configuration per Model:")
        print("="*60)
        for _, row in best_configs.iterrows():
            print(f"{row['model_name']:20s} -> {int(row['n_folds'])} folds (F1={row['f1_score']:.4f})")
        
        return best_configs
    
    def train_with_best_folds(self, best_configs):
        """Train models using their best fold configuration"""
        print("\n" + "="*80)
        print("STEP 5: TRAINING WITH BEST FOLD CONFIGURATIONS")
        print("="*80)
        
        all_models = self.get_all_models()
        
        for _, config in best_configs.iterrows():
            name = config['model_name']
            n_folds = int(config['n_folds'])
            
            if name in self.failed_models:
                continue
            
            print(f"\n{name} (using {n_folds} folds)...", end=' ')
            
            try:
                model = all_models[name]
                start = time.time()
                
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(
                    model, self.X_train_scaled, self.y_train, 
                    cv=cv, scoring='f1', n_jobs=-1
                )
                
                model.fit(self.X_train_scaled, self.y_train)
                
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Confusion matrix
                cm = confusion_matrix(self.y_test, y_pred_test)
                
                self.all_results[name] = {
                    'model_name': name,
                    'best_n_folds': n_folds,
                    'train_accuracy': accuracy_score(self.y_train, y_pred_train),
                    'test_accuracy': accuracy_score(self.y_test, y_pred_test),
                    'precision': precision_score(self.y_test, y_pred_test, zero_division=0),
                    'recall': recall_score(self.y_test, y_pred_test, zero_division=0),
                    'f1': f1_score(self.y_test, y_pred_test, zero_division=0),
                    'auc': roc_auc_score(self.y_test, y_pred_proba),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_min': cv_scores.min(),
                    'cv_max': cv_scores.max(),
                    'true_negative': int(cm[0,0]),
                    'false_positive': int(cm[0,1]),
                    'false_negative': int(cm[1,0]),
                    'true_positive': int(cm[1,1]),
                    'training_time': time.time() - start
                }
                
                self.trained_models[name] = {
                    'model': model,
                    'y_pred_proba': y_pred_proba,
                    'best_n_folds': n_folds
                }
                
                print(f" F1={self.all_results[name]['f1']:.4f}")
                
            except Exception as e:
                print(f" {str(e)[:50]}")
                self.failed_models.append(name)
        
        if self.all_results:
            self.best_model_name = max(self.all_results.items(), 
                                      key=lambda x: x[1]['f1'])[0]
            print(f"\n Overall Best Model: {self.best_model_name}")
            print(f"   F1 Score: {self.all_results[self.best_model_name]['f1']:.4f}")
            print(f"   Best Folds: {self.all_results[self.best_model_name]['best_n_folds']}")
        
        return self
    
    def export_consolidated_results(self):
        """Export all results to CSV files for Power BI"""
        print("\n" + "="*80)
        print("STEP 6: EXPORTING RESULTS FOR POWER BI")
        print("="*80)
        
        results_df = pd.DataFrame(self.all_results).T.reset_index(drop=True)
        results_df = results_df.sort_values('f1', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)
        results_df['performance_category'] = pd.cut(
            results_df['f1'],
            bins=[0, 0.5, 0.6, 0.7, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        results_df['overfit_indicator'] = (
            results_df['train_accuracy'] - results_df['test_accuracy']
        ).apply(lambda x: 'High' if x > 0.1 else 'Moderate' if x > 0.05 else 'Low')
        
        results_df.to_csv(f'{self.output_dir}/01_model_performance_summary.csv', index=False)
        print(f" 01_model_performance_summary.csv ({len(results_df)} models)")
        
        cv_df = pd.DataFrame(self.cv_fold_results)
        cv_df.to_csv(f'{self.output_dir}/02_cv_fold_details.csv', index=False)
        print(f" 02_cv_fold_details.csv ({len(cv_df)} fold results)")
        
        cv_summary = cv_df.groupby(['model_name', 'n_folds']).agg({
            'f1_score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        cv_summary.columns = ['model_name', 'n_folds', 'f1_mean', 'f1_std', 
                              'f1_min', 'f1_max', 'fold_count']
        cv_summary['cv_range'] = cv_summary['f1_max'] - cv_summary['f1_min']
        cv_summary['stability'] = cv_summary['f1_std'].apply(
            lambda x: 'Very Stable' if x < 0.02 else 'Stable' if x < 0.05 else 'Unstable'
        )
        cv_summary.to_csv(f'{self.output_dir}/03_cv_summary_by_folds.csv', index=False)
        print(f" 03_cv_summary_by_folds.csv ({len(cv_summary)} combinations)")
        
        best_fold_config = cv_summary.loc[cv_summary.groupby('model_name')['f1_mean'].idxmax()]
        best_fold_config = best_fold_config[['model_name', 'n_folds', 'f1_mean', 'f1_std', 'stability']]
        best_fold_config.columns = ['model_name', 'best_n_folds', 'best_f1_mean', 'f1_std', 'stability']
        best_fold_config.to_csv(f'{self.output_dir}/04_best_fold_per_model.csv', index=False)
        print(f" 04_best_fold_per_model.csv ({len(best_fold_config)} models)")
        
        metrics_long = []
        metric_types = ['precision', 'recall', 'f1', 'auc', 'test_accuracy', 'cv_mean']
        for _, row in results_df.iterrows():
            for metric in metric_types:
                metrics_long.append({
                    'model_name': row['model_name'],
                    'metric_type': metric.upper().replace('_', ' '),
                    'metric_value': row[metric],
                    'rank': row['rank'],
                    'best_n_folds': row['best_n_folds']
                })
        metrics_df = pd.DataFrame(metrics_long)
        metrics_df.to_csv(f'{self.output_dir}/05_metrics_comparison_long.csv', index=False)
        print(f" 05_metrics_comparison_long.csv ({len(metrics_df)} records)")
        
        confusion_data = []
        for _, row in results_df.iterrows():
            for metric_type, value, category in [
                ('True Negative', row['true_negative'], 'Correct'),
                ('True Positive', row['true_positive'], 'Correct'),
                ('False Positive', row['false_positive'], 'Error'),
                ('False Negative', row['false_negative'], 'Error')
            ]:
                confusion_data.append({
                    'model_name': row['model_name'],
                    'metric_type': metric_type,
                    'value': value,
                    'category': category,
                    'best_n_folds': row['best_n_folds']
                })
        confusion_df = pd.DataFrame(confusion_data)
        confusion_df.to_csv(f'{self.output_dir}/06_confusion_matrix_all_models.csv', index=False)
        print(f" 06_confusion_matrix_all_models.csv ({len(confusion_df)} records)")
        
        training_perf = results_df[['model_name', 'training_time', 'f1', 'auc', 
                                    'test_accuracy', 'cv_mean', 'rank', 'best_n_folds']].copy()
        training_perf['speed_category'] = pd.cut(
            training_perf['training_time'],
            bins=[0, 1, 5, 30, 1000],
            labels=['Very Fast', 'Fast', 'Moderate', 'Slow']
        )
        training_perf.to_csv(f'{self.output_dir}/07_training_performance.csv', index=False)
        print(f" 07_training_performance.csv ({len(training_perf)} records)")
        
        predictions = None
        return results_df
    
    def export_predictions(self):
        """Export prediction results"""
        print("\n Exporting Prediction Results...")
        
        if not self.best_model_name:
            print(" No best model found")
            return
        
        best_proba = self.trained_models[self.best_model_name]['y_pred_proba']
        best_pred = (best_proba >= 0.5).astype(int)
        
        best_predictions = self.X_test.copy()
        best_predictions['actual_churn'] = self.y_test.values
        best_predictions['predicted_churn'] = best_pred
        best_predictions['churn_probability'] = best_proba
        best_predictions['correct_prediction'] = (self.y_test.values == best_pred).astype(int)
        
        
        best_predictions['risk_segment'] = pd.cut(
            best_predictions['churn_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        best_predictions['prediction_confidence'] = best_predictions['churn_probability'].apply(
            lambda x: abs(x - 0.5)
        )
        best_predictions['confidence_category'] = pd.cut(
            best_predictions['prediction_confidence'],
            bins=[0, 0.1, 0.3, 0.5],
            labels=['Low Confidence', 'Medium Confidence', 'High Confidence']
        )
        
        best_predictions.to_csv(f'{self.output_dir}/08_best_model_predictions.csv', index=False)
        print(f" 08_best_model_predictions.csv ({len(best_predictions)} customers)")
        
        risk_summary = best_predictions.groupby('risk_segment').agg({
            'churn_probability': ['count', 'mean', 'min', 'max', 'std'],
            'actual_churn': ['sum', 'mean'],
            'correct_prediction': 'mean'
        }).round(4)
        risk_summary.columns = ['customer_count', 'avg_probability', 'min_probability', 
                               'max_probability', 'std_probability', 'actual_churns', 
                               'churn_rate', 'accuracy']
        risk_summary = risk_summary.reset_index()
        risk_summary.to_csv(f'{self.output_dir}/09_risk_segment_summary.csv', index=False)
        print(f" 09_risk_segment_summary.csv (3 segments)")
        
        high_risk = best_predictions[best_predictions['risk_segment'] == 'High Risk'].copy()
        high_risk = high_risk.sort_values('churn_probability', ascending=False)
        high_risk['priority_rank'] = range(1, len(high_risk) + 1)
        high_risk.to_csv(f'{self.output_dir}/10_high_risk_customers.csv', index=False)
        print(f" 10_high_risk_customers.csv ({len(high_risk)} customers)")
        
        return best_predictions
    
    def export_advanced_analysis(self, best_predictions):
        """Export ROC, PR curves, and advanced metrics"""
        print("\n Exporting Advanced Analysis...")
        
        if best_predictions is None:
                print(" No predictions available")
                return
        
        roc_data = []
        for name, info in self.trained_models.items():
            fpr, tpr, thresholds = roc_curve(self.y_test, info['y_pred_proba'])
            auc_score = self.all_results[name]['auc']
            step = max(1, len(fpr) // 50)
            
            for i in range(0, len(fpr), step):
                roc_data.append({
                    'model_name': name,
                    'model_auc': auc_score,
                    'false_positive_rate': fpr[i],
                    'true_positive_rate': tpr[i],
                    'threshold': thresholds[i] if i < len(thresholds) else 1.0,
                    'best_n_folds': info['best_n_folds']
                })
        
        roc_df = pd.DataFrame(roc_data)
        roc_df.to_csv(f'{self.output_dir}/11_roc_curves.csv', index=False)
        print(f" 11_roc_curves.csv ({len(roc_df)} points)")
        
        pr_data = []
        for name, info in self.trained_models.items():
            precision, recall, thresholds = precision_recall_curve(self.y_test, info['y_pred_proba'])
            step = max(1, len(precision) // 50)
            
            for i in range(0, len(precision), step):
                pr_data.append({
                    'model_name': name,
                    'precision': precision[i],
                    'recall': recall[i],
                    'threshold': thresholds[i] if i < len(thresholds) else 1.0,
                    'f1_score': 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-10),
                    'best_n_folds': info['best_n_folds']
                })
        
        pr_df = pd.DataFrame(pr_data)
        pr_df.to_csv(f'{self.output_dir}/12_precision_recall_curves.csv', index=False)
        print(f" 12_precision_recall_curves.csv ({len(pr_df)} points)")
        
        # Feature Importance
        feature_importance_data = []
        for name, info in self.trained_models.items():
            model = info['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, feature in enumerate(self.X.columns):
                    feature_importance_data.append({
                        'model_name': name,
                        'feature_name': feature,
                        'importance': importances[i],
                        'rank_in_model': np.argsort(importances)[::-1].tolist().index(i) + 1,
                        'best_n_folds': info['best_n_folds']
                    })
        
        if feature_importance_data:
            feature_df = pd.DataFrame(feature_importance_data)
            avg_importance = feature_df.groupby('feature_name')['importance'].mean().reset_index()
            avg_importance.columns = ['feature_name', 'avg_importance']
            avg_importance['overall_rank'] = avg_importance['avg_importance'].rank(ascending=False)
            feature_df = feature_df.merge(avg_importance, on='feature_name')
            feature_df = feature_df.sort_values(['model_name', 'importance'], ascending=[True, False])
            feature_df.to_csv(f'{self.output_dir}/13_feature_importance.csv', index=False)
            print(f" 13_feature_importance.csv ({len(feature_df)} records)")
            
            # Top 20 features
            top_features = avg_importance.sort_values('avg_importance', ascending=False).head(20)
            top_features.to_csv(f'{self.output_dir}/14_top_20_features.csv', index=False)
            print(f" 14_top_20_features.csv (20 features)")
        
        # Cost-Benefit Analysis
        COST_MISSED_CHURN = 50
        COST_FALSE_ALARM = 10
        RETENTION_SUCCESS_RATE = 0.3
        
        cost_benefit_data = []
        for name, info in self.trained_models.items():
            y_pred = (info['y_pred_proba'] >= 0.5).astype(int)
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            cost_missed = fn * COST_MISSED_CHURN
            cost_false_alarms = fp * COST_FALSE_ALARM
            total_cost = cost_missed + cost_false_alarms
            prevented_churn = tp * RETENTION_SUCCESS_RATE
            revenue_saved = prevented_churn * COST_MISSED_CHURN
            net_benefit = revenue_saved - total_cost
            roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0
            
            cost_benefit_data.append({
                'model_name': name,
                'f1_score': self.all_results[name]['f1'],
                'best_n_folds': info['best_n_folds'],
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn,
                'cost_missed_churn': cost_missed,
                'cost_false_alarms': cost_false_alarms,
                'total_cost': total_cost,
                'churns_prevented': int(prevented_churn),
                'revenue_saved': revenue_saved,
                'net_benefit': net_benefit,
                'roi_percentage': roi
            })
        
        cost_benefit_df = pd.DataFrame(cost_benefit_data).sort_values('net_benefit', ascending=False)
        cost_benefit_df['benefit_rank'] = range(1, len(cost_benefit_df) + 1)
        cost_benefit_df.to_csv(f'{self.output_dir}/15_cost_benefit_analysis.csv', index=False)
        print(f" 15_cost_benefit_analysis.csv ({len(cost_benefit_df)} models)")
        
        # Summary Statistics
        summary_stats = {
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model_name,
            'best_f1_score': float(self.all_results[self.best_model_name]['f1']),
            'best_auc_score': float(self.all_results[self.best_model_name]['auc']),
            'best_n_folds': int(self.all_results[self.best_model_name]['best_n_folds']),
            'total_customers': len(self.y_test),
            'actual_churners': int(self.y_test.sum()),
            'churn_rate': float(self.y_test.mean()),
            'high_risk_customers': int((best_predictions['risk_segment'] == 'High Risk').sum()),
            'medium_risk_customers': int((best_predictions['risk_segment'] == 'Medium Risk').sum()),
            'low_risk_customers': int((best_predictions['risk_segment'] == 'Low Risk').sum()),
            'total_features': self.X.shape[1]
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(f'{self.output_dir}/16_summary_statistics.csv', index=False)
        print(f" 16_summary_statistics.csv (1 record)")
    
    def create_data_dictionary(self):
        """Create data dictionary for all exports"""
        print("\n Creating Data Dictionary...")
        
        data_dictionary = {
            '01_model_performance_summary.csv': 'Complete performance metrics with best fold configuration',
            '02_cv_fold_details.csv': 'Individual fold scores for all model-fold combinations',
            '03_cv_summary_by_folds.csv': 'Summary statistics by model and fold configuration',
            '04_best_fold_per_model.csv': 'Best fold configuration identified for each model',
            '05_metrics_comparison_long.csv': 'Long format metrics for easy visualization',
            '06_confusion_matrix_all_models.csv': 'Confusion matrix breakdown for all models',
            '07_training_performance.csv': 'Training time vs accuracy trade-offs',
        }
        
        dict_df = pd.DataFrame(list(data_dictionary.items()), 
                               columns=['filename', 'description'])
        dict_df.to_csv(f'{self.output_dir}/00_DATA_DICTIONARY.csv', index=False)
        print(f" 00_DATA_DICTIONARY.csv (16 files documented)")
    
    def print_final_summary(self):
        """Print final summary of results"""
        print("\n" + "="*80)
        print(" PIPELINE COMPLETE!")
        print("="*80)
        
        print(f"\n LOCATION: ./{self.output_dir}/")
        print(f"\n FILES CREATED: 17 CSV files")
        
        print(f"\n BEST MODEL CONFIGURATION:")
        print(f"   Model: {self.best_model_name}")
        print(f"   F1 Score: {self.all_results[self.best_model_name]['f1']:.4f}")
        print(f"   AUC Score: {self.all_results[self.best_model_name]['auc']:.4f}")
        print(f"   Best Folds: {self.all_results[self.best_model_name]['best_n_folds']}")
        print(f"   CV Mean F1: {self.all_results[self.best_model_name]['cv_mean']:.4f}")
        print(f"   CV Std: {self.all_results[self.best_model_name]['cv_std']:.4f}")
        
        print(f"\n KEY FILES FOR POWER BI:")
        print(f"   1. 01_model_performance_summary.csv - Main dashboard")
        print(f"   2. 02_cv_fold_details.csv - Cross-validation analysis")
        print(f"   3. 04_best_fold_per_model.csv - Optimal fold configurations")
        print(f"   4. 08_best_model_predictions.csv - Customer insights")
        print(f"   5. 15_cost_benefit_analysis.csv - Business value")
        print(f"   6. 10_high_risk_customers.csv - Action plan")
        
     
        
        print(f"\n READ: 00_DATA_DICTIONARY.csv for file descriptions")
        print("\n" + "="*80)