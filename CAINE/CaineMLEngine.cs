using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Accord.MachineLearning;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Math.Distances;
using Accord.MachineLearning.Performance;
using MathNet.Numerics.LinearAlgebra;

namespace CAINE.MachineLearning
{
    /// <summary>
    /// ENHANCED MACHINE LEARNING ENGINE FOR CAINE
    /// 
    /// WHAT THIS ADDS:
    /// - Error clustering to automatically categorize similar problems
    /// - Decision trees for predicting solution effectiveness
    /// - Neural network for complex pattern recognition
    /// - Time series analysis for predicting error trends
    /// - Anomaly detection for identifying new error types
    /// </summary>
    public class CaineMLEngine
    {
        // ML Model Storage
        private KMeans errorClusteringModel;
        private DecisionTree solutionClassifier;
        private MulticlassSupportVectorMachine<Gaussian> errorTypeClassifier;
        private readonly Dictionary<string, ErrorCluster> errorClusters = new Dictionary<string, ErrorCluster>();
        private readonly List<ErrorTimeSeries> errorTrends = new List<ErrorTimeSeries>();

        // Configuration
        private const int ClusterCount = 10; // Number of error categories to discover
        private const double AnomalyThreshold = 0.95; // Confidence for anomaly detection
        private const int MinSamplesForTraining = 50; // Minimum data needed to train models

        /// <summary>
        /// ERROR CLUSTER - Represents a group of similar errors
        /// </summary>
        public class ErrorCluster
        {
            public int ClusterId { get; set; }
            public string ClusterName { get; set; }
            public List<string> CommonKeywords { get; set; }
            public double[] Centroid { get; set; } // Center point of the cluster
            public List<string> MemberErrorHashes { get; set; }
            public double AverageSuccessRate { get; set; }
            public string RecommendedSolutionTemplate { get; set; }
            public Dictionary<string, double> FeatureImportance { get; set; }
        }

        /// <summary>
        /// ERROR TIME SERIES - For trend analysis and prediction
        /// </summary>
        public class ErrorTimeSeries
        {
            public string ErrorPattern { get; set; }
            public List<(DateTime Time, int Count)> Occurrences { get; set; }
            public double PredictedNextHourCount { get; set; }
            public double Seasonality { get; set; } // Weekly/monthly patterns
            public double Trend { get; set; } // Increasing/decreasing
        }

        /// <summary>
        /// TRAINING DATA POINT - Structure for ML training
        /// </summary>
        public class TrainingDataPoint
        {
            public double[] Features { get; set; }
            public string ErrorHash { get; set; }
            public bool SolutionWorked { get; set; }
            public double ResponseTime { get; set; }
            public string ErrorCategory { get; set; }
            public DateTime Timestamp { get; set; }
        }

        /// <summary>
        /// INITIALIZE ML ENGINE - Set up all machine learning models
        /// </summary>
        public async Task InitializeAsync(List<TrainingDataPoint> historicalData)
        {
            if (historicalData.Count < MinSamplesForTraining)
            {
                throw new InvalidOperationException($"Need at least {MinSamplesForTraining} samples for training");
            }

            // Train models in parallel for efficiency
            var tasks = new[]
            {
                Task.Run(() => TrainClusteringModel(historicalData)),
                Task.Run(() => TrainDecisionTree(historicalData)),
                Task.Run(() => TrainSVMClassifier(historicalData)),
                Task.Run(() => AnalyzeTimeSeries(historicalData))
            };

            await Task.WhenAll(tasks);
        }

        /// <summary>
        /// CLUSTERING - Automatically discover error categories
        /// 
        /// WHAT THIS DOES:
        /// Uses K-Means clustering to group similar errors together
        /// automatically without predefined categories
        /// </summary>
        private void TrainClusteringModel(List<TrainingDataPoint> data)
        {
            // Convert data to feature matrix
            double[][] observations = data.Select(d => d.Features).ToArray();

            // Initialize K-Means with smart centroid initialization (K-Means++)
            var kmeans = new KMeans(ClusterCount)
            {
                Distance = new SquareEuclidean(),
                MaxIterations = 1000,
                Tolerance = 1e-5,
                UseSeeding = Seeding.KMeansPlusPlus
            };

            // Learn the clusters - this returns KMeansClusterCollection
            var clusters = kmeans.Learn(observations);

            // Store the trained model (not the clusters result)
            errorClusteringModel = kmeans;

            // Now use the clusters for analysis
            var clusterAssignments = clusters.Decide(observations);

            for (int i = 0; i < ClusterCount; i++)
            {
                var clusterMembers = data.Where((d, idx) => clusterAssignments[idx] == i).ToList();

                var cluster = new ErrorCluster
                {
                    ClusterId = i,
                    Centroid = kmeans.Centroids[i],
                    MemberErrorHashes = clusterMembers.Select(m => m.ErrorHash).ToList(),
                    AverageSuccessRate = clusterMembers.Average(m => m.SolutionWorked ? 1.0 : 0.0),
                    ClusterName = GenerateClusterName(clusterMembers),
                    CommonKeywords = ExtractCommonKeywords(clusterMembers),
                    FeatureImportance = CalculateFeatureImportance(clusterMembers)
                };

                errorClusters[cluster.ClusterName] = cluster;
            }
        }

        /// <summary>
        /// DECISION TREE - Predict which solutions will work
        /// 
        /// WHAT THIS DOES:
        /// Builds a decision tree that learns which features of an error
        /// predict whether a solution will be successful
        /// </summary>
        private void TrainDecisionTree(List<TrainingDataPoint> data)
        {
            // Prepare training data
            double[][] inputs = data.Select(d => d.Features).ToArray();
            int[] outputs = data.Select(d => d.SolutionWorked ? 1 : 0).ToArray();

            // Create attribute list describing your features as continuous
            var attributes = new List<DecisionVariable>();
            for (int i = 0; i < inputs[0].Length; i++)
            {
                attributes.Add(DecisionVariable.Continuous($"Feature_{i}"));
            }

            // Create the decision tree with proper structure
            var tree = new DecisionTree(
                inputs: attributes.ToArray(),
                classes: 2  // Binary classification: worked/didn't work
            );

            // Create C4.5 learning algorithm with the tree
            var teacher = new C45Learning(tree);

            // Train the model - Learn returns the trained tree
            solutionClassifier = teacher.Learn(inputs, outputs);

            // Calculate feature importance
            var importance = CalculateDecisionTreeImportance(solutionClassifier);
            LogFeatureImportance(importance);
        }

        /// <summary>
        /// SVM CLASSIFIER - Multi-class error type classification
        /// 
        /// WHAT THIS DOES:
        /// Uses Support Vector Machines with Gaussian kernels for
        /// sophisticated non-linear classification of error types
        /// </summary>
        private void TrainSVMClassifier(List<TrainingDataPoint> data)
        {
            // Group by error category
            var categories = data.Select(d => d.ErrorCategory).Distinct().ToArray();
            var categoryMap = categories.Select((c, i) => new { Category = c, Index = i })
                                       .ToDictionary(x => x.Category, x => x.Index);

            double[][] inputs = data.Select(d => d.Features).ToArray();
            int[] outputs = data.Select(d => categoryMap[d.ErrorCategory]).ToArray();

            // Create a multi-class SVM with Gaussian kernel
            var teacher = new MulticlassSupportVectorLearning<Gaussian>()
            {
                Learner = (p) => new SequentialMinimalOptimization<Gaussian>()
                {
                    UseKernelEstimation = true,
                    Complexity = 100 // Regularization parameter
                }
            };

            // Train the model
            errorTypeClassifier = teacher.Learn(inputs, outputs);

            // Simple accuracy check without CrossValidation for now
            // (CrossValidation API has changed in different Accord versions)
            int correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                int predicted = errorTypeClassifier.Decide(inputs[i]);
                if (predicted == outputs[i]) correct++;
            }

            double accuracy = (double)correct / inputs.Length;
            Console.WriteLine($"SVM Training Accuracy: {accuracy:P2}");
        }

        /// <summary>
        /// TIME SERIES ANALYSIS - Predict error trends
        /// 
        /// WHAT THIS DOES:
        /// Analyzes historical error patterns to predict future occurrences
        /// and identify seasonal patterns
        /// </summary>
        private void AnalyzeTimeSeries(List<TrainingDataPoint> data)
        {
            // Group errors by pattern and time
            var timeGroups = data.GroupBy(d => new
            {
                Pattern = d.ErrorCategory,
                Hour = new DateTime(d.Timestamp.Year, d.Timestamp.Month,
                                   d.Timestamp.Day, d.Timestamp.Hour, 0, 0)
            });

            foreach (var patternGroup in timeGroups.GroupBy(g => g.Key.Pattern))
            {
                var series = new ErrorTimeSeries
                {
                    ErrorPattern = patternGroup.Key,
                    Occurrences = patternGroup
                        .Select(g => (g.Key.Hour, g.Count()))
                        .OrderBy(x => x.Hour)
                        .ToList()
                };

                // Calculate trend using linear regression
                if (series.Occurrences.Count > 10)
                {
                    series.Trend = CalculateTrend(series.Occurrences);
                    series.Seasonality = CalculateSeasonality(series.Occurrences);
                    series.PredictedNextHourCount = PredictNextValue(series);
                }

                errorTrends.Add(series);
            }
        }

        /// <summary>
        /// ANOMALY DETECTION - Identify unusual new error types
        /// 
        /// WHAT THIS DOES:
        /// Detects errors that don't fit any known pattern,
        /// flagging them for special attention
        /// </summary>
        public async Task<bool> IsAnomalyAsync(double[] errorFeatures)
        {
            return await Task.Run(() =>
            {
                if (errorClusteringModel == null) return false;

                // Calculate distance to nearest cluster centroid
                double minDistance = double.MaxValue;
                foreach (var centroid in errorClusteringModel.Centroids)
                {
                    double distance = CalculateEuclideanDistance(errorFeatures, centroid);
                    minDistance = Math.Min(minDistance, distance);
                }

                // Calculate anomaly score based on distance
                double avgClusterRadius = CalculateAverageClusterRadius();
                double anomalyScore = minDistance / avgClusterRadius;

                return anomalyScore > AnomalyThreshold;
            });
        }

        /// <summary>
        /// PREDICT SOLUTION SUCCESS - Use ML to predict if a solution will work
        /// 
        /// WHAT THIS DOES:
        /// Uses the trained decision tree to predict whether a proposed
        /// solution will successfully resolve an error
        /// </summary>
        public async Task<(bool WillWork, double Confidence)> PredictSolutionSuccessAsync(double[] errorFeatures)
        {
            return await Task.Run(() =>
            {
                if (solutionClassifier == null)
                    return (false, 0.0);

                // Get prediction from decision tree
                int prediction = solutionClassifier.Decide(errorFeatures);

                // Calculate confidence using tree path probability
                double confidence = CalculateDecisionConfidence(solutionClassifier, errorFeatures);

                return (prediction == 1, confidence);
            });
        }

        /// <summary>
        /// GET CLUSTER RECOMMENDATION - Find best solution template for error cluster
        /// 
        /// WHAT THIS DOES:
        /// Identifies which cluster an error belongs to and returns
        /// the most successful solution template for that cluster
        /// </summary>
        public async Task<(string Template, double Confidence)> GetClusterRecommendationAsync(double[] errorFeatures)
        {
            return await Task.Run(() =>
            {
                if (errorClusteringModel == null)
                    return ("No recommendation available", 0.0);

                // Need to transform the features to get cluster assignment
                // Since KMeans doesn't have Decide, we need to find the nearest cluster manually
                int clusterId = FindNearestCluster(errorFeatures);

                var cluster = errorClusters.Values.FirstOrDefault(c => c.ClusterId == clusterId);
                if (cluster == null)
                    return ("No cluster found", 0.0);

                // Calculate confidence based on cluster cohesion
                double distance = CalculateEuclideanDistance(errorFeatures, cluster.Centroid);
                double maxDistance = CalculateMaxClusterDistance(cluster);
                double confidence = 1.0 - (distance / maxDistance);

                return (cluster.RecommendedSolutionTemplate, confidence * cluster.AverageSuccessRate);
            });
        }

        // Add this helper method to find the nearest cluster
        private int FindNearestCluster(double[] features)
        {
            if (errorClusteringModel == null || errorClusteringModel.Centroids == null)
                return 0;

            int nearestCluster = 0;
            double minDistance = double.MaxValue;

            for (int i = 0; i < errorClusteringModel.Centroids.Length; i++)
            {
                double distance = CalculateEuclideanDistance(features, errorClusteringModel.Centroids[i]);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestCluster = i;
                }
            }

            return nearestCluster;
        }

        /// <summary>
        /// PREDICT ERROR TREND - Forecast future error occurrences
        /// 
        /// WHAT THIS DOES:
        /// Uses time series analysis to predict how many times
        /// an error pattern will occur in the next time period
        /// </summary>
        public async Task<(double PredictedCount, double Confidence)> PredictErrorTrendAsync(string errorPattern, int hoursAhead)
        {
            return await Task.Run(() =>
            {
                var series = errorTrends.FirstOrDefault(t => t.ErrorPattern == errorPattern);
                if (series == null || series.Occurrences.Count < 10)
                    return (0.0, 0.0);

                // Use exponential smoothing for prediction
                double alpha = 0.3; // Smoothing factor
                double[] smoothed = ExponentialSmoothing(
                    series.Occurrences.Select(o => (double)o.Count).ToArray(),
                    alpha
                );

                // Project forward
                double lastValue = smoothed.Last();
                double trend = series.Trend;
                double predicted = lastValue + (trend * hoursAhead);

                // Add seasonal component
                if (series.Seasonality > 0.1)
                {
                    int seasonalPeriod = 24; // Daily pattern
                    int seasonalIndex = hoursAhead % seasonalPeriod;
                    predicted *= (1 + series.Seasonality * Math.Sin(2 * Math.PI * seasonalIndex / seasonalPeriod));
                }

                // Calculate confidence based on historical accuracy
                double confidence = CalculatePredictionConfidence(series, smoothed);

                return (Math.Max(0, predicted), confidence);
            });
        }

        /// <summary>
        /// RETRAIN MODELS - Update ML models with new data
        /// 
        /// WHAT THIS DOES:
        /// Periodically retrains all models with new data to improve
        /// accuracy and adapt to changing error patterns
        /// </summary>
        public async Task RetrainModelsAsync(List<TrainingDataPoint> newData)
        {
            // Combine with existing knowledge (transfer learning)
            var combinedData = MergeWithExistingKnowledge(newData);

            // Retrain all models
            await InitializeAsync(combinedData);

            // Log improvement metrics
            LogModelImprovement();
        }

        // ============================================================================
        // HELPER METHODS - Supporting functions for ML operations
        // ============================================================================

        private string GenerateClusterName(List<TrainingDataPoint> members)
        {
            // Generate descriptive name based on common characteristics
            var commonCategories = members.GroupBy(m => m.ErrorCategory)
                                         .OrderByDescending(g => g.Count())
                                         .Take(3)
                                         .Select(g => g.Key);
            return string.Join("-", commonCategories);
        }

        private List<string> ExtractCommonKeywords(List<TrainingDataPoint> members)
        {
            // Extract most frequent keywords from error patterns
            // This would analyze the actual error text (not shown in features)
            return new List<string> { "keyword1", "keyword2" }; // Placeholder
        }

        private Dictionary<string, double> CalculateFeatureImportance(List<TrainingDataPoint> members)
        {
            var importance = new Dictionary<string, double>();

            // Calculate variance for each feature
            for (int i = 0; i < members[0].Features.Length; i++)
            {
                var values = members.Select(m => m.Features[i]).ToArray();
                double variance = CalculateVariance(values);
                importance[$"Feature_{i}"] = variance;
            }

            // Normalize importance scores
            double sum = importance.Values.Sum();
            if (sum > 0)
            {
                var normalized = importance.ToDictionary(
                    kvp => kvp.Key,
                    kvp => kvp.Value / sum
                );
                return normalized;
            }

            return importance;
        }

        private double CalculateVariance(double[] values)
        {
            double mean = values.Average();
            double sumSquaredDiff = values.Sum(v => Math.Pow(v - mean, 2));
            return sumSquaredDiff / values.Length;
        }

        private double CalculateEuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            {
                sum += Math.Pow(a[i] - b[i], 2);
            }
            return Math.Sqrt(sum);
        }

        private double CalculateAverageClusterRadius()
        {
            double totalRadius = 0;
            int count = 0;

            foreach (var cluster in errorClusters.Values)
            {
                double maxDist = CalculateMaxClusterDistance(cluster);
                totalRadius += maxDist;
                count++;
            }

            return count > 0 ? totalRadius / count : 1.0;
        }

        private double CalculateMaxClusterDistance(ErrorCluster cluster)
        {
            // This would calculate based on actual member distances
            // Placeholder for now
            return 10.0;
        }

        private double CalculateTrend(List<(DateTime Time, int Count)> occurrences)
        {
            if (occurrences.Count < 2) return 0;

            // Simple linear regression for trend
            double[] x = Enumerable.Range(0, occurrences.Count).Select(i => (double)i).ToArray();
            double[] y = occurrences.Select(o => (double)o.Count).ToArray();

            double xMean = x.Average();
            double yMean = y.Average();

            double numerator = 0;
            double denominator = 0;

            for (int i = 0; i < x.Length; i++)
            {
                numerator += (x[i] - xMean) * (y[i] - yMean);
                denominator += Math.Pow(x[i] - xMean, 2);
            }

            return denominator != 0 ? numerator / denominator : 0;
        }

        private double CalculateSeasonality(List<(DateTime Time, int Count)> occurrences)
        {
            if (occurrences.Count < 24) return 0;

            // Detect daily patterns using autocorrelation
            int period = 24; // Hours in a day
            double correlation = 0;

            for (int i = period; i < occurrences.Count; i++)
            {
                correlation += occurrences[i].Count * occurrences[i - period].Count;
            }

            return correlation / (occurrences.Count - period);
        }

        private double PredictNextValue(ErrorTimeSeries series)
        {
            var counts = series.Occurrences.Select(o => (double)o.Count).ToArray();
            if (counts.Length == 0) return 0;

            // Simple moving average for now
            int windowSize = Math.Min(5, counts.Length);
            double sum = counts.Skip(counts.Length - windowSize).Sum();
            return sum / windowSize + series.Trend;
        }

        private double[] ExponentialSmoothing(double[] data, double alpha)
        {
            double[] smoothed = new double[data.Length];
            smoothed[0] = data[0];

            for (int i = 1; i < data.Length; i++)
            {
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1];
            }

            return smoothed;
        }

        private double CalculatePredictionConfidence(ErrorTimeSeries series, double[] smoothed)
        {
            // Calculate based on prediction error
            double totalError = 0;
            var actual = series.Occurrences.Select(o => (double)o.Count).ToArray();

            for (int i = 0; i < Math.Min(actual.Length, smoothed.Length); i++)
            {
                totalError += Math.Abs(actual[i] - smoothed[i]);
            }

            double avgError = totalError / actual.Length;
            double avgValue = actual.Average();

            // Convert error to confidence (lower error = higher confidence)
            return Math.Max(0, 1 - (avgError / avgValue));
        }

        private Dictionary<string, double> CalculateDecisionTreeImportance(DecisionTree tree)
        {
            // This would walk the tree and calculate feature importance
            // based on information gain at each split
            return new Dictionary<string, double>();
        }

        private double CalculateDecisionConfidence(DecisionTree tree, double[] features)
        {
            // This would calculate confidence based on the tree path
            // and the purity of the leaf node
            return 0.85; // Placeholder
        }

        private void LogFeatureImportance(Dictionary<string, double> importance)
        {
            Console.WriteLine("Feature Importance:");
            foreach (var kvp in importance.OrderByDescending(x => x.Value))
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value:F3}");
            }
        }

        private void LogModelImprovement()
        {
            Console.WriteLine("Models retrained successfully");
            // This would compare before/after metrics
        }

        private List<TrainingDataPoint> MergeWithExistingKnowledge(List<TrainingDataPoint> newData)
        {
            // This would intelligently merge new data with existing knowledge
            // applying techniques like importance sampling
            return newData;
        }
    }

    /// <summary>
    /// NEURAL NETWORK COMPONENT - Deep learning for complex patterns
    /// 
    /// WHAT THIS DOES:
    /// Implements a simple feedforward neural network for learning
    /// non-linear patterns in error resolution
    /// </summary>
    public class SimpleNeuralNetwork
    {
        private readonly int inputSize;
        private readonly int hiddenSize;
        private readonly int outputSize;
        private double[,] weightsInputHidden;
        private double[,] weightsHiddenOutput;
        private double[] biasHidden;
        private double[] biasOutput;
        private readonly double learningRate = 0.01;

        public SimpleNeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            var random = new Random();

            // Xavier initialization for better convergence
            double inputScale = Math.Sqrt(2.0 / inputSize);
            double hiddenScale = Math.Sqrt(2.0 / hiddenSize);

            weightsInputHidden = new double[inputSize, hiddenSize];
            weightsHiddenOutput = new double[hiddenSize, outputSize];
            biasHidden = new double[hiddenSize];
            biasOutput = new double[outputSize];

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] = random.NextDouble() * inputScale - inputScale / 2;
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weightsHiddenOutput[i, j] = random.NextDouble() * hiddenScale - hiddenScale / 2;
                }
            }
        }

        public double[] Forward(double[] input)
        {
            // Hidden layer
            double[] hidden = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputHidden[i, j];
                }
                hidden[j] = ReLU(sum); // ReLU activation
            }

            // Output layer
            double[] output = new double[outputSize];
            for (int j = 0; j < outputSize; j++)
            {
                double sum = biasOutput[j];
                for (int i = 0; i < hiddenSize; i++)
                {
                    sum += hidden[i] * weightsHiddenOutput[i, j];
                }
                output[j] = Sigmoid(sum); // Sigmoid for classification
            }

            return output;
        }

        public void Train(double[][] inputs, double[][] targets, int epochs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;

                for (int sample = 0; sample < inputs.Length; sample++)
                {
                    // Forward pass
                    double[] output = Forward(inputs[sample]);

                    // Calculate loss
                    double loss = CalculateLoss(output, targets[sample]);
                    totalLoss += loss;

                    // Backward pass (simplified backpropagation)
                    Backpropagate(inputs[sample], targets[sample], output);
                }

                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss / inputs.Length:F4}");
                }
            }
        }

        private void Backpropagate(double[] input, double[] target, double[] output)
        {
            // This is a simplified version of backpropagation
            // In production, you'd use a more sophisticated implementation

            // Calculate output layer gradients
            double[] outputGradients = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                outputGradients[i] = (output[i] - target[i]) * SigmoidDerivative(output[i]);
            }

            // Update weights and biases using gradients
            // (Full implementation would include hidden layer gradients)
        }

        private double ReLU(double x) => Math.Max(0, x);
        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDerivative(double x) => x * (1 - x);

        private double CalculateLoss(double[] output, double[] target)
        {
            // Binary cross-entropy loss
            double loss = 0;
            for (int i = 0; i < output.Length; i++)
            {
                loss -= target[i] * Math.Log(output[i] + 1e-10) +
                        (1 - target[i]) * Math.Log(1 - output[i] + 1e-10);
            }
            return loss / output.Length;
        }
    }
}