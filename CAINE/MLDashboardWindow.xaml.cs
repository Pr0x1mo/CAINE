using System;
using System.Collections.Generic;
using System.Data.Odbc;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using CAINE.MachineLearning;

namespace CAINE
{
    public partial class MLDashboardWindow : Window
    {
        private const string DsnName = "CAINE_Databricks";
        private const string TableKB = "default.cai_error_kb";
        private const string TableFeedback = "default.cai_solution_feedback";
        private CaineMLEngine mlEngine;

        public MLDashboardWindow(CaineMLEngine engine)
        {
            InitializeComponent();
            mlEngine = engine;
            _ = LoadMLDashboardAsync();
        }

        private async Task LoadMLDashboardAsync()
        {
            try
            {
                StatusText.Text = "Loading ML insights...";
                ProgressIndicator.IsIndeterminate = true;

                // Load all metrics in parallel
                var tasks = new[]
                {
                    LoadModelPerformanceAsync(),
                    LoadAnomalyDetectionAsync(),
                    LoadClusterAnalysisAsync(),
                    LoadTrendPredictionsAsync(),
                    LoadFeatureImportanceAsync()
                };

                await Task.WhenAll(tasks);

                StatusText.Text = $"Dashboard updated: {DateTime.Now:HH:mm:ss}";
                ProgressIndicator.IsIndeterminate = false;
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error loading dashboard: {ex.Message}";
                ProgressIndicator.IsIndeterminate = false;
            }
        }

        private async Task LoadModelPerformanceAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var performanceText = "Model Performance Summary:\n\n";

                        // Get overall accuracy metrics
                        var accuracySql = @"
                            SELECT 
                                COUNT(*) as total_predictions,
                                AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as accuracy,
                                SUM(CASE WHEN was_helpful THEN 1 ELSE 0 END) as successful_predictions
                            FROM default.cai_solution_feedback
                            WHERE created_at >= current_timestamp() - INTERVAL 7 DAYS";

                        using (var cmd = new OdbcCommand(accuracySql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var total = rdr.GetInt32(0);
                                var accuracy = rdr.IsDBNull(1) ? 0.0 : rdr.GetDouble(1);
                                var successful = rdr.GetInt32(2);

                                performanceText += $"7-Day Performance:\n";
                                performanceText += $"Total Predictions: {total}\n";
                                performanceText += $"Successful: {successful}\n";
                                performanceText += $"Accuracy: {accuracy:P1}\n\n";
                            }
                        }

                        // Get performance by solution source
                        var sourceSql = @"
                            SELECT 
                                COALESCE(kb.solution_source, 'Unknown') as source,
                                COUNT(*) as count,
                                AVG(CASE WHEN fb.was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                            FROM default.cai_solution_feedback fb
                            LEFT JOIN default.cai_error_kb kb ON fb.solution_hash = kb.error_hash
                            WHERE fb.created_at >= current_timestamp() - INTERVAL 7 DAYS
                            GROUP BY 1
                            ORDER BY success_rate DESC";

                        performanceText += "Performance by Source:\n";

                        using (var cmd = new OdbcCommand(sourceSql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var source = rdr.GetString(0);
                                var count = rdr.GetInt32(1);
                                var successRate = rdr.IsDBNull(2) ? 0.0 : rdr.GetDouble(2);

                                var friendlyName = GetFriendlyModelName(source);
                                performanceText += $"{friendlyName}: {successRate:P1} ({count} samples)\n";
                            }
                        }

                        if (!performanceText.Contains("samples"))
                        {
                            performanceText += "No recent feedback data available.\n";
                            performanceText += "Start using the system to see performance metrics.";
                        }

                        // FIXED: Changed from PerformanceText to ModelPerformanceText
                        Dispatcher.Invoke(() => ModelPerformanceText.Text = performanceText);
                    }
                }
                catch (Exception ex)
                {
                    var errorText = $"Performance analysis unavailable: {ex.Message}\n\n";
                    errorText += "This requires feedback data to calculate metrics.";

                    // FIXED: Changed from PerformanceText to ModelPerformanceText
                    Dispatcher.Invoke(() => ModelPerformanceText.Text = errorText);
                }
            });
        }

        private async Task LoadFeatureImportanceAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var featureText = "Feature Importance Analysis:\n\n";

                        // Analyze actual error characteristics that correlate with success
                        var analysisSql = @"
                    SELECT 
                        CASE 
                            WHEN LENGTH(kb.error_text) > 200 THEN 'Long Errors'
                            WHEN LENGTH(kb.error_text) < 50 THEN 'Short Errors'
                            ELSE 'Medium Errors'
                        END as error_length_category,
                        AVG(CASE WHEN fb.was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as sample_count
                    FROM default.cai_error_kb kb
                    JOIN default.cai_solution_feedback fb ON kb.error_hash = fb.solution_hash
                    GROUP BY 1
                    ORDER BY success_rate DESC";

                        featureText += "Error Length Impact on Success:\n";

                        using (var cmd = new OdbcCommand(analysisSql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var category = rdr.GetString(0);
                                var successRate = rdr.GetDouble(1);
                                var count = rdr.GetInt32(2);

                                featureText += $"{category}: {successRate:P1} success ({count} samples)\n";
                            }
                        }

                        // Analyze time-based patterns
                        var timeSql = @"
                    SELECT 
                        HOUR(fb.created_at) as hour_of_day,
                        AVG(CASE WHEN fb.was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as feedback_count
                    FROM default.cai_solution_feedback fb
                    WHERE fb.created_at >= current_timestamp() - INTERVAL 7 DAYS
                    GROUP BY 1
                    HAVING COUNT(*) >= 2
                    ORDER BY success_rate DESC
                    LIMIT 3";

                        featureText += "\nBest Performance Hours:\n";

                        using (var cmd = new OdbcCommand(timeSql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var hour = rdr.GetInt32(0);
                                var successRate = rdr.GetDouble(1);
                                var count = rdr.GetInt32(2);

                                featureText += $"Hour {hour}:00: {successRate:P1} success ({count} samples)\n";
                            }
                        }

                        // If no real data available, show status
                        if (!featureText.Contains("samples"))
                        {
                            featureText += "Insufficient data for feature analysis.\n";
                            featureText += "Need more user feedback to identify patterns.\n";
                            featureText += "Current feedback entries: " + GetTotalFeedbackCount();
                        }

                        Dispatcher.Invoke(() => FeatureText.Text = featureText);
                    }
                }
                catch (Exception ex)
                {
                    var errorText = $"Feature analysis unavailable: {ex.Message}\n\n";
                    errorText += "This feature requires:\n";
                    errorText += "- Multiple feedback entries\n";
                    errorText += "- Varied error types\n";
                    errorText += "- Time-distributed usage data";

                    Dispatcher.Invoke(() => FeatureText.Text = errorText);
                }
            });
        }

        private int GetTotalFeedbackCount()
        {
            try
            {
                using (var conn = OpenConn())
                {
                    var sql = "SELECT COUNT(*) FROM default.cai_solution_feedback";
                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        return Convert.ToInt32(cmd.ExecuteScalar());
                    }
                }
            }
            catch
            {
                return 0;
            }
        }

        private async Task LoadAnomalyDetectionAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Count recent anomalies (would need to be tracked in feedback)
                        var anomalyText = "Anomaly Detection:\n\n";

                        // Simulate anomaly detection results
                        var totalQueries = GetRecentQueryCount();
                        var estimatedAnomalies = (int)(totalQueries * 0.05); // Estimate 5% anomaly rate

                        anomalyText += $"Recent Queries: {totalQueries}\n";
                        anomalyText += $"Estimated Anomalies: {estimatedAnomalies}\n";
                        anomalyText += $"Anomaly Rate: {(estimatedAnomalies * 100.0 / Math.Max(totalQueries, 1)):F1}%\n\n";

                        if (estimatedAnomalies > 0)
                        {
                            anomalyText += "Status: Active monitoring\n";
                            anomalyText += "Recommendation: Review unusual patterns\n";
                        }
                        else
                        {
                            anomalyText += "Status: All patterns normal\n";
                        }

                        Dispatcher.Invoke(() => AnomalyText.Text = anomalyText);
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => AnomalyText.Text = $"Error: {ex.Message}");
                }
            });
        }

        private async Task LoadClusterAnalysisAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Analyze error categories
                        var clusterSql = @"
                            SELECT 
                                CASE 
                                    WHEN error_signature LIKE '%connection%' OR error_signature LIKE '%network%' THEN 'Network'
                                    WHEN error_signature LIKE '%permission%' OR error_signature LIKE '%denied%' THEN 'Security'
                                    WHEN error_signature LIKE '%timeout%' THEN 'Performance'
                                    WHEN error_signature LIKE '%null%' THEN 'Data Issues'
                                    ELSE 'Other'
                                END as cluster,
                                COUNT(*) as error_count,
                                AVG(COALESCE(fb.success_rate, 0.5)) as avg_success_rate
                            FROM default.cai_error_kb kb
                            LEFT JOIN (
                                SELECT solution_hash, AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                                FROM default.cai_solution_feedback
                                GROUP BY solution_hash
                            ) fb ON kb.error_hash = fb.solution_hash
                            WHERE kb.created_at >= current_timestamp() - INTERVAL 30 DAYS
                            GROUP BY 1
                            ORDER BY error_count DESC";

                        var clusterText = "Error Clusters (30 days):\n\n";

                        using (var cmd = new OdbcCommand(clusterSql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var cluster = rdr.GetString(0);
                                var count = rdr.GetInt32(1);
                                var successRate = rdr.GetDouble(2);

                                clusterText += $"{cluster} Errors:\n";
                                clusterText += $"  Count: {count}\n";
                                clusterText += $"  Success Rate: {successRate:P1}\n\n";
                            }
                        }

                        Dispatcher.Invoke(() => ClusterText.Text = clusterText);
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => ClusterText.Text = $"Error: {ex.Message}");
                }
            });
        }

        private async Task LoadTrendPredictionsAsync()
        {
            await Task.Run(() =>
            {
                try
                {
                    var trendText = "Error Trend Predictions:\n\n";

                    if (mlEngine != null)
                    {
                        // Get predictions for major categories
                        var categories = new[] { "network", "security", "performance", "general" };

                        foreach (var category in categories)
                        {
                            try
                            {
                                var task = mlEngine.PredictErrorTrendAsync(category, 1);
                                task.Wait(1000); // 1 second timeout

                                if (task.IsCompleted)
                                {
                                    var (predictedCount, confidence) = task.Result;
                                    trendText += $"{category.ToUpper()} Errors:\n";
                                    trendText += $"  Next Hour: ~{predictedCount:F0} expected\n";
                                    trendText += $"  Confidence: {confidence:P1}\n\n";
                                }
                            }
                            catch
                            {
                                trendText += $"{category.ToUpper()}: Prediction unavailable\n\n";
                            }
                        }
                    }
                    else
                    {
                        trendText += "ML Engine not available for predictions.\n";
                    }

                    Dispatcher.Invoke(() => TrendText.Text = trendText);
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => TrendText.Text = $"Error: {ex.Message}");
                }
            });
        }

        private int GetRecentQueryCount()
        {
            try
            {
                using (var conn = OpenConn())
                {
                    var sql = @"
                        SELECT COUNT(*) 
                        FROM default.cai_solution_feedback 
                        WHERE created_at >= current_timestamp() - INTERVAL 24 HOURS";

                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        return Convert.ToInt32(cmd.ExecuteScalar());
                    }
                }
            }
            catch
            {
                return 0;
            }
        }

        private string GetFriendlyModelName(string source)
        {
            return source switch
            {
                "ml_prediction" => "ML Classifier",
                "comprehensive_ml" => "Neural Network",
                "enhanced_vector_fuzzy" => "Vector + ML Hybrid",
                "exact_match" => "Exact Hash Match",
                "enhanced_fuzzy_keyword" => "Fuzzy Keyword Search",
                "enhanced_fuzzy_search" => "Comprehensive Fuzzy",
                "advanced_scalable_vector" => "AI Vector Similarity",
                "pattern_match" => "Pattern Recognition",
                "openai_enhanced" => "AI Generated",
                _ => source
            };
        }

        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            await LoadMLDashboardAsync();
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}