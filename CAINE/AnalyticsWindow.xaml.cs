using System;
using System.Data.Odbc;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Collections.Generic;
using System.Linq;

namespace CAINE
{
    public partial class AnalyticsWindow : Window
    {
        private const string DsnName = "CAINE_Databricks";
        private const string TableKB = "default.cai_error_kb";
        private const string TableFeedback = "default.cai_solution_feedback";
        private const string TableSecurityLog = "default.cai_security_log";

        public AnalyticsWindow()
        {
            InitializeComponent();
            _ = LoadAnalyticsAsync(); // Start loading data immediately
        }

        /// <summary>
        /// Main analytics loading function
        /// </summary>
        private async Task LoadAnalyticsAsync()
        {
            try
            {
                UpdateLoadingState("Loading analytics...");

                // Load all metrics in parallel for better performance
                var tasks = new[]
                {
            LoadKnowledgeBaseMetrics(),
            LoadFeedbackMetrics(),
            LoadUserActivityMetrics(),
            LoadSecurityMetrics(),
            LoadRecentActivity(),
            LoadSystemPerformance(),
            LoadMLInsightsAsync() // ADD THIS LINE
        };

                await Task.WhenAll(tasks);

                UpdateSystemHealth();
                UpdateLastUpdatedTime();
            }
            catch (Exception ex)
            {
                UpdateErrorState($"Failed to load analytics: {ex.Message}");
            }
        }
        private async Task LoadMLInsightsAsync()
        {
            try
            {
                using (var conn = OpenConn())
                {
                    // ML Model Performance
                    var mlPerformanceText = "ML Model Performance:\n\n";

                    // [Your existing SQL queries code here...]

                    // Get confidence accuracy 
                    var confidenceAccuracySql = $@"
                SELECT 
                    CASE 
                        WHEN confidence_rating >= 4 THEN 'High Confidence'
                        WHEN confidence_rating >= 3 THEN 'Medium Confidence' 
                        ELSE 'Low Confidence'
                    END as confidence_level,
                    AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as accuracy,
                    COUNT(*) as sample_size
                FROM {TableFeedback}
                WHERE confidence_rating IS NOT NULL
                GROUP BY 1
                ORDER BY confidence_level";

                    using (var cmd = new OdbcCommand(confidenceAccuracySql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        while (rdr.Read())
                        {
                            var level = rdr.GetString(0);
                            var accuracy = rdr.GetDouble(1);
                            var samples = rdr.GetInt32(2);
                            mlPerformanceText += $"{level}: {accuracy:P0} accuracy ({samples} samples)\n";
                        }
                    }

                    // Search method effectiveness
                    mlPerformanceText += "\nSearch Method Effectiveness (Last 7 Days):\n";
                    var methodEffectivenessSql = $@"
                SELECT 
                    solution_source,
                    AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                    COUNT(*) as usage_count
                FROM {TableFeedback}
                WHERE created_at >= current_timestamp() - INTERVAL 7 DAYS
                GROUP BY solution_source
                ORDER BY success_rate DESC";

                    using (var cmd = new OdbcCommand(methodEffectivenessSql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        while (rdr.Read())
                        {
                            var source = rdr.GetString(0);
                            var successRate = rdr.GetDouble(1);
                            var count = rdr.GetInt32(2);
                            var methodName = GetFriendlyMethodName(source);
                            mlPerformanceText += $"{methodName}: {successRate:P0} success ({count} uses)\n";
                        }
                    }

                    // Update UI - use existing TextBlock or append to existing content
                    Dispatcher.Invoke(() =>
                    {
                        // CHOOSE ONE OF THESE OPTIONS:

                        // Option A: Replace existing search strategy text
                        SearchStrategyText.Text = mlPerformanceText;

                        // Option B: Append to existing search strategy text
                        // SearchStrategyText.Text += "\n\n=== ML INSIGHTS ===\n" + mlPerformanceText;

                        // Option C: Use quality distribution text instead
                        // QualityDistributionText.Text = mlPerformanceText;
                    });
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML insights loading failed: {ex.Message}");
            }
        }

        private string GetFriendlyMethodName(string source)
        {
            return source switch
            {
                "exact_match" => "Exact Hash Match",
                "enhanced_fuzzy_keyword" => "Fuzzy Keyword Search",
                "enhanced_fuzzy_search" => "Comprehensive Fuzzy",
                "advanced_scalable_vector" => "AI Vector Similarity",
                "pattern_match" => "Pattern Recognition",
                "comprehensive_ml" => "ML Prediction",
                "openai_enhanced" => "AI Generated",
                _ => source
            };
        }

        /// <summary>
        /// Knowledge base statistics
        /// </summary>
        private async Task LoadKnowledgeBaseMetrics()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Total solutions
                        var totalSolutions = GetScalarValue(conn, $"SELECT COUNT(*) FROM {TableKB}");

                        // Top error category
                        var topCategorySql = $@"
                            SELECT 
                                CASE 
                                    WHEN error_signature LIKE '%login%' THEN 'Authentication'
                                    WHEN error_signature LIKE '%timeout%' THEN 'Connection'
                                    WHEN error_signature LIKE '%permission%' THEN 'Security'
                                    WHEN error_signature LIKE '%syntax%' THEN 'SQL Syntax'
                                    WHEN error_signature LIKE '%deadlock%' THEN 'Concurrency'
                                    WHEN error_signature LIKE '%corrupt%' THEN 'Data Integrity'
                                    ELSE 'Other'
                                END as category,
                                COUNT(*) as count
                            FROM {TableKB}
                            GROUP BY 1
                            ORDER BY count DESC
                            LIMIT 1";

                        var topCategory = "Unknown";
                        var categoryCount = 0;

                        using (var cmd = new OdbcCommand(topCategorySql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                topCategory = rdr.GetString(0);
                                categoryCount = rdr.GetInt32(1);
                            }
                        }

                        // Update UI on main thread
                        Dispatcher.Invoke(() =>
                        {
                            TotalSolutionsText.Text = totalSolutions.ToString("N0");
                            TopErrorCategoryText.Text = $"{topCategory}\n({categoryCount} solutions)";
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => TotalSolutionsText.Text = "Error");
                    System.Diagnostics.Debug.WriteLine($"KB metrics error: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// User feedback and success rate metrics
        /// </summary>
        private async Task LoadFeedbackMetrics()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Total feedback
                        var totalFeedback = GetScalarValue(conn, $@"
                    SELECT COUNT(*) FROM {TableFeedback}
                    WHERE created_at >= current_timestamp() - INTERVAL 30 DAYS");

                        // Average success rate - separate query
                        var successRate = 0.0;
                        var successRateSql = $@"
                    SELECT AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                    FROM {TableFeedback}
                    WHERE created_at >= current_timestamp() - INTERVAL 30 DAYS";

                        using (var cmd = new OdbcCommand(successRateSql, conn))
                        {
                            var result = cmd.ExecuteScalar();
                            successRate = result != null && result != DBNull.Value ? Convert.ToDouble(result) : 0.0;
                        }

                        // Conflicts - separate simple query
                        var conflictCount = GetScalarValue(conn, $@"
                    SELECT COUNT(DISTINCT solution_hash) 
                    FROM {TableFeedback}
                    WHERE was_helpful = false 
                    AND created_at >= current_timestamp() - INTERVAL 30 DAYS");

                        // Update UI
                        Dispatcher.Invoke(() =>
                        {
                            TotalFeedbackText.Text = totalFeedback.ToString("N0");
                            SuccessRateText.Text = (successRate * 100).ToString("F0") + "%";
                            ConflictsText.Text = conflictCount.ToString("N0");

                            // Color-code success rate
                            if (successRate >= 0.8)
                                SuccessRateText.Foreground = new SolidColorBrush(Color.FromRgb(16, 124, 16));
                            else if (successRate >= 0.6)
                                SuccessRateText.Foreground = new SolidColorBrush(Color.FromRgb(255, 140, 0));
                            else
                                SuccessRateText.Foreground = new SolidColorBrush(Color.FromRgb(209, 52, 56));
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        TotalFeedbackText.Text = "Error";
                        SuccessRateText.Text = "Error";
                    });
                    System.Diagnostics.Debug.WriteLine($"Feedback metrics error: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// User activity metrics
        /// </summary>
        private async Task LoadUserActivityMetrics()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Active users (unique sessions in last 24 hours)
                        var activeUsers = GetScalarValue(conn, $@"
                            SELECT COUNT(DISTINCT session_id) 
                            FROM {TableFeedback}
                            WHERE created_at >= current_timestamp() - INTERVAL 24 HOURS");

                        Dispatcher.Invoke(() =>
                        {
                            ActiveUsersText.Text = activeUsers.ToString("N0");
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => ActiveUsersText.Text = "Error");
                    System.Diagnostics.Debug.WriteLine($"User activity error: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Security events and threat monitoring
        /// </summary>
        private async Task LoadSecurityMetrics()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Total security events in last 24 hours
                        var securityEvents = GetScalarValue(conn, $@"
                            SELECT COUNT(*) FROM {TableSecurityLog}
                            WHERE created_at >= current_timestamp() - INTERVAL 24 HOURS");

                        Dispatcher.Invoke(() =>
                        {
                            SecurityEventsText.Text = securityEvents.ToString("N0");

                            // Color-code based on threat level
                            if (securityEvents == 0)
                                SecurityEventsText.Foreground = new SolidColorBrush(Color.FromRgb(16, 124, 16)); // Green
                            else if (securityEvents <= 5)
                                SecurityEventsText.Foreground = new SolidColorBrush(Color.FromRgb(255, 140, 0)); // Orange
                            else
                                SecurityEventsText.Foreground = new SolidColorBrush(Color.FromRgb(209, 52, 56)); // Red
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() => SecurityEventsText.Text = "Error");
                    System.Diagnostics.Debug.WriteLine($"Security metrics error: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Recent system activity log
        /// </summary>
        private async Task LoadRecentActivity()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var activityText = "";

                        // Recent feedback activity
                        var feedbackSql = $@"
                            SELECT 
                                created_at,
                                solution_source,
                                was_helpful,
                                error_signature
                            FROM {TableFeedback}
                            WHERE created_at >= current_timestamp() - INTERVAL 24 HOURS
                            ORDER BY created_at DESC
                            LIMIT 10";

                        using (var cmd = new OdbcCommand(feedbackSql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            activityText += "Recent User Feedback:\n";
                            while (rdr.Read())
                            {
                                var timestamp = rdr.IsDBNull(0) ? "Unknown" : rdr.GetDateTime(0).ToString("HH:mm");
                                var source = rdr.IsDBNull(1) ? "unknown" : rdr.GetString(1);
                                var helpful = rdr.IsDBNull(2) ? false : rdr.GetBoolean(2);
                                var error = rdr.IsDBNull(3) ? "Unknown error" : rdr.GetString(3);

                                var status = helpful ? "👍" : "👎";
                                var shortError = error.Length > 50 ? error.Substring(0, 50) + "..." : error;

                                activityText += $"{timestamp} - {status} {source} - {shortError}\n";
                            }
                        }

                        // Recent security events if any
                        var securitySql = $@"
                            SELECT 
                                created_at,
                                event_type,
                                severity
                            FROM {TableSecurityLog}
                            WHERE created_at >= current_timestamp() - INTERVAL 24 HOURS
                            ORDER BY created_at DESC
                            LIMIT 5";

                        using (var cmd = new OdbcCommand(securitySql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            var hasSecurityEvents = false;
                            while (rdr.Read())
                            {
                                if (!hasSecurityEvents)
                                {
                                    activityText += "\nRecent Security Events:\n";
                                    hasSecurityEvents = true;
                                }

                                var timestamp = rdr.IsDBNull(0) ? "Unknown" : rdr.GetDateTime(0).ToString("HH:mm");
                                var eventType = rdr.IsDBNull(1) ? "unknown" : rdr.GetString(1);
                                var severity = rdr.IsDBNull(2) ? "unknown" : rdr.GetString(2);

                                var icon = severity == "HIGH" ? "🚨" : severity == "MEDIUM" ? "⚠️" : "ℹ️";
                                activityText += $"{timestamp} - {icon} {eventType}\n";
                            }

                            if (!hasSecurityEvents)
                            {
                                activityText += "\nSecurity: No events in last 24 hours ✅\n";
                            }
                        }

                        if (string.IsNullOrEmpty(activityText.Trim()))
                        {
                            activityText = "No recent activity found.\n";
                        }

                        Dispatcher.Invoke(() =>
                        {
                            RecentActivityText.Text = activityText;
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        RecentActivityText.Text = $"Error loading activity: {ex.Message}";
                    });
                }
            });
        }

        /// <summary>
        /// System performance and search strategy effectiveness
        /// </summary>
        private async Task LoadSystemPerformance()
        {
            await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Search strategy effectiveness - basic approach
                        var strategyText = "Search Strategy Performance (7 days):\n\n";

                        try
                        {
                            var exactMatchCount = GetScalarValue(conn, $@"
                        SELECT COUNT(*) FROM {TableFeedback} 
                        WHERE solution_source = 'exact_match' 
                        AND created_at >= current_timestamp() - INTERVAL 7 DAYS");

                            var exactMatchSuccess = 0.0;
                            if (exactMatchCount > 0)
                            {
                                var successSql = $@"
                            SELECT AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END)
                            FROM {TableFeedback} 
                            WHERE solution_source = 'exact_match' 
                            AND created_at >= current_timestamp() - INTERVAL 7 DAYS";

                                using (var cmd = new OdbcCommand(successSql, conn))
                                {
                                    var result = cmd.ExecuteScalar();
                                    exactMatchSuccess = result != null && result != DBNull.Value ? Convert.ToDouble(result) : 0.0;
                                }
                            }

                            strategyText += $"exact_match:\n";
                            strategyText += $"  Uses: {exactMatchCount}\n";
                            strategyText += $"  Success: {(exactMatchSuccess * 100):F0}%\n\n";

                            // Add other sources if they exist
                            var totalFeedback = GetScalarValue(conn, $@"
                        SELECT COUNT(*) FROM {TableFeedback} 
                        WHERE created_at >= current_timestamp() - INTERVAL 7 DAYS");

                            if (totalFeedback == 0)
                            {
                                strategyText = "No feedback data available for the last 7 days.";
                            }
                        }
                        catch (Exception ex)
                        {
                            strategyText = $"Error loading strategy data: {ex.Message}";
                        }

                        // Solution quality - very simple approach
                        var qualityText = "Solution Quality (with feedback):\n\n";

                        try
                        {
                            var totalSolutions = GetScalarValue(conn, $@"
                        SELECT COUNT(DISTINCT solution_hash) FROM {TableFeedback}");

                            var totalFeedbackEntries = GetScalarValue(conn, $@"
                        SELECT COUNT(*) FROM {TableFeedback}");

                            var helpfulFeedback = GetScalarValue(conn, $@"
                        SELECT COUNT(*) FROM {TableFeedback} WHERE was_helpful = true");

                            var overallSuccessRate = totalFeedbackEntries > 0 ? (double)helpfulFeedback / totalFeedbackEntries : 0.0;

                            qualityText += $"Total unique solutions with feedback: {totalSolutions}\n";
                            qualityText += $"Total feedback entries: {totalFeedbackEntries}\n";
                            qualityText += $"Overall success rate: {(overallSuccessRate * 100):F0}%\n\n";

                            if (overallSuccessRate >= 0.8)
                                qualityText += "System Quality: High (80%+ success rate)";
                            else if (overallSuccessRate >= 0.6)
                                qualityText += "System Quality: Medium (60-79% success rate)";
                            else if (overallSuccessRate >= 0.4)
                                qualityText += "System Quality: Low (40-59% success rate)";
                            else
                                qualityText += "System Quality: Poor (<40% success rate)";
                        }
                        catch (Exception ex)
                        {
                            qualityText = $"Error loading quality data: {ex.Message}";
                        }

                        Dispatcher.Invoke(() =>
                        {
                            SearchStrategyText.Text = strategyText;
                            QualityDistributionText.Text = qualityText;
                        });
                    }
                }
                catch (Exception ex)
                {
                    Dispatcher.Invoke(() =>
                    {
                        SearchStrategyText.Text = $"Connection Error: {ex.Message}";
                        QualityDistributionText.Text = $"Connection Error: {ex.Message}";
                    });
                    System.Diagnostics.Debug.WriteLine($"System performance error: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Updates overall system health indicator
        /// </summary>
        private void UpdateSystemHealth()
        {
            try
            {
                var successRateText = SuccessRateText.Text.Replace("%", "");
                var successRate = double.TryParse(successRateText, out var rate) ? rate / 100.0 : 0.5;
                var securityEvents = int.TryParse(SecurityEventsText.Text, out var events) ? events : 0;
                var conflicts = int.TryParse(ConflictsText.Text, out var conflictCount) ? conflictCount : 0;

                string healthStatus;
                Brush healthColor;

                if (successRate >= 0.8 && securityEvents <= 2 && conflicts <= 3)
                {
                    healthStatus = "🟢 Healthy";
                    healthColor = new SolidColorBrush(Color.FromRgb(16, 124, 16));
                }
                else if (successRate >= 0.6 && securityEvents <= 10 && conflicts <= 10)
                {
                    healthStatus = "🟡 Caution";
                    healthColor = new SolidColorBrush(Color.FromRgb(255, 140, 0));
                }
                else
                {
                    healthStatus = "🔴 Needs Attention";
                    healthColor = new SolidColorBrush(Color.FromRgb(209, 52, 56));
                }

                SystemHealthIndicator.Text = $"System Status: {healthStatus}";
                SystemHealthIndicator.Foreground = healthColor;
            }
            catch
            {
                SystemHealthIndicator.Text = "System Status: Unknown";
                SystemHealthIndicator.Foreground = new SolidColorBrush(Colors.Gray);
            }
        }

        /// <summary>
        /// Helper method to get single values from database
        /// </summary>
        private int GetScalarValue(OdbcConnection conn, string sql)
        {
            try
            {
                using (var cmd = new OdbcCommand(sql, conn))
                {
                    var result = cmd.ExecuteScalar();
                    return result == null || result == DBNull.Value ? 0 : Convert.ToInt32(result);
                }
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// Updates UI during loading
        /// </summary>
        private void UpdateLoadingState(string message)
        {
            SystemHealthIndicator.Text = message;
            SystemHealthIndicator.Foreground = new SolidColorBrush(Colors.Orange);
        }

        /// <summary>
        /// Updates UI on error
        /// </summary>
        private void UpdateErrorState(string message)
        {
            SystemHealthIndicator.Text = $"Error: {message}";
            SystemHealthIndicator.Foreground = new SolidColorBrush(Colors.Red);
        }

        /// <summary>
        /// Updates last refresh time
        /// </summary>
        private void UpdateLastUpdatedTime()
        {
            LastUpdated.Text = $"Last updated: {DateTime.Now:HH:mm:ss}";
        }

        /// <summary>
        /// Database connection helper
        /// </summary>
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        // Event Handlers
        private async void RefreshButton_Click(object sender, RoutedEventArgs e)
        {
            RefreshButton.IsEnabled = false;
            try
            {
                await LoadAnalyticsAsync();
            }
            finally
            {
                RefreshButton.IsEnabled = true;
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}