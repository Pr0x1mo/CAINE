using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Data.Odbc;

namespace CAINE.MachineLearning
{
    /// <summary>
    /// Interactive Decision Tree for step-by-step troubleshooting
    /// </summary>
    public class InteractiveSolutionTree
    {
        private readonly string connectionString = "DSN=CAINE_Databricks;";
        private DecisionNode rootNode;
        private Dictionary<string, DecisionNode> nodeCache = new Dictionary<string, DecisionNode>();

        /// <summary>
        /// Represents a decision point in the troubleshooting tree
        /// </summary>
        public class DecisionNode
        {
            public string NodeId { get; set; } = Guid.NewGuid().ToString();
            public string Question { get; set; }
            public string ActionIfYes { get; set; }
            public string ActionIfNo { get; set; }
            public DecisionNode YesChild { get; set; }
            public DecisionNode NoChild { get; set; }
            public double SuccessRate { get; set; } = 0.5; // Default 50%
            public int TimesUsed { get; set; } = 0;
            public string ErrorCategory { get; set; }
            public List<string> RelatedErrorHashes { get; set; } = new List<string>();
            public bool IsLeaf { get; set; } = false;
            public string Solution { get; set; }
        }

        /// <summary>
        /// User's path through the decision tree
        /// </summary>
        public class TreePath
        {
            public List<DecisionNode> NodesVisited { get; set; } = new List<DecisionNode>();
            public List<bool> Decisions { get; set; } = new List<bool>(); // true=Yes, false=No
            public DateTime StartTime { get; set; } = DateTime.Now;
            public DateTime? EndTime { get; set; }
            public bool WasSuccessful { get; set; }
            public string SessionId { get; set; } = Guid.NewGuid().ToString();
            public string ErrorHash { get; set; }
        }

        /// <summary>
        /// Build an interactive tree based on error patterns
        /// </summary>
        public async Task<DecisionNode> BuildInteractiveTreeAsync(string errorHash, string errorText)
        {
            // Check cache first
            if (nodeCache.ContainsKey(errorHash))
                return nodeCache[errorHash];

            // Build tree from database patterns and ML model
            rootNode = await GenerateTreeFromPatternsAsync(errorHash, errorText);

            // Cache the tree
            nodeCache[errorHash] = rootNode;

            return rootNode;
        }

        /// <summary>
        /// Generate tree structure from historical data
        /// </summary>
        //private async Task<DecisionNode> GenerateTreeFromPatternsAsync(string errorHash, string errorText)
        //{
        //    return await Task.Run(() =>
        //    {
        //        // Start with category identification
        //        var root = new DecisionNode
        //        {
        //            Question = "Is this error occurring during connection/network operations?",
        //            ActionIfYes = "Proceed to network troubleshooting",
        //            ActionIfNo = "Check for authentication or permission issues",
        //            ErrorCategory = "root"
        //        };

        //        // Network branch
        //        var networkNode = new DecisionNode
        //        {
        //            Question = "Can you ping the target server?",
        //            ActionIfYes = "Test port connectivity",
        //            ActionIfNo = "Check network configuration and firewall",
        //            ErrorCategory = "network"
        //        };

        //        // Authentication branch
        //        var authNode = new DecisionNode
        //        {
        //            Question = "Are the credentials recently changed?",
        //            ActionIfYes = "Update stored credentials",
        //            ActionIfNo = "Verify account permissions",
        //            ErrorCategory = "authentication"
        //        };

        //        // Build tree structure
        //        root.YesChild = networkNode;
        //        root.NoChild = authNode;

        //        // Add leaf nodes with solutions
        //        networkNode.YesChild = CreateLeafNode("Check if the specific port is open using telnet or nc");
        //        networkNode.NoChild = CreateLeafNode("1. Verify network connectivity\n2. Check firewall rules\n3. Confirm DNS resolution");

        //        authNode.YesChild = CreateLeafNode("1. Update connection string\n2. Clear credential cache\n3. Re-authenticate");
        //        authNode.NoChild = CreateLeafNode("1. Verify account is not locked\n2. Check group permissions\n3. Review recent security changes");

        //        // Load success rates from database
        //        LoadNodeStatistics(root);

        //        return root;
        //    });
        //}
        private async Task<DecisionNode> GenerateTreeFromPatternsAsync(string errorHash, string errorText)
        {
            return await Task.Run(() =>
            {
                DecisionNode root = null;

                try
                {
                    using (var conn = new OdbcConnection(connectionString))
                    {
                        conn.Open();

                        // Try to get REAL solution from your database
                        var sql = $@"
                    SELECT kb.resolution_steps, 
                           COALESCE(fb.success_rate, 0.5) as success_rate
                    FROM default.cai_error_kb kb
                    LEFT JOIN (
                        SELECT solution_hash, 
                               AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM default.cai_solution_feedback
                        GROUP BY solution_hash
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE kb.error_hash = '{errorHash}'
                    LIMIT 1";

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                // Found real solution in database!
                                var steps = rdr.GetString(0);
                                var successRate = rdr.GetDouble(1);

                                // Parse the steps into a simple tree
                                root = new DecisionNode
                                {
                                    Question = "Found solution in database. Ready to apply?",
                                    ActionIfYes = "Apply the solution below",
                                    ActionIfNo = "Try alternative approach",
                                    ErrorCategory = CategorizeError(errorText),
                                    SuccessRate = successRate
                                };

                                // Parse steps if they're array format: ["step1", "step2"]
                                var stepsText = ConvertArrayToString(steps);

                                // Create solution node
                                root.YesChild = CreateLeafNode(stepsText);
                                root.YesChild.SuccessRate = successRate;

                                // Alternative path
                                root.NoChild = new DecisionNode
                                {
                                    Question = "Would you like to try the CAINE API for an alternative solution?",
                                    ActionIfYes = "Use CAINE API",
                                    ActionIfNo = "Return to main window",
                                    ErrorCategory = root.ErrorCategory
                                };

                                return root;
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Database query failed: {ex.Message}");
                }

                // If no database match, return the hardcoded tree
                root = new DecisionNode
                {
                    Question = "Is this error occurring during connection/network operations?",
                    ActionIfYes = "Proceed to network troubleshooting",
                    ActionIfNo = "Check for authentication or permission issues",
                    ErrorCategory = "root"
                };

                // ... rest of your existing hardcoded tree ...

                return root;
            });
        }

        // Add this helper method (copy from MainWindow)
        private static string ConvertArrayToString(object arrayValue)
        {
            if (arrayValue == null) return "";
            var arrayStr = arrayValue.ToString();
            if (string.IsNullOrEmpty(arrayStr)) return "";

            if (arrayStr.StartsWith("[") && arrayStr.EndsWith("]"))
            {
                var content = arrayStr.Trim('[', ']');
                var steps = content.Split(',')
                                  .Select(s => s.Trim(' ', '"', '\''))
                                  .Where(s => !string.IsNullOrEmpty(s));
                return string.Join(Environment.NewLine, steps);
            }

            return arrayStr;
        }

        /// <summary>
        /// Create a leaf node with a solution
        /// </summary>
        private DecisionNode CreateLeafNode(string solution)
        {
            return new DecisionNode
            {
                IsLeaf = true,
                Solution = solution,
                Question = "Solution:",
                SuccessRate = 0.5 // Will be updated from feedback
            };
        }

        /// <summary>
        /// Load historical success rates from database
        /// </summary>
        private void LoadNodeStatistics(DecisionNode node)
        {
            if (node == null) return;

            try
            {
                using (var conn = new OdbcConnection(connectionString))
                {
                    conn.Open();
                    // This would load actual statistics from your feedback table
                    // For now, using placeholder logic

                    // Recursively load for children
                    LoadNodeStatistics(node.YesChild);
                    LoadNodeStatistics(node.NoChild);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Failed to load node statistics: {ex.Message}");
            }
        }

        /// <summary>
        /// Update tree weights based on user's path and outcome
        /// </summary>
        public async Task UpdateTreeWeightsAsync(TreePath path, bool wasSuccessful)
        {
            await Task.Run(() =>
            {
                foreach (var node in path.NodesVisited)
                {
                    node.TimesUsed++;

                    // Update success rate using running average
                    if (wasSuccessful)
                    {
                        node.SuccessRate = ((node.SuccessRate * (node.TimesUsed - 1)) + 1.0) / node.TimesUsed;
                    }
                    else
                    {
                        node.SuccessRate = (node.SuccessRate * (node.TimesUsed - 1)) / node.TimesUsed;
                    }
                }

                // Persist to database
                PersistTreeStatistics(path);
            });
        }

        /// <summary>
        /// Save tree statistics to database
        /// </summary>
        private void PersistTreeStatistics(TreePath path)
        {
            try
            {
                using (var conn = new OdbcConnection(connectionString))
                {
                    conn.Open();

                    // Store the path taken and outcome
                    var sql = @"
                        INSERT INTO cai_tree_paths (
                            session_id, error_hash, path_json, 
                            was_successful, duration_seconds, created_at
                        ) VALUES (?, ?, ?, ?, ?, current_timestamp())";

                    // This would save the actual path data
                    // Implementation depends on your database schema
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Failed to persist tree statistics: {ex.Message}");
            }
        }

        /// <summary>
        /// Get recommended path based on success rates
        /// </summary>
        public List<DecisionNode> GetRecommendedPath(DecisionNode startNode)
        {
            var path = new List<DecisionNode>();
            var current = startNode;

            while (current != null && !current.IsLeaf)
            {
                path.Add(current);

                // Choose path with higher success rate
                if (current.YesChild?.SuccessRate >= current.NoChild?.SuccessRate)
                {
                    current = current.YesChild;
                }
                else
                {
                    current = current.NoChild;
                }
            }

            if (current != null)
                path.Add(current); // Add the leaf node

            return path;
        }

        /// <summary>
        /// Rebuild tree if success rates change significantly
        /// </summary>
        public async Task<bool> ShouldRebuildTreeAsync(string errorHash)
        {
            // Check if patterns have changed significantly
            // This would analyze recent feedback to determine if tree needs restructuring
            return await Task.Run(() =>
            {
                // Placeholder logic - in production, this would check:
                // 1. Number of failed paths
                // 2. Significant changes in success rates
                // 3. New patterns discovered
                return false;
            });
        }

        /// <summary>
        /// Generate tree from ML decision tree model
        /// </summary>
        public async Task<DecisionNode> GenerateTreeFromMLModelAsync(CaineMLEngine mlEngine, string errorText)
        {
            return await Task.Run(() =>
            {
                // Extract features from error
                var features = ExtractFeatures(errorText);

                // Get cluster recommendation
                var (template, confidence) = mlEngine.GetClusterRecommendationAsync(features).Result;

                // Build tree based on error category
                var category = CategorizeError(errorText);

                var root = new DecisionNode
                {
                    Question = GetCategoryQuestion(category),
                    ErrorCategory = category,
                    SuccessRate = confidence
                };

                // Build category-specific branches
                BuildCategoryBranches(root, category, template);

                return root;
            });
        }

        private string GetCategoryQuestion(string category)
        {
            return category switch
            {
                "network" => "Is this a network/connection related error?",
                "security" => "Is this an authentication or permission issue?",
                "nullref" => "Is this a null reference or missing data error?",
                "performance" => "Is this a timeout or performance issue?",
                _ => "Is the service/database currently accessible?"
            };
        }
        private void BuildCategoryBranches(DecisionNode root, string category, string template)
        {
            switch (category)
            {
                case "network":
                    var pingNode = new DecisionNode
                    {
                        Question = "Can you ping the target server?",
                        ActionIfYes = "Test specific port connectivity",
                        ActionIfNo = "Check network configuration",
                        ErrorCategory = "network"
                    };

                    var portNode = new DecisionNode
                    {
                        Question = "Is the required port open (telnet/nc test)?",
                        ActionIfYes = "Check service status on server",
                        ActionIfNo = "Open port in firewall",
                        ErrorCategory = "network"
                    };

                    root.YesChild = pingNode;
                    pingNode.YesChild = portNode;
                    pingNode.NoChild = CreateLeafNode("1. Check network cable/WiFi\n2. Verify IP configuration\n3. Check DNS settings");
                    portNode.YesChild = CreateLeafNode(template ?? "Restart the service and check logs");
                    portNode.NoChild = CreateLeafNode("1. Configure firewall rules\n2. Check port binding\n3. Verify service is listening");

                    root.NoChild = CreateLeafNode("This doesn't appear to be a network issue. Check other categories.");
                    break;

                case "security":
                    var credNode = new DecisionNode
                    {
                        Question = "Have credentials been recently changed?",
                        ActionIfYes = "Update stored credentials",
                        ActionIfNo = "Check account status",
                        ErrorCategory = "security"
                    };

                    root.YesChild = credNode;
                    credNode.YesChild = CreateLeafNode(template ?? "1. Update connection string\n2. Clear credential cache\n3. Re-authenticate");
                    credNode.NoChild = CreateLeafNode("1. Check if account is locked\n2. Verify permissions\n3. Check group membership");

                    root.NoChild = CreateLeafNode("This doesn't appear to be a security issue. Check other categories.");
                    break;

                case "nullref":
                    var dataNode = new DecisionNode
                    {
                        Question = "Is required data present in the database?",
                        ActionIfYes = "Check data mapping",
                        ActionIfNo = "Check data source",
                        ErrorCategory = "nullref"
                    };

                    root.YesChild = dataNode;
                    dataNode.YesChild = CreateLeafNode(template ?? "1. Verify column names\n2. Check data types\n3. Review mapping configuration");
                    dataNode.NoChild = CreateLeafNode("1. Check if data was deleted\n2. Verify data load process\n3. Check for failed imports");

                    root.NoChild = CreateLeafNode("This doesn't appear to be a null reference issue. Check other categories.");
                    break;

                case "performance":
                    var loadNode = new DecisionNode
                    {
                        Question = "Is system under heavy load?",
                        ActionIfYes = "Optimize resources",
                        ActionIfNo = "Check specific bottlenecks",
                        ErrorCategory = "performance"
                    };

                    root.YesChild = loadNode;
                    loadNode.YesChild = CreateLeafNode(template ?? "1. Increase timeout values\n2. Scale resources\n3. Optimize queries");
                    loadNode.NoChild = CreateLeafNode("1. Check for blocking queries\n2. Review execution plans\n3. Check network latency");

                    root.NoChild = CreateLeafNode("This doesn't appear to be a performance issue. Check other categories.");
                    break;

                default:
                    var serviceNode = new DecisionNode
                    {
                        Question = "Is the service/database running?",
                        ActionIfYes = "Check configuration",
                        ActionIfNo = "Start the service",
                        ErrorCategory = "general"
                    };

                    root.YesChild = serviceNode;
                    serviceNode.YesChild = CreateLeafNode(template ?? "1. Review configuration files\n2. Check error logs\n3. Verify connectivity");
                    serviceNode.NoChild = CreateLeafNode("1. Start the service\n2. Check for startup errors\n3. Verify dependencies");

                    root.NoChild = CreateLeafNode("Unable to categorize. Try manual troubleshooting.");
                    break;
            }
        }
        // Add these helper methods from your MainWindow
        private double[] ExtractFeatures(string errorText)
        {
            var features = new List<double>();

            // Text length features
            features.Add(errorText.Length);
            features.Add(errorText.Split(' ').Length); // Word count
            features.Add(errorText.Split('\n').Length); // Line count

            // Keyword presence (binary features)
            string[] importantKeywords = {
        "connection", "timeout", "permission", "denied", "failed",
        "database", "network", "authentication", "invalid", "null"
    };

            foreach (var keyword in importantKeywords)
            {
                features.Add(errorText.ToLower().Contains(keyword) ? 1.0 : 0.0);
            }

            // Error code extraction (if present)
            var errorCode = ExtractErrorCode(errorText);
            features.Add(errorCode);

            // Time-based features
            features.Add(DateTime.Now.Hour); // Hour of day
            features.Add((int)DateTime.Now.DayOfWeek); // Day of week

            // Complexity metrics
            features.Add(CalculateTextComplexity(errorText));
            features.Add(CountSpecialCharacters(errorText));

            return features.ToArray();
        }

        private string CategorizeError(string errorSignature)
        {
            // Simple categorization based on keywords
            if (errorSignature.Contains("connection") || errorSignature.Contains("network"))
                return "network";
            if (errorSignature.Contains("permission") || errorSignature.Contains("denied"))
                return "security";
            if (errorSignature.Contains("null") || errorSignature.Contains("reference"))
                return "nullref";
            if (errorSignature.Contains("timeout"))
                return "performance";

            return "general";
        }

        // You'll also need these helper methods:
        private int ExtractErrorCode(string errorText)
        {
            // Extract numeric error codes if present
            var match = System.Text.RegularExpressions.Regex.Match(errorText, @"(?:error|code)[\s:]*(\d+)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

            if (match.Success && int.TryParse(match.Groups[1].Value, out int code))
            {
                return code;
            }

            return 0;
        }

        private double CalculateTextComplexity(string text)
        {
            // Simple text complexity metric
            var sentences = text.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            var words = text.Split(' ');

            if (sentences.Length == 0) return 0;

            double avgWordsPerSentence = (double)words.Length / sentences.Length;
            double avgWordLength = words.Average(w => w.Length);

            // Flesch Reading Ease approximation
            return Math.Min(100, 206.835 - 1.015 * avgWordsPerSentence - 84.6 * (avgWordLength / 10));
        }

        private int CountSpecialCharacters(string text)
        {
            return text.Count(c => !char.IsLetterOrDigit(c) && !char.IsWhiteSpace(c));
        }
    }
}