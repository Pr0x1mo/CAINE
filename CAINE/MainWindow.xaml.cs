using System;
using System.Collections.Generic;
using System.Data.Odbc;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using CAINE.Security;

namespace CAINE
{
    public partial class MainWindow : Window
    {
        // Database connection settings
        private const string DsnName = "CAINE_Databricks";
        private const string TableKB = "default.cai_error_kb";
        private const string TablePatterns = "default.cai_error_patterns";
        private const string TableFeedback = "default.cai_solution_feedback";
        private const string TableKBVersions = "default.cai_kb_versions";

        // Enhanced ML Configuration with Confidence Scoring
        private const int VectorCandidateLimit = 300;
        private const int LikeTokenMax = 4;
        private const double VectorMinCosine = 0.70;
        private const double FeedbackBoostThreshold = 0.7;

        // Confidence and reliability parameters
        private const int MinFeedbackForConfidence = 5;
        private const double ConflictThreshold = 0.3;
        private const double HighConfidenceThreshold = 0.85;

        // Stop words for tokenization
        private static readonly string[] Stop = new[]
        {
            "error","failed","login","user","query","server","execute","executing","ssis","package",
            "for","the","with","from","could","not","because","property","set","correctly","message",
            "reason","reasons","possible","and","was","is","are","to","in","of","on","by","sqlstate","code"
        };

        // Session tracking with enhanced metadata
        private string currentSessionId = Guid.NewGuid().ToString();
        private string currentSolutionHash = null;
        private string currentSolutionSource = null;
        private double currentSolutionConfidence = 0.0;

        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

        // Enhanced solution metadata for confidence scoring
        public class SolutionResult
        {
            public string Steps { get; set; }
            public string Hash { get; set; }
            public string Source { get; set; }
            public double Confidence { get; set; }
            public int FeedbackCount { get; set; }
            public double SuccessRate { get; set; }
            public bool HasConflicts { get; set; }
            public DateTime LastUpdated { get; set; }
            public string Version { get; set; }
        }

        public MainWindow()
        {
            InitializeComponent();
            EnsureTls12();
            SecurityValidator.InitializeSecurityTables(); // Initialize security first
            InitializeEnhancedTables();
        }

        /// <summary>
        /// ENHANCED DATABASE SETUP: Creates tables with version control and conflict resolution
        /// </summary>
        private async void InitializeEnhancedTables()
        {
            try
            {
                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        // Create knowledge base versions table for version control
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TableKBVersions} (
                                version_id STRING,
                                kb_entry_id STRING,
                                error_hash STRING,
                                resolution_steps ARRAY<STRING>,
                                created_at TIMESTAMP,
                                created_by STRING,
                                change_type STRING,
                                parent_version STRING,
                                confidence_score DOUBLE,
                                feedback_count INT,
                                is_active BOOLEAN
                            ) USING DELTA");

                        // Enhanced feedback table with conflict detection
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TableFeedback} (
                                feedback_id STRING,
                                session_id STRING,
                                solution_hash STRING,
                                solution_source STRING,
                                solution_version STRING,
                                was_helpful BOOLEAN,
                                confidence_rating DOUBLE,
                                user_comment STRING,
                                user_expertise STRING,
                                created_at TIMESTAMP,
                                error_signature STRING,
                                resolution_time_minutes INT,
                                environment_context STRING
                            ) USING DELTA");

                        // Enhanced patterns table with priority and effectiveness tracking
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TablePatterns} (
                                pattern_id STRING,
                                pattern_regex STRING,
                                priority INT,
                                resolution_steps ARRAY<STRING>,
                                created_at TIMESTAMP,
                                description STRING,
                                effectiveness_score DOUBLE,
                                usage_count INT,
                                last_successful_match TIMESTAMP
                            ) USING DELTA");

                        System.Diagnostics.Debug.WriteLine("Enhanced CAINE tables initialized successfully");
                    }
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Enhanced table initialization failed: {ex.Message}");
            }
        }

        /// <summary>
        /// SECURITY FIX: Execute parameterized commands safely
        /// </summary>
        private void ExecuteSecureCommand(OdbcConnection conn, string sql, Dictionary<string, object> parameters = null)
        {
            using (var cmd = new OdbcCommand(sql, conn))
            {
                if (parameters != null)
                {
                    foreach (var param in parameters)
                    {
                        cmd.Parameters.AddWithValue(param.Key, param.Value);
                    }
                }
                cmd.ExecuteNonQuery();
            }
        }

        /// <summary>
        /// SECURITY ENHANCEMENT: Comprehensive input sanitization
        /// </summary>
        private static string SecureEscape(string input)
        {
            if (string.IsNullOrEmpty(input)) return "";

            var cleaned = input.Replace("--", "")
                              .Replace("/*", "")
                              .Replace("*/", "")
                              .Replace(";", "")
                              .Replace("xp_", "x p_")
                              .Replace("sp_", "s p_")
                              .Replace("\\", "\\\\")
                              .Replace("'", "''")
                              .Replace("\"", "\\\"")
                              .Replace("\r", "")
                              .Replace("\n", "\\n")
                              .Replace("\t", "\\t");

            return cleaned.Length > 4000 ? cleaned.Substring(0, 4000) + "..." : cleaned;
        }

        /// <summary>
        /// ENHANCED SEARCH: Main search function with confidence scoring and conflict resolution
        /// </summary>
        private async void BtnSearch_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Security validation first
                var validation = SecurityValidator.ValidateInput(ErrorInput.Text, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"Security validation failed: {validation.ErrorMessage}";
                    return;
                }

                BtnCaineApi.IsEnabled = false;
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Searching with enhanced ML algorithms...";

                var cleanErrorInput = validation.CleanInput;
                var sig = Normalize(cleanErrorInput);
                var hash = Sha256Hex(sig);

                System.Diagnostics.Debug.WriteLine($"DEBUG: Generated hash: {hash}");

                // Reset session tracking
                currentSessionId = Guid.NewGuid().ToString();
                currentSolutionHash = null;
                currentSolutionSource = null;
                currentSolutionConfidence = 0.0;

                // Try exact match first
                var exactResult = await TryExactMatchAsync(hash);
                if (exactResult != null && !string.IsNullOrWhiteSpace(exactResult.Steps))
                {
                    System.Diagnostics.Debug.WriteLine($"DEBUG: Exact match found with confidence: {exactResult.Confidence}");

                    currentSolutionHash = exactResult.Hash;
                    currentSolutionSource = exactResult.Source;
                    currentSolutionConfidence = exactResult.Confidence;

                    // Show result regardless of confidence for debugging
                    var confidenceText = GetConfidenceText(exactResult);
                    ResultBox.Text = $"{confidenceText}\n\n{exactResult.Steps}";

                    EnableFeedbackButtons();
                    return;
                }

                // If no exact match, continue with other strategies...
                ResultBox.Text = "No exact match found. Click 'Use CAINE API' for AI assistance.";
                BtnCaineApi.IsEnabled = true;
            }
            catch (Exception ex)
            {
                ResultBox.Text = $"Search error: {ex.Message}\n\nYou can try 'Use CAINE API'.";
                BtnCaineApi.IsEnabled = true;
            }
            finally
            {
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }

        /// <summary>
        /// ENHANCED AI CONSULTATION: ChatGPT with confidence scoring and learning integration
        /// </summary>
        private async void BtnCaineApi_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                BtnCaineApi.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Consulting CAINE AI (OpenAI) with enhanced context...";

                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                // Security validation for AI input
                var validation = SecurityValidator.ValidateInput(err, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"AI input validation failed: {validation.ErrorMessage}";
                    return;
                }

                // Build enhanced conversation with past successful solutions and confidence data
                var messages = await BuildEnhancedChatHistoryAsync(validation.CleanInput);

                // Ask ChatGPT for advice with enhanced context
                var gptResponse = await AskOpenAIAsync(messages);

                // Track this solution for feedback learning
                currentSolutionHash = Sha256Hex(gptResponse);
                currentSolutionSource = "openai_enhanced";
                currentSolutionConfidence = 0.5; // Default for new AI solutions

                // Format response with confidence indicator
                var confidenceText = "AI Generated Solution (Confidence: Learning) | Please rate this solution to help CAINE learn";
                ResultBox.Text = $"{confidenceText}\n\n{gptResponse}";

                // Extract clean steps for the teaching interface
                var numbered = ExtractNumberedLines(gptResponse).ToArray();
                TeachSteps.Text = numbered.Length > 0
                    ? string.Join(Environment.NewLine, numbered)
                    : gptResponse;

                // Enable feedback buttons for learning
                EnableFeedbackButtons();
            }
            catch (Exception ex)
            {
                ResultBox.Text = "OpenAI consultation failed:\n" + ex.Message;
            }
            finally
            {
                BtnCaineApi.IsEnabled = true;
            }
        }

        /// <summary>
        /// ENHANCED TEACHING: Knowledge addition with version control and conflict resolution
        /// </summary>
        private async void BtnTeach_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                ResultBox.Text = "Teaching CAINE with enhanced learning...";

                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                // Security validation for teaching input
                var validation = SecurityValidator.ValidateInput(err, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"Teaching input validation failed: {validation.ErrorMessage}";
                    return;
                }

                // Parse solution steps
                var raw = (TeachSteps.Text ?? "").Replace("\r", "");
                var lines = raw.Split('\n');
                var stepsLines = lines.Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
                if (stepsLines.Length == 0)
                {
                    ResultBox.Text = "Enter at least one step (one per line).";
                    return;
                }

                // Enhanced teaching with version control
                var cleanErrorInput = validation.CleanInput;
                var sig = Normalize(cleanErrorInput);
                var hash = Sha256Hex(sig);

                // Check if solution already exists and handle conflicts
                var existingConflicts = await CheckForExistingSolutions(hash, stepsLines);

                if (existingConflicts.HasConflicts)
                {
                    var conflictMessage = $"Warning: Similar solution exists with {existingConflicts.ExistingSuccessRate:P0} success rate. ";
                    conflictMessage += "Your solution will be added as an alternative approach.";
                    ResultBox.Text += "\n" + conflictMessage;
                }

                // Create embeddings for enhanced search
                float[] emb = null;
                try
                {
                    emb = await EmbedAsync(cleanErrorInput + "\n\n" + string.Join("\n", stepsLines));
                }
                catch (Exception embedEx)
                {
                    System.Diagnostics.Debug.WriteLine($"Embedding creation failed: {embedEx.Message}");
                }

                // Enhanced UPSERT with version control
                await TeachEnhancedSolution(hash, sig, cleanErrorInput, stepsLines, emb, existingConflicts);

                // Record positive feedback if user is correcting a bad suggestion
                if (!string.IsNullOrEmpty(currentSolutionHash) && currentSolutionHash != hash)
                {
                    currentSolutionHash = hash;
                    currentSolutionSource = "human_teaching";
                    currentSolutionConfidence = 0.8; // High confidence for human-provided solutions
                    await Task.Run(() => RecordEnhancedFeedback(true, "User-provided teaching", 5));
                }

                var successMessage = $"Successfully taught CAINE this solution.\nError Hash: {hash}";
                if (existingConflicts.HasConflicts)
                {
                    successMessage += "\nAdded as alternative approach due to conflicts.";
                }
                ResultBox.Text = successMessage;
            }
            catch (Exception ex)
            {
                ResultBox.Text = "Teaching error:\n" + ex.Message;
            }
            finally
            {
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }

        /// <summary>
        /// ENHANCED EXACT MATCH: With confidence scoring and version control
        /// </summary>
        /// <summary>
        /// ENHANCED EXACT MATCH: With real confidence scoring from feedback data
        /// </summary>
        private async Task<SolutionResult> TryExactMatchAsync(string hash)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Updated SQL to join with feedback table for real confidence data
                        var sql = $@"
                    SELECT 
                        kb.resolution_steps, 
                        kb.error_hash,
                        COALESCE(fb.success_rate, 0.5) as success_rate,
                        COALESCE(fb.feedback_count, 0) as feedback_count,
                        CASE WHEN COALESCE(fb.conflict_rate, 0.0) > {ConflictThreshold} THEN true ELSE false END as has_conflicts
                    FROM {TableKB} kb
                    LEFT JOIN (
                        SELECT 
                            solution_hash,
                            AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                            COUNT(*) as feedback_count,
                            AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE kb.error_hash = '{hash}'
                    ORDER BY kb.created_at DESC
                    LIMIT 1";

                        System.Diagnostics.Debug.WriteLine($"DEBUG: Searching for hash: {hash}");

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : ConvertArrayToString(rdr.GetValue(0));
                                var resultHash = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
                                var successRate = rdr.GetDouble(2);
                                var feedbackCount = rdr.GetInt32(3);
                                var hasConflicts = rdr.GetBoolean(4);

                                // Calculate real confidence using the same method as feedback updates
                                var confidence = CalculateConfidence(successRate, feedbackCount, hasConflicts ? 0.4 : 0.0);

                                System.Diagnostics.Debug.WriteLine($"DEBUG: Found solution - Success: {successRate:P0}, Feedback: {feedbackCount}, Confidence: {confidence:P0}");

                                return new SolutionResult
                                {
                                    Steps = steps,
                                    Hash = resultHash,
                                    Source = "exact_match",
                                    SuccessRate = successRate,
                                    FeedbackCount = feedbackCount,
                                    HasConflicts = hasConflicts,
                                    Confidence = confidence,
                                    Version = "1.0"
                                };
                            }
                            else
                            {
                                System.Diagnostics.Debug.WriteLine("DEBUG: No records found for hash");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"DEBUG: Exception: {ex.Message}");
                }
                return null;
            });
        }
        private static string ConvertArrayToString(object arrayValue)
        {
            if (arrayValue == null) return "";

            var arrayStr = arrayValue.ToString();
            if (string.IsNullOrEmpty(arrayStr)) return "";

            // Handle Databricks array format: ["step1", "step2"]
            if (arrayStr.StartsWith("[") && arrayStr.EndsWith("]"))
            {
                // Remove brackets and quotes, split by comma, join with newlines
                var content = arrayStr.Trim('[', ']');
                var steps = content.Split(',')
                                  .Select(s => s.Trim(' ', '"', '\''))
                                  .Where(s => !string.IsNullOrEmpty(s));
                return string.Join(Environment.NewLine, steps);
            }

            return arrayStr;
        }

        /// <summary>
        /// ENHANCED PATTERN MATCHING: With effectiveness tracking
        /// </summary>
        private async Task<SolutionResult> TryEnhancedPatternMatchAsync(string sig)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            SELECT resolution_steps, effectiveness_score, usage_count
                            FROM {TablePatterns}
                            WHERE ? RLIKE pattern_regex
                            ORDER BY priority DESC, effectiveness_score DESC
                            LIMIT 1";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("signature", sig);
                            using (var rdr = cmd.ExecuteReader())
                            {
                                if (rdr.Read())
                                {
                                    var steps = rdr.GetString(0);
                                    var effectiveness = rdr.IsDBNull(1) ? 0.5 : rdr.GetDouble(1);
                                    var usageCount = rdr.IsDBNull(2) ? 0 : rdr.GetInt32(2);

                                    return new SolutionResult
                                    {
                                        Steps = steps,
                                        Hash = Sha256Hex(steps),
                                        Source = "pattern_match",
                                        SuccessRate = effectiveness,
                                        FeedbackCount = usageCount,
                                        HasConflicts = false,
                                        Confidence = Math.Min(0.9, effectiveness + (usageCount * 0.01)),
                                        Version = "1.0"
                                    };
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Pattern match failed: {ex.Message}");
                }
                return null;
            });
        }

        /// <summary>
        /// KEYWORD SEARCH: With confidence scoring
        /// </summary>
        private async Task<SolutionResult> TryKeywordMatchAsync(string sig)
        {
            return await Task.Run(() =>
            {
                try
                {
                    var tokens = Tokens(sig);
                    if (tokens.Length == 0) return null;

                    using (var conn = OpenConn())
                    {
                        var likeConditions = tokens.Select((t, i) => "kb.error_signature LIKE ?").ToArray();
                        var whereClause = string.Join(" AND ", likeConditions);

                        var sql = $@"
                            SELECT kb.resolution_steps, kb.error_hash,
                                   COALESCE(fb.success_rate, 0.5) as success_rate,
                                   COALESCE(fb.feedback_count, 0) as feedback_count
                            FROM {TableKB} kb
                            LEFT JOIN (
                                SELECT solution_hash,
                                       AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                                       COUNT(*) as feedback_count
                                FROM {TableFeedback}
                                GROUP BY solution_hash
                            ) fb ON kb.error_hash = fb.solution_hash
                            WHERE {whereClause}
                            ORDER BY fb.success_rate DESC, kb.created_at DESC
                            LIMIT 1";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            for (int i = 0; i < tokens.Length; i++)
                            {
                                cmd.Parameters.AddWithValue($"token{i}", $"%{tokens[i]}%");
                            }

                            using (var rdr = cmd.ExecuteReader())
                            {
                                if (rdr.Read())
                                {
                                    var steps = rdr.GetString(0);
                                    var hash = rdr.GetString(1);
                                    var successRate = rdr.GetDouble(2);
                                    var feedbackCount = rdr.GetInt32(3);

                                    return new SolutionResult
                                    {
                                        Steps = steps,
                                        Hash = hash,
                                        Source = "keyword_match",
                                        SuccessRate = successRate,
                                        FeedbackCount = feedbackCount,
                                        HasConflicts = false,
                                        Confidence = CalculateConfidence(successRate, feedbackCount, 0.0),
                                        Version = "1.0"
                                    };
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Keyword match failed: {ex.Message}");
                }
                return null;
            });
        }

        /// <summary>
        /// ENHANCED VECTOR SIMILARITY: With scalable vector database simulation and conflict resolution
        /// </summary>
        private async Task<SolutionResult> TryEnhancedVectorMatchAsync(string rawError, string[] likeTokens)
        {
            try
            {
                var queryEmb = await EmbedAsync(rawError);
                var pageSize = 100;
                var candidates = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>();

                for (int page = 0; page < 3; page++)
                {
                    var pageCandidates = await GetVectorCandidatesPage(likeTokens, page, pageSize);
                    candidates.AddRange(pageCandidates);

                    if (pageCandidates.Count < pageSize) break;
                }

                if (candidates.Count == 0) return null;

                NormalizeInPlace(queryEmb);
                SolutionResult bestResult = null;
                double bestScore = -1;

                foreach (var c in candidates)
                {
                    var e = c.Emb;
                    NormalizeInPlace(e);

                    var cos = Cosine(queryEmb, e);

                    if (cos < VectorMinCosine) continue;

                    var confidence = CalculateConfidence(c.SuccessRate, c.FeedbackCount, c.HasConflicts ? 0.4 : 0.1);
                    var weightedScore = cos * (0.7 + confidence * 0.3);

                    if (weightedScore > bestScore)
                    {
                        bestScore = weightedScore;
                        bestResult = new SolutionResult
                        {
                            Steps = c.Steps,
                            Hash = c.Hash,
                            Source = "vector_similarity",
                            SuccessRate = c.SuccessRate,
                            FeedbackCount = c.FeedbackCount,
                            HasConflicts = c.HasConflicts,
                            Confidence = confidence
                        };
                    }
                }

                return bestResult;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Enhanced vector match failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// SCALABILITY FIX: Paginated vector candidate retrieval
        /// </summary>
        private async Task<List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>>
            GetVectorCandidatesPage(string[] likeTokens, int page, int pageSize)
        {
            return await Task.Run(() =>
            {
                var list = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>();

                try
                {
                    string where = (likeTokens != null && likeTokens.Length > 0)
                        ? "WHERE " + string.Join(" OR ", likeTokens.Select((t, i) => "kb.error_signature LIKE ?"))
                        : "";

                    var sql = $@"
                        SELECT kb.resolution_steps, kb.embedding, kb.error_hash,
                               COALESCE(fb.success_rate, 0.5) as success_rate,
                               COALESCE(fb.feedback_count, 0) as feedback_count,
                               CASE WHEN COALESCE(fb.conflict_rate, 0.0) > {ConflictThreshold} THEN true ELSE false END as has_conflicts
                        FROM {TableKB} kb
                        LEFT JOIN (
                            SELECT solution_hash,
                                   AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                                   COUNT(*) as feedback_count,
                                   AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate
                            FROM {TableFeedback}
                            GROUP BY solution_hash
                        ) fb ON kb.error_hash = fb.solution_hash
                        {where}
                        ORDER BY kb.created_at DESC
                        LIMIT {pageSize} OFFSET {page * pageSize}";

                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        if (likeTokens != null)
                        {
                            for (int i = 0; i < likeTokens.Length; i++)
                            {
                                cmd.Parameters.AddWithValue($"token{i}", $"%{likeTokens[i]}%");
                            }
                        }

                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : (rdr.GetValue(0)?.ToString() ?? "");
                                var hash = rdr.IsDBNull(2) ? "" : rdr.GetString(2);
                                var successRate = rdr.GetDouble(3);
                                var feedbackCount = rdr.GetInt32(4);
                                var hasConflicts = rdr.GetBoolean(5);

                                float[] emb = Array.Empty<float>();
                                if (!rdr.IsDBNull(1))
                                {
                                    var raw = rdr.GetValue(1)?.ToString() ?? "";
                                    emb = ParseFloatArray(raw);
                                }

                                if (emb.Length > 0)
                                {
                                    list.Add((steps, emb, hash, successRate, feedbackCount, hasConflicts));
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Vector candidates page {page} failed: {ex.Message}");
                }
                
                return list;
            });
        }

        // Helper Methods

        

        private double CalculateConfidence(double successRate, int feedbackCount, double conflictRate)
        {
            // Give reasonable default confidence for solutions with no feedback yet
            var baseConfidence = feedbackCount == 0 ? 0.5 : successRate;  // 50% default confidence

            // Adjust for sample size
            var sampleConfidence = feedbackCount == 0 ? 1.0 : Math.Min(1.0, feedbackCount / (double)MinFeedbackForConfidence);

            // Penalty for conflicts
            var conflictPenalty = Math.Max(0.0, 1.0 - conflictRate * 2);

            return baseConfidence * sampleConfidence * conflictPenalty;
        }

        private SolutionResult SelectBestSolution(List<SolutionResult> results)
        {
            if (results.Count == 0) return null;

            var sorted = results
                .Where(r => !string.IsNullOrWhiteSpace(r.Steps))
                .OrderByDescending(r => r.Confidence)
                .ThenByDescending(r => r.SuccessRate)
                .ThenByDescending(r => r.FeedbackCount)
                .ToList();

            var best = sorted.FirstOrDefault();
            return (best?.Confidence >= 0.1) ? best : null;
        }

        private string GetConfidenceText(SolutionResult result)
        {
            var confidenceLevel = result.Confidence >= HighConfidenceThreshold ? "High" :
                                 result.Confidence >= 0.6 ? "Medium" : "Low";

            var conflictWarning = result.HasConflicts ? " ⚠️ Some users reported mixed results." : "";

            return $"🎯 Confidence: {confidenceLevel} ({result.Confidence:P0}) | " +
                   $"Success Rate: {result.SuccessRate:P0} | " +
                   $"Based on {result.FeedbackCount} user ratings{conflictWarning}";
        }

        private async void RecordEnhancedFeedback(bool wasHelpful, string comment = "", int confidenceRating = 3)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                await Task.Run(() =>
                {
                    var escComment = SecureEscape(comment);
                    var escSig = SecureEscape(Normalize(ErrorInput.Text));

                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            INSERT INTO {TableFeedback} VALUES (
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp(), ?, ?, ?
                            )";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("feedback_id", Guid.NewGuid().ToString());
                            cmd.Parameters.AddWithValue("session_id", currentSessionId);
                            cmd.Parameters.AddWithValue("solution_hash", currentSolutionHash);
                            cmd.Parameters.AddWithValue("solution_source", currentSolutionSource);
                            cmd.Parameters.AddWithValue("solution_version", "1.0");
                            cmd.Parameters.AddWithValue("was_helpful", wasHelpful);
                            cmd.Parameters.AddWithValue("confidence_rating", (double)confidenceRating);
                            cmd.Parameters.AddWithValue("user_comment", escComment);
                            cmd.Parameters.AddWithValue("user_expertise", "intermediate");
                            cmd.Parameters.AddWithValue("error_signature", escSig);
                            cmd.Parameters.AddWithValue("resolution_time", 0);
                            cmd.Parameters.AddWithValue("environment", "unknown");

                            cmd.ExecuteNonQuery();
                        }
                    }

                    UpdateSolutionConfidence(currentSolutionHash);
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Enhanced feedback recording failed: {ex.Message}");
            }
        }

        private void UpdateSolutionConfidence(string solutionHash)
        {
            try
            {
                using (var conn = OpenConn())
                {
                    var sql = $@"
                        SELECT 
                            AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                            COUNT(*) as total_feedback,
                            AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate
                        FROM {TableFeedback}
                        WHERE solution_hash = ?";

                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        cmd.Parameters.AddWithValue("hash", solutionHash);
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var successRate = rdr.GetDouble(0);
                                var feedbackCount = rdr.GetInt32(1);
                                var conflictRate = rdr.GetDouble(2);

                                var newConfidence = CalculateConfidence(successRate, feedbackCount, conflictRate);

                                System.Diagnostics.Debug.WriteLine(
                                    $"Solution {solutionHash}: Confidence updated to {newConfidence:F2} " +
                                    $"(Success: {successRate:P0}, Feedback: {feedbackCount}, Conflicts: {conflictRate:P0})");
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Confidence update failed: {ex.Message}");
            }
        }
        private void BtnAnalytics_Click(object sender, RoutedEventArgs e)
        {
            var analyticsWindow = new AnalyticsWindow();
            analyticsWindow.Show();
        }

        // Enhanced feedback button handlers
        private async void BtnFeedbackYes_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                var errorText = ErrorInput.Text;
                var normalizedSig = SecureEscape(Normalize(errorText));

                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
            INSERT INTO {TableFeedback} (
                feedback_id,
                session_id,
                solution_hash,
                solution_source,
                was_helpful,
                user_comment,
                created_at,
                error_signature
            ) VALUES (
                '{Guid.NewGuid()}',
                '{currentSessionId}',
                '{currentSolutionHash}',
                '{currentSolutionSource}',
                true,
                '',
                current_timestamp(),
                '{normalizedSig}'
            )";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        System.Diagnostics.Debug.WriteLine("Positive feedback recorded successfully");
                    }
                });

                DisableFeedbackButtons();

                // ADD THIS NEW CODE HERE - Refresh the confidence display
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);

                    // Update just the first line of the result
                    var lines = ResultBox.Text.Split('\n');
                    lines[0] = newConfidenceText;
                    ResultBox.Text = string.Join("\n", lines);
                }
                // END NEW CODE

                ResultBox.Text += "\n\nThank you! Your positive feedback has been recorded.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Feedback failed: {ex.Message}");
            }
        }
        // Add this helper method anywhere in your MainWindow class
        private async Task<SolutionResult> GetUpdatedConfidence(string solutionHash)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                    SELECT 
                        AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                        COUNT(*) as feedback_count
                    FROM {TableFeedback}
                    WHERE solution_hash = '{solutionHash}'";

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var successRate = rdr.GetDouble(0);
                                var feedbackCount = rdr.GetInt32(1);

                                return new SolutionResult
                                {
                                    SuccessRate = successRate,
                                    FeedbackCount = feedbackCount,
                                    Confidence = CalculateConfidence(successRate, feedbackCount, 0.0),
                                    Hash = solutionHash
                                };
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Updated confidence query failed: {ex.Message}");
                }
                return null;
            });
        }
        private async void BtnFeedbackNo_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                var errorText = ErrorInput.Text;
                var normalizedSig = SecureEscape(Normalize(errorText));

                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                    INSERT INTO {TableFeedback} (
                        feedback_id,
                        session_id,
                        solution_hash,
                        solution_source,
                        was_helpful,
                        user_comment,
                        created_at,
                        error_signature
                    ) VALUES (
                        '{Guid.NewGuid()}',
                        '{currentSessionId}',
                        '{currentSolutionHash}',
                        '{currentSolutionSource}',
                        false,
                        '',
                        current_timestamp(),
                        '{normalizedSig}'
                    )";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        System.Diagnostics.Debug.WriteLine("Negative feedback recorded successfully");
                    }
                });

                DisableFeedbackButtons();

                // ADD THIS NEW CODE HERE - Same as BtnFeedbackYes_Click
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);

                    // Update just the first line of the result
                    var lines = ResultBox.Text.Split('\n');
                    lines[0] = newConfidenceText;
                    ResultBox.Text = string.Join("\n", lines);
                }
                // END NEW CODE

                ResultBox.Text += "\n\nThank you for the feedback. Please consider teaching the correct solution below.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Feedback failed: {ex.Message}");
            }
        }

        private void EnableFeedbackButtons()
        {
            BtnFeedbackYes.IsEnabled = true;
            BtnFeedbackNo.IsEnabled = true;
        }

        private void DisableFeedbackButtons()
        {
            BtnFeedbackYes.IsEnabled = false;
            BtnFeedbackNo.IsEnabled = false;
        }

        // Core AI and utility methods
        private static string GetOpenAIApiKey()
        {
            return Environment.GetEnvironmentVariable("OPENAI_API_KEY")
                   ?? throw new InvalidOperationException("OPENAI_API_KEY environment variable not set");
        }

        private static void EnsureTls12()
        {
            try { ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12; } catch { }
        }

        private async Task<string> AskOpenAIAsync(object[] messages, string model = "gpt-4o-mini")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            var payload = new { model, messages };
            var json = JsonConvert.SerializeObject(payload);

            using (var req = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/chat/completions"))
            {
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                req.Content = new StringContent(json, Encoding.UTF8, "application/json");

                using (var res = await Http.SendAsync(req))
                {
                    var body = await res.Content.ReadAsStringAsync();
                    if (!res.IsSuccessStatusCode)
                        throw new Exception($"OpenAI error {res.StatusCode}: {body}");

                    var obj = JObject.Parse(body);
                    var content =
                        (string)(obj["choices"]?[0]?["message"]?["content"]) ??
                        (string)(obj["choices"]?[0]?["text"]) ?? "";
                    return content;
                }
            }
        }

        private async Task<float[]> EmbedAsync(string text, string model = "text-embedding-3-large")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            var payload = new { model, input = text };
            var json = JsonConvert.SerializeObject(payload);

            using (var req = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/embeddings"))
            {
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                req.Content = new StringContent(json, Encoding.UTF8, "application/json");

                using (var res = await Http.SendAsync(req))
                {
                    var body = await res.Content.ReadAsStringAsync();
                    if (!res.IsSuccessStatusCode)
                        throw new Exception($"OpenAI embeddings error {res.StatusCode}: {body}");

                    var obj = JObject.Parse(body);
                    var arr = (JArray?)obj["data"]?[0]?["embedding"] ?? new JArray();
                    var vec = new float[arr.Count];
                    for (int i = 0; i < arr.Count; i++) vec[i] = (float)arr[i]!;
                    return vec;
                }
            }
        }

        private async Task<object[]> BuildEnhancedChatHistoryAsync(string incomingErrorMessage)
        {
            var incomingLower = (incomingErrorMessage ?? "").ToLowerInvariant();
            var esc = incomingLower.Replace("'", "''");

            var related = await Task.Run(() =>
            {
                var list = new List<(string Sig, string Steps, string Exact, double SuccessRate)>();
                using (var conn = OpenConn())
                using (var cmd = new OdbcCommand($@"
                    SELECT kb.error_signature, kb.resolution_steps, kb.error_text,
                           COALESCE(fb.success_rate, 0.5) as success_rate
                    FROM {TableKB} kb
                    LEFT JOIN (
                        SELECT solution_hash, 
                               AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                        HAVING COUNT(*) >= 3
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE lower(kb.error_signature) LIKE '%{esc}%'
                       OR lower(kb.error_text) LIKE '%{esc}%'
                    ORDER BY success_rate DESC, kb.created_at DESC
                    LIMIT 6
                ", conn))
                using (var rdr = cmd.ExecuteReader())
                {
                    while (rdr.Read())
                    {
                        var row = (
                            Sig: rdr.IsDBNull(0) ? "" : rdr.GetString(0),
                            Steps: rdr.IsDBNull(1) ? "" : (rdr.GetValue(1)?.ToString() ?? ""),
                            Exact: rdr.IsDBNull(2) ? "" : rdr.GetString(2),
                            SuccessRate: rdr.IsDBNull(3) ? 0.5 : rdr.GetDouble(3)
                        );
                        list.Add(row);
                    }
                }
                return list;
            });

            var messages = new List<object>
            {
                new { role = "system", content = "You are CAINE, an expert database debugging assistant with access to proven solutions. Return concise, actionable steps in well-formatted Markdown with headers, bullet points, and code blocks when appropriate." }
            };

            foreach (var r in related)
            {
                var confidence = r.SuccessRate > FeedbackBoostThreshold ? " (Proven high success rate)" : "";
                messages.Add(new
                {
                    role = "user",
                    content = $"Previously solved error:\n'{r.Sig}'{confidence}\n\nProven solution:\n{r.Steps}"
                });
            }

            messages.Add(new
            {
                role = "user",
                content = "Based on the proven solutions above, help me debug this new error. Provide step-by-step instructions in Markdown format:\n\n" + incomingErrorMessage
            });

            return messages.ToArray();
        }

        private static IEnumerable<string> ExtractNumberedLines(string markdown)
        {
            if (markdown == null) yield break;
            foreach (var raw in markdown.Split('\n'))
            {
                var t = raw.Trim();
                if (t.StartsWith("1.") || t.StartsWith("2.") || t.StartsWith("3.") || t.StartsWith("- "))
                    yield return t.TrimStart('-', ' ').Trim();
            }
        }

        private async Task<(bool HasConflicts, double ExistingSuccessRate)> CheckForExistingSolutions(string hash, string[] newSteps)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            SELECT AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                            FROM {TableFeedback} f
                            JOIN {TableKB} kb ON f.solution_hash = kb.error_hash
                            WHERE kb.error_hash = ? OR kb.error_signature = ?";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("hash", hash);
                            cmd.Parameters.AddWithValue("sig", Normalize(string.Join(" ", newSteps)));

                            var result = cmd.ExecuteScalar();
                            if (result != null && result != DBNull.Value)
                            {
                                var successRate = Convert.ToDouble(result);
                                return (successRate < 0.6, successRate);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Conflict check failed: {ex.Message}");
                }
                return (false, 0.5);
            });
        }

        private async Task TeachEnhancedSolution(string hash, string sig, string errorText, string[] stepsLines, float[] emb, (bool HasConflicts, double ExistingSuccessRate) conflicts)
        {
            await Task.Run(() =>
            {
                try
                {
                    string Esc(string s) => (s ?? "")
                        .Replace("\\", "\\\\")
                        .Replace("'", "''")
                        .Replace("\r", "")
                        .Replace("\n", "\\n");

                    string ArrayLiteral(string[] arr) =>
                        "array(" + string.Join(",", arr.Select(s => $"'{Esc(s)}'")) + ")";

                    var steps = ArrayLiteral(stepsLines);
                    var escErr = Esc(errorText);
                    var escSig = Esc(sig);

                    string embSql = emb == null
                        ? "NULL"
                        : $"array({string.Join(",", emb.Select(f => f.ToString("G", CultureInfo.InvariantCulture)))})";

                    var changeType = conflicts.HasConflicts ? "alternative" : "create";

                    using (var conn = OpenConn())
                    {
                        // Insert into main KB
                        var kbSql = $@"
                            MERGE INTO {TableKB} t
                            USING (SELECT
                                     '{Guid.NewGuid()}' AS id,
                                     current_timestamp() AS created_at,
                                     'human_enhanced' AS source,
                                     NULL AS product, NULL AS version, NULL AS env,
                                     '{escErr}' AS error_text,
                                     '{escSig}' AS error_signature,
                                     '{hash}' AS error_hash,
                                     NULL AS context_text,
                                     {steps} AS resolution_steps,
                                     array() AS symptoms,
                                     NULL AS root_cause,
                                     NULL AS verification,
                                     array() AS links,
                                     NULL AS code_before, NULL AS code_after, 
                                     'Enhanced teaching with conflict detection' AS notes,
                                     {embSql} AS embedding
                                   ) s
                            ON t.error_hash = s.error_hash
                            WHEN MATCHED THEN UPDATE SET
                              t.resolution_steps = array_distinct(
                                concat(coalesce(t.resolution_steps, cast(array() as array<string>)),
                                       s.resolution_steps)),
                              t.notes = 'Updated with enhanced teaching'
                            WHEN NOT MATCHED THEN INSERT *";

                        using (var cmd = new OdbcCommand(kbSql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        // Insert into versions table for tracking
                        var versionSql = $@"
                            INSERT INTO {TableKBVersions} VALUES (
                                '{Guid.NewGuid()}', '{Guid.NewGuid()}', '{hash}', {steps},
                                current_timestamp(), '{Environment.UserName}', '{changeType}',
                                NULL, 0.8, 0, true
                            )";

                        using (var versionCmd = new OdbcCommand(versionSql, conn))
                        {
                            versionCmd.ExecuteNonQuery();
                        }
                    }
                }
                catch (Exception ex)
                {
                    throw new Exception($"Enhanced teaching failed: {ex.Message}", ex);
                }
            });
        }

        // Core utility methods
        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "";
            var t = s.ToLowerInvariant();

            t = Regex.Replace(t, @"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<guid>");
            t = Regex.Replace(t, @"[a-z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n]+", "<path>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]\.\[[^\]]+\]", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+\.[a-z0-9_]+", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+", "exec <proc>");
            t = Regex.Replace(t, @"login failed for user\s+'[^']+'", "login failed for user '<user>'");
            t = Regex.Replace(t, @"\b[a-z0-9._-]+\\[a-z0-9.$_-]+\b", "<user>");
            t = Regex.Replace(t, @"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", "<user>");
            t = Regex.Replace(t, @"[^\w\s-]", " ");
            t = Regex.Replace(t, @"\b\d+\b", "<num>");
            t = Regex.Replace(t, @"\s+", " ").Trim();

            return t;
        }

        private static string Sha256Hex(string s)
        {
            using (var sha = SHA256.Create())
            {
                var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(s ?? ""));
                var sb = new StringBuilder(bytes.Length * 2);
                foreach (var b in bytes) sb.Append(b.ToString("x2"));
                return sb.ToString();
            }
        }

        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        private static string[] Tokens(string sig)
        {
            var toks = Regex.Matches(sig ?? "", @"[a-z0-9\-]{4,}")
                            .Cast<Match>().Select(m => m.Value)
                            .Where(v => !Stop.Contains(v))
                            .Distinct()
                            .Take(LikeTokenMax)
                            .ToArray();
            return toks;
        }

        private static float[] ParseFloatArray(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return Array.Empty<float>();
            var t = s.Trim();

            if (t.StartsWith("[")) t = t.Substring(1);
            if (t.EndsWith("]")) t = t.Substring(0, t.Length - 1);
            if (t.StartsWith("array(", StringComparison.OrdinalIgnoreCase)) t = t.Substring(6);
            if (t.EndsWith(")")) t = t.Substring(0, t.Length - 1);

            var parts = t.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var list = new List<float>(parts.Length);
            foreach (var p in parts)
            {
                if (float.TryParse(p, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    list.Add(v);
            }
            return list.ToArray();
        }

        private static void NormalizeInPlace(float[] v)
        {
            if (v == null || v.Length == 0) return;

            double sumsq = 0;
            for (int i = 0; i < v.Length; i++) sumsq += v[i] * v[i];
            var norm = Math.Sqrt(sumsq);

            if (norm <= 1e-8) return;

            for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
        }

        private static double Cosine(float[] a, float[] b)
        {
            if (a == null || b == null) return -1;

            int n = Math.Min(a.Length, b.Length);
            if (n == 0) return -1;

            double dot = 0;
            for (int i = 0; i < n; i++) dot += a[i] * b[i];
            return dot;
        }
    }
}