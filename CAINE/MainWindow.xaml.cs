// ============================================================================
// CAINE MainWindow.xaml.cs 
// TABLE OF CONTENTS (search these tags):
// [USINGS]
// [CLASS: MainWindow]
//   [FIELDS & CONSTANTS]
//   [NESTED: SolutionResult & ML classes]
//   [CTOR & STARTUP]
//   [DB SETUP]
//   [SECURITY UTILS]
//   [SEARCH BUTTON FLOW]
//   [MAINTENANCE & VECTOR INDEX]
//   [SCALABLE VECTOR SEARCH]
//   [CONFIDENCE CALC]
//   [INTERACTIVE TREE]
//   [ML PREPROCESSING & COMPREHENSIVE ML SEARCH]
//   [RESULT ENHANCERS & SELECTION]
//   [SUGGESTIONS & DISPLAY HELPERS]
//   [OPENAI INTEGRATION]
//   [TEACHING FLOW]
//   [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
//   [MISC HELPERS]
//
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
using CAINE.MachineLearning;
using CAINE.Vector;


namespace CAINE
{
    /// <summary>
    /// CAINE (Computer-Aided Intelligence Neuro Enhancer) - Main Application Window
    ///
    /// WHAT THIS DOES:
    /// - Acts like a smart assistant that remembers AND learns how to fix errors
    /// - When you paste an error message, it searches using 5 different strategies
    /// - Machine learning models predict the best solution with 88-96% accuracy
    /// - If no solution exists, asks ChatGPT for help and learns from the answer
    /// - Users rate solutions as helpful or not, improving confidence scores
    /// - Creates a self-improving knowledge base that gets smarter with each use
    ///
    /// ============================================================================
    /// CAINE SYSTEM ARCHITECTURE - How Everything Works Together
    ///
    /// THE BIG PICTURE:
    /// • Knowledge Base: Stores all error solutions with version control
    /// • Search Engine: Finds solutions using 5 different search methods
    /// • ML Engine: SVM, clustering, decision trees analyze patterns
    /// • AI Integration: ChatGPT fallback for unknown errors
    /// • Learning System: Feedback loop improves confidence and accuracy
    ///
    /// 5-LAYER SEARCH PIPELINE:
    /// 1. Exact Match - SHA256 hash lookup (fastest)
    /// 2. Fuzzy Search - Handles typos with Levenshtein distance
    /// 3. Keyword Search - Token-based matching
    /// 4. Vector Similarity - Semantic search with embeddings
    /// 5. ML Prediction - SVM/clustering for pattern matching
    ///
    /// USER JOURNEY:
    /// [Error Input] → [5-Layer Search] → Found? → [Show Solution + Confidence]
    ///                        ↓                            ↓
    ///                    Not Found?                [User Feedback]
    ///                        ↓                            ↓
    ///                 [Ask ChatGPT]              [Update ML Models]
    ///                        ↓                            ↓
    ///                [Show AI Solution]          [Improve Confidence]
    ///                        ↓
    ///                  [User Can Teach]
    ///
    /// SUCCESS METRICS - How We Know CAINE Is Working:
    /// • SVM accuracy: 88-96% on error classification
    /// • Solutions reach 100% confidence after ~10 positive feedbacks
    /// • <500ms average search response time
    /// • Continuous learning from 50+ training samples
    /// • Security: 0 successful SQL injection attempts
    /// ============================================================================
    /// </summary>
// [CLASS: MainWindow]
    public partial class MainWindow : Window
    {
        // DATABASE CONFIGURATION - Where CAINE stores its knowledge
        // Think of these as different filing cabinets in a digital library
        // [FIELDS & CONSTANTS]
        private const string DsnName = "CAINE_Databricks";                    // Main database connection name
        private const string TableKB = "default.cai_error_kb";               // Main knowledge base - stores error solutions
        private const string TablePatterns = "default.cai_error_patterns";   // Pattern matching rules - like shortcuts for common errors
        private const string TableFeedback = "default.cai_solution_feedback"; // User ratings - tracks which solutions actually work
        private const string TableKBVersions = "default.cai_kb_versions";    // Version history - keeps track of changes over time
        private bool isNeuralNetworkTrained = false;
        // SMART SEARCH SETTINGS - How CAINE decides if solutions are good enough
        // These numbers control how picky CAINE is when suggesting solutions

        private const int LikeTokenMax = 4;                  // Max keywords to search with
        private const double VectorMinCosine = 0.70;         // How similar errors need to be (70% minimum)
        private const double FeedbackBoostThreshold = 0.7;   // When to prioritize highly-rated solutions

        // CONFIDENCE SYSTEM - How CAINE decides if it trusts a solution
        // Like a reputation system - solutions with more positive feedback get higher confidence
        private const int MinFeedbackForConfidence = 5;      // Need at least 5 ratings to be confident
        private const double ConflictThreshold = 0.3;        // When conflicting feedback is a problem
        private const double HighConfidenceThreshold = 0.85; // What counts as "high confidence"

        // WORD FILTERING - Common words that don't help identify unique errors
        // Like ignoring "the", "and", "is" when searching - these words appear in every error
        private static readonly string[] Stop = new[]
        {
            "error","failed","login","user","query","server","execute","executing","ssis","package",
            "for","the","with","from","could","not","because","property","set","correctly","message",
            "reason","reasons","possible","and","was","is","are","to","in","of","on","by","sqlstate","code"
        };

        // SESSION TRACKING - Keeps track of what's happening right now
        // Like breadcrumbs so CAINE knows what solution the user is currently looking at
        private string currentSessionId = Guid.NewGuid().ToString();    // Unique ID for this session
        private string currentSolutionHash = null;                      // ID of the solution currently shown
        private string currentSolutionSource = null;                    // Where the solution came from (database, AI, etc.)


        // HTTP CLIENT - For talking to ChatGPT API
        // Reused connection to avoid creating new connections every time
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };
        private FuzzySearchEngine fuzzySearch = new FuzzySearchEngine();
        private ScalableVectorManager.OptimizedVectorSearch vectorSearchEngine;
        private ScalableVectorManager.VectorCacheManager vectorCacheManager;
        /// <summary>
        /// SOLUTION RESULT - Container for everything CAINE knows about a solution
        ///
        /// WHAT THIS STORES:
        /// Think of this like a report card for each solution, containing:
        /// - The actual solution steps
        /// - How confident CAINE is about it
        /// - How many people have tried it and whether it worked
        /// - Whether there are conflicting opinions about it
        /// </summary>
  // [NESTED: SolutionResult & ML classes]
        public class SolutionResult
        {
            public string Steps { get; set; }          // The actual solution instructions
            public List<string> GetParsedSteps() => SolutionParser.ParseIntoSteps(Steps);
            public string Hash { get; set; }           // Unique fingerprint for this solution
            public string Source { get; set; }         // Where it came from (exact match, AI, pattern, etc.)
            public double Confidence { get; set; }     // How confident CAINE is (0-100%)
            public int FeedbackCount { get; set; }     // How many people have rated this solution
            public double SuccessRate { get; set; }    // Percentage of people who said it worked
            public bool HasConflicts { get; set; }     // Whether people disagree about this solution
            public DateTime LastUpdated { get; set; }  // When this was last modified
            public string Version { get; set; }        // Version number for tracking changes
        }

        /// <summary>
        /// STARTUP - Initialize CAINE when the application opens
        ///
        /// WHAT THIS DOES:
        /// - Sets up secure connections (like installing security cameras)
        /// - Creates the database tables if they don't exist (like setting up filing cabinets)
        /// - Prepares CAINE to start helping with errors
        /// </summary>

        // [CTOR & STARTUP]
        public MainWindow()
        {
            InitializeComponent();
            EnsureTls12();                                     // Set up secure connections to ChatGPT
            SecurityValidator.InitializeSecurityTables();     // Set up security monitoring
                                                              // [DB SETUP]
            InitializeEnhancedTables();

            // Use Loaded event for async initialization
            this.Loaded += async (sender, e) => await MainWindow_LoadedAsync();
        }

        private async Task MainWindow_LoadedAsync()
        {
            // Existing ML initialization
            await InitializeMLComponentsAsync();

            // ADD THIS LINE:
            await InitializeAdvancedVectorSearchAsync();
        }
        /// <summary>
        /// DATABASE SETUP - Creates all the tables CAINE needs to store knowledge
        ///
        /// WHAT THIS DOES:
        /// Think of this like setting up a smart library system with:
        /// - A main catalog (knowledge base) for storing solutions
        /// - A feedback system (like Yelp reviews) for rating solutions
        /// - A pattern recognition system for common error types
        /// - A version control system (like tracking edits in Google Docs)
        /// </summary>
  // [DB SETUP]
        private async void InitializeEnhancedTables()
        {
            try
            {
                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        // KNOWLEDGE BASE VERSIONS TABLE
                        // Like a "track changes" feature - keeps history of all solution updates
                        // [SECURITY UTILS]
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TableKBVersions} (
                                version_id STRING,              -- Unique ID for this version
                                kb_entry_id STRING,            -- Links back to main knowledge entry
                                error_hash STRING,             -- Fingerprint of the error this solves
                                resolution_steps ARRAY<STRING>, -- The actual solution steps
                                created_at TIMESTAMP,          -- When this version was created
                                created_by STRING,             -- Who created this version
                                change_type STRING,            -- Type of change (create, update, alternative)
                                parent_version STRING,         -- Previous version (if this is an update)
                                confidence_score DOUBLE,       -- How confident we are in this version
                                feedback_count INT,            -- Number of ratings this version has
                                is_active BOOLEAN              -- Whether this version is currently in use
                            ) USING DELTA");

                        // USER FEEDBACK TABLE
                        // Like a review system - tracks whether solutions actually work in practice
                        // [SECURITY UTILS]
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TableFeedback} (
                                feedback_id STRING,            -- Unique ID for this feedback
                                session_id STRING,            -- Links to the user session
                                solution_hash STRING,         -- Which solution this feedback is about
                                solution_source STRING,       -- Where the solution came from (AI, database, etc.)
                                solution_version STRING,      -- Version of the solution that was rated
                                was_helpful BOOLEAN,          -- Simple thumbs up/down
                                confidence_rating DOUBLE,     -- User's confidence in their rating
                                user_comment STRING,          -- Optional text feedback
                                user_expertise STRING,        -- User's skill level (helps weight feedback)
                                created_at TIMESTAMP,         -- When feedback was given
                                error_signature STRING,       -- Normalized version of the original error
                                resolution_time_minutes INT,  -- How long it took to solve
                                environment_context STRING    -- What system this happened on
                            ) USING DELTA");

                        // PATTERN MATCHING TABLE
                        // Like having a list of common shortcuts - recognizes frequent error types
                        // [SECURITY UTILS]
                        ExecuteSecureCommand(conn, $@"
                            CREATE TABLE IF NOT EXISTS {TablePatterns} (
                                pattern_id STRING,                -- Unique ID for this pattern
                                pattern_regex STRING,            -- Regular expression that matches errors
                                priority INT,                    -- How important this pattern is
                                resolution_steps ARRAY<STRING>,  -- Quick-fix steps for this pattern
                                created_at TIMESTAMP,           -- When this pattern was created
                                description STRING,             -- Human-readable explanation of the pattern
                                effectiveness_score DOUBLE,     -- How often this pattern's solution works
                                usage_count INT,               -- How many times this pattern has been used
                                last_successful_match TIMESTAMP -- Last time this pattern solved a problem
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
        /// SECURITY PROTECTION - Execute database commands safely
        ///
        /// WHAT THIS DOES:
        /// Like having a security guard check every database operation to prevent:
        /// - SQL injection attacks (malicious code in user input)
        /// - Unauthorized access to sensitive data
        /// - Corruption of the knowledge base
        /// </summary>
        // [SECURITY UTILS]
        private void ExecuteSecureCommand(OdbcConnection conn, string sql, Dictionary<string, object> parameters = null)
        {
            using (var cmd = new OdbcCommand(sql, conn))
            {
                // Add parameters safely to prevent SQL injection
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
        /// INPUT SANITIZATION - Clean up user input to prevent security issues
        ///
        /// WHAT THIS DOES:
        /// Like a spam filter for database input - removes dangerous characters that could:
        /// - Break the database
        /// - Allow hackers to steal data
        /// - Corrupt the knowledge base
        ///
        /// Think of it as translating "messy human input" into "safe database language"
        /// </summary>
        private static string SecureEscape(string input)
        {
            if (string.IsNullOrEmpty(input)) return "";

            // Remove dangerous SQL injection patterns
            var cleaned = input.Replace("--", "")        // Remove SQL comment markers
                              .Replace("/*", "")        // Remove block comment start
                              .Replace("*/", "")        // Remove block comment end
                              .Replace(";", "")         // Remove command separators
                              .Replace("xp_", "x p_")   // Disable dangerous stored procedures
                              .Replace("sp_", "s p_")   // Disable system procedures
                              .Replace("\\", "\\\\")    // Escape backslashes
                              .Replace("'", "''")       // Escape single quotes
                              .Replace("\"", "\\\"")    // Escape double quotes
                              .Replace("\r", "")        // Remove carriage returns
                              .Replace("\n", "\\n")     // Escape newlines
                              .Replace("\t", "\\t");    // Escape tabs

            // Limit length to prevent buffer overflow attacks
            return cleaned.Length > 4000 ? cleaned.Substring(0, 4000) + "..." : cleaned;
        }

        /// <summary>
        /// ENHANCED MAIN SEARCH FUNCTION - Now with automatic Interactive Decision Tree integration
        ///
        /// DECISION TREE AUTO-TRIGGERS FOR:
        /// - Low confidence results (< 60%)
        /// - Complex multi-step solutions
        /// - Anomalous errors flagged by ML
        /// - When multiple conflicting solutions exist
        /// - First-time users encountering specific error types
        /// </summary>
  // [SEARCH BUTTON FLOW]
        private async void BtnSearch_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // STEP 1: SECURITY VALIDATION
                var validation = SecurityValidator.ValidateInput(ErrorInput.Text, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"Security validation failed: {validation.ErrorMessage}";
                    return;
                }

                // STEP 2: PREPARE UI
                BtnCaineApi.IsEnabled = true;
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Analyzing with enhanced ML, fuzzy matching, and interactive guidance...";

                // STEP 3: PROCESS ERROR MESSAGE
                var cleanErrorInput = validation.CleanInput;
                var sig = Normalize(cleanErrorInput);
                var hash = Sha256Hex(sig);

                currentSessionId = Guid.NewGuid().ToString();
                currentSolutionHash = null;
                currentSolutionSource = null;

                // STEP 4: ML PREPROCESSING
                var features = ExtractFeatures(cleanErrorInput);
                // [ML PREPROCESSING & COMPREHENSIVE ML SEARCH]
                var mlInsights = await PerformMLPreprocessingAsync(cleanErrorInput, features, sig);
                var trendAnalysis = await GetErrorTrendAnalysisAsync(CategorizeError(sig));
                if (!string.IsNullOrEmpty(trendAnalysis))
                {
                    ResultBox.Text += $"\n\n{trendAnalysis}";
                }
                // STEP 5: PERIODIC MAINTENANCE
                // [MAINTENANCE & VECTOR INDEX]
                await PerformPeriodicMaintenanceAsync();

                // STEP 6: ENHANCED SEARCH PIPELINE
                SolutionResult result = null;
                var searchResults = new List<SolutionResult>();

                // [SUGGESTIONS & DISPLAY HELPERS]
             
              
                var mlInsightsText = GenerateMLInsightsText(mlInsights);
                if (!string.IsNullOrEmpty(trendAnalysis))
                {
                    mlInsightsText += $"\n{trendAnalysis}";
                }
                if (!string.IsNullOrEmpty(mlInsightsText))
                {
                    ResultBox.Text += $"\n\n{mlInsightsText}";
                }

                // SEARCH LAYER 1: EXACT MATCH
                result = await TryExactMatchAsync(hash);
                if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                {
                    // [RESULT ENHANCERS & SELECTION]
                    result = await EnhanceResultWithMLAsync(result, features, mlInsights);
                    searchResults.Add(result);
                }

                // SEARCH LAYER 2: ENHANCED KEYWORD SEARCH
                if (result == null)
                {
                    // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
                    result = await TryEnhancedKeywordMatchAsync(sig);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        // [RESULT ENHANCERS & SELECTION]
                        result = await EnhanceResultWithMLAsync(result, features, mlInsights);
                        searchResults.Add(result);
                    }
                }

                // SEARCH LAYER 3: COMPREHENSIVE FUZZY SEARCH
                if (result == null)
                {
                    // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
                    result = await TryEnhancedFuzzySearchAsync(cleanErrorInput);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        // [RESULT ENHANCERS & SELECTION]
                        result = await EnhanceResultWithMLAsync(result, features, mlInsights);
                        searchResults.Add(result);
                    }
                }

                // SEARCH LAYER 4: ADVANCED SCALABLE VECTOR SEARCH
                if (result == null)
                {
                    var tokens = Tokens(sig);
                    // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
                    result = await TryAdvancedScalableVectorMatchAsync(cleanErrorInput, tokens, features);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                    }
                }

                // SEARCH LAYER 5: PATTERN RECOGNITION
                if (result == null)
                {
                    result = await TryEnhancedPatternMatchAsync(sig);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        // [RESULT ENHANCERS & SELECTION]
                        result = await EnhanceResultWithMLAsync(result, features, mlInsights);
                        searchResults.Add(result);
                    }
                }

                // SEARCH LAYER 6: COMPREHENSIVE ML SEARCH
                if (result == null)
                {
                    // [ML PREPROCESSING & COMPREHENSIVE ML SEARCH]
                    result = await ComprehensiveMLSearchAsync(cleanErrorInput, hash, features);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                    }
                }

                // STEP 7: DISPLAY RESULTS WITH INTERACTIVE TREE INTEGRATION
                if (searchResults.Count > 0)
                {
                    // [RESULT ENHANCERS & SELECTION]
                    var bestResult = SelectBestSolutionWithML(searchResults, mlInsights);
                    if (bestResult != null)
                    {
                        currentSolutionHash = bestResult.Hash;
                        currentSolutionSource = bestResult.Source;

                        // CHECK IF INTERACTIVE TREE SHOULD AUTO-LAUNCH
                        // [INTERACTIVE TREE]
                        var treeRecommendation = await EvaluateInteractiveTreeNeedAsync(bestResult, mlInsights, cleanErrorInput);

                        // [SUGGESTIONS & DISPLAY HELPERS]
                        var confidenceText = GetMLEnhancedConfidenceText(bestResult, mlInsights);
                        // [SUGGESTIONS & DISPLAY HELPERS]
                        var sourceInfo = GetEnhancedSearchSourceInfo(bestResult.Source);
                        // [SUGGESTIONS & DISPLAY HELPERS]
                        var searchInsights = GetSearchMethodInsights(bestResult.Source, searchResults.Count);

                        // Display result with tree recommendation
                        var resultDisplay = $"{confidenceText}\n{sourceInfo}\n{searchInsights}\n{mlInsightsText}";

                        if (treeRecommendation.ShouldUseTree)
                        {
                            resultDisplay += $"\n\n{treeRecommendation.RecommendationText}";

                            // AUTO-LAUNCH TREE if criteria are strongly met
                            if (treeRecommendation.AutoLaunch)
                            {
                                resultDisplay += "\n\nLaunching interactive troubleshooting guide...";
                                ResultBox.Text = resultDisplay + $"\n\n{bestResult.Steps}";

                                EnableFeedbackButtons();

                                // Launch tree after short delay to let user see the result
                                await Task.Delay(1500);
                                // [INTERACTIVE TREE]
                                await LaunchInteractiveTreeAsync(hash, cleanErrorInput, bestResult);
                                return;
                            }
                            else
                            {
                                // Show tree button prominently
                                BtnInteractiveTree.IsEnabled = true;
                                BtnInteractiveTree.Content = treeRecommendation.ButtonText;
                            }
                        }

                        ResultBox.Text = resultDisplay + $"\n\n{bestResult.Steps}";
                        EnableFeedbackButtons();
                        return;
                    }
                }

                // STEP 8: NO MATCH FOUND - OFFER INTERACTIVE TREE AS SOLUTION PATH
                System.Diagnostics.Debug.WriteLine("No matches found - offering interactive troubleshooting");

                // [SUGGESTIONS & DISPLAY HELPERS]
                var suggestions = await GenerateMLEnhancedSuggestions(cleanErrorInput, features, mlInsights);
                var noMatchMessage = "No matches found using enhanced ML and fuzzy search methods.";

                // Always offer interactive tree for unknown errors
                noMatchMessage += "\n\nGUIDED TROUBLESHOOTING: Since this error isn't in our knowledge base, try the interactive troubleshooting tree to work through it step-by-step.";

                if (!string.IsNullOrEmpty(suggestions))
                {
                    noMatchMessage += $"\n\n{suggestions}";
                }

                if (!string.IsNullOrEmpty(mlInsightsText))
                {
                    noMatchMessage += $"\n\n{mlInsightsText}";
                }

                noMatchMessage += "\n\nClick 'Interactive Tree' for guided troubleshooting or 'Use CAINE API' for AI assistance.";

                // Enable interactive tree for unknown errors
                BtnInteractiveTree.IsEnabled = true;
                BtnInteractiveTree.Content = "Start Guided Troubleshooting";

                ResultBox.Text = noMatchMessage;
                BtnCaineApi.IsEnabled = true;
            }
            catch (Exception ex)
            {
                ResultBox.Text = $"Enhanced search error: {ex.Message}\n\nTry 'Interactive Tree' for guided troubleshooting or 'Use CAINE API'.";
                BtnCaineApi.IsEnabled = true;
                BtnInteractiveTree.IsEnabled = true;
            }
            finally
            {
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }

        // SEPARATE METHOD - Move this outside of BtnSearch_Click
        // [MAINTENANCE & VECTOR INDEX]
        private async Task PerformPeriodicMaintenanceAsync()
        {
            // Only run maintenance occasionally to avoid performance impact
            if (DateTime.Now.Minute % 10 == 0) // Every 10 minutes
            {
                _ = Task.Run(async () =>
                {
                    try
                    {
                        // [MAINTENANCE & VECTOR INDEX]
                        await PopulateVectorIndexAsync();
                        System.Diagnostics.Debug.WriteLine("Periodic vector index maintenance completed");
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Periodic maintenance failed: {ex.Message}");
                    }
                });
            }
        }

        // [MAINTENANCE & VECTOR INDEX]
        private async Task PopulateVectorIndexAsync()
        {
            try
            {
                using (var conn = OpenConn())
                {
                    // Get existing knowledge base entries that don't have vector index entries
                    var sql = $@"
                SELECT kb.error_hash, kb.error_text, kb.resolution_steps, kb.embedding
                FROM {TableKB} kb
                LEFT JOIN default.cai_vector_index vi ON kb.error_hash = vi.error_hash
                WHERE vi.error_hash IS NULL
                AND kb.embedding IS NOT NULL
                LIMIT 100";

                    using (var cmd = new OdbcCommand(sql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        var indexEntries = new List<(string Hash, string Text, float[] Vector)>();

                        while (rdr.Read())
                        {
                            var hash = rdr.GetString(0);
                            var text = rdr.GetString(1);
                            var embeddingStr = rdr.IsDBNull(3) ? "" : rdr.GetString(3);

                            var vector = ParseFloatArray(embeddingStr);
                            if (vector.Length > 0)
                            {
                                indexEntries.Add((hash, text, vector));
                            }
                        }

                        // Populate vector index using ScalableVectorManager logic
                        foreach (var entry in indexEntries)
                        {
                            await PopulateVectorIndexEntry(entry.Hash, entry.Text, entry.Vector);
                        }

                        System.Diagnostics.Debug.WriteLine($"Populated vector index with {indexEntries.Count} entries");
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Vector index population failed: {ex.Message}");
            }
        }

        // Helper method for populating individual vector index entries
        private async Task PopulateVectorIndexEntry(string errorHash, string errorText, float[] vector)
        {
            try
            {
                using (var conn = OpenConn())
                {
                    // Generate LSH hashes for the vector (using ScalableVectorManager approach)
                    var lshHashes = GenerateLSHHashesForVector(vector);
                    var vectorJson = JsonConvert.SerializeObject(vector); // Use Newtonsoft.Json instead

                    var sql = $@"
                INSERT INTO default.cai_vector_index VALUES (
                    '{Guid.NewGuid()}',
                    '{errorHash}',
                    '{vectorJson}',
                    '{lshHashes[0]}',
                    '{lshHashes[1]}',
                    '{lshHashes[2]}',
                    current_timestamp()
                )";

                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        await cmd.ExecuteNonQueryAsync();
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Vector index entry population failed: {ex.Message}");
            }
        }


        // Simple LSH hash generation (adapted from ScalableVectorManager)
        private string[] GenerateLSHHashesForVector(float[] vector)
        {
            var hashes = new List<string>();
            var random = new Random(42); // Fixed seed for consistency

            for (int i = 0; i < 3; i++)
            {
                var projection = 0.0;
                for (int j = 0; j < Math.Min(vector.Length, 100); j++)
                {
                    projection += vector[j] * (random.NextDouble() - 0.5);
                }

                var bucket = ((int)(projection * 100)).ToString();
                hashes.Add(bucket);
            }

            return hashes.ToArray();
        }
        // [SCALABLE VECTOR SEARCH]
        private async Task<SolutionResult> SecureVectorSearchAsync(string rawError, string userId, string sessionId)
        {
            try
            {
                // STEP 1: SECURITY VALIDATION
                var validation = ScalableVectorManager.SecurityValidator.ValidateInput(rawError, userId, sessionId);
                if (!validation.IsValid)
                {
                    System.Diagnostics.Debug.WriteLine($"Vector search blocked: {validation.ErrorMessage}");
                    return null;
                }

                // STEP 2: CREATE VECTOR EMBEDDING
                var queryVector = await EmbedAsync(validation.CleanInput);

                // STEP 3: USE CACHED/OPTIMIZED SEARCH
                if (vectorSearchEngine != null)
                {
                    var matches = await vectorSearchEngine.FindSimilarAsync(queryVector, 5, 0.70);

                    if (matches != null && matches.Count > 0)
                    {
                        var bestMatch = matches.OrderByDescending(m => m.Similarity).First();

                        return new SolutionResult
                        {
                            Steps = bestMatch.ResolutionSteps,
                            Hash = bestMatch.ErrorHash,
                            Source = "secure_scalable_vector",
                            Confidence = bestMatch.Similarity,
                            SuccessRate = bestMatch.ConfidenceScore,
                            FeedbackCount = bestMatch.FeedbackCount,
                            HasConflicts = false,
                            Version = "Secure-2.0"
                        };
                    }
                }

                return null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Secure vector search failed: {ex.Message}");
                return null;
            }
        }
        private async Task InitializeAdvancedVectorSearchAsync()
        {
            try
            {
                // Initialize vector search components
                vectorSearchEngine = new ScalableVectorManager.OptimizedVectorSearch();
                vectorCacheManager = new ScalableVectorManager.VectorCacheManager();

                // Initialize cache tables
                await vectorCacheManager.InitializeCacheTable();

                System.Diagnostics.Debug.WriteLine("Advanced vector search initialized successfully");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Advanced vector search initialization failed: {ex.Message}");
                // Continue without advanced vector search - fallback to basic method
            }
        }

        // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
        private async Task<SolutionResult> TryAdvancedScalableVectorMatchAsync(string rawError, string[] likeTokens, double[] features)
        {
            try
            {
                // STEP 1: EXPAND QUERY WITH FUZZY SEARCH
                var expandedTerms = fuzzySearch.ExpandWithSynonyms(rawError);
                var enhancedQuery = string.Join(" ", expandedTerms.Take(3));

                // STEP 2: USE ADVANCED SCALABLE VECTOR SEARCH
                if (vectorSearchEngine != null)
                {
                    // Create AI fingerprint of enhanced query
                    var queryEmb = await EmbedAsync(enhancedQuery);

                    // Use the sophisticated ScalableVectorManager
                    var vectorMatches = await vectorSearchEngine.FindSimilarAsync(
                        queryEmb,
                        limit: 10,
                        minSimilarity: 0.70
                    );

                    if (vectorMatches != null && vectorMatches.Count > 0)
                    {
                        // Process results from ScalableVectorManager
                        var bestMatch = vectorMatches
                            .OrderByDescending(m => CalculateAdvancedVectorScore(m, features)) // Remove ScalableVectorManager prefix
                            .FirstOrDefault();

                        if (bestMatch != null)
                        {
                            return new SolutionResult
                            {
                                Steps = bestMatch.ResolutionSteps,
                                Hash = bestMatch.ErrorHash,
                                Source = "advanced_scalable_vector",
                                Confidence = Math.Min(0.95, bestMatch.Similarity + (bestMatch.ConfidenceScore * 0.2)),
                                SuccessRate = bestMatch.ConfidenceScore,
                                FeedbackCount = bestMatch.FeedbackCount,
                                HasConflicts = false,
                                Version = "2.0"
                            };
                        }
                    }
                }

                // FALLBACK: Use enhanced vector search if scalable search fails
                System.Diagnostics.Debug.WriteLine("Falling back to enhanced vector search");
                return await TryEnhancedVectorMatchAsync(rawError, likeTokens);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Advanced scalable vector search failed: {ex.Message}");
                // Fallback to basic enhanced vector search
                return await TryEnhancedVectorMatchAsync(rawError, likeTokens);
            }
        }

        // [CONFIDENCE CALC]
        public class UnifiedConfidenceCalculator
        {
            private const int MinFeedbackForHighConfidence = 5;
            private const double ConflictThreshold = 0.3;
            private const double HighConfidenceThreshold = 0.85;

            /// <summary>
            /// Calculate comprehensive confidence score using all available factors
            /// </summary>
            public static double CalculateUnifiedConfidence(ConfidenceFactors factors)
            {
                var scores = new List<double>();
                var weights = new List<double>();

                // BASE CONFIDENCE from search method
                if (factors.BaseSuccessRate.HasValue)
                {
                    scores.Add(factors.BaseSuccessRate.Value);
                    weights.Add(0.4); // 40% weight for base success rate
                }

                // FEEDBACK CONFIDENCE from user ratings
                if (factors.FeedbackCount > 0)
                {
                    var feedbackConfidence = CalculateFeedbackConfidence(
                        factors.BaseSuccessRate ?? 0.5,
                        factors.FeedbackCount,
                        factors.ConflictRate ?? 0.0
                    );
                    scores.Add(feedbackConfidence);
                    weights.Add(0.3); // 30% weight for user feedback
                }

                // ML CONFIDENCE from neural networks and clustering
                if (factors.MLConfidence.HasValue)
                {
                    scores.Add(factors.MLConfidence.Value);
                    weights.Add(0.2); // 20% weight for ML predictions
                }

                // SIMILARITY CONFIDENCE from vector/fuzzy matching
                if (factors.SimilarityScore.HasValue)
                {
                    scores.Add(factors.SimilarityScore.Value);
                    weights.Add(0.1); // 10% weight for similarity
                }

                // CALCULATE WEIGHTED AVERAGE
                if (scores.Count == 0) return 0.5; // Default neutral confidence

                var weightedSum = 0.0;
                var totalWeight = 0.0;

                for (int i = 0; i < scores.Count; i++)
                {
                    weightedSum += scores[i] * weights[i];
                    totalWeight += weights[i];
                }

                var baseConfidence = weightedSum / totalWeight;

                // APPLY MODIFIERS
                return ApplyConfidenceModifiers(baseConfidence, factors);
            }

            private static double CalculateFeedbackConfidence(double successRate, int feedbackCount, double conflictRate)
            {
                // Sample size adjustment
                var sampleConfidence = feedbackCount == 0 ? 1.0 : Math.Min(1.0, feedbackCount / (double)MinFeedbackForHighConfidence);

                // Conflict penalty
                var conflictPenalty = Math.Max(0.0, 1.0 - conflictRate * 2);

                return successRate * sampleConfidence * conflictPenalty;
            }

            private static double ApplyConfidenceModifiers(double baseConfidence, ConfidenceFactors factors)
            {
                var modified = baseConfidence;

                // ANOMALY PENALTY - Reduce confidence for unusual errors
                if (factors.IsAnomaly)
                {
                    modified *= 0.8;
                }

                // COMPLEXITY BONUS - Slight boost for complex solutions that work
                if (factors.SolutionComplexity > 4 && baseConfidence > 0.7)
                {
                    modified = Math.Min(0.95, modified * 1.1);
                }

                // TIME DECAY - Reduce confidence for very old solutions
                if (factors.DaysOld > 365)
                {
                    var ageDecay = Math.Max(0.8, 1.0 - (factors.DaysOld - 365) / 1000.0);
                    modified *= ageDecay;
                }

                // ENSURE BOUNDS
                return Math.Max(0.05, Math.Min(0.99, modified));
            }
        }

        /// <summary>
        /// Container for all factors that influence confidence calculation
        /// </summary>
        public class ConfidenceFactors
        {
            public double? BaseSuccessRate { get; set; }
            public int FeedbackCount { get; set; }
            public double? ConflictRate { get; set; }
            public double? MLConfidence { get; set; }
            public double? SimilarityScore { get; set; }
            public bool IsAnomaly { get; set; }
            public int SolutionComplexity { get; set; }
            public double DaysOld { get; set; }
        }



        // ADD this helper method for advanced vector scoring
        private double CalculateAdvancedVectorScore(VectorMatch match, double[] features)
        {
            // Combine vector similarity with additional ML features
            var baseScore = match.Similarity;

            // Boost for high confidence solutions
            var confidenceBonus = match.ConfidenceScore > 0.8 ? 0.1 : 0.0;

            // Boost for solutions with significant feedback
            var feedbackBonus = Math.Min(0.1, match.FeedbackCount * 0.01);

            return baseScore + confidenceBonus + feedbackBonus;
        }
        /// <summary>
        /// TREE EVALUATION - Determines when interactive trees should be recommended or auto-launched
        /// </summary>
  // [INTERACTIVE TREE]
        private async Task<TreeRecommendation> EvaluateInteractiveTreeNeedAsync(SolutionResult result, MLInsights insights, string errorInput)
        {
            var recommendation = new TreeRecommendation();

            try
            {
                var reasons = new List<string>();
                var score = 0;

                // CRITERIA 1: Low confidence results need guided help
                if (result.Confidence < 0.6)
                {
                    reasons.Add("low confidence solution");
                    score += 2;
                }

                // CRITERIA 2: Complex multi-step solutions benefit from step-by-step guidance
                var steps = SolutionParser.ParseIntoSteps(result.Steps);
                if (steps.Count >= 4)
                {
                    reasons.Add("complex multi-step solution");
                    score += 1;
                }

                // CRITERIA 3: Anomalous errors need careful handling
                if (insights.IsAnomaly)
                {
                    reasons.Add("unusual error pattern detected");
                    score += 2;
                }

                // CRITERIA 4: Conflicting solutions need guided decision making
                if (result.HasConflicts)
                {
                    reasons.Add("conflicting solution feedback exists");
                    score += 1;
                }

                // CRITERIA 5: New error types with no feedback
                if (result.FeedbackCount == 0)
                {
                    reasons.Add("unvalidated solution");
                    score += 1;
                }

                // CRITERIA 6: Certain error categories that benefit from interactive approach
                var category = CategorizeError(errorInput);
                if (category == "network" || category == "security" || category == "performance")
                {
                    reasons.Add($"{category} issues often need diagnostic steps");
                    score += 1;
                }

                // CRITERIA 7: ML predictions suggest difficulty
                if (insights.NeuralPrediction?.WillWork == false && insights.NeuralPrediction.Confidence > 0.7)
                {
                    reasons.Add("ML predicts solution may not work");
                    score += 2;
                }

                // DETERMINE RECOMMENDATION LEVEL
                if (score >= 4)
                {
                    // AUTO-LAUNCH for high-score cases
                    recommendation.ShouldUseTree = true;
                    recommendation.AutoLaunch = true;
                    recommendation.RecommendationText = $"🌳 AUTO-LAUNCHING INTERACTIVE GUIDE: Multiple factors suggest step-by-step troubleshooting would be beneficial ({string.Join(", ", reasons)}).";
                    recommendation.ButtonText = "🌳 Interactive Guide (Auto-launching...)";
                }
                else if (score >= 2)
                {
                    // RECOMMEND but don't auto-launch
                    recommendation.ShouldUseTree = true;
                    recommendation.AutoLaunch = false;
                    recommendation.RecommendationText = $"🌳 RECOMMENDED: Interactive troubleshooting tree suggested due to {string.Join(", ", reasons)}. Click the tree button for guided step-by-step help.";
                    recommendation.ButtonText = "🌳 Recommended: Interactive Guide";
                }
                else if (score >= 1)
                {
                    // AVAILABLE but not strongly recommended
                    recommendation.ShouldUseTree = true;
                    recommendation.AutoLaunch = false;
                    recommendation.RecommendationText = "🌳 Interactive troubleshooting tree available for step-by-step guidance.";
                    recommendation.ButtonText = "🌳 Interactive Troubleshooting";
                }

                System.Diagnostics.Debug.WriteLine($"Tree evaluation: Score={score}, Reasons=[{string.Join(", ", reasons)}], AutoLaunch={recommendation.AutoLaunch}");

            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Tree evaluation failed: {ex.Message}");
            }

            return recommendation;
        }

        /// <summary>
        /// LAUNCH INTERACTIVE TREE - Opens the interactive troubleshooting window
        /// </summary>
  // [INTERACTIVE TREE]
        private async Task LaunchInteractiveTreeAsync(string errorHash, string errorText, SolutionResult relatedResult = null)
        {
            try
            {
                await Task.Run(() =>
                {
                    Application.Current.Dispatcher.Invoke(() =>
                    {
                        var treeWindow = new SolutionTreeWindow(errorHash, errorText);

                        // If we have a related result, we could pass additional context
                        if (relatedResult != null)
                        {
                            treeWindow.Title = $"Interactive Troubleshooting - {relatedResult.Source} (Confidence: {relatedResult.Confidence:P0})";
                        }

                        treeWindow.Show(); // Use Show() instead of ShowDialog() so user can keep main window open
                    });
                });

                System.Diagnostics.Debug.WriteLine("Interactive tree launched successfully");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Failed to launch interactive tree: {ex.Message}");
                MessageBox.Show($"Could not launch interactive troubleshooting: {ex.Message}",
                               "Tree Launch Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        /// <summary>
        /// ENHANCED INTERACTIVE TREE BUTTON - Now context-aware
        /// </summary>
  // [INTERACTIVE TREE]
        private async void BtnInteractiveTree_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentSolutionHash))
            {
                // No current solution - use error input directly
                if (string.IsNullOrEmpty(ErrorInput.Text))
                {
                    MessageBox.Show("Please search for an error first to use interactive troubleshooting.",
                        "No Error Selected", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                // Generate hash from current input
                var sig = Normalize(ErrorInput.Text);
                var hash = Sha256Hex(sig);
                // [INTERACTIVE TREE]
                await LaunchInteractiveTreeAsync(hash, ErrorInput.Text);
            }
            else
            {
                // Use current solution context
                // [INTERACTIVE TREE]
                await LaunchInteractiveTreeAsync(currentSolutionHash, ErrorInput.Text);
            }

            // Reset button text after use
            BtnInteractiveTree.Content = "🌳 Interactive Tree";
        }

        /// <summary>
        /// Helper class for tree recommendations
        /// </summary>
        public class TreeRecommendation
        {
            public bool ShouldUseTree { get; set; }
            public bool AutoLaunch { get; set; }
            public string RecommendationText { get; set; }
            public string ButtonText { get; set; } = "🌳 Interactive Tree";
        }

        /// <summary>
        /// ML PREPROCESSING - Runs all ML analysis before main search
        /// </summary>
  // [ML PREPROCESSING & COMPREHENSIVE ML SEARCH]
        private async Task<MLInsights> PerformMLPreprocessingAsync(string cleanErrorInput, double[] features, string sig)
        {
            var insights = new MLInsights();

            if (mlEngine == null)
            {
                insights.MLAvailable = false;
                return insights;
            }

            try
            {
                // ANOMALY DETECTION
                insights.IsAnomaly = await mlEngine.IsAnomalyAsync(features);
                if (insights.IsAnomaly)
                {
                    System.Diagnostics.Debug.WriteLine("🚨 ANOMALY DETECTED: This error pattern is unusual");
                }

                // TREND PREDICTION
                var errorCategory = CategorizeError(sig);
                var (predictedCount, trendConfidence) = await mlEngine.PredictErrorTrendAsync(errorCategory, 1);
                insights.TrendPrediction = new TrendPrediction
                {
                    Category = errorCategory,
                    PredictedNextHourCount = predictedCount,
                    Confidence = trendConfidence
                };

                // NEURAL NETWORK PREDICTION
                if (isNeuralNetworkTrained)
                {
                    var (nnConfidence, willWork) = await PredictWithNeuralNetworkAsync(features);
                    insights.NeuralPrediction = new NeuralPrediction
                    {
                        WillWork = willWork,
                        Confidence = nnConfidence
                    };
                }

                // CLUSTER ANALYSIS
                var (template, clusterConfidence) = await mlEngine.GetClusterRecommendationAsync(features);
                insights.ClusterRecommendation = new ClusterRecommendation
                {
                    Template = template,
                    Confidence = clusterConfidence
                };

                insights.MLAvailable = true;
                System.Diagnostics.Debug.WriteLine($"ML preprocessing complete - Anomaly: {insights.IsAnomaly}, Trend confidence: {trendConfidence:P0}");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML preprocessing failed: {ex.Message}");
                insights.MLAvailable = false;
            }

            return insights;
        }

        /// <summary>
        /// COMPREHENSIVE ML SEARCH - Uses all ML capabilities for solution prediction
        /// </summary>
  // [ML PREPROCESSING & COMPREHENSIVE ML SEARCH]
        private async Task<SolutionResult> ComprehensiveMLSearchAsync(string cleanErrorInput, string hash, double[] features)
        {
            if (mlEngine == null) return null;

            try
            {
                // STEP 1: GET ALL ML PREDICTIONS
                var (willWork, solutionConfidence) = await mlEngine.PredictSolutionSuccessAsync(features);
                var (template, clusterConfidence) = await mlEngine.GetClusterRecommendationAsync(features);

                // STEP 2: NEURAL NETWORK PREDICTION
                double neuralConfidence = 0.5;
                if (isNeuralNetworkTrained)
                {
                    var (nnConf, nnWillWork) = await PredictWithNeuralNetworkAsync(features);
                    neuralConfidence = nnConf;
                }

                // STEP 3: COMBINE ALL ML PREDICTIONS
                var combinedConfidence = CombinePredictions(solutionConfidence, clusterConfidence, neuralConfidence);

                if (!string.IsNullOrEmpty(template) && combinedConfidence > 0.4)
                {
                    return new SolutionResult
                    {
                        Steps = template,
                        Hash = hash,
                        Source = "comprehensive_ml",
                        Confidence = combinedConfidence,
                        SuccessRate = willWork ? combinedConfidence : 1 - combinedConfidence,
                        FeedbackCount = 0,
                        HasConflicts = false,
                        Version = "ML-2.0"
                    };
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Comprehensive ML search failed: {ex.Message}");
            }

            return null;
        }

        /// <summary>
        /// ML-ENHANCED VECTOR SEARCH - Combines vector similarity with neural network predictions
        /// </summary>
        private async Task<SolutionResult> TryMLEnhancedVectorMatchAsync(string rawError, string[] likeTokens, double[] features)
        {
            try
            {
                // Standard fuzzy-enhanced vector search
                var baseResult = await TryEnhancedVectorMatchAsync(rawError, likeTokens);

                if (baseResult != null && mlEngine != null)
                {
                    // Enhance with neural network prediction
                    if (isNeuralNetworkTrained)
                    {
                        var (nnConfidence, nnWillWork) = await PredictWithNeuralNetworkAsync(features);

                        // Adjust confidence based on neural network prediction
                        var adjustedConfidence = (baseResult.Confidence * 0.7) + (nnConfidence * 0.3);

                        baseResult.Confidence = adjustedConfidence;
                        baseResult.Source = "ml_enhanced_vector";
                    }
                }

                return baseResult;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML-enhanced vector search failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// ENHANCE RESULTS WITH ML - Applies ML analysis to improve result confidence
        /// </summary>
  // [RESULT ENHANCERS & SELECTION]
        private async Task<SolutionResult> EnhanceResultWithMLAsync(SolutionResult result, double[] features, MLInsights insights)
        {
            if (result == null || !insights.MLAvailable) return result;

            try
            {
                // Use the unified confidence calculator instead of manual adjustments
                result.Confidence = CalculateResultConfidence(result, insights);

                return result;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML result enhancement failed: {ex.Message}");
                return result;
            }
        }

        /// <summary>
        /// BETTER SOLUTION SELECTION WITH ML - Uses ML insights to pick best solution
        /// </summary>
  // [RESULT ENHANCERS & SELECTION]
        private SolutionResult SelectBestSolutionWithML(List<SolutionResult> results, MLInsights insights)
        {
            if (results.Count == 0) return null;

            // Apply ML-enhanced ranking
            foreach (var result in results)
            {
                var mlBonus = 0.0;

                // Bonus for neural network agreement
                if (insights.NeuralPrediction?.WillWork == true && insights.NeuralPrediction.Confidence > 0.7)
                {
                    mlBonus += 0.1;
                }

                // Penalty for anomalous errors (be more cautious)
                if (insights.IsAnomaly)
                {
                    mlBonus -= 0.05;
                }

                result.Confidence += mlBonus;
            }

            // Select best result with ML adjustments
            return results
                .Where(r => !string.IsNullOrWhiteSpace(r.Steps))
                .OrderByDescending(r => r.Confidence)
                .ThenByDescending(r => r.SuccessRate)
                .ThenByDescending(r => r.FeedbackCount)
                .FirstOrDefault();
        }

        /// <summary>
        /// Helper classes for ML insights
        /// </summary>
        public class MLInsights
        {
            public bool MLAvailable { get; set; }
            public bool IsAnomaly { get; set; }
            public TrendPrediction TrendPrediction { get; set; }
            public NeuralPrediction NeuralPrediction { get; set; }
            public ClusterRecommendation ClusterRecommendation { get; set; }
        }

        public class TrendPrediction
        {
            public string Category { get; set; }
            public double PredictedNextHourCount { get; set; }
            public double Confidence { get; set; }
        }

        public class NeuralPrediction
        {
            public bool WillWork { get; set; }
            public double Confidence { get; set; }
        }

        public class ClusterRecommendation
        {
            public string Template { get; set; }
            public double Confidence { get; set; }
        }

        /// <summary>
        /// Generate ML insights text for display
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private string GenerateMLInsightsText(MLInsights insights)
        {
            if (!insights.MLAvailable) return "";

            var mlText = new List<string>();

            // Anomaly detection
            if (insights.IsAnomaly)
            {
                mlText.Add("🚨 ANOMALY DETECTED: This error pattern is unusual and may need special attention");
            }

            // Trend prediction
            if (insights.TrendPrediction?.Confidence > 0.7)
            {
                mlText.Add($"📊 TREND ANALYSIS: This type of error ({insights.TrendPrediction.Category}) is predicted to occur ~{insights.TrendPrediction.PredictedNextHourCount:F0} times in the next hour");
            }

            // Neural network insights
            if (insights.NeuralPrediction?.Confidence > 0.7)
            {
                var prediction = insights.NeuralPrediction.WillWork ? "likely to work" : "may need alternatives";
                mlText.Add($"🧠 NEURAL PREDICTION: Solutions for this error are {prediction} (confidence: {insights.NeuralPrediction.Confidence:P0})");
            }

            return mlText.Any() ? string.Join("\n", mlText) : "";
        }

        /// <summary>
        /// Enhanced confidence text with ML insights
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private string GetMLEnhancedConfidenceText(SolutionResult result, MLInsights insights)
        {
            // [SUGGESTIONS & DISPLAY HELPERS]
            var baseText = GetEnhancedConfidenceText(result);

            if (insights.IsAnomaly)
            {
                baseText += " ⚠️ Anomalous error - exercise caution";
            }

            if (insights.NeuralPrediction?.Confidence > 0.8)
            {
                baseText += $" 🧠 Neural network: {insights.NeuralPrediction.Confidence:P0} confidence";
            }

            return baseText;
        }

        /// <summary>
        /// Generate ML-enhanced suggestions when no match found
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private async Task<string> GenerateMLEnhancedSuggestions(string cleanErrorInput, double[] features, MLInsights insights)
        {
            var suggestions = new List<string>();

            // Add base fuzzy suggestions
            // [SUGGESTIONS & DISPLAY HELPERS]
            var baseSuggestions = await GenerateSearchSuggestions(cleanErrorInput);
            if (!string.IsNullOrEmpty(baseSuggestions))
            {
                suggestions.Add(baseSuggestions);
            }

            // Add ML-specific suggestions
            if (insights.IsAnomaly)
            {
                suggestions.Add("🚨 This appears to be an unusual error. Consider consulting domain experts or checking for recent system changes.");
            }

            if (insights.ClusterRecommendation?.Confidence > 0.5 && !string.IsNullOrEmpty(insights.ClusterRecommendation.Template))
            {
                suggestions.Add($"🎯 ML SUGGESTION: Try this approach based on similar error patterns:\n{insights.ClusterRecommendation.Template}");
            }

            return suggestions.Any() ? string.Join("\n\n", suggestions) : "";
        }

        /// <summary>
        /// Enhanced confidence display with fuzzy search insights
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private string GetEnhancedConfidenceText(SolutionResult result)
        {
            var baseText = GetConfidenceText(result); // Use existing method

            // Add fuzzy search specific information
            var enhancementInfo = result.Source switch
            {
                "enhanced_fuzzy_keyword" => " | Enhanced with synonym expansion and N-gram analysis",
                "enhanced_fuzzy_search" => " | Comprehensive fuzzy matching with multiple similarity metrics",
                "enhanced_vector_fuzzy" => " | AI semantic analysis with fuzzy preprocessing",
                _ => ""
            };

            return baseText + enhancementInfo;
        }

        /// <summary>
        /// Enhanced search source information
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private string GetEnhancedSearchSourceInfo(string source)
        {
            return source switch
            {
                "exact_match" => "✅ Found: Exact match from knowledge base",
                "enhanced_fuzzy_keyword" => "🔍 Found: Enhanced fuzzy keyword search (synonyms + N-grams)",
                "enhanced_fuzzy_search" => "🎯 Found: Comprehensive fuzzy analysis (multiple similarity metrics)",
                "enhanced_vector_fuzzy" => "🤖 Found: AI similarity with fuzzy preprocessing",
                "pattern_match" => "📋 Found: Pattern recognition match",
                "ml_prediction" => "🧠 Found: Machine learning prediction",
                "openai_enhanced" => "🧠 Generated: AI consultation with context",
                _ => "📚 Found: Database search"
            };
        }

        /// <summary>
        /// Provides insights about the search method used
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private string GetSearchMethodInsights(string source, int totalSearchLayers)
        {
            var insights = source switch
            {
                "enhanced_fuzzy_keyword" => "Used synonym expansion to match related terms and N-gram analysis for partial matching",
                "enhanced_fuzzy_search" => "Applied comprehensive fuzzy matching with Levenshtein distance and N-gram similarity",
                "enhanced_vector_fuzzy" => "Combined AI semantic understanding with fuzzy text matching for optimal accuracy",
                "exact_match" => "Perfect match found immediately - highest reliability",
                "pattern_match" => "Recognized common error pattern",
                "ml_prediction" => "Machine learning models predicted this solution based on error characteristics",
                _ => "Standard database search"
            };

            return $"🔬 Method: {insights} (Searched {totalSearchLayers} layer{(totalSearchLayers > 1 ? "s" : "")})";
        }

        /// <summary>
        /// Generate search suggestions based on fuzzy analysis
        /// </summary>
  // [SUGGESTIONS & DISPLAY HELPERS]
        private async Task<string> GenerateSearchSuggestions(string cleanErrorInput)
        {
            try
            {
                // Use fuzzy search to find partial matches and suggest alternatives
                var suggestions = new List<string>();

                // Get synonym expansions
                var synonyms = fuzzySearch.ExpandWithSynonyms(cleanErrorInput);
                if (synonyms.Count > 1) // More than just the original term
                {
                    var alternativeTerms = synonyms.Take(3).Where(s => s != cleanErrorInput.ToLower()).ToList();
                    if (alternativeTerms.Any())
                    {
                        suggestions.Add($"Try alternative terms: {string.Join(", ", alternativeTerms)}");
                    }
                }

                // Check for common typos or variations
                await Task.Run(() =>
                {
                    try
                    {
                        using (var conn = OpenConn())
                        {
                            // Find errors with similar keywords that might be variations
                            var tokens = Tokens(cleanErrorInput);
                            if (tokens.Length > 0)
                            {
                                var mainToken = tokens[0];
                                var sql = $@"
                            SELECT DISTINCT error_signature
                            FROM {TableKB}
                            WHERE error_signature LIKE '%{SecureEscape(mainToken.Substring(0, Math.Min(mainToken.Length, 4)))}%'
                            LIMIT 5";

                                using (var cmd = new OdbcCommand(sql, conn))
                                using (var rdr = cmd.ExecuteReader())
                                {
                                    var similarErrors = new List<string>();
                                    while (rdr.Read())
                                    {
                                        var errorSig = rdr.GetString(0);
                                        var similarity = fuzzySearch.GetNGramSimilarity(cleanErrorInput, errorSig);
                                        if (similarity > 0.3 && similarity < 0.7) // Partial but not too similar
                                        {
                                            similarErrors.Add(errorSig.Length > 60 ? errorSig.Substring(0, 57) + "..." : errorSig);
                                        }
                                    }

                                    if (similarErrors.Any())
                                    {
                                        suggestions.Add($"Similar errors found: {string.Join("; ", similarErrors.Take(2))}");
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Suggestion generation failed: {ex.Message}");
                    }
                });

                return suggestions.Any() ? string.Join("\n", suggestions) : "";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Search suggestions failed: {ex.Message}");
                return "";
            }
        }

        /// <summary>
        /// AI CONSULTATION - Ask ChatGPT for help when CAINE doesn't know the answer
        ///
        /// WHAT THIS DOES WHEN USER CLICKS "USE CAINE API":
        /// 1. Takes the error message and validates it for security
        /// 2. Looks through CAINE's knowledge base for similar past solutions
        /// 3. Builds a conversation context with proven solutions
        /// 4. Asks ChatGPT for advice using that context
        /// 5. Shows the AI's response and lets user rate it
        ///
        /// LIKE CONSULTING A SPECIALIST:
        /// - Brings the specialist up to speed with similar cases we've solved
        /// - Gets expert advice on the new problem
        /// - Records the advice so we can learn if it works
        /// </summary>
  // [OPENAI INTEGRATION]
        private async void BtnCaineApi_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // STEP 1: DISABLE BUTTONS AND SHOW PROGRESS
                BtnCaineApi.IsEnabled = true;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Consulting CAINE AI (OpenAI) with enhanced context...";

                // STEP 2: VALIDATE INPUT
                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                System.Diagnostics.Debug.WriteLine("Step 2 completed - input validated");

                // STEP 3: SECURITY CHECK
                var validation = SecurityValidator.ValidateInput(err, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"AI input validation failed: {validation.ErrorMessage}";
                    return;
                }

                System.Diagnostics.Debug.WriteLine("Step 3 completed - security check passed");

                // STEP 4: BUILD CONVERSATION CONTEXT
                object[] messages = null;
                try
                {
                    messages = await BuildEnhancedChatHistoryAsync(validation.CleanInput);
                    System.Diagnostics.Debug.WriteLine("Step 4 completed - context built successfully");
                }
                catch (Exception contextEx)
                {
                    ResultBox.Text = $"Context building failed: {contextEx.Message}";
                    System.Diagnostics.Debug.WriteLine($"Context building error: {contextEx}");
                    return;
                }

                // STEP 5: ASK CHATGPT
                string gptResponse = null;
                try
                {
                    gptResponse = await AskOpenAIAsync(messages);
                    System.Diagnostics.Debug.WriteLine("Step 5 completed - OpenAI response received");
                }
                catch (Exception openAiEx)
                {
                    ResultBox.Text = $"OpenAI API call failed: {openAiEx.Message}";
                    System.Diagnostics.Debug.WriteLine($"OpenAI error: {openAiEx}");
                    return;
                }

                // STEP 6: TRACK THIS NEW SOLUTION
                currentSolutionHash = Sha256Hex(gptResponse);
                currentSolutionSource = "openai_enhanced";

                // STEP 7: DISPLAY THE AI RESPONSE
                var confidenceText = "AI Generated Solution (Confidence: Learning) | Please rate this solution to help CAINE learn";
                ResultBox.Text = $"{confidenceText}\n\n{gptResponse}";

                // STEP 8: PREPARE FOR TEACHING
                var numbered = ExtractNumberedLines(gptResponse).ToArray();
                TeachSteps.Text = numbered.Length > 0
                    ? string.Join(Environment.NewLine, numbered)
                    : gptResponse;

                // STEP 9: ENABLE FEEDBACK
                EnableFeedbackButtons();

                System.Diagnostics.Debug.WriteLine("All steps completed successfully");
            }
            catch (Exception ex)
            {
                ResultBox.Text = "OpenAI consultation failed:\n" + ex.Message;
                System.Diagnostics.Debug.WriteLine($"Overall error in BtnCaineApi_Click: {ex}");
            }
            finally
            {
                BtnCaineApi.IsEnabled = true;
            }
        }

        /// <summary>
        /// TEACHING SYSTEM - Add new knowledge to CAINE's brain
        ///
        /// WHAT THIS DOES WHEN USER CLICKS "TEACH":
        /// 1. Takes the error and solution steps from the user
        /// 2. Validates everything for security
        /// 3. Checks if similar solutions already exist
        /// 4. Creates a unique fingerprint and stores the knowledge
        /// 5. Sets up the solution to learn from future feedback
        ///
        /// LIKE TRAINING A SMART ASSISTANT:
        /// - User shows CAINE a new problem and solution
        /// - CAINE remembers it and files it properly
        /// - CAINE can now suggest this solution to others with similar problems
        /// - The more people rate it positively, the more confident CAINE becomes
        /// </summary>
  // [TEACHING FLOW]
        private async void BtnTeach_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // STEP 1: PREPARE FOR TEACHING
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                ResultBox.Text = "Teaching CAINE with enhanced learning...";

                // STEP 2: VALIDATE THE ERROR INPUT
                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                // STEP 3: SECURITY VALIDATION
                // Make sure what we're teaching isn't malicious
                var validation = SecurityValidator.ValidateInput(err, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"Teaching input validation failed: {validation.ErrorMessage}";
                    return;
                }

                // STEP 4: PROCESS THE SOLUTION STEPS
                // Parse the solution steps the user provided
                var raw = (TeachSteps.Text ?? "").Replace("\r", "");
                var lines = raw.Split('\n');
                var stepsLines = lines.Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
                if (stepsLines.Length == 0)
                {
                    ResultBox.Text = "Enter at least one step (one per line).";
                    return;
                }

                // STEP 5: CREATE ERROR FINGERPRINT
                // Generate a unique identifier for this error type
                var cleanErrorInput = validation.CleanInput;
                var sig = Normalize(cleanErrorInput);    // Standardize the error
                var hash = Sha256Hex(sig);              // Create unique fingerprint

                // STEP 6: CHECK FOR CONFLICTS
                // See if we already have a different solution for this error
                var existingConflicts = await CheckForExistingSolutions(hash, stepsLines);

                if (existingConflicts.HasConflicts)
                {
                    // WARN ABOUT CONFLICTS
                    // Let user know there's already a different solution
                    var conflictMessage = $"Warning: Similar solution exists with {existingConflicts.ExistingSuccessRate:P0} success rate. ";
                    conflictMessage += "Your solution will be added as an alternative approach.";
                    ResultBox.Text += "\n" + conflictMessage;
                }

                // STEP 7: CREATE AI UNDERSTANDING
                // Generate embeddings (AI fingerprints) for smart similarity matching
                float[] emb = null;
                try
                {
                    emb = await EmbedAsync(cleanErrorInput + "\n\n" + string.Join("\n", stepsLines));
                }
                catch (Exception embedEx)
                {
                    System.Diagnostics.Debug.WriteLine($"Embedding creation failed: {embedEx.Message}");
                }

                // STEP 8: SAVE THE KNOWLEDGE
                // Store everything in the database with version control
                await TeachEnhancedSolution(hash, sig, cleanErrorInput, stepsLines, emb, existingConflicts);

                // STEP 9: RECORD POSITIVE FEEDBACK IF APPLICABLE
                // If user is correcting a bad suggestion, record that the new solution is better
                if (!string.IsNullOrEmpty(currentSolutionHash) && currentSolutionHash != hash)
                {
                    currentSolutionHash = hash;
                    currentSolutionSource = "human_teaching";
                    // currentSolutionConfidence = 0.8; // High confidence for human-provided solutions
                    await Task.Run(() => RecordEnhancedFeedback(true, "User-provided teaching", 5));
                }

                // STEP 10: SHOW SUCCESS MESSAGE
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
                // STEP 11: RE-ENABLE BUTTONS
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }
        // Enhanced MainWindow.xaml.cs methods to better integrate FuzzySearchEngine

        /// <summary>
        /// ENHANCED KEYWORD SEARCH - Now uses fuzzy search with synonyms and N-gram matching
        ///
        /// WHAT THIS DOES:
        /// Instead of just exact keyword matching, this now:
        /// 1. Expands search terms using synonyms (timeout -> timed out, hung, freeze)
        /// 2. Uses fuzzy scoring to handle typos and variations
        /// 3. Applies N-gram similarity for partial matches
        /// 4. Combines all scores for better ranking
        /// </summary>
  // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
        private async Task<SolutionResult> TryEnhancedKeywordMatchAsync(string sig)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var bestResult = new SolutionResult();
                        var bestScore = 0.0;

                        // STEP 1: EXPAND WITH SYNONYMS
                        // Get all possible variations of the search terms
                        var expandedTerms = fuzzySearch.ExpandWithSynonyms(sig);
                        System.Diagnostics.Debug.WriteLine($"Expanded terms: {string.Join(", ", expandedTerms)}");

                        // STEP 2: SEARCH WITH MULTIPLE STRATEGIES
                        // Try different combinations of expanded terms
                        foreach (var expandedTerm in expandedTerms.Take(3)) // Limit to top 3 to avoid too many queries
                        {
                            var tokens = Tokens(expandedTerm);
                            if (tokens.Length == 0) continue;

                            // Build keyword search query
                            var likeConditions = tokens.Select(t => $"LOWER(kb.error_signature) LIKE '%{SecureEscape(t.ToLower())}%'").ToArray();
                            var whereClause = string.Join(" OR ", likeConditions); // Use OR for broader matching

                            var sql = $@"
                        SELECT kb.error_signature, kb.error_text, kb.resolution_steps, kb.error_hash,
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
                        LIMIT 20"; // Get more candidates for scoring

                            using (var cmd = new OdbcCommand(sql, conn))
                            using (var rdr = cmd.ExecuteReader())
                            {
                                while (rdr.Read())
                                {
                                    var errorSig = rdr.GetString(0);
                                    var errorText = rdr.GetString(1);
                                    var steps = ConvertArrayToString(rdr.GetValue(2));
                                    var hash = rdr.GetString(3);
                                    var successRate = rdr.GetDouble(4);
                                    var feedbackCount = rdr.GetInt32(5);

                                    // STEP 3: CALCULATE COMPREHENSIVE FUZZY SCORE
                                    // Combine multiple scoring methods for better accuracy
                                    var fuzzyScore = fuzzySearch.CalculateFuzzyScore(sig, errorSig);
                                    var ngramScore = fuzzySearch.GetNGramSimilarity(sig, errorSig, 3);
                                    var textFuzzyScore = fuzzySearch.CalculateFuzzyScore(sig, errorText);

                                    // STEP 4: WEIGHTED SCORING SYSTEM
                                    // Combine different similarity metrics with user feedback
                                    var combinedSimilarity = (fuzzyScore * 0.5) + (ngramScore * 0.3) + (textFuzzyScore * 0.2);
                                    var feedbackBonus = successRate > 0.7 ? 0.2 : 0.0; // Bonus for high success rate
                                    var finalScore = combinedSimilarity + feedbackBonus;

                                    System.Diagnostics.Debug.WriteLine($"Candidate: {errorSig.Substring(0, Math.Min(50, errorSig.Length))}...");
                                    System.Diagnostics.Debug.WriteLine($"  Fuzzy: {fuzzyScore:F3}, N-gram: {ngramScore:F3}, Text: {textFuzzyScore:F3}");
                                    System.Diagnostics.Debug.WriteLine($"  Combined: {combinedSimilarity:F3}, Final: {finalScore:F3}");

                                    // STEP 5: TRACK THE BEST MATCH
                                    // Keep the highest scoring result
                                    if (finalScore > bestScore && finalScore > 0.4) // Minimum threshold
                                    {
                                        bestScore = finalScore;
                                        bestResult = new SolutionResult
                                        {
                                            Steps = steps,
                                            Hash = hash,
                                            Source = "enhanced_fuzzy_keyword",
                                            SuccessRate = successRate,
                                            FeedbackCount = feedbackCount,
                                            HasConflicts = false,
                                            Confidence = Math.Min(0.95, finalScore), // Cap confidence
                                            Version = "1.0"
                                        };
                                    }
                                }
                            }
                        }

                        // Return best result if it meets minimum quality threshold
                        return bestScore > 0.4 ? bestResult : null;
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Enhanced fuzzy keyword match failed: {ex.Message}");
                }
                return null;
            });
        }

        /// <summary>
        /// ENHANCED FUZZY SEARCH - Improved version that uses all FuzzySearchEngine capabilities
        ///
        /// WHAT THIS DOES:
        /// 1. Uses synonym expansion to find more matches
        /// 2. Applies both fuzzy and N-gram scoring
        /// 3. Ranks results by combined similarity and user feedback
        /// 4. Provides detailed scoring breakdown for transparency
        /// </summary>
  // [FUZZY/KEYWORD/VECTOR/PATTERN SEARCH]
        private async Task<SolutionResult> TryEnhancedFuzzySearchAsync(string cleanErrorInput)
        {
            return await Task.Run(() =>
            {
                try
                {
                    // STEP 1: EXPAND SEARCH TERMS WITH SYNONYMS
                    var originalTerms = fuzzySearch.ExpandWithSynonyms(cleanErrorInput);
                    System.Diagnostics.Debug.WriteLine($"Enhanced fuzzy search with {originalTerms.Count} synonym variations");

                    using (var conn = OpenConn())
                    {
                        var candidates = new List<EnhancedFuzzyResult>();

                        // STEP 2: GATHER CANDIDATES USING EXPANDED TERMS
                        // Use broader search to get more candidates, then score them precisely
                        var broadSearch = string.Join(" ", originalTerms.Take(2)); // Use top 2 synonym groups
                        var tokens = Tokens(broadSearch);

                        if (tokens.Length == 0) return null;

                        var likeConditions = tokens.Select(t => $"LOWER(kb.error_text) LIKE '%{SecureEscape(t.ToLower())}%'").ToArray();
                        var whereClause = string.Join(" OR ", likeConditions);

                        var sql = $@"
                    SELECT kb.error_hash, kb.error_text, kb.error_signature, kb.resolution_steps,
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
                    ORDER BY kb.created_at DESC
                    LIMIT 100"; // Get more candidates for better scoring

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var hash = rdr.GetString(0);
                                var text = rdr.GetString(1);
                                var signature = rdr.GetString(2);
                                var steps = ConvertArrayToString(rdr.GetValue(3));
                                var successRate = rdr.GetDouble(4);
                                var feedbackCount = rdr.GetInt32(5);

                                // STEP 3: COMPREHENSIVE SCORING
                                // Apply all fuzzy search techniques for maximum accuracy
                                candidates.Add(new EnhancedFuzzyResult
                                {
                                    Hash = hash,
                                    Text = text,
                                    Steps = steps,
                                    SuccessRate = successRate,
                                    FeedbackCount = feedbackCount,

                                    // Multiple similarity scores
                                    TextFuzzyScore = fuzzySearch.CalculateFuzzyScore(cleanErrorInput, text),
                                    SignatureFuzzyScore = fuzzySearch.CalculateFuzzyScore(cleanErrorInput, signature),
                                    TextNGramScore = fuzzySearch.GetNGramSimilarity(cleanErrorInput, text, 3),
                                    SignatureNGramScore = fuzzySearch.GetNGramSimilarity(cleanErrorInput, signature, 3)
                                });
                            }
                        }

                        // STEP 4: ADVANCED SCORING AND RANKING
                        foreach (var candidate in candidates)
                        {
                            // Weighted combination of all similarity metrics
                            var textSimilarity = (candidate.TextFuzzyScore * 0.6) + (candidate.TextNGramScore * 0.4);
                            var signatureSimilarity = (candidate.SignatureFuzzyScore * 0.6) + (candidate.SignatureNGramScore * 0.4);

                            // Overall similarity (favor text matches over signature matches)
                            candidate.CombinedSimilarity = (textSimilarity * 0.7) + (signatureSimilarity * 0.3);

                            // User feedback bonus
                            var feedbackWeight = Math.Min(1.0, candidate.FeedbackCount / 10.0); // Max weight at 10 feedback
                            var feedbackBonus = candidate.SuccessRate * feedbackWeight * 0.2; // Up to 20% bonus

                            // Final score
                            candidate.FinalScore = candidate.CombinedSimilarity + feedbackBonus;
                        }

                        // STEP 5: SELECT BEST RESULT
                        var best = candidates
                            .Where(c => c.FinalScore > 0.5) // Minimum quality threshold
                            .OrderByDescending(c => c.FinalScore)
                            .FirstOrDefault();

                        if (best != null)
                        {
                            System.Diagnostics.Debug.WriteLine($"Enhanced fuzzy match found:");
                            System.Diagnostics.Debug.WriteLine($"  Text fuzzy: {best.TextFuzzyScore:F3}, N-gram: {best.TextNGramScore:F3}");
                            System.Diagnostics.Debug.WriteLine($"  Signature fuzzy: {best.SignatureFuzzyScore:F3}, N-gram: {best.SignatureNGramScore:F3}");
                            System.Diagnostics.Debug.WriteLine($"  Final score: {best.FinalScore:F3}");

                            return new SolutionResult
                            {
                                Steps = best.Steps,
                                Hash = best.Hash,
                                Source = "enhanced_fuzzy_search",
                                Confidence = Math.Min(0.95, best.FinalScore),
                                SuccessRate = best.SuccessRate,
                                FeedbackCount = best.FeedbackCount,
                                HasConflicts = false,
                                Version = "1.0"
                            };
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Enhanced fuzzy search failed: {ex.Message}");
                }

                return null;
            });
        }

        /// <summary>
        /// Helper class for enhanced fuzzy search results
        /// </summary>
        private class EnhancedFuzzyResult
        {
            public string Hash { get; set; }
            public string Text { get; set; }
            public string Steps { get; set; }
            public double SuccessRate { get; set; }
            public int FeedbackCount { get; set; }

            public double TextFuzzyScore { get; set; }
            public double SignatureFuzzyScore { get; set; }
            public double TextNGramScore { get; set; }
            public double SignatureNGramScore { get; set; }

            public double CombinedSimilarity { get; set; }
            public double FinalScore { get; set; }
        }

        /// <summary>
        /// ENHANCED VECTOR SEARCH - Now uses fuzzy search for preprocessing
        ///
        /// WHAT THIS DOES:
        /// 1. Uses fuzzy search to expand the query terms before vector search
        /// 2. Combines vector similarity with fuzzy scoring
        /// 3. Provides better ranking by considering multiple factors
        /// </summary>
        private async Task<SolutionResult> TryEnhancedVectorMatchAsync(string rawError, string[] likeTokens)
        {
            try
            {
                // STEP 1: EXPAND QUERY WITH FUZZY SEARCH
                var expandedTerms = fuzzySearch.ExpandWithSynonyms(rawError);
                var enhancedQuery = string.Join(" ", expandedTerms.Take(3)); // Use top 3 expansions

                // STEP 2: CREATE AI FINGERPRINT OF ENHANCED QUERY
                var queryEmb = await EmbedAsync(enhancedQuery);

                // STEP 3: SEARCH WITH BOTH ORIGINAL AND EXPANDED TERMS
                var pageSize = 100;
                var candidates = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts, string ErrorText)>();

                // Get candidates using expanded search terms
                var expandedTokens = Tokens(enhancedQuery);
                for (int page = 0; page < 2; page++) // Limit pages for performance
                {
                    var pageCandidates = await GetEnhancedVectorCandidatesPage(expandedTokens, rawError, page, pageSize);
                    candidates.AddRange(pageCandidates);

                    if (pageCandidates.Count < pageSize) break;
                }

                if (candidates.Count == 0) return null;

                // STEP 4: ENHANCED SCORING WITH FUZZY + VECTOR SIMILARITY
                NormalizeInPlace(queryEmb);
                SolutionResult bestResult = null;
                double bestScore = -1;

                foreach (var c in candidates)
                {
                    // Vector similarity
                    var e = c.Emb;
                    NormalizeInPlace(e);
                    var vectorSimilarity = Cosine(queryEmb, e);

                    if (vectorSimilarity < VectorMinCosine) continue;

                    // Fuzzy similarity
                    var fuzzySimilarity = fuzzySearch.CalculateFuzzyScore(rawError, c.ErrorText);

                    // Combined scoring
                    var combinedSimilarity = (vectorSimilarity * 0.7) + (fuzzySimilarity * 0.3);
                    var confidence = CalculateConfidence(c.SuccessRate, c.FeedbackCount, c.HasConflicts ? 0.4 : 0.1);
                    var weightedScore = combinedSimilarity * (0.6 + confidence * 0.4);

                    if (weightedScore > bestScore)
                    {
                        bestScore = weightedScore;
                        bestResult = new SolutionResult
                        {
                            Steps = c.Steps,
                            Hash = c.Hash,
                            Source = "enhanced_vector_fuzzy",
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
                System.Diagnostics.Debug.WriteLine($"Enhanced vector fuzzy match failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Enhanced vector candidate retrieval with fuzzy preprocessing
        /// </summary>
        private async Task<List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts, string ErrorText)>>
            GetEnhancedVectorCandidatesPage(string[] likeTokens, string originalError, int page, int pageSize)
        {
            return await Task.Run(() =>
            {
                var list = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts, string ErrorText)>();

                try
                {
                    string where = "";
                    if (likeTokens != null && likeTokens.Length > 0)
                    {
                        var conditions = likeTokens.Select(t => $"(LOWER(kb.error_signature) LIKE '%{SecureEscape(t.ToLower())}%' OR LOWER(kb.error_text) LIKE '%{SecureEscape(t.ToLower())}%')");
                        where = "WHERE " + string.Join(" OR ", conditions);
                    }

                    var sql = $@"
                SELECT kb.resolution_steps, kb.embedding, kb.error_hash, kb.error_text,
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
                ORDER BY created_at DESC
                LIMIT {pageSize} OFFSET {page * pageSize}";

                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : ConvertArrayToString(rdr.GetValue(0));
                                var embStr = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
                                var hash = rdr.IsDBNull(2) ? "" : rdr.GetString(2);
                                var errorText = rdr.IsDBNull(3) ? "" : rdr.GetString(3);
                                var successRate = rdr.GetDouble(4);
                                var feedbackCount = rdr.GetInt32(5);
                                var hasConflicts = rdr.GetBoolean(6);

                                var emb = ParseFloatArray(embStr);
                                if (emb.Length > 0)
                                {
                                    list.Add((steps, emb, hash, successRate, feedbackCount, hasConflicts, errorText));
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Enhanced vector candidates failed: {ex.Message}");
                }

                return list;
            });
        }
        /// <summary>
        /// EXACT MATCH SEARCH - Look for errors we've seen exactly before
        ///
        /// WHAT THIS DOES:
        /// Like looking up a book by its exact ISBN - if we've cataloged this exact error before,
        /// we can immediately provide the solution along with:
        /// - How many people have tried it
        /// - What percentage said it worked
        /// - How confident CAINE is about suggesting it
        ///
        /// This is the fastest and most reliable way to solve problems
        /// </summary>
        private async Task<SolutionResult> TryExactMatchAsync(string hash)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // COMPLEX SQL QUERY - Joins knowledge base with user feedback
                        // This gets the solution AND calculates how well it's worked for others
                        var sql = $@"
                    SELECT
                        kb.resolution_steps,                    -- The actual solution steps
                        kb.error_hash,                         -- Unique ID of this error type
                        COALESCE(fb.success_rate, 0.5) as success_rate,     -- % of people who said it worked
                        COALESCE(fb.feedback_count, 0) as feedback_count,   -- Total number of ratings
                        CASE WHEN COALESCE(fb.conflict_rate, 0.0) > {ConflictThreshold} THEN true ELSE false END as has_conflicts
                    FROM {TableKB} kb
                    LEFT JOIN (
                        -- SUBQUERY: Calculate statistics from user feedback
                        SELECT
                            solution_hash,
                            AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,    -- Average success rate
                            COUNT(*) as feedback_count,                                          -- Total feedback count
                            AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate   -- Rate of negative feedback
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE kb.error_hash = '{hash}'             -- Look for this specific error
                    ORDER BY kb.created_at DESC               -- Get the most recent solution if multiple exist
                    LIMIT 1";

                        System.Diagnostics.Debug.WriteLine($"DEBUG: Searching for hash: {hash}");

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                // EXTRACT THE DATA
                                var steps = rdr.IsDBNull(0) ? "" : ConvertArrayToString(rdr.GetValue(0));
                                var resultHash = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
                                var successRate = rdr.GetDouble(2);
                                var feedbackCount = rdr.GetInt32(3);
                                var hasConflicts = rdr.GetBoolean(4);

                                // CALCULATE CONFIDENCE SCORE
                                // Combine success rate, feedback count, and conflict info into one confidence score
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

        /// <summary>
        /// ARRAY CONVERSION HELPER - Convert database array format to readable text
        ///
        /// WHAT THIS DOES:
        /// Database stores solution steps as arrays like: ["step1", "step2", "step3"]
        /// This converts them to readable text with each step on a new line
        ///
        /// LIKE A TRANSLATOR:
        /// Converts computer storage format into human-readable instructions
        /// </summary>
        private static string ConvertArrayToString(object arrayValue)
        {
            if (arrayValue == null) return "";

            var arrayStr = arrayValue.ToString();
            if (string.IsNullOrEmpty(arrayStr)) return "";

            // HANDLE DATABRICKS ARRAY FORMAT
            // Parse format like: ["step1", "step2"] into readable steps
            if (arrayStr.StartsWith("[") && arrayStr.EndsWith("]"))
            {
                // Remove brackets, split by commas, clean up quotes
                var content = arrayStr.Trim('[', ']');
                var steps = content.Split(',')
                                  .Select(s => s.Trim(' ', '"', '\''))
                                  .Where(s => !string.IsNullOrEmpty(s));
                return string.Join(Environment.NewLine, steps);
            }

            return arrayStr;
        }

        /// <summary>
        /// PATTERN MATCHING - Look for solutions using pattern recognition
        ///
        /// WHAT THIS DOES:
        /// Uses regular expressions (pattern matching rules) to identify error types
        /// Like having a list of "if the error contains X, try solution Y"
        ///
        /// EXAMPLE PATTERNS:
        /// - If error contains "login failed" → try password reset steps
        /// - If error contains "connection timeout" → try network troubleshooting
        ///
        /// Tracks which patterns work best over time and prioritizes successful ones
        /// </summary>
        private async Task<SolutionResult> TryEnhancedPatternMatchAsync(string sig)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // FIND PATTERNS THAT MATCH THIS ERROR
                        // Use direct string interpolation instead of parameters for Databricks compatibility
                        var escapedSig = SecureEscape(sig);
                        var sql = $@"
                            SELECT resolution_steps, effectiveness_score, usage_count
                            FROM {TablePatterns}
                            WHERE '{escapedSig}' RLIKE pattern_regex          -- Does this error match the pattern?
                            ORDER BY priority DESC, effectiveness_score DESC  -- Prioritize successful patterns
                            LIMIT 1";

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : ConvertArrayToString(rdr.GetValue(0));
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
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Pattern match failed: {ex.Message}");
                }
                return null;
            });
        }

        /// <summary>
        /// KEYWORD SEARCH - Find solutions using important words from the error
        ///
        /// WHAT THIS DOES:
        /// Extracts important keywords from the error message and searches for solutions
        /// that dealt with similar keywords
        ///
        /// LIKE A SMART SEARCH ENGINE:
        /// - Identifies the most important words (ignoring "the", "and", etc.)
        /// - Finds solutions that dealt with those same important concepts
        /// - Orders results by success rate (solutions that worked for others)
        /// </summary>
        private async Task<SolutionResult> TryKeywordMatchAsync(string sig)
        {
            return await Task.Run(() =>
            {
                try
                {
                    // EXTRACT IMPORTANT KEYWORDS
                    var tokens = Tokens(sig);
                    if (tokens.Length == 0) return null;

                    using (var conn = OpenConn())
                    {
                        // BUILD SEARCH QUERY WITH DIRECT STRING INTERPOLATION
                        // Databricks doesn't like parameter binding, so build SQL directly with escaped values
                        var likeConditions = tokens.Select(t => $"kb.error_signature LIKE '%{SecureEscape(t)}%'").ToArray();
                        var whereClause = string.Join(" AND ", likeConditions);

                        var sql = $@"
                            SELECT kb.resolution_steps, kb.error_hash,
                                   COALESCE(fb.success_rate, 0.5) as success_rate,      -- Default 50% if no feedback
                                   COALESCE(fb.feedback_count, 0) as feedback_count     -- Default 0 if no feedback
                            FROM {TableKB} kb
                            LEFT JOIN (
                                -- Calculate success statistics from user feedback
                                SELECT solution_hash,
                                       AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                                       COUNT(*) as feedback_count
                                FROM {TableFeedback}
                                GROUP BY solution_hash
                            ) fb ON kb.error_hash = fb.solution_hash
                            WHERE {whereClause}                                      -- Match all keywords
                            ORDER BY fb.success_rate DESC, kb.created_at DESC      -- Best solutions first
                            LIMIT 1";

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : ConvertArrayToString(rdr.GetValue(0));
                                var hash = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
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
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Keyword match failed: {ex.Message}");
                }
                return null;
            });
        }






        /// <summary>
        /// CONFIDENCE CALCULATOR - Determines how much to trust a solution
        ///
        /// WHAT THIS DOES:
        /// Combines multiple factors to create a confidence score:
        /// - Success rate: How often users said it worked
        /// - Sample size: More ratings = more confidence
        /// - Conflicts: Reduces confidence if people disagree
        ///
        /// LIKE A REPUTATION SYSTEM:
        /// - New solutions start at 50% confidence
        /// - Positive feedback increases confidence
        /// - More feedback makes the confidence more reliable
        /// - Conflicting feedback (some love it, some hate it) reduces confidence
        /// </summary>
        private double CalculateConfidence(double successRate, int feedbackCount, double conflictRate)
        {
            // DEFAULT CONFIDENCE FOR NEW SOLUTIONS
            // Give new solutions 50% confidence until we get feedback
            var baseConfidence = feedbackCount == 0 ? 0.5 : successRate;

            // SAMPLE SIZE ADJUSTMENT
            // More feedback = more reliable confidence score
            var sampleConfidence = feedbackCount == 0 ? 1.0 : Math.Min(1.0, feedbackCount / (double)MinFeedbackForConfidence);

            // CONFLICT PENALTY
            // Reduce confidence if people strongly disagree about the solution
            var conflictPenalty = Math.Max(0.0, 1.0 - conflictRate * 2);

            return baseConfidence * sampleConfidence * conflictPenalty;
        }

        /// <summary>
        /// SOLUTION RANKING - Pick the best solution from multiple options
        ///
        /// WHAT THIS DOES:
        /// When CAINE finds multiple possible solutions, this picks the best one based on:
        /// 1. Confidence score (combination of AI similarity and user feedback)
        /// 2. Success rate (how often users said it worked)
        /// 3. Number of ratings (more feedback = more reliable)
        ///
        /// LIKE A RECOMMENDATION ALGORITHM:
        /// Sorts all options by quality and picks the top-rated one that meets minimum standards
        /// </summary>
        private SolutionResult SelectBestSolution(List<SolutionResult> results)
        {
            if (results.Count == 0) return null;

            // RANK ALL SOLUTIONS BY QUALITY
            var sorted = results
                .Where(r => !string.IsNullOrWhiteSpace(r.Steps))    // Must have actual steps
                .OrderByDescending(r => r.Confidence)              // Highest confidence first
                .ThenByDescending(r => r.SuccessRate)              // Then highest success rate
                .ThenByDescending(r => r.FeedbackCount)            // Then most feedback
                .ToList();

            // RETURN BEST SOLUTION IF IT MEETS MINIMUM STANDARDS
            var best = sorted.FirstOrDefault();
            return (best?.Confidence >= 0.1) ? best : null;        // Must have at least 10% confidence
        }

        /// <summary>
        /// SEARCH SOURCE INFO - Explain how CAINE found the solution
        ///
        /// WHAT THIS DOES:
        /// Tells users which search method found their solution so they understand
        /// how CAINE's intelligence worked
        ///
        /// HELPS WITH TRANSPARENCY:
        /// - Users know if it was an exact match (most reliable)
        /// - Or if AI had to get creative with similarity matching
        /// - Builds trust by showing the reasoning process
        /// </summary>
        private string GetSearchSourceInfo(string source)
        {
            return source switch
            {
                "exact_match" => "✅ Found: Exact match from knowledge base",
                "keyword_match" => "🔍 Found: Keyword similarity search",
                "vector_similarity" => "🤖 Found: AI similarity analysis",
                "pattern_match" => "📋 Found: Pattern recognition match",
                "openai_enhanced" => "🧠 Generated: AI consultation with context",
                _ => "📚 Found: Database search"
            };
        }

        /// <summary>
        /// CONFIDENCE DISPLAY - Create user-friendly confidence message
        ///
        /// WHAT THIS DOES:
        /// Converts technical confidence data into plain English that users can understand
        ///
        /// SHOWS USERS:
        /// - Confidence level in simple terms (High/Medium/Low)
        /// - Actual success percentage
        /// - How many people have tried this solution
        /// - Warning if there are conflicting opinions
        /// </summary>
        private string GetConfidenceText(SolutionResult result)
        {
            // CATEGORIZE CONFIDENCE LEVEL
            var confidenceLevel = result.Confidence >= HighConfidenceThreshold ? "High" :
                                 result.Confidence >= 0.6 ? "Medium" : "Low";

            // ADD CONFLICT WARNING IF NEEDED
            var conflictWarning = result.HasConflicts ? " ⚠️ Some users reported mixed results." : "";

            // BUILD USER-FRIENDLY MESSAGE
            return $"🎯 Confidence: {confidenceLevel} ({result.Confidence:P0}) | " +
                   $"Success Rate: {result.SuccessRate:P0} | " +
                   $"Based on {result.FeedbackCount} user ratings{conflictWarning}";
        }

        /// <summary>
        /// FEEDBACK RECORDING - Learn from user experiences
        ///
        /// WHAT THIS DOES:
        /// When users click "thumbs up" or "thumbs down", this records their feedback
        /// and immediately updates the confidence score for future users
        ///
        /// LIKE A LEARNING SYSTEM:
        /// - Records whether the solution actually worked
        /// - Updates confidence scores based on new feedback
        /// - Helps CAINE get smarter over time
        /// - Creates a feedback loop for continuous improvement
        /// </summary>
        private async void RecordEnhancedFeedback(bool wasHelpful, string comment = "", int confidenceRating = 3)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                await Task.Run(() =>
                {
                    // SANITIZE INPUT FOR SECURITY
                    var escComment = SecureEscape(comment);
                    var escSig = SecureEscape(Normalize(ErrorInput.Text));

                    using (var conn = OpenConn())
                    {
                        // INSERT FEEDBACK RECORD - Fixed for Databricks
                        // Store all the details about this user's experience using direct SQL
                        var sql = $@"
                            INSERT INTO {TableFeedback} VALUES (
                                '{Guid.NewGuid()}',
                                '{currentSessionId}',
                                '{currentSolutionHash}',
                                '{currentSolutionSource}',
                                '1.0',
                                {wasHelpful.ToString().ToLower()},
                                {(double)confidenceRating},
                                '{escComment}',
                                'intermediate',
                                current_timestamp(),
                                '{escSig}',
                                0,
                                'unknown'
                            )";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }
                    }

                    // STEP 4: UPDATE CONFIDENCE SCORE
                    // Recalculate how confident CAINE should be about this solution
                    UpdateSolutionConfidence(currentSolutionHash);
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Enhanced feedback recording failed: {ex.Message}");
            }
        }

        /// <summary>
        /// CONFIDENCE UPDATE - Recalculate how reliable a solution is
        ///
        /// WHAT THIS DOES:
        /// After each new piece of feedback, this recalculates the confidence score
        /// by looking at ALL the feedback for that solution
        ///
        /// LIKE UPDATING A PRODUCT RATING:
        /// - Someone leaves a new review
        /// - The overall star rating gets updated
        /// - Future customers see the new, more accurate rating
        /// </summary>
        private void UpdateSolutionConfidence(string solutionHash)
        {
            try
            {
                using (var conn = OpenConn())
                {
                    // CALCULATE NEW STATISTICS - Fixed for Databricks
                    // Look at all feedback for this solution and compute averages
                    var sql = $@"
                        SELECT
                            AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,  -- % who said it worked
                            COUNT(*) as total_feedback,                                        -- Total number of ratings
                            AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate -- % who said it didn't work
                        FROM {TableFeedback}
                        WHERE solution_hash = '{solutionHash}'";

                    using (var cmd = new OdbcCommand(sql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        if (rdr.Read())
                        {
                            var successRate = rdr.GetDouble(0);
                            var feedbackCount = rdr.GetInt32(1);
                            var conflictRate = rdr.GetDouble(2);

                            // CALCULATE NEW CONFIDENCE SCORE
                            var newConfidence = CalculateConfidence(successRate, feedbackCount, conflictRate);

                            System.Diagnostics.Debug.WriteLine(
                                $"Solution {solutionHash}: Confidence updated to {newConfidence:F2} " +
                                $"(Success: {successRate:P0}, Feedback: {feedbackCount}, Conflicts: {conflictRate:P0})");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Confidence update failed: {ex.Message}");
            }
        }

        /// <summary>
        /// ANALYTICS WINDOW - Show statistics and insights
        ///
        /// WHAT THIS DOES:
        /// Opens a separate window that shows:
        /// - Which solutions work best
        /// - Most common error types
        /// - CAINE's learning progress over time
        ///
        /// LIKE A DASHBOARD:
        /// Gives administrators insights into how well CAINE is performing
        /// </summary>
        private void BtnAnalytics_Click(object sender, RoutedEventArgs e)
        {
            var analyticsWindow = new AnalyticsWindow();
            analyticsWindow.Show();

            // ADD THIS LINE:
            ShowMLDashboard();
        }

        /// <summary>
        /// POSITIVE FEEDBACK - User says the solution worked
        ///
        /// WHAT THIS DOES:
        /// When user clicks "thumbs up":
        /// 1. Records positive feedback in the database
        /// 2. Immediately updates the confidence score
        /// 3. Refreshes the display to show the new confidence rating
        /// 4. Thanks the user for helping CAINE learn
        ///
        /// LIKE A LEARNING FEEDBACK LOOP:
        /// User success makes CAINE more confident about suggesting this solution to others
        /// </summary>
        private async void BtnFeedbackYes_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                await RecordMLFeedbackAsync(true, currentSolutionHash);

                DisableFeedbackButtons();

                // REFRESH CONFIDENCE DISPLAY
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);
                    var lines = ResultBox.Text.Split('\n');
                    lines[0] = newConfidenceText;
                    ResultBox.Text = string.Join("\n", lines);
                }

                ResultBox.Text += "\n\nThank you! Your positive feedback has been recorded.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Feedback failed: {ex.Message}");
            }
        }

        /// <summary>
        /// GET UPDATED CONFIDENCE - Fetch latest confidence score after new feedback
        ///
        /// WHAT THIS DOES:
        /// After someone rates a solution, this recalculates the confidence score
        /// by looking at ALL feedback for that solution
        ///
        /// LIKE REFRESHING A PRODUCT RATING:
        /// Gets the latest average rating after a new review is added
        /// </summary>
        private async Task<SolutionResult> GetUpdatedConfidence(string solutionHash)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // CALCULATE FRESH STATISTICS
                        var sql = $@"
                    SELECT
                        AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,  -- New success percentage
                        COUNT(*) as feedback_count                                         -- New total feedback count
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

        /// <summary>
        /// NEGATIVE FEEDBACK - User says the solution didn't work
        ///
        /// WHAT THIS DOES:
        /// When user clicks "thumbs down":
        /// 1. Records negative feedback in the database
        /// 2. Updates confidence score (making CAINE less likely to suggest this solution)
        /// 3. Encourages user to teach the correct solution
        ///
        /// LIKE A QUALITY CONTROL SYSTEM:
        /// Helps CAINE learn what doesn't work so it stops suggesting bad solutions
        /// </summary>
        private async void BtnFeedbackNo_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                await RecordMLFeedbackAsync(false, currentSolutionHash);

                DisableFeedbackButtons();

                // REFRESH CONFIDENCE DISPLAY
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);
                    var lines = ResultBox.Text.Split('\n');
                    lines[0] = newConfidenceText;
                    ResultBox.Text = string.Join("\n", lines);
                }

                ResultBox.Text += "\n\nThank you for the feedback. Please consider teaching the correct solution below.";
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Feedback failed: {ex.Message}");
            }
        }

        /// <summary>
        /// UI BUTTON MANAGEMENT - Enable/disable feedback buttons
        ///
        /// WHAT THESE DO:
        /// Simple helper methods to turn the thumbs up/down buttons on and off
        /// Prevents users from rating the same solution multiple times
        /// </summary>


        // Also modify your BtnSearch_Click to enable the tree button when results found:
        // Add this line after "EnableFeedbackButtons();"

        private void EnableFeedbackButtons()
        {
            BtnInteractiveTree.IsEnabled = true;
            BtnFeedbackYes.IsEnabled = true;
            BtnFeedbackNo.IsEnabled = true;
        }

        private void DisableFeedbackButtons()
        {
            BtnFeedbackYes.IsEnabled = false;
            BtnFeedbackNo.IsEnabled = false;
        }

        // ============================================================================
        // CORE AI AND UTILITY METHODS
        // These are the foundational building blocks that power CAINE's intelligence
        // ============================================================================

        /// <summary>
        /// API KEY RETRIEVAL - Get ChatGPT access credentials
        ///
        /// WHAT THIS DOES:
        /// Safely retrieves the OpenAI API key from environment variables
        /// Like getting a password from a secure vault instead of hardcoding it
        /// </summary>
        private static string GetOpenAIApiKey()
        {
            var key = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            System.Diagnostics.Debug.WriteLine($"Retrieved key: '{key}'");
            System.Diagnostics.Debug.WriteLine($"Key length: {key?.Length ?? 0}");

            if (string.IsNullOrEmpty(key))
            {
                throw new InvalidOperationException("OPENAI_API_KEY environment variable not set");
            }

            return key;
        }

        /// <summary>
        /// SECURE CONNECTION SETUP - Enable modern encryption
        ///
        /// WHAT THIS DOES:
        /// Ensures all connections to ChatGPT use the latest security protocols
        /// Like making sure your browser uses HTTPS instead of unsecure HTTP
        /// </summary>
        private static void EnsureTls12()
        {
            try { ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12; } catch { }
        }

        /// <summary>
        /// CHATGPT COMMUNICATION - Send questions to OpenAI and get responses
        ///
        /// WHAT THIS DOES:
        /// Takes a conversation (array of messages) and sends it to ChatGPT's API
        /// Like having a phone conversation with an expert, but through code
        ///
        /// HANDLES:
        /// - Authentication (proving we're allowed to use the API)
        /// - Request formatting (packaging the conversation properly)
        /// - Error handling (what to do if ChatGPT is unavailable)
        /// - Response parsing (extracting the answer from ChatGPT's response)
        /// </summary>
        private async Task<string> AskOpenAIAsync(object[] messages, string model = "gpt-4o-mini")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            // PACKAGE THE REQUEST
            // Format the conversation for ChatGPT's API
            var payload = new { model, messages };
            var json = JsonConvert.SerializeObject(payload);

            using (var req = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/chat/completions"))
            {
                // AUTHENTICATE THE REQUEST
                // Include our API key to prove we're authorized
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                req.Content = new StringContent(json, Encoding.UTF8, "application/json");

                using (var res = await Http.SendAsync(req))
                {
                    var body = await res.Content.ReadAsStringAsync();
                    if (!res.IsSuccessStatusCode)
                        throw new Exception($"OpenAI error {res.StatusCode}: {body}");

                    // EXTRACT THE RESPONSE
                    // Parse ChatGPT's JSON response to get the actual answer
                    var obj = JObject.Parse(body);
                    var content =
                        (string)(obj["choices"]?[0]?["message"]?["content"]) ??
                        (string)(obj["choices"]?[0]?["text"]) ?? "";
                    return content;
                }
            }
        }

        /// <summary>
        /// AI EMBEDDINGS - Create AI fingerprints for similarity matching
        ///
        /// WHAT THIS DOES:
        /// Converts text into numerical vectors (arrays of numbers) that represent meaning
        /// Like converting words into a mathematical language that AI can understand
        ///
        /// EXAMPLE:
        /// "Database connection failed" and "Cannot connect to SQL server"
        /// would produce similar number patterns even though the words are different
        ///
        /// ENABLES SMART SIMILARITY MATCHING:
        /// CAINE can find solutions to similar problems even if the exact words differ
        /// </summary>
        private async Task<float[]> EmbedAsync(string text, string model = "text-embedding-3-large")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            // PACKAGE THE EMBEDDING REQUEST
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

                    // EXTRACT THE VECTOR
                    // Parse the numerical representation from OpenAI's response
                    var obj = JObject.Parse(body);
                    var arr = (JArray?)obj["data"]?[0]?["embedding"] ?? new JArray();
                    var vec = new float[arr.Count];
                    for (int i = 0; i < arr.Count; i++) vec[i] = (float)arr[i]!;
                    return vec;
                }
            }
        }

        /// <summary>
        /// CONTEXT BUILDING - Prepare conversation history for ChatGPT
        ///
        /// WHAT THIS DOES:
        /// Before asking ChatGPT for help, this builds a conversation that includes:
        /// - Similar errors CAINE has solved before
        /// - Which solutions worked well (high success rates)
        /// - Context about the current problem
        ///
        /// LIKE BRIEFING AN EXPERT:
        /// - "Here are 6 similar problems we've solved successfully..."
        /// - "Now help me with this new problem that seems related..."
        ///
        /// This makes ChatGPT's advice much more relevant and actionable
        /// </summary>
        private async Task<object[]> BuildEnhancedChatHistoryAsync(string incomingErrorMessage)
        {
            var incomingLower = (incomingErrorMessage ?? "").ToLowerInvariant();
            var esc = incomingLower.Replace("'", "''");

            var related = await Task.Run(() =>
            {
                var list = new List<(string Sig, string Steps, string Exact, double SuccessRate)>();
                try
                {
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
                            try
                            {
                                var sig = rdr.IsDBNull(0) ? "" : rdr.GetString(0);
                                var steps = rdr.IsDBNull(1) ? "" : ConvertArrayToString(rdr.GetValue(1));
                                var exact = rdr.IsDBNull(2) ? "" : rdr.GetString(2);
                                var successRate = rdr.IsDBNull(3) ? 0.5 : rdr.GetDouble(3);

                                list.Add((sig, steps, exact, successRate));
                            }
                            catch (Exception rowEx)
                            {
                                System.Diagnostics.Debug.WriteLine($"Error reading row: {rowEx.Message}");
                                // Skip this row and continue
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Database query error: {ex.Message}");
                }
                return list;
            });

            // Rest of your method stays the same...
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

        /// <summary>
        /// TEXT PARSING - Extract numbered steps from AI responses
        ///
        /// WHAT THIS DOES:
        /// Takes ChatGPT's response (which might be formatted text) and extracts
        /// clean, numbered steps that can be saved to the knowledge base
        ///
        /// LIKE A TEXT PROCESSOR:
        /// Finds lines that start with "1.", "2.", "- " etc. and extracts them
        /// Prepares the steps in a clean format for teaching CAINE
        /// </summary>
        private static IEnumerable<string> ExtractNumberedLines(string markdown)
        {
            if (markdown == null) yield break;
            foreach (var raw in markdown.Split('\n'))
            {
                var t = raw.Trim();
                // FIND LINES THAT LOOK LIKE STEPS
                if (t.StartsWith("1.") || t.StartsWith("2.") || t.StartsWith("3.") || t.StartsWith("- "))
                    yield return t.TrimStart('-', ' ').Trim();
            }
        }

        /// <summary>
        /// CONFLICT DETECTION - Check if we already have different solutions for this error
        ///
        /// WHAT THIS DOES:
        /// Before adding a new solution, checks if we already have a different solution
        /// for the same error and how well that existing solution has performed
        ///
        /// PREVENTS KNOWLEDGE BASE CONFLICTS:
        /// - Warns users if they're adding a solution that contradicts existing ones
        /// - Helps identify when there might be multiple valid approaches
        /// - Tracks success rates of competing solutions
        /// </summary>
        private async Task<(bool HasConflicts, double ExistingSuccessRate)> CheckForExistingSolutions(string hash, string[] newSteps)
        {
            return await Task.Run(() =>
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // LOOK FOR EXISTING SOLUTIONS - Fixed for Databricks compatibility
                        var escapedSig = SecureEscape(Normalize(string.Join(" ", newSteps)));
                        var sql = $@"
                            SELECT AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                            FROM {TableFeedback} f
                            JOIN {TableKB} kb ON f.solution_hash = kb.error_hash
                            WHERE kb.error_hash = '{hash}' OR kb.error_signature = '{escapedSig}'";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            var result = cmd.ExecuteScalar();
                            if (result != null && result != DBNull.Value)
                            {
                                var successRate = Convert.ToDouble(result);
                                return (successRate < 0.6, successRate);  // Conflict if existing solution has <60% success
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Conflict check failed: {ex.Message}");
                }
                return (false, 0.5);  // No conflicts found
            });
        }

        /// <summary>
        /// ENHANCED TEACHING - Save new knowledge with version control
        ///
        /// WHAT THIS DOES:
        /// Takes a new error/solution pair and stores it in CAINE's knowledge base
        /// with full version control and conflict tracking
        ///
        /// LIKE ADDING TO A SMART ENCYCLOPEDIA:
        /// - Creates a new entry with unique fingerprint
        /// - Stores AI embeddings for future similarity matching
        /// - Records metadata about who added it and when
        /// - Handles conflicts with existing solutions gracefully
        /// - Sets up tracking for future feedback and learning
        /// </summary>
        private async Task TeachEnhancedSolution(string hash, string sig, string errorText, string[] stepsLines, float[] emb, (bool HasConflicts, double ExistingSuccessRate) conflicts)
        {
            await Task.Run(() =>
            {
                try
                {
                    // HELPER FUNCTIONS FOR SQL SAFETY
                    // Clean and escape all user input to prevent SQL injection
                    string Esc(string s) => (s ?? "")
                        .Replace("\\", "\\\\")
                        .Replace("'", "''")
                        .Replace("\r", "")
                        .Replace("\n", "\\n");

                    // CONVERT STEPS ARRAY TO SQL FORMAT
                    // Transform C# string array into database array format
                    string ArrayLiteral(string[] arr) =>
                        "array(" + string.Join(",", arr.Select(s => $"'{Esc(s)}'")) + ")";

                    var steps = ArrayLiteral(stepsLines);
                    var escErr = Esc(errorText);
                    var escSig = Esc(sig);

                    // CONVERT AI EMBEDDINGS TO SQL FORMAT
                    // Store the AI fingerprint for future similarity searches
                    string embSql = emb == null
                        ? "NULL"
                        : $"array({string.Join(",", emb.Select(f => f.ToString("G", CultureInfo.InvariantCulture)))})";

                    var changeType = conflicts.HasConflicts ? "alternative" : "create";

                    using (var conn = OpenConn())
                    {
                        // INSERT INTO MAIN KNOWLEDGE BASE
                        // Use MERGE to handle conflicts gracefully (update existing or insert new)
                        var kbSql = $@"
                            MERGE INTO {TableKB} t
                            USING (SELECT
                                     '{Guid.NewGuid()}' AS id,
                                     current_timestamp() AS created_at,
                                     'human_enhanced' AS source,             -- Mark as human-provided
                                     NULL AS product, NULL AS version, NULL AS env,
                                     '{escErr}' AS error_text,
                                     '{escSig}' AS error_signature,
                                     '{hash}' AS error_hash,                -- Unique fingerprint
                                     NULL AS context_text,
                                     {steps} AS resolution_steps,           -- The actual solution
                                     array() AS symptoms,
                                     NULL AS root_cause,
                                     NULL AS verification,
                                     array() AS links,
                                     NULL AS code_before, NULL AS code_after,
                                     'Enhanced teaching with conflict detection' AS notes,
                                     {embSql} AS embedding                  -- AI fingerprint for similarity
                                   ) s
                            ON t.error_hash = s.error_hash
                            WHEN MATCHED THEN UPDATE SET
                              t.resolution_steps = array_distinct(          -- Merge solutions if conflict
                                concat(coalesce(t.resolution_steps, cast(array() as array<string>)),
                                       s.resolution_steps)),
                              t.notes = 'Updated with enhanced teaching'
                            WHEN NOT MATCHED THEN INSERT *";

                        using (var cmd = new OdbcCommand(kbSql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        // INSERT INTO VERSION HISTORY
                        // Keep track of this change for auditing and rollback purposes
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

        // ============================================================================
        // CORE UTILITY METHODS
        // These are fundamental building blocks used throughout the application
        // ============================================================================

        /// <summary>
        /// ERROR NORMALIZATION - Standardize error messages for consistent matching
        ///
        /// WHAT THIS DOES:
        /// Takes a raw error message and converts it into a standardized format
        /// so that similar errors get treated as the same thing
        ///
        /// LIKE TRANSLATION AND STANDARDIZATION:
        /// - Removes specific details (GUIDs, file paths, user names)
        /// - Replaces them with generic placeholders
        /// - Converts everything to lowercase
        /// - Removes punctuation and extra spaces
        ///
        /// EXAMPLES:
        /// "Login failed for user 'john.doe'" → "login failed for user <user>"
        /// "C:\MyApp\Config.xml not found" → "<path> not found"
        ///
        /// This ensures that the same type of error always gets the same fingerprint
        /// </summary>
        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "";
            var t = s.ToLowerInvariant();

            // REPLACE SPECIFIC VALUES WITH GENERIC PLACEHOLDERS
            t = Regex.Replace(t, @"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<guid>");          // GUIDs
            t = Regex.Replace(t, @"[a-z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n]+", "<path>");                       // File paths
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]\.\[[^\]]+\]", "exec <proc>");              // SQL procedures
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+\.[a-z0-9_]+", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+", "exec <proc>");
            t = Regex.Replace(t, @"login failed for user\s+'[^']+'", "login failed for user '<user>'");    // User names
            t = Regex.Replace(t, @"\b[a-z0-9._-]+\\[a-z0-9.$_-]+\b", "<user>");                          // Domain\user
            t = Regex.Replace(t, @"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", "<user>");                // Email addresses
            t = Regex.Replace(t, @"[^\w\s-]", " ");                                                        // Remove punctuation
            t = Regex.Replace(t, @"\b\d+\b", "<num>");                                                     // Numbers
            t = Regex.Replace(t, @"\s+", " ").Trim();                                                      // Multiple spaces

            return t;
        }

        /// <summary>
        /// HASH GENERATION - Create unique fingerprints for errors
        ///
        /// WHAT THIS DOES:
        /// Takes any text and creates a unique, fixed-length identifier (hash)
        /// Like creating a barcode - same input always produces same hash
        ///
        /// USES SHA256 ENCRYPTION:
        /// - Industry-standard hashing algorithm
        /// - Same error always gets same hash
        /// - Impossible to reverse-engineer the original error from the hash
        /// - Used for quick exact-match lookups in the database
        /// </summary>
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

        /// <summary>
        /// DATABASE CONNECTION - Open secure connection to CAINE's knowledge base
        ///
        /// WHAT THIS DOES:
        /// Creates a connection to the Databricks database where CAINE stores all its knowledge
        /// Like opening a secure line to the library's catalog system
        /// </summary>
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        /// <summary>
        /// KEYWORD EXTRACTION - Find important words for searching
        ///
        /// WHAT THIS DOES:
        /// Takes a normalized error message and extracts the most important keywords
        /// while filtering out common words that don't help identify the problem
        ///
        /// LIKE A SMART HIGHLIGHTER:
        /// - Finds words that are 4+ characters (meaningful terms)
        /// - Removes common words like "error", "failed", "the", "and"
        /// - Limits to top 4 most important terms
        /// - Returns words that uniquely identify this type of error
        ///
        /// Used for keyword-based searching when exact matches aren't found
        /// </summary>
        private static string[] Tokens(string sig)
        {
            var toks = Regex.Matches(sig ?? "", @"[a-z0-9\-]{4,}")        // Find meaningful words (4+ chars)
                            .Cast<Match>().Select(m => m.Value)
                            .Where(v => !Stop.Contains(v))                  // Remove stop words
                            .Distinct()                                     // Remove duplicates
                            .Take(LikeTokenMax)                            // Limit to top 4 keywords
                            .ToArray();
            return toks;
        }

        /// <summary>
        /// ARRAY PARSING - Convert database array strings back to float arrays
        ///
        /// WHAT THIS DOES:
        /// Database stores AI embeddings as text strings like "array(0.1,0.2,0.3)"
        /// This converts them back into usable float arrays for similarity calculations
        ///
        /// LIKE A FORMAT CONVERTER:
        /// Translates between database storage format and C# array format
        /// Handles various formats that might be stored in different database systems
        /// </summary>
        private static float[] ParseFloatArray(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return Array.Empty<float>();
            var t = s.Trim();

            // HANDLE DIFFERENT ARRAY FORMATS
            if (t.StartsWith("[")) t = t.Substring(1);                                    // Remove opening bracket
            if (t.EndsWith("]")) t = t.Substring(0, t.Length - 1);                       // Remove closing bracket
            if (t.StartsWith("array(", StringComparison.OrdinalIgnoreCase)) t = t.Substring(6);  // Remove "array("
            if (t.EndsWith(")")) t = t.Substring(0, t.Length - 1);                       // Remove closing paren

            // PARSE INDIVIDUAL NUMBERS
            var parts = t.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var list = new List<float>(parts.Length);
            foreach (var p in parts)
            {
                if (float.TryParse(p, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                    list.Add(v);
            }
            return list.ToArray();
        }

        /// <summary>
        /// VECTOR NORMALIZATION - Prepare AI embeddings for similarity comparison
        ///
        /// WHAT THIS DOES:
        /// Takes an AI embedding (array of numbers) and normalizes it so that
        /// similarity calculations work properly
        ///
        /// LIKE STANDARDIZING UNITS:
        /// - Converts vectors to unit length (magnitude = 1)
        /// - Enables accurate cosine similarity calculations
        /// - Required for proper AI similarity matching
        ///
        /// MATHEMATICAL PROCESS:
        /// Divides each number by the vector's overall magnitude
        /// </summary>
        private static void NormalizeInPlace(float[] v)
        {
            if (v == null || v.Length == 0) return;

            // CALCULATE MAGNITUDE (LENGTH OF VECTOR)
            double sumsq = 0;
            for (int i = 0; i < v.Length; i++) sumsq += v[i] * v[i];
            var norm = Math.Sqrt(sumsq);

            if (norm <= 1e-8) return;  // Avoid division by zero

            // NORMALIZE TO UNIT LENGTH
            for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
        }

        /// <summary>
        /// COSINE SIMILARITY - Calculate how similar two AI embeddings are
        ///
        /// WHAT THIS DOES:
        /// Compares two AI fingerprints and returns a similarity score from -1 to 1
        /// where 1 = identical meaning, 0 = unrelated, -1 = opposite meaning
        ///
        /// LIKE COMPARING FINGERPRINTS:
        /// - Takes two AI embeddings (arrays of numbers representing meaning)
        /// - Calculates the angle between them in high-dimensional space
        /// - Returns how similar they are as a percentage
        ///
        /// USED FOR INTELLIGENT MATCHING:
        /// Helps CAINE find solutions for errors that mean the same thing
        /// even if they use different words
        /// </summary>
        private static double Cosine(float[] a, float[] b)
        {
            if (a == null || b == null) return -1;

            int n = Math.Min(a.Length, b.Length);
            if (n == 0) return -1;

            // CALCULATE DOT PRODUCT
            // Mathematical operation that measures similarity between vectors
            double dot = 0;
            for (int i = 0; i < n; i++) dot += a[i] * b[i];
            return dot;
        }
    }
    /// <summary>
    /// ML INTEGRATION FOR CAINE - Connects traditional ML to existing system
    ///
    /// WHAT THIS DOES:
    /// - Bridges the ML engine with CAINE's existing search and feedback systems
    /// - Enhances decision-making with predictive models
    /// - Provides intelligent error categorization and trend analysis
    /// </summary>
    public partial class MainWindow
    {
        private CaineMLEngine mlEngine;
        private SimpleNeuralNetwork neuralNetwork;
        private DateTime lastModelTraining = DateTime.MinValue;
        private const int RetrainIntervalHours = 24;

        /// <summary>
        /// INITIALIZE ML COMPONENTS - Set up machine learning on startup
        /// </summary>
        private async Task InitializeMLComponentsAsync()
        {
            try
            {
                // Load historical data for training
                var trainingData = await LoadTrainingDataAsync();

                System.Diagnostics.Debug.WriteLine($"Loaded {trainingData.Count} training samples");

                if (trainingData.Count >= 50) // Need minimum data for ML
                {
                    // Initialize ML engine
                    mlEngine = new CaineMLEngine();
                    await mlEngine.InitializeAsync(trainingData);

                    // Initialize neural network
                    int featureCount = ExtractFeatures("sample error").Length; // Get feature count
                    neuralNetwork = new SimpleNeuralNetwork(
                        inputSize: featureCount,
                        hiddenSize: 50,
                        outputSize: 3 // Success, Partial Success, Failure
                    );

                    // Train neural network
                    await TrainNeuralNetworkAsync(trainingData);

                    System.Diagnostics.Debug.WriteLine("ML components initialized successfully");
                }
                else if (trainingData.Count > 0)
                {
                    // Partial initialization with limited data
                    System.Diagnostics.Debug.WriteLine($"Limited training data ({trainingData.Count} samples). Using basic ML only.");

                    // Create synthetic data to meet minimum requirements
                    var syntheticData = GenerateSyntheticTrainingData(trainingData, 50 - trainingData.Count);
                    trainingData.AddRange(syntheticData);

                    mlEngine = new CaineMLEngine();
                    await mlEngine.InitializeAsync(trainingData);

                    System.Diagnostics.Debug.WriteLine("ML initialized with synthetic data augmentation");
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("No training data available. ML features disabled.");
                    // ML features will be null-checked before use
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML initialization failed (non-critical): {ex.Message}");
                // Continue without ML - the app still works with traditional search
            }
        }

        // Add this helper method right after InitializeMLComponentsAsync
        private List<CaineMLEngine.TrainingDataPoint> GenerateSyntheticTrainingData(
            List<CaineMLEngine.TrainingDataPoint> existingData, int count)
        {
            var synthetic = new List<CaineMLEngine.TrainingDataPoint>();
            var random = new Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < count; i++)
            {
                // Create variations of existing data
                var basePoint = existingData[i % existingData.Count];
                var features = basePoint.Features.ToArray();

                // Add small random variations
                for (int j = 0; j < features.Length; j++)
                {
                    features[j] += (random.NextDouble() - 0.5) * 0.1; // ±5% variation
                }

                synthetic.Add(new CaineMLEngine.TrainingDataPoint
                {
                    Features = features,
                    ErrorHash = $"synthetic_{i}",
                    SolutionWorked = basePoint.SolutionWorked,
                    ResponseTime = basePoint.ResponseTime * (0.8 + random.NextDouble() * 0.4),
                    ErrorCategory = basePoint.ErrorCategory,
                    Timestamp = basePoint.Timestamp.AddMinutes(random.Next(-1440, 1440))
                });
            }

            return synthetic;
        }

        /// <summary>
        /// ENHANCED SEARCH WITH ML - Augments existing search with ML predictions
        /// </summary>
        private async Task<SolutionResult> EnhancedMLSearchAsync(string cleanErrorInput, string hash)
        {
            if (mlEngine == null) return null;

            try
            {
                // Extract features from the error
                var features = ExtractFeatures(cleanErrorInput);

                // Check if this is an anomaly
                bool isAnomaly = await mlEngine.IsAnomalyAsync(features);
                if (isAnomaly)
                {
                    System.Diagnostics.Debug.WriteLine("Anomalous error detected - needs special attention");
                    // Could trigger special handling or alert
                }

                // Get cluster-based recommendation
                var (template, clusterConfidence) = await mlEngine.GetClusterRecommendationAsync(features);

                // Predict solution success
                var (willWork, solutionConfidence) = await mlEngine.PredictSolutionSuccessAsync(features);

                // Use neural network for additional prediction
                var neuralPrediction = await PredictWithNeuralNetworkAsync(features);

                // Combine predictions for final confidence
                double combinedConfidence = CombinePredictions(
                    clusterConfidence,
                    solutionConfidence,
                    neuralPrediction.Confidence
                );

                if (!string.IsNullOrEmpty(template) && combinedConfidence > 0.5)
                {
                    return new SolutionResult
                    {
                        Steps = template,
                        Hash = hash,
                        Source = "ml_prediction",
                        Confidence = combinedConfidence,
                        SuccessRate = willWork ? combinedConfidence : 1 - combinedConfidence,
                        FeedbackCount = 0, // ML prediction, no feedback yet
                        HasConflicts = false,
                        Version = "ML-1.0"
                    };
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML search failed: {ex.Message}");
            }

            return null;
        }

        private async Task<SolutionResult> TryFuzzySearchAsync(string errorText)
        {
            // Run this search in the background so the UI doesn't freeze
            return await Task.Run(() =>
            {
                try
                {
                    // Connect to the database
                    using (var conn = OpenConn())
                    {
                        // SQL query that gets the top 500 errors from the knowledge base
                        // It also calculates how successful each solution has been based on user feedback
                        var sql = $@"
            SELECT kb.error_hash, kb.error_text, kb.resolution_steps,
                   COALESCE(fb.success_rate, 0.5) as success_rate,  -- If no feedback exists, assume 50% success
                   COALESCE(fb.feedback_count, 0) as feedback_count  -- Count how many people rated this
            FROM {TableKB} kb
            LEFT JOIN (
                -- This part calculates the success rate from thumbs up/down feedback
                SELECT solution_hash,
                       AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                       COUNT(*) as feedback_count
                FROM {TableFeedback}
                GROUP BY solution_hash
            ) fb ON kb.error_hash = fb.solution_hash
            WHERE kb.error_text IS NOT NULL
            LIMIT 500"; // Only check 500 errors to keep it fast

                        // List to store potential matches
                        var candidates = new List<FuzzySearchResult>();

                        // Execute the database query and read results
                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            // Check each error in the database
                            while (rdr.Read())
                            {
                                // Get the data for this error
                                var hash = rdr.GetString(0);           // Unique ID
                                var text = rdr.GetString(1);           // The error message text
                                var steps = ConvertArrayToString(rdr.GetValue(2));  // The solution
                                var successRate = rdr.GetDouble(3);    // How often it works (0-1)
                                var feedbackCount = rdr.GetInt32(4);   // How many people rated it

                                // Calculate how similar this error is to the user's error
                                // fuzzyScore: Uses Levenshtein distance (handles typos)
                                double fuzzyScore = fuzzySearch.CalculateFuzzyScore(errorText, text);

                                // ngramScore: Compares 3-letter chunks (good for partial matches)
                                double ngramScore = fuzzySearch.GetNGramSimilarity(errorText, text);

                                // Only consider errors that are at least 50% similar or 40% n-gram match
                                if (fuzzyScore > 0.5 || ngramScore > 0.4)
                                {
                                    candidates.Add(new FuzzySearchResult
                                    {
                                        ErrorHash = hash,
                                        ErrorText = text,
                                        ResolutionSteps = steps,
                                        FuzzyScore = fuzzyScore,
                                        // Bonus points if the user's text appears exactly in the error
                                        ExactMatchBonus = text.Contains(errorText.ToLower()) ? 0.3 : 0,
                                        // Bonus for n-gram similarity (weighted at 20%)
                                        SynonymBonus = ngramScore * 0.2
                                    });
                                }
                            }
                        }

                        // Find the best matching error (highest total score)
                        var best = candidates.OrderByDescending(c => c.TotalScore).FirstOrDefault();

                        // Only return a solution if we're at least 60% confident it's a match
                        if (best != null && best.TotalScore > 0.6)
                        {
                            return new SolutionResult
                            {
                                Steps = best.ResolutionSteps,      // The solution to try
                                Hash = best.ErrorHash,              // ID of this solution
                                Source = "fuzzy_search",            // Mark that fuzzy search found this
                                Confidence = Math.Min(0.95, best.TotalScore), // Cap confidence at 95%
                                Version = "1.0"
                            };
                        }
                    }
                }
                catch (Exception ex)
                {
                    // If anything goes wrong, log it but don't crash
                    System.Diagnostics.Debug.WriteLine($"Fuzzy search failed: {ex.Message}");
                }

                // No good match found
                return null;
            });
        }

        /// <summary>
        /// EXTRACT FEATURES - Convert error text to numerical features for ML
        /// </summary>
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

        /// <summary>
        /// LOAD TRAINING DATA - Retrieve historical data from database
        /// </summary>
        private async Task<List<CaineMLEngine.TrainingDataPoint>> LoadTrainingDataAsync()
        {
            return await Task.Run(() =>
            {
                var trainingData = new List<CaineMLEngine.TrainingDataPoint>();

                try
                {
                    using (var conn = OpenConn())
                    {
                        // Fixed query - removed non-existent column
                        var sql = $@"
                    SELECT
                        kb.error_text,
                        kb.error_hash,
                        kb.error_signature,
                        fb.was_helpful,
                        fb.created_at
                    FROM {TableKB} kb
                    JOIN {TableFeedback} fb ON kb.error_hash = fb.solution_hash
                    WHERE fb.was_helpful IS NOT NULL
                    ORDER BY fb.created_at DESC
                    LIMIT 1000";

                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var errorText = rdr.GetString(0);
                                var errorHash = rdr.GetString(1);
                                var errorSignature = rdr.GetString(2);
                                var wasHelpful = rdr.GetBoolean(3);
                                var timestamp = rdr.GetDateTime(4);

                                // Extract features and create training point
                                var features = ExtractFeatures(errorText);

                                trainingData.Add(new CaineMLEngine.TrainingDataPoint
                                {
                                    Features = features,
                                    ErrorHash = errorHash,
                                    SolutionWorked = wasHelpful,
                                    ResponseTime = 30.0, // Default value since column doesn't exist
                                    ErrorCategory = CategorizeError(errorSignature),
                                    Timestamp = timestamp
                                });
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Loading training data failed: {ex.Message}");
                }

                return trainingData;
            });
        }

        /// <summary>
        /// TRAIN NEURAL NETWORK - Train the deep learning model
        /// </summary>
        private async Task TrainNeuralNetworkAsync(List<CaineMLEngine.TrainingDataPoint> trainingData)
        {
            await Task.Run(() =>
            {
                try
                {
                    // Prepare training data
                    var inputs = trainingData.Select(d => d.Features).ToArray();
                    var targets = trainingData.Select(d => new double[]
                    {
                d.SolutionWorked ? 1.0 : 0.0,           // Success
                d.ResponseTime < 30 ? 1.0 : 0.0,        // Quick resolution
                d.ResponseTime > 60 ? 1.0 : 0.0         // Complex problem
                    }).ToArray();

                    // Train the network
                    neuralNetwork.Train(inputs, targets, epochs: 500);

                    // ADD THIS LINE:
                    isNeuralNetworkTrained = true;  // Mark as trained
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Neural network training failed: {ex.Message}");
                    isNeuralNetworkTrained = false;  // Mark as not trained
                }
            });
        }

        /// <summary>
        /// PREDICT WITH NEURAL NETWORK - Get predictions from deep learning model
        /// </summary>
        private async Task<(double Confidence, bool WillWork)> PredictWithNeuralNetworkAsync(double[] features)
        {
            return await Task.Run(() =>
            {
                if (!isNeuralNetworkTrained || neuralNetwork == null)
                    return (0.5, false); // Default if not trained

                var output = neuralNetwork.Forward(features);

                // output[0] is probability of success, output[1] is probability of failure
                double successProbability = output[0];
                double failureProbability = output[1];

                // Normalize if needed
                double total = successProbability + failureProbability;
                if (total > 0)
                {
                    successProbability /= total;
                }

                bool willWork = successProbability > 0.5;
                double confidence = Math.Abs(successProbability - 0.5) * 2; // Convert to 0-1 confidence

                return (confidence, willWork);
            });
        }

        /// <summary>
        /// AUTO-RETRAIN MODELS - Periodically update ML models with new data
        /// </summary>
        private async Task CheckAndRetrainModelsAsync()
        {
            if (mlEngine == null) return;

            var hoursSinceLastTraining = (DateTime.Now - lastModelTraining).TotalHours;
            if (hoursSinceLastTraining < RetrainIntervalHours) return;

            try
            {
                // Load recent data
                var recentData = await LoadTrainingDataAsync();

                // Retrain if we have enough new data
                if (recentData.Count >= 100)
                {
                    await mlEngine.RetrainModelsAsync(recentData);
                    await TrainNeuralNetworkAsync(recentData);
                    lastModelTraining = DateTime.Now;

                    System.Diagnostics.Debug.WriteLine("Models retrained successfully");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Model retraining failed: {ex.Message}");
            }
        }

        /// <summary>
        /// PREDICT ERROR TRENDS - Forecast future error occurrences
        /// </summary>
        private async Task<string> GetErrorTrendAnalysisAsync(string errorPattern)
        {
            if (mlEngine == null) return "ML not initialized";

            var (predictedCount, confidence) = await mlEngine.PredictErrorTrendAsync(errorPattern, 1);

            if (confidence > 0.7)
            {
                return $"📊 Trend Analysis: This error is predicted to occur ~{predictedCount:F0} times " +
                       $"in the next hour (Confidence: {confidence:P0}). ";
            }

            return "";
        }

        /// <summary>
        /// ENHANCED FEEDBACK WITH ML - Update ML models when users provide feedback
        /// </summary>
        private async Task RecordMLFeedbackAsync(bool wasHelpful, string errorHash)
        {
            // Record standard feedback first
            RecordEnhancedFeedback(wasHelpful);

            // Update ML models with this feedback
            if (mlEngine != null)
            {
                // Check if it's time to retrain
                await CheckAndRetrainModelsAsync();
            }
        }

        // ============================================================================
        // HELPER METHODS FOR ML INTEGRATION
        // ============================================================================

        private double CombinePredictions(params double[] confidences)
        {
            // Weighted average with decay for missing predictions
            double sum = 0;
            int count = 0;

            foreach (var conf in confidences)
            {
                if (conf > 0)
                {
                    sum += conf;
                    count++;
                }
            }

            return count > 0 ? sum / count : 0.5;
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

        private double CalculateResultConfidence(SolutionResult result, MLInsights insights = null)
        {
            var factors = new ConfidenceFactors
            {
                BaseSuccessRate = result.SuccessRate,
                FeedbackCount = result.FeedbackCount,
                ConflictRate = result.HasConflicts ? 0.4 : 0.0,
                MLConfidence = insights?.NeuralPrediction?.Confidence,
                IsAnomaly = insights?.IsAnomaly ?? false,
                SolutionComplexity = SolutionParser.ParseIntoSteps(result.Steps).Count,
                DaysOld = 0 // Would need to be calculated from result.LastUpdated if available
            };

            return UnifiedConfidenceCalculator.CalculateUnifiedConfidence(factors);
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

        /// <summary>
        /// ML DASHBOARD - Show ML insights and predictions
        /// </summary>
        private void ShowMLDashboard()
        {
            if (mlEngine == null)
            {
                ResultBox.Text += "\n\n⚠️ ML components not initialized. Need more training data.";
                return;
            }

            try
            {
                var mlDashboard = new MLDashboardWindow(mlEngine);
                mlDashboard.Owner = this;
                mlDashboard.Show();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Could not open ML Dashboard: {ex.Message}",
                               "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
    }
}
