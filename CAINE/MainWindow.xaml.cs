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

namespace CAINE
{
    /// <summary>
    /// CAINE (Computer-Aided Intelligence for Network Errors) - Main Application Window
    /// 
    /// WHAT THIS DOES:
    /// - Acts like a smart assistant that remembers how to fix database and network errors
    /// - When you paste an error message, it searches for solutions it has seen before
    /// - If no solution exists, it asks ChatGPT for help and learns from the answer
    /// - Users can rate solutions as helpful or not, making the system smarter over time
    /// - Essentially creates a company knowledge base that gets better with each use
    /// 
    /// ============================================================================
    /// CAINE SYSTEM ARCHITECTURE - How Everything Works Together
    /// 
    /// THE BIG PICTURE:
    /// • Knowledge Base: Stores all error solutions like a library catalog
    /// • Search Engine: Finds relevant solutions using 4 different search methods
    /// • AI Integration: Asks ChatGPT when CAINE doesn't know the answer
    /// • Learning System: Gets smarter from user feedback over time
    /// 
    /// USER JOURNEY:
    /// [Error Input] → [Search Database] → Found? → [Show Solution + Confidence]
    ///                          ↓                            ↓
    ///                      Not Found?                [User Feedback]
    ///                          ↓                            ↓
    ///                   [Ask ChatGPT]              [Update Confidence]
    ///                          ↓
    ///                  [Show AI Solution]
    ///                          ↓
    ///                    [User Can Teach]
    /// 
    /// SUCCESS METRICS - How We Know CAINE Is Working:
    /// • Solutions with >80% success rate from user feedback
    /// • Reduced time to resolve errors (tracked in resolution_time_minutes)
    /// • Growing knowledge base with version control
    /// • Continuous learning from both successes and failures
    /// ============================================================================
    /// </summary>
    public partial class MainWindow : Window
    {
        // DATABASE CONFIGURATION - Where CAINE stores its knowledge
        // Think of these as different filing cabinets in a digital library
        private const string DsnName = "CAINE_Databricks";                    // Main database connection name
        private const string TableKB = "default.cai_error_kb";               // Main knowledge base - stores error solutions
        private const string TablePatterns = "default.cai_error_patterns";   // Pattern matching rules - like shortcuts for common errors
        private const string TableFeedback = "default.cai_solution_feedback"; // User ratings - tracks which solutions actually work
        private const string TableKBVersions = "default.cai_kb_versions";    // Version history - keeps track of changes over time

        // SMART SEARCH SETTINGS - How CAINE decides if solutions are good enough
        // These numbers control how picky CAINE is when suggesting solutions
        private const int VectorCandidateLimit = 300;        // Max number of similar errors to compare against
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
        private double currentSolutionConfidence = 0.0;                 // How confident CAINE is in this solution

        // HTTP CLIENT - For talking to ChatGPT API
        // Reused connection to avoid creating new connections every time
        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

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
        public class SolutionResult
        {
            public string Steps { get; set; }          // The actual solution instructions
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

        public MainWindow()
        {
            InitializeComponent();
            EnsureTls12();                                     // Set up secure connections to ChatGPT
            SecurityValidator.InitializeSecurityTables();     // Set up security monitoring
            InitializeEnhancedTables();

            // Use Loaded event for async initialization
            this.Loaded += async (sender, e) => await MainWindow_LoadedAsync();
        }

        private async Task MainWindow_LoadedAsync()
        {
            // Initialize ML components after window loads
            await InitializeMLComponentsAsync();
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
        /// MAIN SEARCH FUNCTION - The heart of CAINE's intelligence
        /// 
        /// WHAT THIS DOES WHEN USER CLICKS "SEARCH":
        /// 1. Checks if the input is safe (security validation)
        /// 2. Creates a unique fingerprint of the error message
        /// 3. Looks for exact matches in the knowledge base
        /// 4. If found, shows the solution with confidence rating
        /// 5. If not found, suggests using AI assistance
        /// 
        /// LIKE A SMART LIBRARIAN:
        /// - First checks if we've seen this exact book before
        /// - Shows how reliable our answer is based on past feedback
        /// - If we haven't seen it, suggests asking an expert (ChatGPT)
        /// </summary>
        private async void BtnSearch_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // STEP 1: SECURITY CHECK
                // Make sure the input isn't malicious before processing
                var validation = SecurityValidator.ValidateInput(ErrorInput.Text, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"Security validation failed: {validation.ErrorMessage}";
                    return;
                }

                // STEP 2: PREPARE THE UI
                // Disable buttons to prevent double-clicking while searching
                BtnCaineApi.IsEnabled = false;
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Searching with enhanced ML algorithms...";

                // STEP 3: PROCESS THE ERROR MESSAGE
                // Clean up the input and create a unique fingerprint (hash)
                var cleanErrorInput = validation.CleanInput;
                var sig = Normalize(cleanErrorInput);           // Standardize the error message
                var hash = Sha256Hex(sig);                     // Create unique fingerprint

                System.Diagnostics.Debug.WriteLine($"DEBUG: Generated hash: {hash}");

                // STEP 4: RESET SESSION TRACKING
                // Start fresh tracking for this search
                currentSessionId = Guid.NewGuid().ToString();
                currentSolutionHash = null;
                currentSolutionSource = null;
                currentSolutionConfidence = 0.0;

                // STEP 5: MULTI-LAYER INTELLIGENT SEARCH
                // Try multiple search strategies in order of reliability and speed
                SolutionResult result = null;
                var searchResults = new List<SolutionResult>();

                // SEARCH LAYER 1: EXACT MATCH (Fastest, Most Reliable)
                // Check if we've seen this exact error before
                System.Diagnostics.Debug.WriteLine("Trying exact match...");
                result = await TryExactMatchAsync(hash);
                if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                {
                    searchResults.Add(result);
                    System.Diagnostics.Debug.WriteLine($"Exact match found with confidence: {result.Confidence}");
                }

                // SEARCH LAYER 2: KEYWORD MATCHING (Good for similar errors)
                // Look for solutions that dealt with the same important concepts
                if (result == null)
                {
                    System.Diagnostics.Debug.WriteLine("No exact match, trying keyword search...");
                    result = await TryKeywordMatchAsync(sig);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                        System.Diagnostics.Debug.WriteLine($"Keyword match found with confidence: {result.Confidence}");
                    }
                }

                // SEARCH LAYER 3: AI SIMILARITY MATCHING (Most Intelligent)
                // Use AI to find errors that mean the same thing even with different words
                if (result == null)
                {
                    System.Diagnostics.Debug.WriteLine("No keyword match, trying AI vector similarity...");
                    var tokens = Tokens(sig);
                    result = await TryEnhancedVectorMatchAsync(cleanErrorInput, tokens);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                        System.Diagnostics.Debug.WriteLine($"Vector similarity match found with confidence: {result.Confidence}");
                    }
                }

                // SEARCH LAYER 4: PATTERN RECOGNITION (Fallback for Common Types)
                // Check if this matches any known error patterns
                if (result == null)
                {
                    System.Diagnostics.Debug.WriteLine("No vector match, trying pattern recognition...");
                    result = await TryEnhancedPatternMatchAsync(sig);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                        System.Diagnostics.Debug.WriteLine($"Pattern match found with confidence: {result.Confidence}");
                    }
                }

                // SEARCH LAYER 5: ENHANCED ML PREDICTION (Advanced Machine Learning)
                // Use clustering, decision trees, and neural networks for prediction
                if (result == null)
                {
                    System.Diagnostics.Debug.WriteLine("No pattern match, trying ML prediction...");
                    result = await EnhancedMLSearchAsync(cleanErrorInput, hash);
                    if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                    {
                        searchResults.Add(result);
                        System.Diagnostics.Debug.WriteLine($"ML prediction found with confidence: {result.Confidence}");
                    }
                }

                // STEP 6: DISPLAY BEST RESULT IF FOUND
                if (searchResults.Count > 0)
                {
                    var bestResult = SelectBestSolution(searchResults);
                    if (bestResult != null)
                    {
                        // TRACK THE SOLUTION WE'RE SHOWING
                        // Remember what we found so we can learn from user feedback
                        currentSolutionHash = bestResult.Hash;
                        currentSolutionSource = bestResult.Source;
                        currentSolutionConfidence = bestResult.Confidence;

                        // DISPLAY THE RESULT WITH CONFIDENCE RATING AND SOURCE INFO
                        // Show the solution along with how confident we are and how it was found
                        var confidenceText = GetConfidenceText(bestResult);
                        var sourceInfo = GetSearchSourceInfo(bestResult.Source);
                        ResultBox.Text = $"{confidenceText}\n{sourceInfo}\n\n{bestResult.Steps}";

                        // ENABLE FEEDBACK BUTTONS
                        // Let user rate whether the solution worked
                        EnableFeedbackButtons();
                        return;
                    }
                }

                // STEP 7: NO MATCH FOUND ANYWHERE
                // If none of the search methods found anything, suggest using AI
                System.Diagnostics.Debug.WriteLine("No matches found in any search layer");
                ResultBox.Text = "No matches found using any search method. Click 'Use CAINE API' for AI assistance.";
                BtnCaineApi.IsEnabled = true;
            }
            catch (Exception ex)
            {
                ResultBox.Text = $"Search error: {ex.Message}\n\nYou can try 'Use CAINE API'.";
                BtnCaineApi.IsEnabled = true;
            }
            finally
            {
                // STEP 8: RE-ENABLE BUTTONS
                // Always re-enable buttons when done, even if there was an error
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
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
        private async void BtnCaineApi_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // STEP 1: DISABLE BUTTONS AND SHOW PROGRESS
                BtnCaineApi.IsEnabled = false;
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

                // STEP 3: SECURITY CHECK
                // Make sure the input is safe before sending to AI
                var validation = SecurityValidator.ValidateInput(err, Environment.UserName, currentSessionId);
                if (!validation.IsValid)
                {
                    ResultBox.Text = $"AI input validation failed: {validation.ErrorMessage}";
                    return;
                }

                // STEP 4: BUILD CONVERSATION CONTEXT
                // Find similar errors we've solved before to give ChatGPT context
                var messages = await BuildEnhancedChatHistoryAsync(validation.CleanInput);

                // STEP 5: ASK CHATGPT
                // Send the error and context to ChatGPT for expert advice
                var gptResponse = await AskOpenAIAsync(messages);

                // STEP 6: TRACK THIS NEW SOLUTION
                // Remember what ChatGPT suggested so we can learn from feedback
                currentSolutionHash = Sha256Hex(gptResponse);
                currentSolutionSource = "openai_enhanced";
                currentSolutionConfidence = 0.5; // Default confidence for new AI solutions

                // STEP 7: DISPLAY THE AI RESPONSE
                // Show the solution with a note that it's AI-generated and needs rating
                var confidenceText = "AI Generated Solution (Confidence: Learning) | Please rate this solution to help CAINE learn";
                ResultBox.Text = $"{confidenceText}\n\n{gptResponse}";

                // STEP 8: PREPARE FOR TEACHING
                // Extract clean steps in case user wants to save this solution
                var numbered = ExtractNumberedLines(gptResponse).ToArray();
                TeachSteps.Text = numbered.Length > 0
                    ? string.Join(Environment.NewLine, numbered)
                    : gptResponse;

                // STEP 9: ENABLE FEEDBACK
                // Let user rate the AI's suggestion
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
                    currentSolutionConfidence = 0.8; // High confidence for human-provided solutions
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
        /// AI SIMILARITY SEARCH - Find solutions using artificial intelligence
        /// 
        /// WHAT THIS DOES:
        /// Uses AI embeddings (like AI fingerprints) to find errors that "mean the same thing"
        /// even if they use different words
        /// 
        /// LIKE A SMART LIBRARIAN WHO UNDERSTANDS CONTEXT:
        /// - "Database connection failed" and "Can't connect to SQL server" 
        ///   might use different words but mean the same thing
        /// - AI can recognize these similarities and suggest relevant solutions
        /// - Combines similarity matching with user feedback to rank results
        /// 
        /// This is the most advanced search method but also the most powerful
        /// </summary>
        private async Task<SolutionResult> TryEnhancedVectorMatchAsync(string rawError, string[] likeTokens)
        {
            try
            {
                // STEP 1: CREATE AI FINGERPRINT OF THE NEW ERROR
                var queryEmb = await EmbedAsync(rawError);

                // STEP 2: SEARCH THROUGH EXISTING SOLUTIONS IN BATCHES
                // Process in pages to handle large knowledge bases efficiently
                var pageSize = 100;
                var candidates = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>();

                // STEP 3: COLLECT CANDIDATE SOLUTIONS
                // Search through multiple pages of potential matches
                for (int page = 0; page < 3; page++)
                {
                    var pageCandidates = await GetVectorCandidatesPage(likeTokens, page, pageSize);
                    candidates.AddRange(pageCandidates);

                    if (pageCandidates.Count < pageSize) break; // No more results
                }

                if (candidates.Count == 0) return null;

                // STEP 4: FIND THE BEST MATCH
                // Compare the new error's AI fingerprint with all candidates
                NormalizeInPlace(queryEmb);               // Standardize for comparison
                SolutionResult bestResult = null;
                double bestScore = -1;

                foreach (var c in candidates)
                {
                    // CALCULATE AI SIMILARITY
                    var e = c.Emb;
                    NormalizeInPlace(e);                  // Standardize candidate fingerprint
                    var cos = Cosine(queryEmb, e);       // Calculate similarity (0-100%)

                    if (cos < VectorMinCosine) continue;  // Skip if not similar enough

                    // COMBINE AI SIMILARITY WITH USER FEEDBACK
                    // Create a weighted score that considers both AI similarity and user ratings
                    var confidence = CalculateConfidence(c.SuccessRate, c.FeedbackCount, c.HasConflicts ? 0.4 : 0.1);
                    var weightedScore = cos * (0.7 + confidence * 0.3);

                    // TRACK THE BEST OPTION
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
        /// PAGINATION HELPER - Get chunks of similar solutions from database
        /// 
        /// WHAT THIS DOES:
        /// When searching through thousands of solutions, this gets them in manageable chunks
        /// Like reading a thick phone book one page at a time instead of all at once
        /// 
        /// INCLUDES PERFORMANCE OPTIMIZATION:
        /// - Processes data in batches to avoid memory issues
        /// - Joins with feedback data to get success ratings
        /// - Returns both the solutions and their AI fingerprints for comparison
        /// </summary>
        private async Task<List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>>
            GetVectorCandidatesPage(string[] likeTokens, int page, int pageSize)
        {
            return await Task.Run(() =>
            {
                var list = new List<(string Steps, float[] Emb, string Hash, double SuccessRate, int FeedbackCount, bool HasConflicts)>();

                try
                {
                    // BUILD KEYWORD FILTER (OPTIONAL)
                    // If we have keywords, use them to narrow the search
                    string where = (likeTokens != null && likeTokens.Length > 0)
                        ? "WHERE " + string.Join(" OR ", likeTokens.Select((t, i) => "kb.error_signature LIKE ?"))
                        : "";

                    // GET ONE PAGE OF CANDIDATES
                    // Retrieve solutions with their AI fingerprints and feedback data
                    var sql = $@"
                        SELECT kb.resolution_steps, kb.embedding, kb.error_hash,
                               COALESCE(fb.success_rate, 0.5) as success_rate,         -- How often this solution works
                               COALESCE(fb.feedback_count, 0) as feedback_count,       -- How many people have tried it
                               CASE WHEN COALESCE(fb.conflict_rate, 0.0) > {ConflictThreshold} THEN true ELSE false END as has_conflicts
                        FROM {TableKB} kb
                        LEFT JOIN (
                            -- FEEDBACK STATISTICS SUBQUERY
                            SELECT solution_hash,
                                   AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate,
                                   COUNT(*) as feedback_count,
                                   AVG(CASE WHEN was_helpful THEN 0.0 ELSE 1.0 END) as conflict_rate
                            FROM {TableFeedback}
                            GROUP BY solution_hash
                        ) fb ON kb.error_hash = fb.solution_hash
                        {where}
                        ORDER BY kb.created_at DESC
                        LIMIT {pageSize} OFFSET {page * pageSize}";     // Pagination: skip previous pages

                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        // ADD KEYWORD PARAMETERS IF WE'RE FILTERING
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
                                // EXTRACT SOLUTION DATA
                                var steps = rdr.IsDBNull(0) ? "" : (rdr.GetValue(0)?.ToString() ?? "");
                                var hash = rdr.IsDBNull(2) ? "" : rdr.GetString(2);
                                var successRate = rdr.GetDouble(3);
                                var feedbackCount = rdr.GetInt32(4);
                                var hasConflicts = rdr.GetBoolean(5);

                                // EXTRACT AI FINGERPRINT
                                // Parse the AI embedding stored in the database
                                float[] emb = Array.Empty<float>();
                                if (!rdr.IsDBNull(1))
                                {
                                    var raw = rdr.GetValue(1)?.ToString() ?? "";
                                    emb = ParseFloatArray(raw);
                                }

                                // ONLY INCLUDE SOLUTIONS WITH AI FINGERPRINTS
                                // We need the embeddings for similarity comparison
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
                var errorText = ErrorInput.Text;
                var normalizedSig = SecureEscape(Normalize(errorText));

                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        // RECORD POSITIVE FEEDBACK
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

                // REFRESH CONFIDENCE DISPLAY
                // Show the updated confidence score immediately
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);

                    // Update just the confidence line in the display
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
                var errorText = ErrorInput.Text;
                var normalizedSig = SecureEscape(Normalize(errorText));

                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        // RECORD NEGATIVE FEEDBACK
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
                        false,                          -- Mark as NOT helpful
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

                // REFRESH CONFIDENCE DISPLAY
                // Show the updated (likely lower) confidence score
                var updatedResult = await GetUpdatedConfidence(currentSolutionHash);
                if (updatedResult != null)
                {
                    var newConfidenceText = GetConfidenceText(updatedResult);

                    // Update the confidence line in the display
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
            return Environment.GetEnvironmentVariable("OPENAI_API_KEY")
                   ?? throw new InvalidOperationException("OPENAI_API_KEY environment variable not set");
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
            var esc = incomingLower.Replace("'", "''");  // Escape for SQL safety

            // FIND RELATED SUCCESSFUL SOLUTIONS
            // Search for similar errors that have good feedback ratings
            var related = await Task.Run(() =>
            {
                var list = new List<(string Sig, string Steps, string Exact, double SuccessRate)>();
                using (var conn = OpenConn())
                using (var cmd = new OdbcCommand($@"
                    SELECT kb.error_signature, kb.resolution_steps, kb.error_text,
                           COALESCE(fb.success_rate, 0.5) as success_rate
                    FROM {TableKB} kb
                    LEFT JOIN (
                        -- Only include solutions with enough feedback to be reliable
                        SELECT solution_hash, 
                               AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                        HAVING COUNT(*) >= 3
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE lower(kb.error_signature) LIKE '%{esc}%'
                       OR lower(kb.error_text) LIKE '%{esc}%'
                    ORDER BY success_rate DESC, kb.created_at DESC
                    LIMIT 6                                          -- Top 6 most relevant solutions
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

            // BUILD CONVERSATION MESSAGES
            // Create a conversation thread that includes context and the new question
            var messages = new List<object>
            {
                // SYSTEM MESSAGE - Tell ChatGPT what its role is
                new { role = "system", content = "You are CAINE, an expert database debugging assistant with access to proven solutions. Return concise, actionable steps in well-formatted Markdown with headers, bullet points, and code blocks when appropriate." }
            };

            // ADD CONTEXT FROM SUCCESSFUL PAST SOLUTIONS
            foreach (var r in related)
            {
                var confidence = r.SuccessRate > FeedbackBoostThreshold ? " (Proven high success rate)" : "";
                messages.Add(new
                {
                    role = "user",
                    content = $"Previously solved error:\n'{r.Sig}'{confidence}\n\nProven solution:\n{r.Steps}"
                });
            }

            // ADD THE NEW QUESTION
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

                if (trainingData.Count >= 50) // Need minimum data for ML
                {
                    // Initialize ML engine
                    mlEngine = new CaineMLEngine();
                    await mlEngine.InitializeAsync(trainingData);

                    // Initialize neural network
                    int featureCount = ExtractFeatures(ErrorInput.Text).Length;
                    neuralNetwork = new SimpleNeuralNetwork(
                        inputSize: featureCount,
                        hiddenSize: 50,
                        outputSize: 3 // Success, Partial Success, Failure
                    );

                    // Train neural network
                    await TrainNeuralNetworkAsync(trainingData);

                    System.Diagnostics.Debug.WriteLine("ML components initialized successfully");
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine($"Insufficient training data: {trainingData.Count} samples");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"ML initialization failed: {ex.Message}");
            }
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
                        // Load error history with feedback
                        var sql = $@"
                            SELECT 
                                kb.error_text,
                                kb.error_hash,
                                kb.error_signature,
                                fb.was_helpful,
                                fb.created_at,
                                COALESCE(fb.resolution_time_minutes, 30) as response_time
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
                                var responseTime = rdr.GetDouble(5);

                                // Extract features and create training point
                                var features = ExtractFeatures(errorText);

                                trainingData.Add(new CaineMLEngine.TrainingDataPoint
                                {
                                    Features = features,
                                    ErrorHash = errorHash,
                                    SolutionWorked = wasHelpful,
                                    ResponseTime = responseTime,
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
            });
        }

        /// <summary>
        /// PREDICT WITH NEURAL NETWORK - Get predictions from deep learning model
        /// </summary>
        private async Task<(double Confidence, string Category)> PredictWithNeuralNetworkAsync(double[] features)
        {
            return await Task.Run(() =>
            {
                if (neuralNetwork == null)
                    return (0.5, "unknown");

                var output = neuralNetwork.Forward(features);

                // Interpret outputs
                double successProbability = output[0];
                double quickResolutionProb = output[1];
                double complexProblemProb = output[2];

                string category = complexProblemProb > 0.7 ? "complex" :
                                 quickResolutionProb > 0.7 ? "simple" : "moderate";

                return (successProbability, category);
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

            // This would open a new window showing:
            // - Error clusters visualization
            // - Trend predictions
            // - Model accuracy metrics
            // - Feature importance rankings
            // - Anomaly detection alerts
        }
    }
}