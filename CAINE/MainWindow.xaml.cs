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

namespace CAINE
{
    public partial class MainWindow : Window
    {

        // Database connection settings - connects to Databricks via ODBC
        private const string DsnName = "CAINE_Databricks";

        // Knowledge base table that stores error signatures and their resolutions
        private const string TableKB = "default.cai_error_kb";

        // Pattern rules table for regex-based error matching
        private const string TablePatterns = "default.cai_error_patterns";

        // NEW: Feedback tracking table - this is the "brain" that learns from user feedback
        private const string TableFeedback = "default.cai_solution_feedback";

        // ---- Machine Learning Configuration Parameters ----
        // These numbers control how smart the system is
        private const int VectorCandidateLimit = 300;      // How many similar errors to consider
        private const int LikeTokenMax = 4;                // Max keywords to search for
        private const double VectorMinCosine = 0.70;       // Minimum similarity score (70% similar)
        private const double FeedbackBoostThreshold = 0.7; // Solutions with >70% success rate get priority

        // Stop words to ignore when tokenizing error messages for LIKE queries
        // These are common words that don't help identify specific problems
        private static readonly string[] Stop = new[]
        {
            "error","failed","login","user","query","server","execute","executing","ssis","package",
            "for","the","with","from","could","not","because","property","set","correctly","message",
            "reason","reasons","possible","and","was","is","are","to","in","of","on","by","sqlstate","code"
        };

        // Machine Learning Session Tracking - keeps track of what we showed the user
        private string currentSessionId = Guid.NewGuid().ToString();    // Unique ID for this user session
        private string currentSolutionHash = null;                      // Hash of the solution we showed
        private string currentSolutionSource = null;                    // Where solution came from: "kb", "pattern", "vector", "openai"

        // ------------ OpenAI Integration Layer ------------
        // This is how we connect to ChatGPT for AI-powered suggestions

        private static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(60) };

        /// <summary>
        /// Gets the OpenAI API key from environment variables
        /// This is like getting your password to talk to ChatGPT
        /// </summary>
        private static string GetOpenAIApiKey()
        {
            return Environment.GetEnvironmentVariable("OPENAI_API_KEY")
                   ?? throw new InvalidOperationException("OPENAI_API_KEY environment variable not set");
        }

        /// <summary>
        /// Ensures we use secure HTTPS connection (TLS 1.2)
        /// This is like making sure our conversation with ChatGPT is encrypted
        /// </summary>
        private static void EnsureTls12()
        {
            try { ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12; } catch { }
        }

        /// <summary>
        /// CORE AI FUNCTION: Asks ChatGPT for help solving an error
        /// Think of this like having a conversation with an expert who knows about database problems
        /// </summary>
        /// <param name="messages">The conversation history to send to ChatGPT</param>
        /// <param name="model">Which version of ChatGPT to use</param>
        /// <returns>ChatGPT's suggested solution</returns>
        private async Task<string> AskOpenAIAsync(object[] messages, string model = "gpt-4o-mini")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            // Package our question in the format ChatGPT expects
            var payload = new { model, messages };
            var json = JsonConvert.SerializeObject(payload);

            // Send the request to OpenAI's servers
            using (var req = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/chat/completions"))
            {
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", apiKey);
                req.Content = new StringContent(json, Encoding.UTF8, "application/json");

                using (var res = await Http.SendAsync(req))
                {
                    var body = await res.Content.ReadAsStringAsync();
                    if (!res.IsSuccessStatusCode)
                        throw new Exception($"OpenAI error {res.StatusCode}: {body}");

                    // Extract ChatGPT's response from the JSON
                    var obj = JObject.Parse(body);
                    var content =
                        (string)(obj["choices"]?[0]?["message"]?["content"]) ??
                        (string)(obj["choices"]?[0]?["text"]) ?? "";
                    return content;
                }
            }
        }

        /// <summary>
        /// MACHINE LEARNING CORE: Converts text into mathematical vectors (embeddings)
        /// This is like converting words into coordinates on a map where similar meanings are close together
        /// Example: "car won't start" and "vehicle startup failure" would have similar coordinates
        /// </summary>
        /// <param name="text">The error message to convert</param>
        /// <param name="model">Which OpenAI embedding model to use</param>
        /// <returns>An array of numbers that represents the meaning of the text</returns>
        private async Task<float[]> EmbedAsync(string text, string model = "text-embedding-3-large")
        {
            var apiKey = GetOpenAIApiKey();
            EnsureTls12();

            // Ask OpenAI to convert our text into a mathematical representation
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

                    // Extract the vector (array of numbers) from the response
                    var obj = JObject.Parse(body);
                    var arr = (JArray?)obj["data"]?[0]?["embedding"] ?? new JArray();
                    var vec = new float[arr.Count];
                    for (int i = 0; i < arr.Count; i++) vec[i] = (float)arr[i]!;
                    return vec;
                }
            }
        }

        /// <summary>
        /// MACHINE LEARNING: Data structure to hold information about related errors and their success rates
        /// This is like a report card for each solution showing how often it actually worked
        /// </summary>
        private class RelatedRow
        {
            public string Sig;          // The error signature (fingerprint of the problem)
            public string Steps;        // The solution steps
            public string Exact;        // The original error message
            public double SuccessRate;  // NEW ML FEATURE: What % of users said this solution worked (0.0 to 1.0)
        }

        /// <summary>
        /// MACHINE LEARNING CORE: Builds a smart conversation with ChatGPT using past successful solutions
        /// This is like preparing talking points for a consultant meeting, highlighting what has worked before
        /// The ML magic: Solutions that worked well for others get highlighted as "high success rate"
        /// </summary>
        /// <param name="incomingErrorMessage">The new error we need to solve</param>
        /// <returns>A conversation history that makes ChatGPT smarter about our specific problems</returns>
        private async Task<object[]> BuildChatHistoryAsync(string incomingErrorMessage)
        {
            // Clean up the error message for database searching
            var incomingLower = (incomingErrorMessage ?? "").ToLowerInvariant();
            var esc = incomingLower.Replace("'", "''");  // Prevent SQL injection attacks

            // MACHINE LEARNING QUERY: Find similar errors AND rank them by success rate
            // This is the ML magic - we're not just finding similar problems,
            // we're finding similar problems whose solutions ACTUALLY WORKED for real users!
            var related = await Task.Run(() =>
            {
                var list = new List<RelatedRow>();
                using (var conn = OpenConn())
                using (var cmd = new OdbcCommand($@"
                    SELECT kb.error_signature, kb.resolution_steps, kb.error_text,
                           COALESCE(fb.success_rate, 0.5) as success_rate
                    FROM {TableKB} kb
                    LEFT JOIN (
                        -- Calculate success rate for each solution based on user feedback
                        SELECT solution_hash, 
                               AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                        HAVING COUNT(*) >= 3  -- Only consider solutions with at least 3 feedback votes
                    ) fb ON kb.error_hash = fb.solution_hash
                    WHERE lower(kb.error_signature) LIKE '%{esc}%'
                       OR lower(kb.error_text) LIKE '%{esc}%'
                    ORDER BY success_rate DESC, kb.created_at DESC  -- Best solutions first!
                    LIMIT 6
                ", conn))
                using (var rdr = cmd.ExecuteReader())
                {
                    while (rdr.Read())
                    {
                        var row = new RelatedRow
                        {
                            Sig = rdr.IsDBNull(0) ? "" : rdr.GetString(0),
                            Steps = rdr.IsDBNull(1) ? "" : (rdr.GetValue(1)?.ToString() ?? ""),
                            Exact = rdr.IsDBNull(2) ? "" : rdr.GetString(2),
                            SuccessRate = rdr.IsDBNull(3) ? 0.5 : rdr.GetDouble(3)  // Track effectiveness
                        };
                        list.Add(row);
                    }
                }
                return list;
            });

            // Build a conversation like you're briefing a human expert
            var messages = new List<object>
            {
                // Give ChatGPT its instructions
                new { role = "system", content = "You are a helpful database debugging assistant. Return concise, actionable steps. If unsure, say so." }
            };

            // Add examples of past problems and their solutions
            // ML ENHANCEMENT: Tell ChatGPT which solutions have proven track records
            foreach (var r in related)
            {
                // Mark high-performing solutions so ChatGPT knows they're reliable
                var confidence = r.SuccessRate > FeedbackBoostThreshold ? " (High success rate)" : "";
                messages.Add(new
                {
                    role = "user",
                    content = $"Previously, this error occurred:\n'{r.Sig}'{confidence}\n\nThe solution was:\n{r.Steps}"
                });
            }

            // Finally, ask about the current problem
            messages.Add(new
            {
                role = "user",
                content = "Help me debug this new error and return the answer in well-formatted Markdown with headers, bullet points, and code blocks.\n\nError:\n" + incomingErrorMessage
            });

            return messages.ToArray();
        }

        /// <summary>
        /// TEXT PROCESSING: Extracts numbered steps from ChatGPT's markdown response
        /// This helps format the AI's response into clean, actionable steps
        /// </summary>
        /// <param name="markdown">The formatted text response from ChatGPT</param>
        /// <returns>Clean step-by-step instructions</returns>
        private static IEnumerable<string> ExtractNumberedLines(string markdown)
        {
            if (markdown == null) yield break;
            foreach (var raw in markdown.Split('\n'))
            {
                var t = raw.Trim();
                // Look for numbered lists or bullet points
                if (t.StartsWith("1.") || t.StartsWith("2.") || t.StartsWith("3.") || t.StartsWith("- "))
                    yield return t.TrimStart('-', ' ').Trim();
            }
        }

        // ---------------------- Application Initialization ----------------------

        public MainWindow()
        {
            InitializeComponent();
            EnsureTls12();
            InitializeFeedbackTable();  // Set up the machine learning tables
        }

        /// <summary>
        /// DATABASE SETUP: Creates the tables needed for machine learning feedback system
        /// This is like setting up filing cabinets to store what solutions worked and what didn't
        /// </summary>
        private async void InitializeFeedbackTable()
        {
            try
            {
                await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    {
                        // Create the pattern rules table for regex-based matching
                        try
                        {
                            using (var cmd = new OdbcCommand($@"
                                CREATE TABLE IF NOT EXISTS {TablePatterns} (
                                    pattern_id STRING,           -- Unique identifier for this pattern
                                    pattern_regex STRING,        -- Regular expression to match errors
                                    priority INT,                -- Higher priority patterns are checked first
                                    resolution_steps ARRAY<STRING>, -- Steps to fix errors matching this pattern
                                    created_at TIMESTAMP,        -- When this pattern was added
                                    description STRING           -- Human-readable description of what this pattern catches
                                ) USING DELTA
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }
                            System.Diagnostics.Debug.WriteLine("Created patterns table successfully");
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Could not create patterns table: {ex.Message}");
                        }

                        // Create the MACHINE LEARNING feedback table - this is where we learn!
                        try
                        {
                            using (var cmd = new OdbcCommand($@"
                                CREATE TABLE IF NOT EXISTS {TableFeedback} (
                                    feedback_id STRING,       -- Unique ID for this piece of feedback
                                    session_id STRING,        -- Which user session this came from
                                    solution_hash STRING,     -- Which solution we showed (links to knowledge base)
                                    solution_source STRING,   -- Where solution came from: kb/pattern/vector/openai
                                    was_helpful BOOLEAN,      -- Did this solution actually work? (TRUE/FALSE)
                                    user_comment STRING,      -- Optional notes from the user
                                    created_at TIMESTAMP,     -- When we got this feedback
                                    error_signature STRING    -- What error this was feedback for
                                ) USING DELTA
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }
                            System.Diagnostics.Debug.WriteLine("Created feedback table successfully");
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Could not create feedback table: {ex.Message}");
                        }

                        // Set up permissions for the database user
                        try
                        {
                            // Grant database-level permissions
                            using (var cmd = new OdbcCommand($@"
                                GRANT USAGE ON DATABASE default TO `xavierborja@innohubspace.onmicrosoft.com`
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }

                            using (var cmd = new OdbcCommand($@"
                                GRANT SELECT ON DATABASE default TO `xavierborja@innohubspace.onmicrosoft.com`
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }

                            // Grant permissions to all CAINE tables
                            using (var cmd = new OdbcCommand($@"
                                GRANT ALL PRIVILEGES ON TABLE {TableKB} TO `xavierborja@innohubspace.onmicrosoft.com`
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }

                            using (var cmd = new OdbcCommand($@"
                                GRANT ALL PRIVILEGES ON TABLE {TablePatterns} TO `xavierborja@innohubspace.onmicrosoft.com`
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }

                            using (var cmd = new OdbcCommand($@"
                                GRANT ALL PRIVILEGES ON TABLE {TableFeedback} TO `xavierborja@innohubspace.onmicrosoft.com`
                            ", conn))
                            {
                                cmd.ExecuteNonQuery();
                            }

                            System.Diagnostics.Debug.WriteLine("Successfully granted permissions to all CAINE tables");
                        }
                        catch (Exception permEx)
                        {
                            System.Diagnostics.Debug.WriteLine($"Could not grant permissions: {permEx.Message}");
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Could not initialize tables: {ex.Message}");
            }
        }

        /// <summary>
        /// TEXT NORMALIZATION: Converts messy error messages into clean, searchable signatures
        /// This is like creating a standardized filing system where similar problems get the same label
        /// Example: "Login failed for user 'john.doe'" becomes "login failed for user <user>"
        /// </summary>
        /// <param name="s">Raw error message</param>
        /// <returns>Cleaned, normalized version for matching</returns>
        private static string Normalize(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return "";
            var t = s.ToLowerInvariant();

            // Replace unique identifiers with generic placeholders
            // This helps group similar errors together regardless of specific details

            // Replace GUIDs (like 123e4567-e89b-12d3-a456-426614174000) with <guid>
            t = Regex.Replace(t, @"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", "<guid>");

            // Replace Windows file paths (like C:\Program Files\App\file.exe) with <path>
            t = Regex.Replace(t, @"[a-z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n]+", "<path>");

            // Genericize stored procedure calls
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]\.\[[^\]]+\]", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+\[[^\]]+\]", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+\.[a-z0-9_]+", "exec <proc>");
            t = Regex.Replace(t, @"\bexec(?:ute)?\s+[a-z0-9_]+", "exec <proc>");

            // Genericize login failures (replace specific usernames)
            t = Regex.Replace(t, @"login failed for user\s+'[^']+'", "login failed for user '<user>'");

            // Replace usernames and email addresses with generic placeholder
            t = Regex.Replace(t, @"\b[a-z0-9._-]+\\[a-z0-9.$_-]+\b", "<user>");
            t = Regex.Replace(t, @"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", "<user>");

            // Clean up punctuation, replace numbers, normalize whitespace
            t = Regex.Replace(t, @"[^\w\s-]", " ");        // Remove special characters
            t = Regex.Replace(t, @"\b\d+\b", "<num>");     // Replace numbers with <num>
            t = Regex.Replace(t, @"\s+", " ").Trim();      // Normalize whitespace

            return t;
        }

        /// <summary>
        /// CRYPTOGRAPHIC HASHING: Creates a unique fingerprint for each normalized error
        /// This is like creating a barcode for each type of error so we can find it quickly
        /// </summary>
        /// <param name="s">Normalized error string</param>
        /// <returns>SHA256 hash (64-character fingerprint)</returns>
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
        /// DATABASE CONNECTION: Opens a connection to the Databricks database
        /// </summary>
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        /// <summary>
        /// TEXT PROCESSING: Extracts meaningful keywords from error signatures for searching
        /// This removes common words that don't help identify specific problems
        /// Example: "login failed for user because password" -> ["login", "password"]
        /// </summary>
        /// <param name="sig">Normalized error signature</param>
        /// <returns>Array of important keywords</returns>
        private static string[] Tokens(string sig)
        {
            var toks = Regex.Matches(sig ?? "", @"[a-z0-9\-]{4,}")  // Find words 4+ characters long
                            .Cast<Match>().Select(m => m.Value)
                            .Where(v => !Stop.Contains(v))           // Remove stop words
                            .Distinct()                              // Remove duplicates
                            .Take(LikeTokenMax)                      // Limit to top keywords
                            .ToArray();
            return toks;
        }

        /// <summary>
        /// MACHINE LEARNING: Records whether a solution actually worked for the user
        /// This is the core of the learning system - every thumbs up/down makes CAINE smarter
        /// Think of this as a customer satisfaction survey that improves future recommendations
        /// </summary>
        /// <param name="wasHelpful">True if the solution worked, false if it didn't</param>
        /// <param name="comment">Optional feedback from the user</param>
        private async void RecordFeedback(bool wasHelpful, string comment = "")
        {
            // Only record feedback if we actually showed a solution
            if (string.IsNullOrEmpty(currentSolutionHash)) return;

            try
            {
                await Task.Run(() =>
                {
                    // Clean the user input to prevent SQL injection
                    var escComment = comment.Replace("'", "''");
                    var escSig = Normalize(ErrorInput.Text).Replace("'", "''");

                    // Store the learning data: "Solution X worked/didn't work for Error Y at Time Z"
                    // This creates a permanent record that future AI recommendations can use
                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand($@"
                        INSERT INTO {TableFeedback} VALUES (
                            '{Guid.NewGuid()}',           -- Unique feedback ID
                            '{currentSessionId}',         -- This user session
                            '{currentSolutionHash}',      -- Which solution we showed them
                            '{currentSolutionSource}',    -- Where it came from (database, AI, etc.)
                            {(wasHelpful ? "true" : "false")}, -- Did it actually work?
                            '{escComment}',               -- Any additional user notes
                            current_timestamp(),          -- When this feedback was given
                            '{escSig}'                    -- What error it was feedback for
                        )
                    ", conn))
                    {
                        cmd.ExecuteNonQuery();
                    }
                });
            }
            catch (Exception ex)
            {
                // If feedback recording fails, don't crash the app - this is just for learning
                System.Diagnostics.Debug.WriteLine($"Feedback recording failed: {ex.Message}");
            }
        }

        // ---------------------- USER INTERFACE EVENT HANDLERS ----------------------

        /// <summary>
        /// MAIN SEARCH FUNCTION: This is the primary "brain" of CAINE that tries different AI approaches
        /// It works like a smart librarian who tries multiple methods to find the right answer:
        /// 1. Exact match (fastest)
        /// 2. Pattern matching (rule-based)  
        /// 3. Keyword search (simple)
        /// 4. AI similarity search (smartest)
        /// 5. Ask ChatGPT (last resort)
        /// </summary>
        private async void BtnSearch_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Disable buttons while searching
                BtnCaineApi.IsEnabled = false;
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Searching...";

                var err = ErrorInput.Text;
                var sig = Normalize(err);        // Clean up the error message
                var hash = Sha256Hex(sig);       // Create unique fingerprint

                // Reset session tracking for machine learning
                currentSessionId = Guid.NewGuid().ToString();
                currentSolutionHash = null;
                currentSolutionSource = null;

                // SEARCH STRATEGY 1: Exact hash lookup (fastest method)
                // This is like looking up a specific barcode in our catalog
                var sql = $@"
                    SELECT resolution_steps, error_hash
                    FROM {TableKB}
                    WHERE error_hash = '{hash}'
                    ORDER BY created_at DESC
                    LIMIT 1";

                var result = await Task.Run(() =>
                {
                    try
                    {
                        using (var conn = OpenConn())
                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            if (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : rdr.GetString(0);
                                var resultHash = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
                                return new { Steps = steps, Hash = resultHash, SuccessRate = 0.5 };
                            }
                            return null;
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Basic search failed: {ex.Message}");
                        return null;
                    }
                });

                // If we found an exact match, show it
                if (result != null && !string.IsNullOrWhiteSpace(result.Steps))
                {
                    currentSolutionHash = result.Hash;
                    currentSolutionSource = "kb";
                    ResultBox.Text = result.Steps;
                    BtnCaineApi.IsEnabled = true;
                    BtnFeedbackYes.IsEnabled = true;
                    BtnFeedbackNo.IsEnabled = true;
                    return;
                }

                // SEARCH STRATEGY 2: Pattern rules matching
                // This is like having if-then rules: "If error contains X, then solution is Y"
                ResultBox.Text = "No exact hash match. Checking pattern rules...";
                var patternResult = await TryPatternRuleAsync(sig);
                if (!string.IsNullOrWhiteSpace(patternResult))
                {
                    currentSolutionHash = Sha256Hex(patternResult);
                    currentSolutionSource = "pattern";
                    ResultBox.Text = patternResult;
                    BtnCaineApi.IsEnabled = true;
                    BtnFeedbackYes.IsEnabled = true;
                    BtnFeedbackNo.IsEnabled = true;
                    return;
                }

                // SEARCH STRATEGY 3: Simple keyword search
                // This is like searching for books by looking up individual words in the index
                ResultBox.Text = "No pattern match. Trying keyword search...";
                var likeTokens = Tokens(sig);
                if (likeTokens.Length > 0)
                {
                    var like = string.Join(" AND ", likeTokens.Select(t => $"error_signature LIKE '%{t}%'"));
                    var sqlLike = $@"
                        SELECT resolution_steps, error_hash
                        FROM {TableKB}
                        WHERE {like}
                        ORDER BY created_at DESC
                        LIMIT 1";

                    var keywordResult = await Task.Run(() =>
                    {
                        try
                        {
                            using (var conn = OpenConn())
                            using (var cmd = new OdbcCommand(sqlLike, conn))
                            using (var rdr = cmd.ExecuteReader())
                            {
                                if (rdr.Read())
                                {
                                    var steps = rdr.IsDBNull(0) ? "" : rdr.GetString(0);
                                    var hash = rdr.IsDBNull(1) ? "" : rdr.GetString(1);
                                    return new { Steps = steps, Hash = hash };
                                }
                                return null;
                            }
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine($"Keyword search failed: {ex.Message}");
                            return null;
                        }
                    });

                    if (keywordResult != null && !string.IsNullOrWhiteSpace(keywordResult.Steps))
                    {
                        currentSolutionHash = keywordResult.Hash;
                        currentSolutionSource = "keyword";
                        ResultBox.Text = keywordResult.Steps;
                        BtnCaineApi.IsEnabled = true;
                        BtnFeedbackYes.IsEnabled = true;
                        BtnFeedbackNo.IsEnabled = true;
                        return;
                    }
                }

                // SEARCH STRATEGY 4: AI Vector similarity search (machine learning magic!)
                // This uses math to find errors that "mean" similar things, even if they use different words
                ResultBox.Text = "No keyword match. Trying vector similarity...";
                var vectorResult = await TrySimpleVectorMatchAsync(err, likeTokens);
                if (!string.IsNullOrWhiteSpace(vectorResult))
                {
                    currentSolutionHash = Sha256Hex(vectorResult);
                    currentSolutionSource = "vector";
                    ResultBox.Text = vectorResult;
                    BtnCaineApi.IsEnabled = true;
                    BtnFeedbackYes.IsEnabled = true;
                    BtnFeedbackNo.IsEnabled = true;
                    return;
                }

                // SEARCH STRATEGY 5: No matches found - enable manual ChatGPT consultation
                ResultBox.Text = "No results. Click 'Use CAINE API' to try an AI suggestion.";
                BtnCaineApi.IsEnabled = true;
            }
            catch (Exception ex)
            {
                ResultBox.Text = "Search error:\n" + ex.Message + "\nYou can try 'Use CAINE API'.";
                BtnCaineApi.IsEnabled = true;
            }
            finally
            {
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }

        /// <summary>
        /// MACHINE LEARNING: Simple vector similarity matching without complex feedback weighting
        /// This is like finding books with similar themes, even if they don't share exact keywords
        /// Uses AI embeddings to understand semantic meaning of errors
        /// </summary>
        /// <param name="rawError">The original error message</param>
        /// <param name="likeTokens">Keywords to help filter candidates</param>
        /// <returns>Best matching solution or null if none found</returns>
        private async Task<string?> TrySimpleVectorMatchAsync(string rawError, string[] likeTokens)
        {
            try
            {
                // STEP 1: Convert the new error into AI embeddings (mathematical representation)
                var queryEmb = await EmbedAsync(rawError);

                // STEP 2: Get candidate solutions from database, filtered by keywords if available
                string where = (likeTokens != null && likeTokens.Length > 0)
                    ? "WHERE " + string.Join(" OR ", likeTokens.Select(t => $"error_signature LIKE '%{t}%'"))
                    : "";

                var sql = $@"
                    SELECT resolution_steps, embedding, error_hash
                    FROM {TableKB}
                    {where}
                    ORDER BY created_at DESC
                    LIMIT {VectorCandidateLimit}";

                // STEP 3: Load candidate solutions and their embeddings
                var candidates = await Task.Run(() =>
                {
                    var list = new List<(string Steps, float[] Emb, string Hash)>();
                    try
                    {
                        using (var conn = OpenConn())
                        using (var cmd = new OdbcCommand(sql, conn))
                        using (var rdr = cmd.ExecuteReader())
                        {
                            while (rdr.Read())
                            {
                                var steps = rdr.IsDBNull(0) ? "" : (rdr.GetValue(0)?.ToString() ?? "");
                                var hash = rdr.IsDBNull(2) ? "" : rdr.GetString(2);

                                // Parse the stored embedding (mathematical representation)
                                float[] emb = Array.Empty<float>();
                                if (!rdr.IsDBNull(1))
                                {
                                    var raw = rdr.GetValue(1)?.ToString() ?? "";
                                    emb = ParseFloatArray(raw);
                                }
                                if (emb.Length > 0)
                                    list.Add((steps, emb, hash));
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Vector search failed: {ex.Message}");
                    }
                    return list;
                });

                if (candidates.Count == 0) return null;

                // STEP 4: Find the most similar error using cosine similarity
                NormalizeInPlace(queryEmb);  // Normalize for accurate comparison
                string? bestSteps = null;
                string? bestHash = null;
                double bestScore = -1;

                foreach (var c in candidates)
                {
                    var e = c.Emb;
                    NormalizeInPlace(e);

                    // Calculate similarity score using cosine similarity (0 to 1, where 1 is identical)
                    var cos = Cosine(queryEmb, e);

                    if (cos > bestScore && cos >= VectorMinCosine)
                    {
                        bestScore = cos;
                        bestSteps = c.Steps;
                        bestHash = c.Hash;
                    }
                }

                // Return the best match if it's similar enough
                if (!string.IsNullOrWhiteSpace(bestSteps) && !string.IsNullOrWhiteSpace(bestHash))
                {
                    currentSolutionHash = bestHash;
                    return bestSteps;
                }

                return null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Simple vector match failed: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// AI CONSULTATION: Manually asks ChatGPT for help when other methods don't work
        /// This is like calling in a human expert consultant as a last resort
        /// Uses machine learning to show ChatGPT which solutions have worked well in the past
        /// </summary>
        private async void BtnCaineApi_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                BtnCaineApi.IsEnabled = false;
                BtnFeedbackYes.IsEnabled = false;
                BtnFeedbackNo.IsEnabled = false;
                ResultBox.Text = "Consulting CAINE API (OpenAI)…";

                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                // Build a smart conversation that includes past successful solutions
                var messages = await BuildChatHistoryAsync(err);

                // Ask ChatGPT for advice
                var gpt = await AskOpenAIAsync(messages);

                // Track this solution for feedback learning
                currentSolutionHash = Sha256Hex(gpt);
                currentSolutionSource = "openai";

                ResultBox.Text = gpt;

                // Extract clean steps for the teaching interface
                var numbered = ExtractNumberedLines(gpt).ToArray();
                TeachSteps.Text = numbered.Length > 0
                    ? string.Join(Environment.NewLine, numbered)
                    : gpt;

                // Enable feedback buttons so user can rate the AI's suggestion
                BtnFeedbackYes.IsEnabled = true;
                BtnFeedbackNo.IsEnabled = true;
            }
            catch (Exception ex)
            {
                ResultBox.Text = "OpenAI call failed:\n" + ex.Message;
            }
            finally
            {
                BtnCaineApi.IsEnabled = true;
            }
        }

        /// <summary>
        /// MACHINE LEARNING: User clicked "thumbs up" - this solution worked!
        /// Records positive feedback to improve future AI recommendations
        /// </summary>
        private async void BtnFeedbackYes_Click(object sender, RoutedEventArgs e)
        {
            await Task.Run(() => RecordFeedback(true));  // Record that this solution worked
            BtnFeedbackYes.IsEnabled = false;
            BtnFeedbackNo.IsEnabled = false;
            ResultBox.Text += "\n\nThank you for your feedback! This will help improve future recommendations.";
        }

        /// <summary>
        /// MACHINE LEARNING: User clicked "thumbs down" - this solution didn't work
        /// Records negative feedback and encourages user to teach the correct solution
        /// </summary>
        private async void BtnFeedbackNo_Click(object sender, RoutedEventArgs e)
        {
            await Task.Run(() => RecordFeedback(false));  // Record that this solution didn't work
            BtnFeedbackYes.IsEnabled = false;
            BtnFeedbackNo.IsEnabled = false;
            ResultBox.Text += "\n\nThank you for your feedback! Please consider teaching CAINE the correct solution below.";
        }

        /// <summary>
        /// KNOWLEDGE BASE EXPANSION: User teaches CAINE the correct solution to an error
        /// This adds new knowledge to the system and optionally records positive feedback
        /// Think of this as adding a new entry to an expert's handbook
        /// </summary>
        private async void BtnTeach_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                BtnSearch.IsEnabled = false;
                BtnTeach.IsEnabled = false;

                var err = ErrorInput.Text;
                if (string.IsNullOrWhiteSpace(err))
                {
                    ResultBox.Text = "Paste an error first.";
                    return;
                }

                // Parse the solution steps entered by the user
                var raw = (TeachSteps.Text ?? "").Replace("\r", "");
                var lines = raw.Split('\n');
                var stepsLines = lines.Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
                if (stepsLines.Length == 0)
                {
                    ResultBox.Text = "Enter at least one step (one per line).";
                    return;
                }

                // IMPORTANT: Escape special characters so database doesn't get confused
                string Esc(string s) => (s ?? "")
                    .Replace("\\", "\\\\")    // Escape backslashes
                    .Replace("'", "''")       // Escape quotes  
                    .Replace("\r", "")        // Remove carriage returns
                    .Replace("\n", "\\n");    // Escape newlines

                // Convert C# array to SQL array format
                string ArrayLiteral(string[] arr) =>
                    "array(" + string.Join(",", arr.Select(s => $"'{Esc(s)}'")) + ")";

                var sig = Normalize(err);           // Create normalized signature
                var hash = Sha256Hex(sig);          // Create unique fingerprint
                var steps = ArrayLiteral(stepsLines); // Format steps for database
                var escErr = Esc(err);              // Escape original error
                var escSig = Esc(sig);              // Escape signature

                // Optional: Create AI embedding for future vector search
                float[] emb = null;
                try
                {
                    emb = await EmbedAsync(err + "\n\n" + sig);
                }
                catch
                {
                    /* ignore embedding errors - not critical */
                }

                // Format embedding for database storage
                string embSql = emb == null
                    ? "NULL"
                    : $"array({string.Join(",", emb.Select(f => f.ToString("G", CultureInfo.InvariantCulture)))})";

                // UPSERT operation: Update existing record or insert new one
                var sql = $@"
                    MERGE INTO {TableKB} t
                    USING (SELECT
                             '{Guid.NewGuid()}' AS id,
                             current_timestamp() AS created_at,
                             'human' AS source,                    -- Mark this as human-provided knowledge
                             NULL AS product, NULL AS version, NULL AS env,
                             '{escErr}' AS error_text,
                             '{escSig}' AS error_signature,
                             '{hash}'   AS error_hash,
                             NULL AS context_text,
                             {steps}   AS resolution_steps,        -- The solution steps
                             array()   AS symptoms,
                             NULL AS root_cause,
                             NULL AS verification,
                             array()   AS links,
                             NULL AS code_before, NULL AS code_after, NULL AS notes,
                             {embSql}  AS embedding                -- AI representation for similarity search
                           ) s
                    ON t.error_hash = s.error_hash
                    WHEN MATCHED THEN UPDATE SET
                      -- If error exists, combine the solutions
                      t.resolution_steps = array_distinct(
                        concat(coalesce(t.resolution_steps, cast(array() as array<string>)),
                               s.resolution_steps))
                    WHEN NOT MATCHED THEN INSERT *";

                // Execute the database update
                var rows = await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                        return cmd.ExecuteNonQuery();
                });

                // MACHINE LEARNING: If user is correcting a bad suggestion, record this as positive feedback
                if (!string.IsNullOrEmpty(currentSolutionHash) && currentSolutionHash != hash)
                {
                    currentSolutionHash = hash;
                    currentSolutionSource = "human";
                    await Task.Run(() => RecordFeedback(true, "User-provided correction"));
                }

                ResultBox.Text = $"Taught CAINE this fix. Hash: {hash}{Environment.NewLine}Rows affected: {rows}";
            }
            catch (Exception ex)
            {
                ResultBox.Text = "Teach error:\n" + ex.Message;
            }
            finally
            {
                BtnSearch.IsEnabled = true;
                BtnTeach.IsEnabled = true;
            }
        }

        // ---------------------- ENHANCED AI SEARCH LAYERS ----------------------

        /// <summary>
        /// PATTERN MATCHING: Searches for regex-based rules that match the error
        /// This is like having a rulebook: "If error matches pattern X, then solution is Y"
        /// Useful for consistent error formats that can be caught with regular expressions
        /// </summary>
        /// <param name="sig">Normalized error signature</param>
        /// <returns>Matching solution steps or null if no pattern matches</returns>
        private async Task<string?> TryPatternRuleAsync(string sig)
        {
            try
            {
                var escSig = sig.Replace("'", "''");  // Prevent SQL injection

                // Look for regex patterns that match this error signature
                var sql = $@"
                    SELECT resolution_steps
                    FROM {TablePatterns}
                    WHERE '{escSig}' RLIKE pattern_regex    -- RLIKE = regex matching in SQL
                    ORDER BY priority DESC                  -- Higher priority patterns first
                    LIMIT 1";

                return await Task.Run(() =>
                {
                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                    using (var rdr = cmd.ExecuteReader())
                        return rdr.Read() ? rdr.GetString(0) : null;
                });
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// ADVANCED MACHINE LEARNING: Weighted vector similarity that learns from user feedback
        /// This is like having a smart librarian who remembers which book recommendations actually helped people
        /// 
        /// Instead of just finding "similar" errors, we find similar errors whose solutions ACTUALLY WORKED
        /// 
        /// How it works:
        /// 1. Convert error to mathematical coordinates (AI embeddings)
        /// 2. Find errors with similar coordinates (semantic similarity)  
        /// 3. Boost solutions that have high user satisfaction rates
        /// 4. Return the solution with best combined score (similarity + proven effectiveness)
        /// </summary>
        /// <param name="rawError">The new error message to solve</param>
        /// <param name="likeTokens">Keywords to help filter candidates</param>
        /// <returns>Best solution considering both similarity and user feedback</returns>
        private async Task<string?> TryVectorMatchAsync(string rawError, string[] likeTokens)
        {
            try
            {
                // STEP 1: Convert the new error into a mathematical fingerprint
                // Think of this like converting "my car won't start" into coordinates on a map of all problems
                var queryEmb = await EmbedAsync(rawError);

                // STEP 2: Get candidate solutions from database WITH their success rates
                // ML ENHANCEMENT: Also grab how often each solution actually worked for users
                string where = (likeTokens != null && likeTokens.Length > 0)
                    ? "WHERE " + string.Join(" OR ", likeTokens.Select(t => $"kb.error_signature LIKE '%{t}%'"))
                    : "";

                var sql = $@"
                    SELECT kb.resolution_steps, kb.embedding, kb.error_hash,
                           COALESCE(fb.success_rate, 0.5) as success_rate
                    FROM {TableKB} kb
                    LEFT JOIN (
                        -- Calculate success rate for each solution based on user feedback
                        SELECT solution_hash, 
                               AVG(CASE WHEN was_helpful THEN 1.0 ELSE 0.0 END) as success_rate
                        FROM {TableFeedback}
                        GROUP BY solution_hash
                        HAVING COUNT(*) >= 2  -- Only consider solutions with at least 2 feedback votes
                    ) fb ON kb.error_hash = fb.solution_hash
                    {where}
                    ORDER BY kb.created_at DESC
                    LIMIT {VectorCandidateLimit}";

                var candidates = await Task.Run(() =>
                {
                    var list = new List<(string Steps, float[] Emb, string Hash, double SuccessRate)>();
                    using (var conn = OpenConn())
                    using (var cmd = new OdbcCommand(sql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        while (rdr.Read())
                        {
                            var steps = rdr.IsDBNull(0) ? "" : (rdr.GetValue(0)?.ToString() ?? "");
                            var hash = rdr.IsDBNull(2) ? "" : rdr.GetString(2);
                            var successRate = rdr.IsDBNull(3) ? 0.5 : rdr.GetDouble(3);

                            // Parse the stored mathematical representation
                            float[] emb = Array.Empty<float>();
                            if (!rdr.IsDBNull(1))
                            {
                                var raw = rdr.GetValue(1)?.ToString() ?? "";
                                emb = ParseFloatArray(raw);
                            }
                            if (emb.Length > 0)
                                list.Add((steps, emb, hash, successRate));
                        }
                    }
                    return list;
                });

                if (candidates.Count == 0) return null;

                // STEP 3: MACHINE LEARNING MAGIC - Find the best match using AI + user feedback
                // This combines two types of intelligence:
                // 1. AI similarity (how mathematically similar are the errors?)
                // 2. Human feedback (which solutions actually worked in practice?)

                NormalizeInPlace(queryEmb);  // Normalize for accurate comparison
                string? bestSteps = null;
                string? bestHash = null;
                double bestScore = -1;

                foreach (var c in candidates)
                {
                    var e = c.Emb;
                    NormalizeInPlace(e);

                    // Calculate how mathematically similar this old error is to the new one
                    // Cosine similarity: 1.0 = identical meaning, 0.0 = completely different
                    var cos = Cosine(queryEmb, e);

                    // MACHINE LEARNING BOOST: Give extra weight to solutions that have worked well
                    // Formula: If success rate is 90%, get 20% boost. If 10%, get 25% penalty.
                    // This ensures we prioritize solutions that real users found helpful
                    var feedbackMultiplier = 1.0 + (c.SuccessRate - 0.5) * 0.5; // Range: 0.75 to 1.25
                    var weightedScore = cos * feedbackMultiplier;

                    // Pick the solution with the best combined score (similarity + proven effectiveness)
                    if (weightedScore > bestScore && cos >= VectorMinCosine)
                    {
                        bestScore = weightedScore;
                        bestSteps = c.Steps;
                        bestHash = c.Hash;
                    }
                }

                // Return the winning solution if it's good enough
                if (!string.IsNullOrWhiteSpace(bestSteps) && !string.IsNullOrWhiteSpace(bestHash))
                {
                    currentSolutionHash = bestHash;  // Remember what we showed for feedback tracking
                    return bestSteps;
                }

                return null;
            }
            catch
            {
                return null;  // If anything goes wrong, fail gracefully
            }
        }

        // ----- MATHEMATICAL HELPER FUNCTIONS FOR MACHINE LEARNING -----

        /// <summary>
        /// MATH HELPER: Converts a stored array string back into numbers
        /// Database stores embeddings as text, this converts them back to float arrays for calculations
        /// Example: "[0.1, 0.2, 0.3]" becomes [0.1f, 0.2f, 0.3f]
        /// </summary>
        /// <param name="s">String representation of float array</param>
        /// <returns>Array of floating point numbers</returns>
        private static float[] ParseFloatArray(string s)
        {
            if (string.IsNullOrWhiteSpace(s)) return Array.Empty<float>();
            var t = s.Trim();

            // Handle different array formats from database
            if (t.StartsWith("[")) t = t.Substring(1);                                    // Remove [
            if (t.EndsWith("]")) t = t.Substring(0, t.Length - 1);                       // Remove ]
            if (t.StartsWith("array(", StringComparison.OrdinalIgnoreCase)) t = t.Substring(6); // Remove array(
            if (t.EndsWith(")")) t = t.Substring(0, t.Length - 1);                       // Remove )

            // Parse comma-separated numbers
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
        /// MATH HELPER: Normalizes a vector to unit length for accurate similarity comparison
        /// This is like adjusting for different "loudnesses" - ensures we compare meaning, not magnitude
        /// 
        /// Why this matters: Vector [2, 4, 6] has same direction as [1, 2, 3] but different length
        /// Normalization makes them comparable by adjusting them to the same length
        /// </summary>
        /// <param name="v">Vector to normalize in-place</param>
        private static void NormalizeInPlace(float[] v)
        {
            if (v == null || v.Length == 0) return;

            // Calculate the length (magnitude) of the vector
            double sumsq = 0;
            for (int i = 0; i < v.Length; i++) sumsq += v[i] * v[i];
            var norm = Math.Sqrt(sumsq);

            // Avoid division by zero
            if (norm <= 1e-8) return;

            // Scale each component to make the vector unit length
            for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
        }

        /// <summary>
        /// MACHINE LEARNING CORE: Calculates cosine similarity between two vectors
        /// This measures how "similar" two mathematical representations are
        /// 
        /// Think of it like measuring the angle between two arrows:
        /// - Same direction (parallel): similarity = 1.0 (identical meaning)
        /// - Perpendicular: similarity = 0.0 (unrelated)  
        /// - Opposite direction: similarity = -1.0 (opposite meaning)
        /// 
        /// This is the mathematical heart of semantic search - it lets us find errors
        /// that "mean" similar things even if they use completely different words
        /// </summary>
        /// <param name="a">First vector (normalized)</param>
        /// <param name="b">Second vector (normalized)</param>
        /// <returns>Similarity score from -1.0 to 1.0</returns>
        private static double Cosine(float[] a, float[] b)
        {
            if (a == null || b == null) return -1;

            int n = Math.Min(a.Length, b.Length);
            if (n == 0) return -1;

            // Calculate dot product (sum of element-wise multiplication)
            // For normalized vectors, this gives us the cosine of the angle between them
            double dot = 0;
            for (int i = 0; i < n; i++) dot += a[i] * b[i];
            return dot;
        }
    }
}