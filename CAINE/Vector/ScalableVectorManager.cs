using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data.Odbc;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace CAINE.Vector
{
    /// <summary>
    /// SCALABLE VECTOR MANAGER - Advanced AI-powered similarity search system
    /// 
    /// WHAT THIS DOES:
    /// Finds similar errors even when they use completely different words
    /// Like how Spotify finds songs that "feel" similar even with different lyrics
    /// 
    /// HOW IT WORKS:
    /// 1. Converts error messages into mathematical "fingerprints" (vectors)
    /// 2. Compares fingerprints to find similar meanings
    /// 3. Caches results so repeated searches are instant
    /// 4. Protects against hackers trying to break the system
    /// </summary>
    public class ScalableVectorManager
    {
        // Database table names - where all the data is stored
        private const string TableKB = "default.cai_error_kb";                   // Main knowledge base with solutions
        private const string TableVectorIndex = "default.cai_vector_index";      // Fast-lookup index for AI fingerprints
        private const string TableVectorCache = "default.cai_vector_cache";      // Cache for recent searches
        private const string TableSecurityLog = "default.cai_security_log";      // Security event tracking
        private const string DsnName = "CAINE_Databricks";                       // Database connection name

        // Performance limits - prevent the system from getting overwhelmed
        private const int VectorCandidateLimit = 300;  // Don't compare against more than 300 errors (for speed)
        private const int VectorDimensions = 3072;     // Size of AI fingerprints from OpenAI text-embedding-3-large
        private const double SimilarityThreshold = 0.70; // Default 70% similarity = good enough match
        private const int CacheExpiryHours = 24;       // Cache expires after 24 hours (to get fresh results)

        // Security limits - prevent abuse and attacks
        private const int MaxQueryLength = 4000;       // Maximum error message length (prevents buffer overflow)
        private const int MaxDailyQueries = 1000;      // Max searches per user per day (prevents abuse)
        private const int MaxConcurrentSessions = 10;  // Max simultaneous searches (prevents overload)

        // Session limiter - only allows MaxConcurrentSessions at once
        private static readonly SemaphoreSlim SessionSemaphore = new SemaphoreSlim(MaxConcurrentSessions, MaxConcurrentSessions);

        /// <summary>
        /// SECURITY VALIDATOR - Protection against hackers and malicious input
        /// 
        /// WHAT THIS DOES:
        /// Like a bouncer at a club - checks everyone trying to get in and chin checks foo's 
        /// Blocks SQL injection, script attacks, and other hacking attempts
        /// Keeps track of who's being suspicious
        /// </summary>
        public class SecurityValidator
        {
            // Pattern to detect SQL injection attempts (hackers trying to access database)
            // Looks for dangerous SQL commands like DROP TABLE, DELETE, etc.
            private static readonly Regex SqlInjectionPattern = new Regex(
                @"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|MERGE|SELECT|UPDATE|UNION|USE)\b)|" +
                @"(--|/\*|\*/|;|'|xp_|sp_)",
                RegexOptions.IgnoreCase);

            // Pattern to detect cross-site scripting (XSS) attacks
            // Looks for JavaScript code that could steal user data
            private static readonly Regex XssPattern = new Regex(
                @"<script|javascript:|vbscript:|onload|onerror|onclick",
                RegexOptions.IgnoreCase);

            // List of Windows/SQL commands that could damage the system
            private static readonly string[] DangerousCommands = {
                "xp_cmdshell",    // Execute Windows commands
                "sp_oacreate",    // Create COM objects
                "sp_oamethod",    // Call COM methods
                "OPENROWSET",     // Access external data
                "BULK INSERT",    // Mass data insertion
                "bcp",            // Bulk copy program
                "sqlcmd",         // SQL command line
                "osql"            // Old SQL command line
            };

            /// <summary>
            /// VALIDATE INPUT - Main security check for all user input
            /// 
            /// WHAT THIS DOES:
            /// Like airport security - checks everything coming in for dangerous items
            /// Returns cleaned, safe input or blocks the request entirely
            /// </summary>
            public static ValidationResult ValidateInput(string input, string userId, string sessionId)
            {
                // Create result object to track validation outcome
                var result = new ValidationResult { IsValid = true, CleanInput = input };

                // CHECK 1: Make sure input isn't empty
                if (string.IsNullOrWhiteSpace(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input cannot be empty";
                    return result;
                }

                // CHECK 2: Make sure input isn't too long (could be buffer overflow attack)
                if (input.Length > MaxQueryLength)
                {
                    result.IsValid = false;
                    result.ErrorMessage = $"Input exceeds maximum length of {MaxQueryLength} characters";
                    // Log this as suspicious - why is someone sending huge inputs?
                    LogSecurityEvent("EXCESSIVE_LENGTH", userId, sessionId, input.Substring(0, 100));
                    return result;
                }

                // CHECK 3: Look for SQL injection attempts (DROP TABLE, DELETE, etc.)
                if (SqlInjectionPattern.IsMatch(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input contains potentially dangerous SQL patterns";
                    // Log this as an attack attempt
                    LogSecurityEvent("SQL_INJECTION_ATTEMPT", userId, sessionId, input);
                    return result;
                }

                // CHECK 4: Look for JavaScript/script injection attempts
                if (XssPattern.IsMatch(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input contains potentially dangerous script content";
                    // Log this as an attack attempt
                    LogSecurityEvent("XSS_ATTEMPT", userId, sessionId, input);
                    return result;
                }

                // CHECK 5: Look for dangerous system commands
                var inputLower = input.ToLowerInvariant(); // Convert to lowercase for checking
                foreach (var cmd in DangerousCommands)
                {
                    if (inputLower.Contains(cmd.ToLowerInvariant()))
                    {
                        result.IsValid = false;
                        result.ErrorMessage = "Input contains restricted commands";
                        // Log this as suspicious activity
                        LogSecurityEvent("DANGEROUS_COMMAND", userId, sessionId, input);
                        return result;
                    }
                }

                // CHECK 6: Make sure user hasn't exceeded daily limit (rate limiting)
                if (!CheckRateLimit(userId))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Rate limit exceeded. Please try again later.";
                    // Log that someone hit the rate limit
                    LogSecurityEvent("RATE_LIMIT_EXCEEDED", userId, sessionId, "");
                    return result;
                }

                // If all checks pass, clean the input to be extra safe
                result.CleanInput = SanitizeInput(input);
                return result;
            }

            /// <summary>
            /// SANITIZE INPUT - Clean up potentially dangerous characters
            /// 
            /// Like washing vegetables before cooking - removes anything harmful
            /// </summary>
            private static string SanitizeInput(string input)
            {
                return input.Replace("--", "- -")           // Break SQL comment markers
                            .Replace("/*", "/ *")            // Break block comment start
                            .Replace("*/", "* /")            // Break block comment end
                            .Replace(";", ",")               // Replace command separators
                            .Replace("'", "&#39;")           // Escape single quotes (prevent SQL injection)
                            .Replace("\"", "&quot;")         // Escape double quotes
                            .Replace("<", "&lt;")            // Escape HTML tags
                            .Replace(">", "&gt;")            // Escape HTML tags
                            .Replace("&", "&amp;")           // Escape ampersands
                            .Trim();                         // Remove extra spaces
            }

            /// <summary>
            /// CHECK RATE LIMIT - Ensure user isn't spamming the system
            /// 
            /// Like a ticket system - each user gets X queries per day
            /// </summary>
            private static bool CheckRateLimit(string userId)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Count how many queries this user has made in last 24 hours
                        var sql = @"
                            SELECT COUNT(*) 
                            FROM default.cai_security_log 
                            WHERE user_id = ? 
                            AND created_at >= current_timestamp() - INTERVAL 24 HOURS
                            AND event_type IN ('QUERY', 'SEARCH')";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("user_id", userId);
                            var count = Convert.ToInt32(cmd.ExecuteScalar());
                            // Return true if under limit, false if over
                            return count < MaxDailyQueries;
                        }
                    }
                }
                catch
                {
                    // If rate limiting breaks, allow the query (fail open, not closed)
                    return true;
                }
            }

            /// <summary>
            /// LOG SECURITY EVENT - Record suspicious activity for monitoring
            /// 
            /// Like security camera footage - keeps track of all security events
            /// </summary>
            private static void LogSecurityEvent(string eventType, string userId, string sessionId, string details)
            {
                // Run in background so it doesn't slow down the main process
                Task.Run(() =>
                {
                    try
                    {
                        using (var conn = OpenConn())
                        {
                            // Insert security event into log table
                            var sql = $@"
                                INSERT INTO {TableSecurityLog} VALUES (
                                    ?, current_timestamp(), ?, ?, ?, ?, ?
                                )";

                            using (var cmd = new OdbcCommand(sql, conn))
                            {
                                cmd.Parameters.AddWithValue("event_id", Guid.NewGuid().ToString());     // Unique ID
                                cmd.Parameters.AddWithValue("event_type", eventType);                   // What happened
                                cmd.Parameters.AddWithValue("user_id", userId ?? "unknown");            // Who did it
                                cmd.Parameters.AddWithValue("session_id", sessionId ?? "unknown");      // Which session
                                cmd.Parameters.AddWithValue("details", details.Length > 500 ?
                                    details.Substring(0, 500) : details);                               // What they tried
                                cmd.Parameters.AddWithValue("severity", GetSeverityLevel(eventType));   // How bad is it
                                cmd.ExecuteNonQuery();
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        // If logging fails, don't crash - just note it in debug log
                        System.Diagnostics.Debug.WriteLine($"Security logging failed: {ex.Message}");
                    }
                });
            }

            /// <summary>
            /// GET SEVERITY LEVEL - Determine how serious a security event is
            /// 
            /// Like threat levels (green, yellow, red) at airports
            /// </summary>
            private static string GetSeverityLevel(string eventType)
            {
                switch (eventType)
                {
                    case "SQL_INJECTION_ATTEMPT":   // Someone tried to hack the database
                    case "DANGEROUS_COMMAND":        // Someone tried to run system commands
                        return "HIGH";               // Red alert!

                    case "XSS_ATTEMPT":             // Someone tried to inject scripts
                    case "RATE_LIMIT_EXCEEDED":     // Someone is spamming
                        return "MEDIUM";            // Yellow alert

                    default:
                        return "LOW";               // Green - probably fine
                }
            }
        }

        /// <summary>
        /// OPTIMIZED VECTOR SEARCH - Find similar errors using AI fingerprints
        /// 
        /// WHAT THIS DOES:
        /// Like Shazam for error messages - finds matches even with different words
        /// Uses AI to understand meaning, not just exact text
        /// </summary>
        public class OptimizedVectorSearch
        {
            /// <summary>
            /// FIND SIMILAR - Main search function to find similar errors
            /// 
            /// Now with session limiting to prevent overload
            /// Returns list of similar errors ranked by how close they match
            /// </summary>
            public async Task<List<VectorMatch>> FindSimilarAsync(
                float[] queryVector,
                int limit = 10,
                double minSimilarity = SimilarityThreshold)  // Now uses the constant as default
            {
                // VALIDATION: Check vector dimensions are correct
                if (queryVector == null || queryVector.Length != VectorDimensions)
                {
                    throw new ArgumentException(
                        $"Query vector must be exactly {VectorDimensions} dimensions (OpenAI text-embedding-3-large). " +
                        $"Got {queryVector?.Length ?? 0} dimensions.");
                }

                // CONCURRENCY CONTROL: Wait for available session slot
                // This prevents too many searches from running at once and overwhelming the system
                await SessionSemaphore.WaitAsync();
                try
                {
                    // STEP 1: Check if we've searched for this exact thing recently (cache check)
                    // Like checking if you already Googled this today
                    var cacheKey = GenerateVectorCacheKey(queryVector);
                    var cachedResults = await VectorCacheManager.GetCachedResults(cacheKey);
                    if (cachedResults != null) return cachedResults; // Found in cache! Return instantly

                    // STEP 2: Not in cache, do the actual search lazy bum
                    var results = await HierarchicalVectorSearch(queryVector, limit, minSimilarity);

                    // STEP 3: Save results to cache for next time
                    // So if someone searches the same thing again, it's instant
                    await VectorCacheManager.CacheResults(cacheKey, results);

                    return results;
                }
                finally
                {
                    // ALWAYS release the session slot, even if error occurs
                    SessionSemaphore.Release();
                }
            }

            /// <summary>
            /// HIERARCHICAL SEARCH - Two-stage search for efficiency
            /// 
            /// Like looking for a book in a library:
            /// Stage 1: Go to the right section (approximate search)
            /// Stage 2: Find the exact book (precise search)
            /// </summary>
            private async Task<List<VectorMatch>> HierarchicalVectorSearch(float[] queryVector, int limit, double minSimilarity)
            {
                // VALIDATION: Double-check vector dimensions (belt and suspenders approach)
                if (queryVector.Length != VectorDimensions)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"WARNING: Vector dimension mismatch. Expected {VectorDimensions}, got {queryVector.Length}");
                    // Continue anyway but log the warning
                }

                // STAGE 1: Quick approximate search to find candidates
                // Like narrowing down to "computer books" section in library
                var candidates = await GetIndexCandidates(queryVector, Math.Min(VectorCandidateLimit, limit * 5));

                if (candidates.Count == 0) return new List<VectorMatch>(); // No candidates found

                // STAGE 2: Precise calculation on the candidates
                // Like reading each book's summary to find the best match
                var preciseMatches = new List<VectorMatch>();

                await Task.Run(() =>
                {
                    // Normalize the query vector (convert to standard scale)
                    NormalizeVector(queryVector);

                    // Check each candidate to see how similar it really is
                    foreach (var candidate in candidates)
                    {
                        // Parse the candidate's AI fingerprint from database
                        var candidateVector = ParseVector(candidate.VectorData);
                        if (candidateVector.Length == 0) continue; // Skip if invalid

                        // DIMENSION CHECK: Warn if candidate has wrong dimensions
                        if (candidateVector.Length != VectorDimensions)
                        {
                            System.Diagnostics.Debug.WriteLine(
                                $"WARNING: Candidate vector has {candidateVector.Length} dimensions, expected {VectorDimensions}. " +
                                $"Error hash: {candidate.ErrorHash}");
                            continue; // Skip this candidate
                        }

                        // Normalize candidate vector too
                        NormalizeVector(candidateVector);

                        // Calculate actual similarity (0 to 1, where 1 = identical)
                        var similarity = CosineSimilarity(queryVector, candidateVector);

                        // Only keep if similarity is above threshold
                        if (similarity >= minSimilarity)
                        {
                            preciseMatches.Add(new VectorMatch
                            {
                                ErrorHash = candidate.ErrorHash,
                                ResolutionSteps = candidate.ResolutionSteps,
                                Similarity = similarity,
                                ConfidenceScore = candidate.ConfidenceScore,
                                FeedbackCount = candidate.FeedbackCount
                            });
                        }
                    }
                });

                // Return top matches, best first
                // Weighted score: 70% similarity + 30% user feedback confidence
                return preciseMatches
                    .OrderByDescending(m => m.Similarity * (0.7 + m.ConfidenceScore * 0.3))
                    .Take(limit)
                    .ToList();
            }

            /// <summary>
            /// GET INDEX CANDIDATES - Fast approximate search using hash buckets
            /// 
            /// Like using a library's card catalog instead of checking every book
            /// LSH = Locality Sensitive Hashing (groups similar items together)
            /// </summary>
            private async Task<List<VectorCandidate>> GetIndexCandidates(float[] queryVector, int candidateLimit)
            {
                return await Task.Run(() =>
                {
                    var candidates = new List<VectorCandidate>();

                    try
                    {
                        // Generate hash codes that group similar vectors together
                        // Like organizing books by topic instead of alphabetically
                        var queryHashes = GenerateLSHHashes(queryVector);

                        using (var conn = OpenConn())
                        {
                            // Build SQL to find vectors in the same hash buckets
                            var hashConditions = queryHashes.Select((hash, i) => $"lsh_hash_{i % 3} = ?").ToArray();
                            var whereClause = string.Join(" OR ", hashConditions);

                            // Query: Find all errors that hash to similar buckets
                            var sql = $@"
                                SELECT DISTINCT kb.error_hash, kb.resolution_steps, idx.vector_data, 
                                       kb.confidence_score, kb.feedback_count
                                FROM {TableKB} kb
                                JOIN {TableVectorIndex} idx ON kb.error_hash = idx.error_hash
                                WHERE {whereClause}
                                ORDER BY kb.confidence_score DESC
                                LIMIT {candidateLimit}";

                            using (var cmd = new OdbcCommand(sql, conn))
                            {
                                // Add hash parameters
                                for (int i = 0; i < queryHashes.Length && i < 3; i++)
                                {
                                    cmd.Parameters.AddWithValue($"hash{i}", queryHashes[i]);
                                }

                                // Read results
                                using (var rdr = cmd.ExecuteReader())
                                {
                                    while (rdr.Read())
                                    {
                                        candidates.Add(new VectorCandidate
                                        {
                                            ErrorHash = rdr.GetString(0),           // Unique ID
                                            ResolutionSteps = rdr.GetString(1),     // The solution
                                            VectorData = rdr.GetString(2),          // AI fingerprint
                                            ConfidenceScore = rdr.GetDouble(3),     // How confident
                                            FeedbackCount = rdr.GetInt32(4)         // User ratings
                                        });
                                    }
                                }
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Index candidate retrieval failed: {ex.Message}");
                    }

                    return candidates;
                });
            }

            /// <summary>
            /// GENERATE LSH HASHES - Create hash buckets for fast searching
            /// 
            /// Like creating a zip code for vectors - similar vectors get similar codes
            /// This makes searching 1000x faster than comparing every single vector
            /// </summary>
            private string[] GenerateLSHHashes(float[] vector)
            {
                var hashes = new List<string>();

                // Use fixed random seed so same vector always gets same hashes
                var random = new Random(42); // 42 is arbitrary but must be consistent

                // Generate 3 different hash codes (like having 3 different indexes)
                for (int i = 0; i < 3; i++)
                {
                    var projection = 0.0;

                    // Project vector onto random direction (mathematical trick for hashing)
                    // Only use first 100 dimensions for speed (3072 would be slow)
                    int dimensionsToUse = Math.Min(vector.Length, 100);
                    for (int j = 0; j < dimensionsToUse; j++)
                    {
                        projection += vector[j] * (random.NextDouble() - 0.5);
                    }

                    // Convert projection to hash bucket number
                    var bucket = ((int)(projection * 100)).ToString();
                    hashes.Add(bucket);
                }

                return hashes.ToArray();
            }
        }

        /// <summary>
        /// VECTOR CACHE MANAGER - Speeds up repeated searches
        /// 
        /// WHAT THIS DOES:
        /// Like browser cache - if you search the same thing twice, second time is instant
        /// Remembers recent searches for 24 hours then refreshes
        /// </summary>
        public class VectorCacheManager
        {
            /// <summary>
            /// INITIALIZE CACHE TABLE - Create database tables for caching
            /// 
            /// Sets up the storage system for fast lookups
            /// </summary>
            public async Task InitializeCacheTable()
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Create table to store cached search results
                        var sql = $@"
                            CREATE TABLE IF NOT EXISTS {TableVectorCache} (
                                cache_key STRING,           -- Unique ID for this search
                                query_hash STRING,          -- Hash of what was searched
                                results_json STRING,        -- The search results (as JSON)
                                created_at TIMESTAMP,       -- When this was cached
                                hit_count INT DEFAULT 1,    -- How many times it's been used
                                last_accessed TIMESTAMP     -- When it was last used
                            ) USING DELTA";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        // Create index table for fast vector lookups
                        var indexSql = $@"
                            CREATE TABLE IF NOT EXISTS {TableVectorIndex} (
                                index_id STRING,        -- Unique ID for this index entry
                                error_hash STRING,      -- Links to the error
                                vector_data STRING,     -- The AI fingerprint (as JSON)
                                lsh_hash_0 STRING,      -- Hash bucket 1 (for fast search)
                                lsh_hash_1 STRING,      -- Hash bucket 2 (for fast search)
                                lsh_hash_2 STRING,      -- Hash bucket 3 (for fast search)
                                created_at TIMESTAMP    -- When this was indexed
                            ) USING DELTA";

                        using (var indexCmd = new OdbcCommand(indexSql, conn))
                        {
                            indexCmd.ExecuteNonQuery();
                        }

                        // Create security log table for tracking suspicious activity
                        var securitySql = $@"
                            CREATE TABLE IF NOT EXISTS {TableSecurityLog} (
                                event_id STRING,        -- Unique ID for this event
                                created_at TIMESTAMP,   -- When it happened
                                event_type STRING,      -- What type of event
                                user_id STRING,         -- Who did it
                                session_id STRING,      -- Which session
                                details STRING,         -- What happened
                                severity STRING         -- How serious (HIGH/MEDIUM/LOW)
                            ) USING DELTA";

                        using (var secCmd = new OdbcCommand(securitySql, conn))
                        {
                            secCmd.ExecuteNonQuery();
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Cache initialization failed: {ex.Message}");
                }
            }

            /// <summary>
            /// GET CACHED RESULTS - Check if we have this search cached
            /// 
            /// Like checking browser history - have we done this search recently?
            /// </summary>
            public static async Task<List<VectorMatch>> GetCachedResults(string cacheKey)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Look for cached results less than CacheExpiryHours old
                        var sql = $@"
                            SELECT results_json 
                            FROM {TableVectorCache} 
                            WHERE cache_key = ? 
                            AND created_at > current_timestamp() - INTERVAL {CacheExpiryHours} HOURS";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("cache_key", cacheKey);
                            var json = cmd.ExecuteScalar()?.ToString();

                            if (!string.IsNullOrEmpty(json))
                            {
                                // Found in cache! Update the hit counter
                                await UpdateCacheHitCount(cacheKey);
                                // Convert JSON back to objects and return
                                return JsonConvert.DeserializeObject<List<VectorMatch>>(json);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Cache retrieval failed: {ex.Message}");
                }

                return null; // Not in cache
            }

            /// <summary>
            /// CACHE RESULTS - Save search results for future use
            /// 
            /// Like saving a Google search so you don't have to search again
            /// </summary>
            public static async Task CacheResults(string cacheKey, List<VectorMatch> results)
            {
                try
                {
                    // Convert results to JSON for storage
                    var json = JsonConvert.SerializeObject(results);

                    using (var conn = OpenConn())
                    {
                        // MERGE: Update if exists, insert if new
                        var sql = $@"
                            MERGE INTO {TableVectorCache} t
                            USING (SELECT ? as cache_key, ? as results_json, current_timestamp() as created_at) s
                            ON t.cache_key = s.cache_key
                            WHEN MATCHED THEN UPDATE SET 
                                t.results_json = s.results_json,      -- Update with new results
                                t.created_at = s.created_at,          -- Reset expiry timer
                                t.hit_count = t.hit_count + 1         -- Increment usage counter
                            WHEN NOT MATCHED THEN INSERT VALUES (     -- New cache entry
                                s.cache_key, ?, s.results_json, s.created_at, 1, s.created_at
                            )";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("cache_key", cacheKey);
                            cmd.Parameters.AddWithValue("results_json", json);
                            cmd.Parameters.AddWithValue("query_hash", GenerateQueryHash(results));
                            await cmd.ExecuteNonQueryAsync();
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Cache storage failed: {ex.Message}");
                }
            }

            /// <summary>
            /// UPDATE CACHE HIT COUNT - Track how often cached results are used
            /// 
            /// Helps identify frequently searched items for optimization
            /// </summary>
            public static async Task UpdateCacheHitCount(string cacheKey)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        // Increment hit counter and update last access time
                        var sql = $@"
                            UPDATE {TableVectorCache} 
                            SET hit_count = hit_count + 1,           -- One more use
                                last_accessed = current_timestamp()   -- Update access time
                            WHERE cache_key = ?";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.Parameters.AddWithValue("cache_key", cacheKey);
                            await cmd.ExecuteNonQueryAsync();
                        }
                    }
                }
                catch
                {
                    // Don't fail if we can't update the counter
                    // Cache hit tracking is nice-to-have, not critical
                }
            }
        }

        // ============================================================================
        // HELPER METHODS - Utility functions used throughout the system
        // ============================================================================

        /// <summary>
        /// OPEN CONNECTION - Create database connection
        /// 
        /// Like dialing a phone number to the database
        /// </summary>
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        /// <summary>
        /// GENERATE VECTOR CACHE KEY - Create unique ID for a vector search
        /// 
        /// Like creating a fingerprint for the search query
        /// Same search always gets same key
        /// </summary>
        private static string GenerateVectorCacheKey(float[] vector)
        {
            using (var sha = SHA256.Create())
            {
                // Convert vector to bytes (4 bytes per float)
                var bytes = new byte[vector.Length * 4];
                Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);

                // Create SHA256 hash (like a fingerprint)
                var hash = sha.ComputeHash(bytes);

                // Convert to string and take first 16 characters
                return Convert.ToBase64String(hash).Substring(0, 16);
            }
        }

        /// <summary>
        /// GENERATE QUERY HASH - Create ID for search results
        /// 
        /// Used to identify unique result sets
        /// </summary>
        private static string GenerateQueryHash(List<VectorMatch> results)
        {
            // Combine all result IDs into one string
            var content = string.Join("|", results.Select(r => r.ErrorHash));

            using (var sha = SHA256.Create())
            {
                // Create hash of the combined IDs
                var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
                return Convert.ToBase64String(hash).Substring(0, 16);
            }
        }

        /// <summary>
        /// PARSE VECTOR - Convert JSON string back to number array
        /// 
        /// Database stores vectors as JSON text, this converts back to numbers
        /// </summary>
        private static float[] ParseVector(string vectorData)
        {
            try
            {
                // Use Newtonsoft.Json to parse JSON array
                var vector = JsonConvert.DeserializeObject<float[]>(vectorData) ?? Array.Empty<float>();

                // VALIDATION: Log warning if vector has wrong dimensions
                if (vector.Length > 0 && vector.Length != VectorDimensions)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"WARNING: Parsed vector has {vector.Length} dimensions, expected {VectorDimensions}");
                }

                return vector;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Vector parsing failed: {ex.Message}");
                // If parsing fails, return empty array
                return Array.Empty<float>();
            }
        }

        /// <summary>
        /// NORMALIZE VECTOR - Convert vector to unit length
        /// 
        /// Like adjusting volume so all songs play at same level; mega compression 
        /// Makes similarity comparison work correctly
        /// </summary>
        private static void NormalizeVector(float[] vector)
        {
            if (vector.Length == 0) return;

            // Calculate magnitude (length) of vector
            // Square root of sum of squares (Pythagorean theorem in N dimensions)
            var magnitude = Math.Sqrt(vector.Sum(x => x * x));

            // Avoid division by zero
            if (magnitude > 1e-8)
            {
                // Divide each element by magnitude to get unit vector
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] = (float)(vector[i] / magnitude);
                }
            }
        }

        /// <summary>
        /// COSINE SIMILARITY - Calculate how similar two vectors are
        /// 
        /// Returns value from -1 to 1:
        /// 1 = identical meaning
        /// 0 = unrelated
        /// -1 = opposite meaning
        /// 
        /// Like comparing two songs to see how similar they sound
        /// </summary>
        private static double CosineSimilarity(float[] a, float[] b)
        {
            // Vectors must be same length
            if (a.Length != b.Length)
            {
                System.Diagnostics.Debug.WriteLine(
                    $"WARNING: Vector dimension mismatch in cosine similarity. " +
                    $"Vector A: {a.Length} dims, Vector B: {b.Length} dims");
                return 0;
            }

            // Calculate dot product (sum of element-wise multiplication)
            var dotProduct = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
            }

            // For normalized vectors, dot product = cosine similarity
            return dotProduct;
        }
    }

    // ============================================================================
    // DATA STRUCTURES - Objects used to pass data around
    // ============================================================================

    /// <summary>
    /// VALIDATION RESULT - Output from security validation
    /// </summary>
    public class ValidationResult
    {
        public bool IsValid { get; set; }          // Did it pass security checks?
        public string CleanInput { get; set; }     // Sanitized safe version of input
        public string ErrorMessage { get; set; }   // Why it failed (if it did)
    }

    /// <summary>
    /// VECTOR MATCH - A similar error found by vector search
    /// </summary>
    public class VectorMatch
    {
        public string ErrorHash { get; set; }          // Unique ID of the error
        public string ResolutionSteps { get; set; }    // The solution text
        public double Similarity { get; set; }         // How similar (0-1 scale)
        public double ConfidenceScore { get; set; }    // User feedback confidence
        public int FeedbackCount { get; set; }         // Number of user ratings
    }

    /// <summary>
    /// VECTOR CANDIDATE - Potential match from initial search
    /// </summary>
    public class VectorCandidate
    {
        public string ErrorHash { get; set; }          // Unique ID
        public string ResolutionSteps { get; set; }    // Solution text
        public string VectorData { get; set; }         // AI fingerprint (as JSON)
        public double ConfidenceScore { get; set; }    // Confidence from feedback
        public int FeedbackCount { get; set; }         // Number of ratings
    }
}