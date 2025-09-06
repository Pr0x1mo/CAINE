using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data.Odbc;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CAINE.Vector
{
    public class ScalableVectorManager
    {
        private const string TableKB = "default.cai_error_kb";
        private const string TableVectorIndex = "default.cai_vector_index";
        private const string TableVectorCache = "default.cai_vector_cache";
        private const string TableSecurityLog = "default.cai_security_log";
        private const string DsnName = "CAINE_Databricks";
        private const int VectorCandidateLimit = 300;  // Max number of similar errors to compare against
        // Performance optimization parameters
        private const int VectorDimensions = 3072; // text-embedding-3-large dimensions
        private const int IndexPageSize = 1000;
        private const int CacheExpiryHours = 24;
        private const double SimilarityThreshold = 0.70;

        // Security parameters
        private const int MaxQueryLength = 4000;
        private const int MaxDailyQueries = 1000;
        private const int MaxConcurrentSessions = 10;

        /// <summary>
        /// SECURITY LAYER: Comprehensive input validation and sanitization
        /// </summary>
        public class SecurityValidator
        {
            private static readonly Regex SqlInjectionPattern = new Regex(
                @"(\b(ALTER|CREATE|DELETE|DROP|EXEC(UTE)?|INSERT|MERGE|SELECT|UPDATE|UNION|USE)\b)|" +
                @"(--|/\*|\*/|;|'|xp_|sp_)",
                RegexOptions.IgnoreCase);

            private static readonly Regex XssPattern = new Regex(
                @"<script|javascript:|vbscript:|onload|onerror|onclick",
                RegexOptions.IgnoreCase);

            private static readonly string[] DangerousCommands = {
                "xp_cmdshell", "sp_oacreate", "sp_oamethod", "OPENROWSET",
                "BULK INSERT", "bcp", "sqlcmd", "osql"
            };

            public static ValidationResult ValidateInput(string input, string userId, string sessionId)
            {
                var result = new ValidationResult { IsValid = true, CleanInput = input };

                if (string.IsNullOrWhiteSpace(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input cannot be empty";
                    return result;
                }

                // Length validation
                if (input.Length > MaxQueryLength)
                {
                    result.IsValid = false;
                    result.ErrorMessage = $"Input exceeds maximum length of {MaxQueryLength} characters";
                    LogSecurityEvent("EXCESSIVE_LENGTH", userId, sessionId, input.Substring(0, 100));
                    return result;
                }

                // SQL injection detection
                if (SqlInjectionPattern.IsMatch(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input contains potentially dangerous SQL patterns";
                    LogSecurityEvent("SQL_INJECTION_ATTEMPT", userId, sessionId, input);
                    return result;
                }

                // XSS detection
                if (XssPattern.IsMatch(input))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Input contains potentially dangerous script content";
                    LogSecurityEvent("XSS_ATTEMPT", userId, sessionId, input);
                    return result;
                }

                // Dangerous command detection
                // Dangerous command detection
                var inputLower = input.ToLowerInvariant();
                foreach (var cmd in DangerousCommands)
                {
                    if (inputLower.Contains(cmd.ToLowerInvariant()))
                    {
                        result.IsValid = false;
                        result.ErrorMessage = "Input contains restricted commands";
                        LogSecurityEvent("DANGEROUS_COMMAND", userId, sessionId, input);
                        return result;
                    }
                }

                // Rate limiting check
                if (!CheckRateLimit(userId))
                {
                    result.IsValid = false;
                    result.ErrorMessage = "Rate limit exceeded. Please try again later.";
                    LogSecurityEvent("RATE_LIMIT_EXCEEDED", userId, sessionId, "");
                    return result;
                }

                // Clean and sanitize input
                result.CleanInput = SanitizeInput(input);
                return result;
            }

            private static string SanitizeInput(string input)
            {
                return input.Replace("--", "- -")           // Break SQL comments
                            .Replace("/*", "/ *")            // Break block comments
                            .Replace("*/", "* /")
                            .Replace(";", ",")               // Replace statement terminators
                            .Replace("'", "&#39;")           // Escape single quotes
                            .Replace("\"", "&quot;")         // Escape double quotes
                            .Replace("<", "&lt;")            // Escape HTML
                            .Replace(">", "&gt;")
                            .Replace("&", "&amp;")
                            .Trim();
            }

            private static bool CheckRateLimit(string userId)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
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
                            return count < MaxDailyQueries;
                        }
                    }
                }
                catch
                {
                    return true; // Fail open if rate limiting system is down
                }
            }

            private static void LogSecurityEvent(string eventType, string userId, string sessionId, string details)
            {
                Task.Run(() =>
                {
                    try
                    {
                        using (var conn = OpenConn())
                        {
                            var sql = $@"
                                INSERT INTO {TableSecurityLog} VALUES (
                                    ?, current_timestamp(), ?, ?, ?, ?, ?
                                )";

                            using (var cmd = new OdbcCommand(sql, conn))
                            {
                                cmd.Parameters.AddWithValue("event_id", Guid.NewGuid().ToString());
                                cmd.Parameters.AddWithValue("event_type", eventType);
                                cmd.Parameters.AddWithValue("user_id", userId ?? "unknown");
                                cmd.Parameters.AddWithValue("session_id", sessionId ?? "unknown");
                                cmd.Parameters.AddWithValue("details", details.Length > 500 ? details.Substring(0, 500) : details);
                                cmd.Parameters.AddWithValue("severity", GetSeverityLevel(eventType));
                                cmd.ExecuteNonQuery();
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Security logging failed: {ex.Message}");
                    }
                });
            }

            private static string GetSeverityLevel(string eventType)
            {
                switch (eventType)
                {
                    case "SQL_INJECTION_ATTEMPT":
                    case "DANGEROUS_COMMAND":
                        return "HIGH";
                    case "XSS_ATTEMPT":
                    case "RATE_LIMIT_EXCEEDED":
                        return "MEDIUM";
                    default:
                        return "LOW";
                }
            }
        }

        /// <summary>
        /// OPTIMIZED VECTOR SEARCH: Scalable similarity search with intelligent caching
        /// </summary>
        public class OptimizedVectorSearch
        {
            public async Task<List<VectorMatch>> FindSimilarAsync(float[] queryVector, int limit = 10, double minSimilarity = 0.70)
            {
                // Check cache first
                var cacheKey = GenerateVectorCacheKey(queryVector);
                var cachedResults = await VectorCacheManager.GetCachedResults(cacheKey);
                if (cachedResults != null) return cachedResults;

                // Perform hierarchical search for scalability
                var results = await HierarchicalVectorSearch(queryVector, limit, minSimilarity);

                // Cache results for future queries
                await VectorCacheManager.CacheResults(cacheKey, results);

                return results;
            }

            /// <summary>
            /// HIERARCHICAL SEARCH: Multi-level approach for large datasets
            /// Level 1: Index-based approximate search
            /// Level 2: Precise similarity calculation on candidates
            /// </summary>
            private async Task<List<VectorMatch>> HierarchicalVectorSearch(float[] queryVector, int limit, double minSimilarity)
            {
                // LEVEL 1: Fast approximate search using vector index
                var candidates = await GetIndexCandidates(queryVector, Math.Min(VectorCandidateLimit, limit * 5));

                if (candidates.Count == 0) return new List<VectorMatch>();

                // LEVEL 2: Precise similarity calculation
                var preciseMatches = new List<VectorMatch>();

                await Task.Run(() =>
                {
                    NormalizeVector(queryVector);

                    foreach (var candidate in candidates)
                    {
                        var candidateVector = ParseVector(candidate.VectorData);
                        if (candidateVector.Length == 0) continue;

                        NormalizeVector(candidateVector);
                        var similarity = CosineSimilarity(queryVector, candidateVector);

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

                // Return top matches sorted by weighted score
                return preciseMatches
                    .OrderByDescending(m => m.Similarity * (0.7 + m.ConfidenceScore * 0.3))
                    .Take(limit)
                    .ToList();
            }

            /// <summary>
            /// INDEX-BASED CANDIDATE RETRIEVAL: Fast approximate search
            /// Uses LSH (Locality Sensitive Hashing) simulation for scalability
            /// </summary>
            private async Task<List<VectorCandidate>> GetIndexCandidates(float[] queryVector, int candidateLimit)
            {
                return await Task.Run(() =>
                {
                    var candidates = new List<VectorCandidate>();

                    try
                    {
                        // Generate hash buckets for LSH-style approximate matching
                        var queryHashes = GenerateLSHHashes(queryVector);

                        using (var conn = OpenConn())
                        {
                            // Query multiple hash buckets to find candidates
                            var hashConditions = queryHashes.Select((hash, i) => $"lsh_hash_{i % 3} = ?").ToArray();
                            var whereClause = string.Join(" OR ", hashConditions);

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
                                for (int i = 0; i < queryHashes.Length && i < 3; i++)
                                {
                                    cmd.Parameters.AddWithValue($"hash{i}", queryHashes[i]);
                                }

                                using (var rdr = cmd.ExecuteReader())
                                {
                                    while (rdr.Read())
                                    {
                                        candidates.Add(new VectorCandidate
                                        {
                                            ErrorHash = rdr.GetString(0),
                                            ResolutionSteps = rdr.GetString(1),
                                            VectorData = rdr.GetString(2),
                                            ConfidenceScore = rdr.GetDouble(3),
                                            FeedbackCount = rdr.GetInt32(4)
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
            /// LSH HASH GENERATION: Creates locality-sensitive hashes for fast approximate matching
            /// </summary>
            private string[] GenerateLSHHashes(float[] vector)
            {
                var hashes = new List<string>();

                // Simple LSH using random projections
                var random = new Random(42); // Fixed seed for consistency

                for (int i = 0; i < 3; i++) // Generate 3 hash buckets
                {
                    var projection = 0.0;
                    for (int j = 0; j < Math.Min(vector.Length, 100); j++) // Use first 100 dimensions
                    {
                        projection += vector[j] * (random.NextDouble() - 0.5);
                    }

                    // Convert to hash bucket
                    var bucket = ((int)(projection * 100)).ToString();
                    hashes.Add(bucket);
                }

                return hashes.ToArray();
            }
        }

        /// <summary>
        /// INTELLIGENT CACHING: Reduces computation for repeated queries
        /// </summary>
        public class VectorCacheManager
        {
            public async Task InitializeCacheTable()
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            CREATE TABLE IF NOT EXISTS {TableVectorCache} (
                                cache_key STRING,
                                query_hash STRING,
                                results_json STRING,
                                created_at TIMESTAMP,
                                hit_count INT DEFAULT 1,
                                last_accessed TIMESTAMP
                            ) USING DELTA";

                        using (var cmd = new OdbcCommand(sql, conn))
                        {
                            cmd.ExecuteNonQuery();
                        }

                        // Create index table for LSH
                        var indexSql = $@"
                            CREATE TABLE IF NOT EXISTS {TableVectorIndex} (
                                index_id STRING,
                                error_hash STRING,
                                vector_data STRING,
                                lsh_hash_0 STRING,
                                lsh_hash_1 STRING,
                                lsh_hash_2 STRING,
                                created_at TIMESTAMP
                            ) USING DELTA";

                        using (var indexCmd = new OdbcCommand(indexSql, conn))
                        {
                            indexCmd.ExecuteNonQuery();
                        }

                        // Create security log table
                        var securitySql = $@"
                            CREATE TABLE IF NOT EXISTS {TableSecurityLog} (
                                event_id STRING,
                                created_at TIMESTAMP,
                                event_type STRING,
                                user_id STRING,
                                session_id STRING,
                                details STRING,
                                severity STRING
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

            public static async Task<List<VectorMatch>> GetCachedResults(string cacheKey)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
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
                                // Update hit count
                                await UpdateCacheHitCount(cacheKey);
                                return JsonConvert.DeserializeObject<List<VectorMatch>>(json);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Cache retrieval failed: {ex.Message}");
                }

                return null;
            }

            public static async Task CacheResults(string cacheKey, List<VectorMatch> results)
            {
                try
                {
                    var json = JsonConvert.SerializeObject(results);

                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            MERGE INTO {TableVectorCache} t
                            USING (SELECT ? as cache_key, ? as results_json, current_timestamp() as created_at) s
                            ON t.cache_key = s.cache_key
                            WHEN MATCHED THEN UPDATE SET 
                                t.results_json = s.results_json, 
                                t.created_at = s.created_at,
                                t.hit_count = t.hit_count + 1
                            WHEN NOT MATCHED THEN INSERT VALUES (
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

            public static async Task UpdateCacheHitCount(string cacheKey)
            {
                try
                {
                    using (var conn = OpenConn())
                    {
                        var sql = $@"
                            UPDATE {TableVectorCache} 
                            SET hit_count = hit_count + 1, last_accessed = current_timestamp()
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
                    // Ignore cache update failures
                }
            }
        }

        // Helper methods and data structures
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        private static string GenerateVectorCacheKey(float[] vector)
        {
            using (var sha = SHA256.Create())
            {
                var bytes = new byte[vector.Length * 4];
                Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
                var hash = sha.ComputeHash(bytes);
                return Convert.ToBase64String(hash).Substring(0, 16);
            }
        }

        private static string GenerateQueryHash(List<VectorMatch> results)
        {
            var content = string.Join("|", results.Select(r => r.ErrorHash));
            using (var sha = SHA256.Create())
            {
                var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(content));
                return Convert.ToBase64String(hash).Substring(0, 16);
            }
        }

        private static float[] ParseVector(string vectorData)
        {
            try
            {
                return JsonConvert.DeserializeObject<float[]>(vectorData) ?? Array.Empty<float>();
            }
            catch
            {
                return Array.Empty<float>();
            }
        }

        private static void NormalizeVector(float[] vector)
        {
            if (vector.Length == 0) return;

            var magnitude = Math.Sqrt(vector.Sum(x => x * x));
            if (magnitude > 1e-8)
            {
                for (int i = 0; i < vector.Length; i++)
                {
                    vector[i] = (float)(vector[i] / magnitude);
                }
            }
        }

        private static double CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length) return 0;

            var dotProduct = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                dotProduct += a[i] * b[i];
            }

            return dotProduct;
        }
    }

    // Data structures
    public class ValidationResult
    {
        public bool IsValid { get; set; }
        public string CleanInput { get; set; }
        public string ErrorMessage { get; set; }
    }

    public class VectorMatch
    {
        public string ErrorHash { get; set; }
        public string ResolutionSteps { get; set; }
        public double Similarity { get; set; }
        public double ConfidenceScore { get; set; }
        public int FeedbackCount { get; set; }
    }

    public class VectorCandidate
    {
        public string ErrorHash { get; set; }
        public string ResolutionSteps { get; set; }
        public string VectorData { get; set; }
        public double ConfidenceScore { get; set; }
        public int FeedbackCount { get; set; }
    }
}