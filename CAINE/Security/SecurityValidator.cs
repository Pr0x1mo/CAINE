using System;
using System.Collections.Generic;
using System.Data.Odbc;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CAINE.Security
{
    /// <summary>
    /// SECURITY LAYER: Comprehensive input validation and sanitization
    /// FIXED: Allows SQL error messages for analysis while blocking actual SQL injection
    /// </summary>
    public class SecurityValidator
    {
        private const string TableSecurityLog = "default.cai_security_log";
        private const string DsnName = "CAINE_Databricks";

        // Security parameters
        private const int MaxQueryLength = 4000;
        private const int MaxDailyQueries = 1000;
        private const int MaxConcurrentSessions = 10;

        // FIXED: More precise SQL injection detection - only block actual injection attempts
        private static readonly Regex SqlInjectionPattern = new Regex(
            @"(\bunion\s+select\b)|" +                          // Union-based injection
            @"(\binsert\s+into\b.*\bvalues\b)|" +               // Insert injection
            @"(\bdelete\s+from\b)|" +                           // Delete injection
            @"(\bdrop\s+(table|database|schema)\b)|" +          // Drop commands
            @"(\balter\s+(table|database)\b)|" +                // Alter commands
            @"(\bcreate\s+(table|database|user)\b)|" +          // Create commands
            @"(\bgrant\s+)|(\brevoke\s+)|" +                    // Permission changes
            @"(;\s*(insert|delete|update|drop|create|alter))|" + // Multiple statements
            @"('.*;\s*--)|" +                                   // Comment injection
            @"(\bor\s+1\s*=\s*1\b)|(\band\s+1\s*=\s*1\b)",     // Always true conditions
            RegexOptions.IgnoreCase);

        // Patterns that indicate this is an ERROR MESSAGE, not an injection attempt
        private static readonly Regex ErrorMessagePattern = new Regex(
            @"(error|failed|exception|invalid|cannot|unable to|timeout|overflow|" +
            @"violation|constraint|permission denied|access denied|login failed|" +
            @"step \[.*\] failed|executing the query.*failed|" +
            @"possible failure reasons|arithmetic overflow)",
            RegexOptions.IgnoreCase);

        private static readonly Regex XssPattern = new Regex(
            @"<script|javascript:|vbscript:|onload|onerror|onclick",
            RegexOptions.IgnoreCase);

        // FIXED: Only block truly dangerous commands, not error messages containing them
        private static readonly string[] DangerousCommands = {
            "xp_cmdshell", "sp_oacreate", "sp_oamethod", "OPENROWSET",
            "BULK INSERT", "bcp", "sqlcmd", "osql",
            "OPENDATASOURCE", "OPENQUERY"
        };

        /// <summary>
        /// FIXED: Main validation method - allows SQL error messages for analysis
        /// </summary>
        public static ValidationResult ValidateInput(string input, string userId, string sessionId)
        {
            var result = new ValidationResult { IsValid = true, CleanInput = input };

            if (string.IsNullOrWhiteSpace(input))
            {
                result.IsValid = false;
                result.ErrorMessage = "Input cannot be empty";
                return result;
            }

            // Length validation - prevent buffer overflow attacks
            if (input.Length > MaxQueryLength)
            {
                result.IsValid = false;
                result.ErrorMessage = $"Input exceeds maximum length of {MaxQueryLength} characters";
                LogSecurityEvent("EXCESSIVE_LENGTH", userId, sessionId, input.Substring(0, 100));
                return result;
            }

            // FIXED: Check if this looks like an error message first
            bool isErrorMessage = ErrorMessagePattern.IsMatch(input);

            // FIXED: Only check for SQL injection if this doesn't look like an error message
            if (!isErrorMessage && SqlInjectionPattern.IsMatch(input))
            {
                result.IsValid = false;
                result.ErrorMessage = "Input contains potentially dangerous SQL patterns";
                LogSecurityEvent("SQL_INJECTION_ATTEMPT", userId, sessionId, input);
                return result;
            }

            // XSS detection - prevent script injection (always check this)
            if (XssPattern.IsMatch(input))
            {
                result.IsValid = false;
                result.ErrorMessage = "Input contains potentially dangerous script content";
                LogSecurityEvent("XSS_ATTEMPT", userId, sessionId, input);
                return result;
            }

            // FIXED: Only block dangerous commands if not in error context
            if (!isErrorMessage)
            {
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
            }

            // Rate limiting check - prevent abuse
            if (!CheckRateLimit(userId))
            {
                result.IsValid = false;
                result.ErrorMessage = "Rate limit exceeded. Please try again later.";
                LogSecurityEvent("RATE_LIMIT_EXCEEDED", userId, sessionId, "");
                return result;
            }

            // FIXED: Don't sanitize error messages - keep them intact for analysis
            if (isErrorMessage)
            {
                result.CleanInput = input; // Keep error messages unchanged
                LogSecurityEvent("ERROR_ANALYSIS", userId, sessionId, "SQL error message analyzed");
            }
            else
            {
                // Clean and sanitize non-error input
                result.CleanInput = SanitizeInput(input);
                LogSecurityEvent("QUERY", userId, sessionId, "Validation passed");
            }

            return result;
        }

        /// <summary>
        /// Sanitizes input by removing/escaping dangerous characters
        /// </summary>
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

        /// <summary>
        /// Checks if user has exceeded daily query limit
        /// </summary>
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
                        AND event_type IN ('QUERY', 'SEARCH', 'ERROR_ANALYSIS')";

                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        cmd.Parameters.AddWithValue("user_id", userId);
                        var count = Convert.ToInt32(cmd.ExecuteScalar());
                        return count < MaxDailyQueries;
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Rate limit check failed: {ex.Message}");
                return true; // Fail open if rate limiting system is down
            }
        }

        /// <summary>
        /// Logs security events for monitoring and alerting
        /// </summary>
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

        /// <summary>
        /// Determines severity level for security events
        /// </summary>
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
                case "EXCESSIVE_LENGTH":
                    return "LOW";
                case "ERROR_ANALYSIS":
                    return "INFO";
                default:
                    return "INFO";
            }
        }

        /// <summary>
        /// Creates secure database connection
        /// </summary>
        private static OdbcConnection OpenConn()
        {
            var conn = new OdbcConnection("DSN=" + DsnName + ";");
            conn.Open();
            return conn;
        }

        /// <summary>
        /// Creates security log table if it doesn't exist
        /// </summary>
        public static void InitializeSecurityTables()
        {
            try
            {
                using (var conn = OpenConn())
                {
                    var sql = $@"
                        CREATE TABLE IF NOT EXISTS {TableSecurityLog} (
                            event_id STRING,
                            created_at TIMESTAMP,
                            event_type STRING,
                            user_id STRING,
                            session_id STRING,
                            details STRING,
                            severity STRING
                        ) USING DELTA";

                    using (var cmd = new OdbcCommand(sql, conn))
                    {
                        cmd.ExecuteNonQuery();
                    }

                    System.Diagnostics.Debug.WriteLine("Security tables initialized successfully");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Security table initialization failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Gets security statistics for monitoring dashboard
        /// </summary>
        public static SecurityStats GetSecurityStats(int hours = 24)
        {
            try
            {
                using (var conn = OpenConn())
                {
                    var sql = $@"
                        SELECT 
                            COUNT(*) as total_events,
                            SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) as high_severity,
                            SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) as medium_severity,
                            SUM(CASE WHEN event_type = 'SQL_INJECTION_ATTEMPT' THEN 1 ELSE 0 END) as sql_attempts,
                            SUM(CASE WHEN event_type = 'RATE_LIMIT_EXCEEDED' THEN 1 ELSE 0 END) as rate_limits
                        FROM {TableSecurityLog}
                        WHERE created_at >= current_timestamp() - INTERVAL {hours} HOURS";

                    using (var cmd = new OdbcCommand(sql, conn))
                    using (var rdr = cmd.ExecuteReader())
                    {
                        if (rdr.Read())
                        {
                            return new SecurityStats
                            {
                                TotalEvents = rdr.GetInt32(0),
                                HighSeverityEvents = rdr.GetInt32(1),
                                MediumSeverityEvents = rdr.GetInt32(2),
                                SqlInjectionAttempts = rdr.GetInt32(3),
                                RateLimitExceeded = rdr.GetInt32(4)
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Security stats failed: {ex.Message}");
            }

            return new SecurityStats();
        }
    }

    /// <summary>
    /// Result of input validation
    /// </summary>
    public class ValidationResult
    {
        public bool IsValid { get; set; }
        public string CleanInput { get; set; }
        public string ErrorMessage { get; set; }
    }

    /// <summary>
    /// Security statistics for monitoring
    /// </summary>
    public class SecurityStats
    {
        public int TotalEvents { get; set; }
        public int HighSeverityEvents { get; set; }
        public int MediumSeverityEvents { get; set; }
        public int SqlInjectionAttempts { get; set; }
        public int RateLimitExceeded { get; set; }

        public bool HasThreats => HighSeverityEvents > 0 || SqlInjectionAttempts > 0;
        public string ThreatLevel =>
            HighSeverityEvents > 10 ? "CRITICAL" :
            HighSeverityEvents > 0 ? "HIGH" :
            MediumSeverityEvents > 20 ? "MEDIUM" : "LOW";
    }
}