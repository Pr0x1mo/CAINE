using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace CAINE
{
    public class FuzzySearchEngine
    {
        // Synonym mappings
        private readonly Dictionary<string, HashSet<string>> synonyms = new Dictionary<string, HashSet<string>>
        {
            ["timeout"] = new HashSet<string> { "timed out", "time out", "hung", "freeze", "frozen" },
            ["connection"] = new HashSet<string> { "connectivity", "network", "conn", "connect" },
            ["permission"] = new HashSet<string> { "access denied", "unauthorized", "forbidden", "denied" },
            ["failed"] = new HashSet<string> { "failure", "error", "exception", "fail" },
            ["database"] = new HashSet<string> { "db", "sql", "table", "schema" },
            ["login"] = new HashSet<string> { "logon", "signin", "authenticate", "auth" }
        };

        /// <summary>
        /// Calculate Levenshtein distance for fuzzy matching
        /// </summary>
        private int LevenshteinDistance(string s1, string s2)
        {
            s1 = s1.ToLower();
            s2 = s2.ToLower();

            int[,] d = new int[s1.Length + 1, s2.Length + 1];

            for (int i = 0; i <= s1.Length; i++)
                d[i, 0] = i;
            for (int j = 0; j <= s2.Length; j++)
                d[0, j] = j;

            for (int i = 1; i <= s1.Length; i++)
            {
                for (int j = 1; j <= s2.Length; j++)
                {
                    int cost = s1[i - 1] == s2[j - 1] ? 0 : 1;
                    d[i, j] = Math.Min(
                        Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                        d[i - 1, j - 1] + cost
                    );
                }
            }

            return d[s1.Length, s2.Length];
        }

        /// <summary>
        /// Calculate similarity score (0-1) between two strings
        /// </summary>
        public double GetSimilarity(string s1, string s2)
        {
            if (string.IsNullOrEmpty(s1) || string.IsNullOrEmpty(s2))
                return 0;

            if (s1.Equals(s2, StringComparison.OrdinalIgnoreCase))
                return 1.0;

            int distance = LevenshteinDistance(s1, s2);
            int maxLength = Math.Max(s1.Length, s2.Length);

            return 1.0 - ((double)distance / maxLength);
        }

        /// <summary>
        /// Expand query with synonyms
        /// </summary>
        public List<string> ExpandWithSynonyms(string query)
        {
            var expanded = new HashSet<string> { query.ToLower() };
            var words = query.ToLower().Split(' ');

            foreach (var word in words)
            {
                foreach (var synGroup in synonyms)
                {
                    if (synGroup.Key == word || synGroup.Value.Contains(word))
                    {
                        expanded.Add(synGroup.Key);
                        foreach (var syn in synGroup.Value)
                        {
                            expanded.Add(syn);
                        }
                    }
                }
            }

            return expanded.ToList();
        }

        /// <summary>
        /// Fuzzy search with scoring
        /// </summary>
        public double CalculateFuzzyScore(string query, string target)
        {
            query = query.ToLower();
            target = target.ToLower();

            double score = 0;

            // 1. Exact match = highest score
            if (target.Contains(query))
                return 1.0;

            // 2. Check individual words
            var queryWords = query.Split(' ').Where(w => w.Length > 2).ToArray();
            var targetWords = target.Split(' ').Where(w => w.Length > 2).ToArray();

            // Word match scoring
            foreach (var qWord in queryWords)
            {
                double bestWordScore = 0;

                foreach (var tWord in targetWords)
                {
                    // Exact word match
                    if (qWord == tWord)
                    {
                        bestWordScore = Math.Max(bestWordScore, 0.9);
                    }
                    // Fuzzy word match
                    else
                    {
                        double similarity = GetSimilarity(qWord, tWord);
                        if (similarity > 0.8) // 80% similarity threshold
                        {
                            bestWordScore = Math.Max(bestWordScore, similarity * 0.7);
                        }
                    }
                }

                score += bestWordScore;
            }

            // 3. Check synonyms
            var expandedQuery = ExpandWithSynonyms(query);
            foreach (var synonym in expandedQuery)
            {
                if (target.Contains(synonym))
                {
                    score += 0.5;
                    break;
                }
            }

            // Normalize score
            return Math.Min(1.0, score / Math.Max(1, queryWords.Length));
        }

        /// <summary>
        /// N-gram matching for partial matches
        /// </summary>
        public double GetNGramSimilarity(string s1, string s2, int n = 3)
        {
            var ngrams1 = GetNGrams(s1.ToLower(), n);
            var ngrams2 = GetNGrams(s2.ToLower(), n);

            if (!ngrams1.Any() || !ngrams2.Any())
                return 0;

            var intersection = ngrams1.Intersect(ngrams2).Count();
            var union = ngrams1.Union(ngrams2).Count();

            return (double)intersection / union;
        }

        private HashSet<string> GetNGrams(string text, int n)
        {
            var ngrams = new HashSet<string>();
            for (int i = 0; i <= text.Length - n; i++)
            {
                ngrams.Add(text.Substring(i, n));
            }
            return ngrams;
        }
    }

    /// <summary>
    /// Enhanced search result with fuzzy scoring
    /// </summary>
    public class FuzzySearchResult
    {
        public string ErrorHash { get; set; }
        public string ErrorText { get; set; }
        public string ResolutionSteps { get; set; }
        public double FuzzyScore { get; set; }
        public double ExactMatchBonus { get; set; }
        public double SynonymBonus { get; set; }
        public double TotalScore => FuzzyScore + ExactMatchBonus + SynonymBonus;
    }
}