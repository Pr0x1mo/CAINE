using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace CAINE
{
    /// <summary>
    /// SOLUTION PARSER - Breaks solution text into individual steps
    /// 
    /// WHAT THIS DOES:
    /// Takes a blob of text containing a solution and intelligently splits it into
    /// individual steps that can be displayed one at a time or in a list.
    /// 
    /// LIKE A SMART TEXT SPLITTER:
    /// Recognizes different formats (numbered lists, bullet points, JSON arrays)
    /// and breaks them apart correctly so each step can be shown separately.
    /// </summary>
    public class SolutionParser
    {
        /// <summary>
        /// PARSE INTO STEPS - Convert solution text into a list of individual steps
        /// 
        /// HANDLES MULTIPLE FORMATS:
        /// - JSON arrays: ["step1", "step2", "step3"]
        /// - Numbered lists: 1. First step\n2. Second step
        /// - Bullet points: • Do this\n• Then that
        /// - Line breaks: Step one\nStep two\nStep three
        /// - Single paragraph: Returns as one step
        /// </summary>
        public static List<string> ParseIntoSteps(string solution)
        {
            // Create an empty list to store the individual steps we find
            var steps = new List<string>();

            // If there's no solution text (null, empty, or just whitespace), return empty list
            // This prevents crashes when there's no solution to parse
            if (string.IsNullOrWhiteSpace(solution))
                return steps;

            // FORMAT 1: JSON ARRAY - Database might store steps as ["step1", "step2"]
            // Check if the text starts with [ and ends with ] (JSON array format)
            if (solution.StartsWith("[") && solution.EndsWith("]"))
            {
                // Remove the [ and ] brackets from beginning and end
                solution = solution.Trim('[', ']');

                // Split by commas to get individual items
                // Remove quotes and spaces from each item
                // Filter out any empty items
                // Convert to a list
                steps = solution.Split(',')                           // Split: ["step1", "step2"] → ["step1"] and ["step2"]
                              .Select(s => s.Trim(' ', '"', '\''))    // Clean: ["step1"] → step1
                              .Where(s => !string.IsNullOrEmpty(s))   // Remove: any empty strings
                              .ToList();                              // Convert: array to list
            }
            // FORMAT 2: NUMBERED STEPS - Like "1. Do this" or "1) Do that"
            // Uses regex (pattern matching) to find lines starting with numbers followed by . or )
            else if (Regex.IsMatch(solution, @"^\d+[\.\)]", RegexOptions.Multiline))
            {
                // Split the text wherever we find a number followed by . or )
                // The (?=...) means "split here but keep the number as part of the next item"
                var matches = Regex.Split(solution, @"(?=^\d+[\.\)])", RegexOptions.Multiline);

                // Remove any empty or whitespace-only entries and convert to list
                steps = matches.Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            }
            // FORMAT 3: BULLET POINTS - Using • or - or * as bullets
            // Check if the text contains newline followed by a bullet character
            else if (solution.Contains("\n•") || solution.Contains("\n-") || solution.Contains("\n*"))
            {
                // Split by newline characters (line breaks)
                // Keep only non-empty lines
                steps = solution.Split('\n')                              // Split at each line break
                              .Where(s => !string.IsNullOrWhiteSpace(s))  // Keep only lines with actual text
                              .ToList();                                  // Convert to list
            }
            // FORMAT 4: NEWLINE-SEPARATED - Multiple lines of text (at least 3 lines)
            // Check if there are line breaks and at least 3 separate lines
            else if (solution.Contains("\n") && solution.Split('\n').Length > 2)
            {
                // Split by newlines and keep non-empty lines
                // Assumes each line is a separate step
                steps = solution.Split('\n')                              // Split at each line break
                              .Where(s => !string.IsNullOrWhiteSpace(s))  // Remove empty lines
                              .ToList();                                  // Convert to list
            }
            // FORMAT 5: SINGLE STEP - Just one paragraph or sentence
            else
            {
                // No special formatting found, treat the entire text as one step
                // This handles simple one-line solutions like "Restart the service"
                steps.Add(solution);
            }

            // Return the list of parsed steps
            // Could be empty list, single item, or multiple items
            return steps;
        }
    }
}