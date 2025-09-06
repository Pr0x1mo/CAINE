using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace CAINE
{
    public class SolutionParser
    {
        public static List<string> ParseIntoSteps(string solution)
        {
            var steps = new List<string>();

            if (string.IsNullOrWhiteSpace(solution))
                return steps;

            // Check if it's already JSON array format
            if (solution.StartsWith("[") && solution.EndsWith("]"))
            {
                // Parse JSON array
                solution = solution.Trim('[', ']');
                steps = solution.Split(',')
                              .Select(s => s.Trim(' ', '"', '\''))
                              .Where(s => !string.IsNullOrEmpty(s))
                              .ToList();
            }
            // Check for numbered steps (1. 2. 3. or 1) 2) 3))
            else if (Regex.IsMatch(solution, @"^\d+[\.\)]", RegexOptions.Multiline))
            {
                var matches = Regex.Split(solution, @"(?=^\d+[\.\)])", RegexOptions.Multiline);
                steps = matches.Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
            }
            // Check for bullet points
            else if (solution.Contains("\n•") || solution.Contains("\n-") || solution.Contains("\n*"))
            {
                steps = solution.Split('\n')
                              .Where(s => !string.IsNullOrWhiteSpace(s))
                              .ToList();
            }
            // Check for newline-separated steps
            else if (solution.Contains("\n") && solution.Split('\n').Length > 2)
            {
                steps = solution.Split('\n')
                              .Where(s => !string.IsNullOrWhiteSpace(s))
                              .ToList();
            }
            else
            {
                // Single step solution
                steps.Add(solution);
            }

            return steps;
        }
    }
}