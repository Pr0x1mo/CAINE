# CAINE - Computer-Aided Intelligence for Network Errors

## What is CAINE?

CAINE is an intelligent error resolution system that learns from your organization's experience. Think of it as a smart assistant that remembers every database and network error your team has ever solved, and gets better at solving new problems over time.

### In Simple Terms
When you encounter an error message:
1. Paste it into CAINE
2. CAINE searches its memory for similar problems
3. If found, it shows you the solution that worked before
4. If not found, it asks ChatGPT for help
5. You can teach it new solutions
6. Users rate solutions (thumbs up/down)
7. CAINE learns what works and what doesn't

## Version 2.1 - Interactive Troubleshooting Assistant

### New Feature: Decision Tree Navigation
CAINE now includes an interactive troubleshooting assistant that guides users through problem-solving step by step.

#### Features:
- **Visual Decision Tree**: Navigate through yes/no questions to find solutions
- **Adaptive Learning**: Tree adapts based on user feedback
- **Database Integration**: Pulls real solutions from your knowledge base
- **Success Rate Tracking**: Shows confidence for each path
- **Visual Flowchart**: Graphical representation of troubleshooting paths

#### How to Use:
1. Search for an error first
2. Click "🌳 Interactive" button
3. Answer yes/no questions
4. Follow the guided path to resolution
5. Rate if the solution worked

#### Technical Implementation:
- Custom WPF tree visualization control
- Real-time path weight updates
- Integration with ML clustering for categorization
- Persistent path tracking in database

## Version 2.0 - Machine Learning Enhancement

### What's New
CAINE now has its own "brain" that learns patterns from your errors, not just memorizing exact matches. It can recognize that "connection timeout" and "network unreachable" are similar types of problems, even if the exact words differ.

### Machine Learning Components

#### For Non-Technical Users
- **Clustering**: CAINE automatically groups similar errors together (like sorting mail into categories)
- **Neural Network**: Learns complex patterns, like a brain learning from experience
- **Decision Trees**: Makes yes/no decisions to predict if a solution will work
- **Trend Analysis**: Predicts when errors are likely to happen again
- **Anomaly Detection**: Spots unusual new problems that need special attention

#### For Technical Users
- **K-Means Clustering**: Automatically categorizes errors into 10 discovered groups
- **C4.5 Decision Trees**: Predicts solution effectiveness based on error features
- **Support Vector Machines (SVM)**: Non-linear classification with Gaussian kernels
- **Custom Neural Network**: 3-layer feedforward with backpropagation
- **Time Series Analysis**: Exponential smoothing for occurrence prediction
- **Auto-retraining Pipeline**: Models update every 24 hours with new data

## How CAINE Searches (5 Layers)

1. **Exact Match** - Have we seen this exact error before?
2. **Keyword Search** - Find errors with the same important words
3. **AI Similarity** - Use OpenAI embeddings to find similar meanings
4. **Pattern Recognition** - Check if it matches known error patterns
5. **ML Prediction** (v2.0) - Use machine learning to predict best solution

## System Requirements

### Software
- Windows with .NET Framework 4.7.2+
- Visual Studio 2019 or later
- Databricks account with ODBC connection
- OpenAI API key

### NuGet Packages
```xml
<PackageReference Include="Accord.MachineLearning" Version="3.8.0" />
<PackageReference Include="Accord.MachineLearning.DecisionTrees" Version="3.8.0" />
<PackageReference Include="Accord.Statistics" Version="3.8.0" />
<PackageReference Include="Accord.Math" Version="3.8.0" />
<PackageReference Include="MathNet.Numerics" Version="5.0.0" />
<PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pr0x1mo/CAINE.git
   ```

2. Set up environment variables:
   ```bash
   setx OPENAI_API_KEY "your-api-key-here"
   ```

3. Configure Databricks ODBC:
   - Create DSN named "CAINE_Databricks"
   - Point to your Databricks workspace

4. Open in Visual Studio and restore NuGet packages

5. Build and run

## Database Setup

CAINE will automatically create these tables on first run:
- `cai_error_kb` - Knowledge base of error solutions
- `cai_solution_feedback` - User ratings of solutions
- `cai_error_patterns` - Pattern matching rules
- `cai_kb_versions` - Version history
- `cai_tree_paths` - Interactive troubleshooting paths (v2.1)

For the interactive tree feature, manually create:
```sql
CREATE TABLE IF NOT EXISTS default.cai_tree_paths (
    path_id STRING,
    session_id STRING,
    error_hash STRING,
    path_json STRING,
    was_successful BOOLEAN,
    duration_seconds INT,
    nodes_visited INT,
    created_at TIMESTAMP,
    created_by STRING
) USING DELTA;
```

## Machine Learning Activation

### Requirements
- Minimum 50 feedback entries (user ratings) in database
- At least 4GB RAM for model training

### Check Your Status
```sql
SELECT COUNT(*) as training_samples
FROM default.cai_error_kb kb
JOIN default.cai_solution_feedback fb ON kb.error_hash = fb.solution_hash
WHERE fb.was_helpful IS NOT NULL;
```

If < 50: ML features partially active (clustering only)
If ≥ 50: Full ML features activate on next startup

## How It Works

### For End Users
1. **Search**: Paste error → Click Search → Get solution with confidence rating
2. **Feedback**: Click 👍 or 👎 to rate if solution worked
3. **Teach**: If no solution found, enter steps that work → Click Teach
4. **AI Help**: Click "Use CAINE API" for ChatGPT assistance
5. **Interactive**: Click "🌳 Interactive" for guided troubleshooting (v2.1)

### What Happens Behind the Scenes
1. **Error Processing**: Normalizes and creates SHA256 hash fingerprint
2. **Multi-Layer Search**: Tries 5 different search strategies
3. **Confidence Scoring**: Combines success rates, feedback count, and ML predictions
4. **Learning**: Updates models based on user feedback
5. **Auto-Improvement**: Retrains ML models every 24 hours
6. **Path Tracking**: Records troubleshooting paths for continuous improvement (v2.1)

## Key Features

### Security
- SQL injection prevention
- Input sanitization
- Audit logging
- Session tracking

### Intelligence
- OpenAI GPT integration for unknown errors
- Semantic similarity matching
- Pattern recognition
- Predictive analytics
- Interactive decision trees (v2.1)

### Learning System
- User feedback tracking
- Success rate calculation
- Conflict detection
- Version control for solutions
- Adaptive path optimization (v2.1)

## Analytics Dashboard

View system performance:
- Total solutions and success rates
- Most common error types
- User activity tracking
- Solution quality metrics
- Security event monitoring

## Architecture

```
User Interface (WPF)
        ↓
    ┌───────────────┐
    │ Interactive   │ ← v2.1 Decision Tree Navigation
    │ Troubleshoot  │
    └───────┬───────┘
            ↓
Search Engine (5 Layers)
        ↓
    ┌───┴───┐
    │   ML  │ ← Machine Learning Engine
    │Engine │   (Clustering, Neural Net, SVM)
    └───┬───┘
        ↓
Databricks Database ← → OpenAI API
```

## File Structure

```
CAINE/
├── MainWindow.xaml.cs           (Core CAINE logic + ML integration)
├── MainWindow.xaml              (Main UI)
├── CaineMLEngine.cs            (Machine learning models)
├── InteractiveSolutionTree.cs  (v2.1 Decision tree logic)
├── SolutionTreeWindow.xaml     (v2.1 Tree UI)
├── SolutionTreeWindow.xaml.cs  (v2.1 Tree interaction)
├── TreeVisualizationControl.cs (v2.1 Visual flowchart)
├── Security/
│   └── SecurityValidator.cs    (Security features)
└── Analytics/
    └── AnalyticsWindow.cs      (Analytics dashboard)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Version History

- **v2.1** - Interactive Troubleshooting Assistant with decision trees
- **v2.0** - Machine Learning enhancement (clustering, neural networks, SVM)
- **v1.0** - Initial release with semantic search and ChatGPT integration

## Future Enhancements

- Web interface version
- Real-time error monitoring
- Automated error capture from logs
- Integration with ticketing systems
- Multi-language support
- Distributed team sharing
- Voice-guided troubleshooting
- Mobile companion app

## License

[ProximusDaviticus]

## Support

For issues or questions, please open an issue on GitHub or contact your system administrator.

## Acknowledgments

- OpenAI for GPT API
- Accord.NET for machine learning framework
- Databricks for scalable data storage
- All contributors and users who help CAINE learn

---

**Remember**: CAINE gets smarter with every use. The more feedback you provide, the better it becomes at solving your team's specific problems!