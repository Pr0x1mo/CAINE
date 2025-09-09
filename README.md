# CAINE - Computer-Aided Intelligence Neuro Enhancer

## What is CAINE?

CAINE is an enterprise-grade intelligent error resolution system that combines traditional machine learning with modern AI to solve database and network errors. It learns from your organization's collective experience and improves with every use.

### Core Capabilities
- **Multi-Layer Search**: 6 different search strategies from exact matching to ML predictions
- **Machine Learning Engine**: SVM (88-96% accuracy), clustering, decision trees, neural networks
- **Enhanced Fuzzy Search**: Handles typos and variations with Levenshtein distance and N-gram analysis
- **Distributed Computing**: Leverages Databricks SQL Warehouse for scalability
- **Interactive Troubleshooting**: Visual decision trees with auto-launch capabilities
- **Security First**: SQL injection protection, rate limiting, audit logging
- **Real-time Analytics**: Live ML performance dashboard with actual usage metrics

## Version 3.0 - Full ML Enhancement with Production Dashboard

### What's New in v3.0
- **GPU-Accelerated Neural Networks** (TorchSharp integration active)
- **Auto-Launching Interactive Trees** based on ML confidence analysis
- **Comprehensive Fuzzy Search Engine** with synonym matching and n-gram analysis
- **Enhanced Security Layer** with intelligent error/attack discrimination
- **Scalable Vector Search** with LSH hashing and intelligent caching
- **Production ML Dashboard** showing real usage metrics and performance data
- **Unified Confidence Calculator** combining multiple ML predictions
- **Fixed Performance Tracking** with actual user feedback integration

### Machine Learning Components

#### Active ML Models
- **Support Vector Machines (SVM)**: 88-96% accuracy on error classification
- **K-Means Clustering**: Automatically discovers error categories
- **C4.5 Decision Trees**: Predicts solution effectiveness
- **Neural Networks**: TorchSharp-powered deep learning with GPU acceleration
- **Time Series Analysis**: Forecasts error trends and patterns
- **Anomaly Detection**: Identifies new error types requiring attention

#### ML Dashboard Data Sources
- **Model Performance**: Real accuracy from actual user feedback
- **Feature Importance**: Analysis of error characteristics that correlate with success
- **Cluster Analysis**: Automatic categorization of errors (Network, Security, Performance, Data Issues)
- **Trend Predictions**: ML-based forecasting of error patterns
- **Anomaly Detection**: Statistical analysis of unusual error patterns

#### ML Activation Thresholds
- **< 50 samples**: Basic ML mode (clustering only)
- **≥ 50 samples**: Full ML mode (all models active)
- **≥ 100 samples**: Neural network training begins automatically

## System Architecture

```
┌─────────────────────────────────────┐
│     User Interface (WPF)            │
│  ┌──────────┬──────────┬─────────┐  │
│  │ Search   │Analytics │ Tree    │  │
│  │ Engine   │Dashboard │ Visual  │  │
│  └──────────┴──────────┴─────────┘  │
└──────────────┬──────────────────────┘
               ↓
┌──────────────────────────────────────┐
│     6-Layer Search & ML Pipeline     │
│  1. Exact Match (SHA256 hash)        │
│  2. Enhanced Fuzzy Keyword Search    │
│  3. Comprehensive Fuzzy Search       │
│  4. Advanced Scalable Vector Search  │
│  5. Pattern Recognition              │
│  6. Comprehensive ML Search          │
└──────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│     Data Layer                      │
│  ┌─────────────┬─────────────┐      │
│  │ Databricks  │    OpenAI   │      │
│  │SQL Warehouse│    API      │      │
│  └─────────────┴─────────────┘      │
└─────────────────────────────────────┘
```

## Key Features

### Advanced Search Capabilities
- **Exact Match**: SHA256 hash-based instant lookup
- **Enhanced Fuzzy Search**: Tolerates typos with multiple similarity metrics
- **Synonym Expansion**: Understands "timeout" = "timed out" = "hung" = "freeze"
- **N-gram Analysis**: Partial text matching for robust similarity detection
- **Vector Similarity**: Semantic search using OpenAI embeddings with LSH indexing
- **ML Pattern Recognition**: SVM and neural network-based pattern matching

### Interactive Troubleshooting
- **Auto-Launch Detection**: ML determines when guided troubleshooting is beneficial
- **Decision Tree Visualization**: Full flowchart view of troubleshooting paths
- **Step-by-Step Guidance**: Interactive yes/no questions leading to solutions
- **Path Optimization**: ML learns which paths work best for different error types
- **Success Tracking**: Records which troubleshooting paths lead to resolution

### Security Features
- **Intelligent Input Validation**: Distinguishes error messages from actual attacks
- **SQL Injection Prevention**: Multi-layer validation with context awareness
- **Rate Limiting**: 1000 queries/day per user
- **Input Sanitization**: XSS and command injection protection
- **Audit Logging**: Complete security event tracking with severity levels

### Learning System
- **Feedback Loop**: Thumbs up/down ratings improve confidence
- **Auto-retraining**: Models update every 24 hours with new data
- **Conflict Detection**: Identifies contradictory solutions
- **Success Tracking**: Monitors solution effectiveness across all search methods
- **Unified Confidence**: Combines ML predictions, user feedback, and similarity scores

### Production Analytics Dashboard
- **Real-time Metrics**: Actual solution count, success rates, active users, security events
- **ML Performance**: Model accuracy from real user feedback, confidence distributions
- **Search Strategy Analytics**: Effectiveness of each search layer based on actual usage
- **Interactive Tree Metrics**: Success rates and usage patterns from real troubleshooting sessions
- **System Health**: Overall status with ML-enhanced monitoring
- **Feature Analysis**: Hour-of-day performance patterns from actual user data
- **Error Clustering**: Automatic categorization based on real error signatures

## Installation

### Prerequisites
- Windows 10/11 with .NET Framework 4.8
- Visual Studio 2019 or later
- Databricks account with SQL Warehouse
- OpenAI API key
- Optional: NVIDIA GPU for neural network acceleration

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CAINE.git
   ```

2. **Configure environment variables**
   ```bash
   setx OPENAI_API_KEY "your-api-key-here"
   ```

3. **Set up Databricks ODBC**
   - Create System DSN named "CAINE_Databricks"
   - Configure with your SQL Warehouse credentials

4. **Install NuGet packages**
   ```xml
   <PackageReference Include="Accord.MachineLearning" Version="3.8.2-alpha" />
   <PackageReference Include="Accord.Statistics" Version="3.8.2-alpha" />
   <PackageReference Include="MathNet.Numerics" Version="6.0.0-beta2" />
   <PackageReference Include="Newtonsoft.Json" Version="13.0.1" />
   <PackageReference Include="TorchSharp" Version="0.100.7" />
   <PackageReference Include="WpfAnimatedGif" Version="2.0.2" />
   ```

5. **Build and run**
   - Open CAINE.sln in Visual Studio
   - Restore NuGet packages
   - Build solution (x64 configuration)
   - Run

## Database Schema

CAINE automatically creates these tables:

```sql
-- Main knowledge base
CREATE TABLE default.cai_error_kb (
    id STRING,
    error_hash STRING,
    error_signature STRING,
    error_text STRING,
    resolution_steps ARRAY<STRING>,
    embedding ARRAY<FLOAT>,
    confidence_score DOUBLE,
    solution_source STRING,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    source STRING,
    notes STRING
) USING DELTA;

-- User feedback with enhanced tracking
CREATE TABLE default.cai_solution_feedback (
    feedback_id STRING,
    session_id STRING,
    solution_hash STRING,
    solution_source STRING,
    solution_version STRING,
    was_helpful BOOLEAN,
    confidence_rating DOUBLE,
    user_comment STRING,
    user_expertise STRING,
    created_at TIMESTAMP,
    error_signature STRING,
    resolution_time_minutes INT,
    environment_context STRING
) USING DELTA;

-- Vector index for scalable similarity search
CREATE TABLE default.cai_vector_index (
    index_id STRING,
    error_hash STRING,
    vector_data STRING,
    lsh_hash_0 STRING,
    lsh_hash_1 STRING,
    lsh_hash_2 STRING,
    created_at TIMESTAMP
) USING DELTA;

-- Vector caching for performance
CREATE TABLE default.cai_vector_cache (
    cache_key STRING,
    query_hash STRING,
    results_json STRING,
    created_at TIMESTAMP,
    hit_count INT,
    last_accessed TIMESTAMP
) USING DELTA;

-- Security audit log
CREATE TABLE default.cai_security_log (
    event_id STRING,
    created_at TIMESTAMP,
    event_type STRING,
    user_id STRING,
    session_id STRING,
    details STRING,
    severity STRING
) USING DELTA;

-- Interactive tree paths with success tracking
CREATE TABLE default.cai_tree_paths (
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

-- Pattern matching rules
CREATE TABLE default.cai_error_patterns (
    pattern_id STRING,
    pattern_regex STRING,
    priority INT,
    resolution_steps ARRAY<STRING>,
    created_at TIMESTAMP,
    description STRING,
    effectiveness_score DOUBLE,
    usage_count INT,
    last_successful_match TIMESTAMP
) USING DELTA;
```

## Usage Guide

### Basic Workflow
1. **Search**: Paste error → Click Search
2. **ML Analysis**: System runs 6-layer search with ML preprocessing
3. **Auto-Tree**: Interactive troubleshooting may launch automatically for complex errors
4. **Review**: Check confidence score and ML insights
5. **Apply**: Follow the solution steps
6. **Feedback**: Click 👍 or 👎 to improve ML models

### ML Dashboard Access
- Click "Analytics" button in main window
- View real-time performance metrics from actual usage
- Monitor ML model accuracy and effectiveness
- Track error clustering and trend predictions
- Review feature importance analysis

### Interactive Troubleshooting
- **Auto-Launch**: Triggered automatically for low-confidence or complex errors
- **Manual Access**: Click "🌳 Interactive" button anytime
- **Guided Process**: Answer yes/no questions through decision tree
- **Visual Tree**: View full flowchart of troubleshooting paths
- **ML Optimization**: System learns which paths work best

### Teaching New Solutions
1. If no solution found, click "Teach CAINE"
2. Enter step-by-step solution
3. System checks for conflicts with existing solutions
4. ML models incorporate new knowledge automatically

### Using the API Fallback
- "Use CAINE API" button provides ChatGPT-powered solutions
- Builds context from similar past solutions
- Useful for comparing local vs AI solutions
- Results can be taught back to the system

## Performance Metrics

- **Search Speed**: < 500ms average response time across 6 layers
- **ML Accuracy**: 88-96% SVM classification accuracy
- **Neural Network**: GPU-accelerated training and inference
- **Scalability**: Handles 1000+ concurrent users with LSH indexing
- **Learning Rate**: Improves 2-3% per 100 feedback entries
- **Cache Hit Rate**: 60-70% for repeated queries with intelligent caching
- **Confidence Growth**: 20% → 100% after ~10 positive feedbacks

## ML Dashboard Data Types

### Real Usage Data (from your actual system usage)
- **Model Performance**: Calculated from real user thumbs up/down feedback
- **Feature Importance**: Error length analysis from actual KB entries
- **Time-based Patterns**: Hour-of-day success rates from real feedback timestamps
- **Error Clustering**: Categorization based on actual error signatures in your database
- **Search Method Effectiveness**: Success rates by search type (exact match, fuzzy, etc.)

### Estimated/Simulated Data (shown when insufficient real data)
- **Anomaly Detection**: Estimates 5% anomaly rate when no ML model available
- **Trend Predictions**: Shows "Prediction unavailable" if ML engine not trained
- **Cold Start Messages**: "No recent feedback data available" until you use the system

### Dashboard Activation Requirements
- **Basic Dashboard**: Available immediately (shows "Loading..." messages)
- **Real Metrics**: Requires actual system usage and user feedback
- **Full ML Insights**: Requires 50+ feedback entries for meaningful analysis

## File Structure

```
CAINE/
├── MainWindow.xaml.cs             # Core search and ML integration
├── CaineMLEngine.cs               # Machine learning models
├── FuzzySearchEngine.cs           # Fuzzy matching algorithms
├── InteractiveSolutionTree.cs     # Decision tree logic
├── SolutionTreeWindow.xaml.cs     # Interactive UI window
├── TreeVisualizationControl.cs    # Visual tree rendering
├── SolutionParser.cs              # Solution step parsing
├── AnalyticsWindow.xaml.cs        # Main analytics dashboard
├── MLDashboardWindow.xaml.cs      # ML-specific metrics dashboard
├── Security/
│   └── SecurityValidator.cs       # Security validation layer
└── Vector/
    └── ScalableVectorManager.cs   # Advanced vector search engine
```

## Troubleshooting

### Common Issues

**Neural network not loading**
- TorchSharp will auto-detect GPU vs CPU
- Falls back gracefully to SVM (88-96% accuracy sufficient)
- Check CUDA installation for GPU acceleration

**ODBC connection fails**
- Verify Databricks SQL Warehouse is running
- Check DSN configuration in ODBC Data Sources
- Ensure firewall allows connection

**Dashboard showing "no data available"**
- Normal for new installations
- Use CAINE to search for errors and provide feedback
- Dashboard becomes meaningful after 10+ uses with feedback

**Low confidence scores**
- Need 50+ feedback entries for full ML activation
- Check for conflicting solutions in Analytics Dashboard
- Review error categorization patterns

**Interactive trees not auto-launching**
- Requires ML models to be trained (50+ samples)
- Check ML insights in search results
- Manual launch always available via button

### Security Events
Monitor the Analytics Dashboard for:
- HIGH severity: SQL injection attempts blocked
- MEDIUM severity: Rate limiting activated
- INFO: Normal error analysis operations

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open Pull Request

## Version History

- **v3.0** (2024) - Full ML enhancement with neural networks, auto-launching trees, advanced vector search, and production analytics dashboard
- **v2.1** (2024) - Interactive troubleshooting with decision trees
- **v2.0** (2024) - Machine learning integration
- **v1.0** (2024) - Initial release with ChatGPT integration

## Future Roadmap

- [ ] Web-based interface
- [ ] Real-time log monitoring with ML anomaly detection
- [ ] Automated error capture from system logs
- [ ] Team collaboration features with shared knowledge bases
- [ ] Natural language queries ("Find connection timeouts from last week")
- [ ] Predictive maintenance alerts based on error trends
- [ ] Integration with ticketing systems (Jira, ServiceNow)
- [ ] Mobile companion app for on-call support
- [ ] Export/import knowledge bases between organizations
- [ ] Multi-tenant support with role-based access

## License

[ProximusDaviticus]

## Support

For issues or questions:
- Open an issue on GitHub
- Check the Analytics Dashboard for system health
- Review ML Dashboard for performance metrics
- Review security logs for blocked threats
- Contact your system administrator

## Acknowledgments

- OpenAI for GPT API and embeddings
- Accord.NET for ML framework
- Databricks for scalable data platform
- TorchSharp for neural network capabilities
- All contributors and users who help CAINE learn

---

**Remember**: CAINE improves with every use. The more feedback you provide, the better it becomes at solving your specific problems. The ML Dashboard shows real metrics from your actual usage - with 50+ feedback entries, full ML capabilities activate automatically, including neural networks and auto-launching interactive troubleshooting!