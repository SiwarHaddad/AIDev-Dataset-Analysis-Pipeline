# AIDev Dataset Analysis Pipeline

## Research Goal
Analyze the quality and maintainability characteristics of AI-generated code contributions to understand long-term implications on software sustainability.

## Repository Structure
```
AIDev-Dataset-Analysis-Pipeline/
├── SiwarHaddad__AIDevAct_Code.ipynb    # Main analysis pipeline
├── outputs/                             # Analysis results
│   ├── rq1_results.csv
│   ├── rq1_visualizations.png
│   ├── rq2_results.csv
│   ├── rq2_visualizations.png
│   ├── rq3_feature_importance.csv
│   ├── rq3_model_performance.csv
│   ├── rq3_visualizations.png
│   └── executive_summary.txt
└── README.md                            # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Google Colab (recommended) or local Jupyter environment
- Internet connection for dataset download

### Installation
```bash
# Install required packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Running the Pipeline

#### Option 1: Google Colab (Recommended)
1. Open the notebook: [SiwarHaddad__AIDevAct_Code.ipynb](https://github.com/SiwarHaddad/AIDev-Dataset-Analysis-Pipeline/blob/main/SiwarHaddad__AIDevAct_Code.ipynb)
2. Click "Open in Colab"
3. Run all cells sequentially
4. Results will be saved to `content/outputs/`

#### Option 2: Local Environment
```bash
# Clone the repository
git clone https://github.com/SiwarHaddad/AIDev-Dataset-Analysis-Pipeline.git
cd AIDev-Dataset-Analysis-Pipeline

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt

# Run Jupyter notebook (make sure you have a directory ./outputs and that output_path='./outputs')
jupyter notebook SiwarHaddad__AIDevAct_Code.ipynb
```

## Dataset Characterization

### Source Dataset
- **Primary**: AIDev-pop (curated subset with repos >500 stars)
- **Comparison**: Human-PRs from same repositories (2025)
- **Source**: Hugging Face (`hao-li/AIDev`)

### Inclusion Criteria
1. **Repository Quality**: Repositories with ≥500 GitHub stars
2. **Temporal Scope**: PRs created between Jan 1, 2025 - June 22, 2025
3. **Completeness**: PRs with complete commit metadata available
4. **Attribution**: Clear author attribution (Agent or Human)

### Exclusion Criteria
1. Incomplete metadata (missing commits, reviews, or timelines)
2. Non-agent bots (excluded from Agentic-PR category)
3. Ambiguous attribution (cannot determine agent/human authorship)
4. PRs outside temporal scope

### Final Dataset Size
```
Total PRs: 13,582
├── Agentic-PRs: 7,333 (54.0%)
│   ├── OpenAI Codex: 2,686
│   ├── Devin: 2,729
│   ├── GitHub Copilot: 1,462
│   ├── Cursor: 144
│   └── Claude Code: 101
└── Human-PRs: 6,249 (46.0%)

Repositories: 870 (with >500 stars)
Programming Languages: 51
Date Range: 2025-01-01 to 2025-06-22
```

## Research Questions & Metrics

### RQ1: Structural Code Quality Differences

**Question**: How do AI-generated code contributions differ from human contributions in structural quality?

**Metrics Calculated**:
- `loc_changed`: Total lines of code changed (additions + deletions)
- `files_touched`: Number of files modified per PR
- `additions`: Lines of code added
- `deletions`: Lines of code deleted
- `complexity_score`: Proxy measure (additions × 0.1 + files_touched × 2)
- `change_dispersion`: Files touched per LOC changed
- `has_tests`: Binary indicator of test file presence
- `add_del_ratio`: Ratio of additions to deletions

**Analysis Method**:
- Mann-Whitney U test for continuous metrics
- Chi-square test for categorical metrics (has_tests)
- Cliff's Delta for effect size
- Significance level: α = 0.05

### RQ2: Post-Merge Maintenance Effort

**Question**: What is the relationship between AI-generated code and maintenance effort?

**Metrics Calculated**:
- `is_accepted`: PR merge status (1 = merged, 0 = rejected)
- `review_time_hours`: Time from creation to closure
- `num_reviews`: Number of review submissions
- `num_comments`: Total comments (reviews + discussion)
- `linked_issues`: Number of linked issues (proxy for bug fixes)
- `is_bug_fix`: Binary indicator based on task type

**Analysis Method**:
- Chi-square test for acceptance rates
- Mann-Whitney U test for review time
- Median comparison for review activity

### RQ3: Success Prediction

**Question**: Which characteristics of Agentic-PRs predict successful integration?

**Features Used** (15 total):
- Code metrics: loc_changed, files_touched, additions, deletions
- Complexity: complexity_score, change_dispersion
- Quality indicators: has_tests, is_bug_fix, linked_issues
- Documentation: title_length, body_length, has_description
- Context: stars, language_encoded, type_encoded

**Model**: Random Forest Classifier
- n_estimators: 100
- max_depth: 10
- class_weight: balanced
- Train/test split: 70/30
- Cross-validation: 5-fold

**Evaluation Metrics**:
- Accuracy
- ROC-AUC
- Precision, Recall, F1-score
- Feature importance rankings

## Key Findings

### RQ1: Structural Code Quality

**Major Findings**:
1. **Agent PRs are significantly larger**:
   - Agent median LOC: 103 vs Human: 0
   - Agent median files: 3 vs Human: 0
   - All differences statistically significant (p < 0.0001)

2. **Test inclusion rate**:
   - Agent: 44.2% include tests
   - Human: 0.0% (due to missing data)

3. **Task type preferences**:
   - Agents focus on: feat (31.6%), fix (32.9%), docs (12.6%)
   - Similar distribution to human developers

**Note on Human Data**: The Human-PR baseline shows 0 values for most code metrics because the human_pull_request table lacks detailed commit information in the AIDev dataset. This is a known limitation.

### RQ2: Maintenance Effort

**Major Findings**:
1. **Lower acceptance rates for agents**:
   - Agent: 55.8% acceptance
   - Human: 77.6% acceptance
   - Difference: -21.8 percentage points (p < 0.0001)

2. **Faster review times for agents**:
   - Agent median: 4.1 hours
   - Human median: 5.6 hours
   - Difference not statistically significant (p = 0.139)

3. **Higher review activity**:
   - Agent median comments: 2.0
   - Human median comments: 0.0
   - Significant difference (p < 0.0001)

### RQ3: Success Predictors

**Model Performance**:
- Accuracy: 70.7%
- ROC-AUC: 74.5%
- Cross-validation ROC-AUC: 61.0% (±23.4%)

**Top 5 Feature Importance**:
1. Repository stars: 14.5%
2. Body length: 13.4%
3. Title length: 9.4%
4. Additions: 9.2%
5. Complexity score: 8.2%

**Interpretation**: Repository popularity and PR description quality are stronger predictors of acceptance than code complexity metrics.

## Pipeline Execution Steps

### Step 1: Data Loading
```python
analyzer = AIDevAnalyzer(output_path='./outputs/')
analyzer.load_data()
```
- Loads 12 tables from Hugging Face
- Total: 33,596 Agentic-PRs + 6,618 Human-PRs

### Step 2: Apply Filters
```python
analyzer.apply_inclusion_exclusion_criteria(min_stars=500)
```
- Filters by repository stars (≥500)
- Applies date range (2025-01-01 to 2025-06-22)
- Ensures complete metadata
- Final: 13,582 PRs

### Step 3: Calculate Metrics
```python
analyzer.calculate_metrics()
```
- Aggregates commit details per PR
- Computes 44 features total
- Handles missing values appropriately
- Creates analysis-ready dataset

### Step 4: Analyze RQs
```python
analyzer.analyze_rq1()  # Structural quality
analyzer.analyze_rq2()  # Maintenance effort
analyzer.analyze_rq3()  # Success prediction
```
- Performs statistical tests
- Trains ML models
- Generates visualizations
- Saves results to CSV

### Step 5: Generate Summary
```python
analyzer.generate_summary_report()
```
- Creates executive summary
- Consolidates key findings
- Provides recommendations

## Limitations & Discussion

### Data Limitations

1. **Human Baseline Issues**:
   - Human-PR data lacks detailed commit information
   - Most code metrics show 0 for human contributions
   - This severely limits RQ1 comparisons
   - **Mitigation**: Focus analysis on agent-to-agent comparisons and use acceptance rates as proxy for quality

2. **Dataset Temporal Bias**:
   - Short observation window (6 months)
   - Early adoption period may not reflect mature usage
   - **Impact**: Results may change as tools evolve

3. **Missing Metadata**:
   - Not all PRs have complete review/comment data
   - Some repositories missing language information
   - **Handling**: Filled with 0 or excluded from specific analyses

### Methodological Limitations

1. **Complexity Proxy**:
   - Used simple proxy (additions × 0.1 + files × 2)
   - Cannot calculate true cyclomatic complexity without AST parsing
   - **Justification**: Computational constraints for 13K+ PRs

2. **Test Detection Heuristic**:
   - Keyword-based (test, spec, __tests__)
   - May miss non-standard test patterns
   - **Accuracy**: Estimated 80-90% precision based on manual sampling

3. **Attribution Ambiguity**:
   - Some agents (OpenAI Codex) don't clearly mark authorship
   - Relied on branch naming conventions
   - **Validation**: Manual spot-checks confirm >95% accuracy

### Statistical Limitations

1. **Class Imbalance**:
   - PRs accepted vs rejected
   - Used balanced class weights in RF model
   - **Impact**: Model may overpredict acceptance

2. **Feature Engineering**:
   - Limited to GitHub metadata
   - No AST-level code analysis
   - No semantic code understanding
   - **Future work**: Integrate LLM-based code quality metrics

3. **Cross-Validation Variance**:
   - High std (±23.4%) in CV ROC-AUC
   - Suggests model instability across folds
   - **Possible causes**: Dataset heterogeneity, agent diversity

### Construct Validity

1. **Acceptance ≠ Quality**:
   - PR merge doesn't guarantee code quality
   - May reflect project practices, not agent capability
   - **Consideration**: Need long-term bug tracking

2. **Review Time Interpretation**:
   - Faster review ≠ better quality
   - Could indicate superficial review
   - **Context needed**: Review depth analysis

### External Validity

1. **Generalizability**:
   - Limited to popular open-source repos (>500 stars)
   - May not apply to private/enterprise settings
   - Different workflows in corporate environments

2. **Agent Evolution**:
   - Tools rapidly evolving (2025 data)
   - Findings may be outdated quickly
   - **Recommendation**: Periodic re-analysis

## Refinements from Part 1

### Changes Made
1. **Simplified RQ1**: Removed cyclomatic complexity (computation intensive)
2. **Added Metrics**: 
   - `change_dispersion` for code scatter analysis
   - `complexity_score` as lightweight proxy
3. **Enhanced RQ3**: Added repository context (stars, language)
4. **Improved Filtering**: More strict date range for consistency

### Justifications
- Focus on achievable metrics with available data
- Balance depth vs. computational feasibility
- Prioritize reproducibility and interpretability

## Recommendations

### For Developers
1. Include tests when using AI coding agents (44% currently do)
2. Provide detailed PR descriptions (strong predictor of acceptance)
3. Review AI-generated code thoroughly despite fast turnaround

### For Tool Designers
1. Optimize for acceptance, not just speed
2. Improve test generation capabilities
3. Better documentation of code rationale

### For Researchers
1. Develop better human baseline datasets
2. Investigate long-term code survival rates
3. Study review depth vs. review speed tradeoffs

## Contact & Support

- GitHub: [SiwarHaddad](https://github.com/SiwarHaddad)
- Repository: [AIDev-Dataset-Analysis-Pipeline](https://github.com/SiwarHaddad/AIDev-Dataset-Analysis-Pipeline)
- Issues: Please open a GitHub issue for questions or problems

## Acknowledgments

- AIDev Dataset: Li et al. (2025)
- MSR 2026 Challenge organizers

---

**Last Updated**: January 3, 2026
**Version**: 1.0.0
