# Course Material Analysis

## Summary of Topics Covered

Based on analysis of all 12 PDFs:

### 1. Retrieval/Ranking Methods Covered

**✅ QLD (Query Likelihood Dirichlet)** - **HEAVILY COVERED**
- Found in 10 out of 12 PDFs
- Detailed coverage in PDF 10 with examples
- Formula: `p_Dir(t|M_d) = (tf(t,d) + μ*p_MLE(t|M_C)) / (|d| + μ)`
- Parameter: `μ` (Dirichlet prior/smoothing parameter)
- Pyserini usage: `searcher.set_qld(mu=1000)`
- Template uses: `searcher.set_qld(mu=1000)`

**⚠️ BM25** - **MENTIONED BUT NOT DEEPLY COVERED**
- Found in only 2 PDFs (5.pdf and 10.pdf)
- PDF 10 mentions: "Optionally, configure BM25 parameters"
- No detailed formula or examples found
- Not used in template

**✅ TF-IDF / Vector Space Model** - **COVERED**
- Found in 5-6 PDFs
- Covered in PDFs 3, 4, 6, 7, 9, 10
- Not used in template (template uses QLD)

**❌ Hybrid QLD+BM25** - **NOT COVERED**
- No evidence of combining QLD and BM25 scores
- Hybrid mentions found are for other purposes (term/entity feedback fusion)

**⚠️ Reciprocal Rank Fusion** - **MENTIONED BUT DIFFERENT PURPOSE**
- Found in PDF 12
- Used for combining term and entity feedback, not for combining QLD/BM25

### 2. Key Findings

1. **Primary Method**: QLD (Query Likelihood Dirichlet) is the main retrieval method taught
2. **Template Alignment**: Template uses only QLD with `mu=1000`
3. **BM25 Status**: BM25 is mentioned but not deeply covered - likely optional/advanced
4. **No Hybrid**: Combining QLD and BM25 was not taught in the course

### 3. What Should Be in the Implementation

**✅ SHOULD INCLUDE:**
- QLD with configurable `mu` parameter (main method)
- Experimentation with different `mu` values (500, 1000, 2000, 3000)
- Pyserini usage: `searcher.set_qld(mu=X)`

**⚠️ OPTIONAL (if mentioned but not required):**
- BM25 as an alternative method (if you want to experiment)
- But should not be the primary focus

**❌ SHOULD REMOVE OR SIMPLIFY:**
- Hybrid QLD+BM25 combination (not covered in course)
- Complex score normalization for hybrid (not taught)

### 4. Recommendations

1. **Primary Focus**: QLD with different `mu` values
2. **Secondary**: BM25 can be included as an optional alternative
3. **Remove**: Hybrid QLD+BM25 approach (not covered)
4. **Keep**: Prompt engineering improvements (allowed)
5. **Keep**: Better context handling (fixing template bugs)

