# Course Material Alignment Summary

## Answers to Your Questions

### 1. Which retrieval/ranking methods were covered?

**✅ QLD (Query Likelihood Dirichlet)** - **HEAVILY COVERED**
- Found in 10 out of 12 PDFs
- Detailed coverage in PDF 10 with formulas and examples
- Formula: `p_Dir(t|M_d) = (tf(t,d) + μ*p_MLE(t|M_C)) / (|d| + μ)`
- Parameter: `μ` (Dirichlet prior/smoothing parameter)
- Pyserini usage: `searcher.set_qld(mu=1000)`
- **This is the PRIMARY method taught and used in the template**

**⚠️ BM25** - **MENTIONED BUT NOT DEEPLY COVERED**
- Found in only 2 PDFs (5.pdf and 10.pdf)
- PDF 10 mentions: "Optionally, configure BM25 parameters"
- No detailed formula or examples found
- **Not used in template** - only QLD is used

**✅ TF-IDF / Vector Space Model** - **COVERED**
- Found in 5-6 PDFs
- Covered in PDFs 3, 4, 6, 7, 9, 10
- Not used in template (template uses QLD)

**❌ Hybrid QLD+BM25** - **NOT COVERED**
- No evidence of combining QLD and BM25 scores
- Hybrid mentions found are for other purposes (term/entity feedback fusion in PDF 12)

### 2. Any specific techniques or algorithms emphasized?

**Primary Emphasis:**
- **Query Likelihood Dirichlet (QLD)** with Dirichlet smoothing
- Parameter tuning for `mu` (Dirichlet smoothing parameter)
- Pyserini usage with `set_qld(mu=X)`

**Secondary Mentions:**
- Jelinek-Mercer smoothing (mentioned but Dirichlet is primary)
- BM25 (mentioned but not emphasized)
- Query expansion (mentioned in multiple PDFs but not required for assignment)

### 3. Any methods mentioned that we should avoid?

**Should Remove/Simplify:**
- ❌ **Hybrid QLD+BM25 combination** - Not covered in course material
- The hybrid approach I initially implemented combines QLD and BM25 scores, which was not taught

**Should Keep but De-emphasize:**
- ⚠️ **BM25** - Mentioned but not deeply covered, can be kept as optional alternative
- Focus should be on QLD parameter tuning

## Changes Made to Align with Course

### ✅ Updated Implementation

1. **Removed Hybrid Method**
   - Removed `get_context_hybrid()` function
   - Removed hybrid option from configuration
   - Updated error messages to only mention QLD and BM25

2. **Emphasized QLD as Primary**
   - QLD is now clearly marked as the primary method
   - BM25 is marked as optional alternative
   - Default method is QLD (matching template)

3. **Updated Documentation**
   - APPROACH.md updated to reflect QLD as primary method
   - Removed hybrid references
   - Emphasized QLD parameter tuning (mu values)

4. **Updated Code Comments**
   - Added notes explaining QLD is from course material
   - BM25 marked as optional/experimental
   - Removed hybrid implementation

## Current Implementation Status

### ✅ Aligned with Course Material

- **Primary Method**: QLD (Query Likelihood Dirichlet) - ✅ Covered in course
- **Parameter Tuning**: Focus on `mu` values (500, 1000, 2000, 3000) - ✅ From course
- **Pyserini Usage**: `searcher.set_qld(mu=1000)` - ✅ Matches template and course
- **Prompt Engineering**: Improved prompts (allowed) - ✅ Allowed per assignment
- **Bug Fixes**: Fixed template bugs - ✅ Necessary improvements

### ⚠️ Optional (Not Required)

- **BM25**: Included as optional alternative for experimentation
  - Mentioned in course but not deeply covered
  - Can be used but focus should be on QLD

### ❌ Removed (Not Covered)

- **Hybrid QLD+BM25**: Removed - not covered in course material

## Recommendation

The implementation is now aligned with the course material:
- **Primary focus**: QLD with different `mu` parameter values
- **Optional**: BM25 for experimentation
- **Removed**: Hybrid combination (not taught)

The system is ready to use and focuses on the methods actually covered in your course.

