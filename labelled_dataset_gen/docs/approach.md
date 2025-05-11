# Approach for High-Quality NL2SQL Dataset Generation (TPCDS)

## 1. **Objective**
- Generate a high-quality, diverse, and fully-covered NL2SQL dataset using the TPCDS schema (24 tables, 427 columns).
- Ensure the dataset is suitable for both testing and training advanced NL2SQL models, including those using deep learning and contrastive learning techniques.

---

## 2. **Key Requirements**
- **Full Schema Coverage:** Every table and column in TPCDS should be used in at least one NL/SQL pair.
- **Diversity:** Include a wide range of business questions, SQL features, and NL paraphrases.
- **Quality:** Ensure all SQL is valid, all NL is clear, and all pairs are realistic and useful.
- **Metadata:** Annotate each example with tables/columns used, business intent, complexity, and other relevant info.
- **Reproducibility:** Allow for reproducible dataset generation (random seed, config management).

---

## 3. **Recommended Techniques**

### a. **Schema Coverage Tracking**
- Track which tables and columns have been used.
- Prioritize underrepresented schema elements in subsequent generations.

### b. **Prompt Engineering**
- Provide only relevant schema context in each prompt (rotate tables/columns).
- Explicitly instruct the LLM to use only the provided schema.
- Request output in a strict JSON format with all required metadata.
- Include example NL/SQL pairs in the prompt for style guidance.

### c. **Diversity Generation**
- Use LLMs or paraphrase models to generate multiple NL variants for each SQL.
- Include a mix of query complexities and business categories.
- Add ambiguous and edge-case questions for robustness.

### d. **Contrastive Learning Preparation**
- For each NL/SQL pair, generate negative pairs (e.g., mismatched NL and SQL) for use in contrastive learning.
- Optionally, include paraphrases and hard negatives (similar but incorrect SQL).

### e. **Validation and Quality Assurance**
- Validate SQL queries against the schema (using a parser or test DB).
- Human review a sample of generated data.
- Deduplicate NL/SQL pairs.

### f. **Extensibility**
- Design scripts to support other schemas and SQL dialects in the future.

---

## 4. **Prompt Template Guidelines**
- Include a subset of the schema (rotating tables/columns for coverage).
- Explicitly request: NL question, 2-3 paraphrases, SQL, tables/columns used, business intent, complexity, and alternatives.
- Specify output format (JSON with required fields).
- Provide clear instructions and examples in the prompt.

---

## 5. **Implementation Checklist**
- [ ] Load and parse full TPCDS schema.
- [ ] Track table and column coverage during generation.
- [ ] Construct prompts with rotating schema context.
- [ ] Generate NL/SQL pairs with required metadata.
- [ ] Generate paraphrases and negative pairs.
- [ ] Validate SQL and NL quality.
- [ ] Annotate and store all metadata.
- [ ] Ensure reproducibility (random seed, config).
- [ ] Review and deduplicate dataset.
- [ ] Report on schema coverage and dataset statistics.

---

## 6. **References**
- TPCDS Schema Documentation
- NL2SQL Research Papers (Spider, WikiSQL, etc.)
- Contrastive Learning Literature

---

**This document should be referenced before and during the development of any scripts for labelled NL2SQL dataset generation.** 