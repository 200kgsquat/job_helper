# BERT NER Predictions — Analysis

This document summarizes predictions made by the BERT NER model (`bert-base-cased`) on the `jjzha/skillspan` dataset.

---

## 1. Overall Performance

# Model Performance Comparison

## BERT-NER

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| \_               | 0.51      | 0.51   | 0.51     | 1091    |
| **micro avg**    | 0.51      | 0.51   | 0.51     | 1091    |
| **macro avg**    | 0.51      | 0.51   | 0.51     | 1091    |
| **weighted avg** | 0.51      | 0.51   | 0.51     | 1091    |

> Note: The model struggles to correctly identify entities beyond simple `O` labels. `B` and `I` tags are often missed or under-predicted.

## LR + CRF

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| \_               | 0.34      | 0.26   | 0.30     | 755     |
| **micro avg**    | 0.34      | 0.26   | 0.30     | 755     |
| **macro avg**    | 0.34      | 0.26   | 0.30     | 755     |
| **weighted avg** | 0.34      | 0.26   | 0.30     | 755     |

**Span-level F1:** 0.296

> Improved CRF model saved as `'crf_model_improved.joblib'`.


## 2. Sample Predictions

**Example 1:**  
Sentence: "The collaboration also covers acute patients during peak periods ."  
Predicted labels: `O O O O O O O O O O`  
True labels: `O O O O O O O O O O`  
> Model correctly predicted all tokens as `O`.

**Example 2:**  
Sentence: "4 ) Work on a mix of front-end back-end and cloud technologies ."  
Predicted labels: `O O O O O O O O O O O O O`  
True labels: `O O B I I I I I I I I I O`  
> Model failed to identify `front-end` and `back-end` as entities.

**Example 3:**  
Sentence: "Industry: Financial Technology SaaS wealthtech"  
Predicted labels: `O O O O O`  
True labels: `O O O O O`  
> Correctly predicted all tokens as `O`.

**Example 4:**  
Sentence: "Manage and be part of all processes of a typical sales cycle from A-Z ."  
Predicted labels: `B I I I I I I O O O I I O O O`  
True labels: `B I I I I I I I I I I I O O O`  
> Model partially recognized the entity span but missed some `I` tokens.

**Example 5:**  
Sentence: "In 14 technology hubs worldwide our team of 40,000+ technologists design build and deploy everything from enterprise technology initiatives to big data and mobile solutions as well as innovations in electronic payments cybersecurity machine learning and cloud development ."  
Predicted labels: `O O O O ...` (all `O`)  
True labels: `O O O O ...` (all `O`)  
> Model predicted all correctly for sentences without entities.

---

## 3. Observations

1. **High number of `O` predictions:**  
   The model often predicts `O` even when entities (`B`/`I`) are present.

2. **Entities spanning multiple tokens (`B`/`I`) are under-predicted:**  
   Subword splitting and token alignment issues may cause the model to miss `I` labels.

3. **Short or single-token entities are more accurately predicted.**

4. **Common failure cases:**  
   - Hyphenated words (`front-end`, `back-end`)  
   - Long sequences of entities  
   - Rare or domain-specific terms  


## 4. Running Training

* **Number of examples:** 7,974
* **Number of epochs:** 3
* **Batch size per device:** 16
* **Total train batch size** (with parallel, distributed & accumulation): 16
* **Gradient accumulation steps:** 1
* **Total optimization steps:** 1,497
* **Number of trainable parameters:** 107,721,987

