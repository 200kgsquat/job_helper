# Model Comparison Report

## 📊 Dataset Overview
- **Task**: Industry Classification (10 classes)
- **Test Samples**: 1,591

## 📈 Overall Performance Metrics

| Metric | BERT | TF-IDF+LR |
|--------|------|-----------|
| **Accuracy** | 74.86% | 73.98% |
| **Macro F1-Score** | 77.19% | 76.43% |
| **Inference Speed** | ~15 samples/sec | ~1200 samples/sec |

## 🔍 Key Observations

### 1. Performance Difference
- BERT slightly outperforms TF-IDF+LR in both accuracy (+0.88%) and macro F1-score (+0.76%)
- Both models show similar overall performance patterns

### 2. Class-wise Analysis
| Industry Category | BERT F1-Score | TF-IDF F1-Score | Notes |
|-------------------|---------------|-----------------|-------|
| Education Management | 0.97 | 0.96 | Both models achieve excellent performance |
| Hospital & Health Care | 0.89 | 0.89 | Comparable high performance |
| Oil & Energy | 0.87 | 0.85 | TF-IDF shows perfect precision (1.00) while BERT achieves better recall |
| Telecommunications | 0.69 | 0.48 | BERT shows significantly better recall (0.69 vs 0.48) |
| Computer Software | 0.75 | 0.78 | TF-IDF achieves better balance between precision and recall |
| Marketing and Advertising | 0.70 | 0.74 | TF-IDF shows more balanced performance |

### 3. Model Strengths

#### BERT
- Better handling of semantic nuances in text
- Superior performance on classes requiring contextual understanding
- More consistent recall across most categories
- Better at handling ambiguous industry descriptions

#### TF-IDF+LR
- Computationally more efficient (80x faster inference)
- Better precision in several categories (Computer Software, Telecommunications)
- Strong performance on clearly defined industry terms
- Lower resource requirements for deployment

### 4. Model Weaknesses

#### BERT
- Lower precision for Marketing and Advertising (0.62)
- Higher computational requirements (GPU recommended)
- Slower inference speed (15 samples/sec vs 1200 samples/sec)

#### TF-IDF+LR
- Struggles with semantic variations (Telecommunications recall: 0.48)
- Less effective with complex linguistic patterns
- Performance depends heavily on feature engineering

## ⚡ Inference Speed Comparison

| Metric | BERT | TF-IDF+LR |
|--------|------|-----------|
| **Samples per Second** | ~15 | ~1200 |
| **Hardware Requirements** | GPU recommended | CPU only |
| **Batch Processing** | More efficient with batching | Consistently fast |

## 🎯 Conclusion

Both models achieve comparable overall performance, with BERT showing a slight advantage in overall metrics (accuracy +0.88%, F1 +0.76%). BERT demonstrates better contextual understanding while TF-IDF+LR remains a strong baseline for classification tasks where computational efficiency is important.

### Decision Factors:
- **BERT is preferable when**: Accuracy is critical, computational resources are available, and inference speed is not the primary concern
- **TF-IDF+LR is preferable when**: Inference speed is critical (real-time applications), computational resources are limited, or explainability is important

The 80x speed advantage of TF-IDF+LR makes it more suitable for high-throughput applications, while BERT's slightly better accuracy may justify its use in accuracy-critical applications where speed is less important.

---

*Report generated on: 20.08.2025*  
*Analysis based on 1,591 test samples across 10 industry categories*  
*Hardware: NVIDIA RTX 3080 for BERT, Intel i7-10700K for TF-IDF+LR*