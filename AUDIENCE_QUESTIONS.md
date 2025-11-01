# Audience Questions: System Learning & Prediction

## 1. How Does the System Learn?

The system uses two complementary machine learning approaches:

### A. Ad Click Prediction (Random Forest Classifier)
**Learning Method:** Supervised Learning
- **Algorithm:** Random Forest with 80 decision trees
- **Training Process:**
  - The model learns patterns from historical customer interactions
  - It analyzes 10,000+ interaction records to identify what features correlate with clicks
  - Uses ensemble learning (multiple decision trees voting together) for robust predictions
  - Applies class weighting to handle imbalanced data (more non-clicks than clicks)

**What it learns:**
- Which customer segments are more likely to click on ads
- How product characteristics (price, rating, category) influence click behavior
- Time patterns (hour of day, day of week) that affect engagement
- Device and regional preferences

### B. Product Recommendations (SVD - Singular Value Decomposition)
**Learning Method:** Collaborative Filtering (Unsupervised Learning)
- **Algorithm:** Truncated SVD with 20 latent factors
- **Training Process:**
  - Creates a user-item interaction matrix
  - Decomposes this matrix into user and product "embeddings" (latent features)
  - Captures hidden patterns in user preferences and product similarities
  - Uses weighted scoring: views (0.05) + clicks (0.5) + purchases (2.5)

**What it learns:**
- User preference patterns across product categories
- Product similarities based on who interacted with them
- Latent factors (hidden features) that explain user-product affinity

---

## 2. How Does the System Predict?

### Ad Click Prediction Process:
1. **Input:** User profile + Product features + Context (time, device)
2. **Feature Engineering:** Convert categorical variables to numerical (one-hot encoding)
3. **Prediction:** Random Forest outputs probability score (0-1) for click likelihood
4. **Output:** Ranked list of products with click probability scores

**Example:** For a 35-year-old user in the East region, the model predicts 0.73 probability they'll click on electronics ads shown on mobile at 8 PM.

### Product Recommendation Process:
1. **Input:** User ID
2. **Retrieval:** Generate candidate products based on collaborative filtering
3. **Ranking:** Score candidates using the click prediction model
4. **Filtering:** Remove already-seen/purchased items
5. **Output:** Top N personalized product recommendations

**Example:** User123 gets recommended products P401, P205, P789 because similar users purchased these items.

---

## 3. What Data Does the System Learn From?

### Primary Dataset: Customer Interaction Records
**Size:** 10,000 interaction records across 1,000 users and 500 products

### Data Categories:

#### User Demographics
- **Age:** Customer age (18-65 range)
- **Gender:** Male/Female/Other
- **Region:** Geographic location (North, South, East, West)
- **Device:** Desktop, Mobile, Tablet

#### Product Information
- **Product ID:** Unique identifier
- **Category:** Electronics, Clothing, Home, Books, etc.
- **Price:** Product price point ($)
- **Rating:** Customer rating (1-5 stars)

#### Interaction Behavior
- **Clicked:** Binary (0/1) - Did user click on the product?
- **Purchased:** Binary (0/1) - Did user purchase the product?
- **Timestamp:** When the interaction occurred
- **Hour:** Hour of day (0-23)
- **Day of Week:** Monday-Sunday (0-6)

### Data Weighting Strategy
The system assigns importance scores to different interactions:
- **View/Impression:** 0.05 points
- **Click:** 0.5 points
- **Purchase:** 2.5 points

This helps the system prioritize strong signals (purchases) over weak signals (views).

---

## 4. Common Audience Questions

### Q: How accurate is the system?
**A:** The Random Forest classifier achieves ~80% accuracy with balanced precision/recall. Performance metrics are tracked via ROC-AUC score and classification reports (see Customer_Interactions.ipynb for details).

### Q: Does the system improve over time?
**A:** Currently, the models are pre-trained and static. To improve over time, you would need to:
- Periodically retrain with new interaction data
- Implement online learning or incremental updates
- Monitor model drift and performance degradation

### Q: What if we have a new user with no history?
**A:** This is the "cold start problem":
- For new users: System uses demographic-based predictions (age, region, device)
- For new products: System relies on content features (category, price, rating)
- Collaborative filtering won't work until sufficient interaction data exists

### Q: How do you prevent showing the same recommendations repeatedly?
**A:** The system filters out already-seen products from the recommendation list (line 102-104 in ecom.py):
```python
seen = set(user_item.loc[user_id][user_item.loc[user_id] > 0].index)
recs = [pid for pid, sc in ranked if pid not in seen]
```

### Q: Can the system explain why it recommends something?
**A:** Partial explainability:
- **Random Forest:** Can show feature importance (which factors most influence predictions)
- **SVD:** Harder to interpret (latent factors are abstract mathematical concepts)
- For better explainability, consider adding rule-based explanations or SHAP values

### Q: What about privacy and bias?
**A:** Important considerations:
- **Privacy:** System uses anonymized user_ids, but still processes demographic data
- **Bias:** Gender/age/region are used as features, which could perpetuate stereotypes
- **Mitigation:** Audit model predictions across demographic groups, apply fairness constraints

### Q: How scalable is this system?
**A:** Current implementation:
- **Handles:** ~1,000 users, ~500 products
- **Performance:** In-memory processing with joblib model serialization
- **Scaling up:** Would require:
  - Database integration (not CSV files)
  - Distributed computing (Spark, Ray)
  - Model serving infrastructure (APIs, caching)
  - Real-time streaming for live predictions

---

## 5. Technical Architecture Summary

```
[Data Sources] → [Feature Engineering] → [Model Training] → [Prediction] → [Results]
     ↓                    ↓                      ↓               ↓            ↓
Customer Data    One-hot encoding      Random Forest     Click prob    Recommendations
Product Data     User-item matrix           SVD          Rankings      Targeted ads
Interactions     Temporal features      80 trees      Collaborative   Promotions
                                       20 factors        filtering
```

### Files in the System:
- **Customer_Interactions.xlsx:** Raw customer interaction data (10K records)
- **Customer_Interactions.ipynb:** Model training notebook
- **ecom.py:** Streamlit web app for predictions
- **ecom_synthetic/customers.csv:** Processed customer data
- **ecom_synthetic/ad_click_model_rf.joblib:** Trained Random Forest model
- **ecom_synthetic/recommender_svd.joblib:** Trained SVD model

---

## 6. Demo Flow

When you run the Streamlit app (`streamlit run ecom.py`):

1. **Select a customer** from the dropdown
2. **View their profile:** age, gender, region, device
3. **Click "Recommend"** button
4. **System generates:**
   - Top N product recommendations (collaborative filtering)
   - Click probability scores for each product (Random Forest)
   - Personalized promotions (rule-based on price percentiles)
   - Targeted ad suggestions (category-based)
5. **Download results** as CSV files

---

## 7. Key Performance Metrics

From the training notebook (Customer_Interactions.ipynb):

- **Click-Through Rate (CTR):** ~5-10% baseline
- **Purchase Rate:** ~2-5% baseline
- **Model Performance:**
  - Precision/Recall tracked via classification report
  - ROC-AUC score for click prediction model
  - Collaborative filtering evaluated by seen/unseen item separation

---

## Questions for Further Discussion

1. Should we implement A/B testing to validate recommendation quality?
2. Do we need real-time predictions or batch processing is sufficient?
3. What's the acceptable latency for generating recommendations?
4. Should we add content-based filtering to complement collaborative filtering?
5. How often should we retrain models with new data?

---

**Last Updated:** 2025-11-01
**System Version:** 1.0
**Contact:** [Your team contact information]
