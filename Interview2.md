# Attention Model (Hand-crafted)

- **Core Idea:** The Attention model computes similarity scores (e.g., dot-product or learnable weights) between the Query and Key, and performs a weighted sum of the Value in order to focus on the important parts of the input.

## Formula

- **Input:**
  1. **Query Matrix:** \( Q \in \mathbb{R}^{n_q \times d_k} \)
  2. **Key Matrix:** \( K \in \mathbb{R}^{n_k \times d_k} \)
  3. **Value Matrix:** \( V \in \mathbb{R}^{n_k \times d_v} \)

- **Steps:**
  1. **Calculate similarity scores (attention scores):**
     \[
     \text{Score}(Q, K) = Q K^\top
     \]
     This is the dot-product operation between Query and Key that represents their similarity.
  2. **Normalization (Scaled Dot-Product Attention):**
     \[
     \text{Scaled Score}(Q, K) = \frac{\text{Score}(Q, K)}{\sqrt{d_k}}
     \]
     The scaling factor \(\sqrt{d_k}\) is used to prevent the values from becoming too large, which may lead to unstable gradients.
  3. **Compute weights (Softmax normalization):**
     \[
     \text{Attention Weights} = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right)
     \]
     Softmax converts the scores into a probability distribution that is used to weight the Value.
  4. **Weighted summation:**
     \[
     \text{Attention Output} = \text{Attention Weights} \cdot V
     \]

## Example

Assume the data as follows:
1. **Input Dimensions:**
   - **Query (Q):** \(2 \times 4\) (2 Queries, each of dimension 4).
   - **Key (K):** \(3 \times 4\) (3 Keys, each of dimension 4).
   - **Value (V):** \(3 \times 2\) (3 Values, each of dimension 2).
     
     **Note:** Here \(d_k=4, d_v=2\).
2. **Specific Values:**
   - \( Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix} \)
   - \( K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix} \)
   - \( V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \)

### Step 1: Calculate Similarity Scores
Dot product calculation \( QK^\top \):
\[
QK^\top = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}^\top = \begin{bmatrix} 2 & 0 & 2 \\ 0 & 2 & 2 \end{bmatrix}
\]

### Step 2: Scale the Scores
Scaling factor: \(\sqrt{d_k} = \sqrt{4} = 2\).

Divide the score matrix by \(\sqrt{d_k}\):
\[
\text{Scaled Scores} = \frac{QK^\top}{\sqrt{d_k}} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
\]

### Step 3: Softmax Normalization
Apply Softmax on each row (to calculate weights):
- **For the first row:**
  \[
  \text{Softmax}(1, 0, 1) = \left[\frac{e^1}{e^1 + e^0 + e^1}, \frac{e^0}{e^1 + e^0 + e^1}, \frac{e^1}{e^1 + e^0 + e^1}\right] = \left[\frac{e}{2e+1}, \frac{1}{2e+1}, \frac{e}{2e+1}\right]
  \]
- **For the second row:**
  \[
  \text{Softmax}(0, 1, 1) = \left[\frac{e^0}{e^0 + e^1 + e^1}, \frac{e^1}{e^0 + e^1 + e^1}, \frac{e^1}{e^0 + e^1 + e^1}\right] = \left[\frac{1}{1+2e}, \frac{e}{1+2e}, \frac{e}{1+2e}\right]
  \]

Resulting weight matrix:
\[
\text{Attention Weights} = \begin{bmatrix} \frac{e}{2e+1} & \frac{1}{2e+1} & \frac{e}{2e+1} \\ \frac{1}{1+2e} & \frac{e}{1+2e} & \frac{e}{1+2e} \end{bmatrix}
\]

### Step 4: Compute Attention Output
Multiply the weight matrix with \( V \):
\[
\text{Attention Output} = \text{Attention Weights} \cdot V
\]
Computed row-by-row, for example:
- **For the first row:**
  \[
  \text{Output}_1 = \frac{e}{2e+1} \cdot [1, 2] + \frac{1}{2e+1} \cdot [3, 4] + \frac{e}{2e+1} \cdot [5, 6]
  \]
- **For the second row:**
  \[
  \text{Output}_2 = \frac{1}{1+2e} \cdot [1, 2] + \frac{e}{1+2e} \cdot [3, 4] + \frac{e}{1+2e} \cdot [5, 6]
  \]

## Output
- The Attention Output is a matrix representing the weighted sum of \( V \).
- **Steps:**
  1. Calculate the dot-product of Query and Key, representing similarity.
  2. Normalize by \(\sqrt{d_k}\) to avoid large values.
  3. Use softmax to derive probability weights.
  4. Perform weighted summation of Value to obtain the final output.

---

# BERT and GPT

- **BERT:**
  - **Full Name:** Bidirectional Encoder Representations from Transformers.
  - **Features:** Bidirectional Transformer capable of capturing contextual information.
  - **Training:** Utilizes Masked Language Model (MLM) and Next Sentence Prediction (NSP) tasks.
  - **Applications:** Question answering, text classification, language understanding.

- **GPT:**
  - **Full Name:** Generative Pre-trained Transformer.
  - **Features:** Unidirectional Transformer (only looks at preceding text), excels in generative tasks.
  - **Training:** Based on language modeling (predicting the next word).
  - **Applications:** Text generation, dialogue systems, code generation.

---

# Logistic Regression Frequently Asked Questions

- **Advantages:**
  1. Simple and efficient, ideal for binary classification.
  2. Outputs probabilities, which are easy to interpret.
- **Common Issues:**
  1. **Linear Separability:** Cannot address non-linear problems.
  2. **Feature Engineering Dependency:** Requires manual feature design.
  3. **Multicollinearity:** Performance decreases when features are highly correlated.

---

# Recommendation Systems

## Common Methods

### 1. Content-Based Recommendation
- **Core Principle:**  
  Based on users' historical behavior (such as ratings or browsing history) and the intrinsic features of the content, recommend items that are similar to what the user has experienced.
- **Implementation Steps:**
  1. Extract features from the content (e.g., movie genres, actors, directors).
  2. Construct a user interest model based on past behavior (e.g., preferences for certain features).
  3. Compute the similarity between the user model and item features, recommending the most similar items.
- **Advantages:**
  - Personalized recommendations independent of other users.
  - Can recommend less popular items.
- **Disadvantages:**
  - Requires detailed feature engineering.
  - May lead to a "filter bubble" of similar items.

### 2. Collaborative Filtering
Collaborative filtering analyzes user behaviors (ratings, clicks, purchases) to find similarities either among users or among items for recommendation.

#### (1) User-Based Collaborative Filtering
- **Core Principle:**  
  Identify users with similar interests and recommend items that these similar users liked.
- **Implementation Steps:**
  1. Compute similarity between users (using methods such as Euclidean distance or cosine similarity).
  2. Identify the group of users most similar to the target user.
  3. Recommend items liked by these similar users that the target user has not yet interacted with.
- **Advantages:**
  - No need to analyze item features.
  - Straightforward to implement.
- **Disadvantages:**
  - **Data Sparsity:** Typically, the user-item matrix is sparse.
  - **Cold-Start:** New users or items may lack sufficient data.

#### (2) Item-Based Collaborative Filtering
- **Core Principle:**  
  Find items that are similar to the target item and recommend them.
- **Implementation Steps:**
  1. Compute similarity between items (based on user ratings).
  2. Identify items similar to what the user has liked.
  3. Recommend these similar items.
- **Advantages:**
  - More stable since item similarity is generally less volatile.
- **Disadvantages:**
  - **New Item Issue:** New items with little data have unreliable similarity measures.

### 3. Hybrid Model
- **Core Principle:**  
  Combine content-based and collaborative filtering methods to leverage the strengths of both and mitigate their limitations.
- **Implementation Methods:**
  - **Weighted Hybrid:** Average the results from both methods.
  - **Switching Methods:** Dynamically choose which method to use (e.g., content-based for cold-start, collaborative with sufficient data).
  - **Cascading:** Use one method to form an initial candidate list and refine it with the other.
- **Advantages:**
  - Provides balanced recommendations.
- **Disadvantages:**
  - More complex design and higher computational demands.

### 4. Deep Learning-Based Recommendation
- **Core Principle:**  
  Use deep learning methods to encode complex features and capture non-linear user–item relationships.
- **Common Models:**
  1. **Wide & Deep Learning:**  
     - **Wide Part:** Captures explicit features.
     - **Deep Part:** Learns high-dimensional implicit features.
  2. **Transformer-based Models:**  
     - Uses Transformer architectures to model sequential dependencies in user behavior.
  3. **Autoencoder:**  
     - Reconstructs sparse user behavior data into a compact representation.
  4. **Reinforcement Learning:**  
     - Models recommendations as sequential decision-making to optimize long-term user satisfaction.
- **Advantages:**
  - Can learn complex, non-linear relationships.
  - Leverages large-scale data.
- **Disadvantages:**
  - High computational cost.
  - Requires a large volume of labeled data.

## Challenges in Recommendation Systems

### 1. Cold Start Problem
- **Definition:**  
  Difficulty in recommending for new users or items lacking historical data.
- **Solutions:**
  1. **For New Users:**
     - Use content-based recommendations relying on user features.
     - Collect preference information during registration.
  2. **For New Items:**
     - Use item metadata (e.g., tags, descriptions).
     - Gather initial data through promotions.

### 2. Data Sparsity
- **Definition:**  
  Very sparse user-item interactions can distort similarity calculations in collaborative filtering.
- **Solutions:**
  1. **Matrix Factorization:**  
     - Use methods like SVD or ALS to derive latent representations.
  2. **Deep Learning:**  
     - Apply neural networks to complete or capture the latent space.
  3. **Metadata Fusion:**  
     - Incorporate item features to alleviate sparsity.

### 3. Real-Time Requirements
- **Definition:**  
  Systems must respond rapidly to high-traffic scenarios.
- **Solutions:**
  1. **Caching:**  
     - Cache popular recommendations.
  2. **Online Learning:**  
     - Update models using streaming data.
  3. **Efficient Models:**  
     - Employ lightweight or distilled models.
  4. **Distributed Systems:**  
     - Use frameworks like Hadoop or Spark for scalability.

---

# IC Hard and Shortest Path Graph Theory

- **IC Hard:**  
  Refers to problems that are NP-hard or belong to a broader group where approximations may be even trickier than NP-complete problems.
  - **Example:** Graph Coloring Problem.
- **Shortest Path:**  
  - **Common Algorithms:**
    1. **Dijkstra:** Single-source for non-negative weights.
    2. **Bellman-Ford:** Handles negative weights.
    3. **Floyd-Warshall:** All-pairs shortest path via dynamic programming.
    4. **A\*:** Heuristic search.
  - **Applications:** Navigation, network routing optimization.

---

# Transformer-based Recommendation Systems: Core Idea

The Transformer model in recommendation systems plays a key role by:

1. Capturing patterns in user behavior sequences through self-attention to model long- and short-term interests.
2. Predicting the next item in a sequence.
3. Flexibly incorporating multi-modal data (e.g., text, images, categorical features).

## Implementation Steps

### 1. Data Preparation
Transformer-based recommendation systems require:
- **User Behavior Sequences:**  
  Time-ordered histories (clicks, purchases, ratings).  
  _Example:_ User A’s behavior: item1, item3, item5, item8.
- **Item Features:**  
  Attributes such as category, price, and description.
- **User Features (Optional):**  
  Demographic data like age, gender, location.

**Input Data Format:**
- Each user forms a sequence: item1, item2, item3, …
- The sequence is used to predict the next item (Next-Item Prediction).

### 2. Model Architecture

#### (1) Input Embedding Layer
- **Embedding Content:**
  1. **Item Embedding:** Map item IDs to fixed-dimensional vectors.
  2. **Time Embedding (Optional):** Encode timestamp information.
  3. **Position Embedding:** Add positional information (e.g., first, second item).
- **Embedding Equation:**
  \[
  E = \text{Embedding}(ItemID) + \text{PositionEmbedding} + \text{TimeEmbedding}
  \]

#### (2) Transformer Encoder
- **Core Modules:**
  - **Self-Attention:** Captures inter-item relationships.
  - **Residual Connections and Layer Normalization:** Stabilize training and prevent gradient issues.
- **Dimensions:**
  - **Input:** Sequence length \(L\) and embedding dimension \(d\).
  - **Output:** Contextualized user sequence representation.
- **Equation:**
  \[
  Z = \text{MultiHeadAttention}(Q, K, V) + \text{FeedForward}(Z)
  \]

#### (3) Output Layer
- **Task Objective:**  
  Predict the next item.
- **Approach:**  
  Map the Encoder output through a fully connected layer to a probability distribution over items (using Softmax).
- **Equation:**
  \[
  P(item_{t+1} \mid sequence) = \text{Softmax}(W \cdot Z_t + b)
  \]

### 3. Training Process

#### (1) Objective Function
Typically, cross-entropy loss is used:
\[
\mathcal{L} = - \sum_{i=1}^{n} \log P(item_{t+1} \mid sequence)
\]

#### (2) Negative Sampling
Generate negative samples for each positive example from user sequences.

#### (3) Optimizer
Usually, Adam or AdamW is employed.

### 4. Inference Process
During inference:
1. User behavior sequence is provided.
2. Transformer encoding yields a contextual representation.
3. The output layer predicts the next item or a recommendation list.

## Application Cases

### Case 1: SASRec (Self-Attentive Sequential Recommendation)
- **Overview:**  
  SASRec uses Transformer-based self-attention to model user sequences and predict the next item.
- **Features:**  
  - Captures both short- and long-term interests.
  - Avoids complex feature engineering.

### Case 2: BERT4Rec
- **Overview:**  
  BERT4Rec leverages a bidirectional Transformer (like BERT) to capture global context from user sequences with masked items.
- **Features:**  
  - Similar to MLM in BERT.
  - Better at capturing overall context.

## Advantages of Transformer-based Recommendation Systems

1. **Global Dependency Capture:**  
   Self-attention models relationships between any two positions.
2. **Parallel Computation:**  
   Accelerates training by processing all tokens simultaneously.
3. **Flexibility:**  
   Able to integrate multi-modal data.

## Challenges and Solutions

### 1. Long Sequences
- **Issue:**  
  O(\(L^2\)) complexity makes long sequences computationally intensive.
- **Solutions:**
  - Sparse Attention (e.g., Longformer, Reformer).
  - Sequence truncation to recent actions.

### 2. Cold Start Problem
- **Issue:**  
  New users or items lack data.
- **Solutions:**
  - Use content-based features.
  - Leverage pre-trained models.

### 3. Data Sparsity
- **Issue:**  
  Sparse user interactions affect similarity accuracy.
- **Solutions:**
  - Use pretraining for robust sequence representations.

## Summary
Transformer-based recommendation systems leverage self-attention to efficiently model complex user behavior dependencies. They are suited for next-item prediction, sequential recommendations, and multi-modal setups. Despite high computational costs, optimizations like sparse attention and pretraining help address practical challenges.

---

# How to Leverage Transformer Pretraining Capabilities

## Core Idea:
Through the Pretraining-Finetuning paradigm, first pretrain the Transformer on large-scale unsupervised data to learn general sequence representations, and then fine-tune for the recommendation task, enabling efficient transfer learning.

- **Key Points:**
  1. Pretrain using unlabeled data (massive user or related sequence data) to capture global context.
  2. Fine-tune on specific tasks (e.g., next-item prediction) to adapt to task-specific objectives.

### Common Pretraining Approaches:

#### 1. Masked Language Model (MLM) – Similar to BERT
- **Principle:**
  1. Randomly mask a fraction of items in the user sequence (e.g., item1, item2, item3, item4 → item1, [MASK], item3, item4).
  2. The model predicts the masked items using context.
- **In Recommendation Systems:**
  - **Input:** User sequence with certain items masked.
  - **Objective:** Predict the masked item ID.
  - **Benefits:** Captures global context beyond just the next-item.
- **Example (BERT4Rec):**
  Mask part of the sequence (e.g., item1, [MASK], item3, item4) so the model can learn to predict the missing item (item2).

#### 2. Causal Language Model (CLM) – Similar to GPT
- **Principle:**
  1. Predict subsequent items in an auto-regressive manner.
  2. The model uses only the previous context for prediction.
- **In Recommendation Systems:**
  - **Input:** User sequence (e.g., item1, item2, item3).
  - **Objective:** Predict the next item (item4).
  - **Benefits:** Suitable for sequential tasks.
- **Example (SASRec):**
  Input sequence of item1, item2, item3 with prediction of item4.

---

# BERT4Rec

BERT4Rec is an innovative model in recommendation that adapts BERT’s success in NLP. It uses a bidirectional Transformer to capture the global context from user behavior sequences, outperforming traditional sequential models (such as RNN, GRU, or unidirectional Transformers).

## 1. Introduction

- **Core Objective:**  
  BERT4Rec aims to predict masked items in a user behavior sequence, thus learning a representation of user interest for tasks like next-item prediction.
  
- **Key Improvements:**
  - Traditional models capture only unidirectional dependencies.
  - BERT4Rec uses a bidirectional Transformer to capture both preceding and following contexts.
  - It adopts masking (similar to MLM in NLP) to prevent information leakage during self-supervision.
  
- **Applicable Scenarios:**
  Suitable for sequential recommendation tasks such as:
  - Next-item Prediction
  - Modeling short-term user interests
  - Capturing complex dependencies in behavior sequences

## 2. Input and Output

**Input:**
- **User Behavior Sequence:**  
  A sequence \([v_1, v_2, v_3, \dots, v_n]\) where each \(v_i\) represents a user interaction (click, purchase, view).
- **Sequence Mask:**  
  Randomly mask a portion of the items (e.g., \([v_1, v_2, v_3, v_4]\) → \([v_1, [MASK], v_3, v_4]\)).

**Output:**
- **Predicted Masked Items:**  
  Predicts the probability distribution over items for the masked positions.

## 3. Features

### (1) Bidirectional Transformer
- **Differences from Unidirectional Models:**  
  RNNs, GRU, or unidirectional Transformers (e.g., GPT) capture only one directional dependency.
- **Advantages:**
  - Captures context from both directions, leading to a more accurate sequence model.

### (2) Masked Language Model (MLM)
- **Inspired by BERT in NLP:**  
  Introduces masking to force the model to learn a global representation, not solely relying on predicting the next item.
- **Purpose of Masking:**
  - Prevents leakage of information.
  - Encourages the model to capture more generalized representations.

### (3) Flexibility
- **Global Dependency Modeling:**  
  Self-attention allows learning dependencies across any two positions.
- **No Fixed Order Assumption:**  
  Supports variable-length sequences.

## 4. Architecture of BERT4Rec

### (1) Input Embedding
- **Item Embedding:**  
  Map item IDs to fixed-dimensional vectors (e.g., 128 dimensions).
- **Positional Embedding:**  
  Add position information (first, second, etc.) to preserve sequence ordering.
- **Final Input:**  
  The embedding is the sum of item and positional embeddings:
  \[
  E = \text{Embedding}(ItemID) + \text{PositionEmbedding}
  \]

### (2) Bidirectional Transformer Encoder
- **Self-Attention Mechanism:**  
  Uses multi-head attention to model dependencies.
- **Residual Connections and Layer Normalization:**  
  Stabilize training.
- **Multiple Stacked Layers:**  
  Typically 2–12 layers for deeper representations.

### (3) Output Layer
- **Task Objective:**  
  Predict masked items.
- **Implementation:**  
  A fully-connected layer maps the Transformer outputs to a probability distribution (using Softmax).

## 5. Training Process

### (1) Mask Strategy
- Randomly mask 10%–20% of items in the sequence.
- _Example:_ \([item1, item2, item3, item4]\) masked as \([item1, [MASK], item3, [MASK]]\).

### (2) Loss Function
- Use cross-entropy loss:
  \[
  \mathcal{L} = - \sum_{i=1}^{N} \log P(v_i \mid \text{context})
  \]
  Where \(v_i\) are the masked items and context refers to the rest of the sequence.

### (3) Optimizer
- Typically AdamW with learning rate scheduling (e.g., Warmup and Cosine Decay).

## 6. Inference Process
- During inference, the model uses the complete sequence (without masking) to generate recommendations.

## 7. Advantages of BERT4Rec
1. **Global Context Modeling:**  
   Can capture dependencies between any positions.
2. **Flexibility:**  
   Works with variable-length sequences.
3. **Versatility:**  
   Applicable to various tasks such as next-item prediction and click-through rate prediction.
4. **Superior Performance:**  
   Often outperforms traditional models like GRU4Rec and SASRec.

---

# Multi-Head Attention (Double-Head Transformer) Computation Process

Multi-head (a.k.a. double-head) Transformer’s key component is the multi-head attention mechanism. It computes multiple attention distributions in parallel to capture diverse patterns in the input sequence.

## 1. Transformer Input
The input to a Transformer is a sequence matrix:
\[
X \in \mathbb{R}^{L \times d_{\text{model}}}
\]
- \(L\): Sequence length (e.g., 3 items).
- \(d_{\text{model}}\): Dimensionality of the input embeddings (e.g., 4).
  
In multi-head attention, \(X\) is projected into multiple Query (Q), Key (K), and Value (V) matrices.

## 2. Core Formulas

### (1) Linear Transformation of Input
For each attention head \(i\), generate:
\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]
- \(W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}\) are learnable.
- \(d_k = \frac{d_{\text{model}}}{h}\) where \(h\) is the number of heads.

### (2) Scaled Dot-Product Attention
For each head:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]
- \(QK^\top\): Dot-product similarity.
- \(\sqrt{d_k}\): Scaling factor.
- Softmax normalizes the scores.

### (3) Multi-Head Combination
Concatenate all heads and linearly transform:
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W_O
\]
- Each \(\text{head}_i = \text{Attention}(Q, K, V)\).
- \(W_O \in \mathbb{R}^{(d_k \cdot h) \times d_{\text{model}}}\).

## 3. Example

### Input Conditions:
- **Sequence Length:** \(L = 3\) (3 items).
- **Input Dimension:** \(d_{\text{model}} = 4\).
- **Number of Heads:** \(h = 2\) (double-head).
- **Each Head’s Dimension:** \(d_k = \frac{4}{2} = 2\).

Assume:
\[
X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 1 & 1 \end{bmatrix}
\]

### Step 1: Generate Q, K, V via Linear Transformation
Assume:
\[
W_Q = W_K = W_V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}
\]
Then:
\[
Q = K = V = XW_Q = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 2 & 2 \end{bmatrix}
\]

### Step 2: Attention Computation (Per Head)
Compute:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]
Where:
\[
QK^\top = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 2 & 2 \end{bmatrix} \cdot \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 2 & 2 \end{bmatrix}^\top = \begin{bmatrix} 4 & 0 & 4 \\ 0 & 4 & 4 \\ 4 & 4 & 8 \end{bmatrix}
\]
Scaling by \(\sqrt{2}\) (since \(d_k=2\)):
\[
\frac{QK^\top}{\sqrt{2}} = \begin{bmatrix} 2.83 & 0 & 2.83 \\ 0 & 2.83 & 2.83 \\ 2.83 & 2.83 & 5.66 \end{bmatrix}
\]
Then, softmax each row:
- First row: \(\text{softmax}(2.83, 0, 2.83) = [0.422, 0.156, 0.422]\)
- Second row: \(\text{softmax}(0, 2.83, 2.83) = [0.156, 0.422, 0.422]\)
- Third row: \(\text{softmax}(2.83, 2.83, 5.66) = [0.156, 0.156, 0.688]\)

Resulting in:
\[
\text{Attention Weights} = \begin{bmatrix} 0.422 & 0.156 & 0.422 \\ 0.156 & 0.422 & 0.422 \\ 0.156 & 0.156 & 0.688 \end{bmatrix}
\]
And then:
\[
\text{Attention Output} = \text{Attention Weights} \cdot V
\]

### Step 3: Combining Heads
Concatenate outputs from all heads:
\[
\text{Concat}(\text{head}_1, \text{head}_2)
\]
For example, if:
- \(\text{head}_1 = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \in \mathbb{R}^{2 \times 2}\)
- \(\text{head}_2 = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \in \mathbb{R}^{2 \times 2}\)

Then:
\[
\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix} 1 & 2 & 5 & 6 \\ 3 & 4 & 7 & 8 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
\]
Followed by a linear transformation:
Assume:
\[
W_O = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4}
\]
Then:
\[
Z = \text{Concat}(\text{head}_1, \text{head}_2) W_O = \begin{bmatrix} 6 & 8 & 6 & 8 \\ 10 & 12 & 10 & 12 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
\]

## Summary
Multi-head attention computes multiple attention heads in parallel, each capturing different aspects of the input. Their outputs are concatenated and linearly transformed to integrate diverse information, enhancing the Transformer’s capability to model context.

---

# Regression and Classification Model Evaluation Metrics

## Regression Models

### 1. Mean Squared Error (MSE)
\[
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]
- **Explanation:**  
  MSE is the mean of the squared differences between predicted values and actual values. Squaring emphasizes larger errors, making MSE sensitive to outliers.
- **Pros:**  
  Emphasizes large deviations.
- **Cons:**  
  Too sensitive to outliers.
- **Suitable For:**  
  Tasks where large errors should be penalized, e.g., financial forecasting.

### 2. Mean Absolute Error (MAE)
\[
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]
- **Explanation:**  
  The average of absolute differences between predictions and actual values.
- **Pros:**  
  More robust to outliers.
- **Cons:**  
  Does not penalize large errors as strongly.
- **Suitable For:**  
  Tasks where all errors are weighted equally.

### 3. Coefficient of Determination (\(R^2\))
- **Pros:**  
  Provides a clear measure of explanatory power.
- **Cons:**  
  Sensitive to outliers.
- **Suitable For:**  
  Comparing model performance, particularly for assessing variance explanation.

## Classification Models

### 1. Accuracy
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

### 2. Precision, Recall, and F1 Score
\[
\text{Precision} = \frac{TP}{TP+FP} \quad \quad \text{Recall} = \frac{TP}{TP+FN}
\]
\[
F1 = \frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
\]
- **Precision:** Proportion of true positives among predicted positives.
- **Recall:** Proportion of true positives among actual positives.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Range:** [0,1] with values closer to 1 indicating better performance.

### 3. AUC-ROC Curve

#### (1) ROC Curve
- **Definition:**  
  Plots True Positive Rate (TPR) against False Positive Rate (FPR) at various thresholds.
  - \( \text{TPR} = \frac{TP}{TP+FN} \) (Recall)
  - \( \text{FPR} = \frac{FP}{FP+TN} \)

#### (2) AUC (Area Under Curve)
- **Definition:**  
  The area under the ROC curve, ranging from [0,1]. Closer to 1 denotes better performance.
- **Pros:**  
  Threshold-independent and works well with imbalanced classes.
- **Cons:**  
  May not directly address specific business needs.
- **Suitable For:**  
  Assessing the model’s discrimination ability, especially on imbalanced datasets.

### AUC Details:
- **AUC = 1:** Perfect separation.
- **AUC = 0.5:** Random guessing.
- **AUC < 0.5:** Worse than random.

---

# Differences Between MAE and MSE

- **MAE:**
  - Averages absolute errors.
  - **Pros:** Robust to outliers.
  - **Cons:** May underemphasize large errors.
- **MSE:**
  - Averages squared errors.
  - **Pros:** Penalizes larger errors more.
  - **Cons:** Overly sensitive to outliers.

---

# Loss Functions for Classification Models

- **Common Loss Functions:**
  1. **Cross-Entropy Loss**
  2. **Hinge Loss** (often used for SVM)

### Cross-Entropy Loss vs Hinge Loss

| Metric                   | Cross-Entropy Loss                         | Hinge Loss                           |
|--------------------------|--------------------------------------------|--------------------------------------|
| **Application Scenario** | Probabilistic models (e.g., Logistic Regression, Neural Networks) | Maximum margin classifiers (e.g., SVM) |
| **Tasks**                | Binary/Multiclass                          | Binary/Multiclass                    |
| **Output Characteristics** | Probabilistic output                     | Maximizes classification margin      |
| **Sensitivity to Outliers** | Sensitive                              | Also sensitive                       |
| **Probability Output**   | Yes                                        | No                                   |
| **Loss Growth Trend**    | Exponential                                | Linear                               |
| **Suitable Models**      | Logistic Regression, Neural Networks       | SVM, Support Vector Classifier       |

---

# Principles and Improvements of LSTM

## 1. Principles of LSTM
LSTM enhances RNNs by introducing a **Cell State** and gating mechanisms to capture long-term dependencies while limiting irrelevant or outdated information.

### 1.1 Issues in Traditional RNNs
- **Long-Term Dependency Problem:**  
  Information may be lost when passed through many time steps.
- **Vanishing and Exploding Gradients:**  
  Gradients can exponentially diminish or grow, impeding effective training.

### 1.2 Core Components of LSTM
LSTM uses a cell state \(C_t\) and three gates (Input, Forget, Output).

#### (1) Cell State \(C_t\)
- Serves as the central memory that stores past information.
- Controlled by the gates to retain, update, or erase information.

#### (2) Gating Mechanisms
Gates use a sigmoid function to produce values in \([0,1]\) to regulate information.

##### a. Forget Gate \(f_t\)
- **Function:** Decides which parts of the cell state to discard.
- **Formula:**
  \[
  f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
  \]
  
##### b. Input Gate \(i_t\)
- **Function:** Determines new information to add.
- **Formula:**
  \[
  i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
  \]
- Generates candidate cell state:
  \[
  \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
  \]

##### c. Output Gate \(o_t\)
- **Function:** Controls the influence of the cell state on the hidden state.
- **Formula:**
  \[
  o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
  \]

#### (3) Cell State Update
Combine previous state and new candidate:
\[
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
\]
(\(\odot\) denotes element-wise multiplication)

#### (4) Hidden State Update
The hidden state is computed as:
\[
h_t = o_t \odot \tanh(C_t)
\]

## 2. Improvements in LSTM
Enhancements over traditional RNNs include:

### 2.1 Memory Cell for Long-Term Dependencies
- **Improvement:**  
  Retains relevant long-term information via \(C_t\) rather than relying solely on \(h_t\).
- **Benefit:**  
  Effective at handling long sequences.

### 2.2 Gating Mechanisms to Mitigate Vanishing Gradients
- **Improvement:**  
  Gates control information flow, avoiding repeated gradient diminishment.
- **Benefit:**  
  Supports training of deep networks.

### 2.3 Forget Gate to Suppress Irrelevant Information
- **Improvement:**  
  Selectively drops unimportant data.
- **Benefit:**  
  Enhances model generalization.

### 2.4 Input Gate for Selective New Information
- **Improvement:**  
  Determines which new information is important.
- **Benefit:**  
  Prevents all inputs from affecting the cell state.

### 2.5 Output Gate to Control Information Flow
- **Improvement:**  
  Regulates how much of \(C_t\) is exposed through \(h_t\).
- **Benefit:**  
  Achieves a balance between retaining memory and current processing.

### 2.6 Advantages Over Traditional RNNs

| Issue                     | Traditional RNN                         | LSTM Improvement                                   |
|---------------------------|-----------------------------------------|----------------------------------------------------|
| Long-Term Dependency      | Hidden state gets overwritten           | Uses \( C_t \) to selectively retain dependencies  |
| Vanishing Gradient        | Gradients diminish exponentially        | Gated mechanisms help maintain gradient flow       |
| Noise Interference        | All inputs affect hidden state          | Forget gate discards irrelevant input              |
| Loss of Hidden Information| Hidden state can be easily overwritten  | Input/Output gates provide flexible information flow|

## 3. Variants and Further Improvements of LSTM

### 3.1 GRU (Gated Recurrent Unit)
- Simplified version of LSTM that merges the input and forget gates.

### 3.2 BiLSTM (Bidirectional LSTM)
- Processes sequences in both forward and backward directions.

### 3.3 Peephole LSTM
- Includes the cell state in the gating mechanism for finer control.

## 4. Application Scenarios
LSTM is excellent for tasks that involve sequential data, such as:
- **Natural Language Processing:** Machine translation, text generation, sentiment analysis.
- **Speech Recognition:** Translating audio signals to text.
- **Time Series Forecasting:** Predicting stock prices, weather.
- **Video Analysis:** Modeling video frame sequences.
- **Reinforcement Learning:** Incorporating temporal dependencies in policy networks.

---

# Gradient Vanishing and Exploding Solutions

## 1. Gradient Clipping
- **Principle:**  
  Set a maximum threshold for gradients to prevent them from exploding.
- **Process:**
  1. Compute the L2 norm:
     \[
     \|g\|_2 = \sqrt{\sum_i g_i^2}
     \]
  2. If \(\|g\|_2\) exceeds the threshold, scale:
     \[
     g = g \cdot \frac{\text{threshold}}{\|g\|_2}
     \]
- **Purpose:**  
  Stabilizes parameter updates.

## 2. Using ReLU Activation
- **Principle:**  
  ReLU defined as:
  \[
  f(x) = \max(0, x)
  \]
  And its derivative:
  \[
  f'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}
  \]
- **Benefit:**  
  Avoids vanishing gradients in the positive domain.
- **Drawback:**  
  May cause "neuron death."
- **Improvement:**  
  **Leaky ReLU:**
  \[
  f(x) = \begin{cases} x, & x > 0 \\ \alpha x, & x \leq 0 \end{cases}
  \]
  where \(\alpha\) is a small constant.

## 3. Batch Normalization
- **Principle:**  
  Normalize layer inputs to have zero mean and unit variance.
- **Process:**
  1. Compute mean and variance:
     \[
     \mu = \frac{1}{m}\sum_{i=1}^{m} x_i,\quad \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu)^2
     \]
  2. Normalize:
     \[
     \hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
     \]
  3. Scale and shift with \(\gamma\) and \(\beta\):
     \[
     y_i = \gamma\hat{x}_i + \beta
     \]
- **Purpose:**  
  Mitigates gradient issues and speeds up convergence.

## 4. Residual Connections (ResNet)
- **Principle:**  
  Adds a shortcut from input \(x\) to the output:
  \[
  y = F(x) + x
  \]
- **Purpose:**  
  Helps gradients flow directly, preventing vanishing.
- **Application:**  
  Used in very deep networks (e.g., ResNet-50).

---

# Common Tree Models

## 1. Decision Trees
- **Principle:**  
  Use tree-structured splits on features to partition the space for classification/regression.
- In classification:
  1. **Split Data:**  
     Split recursively (e.g., \(x_i \leq t\)) optimizing purity.
  2. **Purity Metrics:**
     - **Information Gain:**
       \[
       \text{Information Gain} = \text{Entropy}(D) - \sum_{i=1}^{k}\frac{|D_i|}{|D|}\cdot \text{Entropy}(D_i)
       \]
     - **Gini Index:**
       \[
       \text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2
       \]
     - **MSE for regression.**
  3. **Stopping Criteria:**  
     Maximum depth, minimum sample count, or minimal improvement.
- **Advantages:**
  - Easy to interpret.
  - Works with both categorical and continuous features.
- **Disadvantages:**
  - Prone to overfitting.
  - Not suitable for small or high-dimensional data.

## 2. Random Forests
- **Principle:**  
  An ensemble of decision trees using bootstrap sampling and random feature subsets.
- **Randomness:**
  1. **Bootstrap Sampling:**  
     Sampling with replacement.
  2. **Feature Subset Selection:**  
     Randomly select features at each split.
- **For Classification:**
  1. Build multiple trees.
  2. Apply majority voting.
- **Advantages:**
  - Reduces overfitting.
  - Performs well on large, high-dimensional datasets.
- **Disadvantages:**
  - More complex and slower.
  - Harder to interpret individual tree decisions.

## 3. Gradient Boosting Trees (GBDT)
- **Principle:**  
  Sequentially build trees to fit the residuals of previous trees.
- **Workflow:**
  1. Initialize with a simple model.
  2. Build trees on residuals.
  3. Update using a learning rate.
  4. Repeat until convergence.
- **Improved Versions:**
  - **XGBoost:** Adds regularization and supports distributed training.
  - **LightGBM:** Uses histogram-based algorithms for speed.
  - **CatBoost:** Optimizes for categorical features.
- **Advantages:**
  - Excellent performance.
  - Can handle various loss functions.
- **Disadvantages:**
  - Slower training, sensitive to hyperparameter tuning.

---

# Graph Neural Networks (GNN) Background

GNNs are deep learning models designed for graph-structured data via message passing.

## 1. Introduction

### 1.1 What is Graph-Structured Data?
Graphs consist of nodes (entities) and edges (relationships). Examples include:
- **Social Networks:** Users and friendships.
- **Molecular Structures:** Atoms and chemical bonds.
- **Knowledge Graphs:** Entities and their relationships.
- **Traffic Networks:** Intersections and roads.

A graph is defined as:
\[
G = (V, E)
\]
- \(V\): Nodes.
- \(E\): Edges.
- \(A\): Adjacency matrix.
- \(X\): Node feature matrix.

### 1.2 Core Idea of GNNs
GNNs learn node embeddings by leveraging the graph’s structure.
- **Message Passing:**  
  Nodes aggregate information from neighbors.
- **Aggregation:**  
  Summing, averaging, or using attention.
- **Update:**  
  Combine aggregated information with node features.

### 1.3 Basic Computation Framework
1. **Message Passing:**  
   Each node collects messages from its neighbors.
2. **Node Update:**  
   \[
   h_v^{(t+1)} = \text{Update}\left(h_v^{(t)}, \text{Aggregate}\left(\{h_u^{(t)}: u \in \mathcal{N}(v)\}\right)\right)
   \]
3. **Output:**  
   Use final representations for classification, regression, etc.

## 2. Application Scenarios
- **Graph Classification:** Predict overall graph labels (e.g., molecule toxicity).
- **Node Classification:** Predict labels for individual nodes (e.g., user interests).
- **Link Prediction:** Predict the existence or attributes of edges.
- **Other Applications:** Graph generation, path planning.

## 3. Common GNN Variants
- **Graph Convolutional Network (GCN):** Uses convolution operations.
- **Graph Attention Network (GAT):** Employs attention for flexible aggregation.
- **Graph Isomorphism Network (GIN):** Highly expressive in distinguishing graphs.
- **Dynamic GNN:** Models graphs that change over time.

## 4. Summary
- **Flexible:** Handles non-regular data.
- **Expressive:** Captures rich topological and feature information.
- **Broad Applications:** From social networks to molecular design.

---

# Stochastic Gradient Descent (SGD)

- **Principle:**  
  Updates model parameters using one or a small batch of samples per iteration.
- **Pros:**  
  High computational efficiency.
- **Cons:**  
  Convergence can be unstable.

---

# NLP Bayesian

## 1.1 Bayes' Theorem
Bayes' Theorem calculates the posterior probability:
\[
P(C \mid X) = \frac{P(X \mid C)P(C)}{P(X)}
\]
- \(P(C \mid X)\): Posterior probability of class \(C\) given features \(X\).
- \(P(X \mid C)\): Likelihood of features \(X\) under class \(C\).
- \(P(C)\): Prior probability of class \(C\).
- \(P(X)\): Marginal probability of features \(X\).

For classification, since \(P(X)\) is constant:
\[
P(C \mid X) \propto P(X \mid C) \cdot P(C)
\]

## 3. Variations of Naive Bayes

### 3.1 Multinomial Naive Bayes
- Assumes features follow a multinomial distribution.
- Commonly used in text classification based on word counts.

### 3.2 Bernoulli Naive Bayes
- Assumes binary features (0 or 1).
- Suitable for short text classification.

### 3.3 Gaussian Naive Bayes
- Assumes features follow a normal distribution.
- Used for continuous feature classification.

---

# Transformer and Self-Attention

- **Transformer:**  
  Utilizes self-attention to capture global dependencies.
- **Self-Attention:**  
  Compares each position in the sequence to every other to compute weights.

---

# Masked Attention: Is Mask Added at Every Layer?

Yes, each layer should include a mask to ensure that only the current and preceding positions are attended to (e.g., in GPT).

**Core Idea:**  
During attention computation, a mask matrix is added to block out certain positions so that those positions do not contribute to the final attention scores.

### 2.1 Role of the Mask
In the attention formula:
\[
\text{Masked Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
\]
- \(M\) is a mask with the same shape as \(QK^T\).
- If \(M_{ij} = -\infty\), the score is effectively nullified (softmax outputs 0).
- If \(M_{ij} = 0\), there is no effect.

### 2.2 Types of Masks

#### (1) Padding Mask
- **Purpose:**  
  When sequences are padded to equal length, padding positions should be ignored.
- **Mask:**  
  Padding positions are set to \(-\infty\); non-padding positions are 0.
- **Example:**
  - Input Sequence: \([1, 2, 3, 0, 0]\)
  - Padding Mask: \([0, 0, 0, -\infty, -\infty]\)

#### (2) Causal Mask
- **Purpose:**  
  In autoregressive tasks, ensure that the output at a time step does not depend on future steps.
- **Mask:**  
  Each time step only attends to itself and previous positions.
- **Example:**  
  For a sequence of length 4:
  \[
  M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}
  \]
  - Row 1: Only itself.
  - Row 2: Can attend to first and itself.
  - And so on.

### 3. Application in Transformers
- **Encoder:**  
  Typically uses a Padding Mask.
- **Decoder:**  
  Uses both Causal and Padding Masks.

---

# Softmax

- **Purpose:**  
  Converts a vector of values into a probability distribution.
- **Formula:**
\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

---

# CNN

- **Purpose:**  
  Extracts local features from images.
- **Core Components:**  
  Convolution layers, pooling layers, fully-connected layers.

---

# Handling Peak Traffic

Peak traffic refers to periods when the system experiences extreme load (e.g., during flash sales or marquee events). To ensure stability, several strategies are used:

## 1. Caching

### 1.1 Principle
Cache stores data copies near users or at the front end, providing quick responses without recalculating.

### 1.2 Use Cases
- **Static Resources:**  
  CDN caching for images, JS, CSS.
- **Hot Data:**  
  Use in-memory caches (e.g., Redis) for high-demand data.
- **Query Caching:**  
  Cache database query results.

### 1.3 Implementations
1. **CDN Caching:**  
   Distribute static files on CDN nodes.
2. **In-Memory Caching:**  
   Use Redis or Memcached for hot data.
3. **Local Caching:**  
   Cache on the client side.
- **Advantages:**  
  Fast, reduced backend load.
- **Disadvantages:**  
  Data consistency and cache expiration can be challenging.

## 2. Dynamic Load Balancing

### 2.1 Principle
Distribute incoming requests among multiple servers using a load balancer.

### 2.2 Use Cases
- Web service request distribution.
- Database query distribution.
- Distributed computational tasks.

### 2.3 Implementations
1. **Hardware Load Balancers:**  
   Devices like F5 or A10.
2. **Software Load Balancers:**  
   Tools like Nginx or HAProxy.
3. **DNS Load Balancing:**  
   Use domain resolution for traffic distribution.
4. **Dynamic Scheduling Algorithms:**  
   Round-robin, weighted round-robin, least connections.
- **Advantages:**  
  Improves availability and scalability.
- **Disadvantages:**  
  Complexity and potential bottlenecks in the load balancer itself.

## 3. Rate Limiting and Degradation

### 3.1 Rate Limiting
- **Principle:**  
  Limit the number of requests in a given time frame.
- **Common Algorithms:**
  1. **Fixed Window:**  
     Limit by fixed time intervals.
  2. **Sliding Window:**  
     Continuous adjustment.
  3. **Token Bucket:**  
     Tokens are generated at a fixed rate; each request consumes a token.
  4. **Leaky Bucket:**  
     Requests flow out at a constant rate.
- **Use Cases:**  
  API request restrictions, safeguarding backend systems.

### 3.2 Degradation
- **Principle:**  
  Under extreme load, reduce or disable non-critical features to maintain core functionality.
- **Strategies:**
  1. **Static Degradation:**  
     Return static pages or defaults.
  2. **Functional Degradation:**  
     Disable features such as recommendations or logging.
  3. **Delayed Processing:**  
     Process non-critical tasks asynchronously.
  4. **Gray Degradation:**  
     Gradually lower service quality for less critical users.
- **Use Cases:**  
  E-commerce flash sales, content delivery during high traffic.

## 4. Comprehensive Strategy
Combine multiple techniques:
1. **Frontend Caching:**  
   CDN and Redis for hot data.
2. **Rate Limiting:**  
   Via proxies or API gateways.
3. **Backend Degradation:**  
   Prioritize core services.
4. **Dynamic Scaling:**  
   Use auto-scaling in cloud environments.
5. **Load Balancing:**  
   Across multiple servers and data centers.

## 5. Summary Table

| Method                      | Applicable Scenarios         | Advantages                               | Disadvantages                  |
|-----------------------------|------------------------------|------------------------------------------|--------------------------------|
| **Caching**                 | Hot data, static resources   | Reduces backend load, fast response      | Data consistency issues        |
| **Dynamic Load Balancing**  | High concurrency, multi-server | Enhances availability and scaling        | Complexity, potential bottlenecks |
| **Rate Limiting & Degradation** | Peak traffic protection    | Prevents overload, secures core functions | Can lower user experience      |

---
# In-Context Learning

## Introduction
In-Context Learning (ICL) is a reasoning capability in large language models (LLMs) that allows the model to learn specific tasks through contextual prompts without additional training or parameter adjustments. This capability enables the model to perform various tasks during inference without fine-tuning for each task.

---

## 1. What is In-Context Learning?
The core idea of In-Context Learning is that the model performs tasks during inference by providing input-output examples or a brief task description in the context.

- **No additional training:** No need to update model parameters.
- **Context-dependent prompting:** The model learns task patterns through the constructed context.

### Example: Text Classification Task
Suppose the goal is to classify sentences as "Positive" or "Negative":

**Input Context:**
```plaintext
Task: Classify the sentiment of the sentences.
Example 1: "I love this movie!" -> Positive
Example 2: "This is the worst day ever." -> Negative
Sentence: "The food was amazing!" ->
```

**Model Output:**
```plaintext
Positive
```

The model understands this is a sentiment classification task and generates the correct label based on the examples.

---

## 2. Characteristics

### 2.1 No Additional Training
- Unlike traditional fine-tuning methods, ICL does not update model weights.
- The model relies solely on input context to complete tasks.

### 2.2 Flexibility
- Supports various tasks such as text classification, translation, summarization, and code generation.
- Task definition depends on the design of the context prompt.

### 2.3 Generalization Ability
- Since large language models learn broad knowledge and language patterns during pre-training, they can quickly generalize to new tasks with a few examples.

---

## 3. Implementation Methods
ICL relies on **prompt design**, commonly using the following methods:

### 3.1 Zero-Shot Learning
- **Pattern:** Provide task instructions without examples.
- **Advantage:** No examples required, suitable for simple tasks or clear language descriptions.

**Example:**
```plaintext
Task: Translate the following sentence from English to French.
Input: "How are you?"
Output:
```
**Model Output:**
```plaintext
Comment ça va?
```

### 3.2 Few-Shot Learning
- **Pattern:** Provide a few input-output examples in the context to help the model understand the task.
- **Advantage:** Suitable for more complex tasks by showing patterns through examples.

**Example:**
```plaintext
Task: Translate the following sentences from English to French.
Example 1: "Hello, my friend." -> "Bonjour, mon ami."
Example 2: "Good morning." -> "Bonjour."
Input: "I am happy." ->
```
**Model Output:**
```plaintext
Je suis heureux.
```

### 3.3 Chain-of-Thought Prompting
- **Pattern:** Include intermediate reasoning steps in the context to help the model perform logical reasoning.
- **Advantage:** Suitable for multi-step reasoning or complex tasks.

**Example (Math Problem):**
```plaintext
Question: If you have 3 apples and buy 2 more, how many apples do you have in total?
Step-by-step reasoning:
- Start with 3 apples.
- Buy 2 more apples.
- Total apples = 3 + 2 = 5.
Answer: 5
```

---

## 4. Application Scenarios
ICL performs excellently in multiple NLP tasks. Here are some typical applications:

### 4.1 Text Classification
**Task:** Classify the topic of the sentences.

**Example:**
```plaintext
Task: Classify the topic of the following sentences.
Example 1: "The stock market hit a record high today." -> Finance
Example 2: "The team won the championship!" -> Sports
Sentence: "NASA launched a new satellite into orbit." ->
```
**Model Output:**
```plaintext
Science
```

### 4.2 Machine Translation
**Task:** Translate text from English to Chinese.

**Example:**
```plaintext
Task: Translate the following sentences from English to Chinese.
Example 1: "Hello, how are you?" -> "你好，你好吗？"
Example 2: "Thank you for your help." -> "谢谢你的帮助。"
Input: "What is your name?" ->
```
**Model Output:**
```plaintext
你叫什么名字？
```

### 4.3 Text Summarization
**Task:** Generate a brief summary for the given text.

**Example:**
```plaintext
Task: Summarize the following paragraph.
Paragraph: "The new iPhone was released today, featuring an improved camera, faster processor, and longer battery life. Customers lined up at stores worldwide to purchase the new device."
Summary:
```
**Model Output:**
```plaintext
The new iPhone features a better camera, faster processor, and longer battery life.
```

### 4.4 Code Generation
**Task:** Generate code based on natural language descriptions.

**Example:**
```plaintext
Task: Write a Python function to calculate the factorial of a number.
Function:
```
**Model Output:**
```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
```

### 4.5 Logical Reasoning
**Task:** Perform step-by-step logical reasoning to solve complex problems.

**Example:**
```plaintext
Question: John has 3 red balls and 2 blue balls. Mary gives him 4 more blue balls. How many blue balls does John have now?
Step-by-step reasoning:
- John initially has 2 blue balls.
- Mary gives him 4 more blue balls.
- Total blue balls = 2 + 4 = 6.
Answer: 6
```

## 5. Advantages and Limitations

### 5.1 Advantages

1. **No Additional Training Required:**  
   - ICL directly leverages pre-trained models without updating model parameters.

2. **Strong Task Transferability:**  
   - Quickly adapts to new tasks by defining them through context design.

3. **Saves Computational Resources:**  
   - Compared to fine-tuning, it avoids the time and computational costs of large-scale training.

### 5.2 Limitations

1. **Prompt Design Dependency:**  
   - Different prompt designs may lead to performance variations, requiring repeated trials to find the optimal setup.

2. **Context Length Limitation:**  
   - Current models have input context length constraints (e.g., GPT-3's 2048 token limit).

3. **Not Suitable for All Tasks:**  
   - For complex tasks requiring extensive domain knowledge, ICL performance may be insufficient.

4. **Potential Instability:**  
   - Model outputs may be inconsistent, especially in zero-shot scenarios.

---

## 6. Conclusion

In-Context Learning is a powerful capability of modern large language models, using contextual prompts to complete tasks without additional training. This method offers great flexibility and generality, particularly excelling in the following scenarios:

| Scenario        | Example Tasks                             |
|-----------------|-------------------------------------------|
| Language Understanding | Text classification, sentiment analysis   |
| Language Generation    | Machine translation, text summarization   |
| Logical Reasoning      | Mathematical calculations, complex reasoning |
| Code Generation        | Generating code from natural language descriptions |

Although In-Context Learning has some limitations, with the expansion of model scales and improvements in prompt engineering, its potential in NLP applications continues to grow.

---

## 2. Inference Process of ICL

The ICL inference process can be divided into the following steps:

### 2.1 Input Construction

ICL inference starts by constructing an **input sequence (prompt)**. The input sequence typically consists of the following parts:

1. **Task Description (Optional):**  
   - Defines the task goal, providing context information.  
   - Example:
     ```plaintext
     Task: Classify the sentiment of the sentences below.
     ```

2. **Few-Shot Examples (Optional):**  
   - Provides known input-output pairs to help the model understand task patterns.  
   - Example:
     ```plaintext
     Example 1: "I love this movie!" -> Positive
     Example 2: "This is the worst experience ever." -> Negative
     ```

3. **Query Input:**  
   - The target input that the model needs to infer.  
   - Example:
     ```plaintext
     Sentence: "The weather is wonderful today!" ->
     ```

**Complete Input Example:**
```plaintext
Task: Classify the sentiment of the sentences below.
Example 1: "I love this movie!" -> Positive
Example 2: "This is the worst experience ever." -> Negative
Sentence: "The weather is wonderful today!" ->
```

---

### 2.2 Input Sequence Encoding

The language model processes the entire input sequence as a text chunk, encoding it into embedding vectors. The specific steps are:

1. **Tokenization:**  
   - Breaks the input sequence into a series of tokens.  
   - Example:
     ```json
     Tokens: ["Task:", "Classify", "the", "sentiment", "of", ..., "wonderful", "today", "!"]
     ```

2. **Embedding Mapping:**  
   - Each token is mapped to a high-dimensional vector representing its semantics and contextual information.

3. **Sequence Processing:**  
   - The entire input sequence is fed into the Transformer model, which extracts patterns through attention mechanisms.

---

### 2.3 Model Attention Mechanism: Understanding Context

The core of ICL lies in the Transformer model's **Attention Mechanism**, which dynamically focuses on different parts of the input sequence to infer task patterns. The detailed breakdown of the inference process includes:

1. **Building Contextual Associations:**  
   - The model learns the relationships between tokens in the input sequence through attention.  
   - Example:  
     - Observes "I love this movie!" -> Positive and "This is the worst experience ever." -> Negative.
     - The attention mechanism captures the association between sentiment words (**love**, **worst**) and sentiment labels.

2. **Pattern Extraction:**  
   - During inference, the model uses pre-trained language knowledge to extract patterns from the context.  
   - Example:  
     - Learns that "love" and "wonderful" are positive sentiment words mapping to **Positive**.
     - Learns that "worst" is a negative sentiment word mapping to **Negative**.

3. **Query Input Inference:**  
   - When processing the query input, the model embeds it into the context and maps it using the previously extracted pattern.  
   - Example:  
     - For the input "The weather is wonderful today!", the model recognizes **"wonderful"** as a positive sentiment word and infers **Positive**.

---

### 2.4 Output Generation

1. **Softmax Score Calculation:**  
   - The model generates a probability distribution for each possible output (e.g., **Positive** and **Negative**).  
   - These probabilities are based on patterns extracted from the context prompt and the **Query Input**.

2. **Output Selection:**  
   - The model selects the output with the highest probability as the final result.  
   - Example:
     ```plaintext
     Output: "Positive"
     ```
# Selection Bias

## Definition:
Data sampling that fails to reflect the true distribution.  

### Solutions:
- Enhance data sampling strategies.  
- Use reweighting methods.  

Selection Bias is a common issue in statistics and machine learning. It refers to the problem where the data collected during sampling does not accurately reflect the true overall distribution, leading to biased model training or analysis results.  

## 1. Definition

### 1.1 What is Selection Bias?  
Selection bias is when the sample data does not match the distribution of the true population, resulting in systematic errors in the model's inference or predictions.  

**Core Issue:**  
The distribution of the training data \( P_{\text{train}}(X) \) does not match the true data distribution \( P_{\text{true}}(X) \).  
Because the training data distribution is biased, the model cannot generalize well to real-world scenarios.  

### 1.2 Examples  

- **Survey Bias:**  
  Conducting surveys only for a specific group (e.g., urban residents), which does not reflect the true opinions of the entire population.  

- **Medical Data Bias:**  
  Hospital data might focus more on severe patients, ignoring mild cases or healthy individuals, which leads to a model that is inaccurate for the general population.  

- **Recommender Systems:**  
  Training data might come from users' historical click behavior, and content that users did not click on (potential interests) is under-sampled.  

---

## 2. Types of Selection Bias

### 2.1 Non-Random Sampling  
The data sampling process is not random, and certain data are systematically over-sampled or under-sampled.  

**Example:**  
In fraud detection, the sampling ratio might lean towards normal transactions, making it difficult for the model to detect the few abnormal transactions.  

### 2.2 Ignoring Certain Groups  
The data for certain groups may be systematically ignored or underestimated, leading to poor predictive performance for those groups.  

**Example:**  
In facial recognition, if the training data is biased towards a particular skin color or gender, the model may perform poorly on other groups.  

### 2.3 Self-Selection Bias  
Individuals (or systems) deciding whether to participate in data generation or sampling can lead to a mismatch between the sample distribution and the true distribution.  

**Example:**  
An online survey may only attract people who are interested in a particular topic, while others might not participate.  

### 2.4 Survivorship Bias  
Only sampling data from subjects that have "survived" a certain condition, while ignoring those that did not.  

**Example:**  
Analyzing World War II airplane protection only based on the planes that returned, ignoring those that were shot down.  

---

## 3. Impact of Selection Bias  

- **Decreased model generalization ability:**  
  The model may perform well on the training data but perform poorly on unseen data or in real-world scenarios.  

- **Systematic error:**  
  Analysis results may be biased towards certain groups or specific features, failing to reflect overall patterns.  

- **Unfairness:**  
  The model may be unfair to certain groups (e.g., gender, race).  

---

## 4. Solutions  

### 4.1 Enhance Data Sampling Strategies  

#### Method 1: Random Sampling  
Ensure that the data sampling process is as random as possible to reduce human or systematic bias.  

**Example:**  
In a survey, randomly select participants rather than choosing only a specific group.  

#### Method 2: Stratified Sampling  
Divide the data into multiple sub-groups (strata) based on specific characteristics, then perform random sampling within each stratum.  

**Example:**  
In a gender classification task, ensure that the sampled data has a gender distribution close to the true population distribution.  

#### Method 3: Data Supplementation  
Use external data sources to supplement the under-sampled portions so that the overall distribution is better represented.  

**Example:**  
In medical data, collect data from community hospitals or general clinics to better cover healthy individuals.  

---

### 4.2 Using Reweighting Methods  

#### Method 1: Adjusting Sample Weights  
Assign weights to each sample so that the overall training data distribution approximates the true distribution.  

**Implementation:**  
Calculate weights based on the sampling bias:  

\[
w(x) = \frac{P_{\text{true}}(x)}{P_{\text{train}}(x)}
\]

Adjust the loss function during training using these sample weights.  

#### Method 2: Importance Sampling  
Adjust the sampling probabilities to correct the training data distribution so that it approximates the true distribution.  

**Example:**  
For under-sampled categories in a classification task, increase the sampling probability for those categories.  

---

### 4.3 Data Augmentation  

#### Method 1: Generative Data Augmentation  
Use generative models (e.g., GANs or data augmentation methods) to generate data for under-sampled categories.  

**Example:**  
In image classification tasks, generate more samples through data augmentation (e.g., rotation, flipping, cropping).  

#### Method 2: Over-sampling and Under-sampling  
- **Over-sampling:** Increase the number of samples for minority groups (e.g., using the SMOTE method).  
- **Under-sampling:** Decrease the number of samples for majority groups.  

---

### 4.4 Model-Level Adjustments  

#### Method 1: Regularization  
Add regularization terms to the model to reduce overfitting to biased patterns in the training data.  

**Example:**  
L2 regularization can prevent the model from overfitting to noise in the data.  

#### Method 2: Fairness Constraints  
Incorporate constraints into the model to ensure that the predictive performance is similar across different groups.  

**Example:**  
Enforce fairness in model outputs on sensitive attributes (e.g., gender, race).  

---

### 4.5 Evaluation After Bias Correction  

After addressing selection bias, evaluate the model to ensure its performance on the true distribution:  
- Use an independent test set that as much as possible reflects the true distribution.  
- Use cross-validation to examine the model's performance across different groups.  
# RAG (Retrieval-Augmented Generation)

RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation to enhance the capabilities of generative models, especially for tasks that require external knowledge or conversation context. It is a hybrid approach that integrates information retrieval with generative models, enabling the generation of more accurate and knowledge-rich content.

---

## 1. Full Name and Definition  

### 1.1 Full Name:  
**Retrieval-Augmented Generation**  

### 1.2 Definition:  
RAG introduces a retrieval module that dynamically incorporates relevant information from external knowledge bases into the generation process, allowing the generative model to utilize the latest and most comprehensive knowledge when generating answers or text.  

---

## 2. Features of RAG  

### 2.1 Combination of Retrieval and Generation  
- The retrieval module fetches information related to the input query from an external knowledge base or document collection.  
- The generation module (e.g., GPT, T5) utilizes the retrieved information to generate an answer or text.  

### 2.2 Dynamic Knowledge Enhancement  
- The dynamically introduced knowledge from the retrieval module enables the model to use up-to-date information during generation without needing to retrain the generative model.  
- Applicable for tasks that require external or rich domain knowledge (e.g., knowledge Q&A, document generation).  

### 2.3 High Scalability  
- The retrieval module can use any external knowledge base (e.g., Wikipedia, internal company databases, document repositories).  
- The generation module can be any powerful generative language model (e.g., GPT, T5).  

### 2.4 End-to-End Optimization  
- RAG can jointly optimize the weights of the retrieval module and the generation module to achieve better overall performance.  

---

## 3. Workflow of RAG  

The workflow of RAG typically involves the following steps:  

---

### 3.1 Input Query  

The user inputs a query, for example:  

```plaintext
"Explain the significance of quantum computing in modern technology."
