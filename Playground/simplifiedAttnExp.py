import torch

# Demonstrative simplified fixed attention weight with just dot product similarity

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 1. Calculate attention weights for one token only inputs[1]
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# 2. Calculate context vector for inputs[1] by multiplying input tokens by attention weights, and summing
query = inputs[1] # 2nd input token is the query
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
# print(context_vec_2)

# 3. Now we can compute attention weights for all the tokens
attn_scores = torch.empty(6, 6)
for r, x_r in enumerate(inputs):
    for c, x_c in enumerate(inputs):
        attn_scores[r, c] = torch.dot(x_r, x_c)
print(attn_scores)

# 4. This syntax is actually the same as 3. We can visually confirm by print out
attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
print("Previous 2nd context vector:", context_vec_2)