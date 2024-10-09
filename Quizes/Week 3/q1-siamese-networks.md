# Siamese Networks

## Practice Quiz

### Question 1

Classification allows you to identify similarity between two things while siamese networks allow you to categorize things.

- True
- False

Answer: B

### Question 2

Do the two subnetworks in a siamese network share the same parameters?

- Yes
- No

Answer: A

### Question 3

When training a siamese network to identify duplicates, which pairs of questions from the following questions do you expect to have the highest cosine similarity?

```text
Is learning NLP useful for me to get a job? (ANCHOR)

What should I learn to get a job? (POSITIVE)

Where is the job? (NEGATIVE)
```

- Anchor, Positive
- Anchor, Negative
- Negative, Positive

Answer: A

### Question 4

In the triplet loss function below, will decreasing the hyperparameter $\alpha$ from 0.5 to 0.2 require more, or less, optimization during training ?

$$\text{diff} = s(A, N) - s(A, P)$$

$$\mathcal{L}(A,P,N) = \max(\text{diff}+\alpha,0)$$

- Less
- More

Answer: A

Explanation: $\alpha$ is the margin, so the smaller it is the less you have to optimize.

### Question 5

The orange square below corresponds to the similarity score of question duplicates?

$$
\begin{pmatrix}
    \textcolor{green}{0.7} & \textcolor{orange}{-0.6} & \textcolor{orange}{-0.4} \\
    \textcolor{orange}{-0.6} & \textcolor{green}{0.4} & \textcolor{orange}{0.1} \\
    \textcolor{orange}{-0.4} & \textcolor{orange}{0.1} & \textcolor{green}{0.5}
\end{pmatrix}
$$

- True
- False

Answer: B

Explanation: They correspond to non question duplicates.

### Question 6

What is the closest negative in this set of numbers assuming a duplicate pair similarity of 0.6?

```python
[-0.9, -0.4, 0.4, 0.8]
```

- $-0.9$
- $-0.4$
- $0.4$
- $0.8$

Answer: C

### Question 7

In one shot learning, is any retraining required when new classes are added? For example, a new bank customer’s signature.

- Yes
- No

Answer: B

### Question 8

During training, you have to update the weights of each of the subnetworks independently.

- True
- False

Answer: B

Explanation: You update the same weight.

### Question 9

The mean negative is defined as the closest off-diagonal value to the diagonal in each row (excluding the diagonal).

- True
- False

Answer: B

### Question 10

In what order are Siamese networks performed in lecture?

- 1. Convert each input into an array of numbers
  1. Feed arrays into your model
  1. Compare $v_1$, $v_2$ using cosine similarity
  1. Test against a threshold
- 1. Convert each input into an array of numbers
  1. Feed arrays into your model
  1. Run logistic regression classifier
  1. Classify by using the probability
- 1. Convert each input into an array of numbers
  1. Feed arrays into your model
  1. Run soft-max classifier for all classes
  1. Take the arg-max of the probabilities
- 1. Convert each input into an array of numbers
  1. Feed arrays into your model
  1. Compare $v_1$, $v_2$ using euclidean distance
  1. Test against a threshold

Answer: A
