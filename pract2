from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Select a few words to visualize
words_to_visualize = [
    'King', 'Queen', 'Man', 'Woman'
]

# Filter out words not in the model's vocabulary
filtered_words = [word for word in words_to_visualize if word in model.key_to_index]

# Get the vectors for the filtered words
vectors = [model[word] for word in filtered_words]

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Create a scatter plot
plt.figure(figsize=(5, 1))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], edgecolors='k', c='skyblue')

for i, word in enumerate(filtered_words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), textcoords="offset points", xytext=(5,5), ha='center')

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()





# Perform vector arithmetic: king - man + woman
# This operation is expected to result in a vector close to 'queen'.
result_vector = model['king'] - model['man'] + model['woman']

# Find the words most similar to the result vector
print("Words most similar to 'king' - 'man' + 'woman':")
similar_words = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)

for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")




import plotly.express as px
import pandas as pd

# Create data_for_plotly using the filtered_words and vectors_2d from the PCA visualization
data_for_plotly = pd.DataFrame({
    'word': filtered_words,
    'x': vectors_2d[:, 0],
    'y': vectors_2d[:, 1]
})

# Get the list of words from the DataFrame used for PCA visualization
words = data_for_plotly['word'].tolist()

# Initialize an empty similarity matrix
similarity_matrix = pd.DataFrame(index=words, columns=words)

# Populate the similarity matrix
for i in range(len(words)):
    for j in range(len(words)):
        word1 = words[i]
        word2 = words[j]
        # model.similarity() calculates cosine similarity
        similarity_matrix.loc[word1, word2] = model.similarity(word1, word2)

# Convert the matrix to float type for the heatmap
similarity_matrix = similarity_matrix.astype(float)

# Create an interactive heatmap using Plotly Express
fig = px.imshow(
    similarity_matrix,
    text_auto=True, # Show similarity values on the heatmap cells
    aspect="auto",
    title='Cosine Similarity Heatmap for Word Embeddings',
    labels=dict(x="Word 2", y="Word 1", color="Similarity"),
    color_continuous_scale=px.colors.sequential.Viridis # Choose a color scale
)

# Update layout for better readability, especially for small matrices
fig.update_xaxes(side="top")
fig.update_layout(xaxis_nticks=len(words), yaxis_nticks=len(words))

fig.show()
