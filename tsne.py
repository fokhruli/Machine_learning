# import all neccsary module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import manifold
%matplotlib inline

# data collect
pixel_value, labels  = datasets.fetch_openml(
    'mnist_784',
    version = 1,
    return_X_y = True
    )

pixel_value = pixel_value.iloc[:,:].values
labels = np.array(labels.astype('int'))

# show a images 
single_image = pixel_value[1].reshape(28, 28)
plt.imshow(single_image, cmap='gray')

# TSNE calculate
tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(pixel_value[:3000, :])

# buils data frame for R^2 representation
tsne_df = pd.DataFrame(
np.column_stack((transformed_data, labels[:3000])),
columns=["x", "y", "targets"]
)
tsne_df.loc[:, "targets"] = tsne_df.targets.astype(int)

grid = sns.FacetGrid(tsne_df, hue="targets", size=8)
grid.map(plt.scatter, "x", "y").add_legend()