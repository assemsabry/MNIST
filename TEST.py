import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)  

y_test_oh = to_categorical(y_test, num_classes=10)

model = load_model("my_model.h5")  

loss, accuracy = model.evaluate(x_test, y_test_oh)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

index = 0  
image = x_test[index]
true_label = y_test[index]

prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

print(f"True label: {true_label}")
print(f"Predicted label: {predicted_label}")

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.axis("off")
plt.show()
