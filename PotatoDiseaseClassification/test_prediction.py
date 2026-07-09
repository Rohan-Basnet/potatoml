from src.predict import predict_image


image_path = "/Users/rohanbasnet/Downloads/Cercospora_Melonganae.jpg"
result = predict_image(image_path)

print("Disease:", result["disease"])
print("Scientific Name:", result["scientific_name"])
print("Description:", result["description"])
print("Treatment:", result["treatment"])

