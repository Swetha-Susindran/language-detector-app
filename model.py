import pickle

with open('model.pkl', 'rb') as file:
  data = pickle.load(file)

print(data)
